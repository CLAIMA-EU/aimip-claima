"""Job side of the DLESyM / ArchesClimate-SSP / ArchesWeather families. Runs in the emulator venv (started by `aimip.emu`):

    python -I -m aimip.emu_job stage <run-dir>     DLESyM: ERA5 state at <start> from ARCO-ERA5 -> <run-dir>/ic_era5.pt
    python -I -m aimip.emu_job run <run-dir>       the rollout -> <run-dir>/{atmosphere,ocean}/monthly_mean_predictions.nc

Outputs follow the fme layout so `aimip summary` reads them: variables (sample, time, lat, lon), `valid_time`
(sample, time) and `counts` (sample, time). For ArchesClimate-SSP, dataset item 0 uses state 2015-02 with previous
state 2015-01, so the first output month is March 2015. ArchesWeather writes the pressure-level variables to
`atmosphere/monthly_levels.nc` (sample, time, level, lat, lon) and its last states to `restart.nc` (continuation).
"""
from __future__ import annotations

import json
import os
import shutil
import sys
import tempfile
import time

TIME_UNITS = "days since 1970-01-01 00:00:00"
T2M_BAND_K = (200.0, 330.0)   # a global-mean 2-m temperature outside this band means the rollout has blown up
TP6_EPS = 1e-8   # DLESyM v1.1 tp6 = log(P + eps) - log(eps), P = 6-h precipitation in m
EXIT_UNSTABLE = 5


def _log(msg: str) -> None:
    print(f"[aimip.emu_job] {msg}", flush=True)


def _setup_tmp() -> str:
    import multiprocessing.util

    multiprocessing.util.get_temp_dir()          # AF_UNIX sockets stay in the short system /tmp
    tmp = os.environ.get("AIMIP_EMU_TMP") or tempfile.mkdtemp()
    os.makedirs(tmp, mode=0o700, exist_ok=True)
    for sub in ("e2s-cache", "mpl"):
        os.makedirs(os.path.join(tmp, sub), exist_ok=True)
    tempfile.tempdir = tmp
    os.environ["TMPDIR"] = tmp
    return tmp


def _load_job(out_dir: str) -> dict:
    real = os.path.realpath(out_dir)
    with open(os.path.join(real, "config.yaml"), encoding="utf-8") as fh:
        job = json.load(fh)
    if os.path.realpath(job.get("out_dir", "")) != real:
        raise SystemExit(f"config.yaml out_dir {job.get('out_dir')!r} does not match {real}")
    return job


# ------------------------------------------------------------------ monthly-mean netCDF writer (fme-like layout)

class MonthlyWriter:
    """Monthly means of one realm, written month by month to <out>/<realm>/monthly_mean_predictions.nc.

    With *levels*, every variable gets a `level` dimension: (sample, time, level, lat, lon)."""

    def __init__(self, path: str, names: list, lat, lon, n_members: int, attrs: dict | None = None,
                 extra: dict | None = None, levels=None, level_units: str = "hPa"):
        import netCDF4
        import numpy as np

        self.np = np
        self.path = path
        self.lat = np.asarray(lat, dtype="f8")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self.ds = netCDF4.Dataset(path + ".part", "w", format="NETCDF4")
        self.ds.createDimension("sample", n_members)
        self.ds.createDimension("time", None)
        self.ds.createDimension("lat", len(lat))
        self.ds.createDimension("lon", len(lon))
        dims: tuple = ("sample", "time", "lat", "lon")
        chunks: tuple = (1, 1, len(lat), len(lon))
        if levels is not None:
            self.ds.createDimension("level", len(levels))
            lv = self.ds.createVariable("level", "f8", ("level",))
            lv[:] = np.asarray(levels, dtype="f8")
            lv.units = level_units
            dims = ("sample", "time", "level", "lat", "lon")
            chunks = (1, 1, len(levels), len(lat), len(lon))
        for name, vals, units in (("lat", lat, "degrees_north"), ("lon", lon, "degrees_east")):
            v = self.ds.createVariable(name, "f8", (name,))
            v[:] = np.asarray(vals, dtype="f8")
            v.units = units
        t = self.ds.createVariable("time", "f8", ("time",))
        t.units, t.calendar, t.long_name = TIME_UNITS, "proleptic_gregorian", "first day of the month"
        vt = self.ds.createVariable("valid_time", "f8", ("sample", "time"))
        vt.units, vt.calendar = TIME_UNITS, "proleptic_gregorian"
        self.ds.createVariable("counts", "i4", ("sample", "time"))
        self.vars = {}
        for n in names:
            meta = (extra or {}).get(n, {})
            v = self.ds.createVariable(n, "f4", dims, zlib=True, complevel=1, shuffle=True, chunksizes=chunks)
            for k, val in meta.items():
                setattr(v, k, val)
            self.vars[n] = v
        for k, val in (attrs or {}).items():
            setattr(self.ds, k, val)
        self.months: list = []                   # month labels in time order (shared by all members)

    def _index(self, month) -> int:
        if month not in self.months:
            self.months.append(month)
            self.months.sort()
        return self.months.index(month)

    def write(self, member: int, month, fields: dict, count: int) -> None:
        np = self.np
        i = self._index(month)
        day = (np.datetime64(str(month) + "-01") - np.datetime64("1970-01-01")).astype("timedelta64[D]").astype(float)
        self.ds["time"][i] = day
        self.ds["valid_time"][member, i] = day
        self.ds["counts"][member, i] = int(count)
        for n, arr in fields.items():
            self.vars[n][member, i, ...] = arr

    def close(self) -> None:
        self.ds.close()
        os.replace(self.path + ".part", self.path)


class MonthAccumulator:
    def __init__(self, torch):
        self.torch = torch
        self.month, self.sum, self.n = None, None, 0

    def add(self, month, x):
        flushed = None
        if self.month is not None and month != self.month:
            flushed = self.pop()
        if self.sum is None:
            self.month, self.sum, self.n = month, self.torch.zeros_like(x, dtype=self.torch.float64), 0
        self.sum += x.double()
        self.n += 1
        return flushed

    def pop(self):
        if self.sum is None:
            return None
        out = (self.month, (self.sum / self.n).float().cpu().numpy(), self.n)
        self.month, self.sum, self.n = None, None, 0
        return out


# ------------------------------------------------------------------ DLESyM

class Unstable(Exception):
    """A month with non-finite values or an out-of-band global-mean t2m."""

    def __init__(self, month, member, why):
        super().__init__(f"member {member}: {why} in {month}")
        self.month, self.member, self.why = str(month), member, why


def _check_month(names, arr, lat, month, member):
    import numpy as np

    if not np.isfinite(arr).all():
        bad = [n for i, n in enumerate(names) if not np.isfinite(arr[i]).all()]
        raise Unstable(month, member, f"non-finite values in {', '.join(bad[:4])}")
    if "t2m" in names:
        w = np.cos(np.deg2rad(lat))[:, None] * np.ones((1, arr.shape[-1]))
        g = float((arr[names.index("t2m")] * w).sum() / w.sum())
        if not T2M_BAND_K[0] <= g <= T2M_BAND_K[1]:
            raise Unstable(month, member, f"global-mean t2m {g:.1f} K outside {T2M_BAND_K[0]:.0f}-{T2M_BAND_K[1]:.0f} K")


def _healpix_pad_backend() -> str:
    """Select earth2grid's HEALPix padding backend and record it in emu-result.json.

    earth2grid is built with its CUDA padding kernel (`healpixpad_cuda`); if an environment was built without it,
    fall back to the pure-PyTorch `indexing` backend and warn that DLESyM v1.1 values differ from the CUDA kernel."""
    try:
        import earth2grid.healpix._padding as hp_pad
        from earth2grid.healpix._padding import cuda as hp_cuda
    except ImportError:
        return "unavailable"
    if getattr(hp_cuda, "healpixpad_cuda", None) is None and hp_pad.current_pad_backend() == hp_pad.PaddingBackends.cuda:
        hp_pad._backend = hp_pad.PaddingBackends.indexing
        _log("WARNING: healpixpad_cuda is not installed; using earth2grid's indexing padding (DLESyM v1.1 values "
             "differ from the CUDA kernel). Rebuild with install/install-container.sh or install/install-native.sh --with-emu.")
    return hp_pad.current_pad_backend().name


def _dlesym_model(job: dict, device: str):
    from earth2studio.models.auto import Package

    _log(f"earth2grid HEALPix padding backend: {_healpix_pad_backend()}")

    pkg = Package(job["package_dir"], cache=False)
    if job["flavor"] == "v0":
        from earth2studio.models.px.dlesym_v0_isccp_era5 import DLESyMv0_ISCCP_ERA5LatLon as M

        model = M.load_model(pkg, use_ttr=True)
    else:
        from earth2studio.models.px import DLESyMLatLon

        model = DLESyMLatLon.load_model(pkg, version=job["version"])
    return model.to(device)


def month_weights(t, starts) -> tuple[int, float]:
    """(k, w): the value at time *t* is (1 - w) * month[k] + w * month[k + 1], linear in time between the month-start
    stamps *starts* (numpy datetime64, ascending), as the authors' DLESyM AIMIP forcing (monthly -> daily `interp`)."""
    import numpy as np

    t = np.datetime64(t, "ns")
    starts = np.asarray(starts).astype("datetime64[ns]")
    k = int(np.searchsorted(starts, t, side="right")) - 1
    if k == len(starts) - 1 and k >= 1 and t == starts[-1]:
        return k - 1, 1.0                   # exactly the last stamp
    if k < 0 or k + 1 >= len(starts):
        raise SystemExit(f"the prescribed SST does not cover {t} (forcing {starts[0]} .. {starts[-1]})")
    return k, float((t - starts[k]) / (starts[k + 1] - starts[k]))


class PrescribedSST:
    """AIMIP SST (Zenodo 17065758, 0.25 deg monthly, stamped at the first of each month) on DLESyM's HEALPix grid.

    Land cells are filled with the model's own SST NaN interpolation and the field is mapped with the model's own
    lat-lon -> HEALPix regridder, exactly as Earth2Studio prepares the SST of the ERA5 initial state. As a front and
    rear hook it replaces the SST of every input and output lead time, so the atmosphere and the ocean coupling see the
    prescribed SST (the authors' "ground-truth" ocean model) and the ocean output is the prescribed SST."""

    def __init__(self, model, path: str, device: str):
        import netCDF4
        import numpy as np

        self.model, self.device = model, device
        self.ds = netCDF4.Dataset(path)
        lat = np.asarray(self.ds["latitude"][:], "f8")
        lon = np.asarray(self.ds["longitude"][:], "f8")
        if lat.shape != (721,) or lon.shape != (1440,) or abs(lat[0] - 90) > 1e-6 or abs(lon[1] - 0.25) > 1e-6:
            raise SystemExit(f"{path}: expected the 0.25-degree grid 90..-90, 0..359.75")
        tv = self.ds["time"]
        dates = netCDF4.num2date(tv[:], tv.units, getattr(tv, "calendar", "standard"),
                                 only_use_cftime_datetimes=False, only_use_python_datetimes=True)
        self.starts = np.array([np.datetime64(d.replace(tzinfo=None), "ns") for d in dates])
        if not all(np.datetime64(s, "D") == np.datetime64(np.datetime64(s, "M"), "D") for s in self.starts):
            raise SystemExit(f"{path}: SST time stamps are not month starts")
        self.lat, self.lon = lat, lon
        self.cache: dict = {}
        self.used: set = set()

    def _month(self, k: int):
        from collections import OrderedDict

        import numpy as np
        import torch

        if k not in self.cache:
            v = np.ma.filled(self.ds["sea_surface_temperature"][k], np.nan).astype("f4")
            x = torch.from_numpy(v)[None, None, None, None]
            coords = OrderedDict(batch=np.zeros(1), time=np.zeros(1), lead_time=np.zeros(1),
                                 variable=np.array(["sst"]), lat=self.lat, lon=self.lon)
            filled = self.model._nan_interpolate_sst(x, coords)
            hpx = self.model.to_hpx(filled.to(self.device))[0, 0, 0, 0]
            if not torch.isfinite(hpx).all():
                raise SystemExit(f"prescribed SST month {self.starts[k]} has non-finite values on HEALPix")
            self.cache = {kk: vv for kk, vv in self.cache.items() if kk >= k - 1}   # the two months in use
            self.cache[k] = hpx
            self.used.add(k)
        return self.cache[k]

    def at(self, t):
        k, w = month_weights(t, self.starts)
        return (1.0 - w) * self._month(k) + w * self._month(k + 1)

    def hook(self, x, coords):
        import numpy as np
        import torch

        if x.is_inference() and not torch.is_inference_mode_enabled():
            x = x.clone()   # Earth2Studio's tensors come from inference mode; in-place updates need a normal tensor
        vi = list(coords["variable"]).index("sst")
        for ti, t0 in enumerate(coords["time"]):
            for li, lt in enumerate(coords["lead_time"]):
                x[:, ti, li, vi] = self.at(np.datetime64(t0) + lt)
        return x, coords


def stage_dlesym(out_dir: str, job: dict) -> None:
    import numpy as np
    import torch
    from earth2studio.data import ARCO_ERA5
    from earth2studio.data.utils import fetch_data

    model = _dlesym_model(job, "cpu")
    ic = model.input_coords()
    t0 = np.array([np.datetime64(job["start"])])
    x, coords = fetch_data(source=ARCO_ERA5(cache=False), time=t0, variable=np.array(ic["variable"]),
                           lead_time=ic["lead_time"], device="cpu")
    # ERA5 SST is NaN over land; DLESyM fills those cells internally (dlesym.py:1564-1590).
    names = [str(v) for v in coords.get("variable", ic["variable"])]
    nan_vars = [n for i, n in enumerate(names) if not torch.isfinite(x[..., i, :, :]).all()]
    if [n for n in nan_vars if n != "sst"]:
        raise SystemExit(f"ERA5 initial state has non-finite values in {nan_vars} (only sst may, over land)")
    target = os.path.join(out_dir, "ic_era5.pt")
    torch.save({"x": x, "coords": {k: np.asarray(v) for k, v in coords.items()}, "start": job["start"],
                "model": job["model"]}, target + ".part")
    os.replace(target + ".part", target)
    print(
        f"[aimip.emu_job] STAGED {job['model']} {tuple(x.shape)} variables={len(ic['variable'])} -> {target}",
        file=sys.stderr,
        flush=True,
    )


def run_dlesym(out_dir: str, job: dict) -> dict:
    from collections import OrderedDict

    import numpy as np
    import torch

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model = _dlesym_model(job, dev)
    sst = None
    if job.get("sst_forcing"):
        sst = PrescribedSST(model, job["sst_forcing"], dev)
        model.front_hook = sst.hook      # Earth2Studio applies both on the HEALPix grid around every coupled step
        model.rear_hook = sst.hook
        _log(f"prescribed SST from {job['sst_forcing']} ({sst.starts[0]} .. {sst.starts[-1]})")
    staged = torch.load(os.path.join(out_dir, "ic_era5.pt"), weights_only=False)
    start = np.datetime64(job["start"])
    end = start + np.timedelta64(6 * int(job["steps"]), "h")
    hours_per_step = int(model.atmos_output_times[-1] / np.timedelta64(1, "h"))
    n_iter = int(np.ceil(6 * int(job["steps"]) / hours_per_step))
    n_members = int(job["members"])
    attrs = {"source": f"{job['model']} (Earth2Studio DLESyM, checkpoint {job['version']})", "start": job["start"],
             "aimip_family": "e2s-dlesym"}
    if sst is not None:
        attrs["sst"] = ("prescribed: AIMIP Phase 1 forcing (Zenodo 17065758), linear in time between the monthly "
                        "stamps, land filled by the model's SST interpolation; ocean files hold the prescribed SST")
    files = {"atmosphere": "atmosphere/monthly_mean_predictions.nc", "ocean": "ocean/monthly_mean_predictions.nc",
             "precip": "atmosphere/monthly_precip.nc"}
    precip_meta = {"tp6": {"long_name": "monthly mean of tp6 = log(P + eps) - log(eps)"},
                   "tp6_exp": {"long_name": "monthly mean of exp(tp6)"},
                   "tp06": {"units": "m per 6 h", "long_name": "6-hour precipitation, monthly mean",
                            "comment": f"eps * (mean(exp(tp6)) - 1), eps = {TP6_EPS} (inferred from the authors' pipeline)"}}
    writers: dict = {}
    written: dict = {}

    def flush(key, names, member, done):
        month, arr, count = done
        _check_month(names, arr, writers[key].lat, month, member)
        fields = {n: arr[i] for i, n in enumerate(names)}
        if key == "precip":
            fields["tp06"] = (TP6_EPS * (arr[1].astype("f8") - 1.0)).astype("f4")
        writers[key].write(member, month, fields, count)
        written[key] = written.get(key, 0) + 1

    unstable = None
    try:
        _dlesym_members(model, staged, job, end, n_iter, hours_per_step, writers, files, attrs, precip_meta,
                        flush, n_members, dev)
    except Unstable as exc:
        unstable = exc
        _log(f"UNSTABLE {exc}: the run stops; months before it are written")
    for w in writers.values():
        w.close()
    out = {"iterations_per_member": n_iter, "hours_per_iteration": hours_per_step, "months_written": written,
           "members": n_members, "healpix_pad_backend": _healpix_pad_backend()}
    if sst is not None:
        used = sorted(sst.used)
        out["prescribed_sst"] = {"file": job["sst_forcing"], "months_used": len(used),
                                 "first": str(sst.starts[used[0]])[:10] if used else None,
                                 "last": str(sst.starts[used[-1]])[:10] if used else None}
    if unstable is not None:
        out.update({"unstable_from": unstable.month, "unstable_member": unstable.member, "unstable_reason": unstable.why})
    return out


def _dlesym_members(model, staged, job, end, n_iter, hours_per_step, writers, files, attrs, precip_meta, flush,
                    n_members, dev):
    from collections import OrderedDict

    import numpy as np
    import torch

    fresh_seed = torch.initial_seed()      # reset torch's global RNG per member so member results are order-independent
    for member in range(n_members):
        seed = int(job["seed"]) + member
        # Every member starts from the same initial process RNG state; members differ through the model's own noise
        # generator below (`seed + member`).
        torch.manual_seed(fresh_seed)
        if getattr(model, "use_cln", False):
            model.set_rng(seed)            # v1.1 samples conditional-layer-norm noise every step
        it = model.create_iterator(staged["x"].to(dev), OrderedDict(staged["coords"]))
        next(it)                           # the initial state
        acc: dict = {}
        names_of: dict = {}
        tw = time.time()
        for _ in range(n_iter):
            y, yc = next(it)
            for realm, fn in (("atmosphere", model.retrieve_valid_atmos_outputs),
                              ("ocean", model.retrieve_valid_ocean_outputs)):
                ya, yac = fn(y, yc)
                names = [str(v) for v in yac["variable"]]
                base = np.asarray(yac["time"])[0]
                leads = np.asarray(yac["lead_time"])
                step_h = int((leads[1] - leads[0]) / np.timedelta64(1, "h")) if len(leads) > 1 else hours_per_step
                groups = []
                if "tp6" in names:         # accumulate exp(tp6) too: its monthly mean gives P = eps * (mean - 1)
                    k = names.index("tp6")
                    tp = ya[..., k:k + 1, :, :]
                    groups.append(("precip", torch.cat([tp, torch.exp(tp.double()).to(ya.dtype)], dim=-3), ["tp6", "tp6_exp"]))
                    keep = [i for i, v in enumerate(names) if v != "tp6"]
                    ya, names = ya[..., keep, :, :], [names[i] for i in keep]
                groups.insert(0, (realm, ya, names))
                for key, fields, fnames in groups:
                    if key not in writers:
                        all_names = fnames + (["tp06"] if key == "precip" else [])
                        writers[key] = MonthlyWriter(os.path.join(job["out_dir"], files[key]), all_names, np.asarray(yac["lat"]),
                                                     np.asarray(yac["lon"]), n_members, attrs=attrs,
                                                     extra=precip_meta if key == "precip" else None)
                    names_of[key] = fnames
                    a = acc.setdefault(key, MonthAccumulator(torch))
                    for j, lt in enumerate(leads):
                        t = base + lt
                        if t > end:
                            break
                        if key == "precip":
                            t = t - np.timedelta64(step_h // 2, "h")   # a 6-h accumulation belongs to its interval
                        done = a.add(np.datetime64(t, "M"), fields[0, j] if fields.dim() == 5 else fields[0, 0, j])
                        if done is not None:
                            flush(key, fnames, member, done)
        for key, a in acc.items():
            done = a.pop()
            if done is not None:
                flush(key, names_of[key], member, done)
        _log(f"member {member} seed {seed} done in {time.time() - tw:.1f}s")


# ------------------------------------------------------------------ ArchesClimate-SSP

def _arches_code_dir(data_root: str | None = None) -> str:
    """Resolve `$AIMIP_DATA/code/ArchesClimate-SSP` (downloaded from its author at first use)."""
    override = os.environ.get("AIMIP_ARCHES_CODE")
    if override:
        if os.path.isdir(override):
            return override
        raise FileNotFoundError(f"AIMIP_ARCHES_CODE does not exist: {override}")
    candidates: list[str] = []
    if data_root:
        candidates.append(os.path.join(data_root, "code", "ArchesClimate-SSP"))
    env_data = os.environ.get("AIMIP_DATA")
    if env_data:
        candidates.append(os.path.join(env_data, "code", "ArchesClimate-SSP"))
    for home_var in ("AIMIP_HOME", "AIMIP_PLAYGROUND"):
        root = os.environ.get(home_var)
        if root:
            candidates.append(os.path.join(root, "data", "code", "ArchesClimate-SSP"))
    for cand in candidates:
        if os.path.isdir(cand):
            return cand
    target = candidates[0] if candidates else "$AIMIP_DATA/code/ArchesClimate-SSP"
    raise FileNotFoundError(
        f"ArchesClimate-SSP source directory not found at {target}; "
        "run `aimip fetch -m archesclimate-ssp` first."
    )


def run_arches(out_dir: str, job: dict) -> dict:
    tmp = os.environ.get("AIMIP_EMU_TMP") or tempfile.gettempdir()
    data_root = os.path.dirname(os.path.dirname(job["memmap_dir"]))
    # The package reads its statistics through importlib.resources (ArchesClimate/stats/); the data tree may be
    # read-only, so a private copy of the code gets the downloaded statistics file linked in.
    src = _arches_code_dir(data_root)
    if not os.environ.get("AIMIP_ARCHES_CODE"):
        from .setup_data import verify_arches_code_dir

        verify_arches_code_dir(os.path.dirname(os.path.dirname(src)))
    code_root = os.path.join(tmp, "code")
    dst = os.path.join(code_root, "ArchesClimate")
    shutil.copytree(src, dst, symlinks=True, ignore=shutil.ignore_patterns(".git", "*.ipynb", "logs"))
    for walk_root, _, _ in os.walk(dst, followlinks=False):
        if not os.path.islink(walk_root):
            os.chmod(walk_root, 0o755)
    stats_link = os.path.join(dst, "stats", "cmip_stats_new_ozone.pt")
    if os.path.lexists(stats_link):
        os.unlink(stats_link)
    os.symlink(job["stats_file"], stats_link)
    sys.path[:0] = [code_root, dst]          # `ArchesClimate.*` and the repo's top-level `analysis`

    import inspect

    import hydra
    import numpy as np
    import torch
    from omegaconf import open_dict

    import ArchesClimate
    from ArchesClimate.analysis.compare_runs import load_run_all_vars, load_run_lev, load_run_level_all

    if not os.path.realpath(ArchesClimate.__file__).startswith(os.path.realpath(dst)):
        raise SystemExit(f"ArchesClimate imported from {ArchesClimate.__file__}, expected the private copy {dst}")
    torch.manual_seed(int(job["seed"]))
    dotlist = [
        f"dataloader.dataset.path={job['memmap_dir']}",
        f"dataloader.dataset.forcings_path={job['reference_dir']}/",
        f"cluster.data_path={data_root}/",
        f"cluster.work_path={tmp}/",
        "dataloader.dataset.skip_pf=true",
    ]
    # Not load_module(): with a directory it loads the NEWEST *.ckpt it finds (base_module.py:79-117), so an extra file
    # would replace the pinned weights. Perform the same steps directly with the verified checkpoint file.
    from omegaconf import OmegaConf

    cfg = OmegaConf.load(os.path.join(job["model_dir"], "config.yaml"))
    cfg.merge_with_dotlist(dotlist)
    module = hydra.utils.instantiate(cfg.module.module, cfg.module)
    if not os.path.isfile(job["checkpoint"]):
        raise SystemExit(f"checkpoint {job['checkpoint']} is not a file")
    module.init_from_ckpt(job["checkpoint"], ckpt_fname=None, missing_warning=False, use_ema=True)
    module = module.to("cuda").eval()
    ds_cfg = cfg.dataloader.dataset
    sig = inspect.signature(hydra.utils.get_class(ds_cfg._target_).__init__).parameters
    dropped = []
    if not any(p.kind == p.VAR_KEYWORD for p in sig.values()):
        dropped = [k for k in list(ds_cfg.keys()) if k != "_target_" and k not in sig]
        with open_dict(ds_cfg):
            for k in dropped:
                ds_cfg.pop(k)
    _log(f"dataset keys unknown to the public code, dropped: {dropped}")
    dataset = hydra.utils.instantiate(ds_cfg, domain=job["domain"], limit_examples=0, skip_pf=True)
    # The scenario reaches the dataset only through domain + the checkpoint's val/test filters: verify the file opened.
    ds_files = [str(f) for f in (getattr(dataset, "files", None) or [])]
    filters = {k: list(ds_cfg.get(k) or []) for k in ("val_filter", "test_filter")}
    _log(f"dataset files: {ds_files}; filters: {filters}")
    tag = f"_{job['scenario']}_interpolation.memmap"
    if len(ds_files) != 1 or tag not in ds_files[0]:
        raise SystemExit(f"domain {job['domain']!r} opened {ds_files}, expected exactly one file containing {tag!r}")
    steps = int(job["steps"])
    if len(dataset) < steps:
        raise SystemExit(f"scenario {job['scenario']} has {len(dataset)} usable months, fewer than --steps {steps}")
    chunks = os.path.join(tmp, "rollout")
    os.makedirs(chunks, exist_ok=True)
    tw = time.time()
    with torch.inference_mode():
        module.generate_rollouts(cfg=cfg, batch_index=0, test_dataset=dataset, target_name=job["domain"],
                                 out_dir=chunks, start_member=0, end_member=1, flat_forcings=False,
                                 num_rollout_steps=steps, num_perturbations_per_member=1, debug=False,
                                 seed_index=0, save_all_vars=True)
    torch.cuda.synchronize()
    roll_s = time.time() - tw
    surf = load_run_all_vars(chunks, dataset.data_mean["surface"].squeeze(-3), dataset.data_std["surface"].squeeze(-3))[0].numpy()
    lvl = load_run_level_all(chunks, dataset.data_mean["level"], dataset.data_std["level"])[0].numpy()
    lev = load_run_lev(chunks, dataset.data_mean["lev"].squeeze(0), dataset.data_std["lev"].squeeze(0))[0].numpy()
    n = surf.shape[0]
    if n != steps:
        raise SystemExit(f"rollout returned {n} months, expected {steps}")
    lat = np.linspace(-89.375, 89.375, 144)
    lon = np.linspace(1.25, 358.75, 144)
    y0, m0 = (int(v) for v in job["first_output_month"].split("-"))
    months = [np.datetime64(f"{y0 + (m0 - 1 + k) // 12:04d}-{(m0 - 1 + k) % 12 + 1:02d}", "M") for k in range(n)]
    units = {"tas": "K", "psl": "Pa", "ps": "Pa", "pr": "kg m-2 s-1", "uas": "m s-1", "vas": "m s-1",
             "net_flux": "W m-2", "evspsbl": "kg m-2 s-1"}
    surf_names = [str(v) for v in cfg.module.surface_variables]
    attrs = {"source": f"ArchesClimate-SSP deterministic, IPSL-CM6A-LR r1i1p1f1 {job['scenario']} from 2015",
             "aimip_family": "arches", "first_output_month": job["first_output_month"]}
    w = MonthlyWriter(os.path.join(out_dir, "atmosphere", "monthly_mean_predictions.nc"), surf_names, lat, lon, 1,
                      attrs=attrs, extra={v: {"units": units.get(v, "")} for v in surf_names})
    for k in range(n):
        w.write(0, months[k], {v: surf[k, i] for i, v in enumerate(surf_names)}, 1)
    w.close()
    thetao = lev.astype("f4")
    if float(np.nanmean(thetao[:, 0])) > 100.0:
        thetao = thetao - 273.15
    wo = MonthlyWriter(os.path.join(out_dir, "ocean", "monthly_mean_predictions.nc"), ["thetao_0"], lat, lon, 1,
                       attrs=attrs, extra={"thetao_0": {"units": "degC", "long_name": "sea water potential temperature, top level"}})
    for k in range(n):
        wo.write(0, months[k], {"thetao_0": thetao[k, 0]}, 1)
    wo.close()
    _write_levels(os.path.join(out_dir, "atmosphere", "monthly_levels.nc"), lvl, [str(v) for v in cfg.module.level_variables],
                  np.array([p / 100.0 for p in cfg.module.pressure_levels]), lat, lon, months, "level", "hPa", attrs)
    _write_levels(os.path.join(out_dir, "ocean", "monthly_thetao.nc"), thetao[:, None], ["thetao"],
                  np.array(list(cfg.module.depth_levels), dtype="f8"), lat, lon, months, "depth", "m", attrs)
    shutil.rmtree(chunks, ignore_errors=True)
    tas = surf[:, surf_names.index("tas")] if "tas" in surf_names else None
    first_year = None
    if tas is not None:
        wgt = np.cos(np.deg2rad(lat))[:, None] * np.ones((1, len(lon)))
        first_year = float((tas[: min(12, n)] * wgt).sum(axis=(1, 2)).mean() / wgt.sum())
    return {"months": n, "first_month": str(months[0]), "last_month": str(months[-1]), "rollout_seconds": round(roll_s, 1),
            "global_mean_tas_first_12_months_K": first_year, "dropped_dataset_keys": dropped,
            "dataset_files": ds_files, "dataset_filters": filters}


def _write_levels(path, arr, names, levels, lat, lon, months, dim, units, attrs) -> None:
    """(time, var, level, lat, lon) -> one netCDF with (time, level, lat, lon) per variable."""
    import netCDF4
    import numpy as np

    os.makedirs(os.path.dirname(path), exist_ok=True)
    with netCDF4.Dataset(path + ".part", "w") as ds:
        ds.createDimension("time", len(months))
        ds.createDimension(dim, len(levels))
        ds.createDimension("lat", len(lat))
        ds.createDimension("lon", len(lon))
        t = ds.createVariable("time", "f8", ("time",))
        t.units, t.calendar = TIME_UNITS, "proleptic_gregorian"
        t[:] = [(np.datetime64(str(m) + "-01") - np.datetime64("1970-01-01")).astype("timedelta64[D]").astype(float) for m in months]
        lv = ds.createVariable(dim, "f8", (dim,))
        lv[:], lv.units = levels, units
        for name, vals, u in (("lat", lat, "degrees_north"), ("lon", lon, "degrees_east")):
            v = ds.createVariable(name, "f8", (name,))
            v[:], v.units = vals, u
        for i, n in enumerate(names):
            v = ds.createVariable(n, "f4", ("time", dim, "lat", "lon"), zlib=True, complevel=1, shuffle=True,
                                  chunksizes=(1, len(levels), len(lat), len(lon)))
            v[:] = arr[:, i]
        for k, val in attrs.items():
            setattr(ds, k, val)
    os.replace(path + ".part", path)


# ------------------------------------------------------------------ ArchesWeather / ArchesWeatherGen (geoarches)

# Buffers the backbone computes when it is built (the checkpoints do not store them).
AW_OK_MISSING = ("attn.earth_position_index",)
# geoarches name -> (output name, units); geopotential is written as geopotential height zg = z / g0.
AW_SURFACE_OUT = {"10m_u_component_of_wind": ("uas", "m s-1"), "10m_v_component_of_wind": ("vas", "m s-1"),
                  "2m_temperature": ("tas", "K"), "mean_sea_level_pressure": ("psl", "Pa"),
                  "sea_surface_temperature": ("sst", "K"), "sea_ice_cover": ("sic", "1")}
AW_LEVEL_OUT = {"geopotential": ("zg", "m"), "u_component_of_wind": ("ua", "m s-1"),
                "v_component_of_wind": ("va", "m s-1"), "temperature": ("ta", "K"),
                "specific_humidity": ("hus", "kg kg-1"), "vertical_velocity": ("wap", "Pa s-1")}
G0 = 9.80665


def _geoarches_code() -> str:
    """Verify the pinned geoarches tree next to this interpreter's environment and import it from there."""
    from .geoarches_src import GEOARCHES_COMMIT, code_dir, verify

    root = code_dir()
    if os.environ.get("AIMIP_GEOARCHES_CODE"):
        _log(f"WARNING: geoarches from AIMIP_GEOARCHES_CODE={root}; the pinned tree digest is not checked")
    else:
        verify(root)
    # A native tree is writable by its owner: bytecode written into it would change the tree digest for the next run.
    sys.dont_write_bytecode = True
    sys.path.insert(0, root)
    import geoarches

    if not os.path.realpath(geoarches.__file__).startswith(os.path.realpath(root) + os.sep):
        raise SystemExit(f"geoarches imported from {geoarches.__file__}, expected {root}")
    _log(f"geoarches {GEOARCHES_COMMIT[:12]} from {root}")
    return root


def _aw_module(model_dir: str, dotlist: list, torch, allow_unexpected: tuple = ()):
    """Build one geoarches module from its config and load the pinned checkpoint file (weights only, no pickle).

    Not geoarches' load_module(): it loads the newest *.ckpt in the directory with weights_only=False and ignores
    every key mismatch. Here only the buffers in AW_OK_MISSING may be missing, and only keys with a prefix in
    *allow_unexpected* may be left unused."""
    from hydra.utils import instantiate
    from omegaconf import OmegaConf

    from .emu_catalog import AW_CKPT

    cfg = OmegaConf.load(os.path.join(model_dir, "config.yaml"))
    cfg.merge_with_dotlist(list(dotlist))
    module = instantiate(cfg.module.module, cfg.module, cfg.stats)
    ckpt = torch.load(os.path.join(model_dir, AW_CKPT), map_location="cpu", weights_only=True)
    res = module.load_state_dict(ckpt["state_dict"], strict=False)
    bad_missing = [k for k in res.missing_keys if not k.endswith(AW_OK_MISSING)]
    bad_unused = [k for k in res.unexpected_keys if not k.startswith(allow_unexpected)] if allow_unexpected else list(res.unexpected_keys)
    if bad_missing or bad_unused:
        raise SystemExit(f"{model_dir}: checkpoint does not match the model (missing {bad_missing[:4]}, "
                         f"unused {bad_unused[:4]})")
    unused = len(res.unexpected_keys)
    return module.eval(), cfg, unused


def _aw_write_restart(path: str, job: dict, saved: list, *, valid_time: str, step_index: int, month: str,
                      sv: list, lv: list, levels: list, lat_internal, lon_internal) -> None:
    """Write the states the next step needs, per member, exactly as the model consumes them.

    state_* / prev_state_*: normalized float32 tensors on geoarches' internal grid (latitude north first, longitude
    rolled by half the globe); surface SST / sea-ice cover are 0 over land, as fed back. month_sum_* / month_days: the
    float64 sums of the physical daily states of *month* when the run stopped inside it (written only then), so the
    continuation completes that month's mean."""
    import netCDF4
    import numpy as np

    from .emu_catalog import AW_RESTART_FORMAT
    from .geoarches_src import GEOARCHES_COMMIT

    s0 = saved[0]
    ds = netCDF4.Dataset(path + ".part", "w", format="NETCDF4")
    try:
        for name, size in (("sample", len(saved)), ("surface_variable", s0["state_surface"].shape[0]),
                           ("surface_level", s0["state_surface"].shape[1]),
                           ("level_variable", s0["state_level"].shape[0]), ("level", s0["state_level"].shape[1]),
                           ("lat", s0["state_level"].shape[2]), ("lon", s0["state_level"].shape[3])):
            ds.createDimension(name, size)
        for name, vals, units in (("level", levels, "hPa"), ("lat", lat_internal, "degrees_north"),
                                  ("lon", lon_internal, "degrees_east")):
            v = ds.createVariable(name, "f8", (name,))
            v[:] = np.asarray(vals, dtype="f8")
            v.units = units
        sdims = ("sample", "surface_variable", "surface_level", "lat", "lon")
        ldims = ("sample", "level_variable", "level", "lat", "lon")
        carry = month != ""
        arrays = [("state_surface", "f4", sdims), ("prev_state_surface", "f4", sdims),
                  ("state_level", "f4", ldims), ("prev_state_level", "f4", ldims)]
        if carry:
            arrays += [("month_sum_surface", "f8", sdims), ("month_sum_level", "f8", ldims)]
        for name, dtype, dims in arrays:
            v = ds.createVariable(name, dtype, dims, zlib=True, complevel=1, fill_value=False,
                                  chunksizes=(1, 1) + tuple(len(ds.dimensions[d]) for d in dims[2:]))
            for k, m in enumerate(saved):
                v[k] = m[name]
        days = ds.createVariable("month_days", "i4", ("sample",))
        days[:] = np.asarray([m["month_days"] for m in saved], dtype="i4")
        attrs = {"aimip_restart_format": AW_RESTART_FORMAT, "model": job["model"], "valid_time": valid_time,
                 "step_index": int(step_index), "seed": int(job["seed"]), "package_digest": job["package_digest"],
                 "geoarches_commit": GEOARCHES_COMMIT, "month": month,
                 "surface_variables": ",".join(sv), "level_variables": ",".join(lv),
                 "comment": "states at valid_time and one day earlier, normalized with the pinned statistics, on "
                            "geoarches' internal grid; continue with: aimip run -m " + job["model"] + " --ic <this run "
                            "directory> --start " + valid_time}
        for k, val in attrs.items():
            setattr(ds, k, val)
    finally:
        ds.close()
    os.replace(path + ".part", path)


def _aw_read_restart(path: str, job: dict, *, sv: list, lv: list, levels: list, shapes: dict) -> dict:
    """Read a restart written by _aw_write_restart and check it against this job (the file's SHA-256 was verified
    against the manifest before the job started)."""
    import netCDF4
    import numpy as np

    from .emu_catalog import AW_RESTART_FORMAT
    from .geoarches_src import GEOARCHES_COMMIT

    with netCDF4.Dataset(path) as ds:
        ds.set_auto_mask(False)
        a = {k: ds.getncattr(k) for k in ds.ncattrs()}
        want = {"aimip_restart_format": AW_RESTART_FORMAT, "model": job["model"], "valid_time": job["start"],
                "seed": int(job["seed"]), "package_digest": job["package_digest"], "geoarches_commit": GEOARCHES_COMMIT,
                "surface_variables": ",".join(sv), "level_variables": ",".join(lv)}
        bad = [k for k, v in want.items() if str(a.get(k)) != str(v)]
        if len(ds.dimensions["sample"]) != int(job["members"]):
            bad.append("sample")
        if not np.array_equal(np.asarray(ds["level"][:], dtype="f8"), np.asarray(levels, dtype="f8")):
            bad.append("level")
        for name in ("state_surface", "state_level"):
            if tuple(ds[name].shape[1:]) != tuple(shapes[name]):
                bad.append(name)
        if bad:
            raise SystemExit(f"restart {path} does not match this run: {', '.join(bad)}")
        month = str(a.get("month", ""))
        members = []
        for k in range(int(job["members"])):
            m = {n: np.array(ds[n][k]) for n in ("state_surface", "prev_state_surface", "state_level", "prev_state_level")}
            m["month_days"] = int(ds["month_days"][k])
            if month:
                m["month_sum_surface"] = np.array(ds["month_sum_surface"][k])
                m["month_sum_level"] = np.array(ds["month_sum_level"][k])
            elif m["month_days"]:
                raise SystemExit(f"restart {path}: member {k} has month_days without a month")
            members.append(m)
    return {"month": month, "step_index": int(a["step_index"]), "members": members}


def run_geoarches(out_dir: str, job: dict) -> dict:
    """ArchesWeather-Mx4 / ArchesWeatherGen rollout as described in arXiv:2605.29976 section 2.3.

    - initial state X_t = ERA5 daily mean of 1978-09-30, previous state 1978-09-29 (era5_1978_00h_sliced.nc);
    - each daily step predicts X_{t+1} from (X_t, X_{t-1}) and the monthly SST / sea-ice forcing of the month of
      t+1, conditioned on the day of year (and hour 0) of t, like geoarches' own multistep loop;
    - ArchesWeather-Mx4: the four ArchesWeather-M networks are averaged at every step;
    - ArchesWeatherGen: flow-matching sample (25 steps) around the Mx4 prediction; step i of member k uses the
      noise seed (seed + k) + 1000 i, the authors' sample_rollout scheme;
    - predicted SST and sea-ice cover are set to 0 (normalized) where the initial state has no ocean value before
      they are fed back, and written as NaN there;
    - steps are counted from 1978-09-30, also in a continuation (job["restart_file"]): it reads the two states and
      the unfinished month's sums from restart.nc and continues the same noise sequence, so the result equals one
      longer run. Every run that ends without an unstable month writes restart.nc.
    """
    _geoarches_code()
    import numpy as np
    import pandas as pd
    import torch

    from geoarches.dataloaders import nan_util
    from geoarches.dataloaders.era5 import Era5Forecast
    from tensordict import TensorDict

    from .emu_catalog import AW_RESTART, AW_START

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    torch.set_grad_enabled(False)
    t_load = time.time()
    norm = [f"stats.module.norm_file={job['norm_stats_file']}"]
    seeds = []
    for d in job["seed_dirs"]:
        m, seed_cfg, _ = _aw_module(d, norm, torch)
        seeds.append(m.to(dev))
    mx4 = _MeanOf(seeds, torch)
    gen_unused = None
    if job["kind"] == "gen":
        # The generative checkpoint also stores one deterministic network (det_model.*). The published AIMIP config
        # (load_deterministic_model = the four seeds) replaces it with ArchesWeather-Mx4, as geoarches'
        # load_module does; aimip builds the module without it and attaches the four verified seeds.
        dot = norm + [f"stats.module.residual_stats_path={job['residual_stats_file']}",
                      "module.module.load_deterministic_model=false"]
        gen, cfg, gen_unused = _aw_module(job["gen_dir"], dot, torch, allow_unexpected=("det_model.",))
        # geoarches pins tensordict < 0.14, where +/- between TensorDicts of different batch sizes act leaf by leaf.
        # tensordict 0.15 broadcasts the batch dimensions instead: in DiffusionModule.sample the decoder output (batch
        # size [1]) meets states that geoarches' own helpers give auto batch sizes ([1, 6]), and the shapes blow up.
        # Giving the decoder output and the Mx4 prediction auto batch sizes too restores the leaf-by-leaf arithmetic.
        decode = gen.embedder.decode
        gen.embedder.decode = lambda x: decode(x).auto_batch_size_()
        gen.det_model = lambda b: mx4(b).auto_batch_size_()
        gen = gen.to(dev)
        step_fn = lambda batch, i, member_seed: gen.sample(batch, seed=member_seed + 1000 * i, disable_tqdm=True)
    else:
        cfg = seed_cfg
        step_fn = lambda batch, i, member_seed: mx4(batch)
    load_s = time.time() - t_load

    ic_dir = os.path.join(os.environ.get("AIMIP_EMU_TMP") or tempfile.gettempdir(), "aw-ic")
    os.makedirs(ic_dir, exist_ok=True)
    link = os.path.join(ic_dir, os.path.basename(job["ic_file"]))
    if not os.path.lexists(link):
        os.symlink(job["ic_file"], link)
    ds = Era5Forecast(stats_cfg=cfg.stats, path=ic_dir + "/", domain="aimip_rollout_z00", lead_time_hours=24,
                      load_prev=True, multistep=1, variables=cfg.stats.module.variables,
                      dimension_indexers=dict(level=cfg.stats.module.levels), warning_on_nan=False,
                      interpolate_input="zero_after_norm", forcings_path=job["forcing_file"],
                      forcings_stats_path=job["forcing_stats_file"],
                      forcing_vars=["sea_surface_temperature", "sea_ice_cover"])
    ic = ds[0]
    t_aw = pd.Timestamp(int(ic["timestamp"]), unit="s")
    if t_aw != pd.Timestamp(AW_START):
        raise SystemExit(f"initial state is {t_aw}, expected {AW_START}")
    i0 = (pd.Timestamp(job["start"]) - t_aw).days          # global index of the first step (0 unless continuing)
    if job.get("restart_file") is None and i0 != 0:
        raise SystemExit(f"start {job['start']} without a restart")
    land = {k: ic["nan_mask"][k][None].to(dev) for k in ("surface", "level")}
    sv = list(cfg.stats.module.variables["surface"])
    lv = list(cfg.stats.module.variables["level"])
    levels = [float(x) for x in cfg.stats.module.levels]
    if set(sv) != set(AW_SURFACE_OUT) or set(lv) != set(AW_LEVEL_OUT):
        raise SystemExit(f"unexpected model variables {sv} / {lv}")
    if bool(land["level"].any()) or [i for i, v in enumerate(sv) if bool(land["surface"][0, i].any())
                                     and v not in ("sea_surface_temperature", "sea_ice_cover")]:
        raise SystemExit("the initial state has missing values outside SST / sea-ice cover")
    # aimip's own arithmetic below works on plain tensors (leaf by leaf), never on TensorDicts.
    mean = {k: ds.data_mean[k].to(dev) for k in ("surface", "level")}
    std = {k: ds.data_std[k].to(dev) for k in ("surface", "level")}

    forcing_cache: dict = {}

    def forcing_for(state_time):
        """Normalized SST / sea-ice forcing of the month of state_time + 1 day (geoarches' load_forcings)."""
        key = (state_time + pd.Timedelta(days=1)).strftime("%Y-%m")
        if key not in forcing_cache:
            forcing_cache.clear()
            f = ds.load_forcings(np.datetime64(state_time.to_datetime64()))
            f = nan_util.post_norm_interpolate_nans((f - ds.forcings_mean) / ds.forcings_std, "zero_after_norm")
            forcing_cache[key] = f[None].to(dev)
        return forcing_cache[key]

    # geoarches' convert_to_tensordict flips latitude to north-first and rolls longitude by half the globe ("focus on
    # Europe"); its convert_to_xarray undoes both. The writer below undoes the roll, so the files use the input grid.
    lat = np.sort(np.asarray(ds.coords["latitude"], dtype="f8"))[::-1]
    lon = np.asarray(ds.coords["longitude"], dtype="f8")
    half = len(lon) // 2
    restart = None
    if job.get("restart_file"):
        restart = _aw_read_restart(job["restart_file"], job, sv=sv, lv=lv, levels=levels,
                                   shapes={"state_surface": ic["state"]["surface"].shape,
                                           "state_level": ic["state"]["level"].shape})
        if restart["step_index"] != i0:
            raise SystemExit(f"restart step_index {restart['step_index']} != {i0}")
    first_day = t_aw + pd.Timedelta(days=i0 + 1)
    attrs = {"source": f"{job['model']} (geoarches, Zenodo 22795233), daily steps from the ERA5 daily mean of "
                       "1978-09-30", "aimip_family": "geoarches", "first_output_day": first_day.strftime("%Y-%m-%d"),
             "comment": "monthly means of the model's daily-mean states; sst and sic are the model's own daily "
                        "predictions (it also receives the monthly forcing), NaN over land; counts = days in the mean"}
    if restart is not None:
        attrs["continues"] = (f"{job['restart_file']} (step {i0}, state of {job['start'][:10]})"
                              + (f"; month {restart['month']} includes the days of the run it continues"
                                 if restart["month"] else ""))
    surf_names = [AW_SURFACE_OUT[v][0] for v in sv]
    lev_names = [AW_LEVEL_OUT[v][0] for v in lv]
    n_members = int(job["members"])
    ws = MonthlyWriter(os.path.join(out_dir, "atmosphere", "monthly_mean_predictions.nc"), surf_names, lat, lon,
                       n_members, attrs=attrs, extra={AW_SURFACE_OUT[v][0]: {"units": AW_SURFACE_OUT[v][1]} for v in sv})
    wl = MonthlyWriter(os.path.join(out_dir, "atmosphere", "monthly_levels.nc"), lev_names, lat, lon, n_members,
                       attrs=attrs, extra={AW_LEVEL_OUT[v][0]: {"units": AW_LEVEL_OUT[v][1]} for v in lv},
                       levels=levels, level_units="hPa")
    zg_i = lv.index("geopotential")
    w_lat = np.cos(np.deg2rad(lat))[:, None] * np.ones((1, len(lon)))

    def flush(member, month, acc, n):
        surf = (acc["surface"][0, :, 0] / n).float().cpu().numpy()
        lev = (acc["level"][0] / n).float().cpu().numpy()
        lev[zg_i] = lev[zg_i] / G0
        ocean = ~land["surface"][0, :, 0].cpu().numpy()
        bad = [v for i, v in enumerate(sv)
               if not np.isfinite(surf[i][ocean[i]] if v in ("sea_surface_temperature", "sea_ice_cover") else surf[i]).all()]
        bad += [v for i, v in enumerate(lv) if not np.isfinite(lev[i]).all()]
        if bad:
            raise Unstable(month, member, f"non-finite values in {', '.join(bad[:4])}")
        g = float((surf[sv.index("2m_temperature")] * w_lat).sum() / w_lat.sum())
        if not T2M_BAND_K[0] <= g <= T2M_BAND_K[1]:
            raise Unstable(month, member, f"global-mean tas {g:.1f} K outside {T2M_BAND_K[0]:.0f}-{T2M_BAND_K[1]:.0f} K")
        surf, lev = np.roll(surf, -half, axis=-1), np.roll(lev, -half, axis=-1)
        ws.write(member, month, {surf_names[i]: surf[i] for i in range(len(sv))}, n)
        wl.write(member, month, {lev_names[i]: lev[i] for i in range(len(lv))}, n)
        return g

    steps = int(job["steps"])
    t_end = t_aw + pd.Timedelta(days=i0 + steps)              # last output day = state time of the restart
    carry = (t_end + pd.Timedelta(days=1)).month == t_end.month
    saved: list = []
    unstable = None
    first_year: list = []
    tw = time.time()
    try:
        for member in range(n_members):
            member_seed = int(job["seed"]) + member
            torch.manual_seed(member_seed)
            batch = {k: v[None].to(dev) for k, v in ic.items() if k not in ("next_state", "nan_mask")}
            for k in ("state", "prev_state"):
                batch[k] = batch[k].auto_batch_size_()
            acc, n, month = None, 0, None
            if restart is not None:
                r = restart["members"][member]
                for k in ("state", "prev_state"):
                    # device=dev like the initial state's TensorDict: geoarches moves its constant masks to state.device
                    batch[k] = TensorDict(surface=torch.from_numpy(r[f"{k}_surface"])[None],
                                          level=torch.from_numpy(r[f"{k}_level"])[None], device=dev).auto_batch_size_()
                if r["month_days"]:
                    month = np.datetime64(restart["month"], "M")
                    acc = {k: torch.from_numpy(r[f"month_sum_{k}"])[None].to(dev) for k in ("surface", "level")}
                    n = r["month_days"]
            for i in range(i0, i0 + steps):
                t_cur = t_aw + pd.Timedelta(days=i)
                t_next = t_cur + pd.Timedelta(days=1)
                batch["forcings"] = forcing_for(t_cur)
                batch["timestamp"] = torch.tensor([int(t_cur.timestamp())], device=dev)
                batch["hour_of_day"] = torch.tensor([0], device=dev)
                batch["day_of_year"] = torch.tensor([t_cur.dayofyear], device=dev)
                batch["month"] = torch.tensor([t_cur.month], device=dev)
                pred = step_fn(batch, i, member_seed)
                phys = {k: (pred[k] * std[k] + mean[k]).double() for k in ("surface", "level")}
                phys["surface"] = torch.where(land["surface"], torch.nan, phys["surface"])
                m_next = np.datetime64(t_next.strftime("%Y-%m"), "M")
                if month is not None and m_next != month:
                    g = flush(member, month, acc, n)
                    if member == 0 and len(first_year) < 12:
                        first_year.append(g)
                    acc, n = None, 0
                month = m_next
                acc = phys if acc is None else {k: acc[k] + phys[k] for k in acc}
                n += 1
                fed = TensorDict(surface=torch.where(land["surface"], 0.0, pred["surface"]),
                                 level=pred["level"].clone()).auto_batch_size_()
                batch["prev_state"], batch["state"] = batch["state"], fed
                done = i + 1 - i0
                if done % 365 == 0:
                    _log(f"member {member}: {done} days, {(time.time() - tw) / (done + member * steps):.3f} s/day")
            saved.append({"state_surface": batch["state"]["surface"][0].cpu().numpy(),
                          "state_level": batch["state"]["level"][0].cpu().numpy(),
                          "prev_state_surface": batch["prev_state"]["surface"][0].cpu().numpy(),
                          "prev_state_level": batch["prev_state"]["level"][0].cpu().numpy(),
                          "month_days": n if carry else 0,
                          **({"month_sum_surface": acc["surface"][0].cpu().numpy(),
                              "month_sum_level": acc["level"][0].cpu().numpy()} if carry else {})})
            g = flush(member, month, acc, n)
            if member == 0 and len(first_year) < 12:
                first_year.append(g)
    except Unstable as exc:
        unstable = exc
        _log(f"UNSTABLE {exc}: the run stops; months before it are written")
    ws.close()
    wl.close()
    restart_path = None
    if unstable is None:
        restart_path = os.path.join(out_dir, AW_RESTART)
        _aw_write_restart(restart_path, job, saved, valid_time=t_end.strftime("%Y-%m-%dT%H:%M:%S"),
                          step_index=i0 + steps, month=t_end.strftime("%Y-%m") if carry else "", sv=sv, lv=lv,
                          levels=levels, lat_internal=lat, lon_internal=np.roll(lon, half))
    roll_s = time.time() - tw
    out = {"days_per_member": steps, "members": n_members, "load_seconds": round(load_s, 1),
           "rollout_seconds": round(roll_s, 1), "seconds_per_day": round(roll_s / max(1, steps * n_members), 4),
           "global_mean_tas_first_12_months_K": float(np.mean(first_year)) if first_year else None,
           "device": torch.cuda.get_device_name(0) if dev == "cuda" else "cpu"}
    if gen_unused is not None:
        out["gen_checkpoint_unused_det_model_keys"] = gen_unused
    out["first_step_index"] = i0
    if restart_path is not None:
        out["restart"] = {"path": restart_path, "valid_time": t_end.strftime("%Y-%m-%dT%H:%M:%S"),
                          "step_index": i0 + steps, "open_month": t_end.strftime("%Y-%m") if carry else None}
    if unstable is not None:
        out.update({"unstable_from": unstable.month, "unstable_member": unstable.member, "unstable_reason": unstable.why})
    return out


class _MeanOf:
    """Average of several geoarches modules' predictions at one step (ArchesWeather-Mx4)."""

    def __init__(self, modules, torch):
        self.modules, self.torch = list(modules), torch

    def __call__(self, batch):
        return self.torch.stack([m(batch) for m in self.modules], dim=0).mean(dim=0)


# ------------------------------------------------------------------ entry point

# ------------------------------------------------------------------ MD-1.5 (Monthly Diffusion v0.9)

# The checkpoint's tensors fit the published CVAE with these settings; its config.pkl was written by earlier code whose
# arguments the published constructor does not take (kl_weight1, rank, conditioning_rank, ...). Read from the shapes:
# encoder rank 64 / embed 128 / Driscoll-Healy conditioning, decoder rank 256 / embed 1024 / conditioning rank 4,
# diffusion rank 256 / embed 512 / Driscoll-Healy conditioning, seasonality basis 3.
MD_ARCH = {
    "seasonality_basis": 3,
    "encoder_sfno_embed_dim": 128, "encoder_rank": 64, "encoder_operator_type": "spectral_fc",
    "encoder_conditioning_operator_type": "driscoll-healy", "encoder_conditioning_rank": 4,
    "decoder_sfno_embed_dim": 1024, "decoder_rank": 256, "decoder_operator_type": "spectral_fc",
    "decoder_conditioning_operator_type": "spectral_fc", "decoder_conditioning_rank": 4,
    "diffusion_sfno_embed_dim": 512, "diffusion_rank": 256, "diffusion_operator_type": "spectral_fc",
    "diffusion_conditioning_operator_type": "driscoll-healy", "diffusion_conditioning_rank": 4,
}
MD_CONFIG_KEYS = ("forcing_dim", "data_dimensions", "activation", "n_statics", "statics", "hidden_channels",
                  "squeeze_factors", "variable_names", "seasonality_dim")
MD_LEVELS = [1000.0, 850.0, 700.0, 500.0, 250.0, 100.0, 50.0]
MD_SURFACE = {"tas": "VAR_2TSFC", "ts": "SKTSFC", "ps": "SPSFC", "psl": "MSLSFC", "uas": "VAR_10USFC",
              "vas": "VAR_10VSFC", "tdps": "VAR_2DSFC", "pr": "MTPRSFC", "zg500": "z500.0"}
MD_UNITS = {"tas": "K", "ts": "K", "ps": "Pa", "psl": "Pa", "uas": "m s-1", "vas": "m s-1", "tdps": "K",
            "pr": "kg m-2 s-1", "zg500": "m", "ta": "K", "hus": "kg kg-1", "ua": "m s-1", "va": "m s-1"}
GRAVITY = 9.80665


def _md_edges(c, clip):
    """The authors' `edges_from_centers` (data-pipeline notebooks): midpoints, half a cell beyond the ends, clipped."""
    import numpy as np

    c = np.asarray(c, dtype="f8")
    d = np.diff(c)
    e = np.concatenate(([c[0] - d[0] / 2], (c[:-1] + c[1:]) / 2, [c[-1] + d[0] / 2]))
    return np.clip(e, *clip)


def _md_overlap(dst_edges, src_edges, f):
    """Overlap weights (n_dst, n_src) of 1-D cells; f maps a coordinate to its cumulative measure."""
    import numpy as np

    lo = np.maximum(dst_edges[:-1, None], src_edges[None, :-1])
    hi = np.minimum(dst_edges[1:, None], src_edges[None, 1:])
    return np.where(hi > lo, f(hi) - f(lo), 0.0)


class MDForcing:
    """AIMIP SST, sea-ice cover and land-sea mask (Zenodo 17065758, 0.25 deg) on MD-1.5's grid, regridded as the
    authors' regrid_forcings.ipynb: first-order conservative with the valid fraction normalized (xESMF
    conservative_normed; mask of the first month for SST and sea ice), cell edges from `edges_from_centers`. On regular
    grids this is separable: area in sin(latitude) times longitude overlap."""

    def __init__(self, path: str):
        import netCDF4
        import numpy as np

        self.ds = netCDF4.Dataset(path)
        lat = np.asarray(self.ds["latitude"][:], "f8")
        lon = np.asarray(self.ds["longitude"][:], "f8")
        self.flip = lat[0] > lat[-1]
        lat = lat[::-1] if self.flip else lat
        self.lat_out, self.lon_out = np.linspace(-90, 90, 121), np.arange(0, 360, 1.5)
        sinf = lambda x: np.sin(np.deg2rad(x))  # noqa: E731
        self.wlat = _md_overlap(_md_edges(self.lat_out, (-90, 90)), _md_edges(lat, (-90, 90)), sinf)
        self.wlon = _md_overlap(_md_edges(self.lon_out, (0, 360)), _md_edges(lon, (0, 360)), lambda x: x)
        tv = self.ds["time"]
        dates = netCDF4.num2date(tv[:], tv.units, getattr(tv, "calendar", "standard"),
                                 only_use_cftime_datetimes=False, only_use_python_datetimes=True)
        self.months = [f"{d.year:04d}-{d.month:02d}" for d in dates]
        self.masks = {v: np.isfinite(self._raw(v, 0)).astype("f8") for v in ("sea_surface_temperature", "sea_ice_cover")}

    def _raw(self, var, i):
        import numpy as np

        v = self.ds[var]
        x = np.ma.filled(v[i] if v.ndim == 3 else v[:], np.nan).astype("f8")
        return x[::-1] if self.flip else x

    def _regrid(self, x, m):
        """Like ESMF's sparse weights: a NaN inside the mask makes only the destination cells it overlaps NaN (a dense
        product would spread it everywhere, since 0 * NaN = NaN)."""
        import numpy as np

        ok = (m > 0) & np.isfinite(x)
        num = self.wlat @ np.where(ok, x, 0.0) @ self.wlon.T
        den = self.wlat @ m @ self.wlon.T
        out = np.where(den > 0, num / np.where(den > 0, den, 1.0), np.nan)
        bad = (m > 0) & ~np.isfinite(x)
        if bad.any():
            out[(self.wlat @ bad.astype("f8") @ self.wlon.T) > 0] = np.nan
        return out

    def fields(self, months: list):
        """(len(months), 3, 121, 240): SSTKSFC, CISFC, LSMSFC (physical, NaN where no ocean cell overlaps)."""
        import numpy as np

        lsm = self._raw("land_sea_mask", 0)
        lsm_r = self._regrid(lsm, np.isfinite(lsm).astype("f8"))
        out = np.empty((len(months), 3, 121, 240))
        for k, mo in enumerate(months):
            i = self.months.index(mo)
            for j, v in enumerate(("sea_surface_temperature", "sea_ice_cover")):
                out[k, j] = self._regrid(self._raw(v, i), self.masks[v])
            out[k, 2] = lsm_r
        return out


class _MDConfigUnpickler:
    """config.pkl through an allow-list: the classes and functions of plain tensors, a GELU class and a device; the
    statics tensor nested in it is read with torch.load(weights_only=True)."""

    @staticmethod
    def load(path: str, device: str):
        import collections
        import io
        import pickle

        import numpy as np
        import torch

        allowed = {
            ("collections", "OrderedDict"): collections.OrderedDict,
            ("torch.nn.modules.activation", "GELU"): torch.nn.GELU,
            ("torch", "device"): torch.device,
            ("torch._utils", "_rebuild_tensor_v2"): torch._utils._rebuild_tensor_v2,
            ("torch.storage", "_load_from_bytes"):
                lambda b: torch.load(io.BytesIO(b), map_location=device, weights_only=True),
            ("numpy", "dtype"): np.dtype,
        }

        class U(pickle.Unpickler):
            def find_class(self, module, name):
                if (module, name) in allowed:
                    return allowed[(module, name)]
                if module in ("numpy._core.multiarray", "numpy.core.multiarray") and name == "scalar":
                    try:
                        from numpy._core.multiarray import scalar
                    except ImportError:            # numpy < 2
                        from numpy.core.multiarray import scalar
                    return scalar
                raise SystemExit(f"{path} refers to {module}.{name}, which is not allowed")

        with open(path, "rb") as fh:
            return U(fh).load()


def _md_module(code_dir: str):
    """Import the published `md` package from the pinned files without writing bytecode next to them."""
    import importlib
    import types

    sys.dont_write_bytecode = True
    sys.modules.setdefault("src", types.ModuleType("src"))     # md/prep_data.py imports a module named `src`
    sys.path.insert(0, code_dir)
    md = importlib.import_module("md")
    if os.path.realpath(os.path.dirname(md.__file__)) != os.path.realpath(os.path.join(code_dir, "md")):
        raise SystemExit(f"imported md from {md.__file__}, not from {code_dir}")
    return md


def _md_model(md, job: dict, device: str):
    import contextlib

    import torch

    config = _MDConfigUnpickler.load(job["config_file"], device)
    kw = {k: config[k] for k in MD_CONFIG_KEYS}
    kw.update(MD_ARCH, device=torch.device(device), loading=True)
    with contextlib.redirect_stdout(sys.stderr):
        model = md.CVAE(**kw)
    state = torch.load(job["checkpoint"], map_location=device, weights_only=True)
    unused = sorted(k for k in state if k.endswith("l2loss_sphere.quad_weights"))   # loss buffers of the old code
    res = model.load_state_dict({k: v for k, v in state.items() if k not in unused}, strict=False)
    if res.unexpected_keys or set(res.missing_keys) - {"predictor.latent_mu", "predictor.latent_log_sigma"}:
        raise SystemExit(f"checkpoint does not fit MD_ARCH: missing {res.missing_keys[:5]}, unexpected "
                         f"{res.unexpected_keys[:5]}")
    model.eval()
    return model, list(config["variable_names"][: config["data_dimensions"][0]]), unused


def run_md(out_dir: str, job: dict) -> dict:
    """The authors' scripts/run_aimip_runs.py and CVAE.forced_run, one member at a time (torch.manual_seed(seed + k);
    the authors ran five unseeded members in one batch): encode the October 1978 state, then every month sample the
    next latent mean with the diffusion model and decode it under that month's forcing. Step 0 is the decoded initial
    state; forcing normalization statistics are fitted on 1985-2014 of the forcing itself, as the authors did."""
    import json as _json

    import numpy as np
    import torch
    import xarray as xr

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    md = _md_module(job["code_dir"])
    model, state_names, unused = _md_model(md, job, dev)
    stats = _json.load(open(job["stats_file"], encoding="utf-8"))
    n_out = int(job["steps"]) + 1
    months = [str(np.datetime64("1978-10", "M") + np.timedelta64(i, "M")) for i in range(n_out)]

    forcing = MDForcing(job["sst_forcing"])
    lat, lon = forcing.lat_out, forcing.lon_out
    all_months = forcing.months[: forcing.months.index("2024-12") + 1]
    f_da = xr.DataArray(forcing.fields(all_months), dims=("time", "varlev", "lat", "lon"),
                        coords={"time": np.array([np.datetime64(m + "-01", "ns") for m in all_months]),
                                "varlev": ["SSTKSFC", "CISFC", "LSMSFC"], "lat": lat, "lon": lon})
    _, f_params, _ = md.preprocess_by_variables(f_da.sel(time=slice("1985-01-01", "2014-12-01")), group_levels=False)
    f_norm, _, _ = md.preprocess_by_variables(f_da, dict_to_undo=f_params, group_levels=False)
    f_norm = f_norm.fillna(0).transpose("time", "varlev", "lat", "lon").sel(time=f_da.time.values[:n_out])
    forcings = torch.tensor(f_norm.values, dtype=torch.float32, device=dev)

    with xr.open_dataset(job["ic_file"]) as icds:
        ic = icds["era5"].sel(varlev=state_names).load()
    if not np.allclose(ic.lat.values, lat) or not np.allclose(ic.lon.values, lon):
        raise SystemExit(f"{job['ic_file']}: grid differs from the model grid")
    ic_norm, _, _ = md.preprocess_by_variables(ic.expand_dims(time=[np.datetime64("1978-10-01", "ns")]),
                                               dict_to_undo=stats, group_levels=False)
    ic_t = torch.tensor(ic_norm.fillna(0).transpose("time", "varlev", "lat", "lon").values, dtype=torch.float32,
                        device=dev)
    month_no = torch.tensor([[float(np.datetime64(m, "M").astype(object).month)] for m in months], device=dev)
    template = xr.DataArray(np.zeros((1, len(state_names), len(lat), len(lon)), "f4"),
                            dims=("time", "varlev", "lat", "lon"),
                            coords={"varlev": state_names, "lat": lat, "lon": lon})

    attrs = {"source": "MD-1.5 (Monthly Diffusion v0.9, MD-1p5.pth; UMD PARETO), aimip", "start": job["start"],
             "aimip_family": "md",
             "comment": "monthly-mean states; the first month is the model's decoding of the ERA5 initial state; zg500 "
                        "= z500 / 9.80665; ta, hus, ua, va on 7 pressure levels in monthly_levels.nc"}
    n_members = int(job["members"])
    ws = MonthlyWriter(os.path.join(out_dir, "atmosphere", "monthly_mean_predictions.nc"), list(MD_SURFACE), lat,
                       lon, n_members, attrs=attrs, extra={k: {"units": MD_UNITS[k]} for k in MD_SURFACE})
    wl = MonthlyWriter(os.path.join(out_dir, "atmosphere", "monthly_levels.nc"), ["ta", "hus", "ua", "va"], lat, lon,
                       n_members, attrs=attrs, extra={k: {"units": MD_UNITS[k]} for k in ("ta", "hus", "ua", "va")},
                       levels=MD_LEVELS, level_units="hPa")
    idx = {n: i for i, n in enumerate(state_names)}
    w = np.cos(np.deg2rad(lat))[:, None] * np.ones((1, len(lon)))
    unstable = None

    def emit(member, step, decoded):
        phys = md.undo_preprocess_by_variables(template.copy(data=decoded), dict_to_undo=stats, group_levels=False)
        a = phys.transpose("time", "varlev", "lat", "lon").values[0]
        if not np.isfinite(a).all():
            raise Unstable(months[step], member, "non-finite values")
        g = float((a[idx["VAR_2TSFC"]] * w).sum() / w.sum())
        if not T2M_BAND_K[0] <= g <= T2M_BAND_K[1]:
            raise Unstable(months[step], member, f"global-mean tas {g:.1f} K outside the plausible band")
        fields = {k: a[idx[v]] for k, v in MD_SURFACE.items()}
        fields["zg500"] = fields["zg500"] / GRAVITY
        ws.write(member, months[step], fields, 1)
        wl.write(member, months[step], {n: np.stack([a[idx[f"{s}{lev}"]] for lev in MD_LEVELS])
                                        for n, s in (("ta", "t"), ("hus", "q"), ("ua", "u"), ("va", "v"))}, 1)

    try:
        with torch.no_grad():
            for k in range(n_members):
                seed = int(job["seed"]) + k
                torch.manual_seed(seed)
                tw = time.time()
                f0 = forcings[0:1]
                cond = torch.cat([f0, model.seasonality_embedding(month_no[0:1])], dim=1)
                mu, _, cs = model.encoder(ic_t, cond)
                emit(k, 0, model.decoder(mu, cs).cpu().numpy())
                for step in range(1, n_out):
                    mu = model.predictor(mu, cs[-1])
                    cond = torch.cat([forcings[step:step + 1], model.seasonality_embedding(month_no[step:step + 1])],
                                     dim=1)
                    cs = model.encoder.embed_conditions(cond)
                    emit(k, step, model.decoder(mu, cs).cpu().numpy())
                _log(f"member {k} seed {seed}: {n_out} months in {time.time() - tw:.1f}s")
    except Unstable as exc:
        unstable = exc
        _log(f"UNSTABLE {exc}: the run stops; months before it are written")
    ws.close()
    wl.close()
    out = {"members": n_members, "months_per_member": n_out, "device": dev, "unused_checkpoint_keys": unused,
           "forcing_normalization": f_params}
    if unstable is not None:
        out.update({"unstable_from": unstable.month, "unstable_member": unstable.member, "unstable_reason": unstable.why})
    return out


def main(argv: list) -> int:
    if len(argv) != 2 or argv[0] not in ("stage", "run"):
        print(__doc__, file=sys.stderr)
        return 2
    mode, out_dir = argv
    _setup_tmp()
    job = _load_job(out_dir)
    t0 = time.time()
    if mode == "stage":
        if job["family"] != "e2s-dlesym":
            raise SystemExit("only DLESyM needs a staged initial state")
        stage_dlesym(job["out_dir"], job)
        return 0
    if job["family"] == "e2s-dlesym":
        result = run_dlesym(job["out_dir"], job)
    elif job["family"] == "arches":
        result = run_arches(job["out_dir"], job)
    elif job["family"] == "geoarches":
        result = run_geoarches(job["out_dir"], job)
    elif job["family"] == "md":
        result = run_md(job["out_dir"], job)
    else:
        raise SystemExit(f"unknown family {job['family']!r}")
    result["process_seconds"] = round(time.time() - t0, 1)
    rc = EXIT_UNSTABLE if result.get("unstable_from") else 0
    target = os.path.join(job["out_dir"], "emu-result.json")
    with open(target + ".part", "w", encoding="utf-8") as fh:
        json.dump(result, fh, indent=2)
    os.replace(target + ".part", target)
    _log(f"DONE {job['model']} {json.dumps(result)}")
    return rc


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
