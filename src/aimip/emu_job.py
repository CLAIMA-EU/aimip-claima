"""Job side of the DLESyM / ArchesClimate-SSP families. Runs in the emulator venv (started by `aimip.emu`):

    python -I -m aimip.emu_job stage <run-dir>     DLESyM: ERA5 state at <start> from ARCO-ERA5 -> <run-dir>/ic_era5.pt
    python -I -m aimip.emu_job run <run-dir>       the rollout -> <run-dir>/{atmosphere,ocean}/monthly_mean_predictions.nc

Outputs follow the fme layout so `aimip summary` reads them: variables (sample, time, lat, lon), `valid_time`
(sample, time) and `counts` (sample, time). For ArchesClimate-SSP, dataset item 0 uses state 2015-02 with previous
state 2015-01, so the first output month is March 2015.
"""
from __future__ import annotations

import json
import os
import shutil
import sys
import tempfile
import time

# Never emit `.pyc` / `__pycache__` inside `$AIMIP_DATA/code/*` trees verified by `find -type f` SHA-256.
sys.dont_write_bytecode = True
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"

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
    """Monthly means of one realm, written month by month to <out>/<realm>/monthly_mean_predictions.nc."""

    def __init__(self, path: str, names: list, lat, lon, n_members: int, attrs: dict | None = None,
                 extra: dict | None = None):
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
            v = self.ds.createVariable(n, "f4", ("sample", "time", "lat", "lon"), zlib=True, complevel=1, shuffle=True,
                                       chunksizes=(1, 1, len(lat), len(lon)))
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
            self.vars[n][member, i, :, :] = arr

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
    staged = torch.load(os.path.join(out_dir, "ic_era5.pt"), weights_only=False)
    start = np.datetime64(job["start"])
    end = start + np.timedelta64(6 * int(job["steps"]), "h")
    hours_per_step = int(model.atmos_output_times[-1] / np.timedelta64(1, "h"))
    n_iter = int(np.ceil(6 * int(job["steps"]) / hours_per_step))
    n_members = int(job["members"])
    attrs = {"source": f"{job['model']} (Earth2Studio DLESyM, checkpoint {job['version']})", "start": job["start"],
             "aimip_family": "e2s-dlesym"}
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


# ------------------------------------------------------------------ entry point

def main(argv: list) -> int:
    if len(argv) != 2 or argv[0] not in ("stage", "run"):
        print(__doc__, file=sys.stderr)
        return 2
    mode, out_dir = argv
    _setup_tmp()
    job = _load_job(out_dir)
    t0 = time.time()
    if mode == "stage":
        if job["family"] == "e2s-dlesym":
            stage_dlesym(job["out_dir"], job)
            return 0
        if job["family"] == "samudra":
            from .samudra import stage_samudra

            stage_samudra(job["out_dir"], job)
            return 0
        raise SystemExit("only DLESyM and Samudra need staged initial/window data")
    if job["family"] == "e2s-dlesym":
        result = run_dlesym(job["out_dir"], job)
    elif job["family"] == "arches":
        result = run_arches(job["out_dir"], job)
    elif job["family"] == "samudra":
        from .samudra import run_samudra

        result = run_samudra(job["out_dir"], job)
    elif job["family"] == "md":
        from .md15 import run_md15

        result = run_md15(job["out_dir"], job)
    elif job["family"] == "camulator":
        from .camulator import run_camulator

        result = run_camulator(job["out_dir"], job)
    elif job["family"] == "cbottle":
        from .cbottle import run_cbottle

        result = run_cbottle(job["out_dir"], job)
    elif job["family"] == "sdyff":
        from .sdyff import run_sdyff

        result = run_sdyff(job["out_dir"], job)
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
