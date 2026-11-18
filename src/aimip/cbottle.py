"""cBottle-3d (`cbottle-3d`, conditional atmospheric state generator) adapter and runner.

Upstream provenance & licences:
- Code (Apache-2.0): `NVlabs/cBottle` commit `f0f519987589cc0b368ab892a1adfb2fb30a5587`
  (`183,593` B archive, SHA-256 `904e2e2e...`, 136 files, tree SHA-256 `a91c8256...`)
  wrapped by `NVIDIA/earth2studio@8fa7acb28b082e83a0718e8d2ed137fd375a11fe`
  (`earth2studio.data.CBottle3D`; Brenowitz et al., 2025, `arXiv:2505.06474`).
- Weights (Apache-2.0): `hf://nvidia/cbottle@e60ce82b236b953d79d3ead7f43445a3bb8e37d3`
  (`cBottle-3d/training-state-{000512000,002048000,009856000}.checkpoint`).
  Fetched from revision `e60ce82b` (whose card metadata and text declare Apache-2.0 only),
  never from Zenodo `20832634` (a different paper's evaluation-licensed artefact) and never
  via `CBottle3D.load_default_package()` (which names the older `eebd93c8` revision).
- Conditioning SST data (CC-BY-4.0): `amip_midmonth_sst.nc` (`PCMDI-AMIP-1-1-9`, `tosbcs`
  in `degC`, monthly midpoints `1870-01 .. 2022-12`; file global `license` attribute states
  CC BY 4.0).

Generator & security invariants:
- `cbottle-3d` is a conditional generator without state memory: every state is an independent
  diffusion draw conditioned only on `(day_of_year, hour_of_day)` and linearly interpolated
  monthly SST (`tosbcs`). Passing `--ic` or `--ocean-ic` is refused.
- Valid date window: `1940-01-01T00:00:00 .. 2022-12-16T00:00:00` (dates from `2018` onward
  lie outside the `1980-2017` ERA5 training range).
- Output depends on `(seed, batch_size, date_order)`: `batch_size = 4` and chronological
  date ordering (`daily_synoptic` rotating `00/06/12/18 UTC` by default, or `6h`) are fixed
  and recorded in `config.yaml`, `plan.json`, `aimip-manifest.json`, and NetCDF attributes.
- Every `.checkpoint` plain ZIP archive is opened via `O_NOFOLLOW`, verified by SHA-256 on
  the open descriptor, and checked for `metadata.json` + `weights_only=True` `net_state.pth`;
  `verified_cbottle_pretrained_loader` also wraps `zipfile.ZipFile` and `torch.load` around
  `MixtureOfExpertsDenoiser.from_pretrained` so upstream reads directly from the verified
  descriptors with `weights_only=True`.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping
import contextlib
import datetime
import hashlib
import io
import json
import os
import time
from typing import Any
import zipfile

from .errors import ConfigError, ValidationError
from .safe_pickle import normalize_sha256, open_verified_hash_stream
from .validators import validate_safe_path

CBOTTLE_CODE_REPO = "NVlabs/cBottle"
CBOTTLE_CODE_COMMIT = "f0f519987589cc0b368ab892a1adfb2fb30a5587"
CBOTTLE_CODE_LICENCE = "Apache-2.0"
CBOTTLE_CODE_ARCHIVE_URL = (
    f"https://github.com/{CBOTTLE_CODE_REPO}/archive/{CBOTTLE_CODE_COMMIT}.tar.gz"
)
CBOTTLE_CODE_ARCHIVE_SIZE = 183_593
CBOTTLE_CODE_ARCHIVE_SHA256 = (
    "904e2e2e563b0c035f961e235d4081c4800fc4b1ab2767a507016ae2f2144fc1"
)
CBOTTLE_TREE_FILES = 136
CBOTTLE_TREE_SHA256 = (
    "a91c82566c6499efc9f30dff5d9490b3472dc49c4b355420b6bc32465bda107d"
)

CBOTTLE_E2S_COMMIT = "8fa7acb28b082e83a0718e8d2ed137fd375a11fe"
CBOTTLE_HF_REPO = "nvidia/cbottle"
CBOTTLE_HF_REVISION = "e60ce82b236b953d79d3ead7f43445a3bb8e37d3"
CBOTTLE_WEIGHTS_LICENCE = "Apache-2.0"
CBOTTLE_DATA_LICENCE = "CC-BY-4.0"

CBOTTLE_EXPERT_STEPS: tuple[str, ...] = ("000512000", "002048000", "009856000")
CBOTTLE_SIGMA_THRESHOLDS: tuple[float, float] = (100.0, 10.0)

CBOTTLE_CKPT_512K_REL_PATH = "models/cbottle-3d/cBottle-3d/training-state-000512000.checkpoint"
CBOTTLE_CKPT_2048K_REL_PATH = "models/cbottle-3d/cBottle-3d/training-state-002048000.checkpoint"
CBOTTLE_CKPT_9856K_REL_PATH = "models/cbottle-3d/cBottle-3d/training-state-009856000.checkpoint"
CBOTTLE_SST_REL_PATH = "models/cbottle-3d/amip_midmonth_sst.nc"

CBOTTLE_CHECKPOINT_REL_PATHS: tuple[str, ...] = (
    CBOTTLE_CKPT_512K_REL_PATH,
    CBOTTLE_CKPT_2048K_REL_PATH,
    CBOTTLE_CKPT_9856K_REL_PATH,
)

CBOTTLE_FILE_SPECS: dict[str, dict[str, Any]] = {
    "cBottle-3d/training-state-000512000.checkpoint": {
        "rel_path": CBOTTLE_CKPT_512K_REL_PATH,
        "size": 595_471_094,
        "sha256": "0f3c78a49ac331ff5fcb503f282eec02aaa75ad2df5564e2591463dc5dda930f",
    },
    "cBottle-3d/training-state-002048000.checkpoint": {
        "rel_path": CBOTTLE_CKPT_2048K_REL_PATH,
        "size": 595_471_094,
        "sha256": "a037294f096333a2ab9f4a73778bf5cb5600332698845fc67b6717f33ccf754f",
    },
    "cBottle-3d/training-state-009856000.checkpoint": {
        "rel_path": CBOTTLE_CKPT_9856K_REL_PATH,
        "size": 1_786_464_750,
        "sha256": "95894e3e04e10e0e33d807b9605f33180bf1ca6feb1a79ccded23c57c40c97d1",
    },
    "amip_midmonth_sst.nc": {
        "rel_path": CBOTTLE_SST_REL_PATH,
        "size": 476_174_130,
        "sha256": "388d06b45aea206c3eda982f7cf7d3020d607ed685874951d5b62dd87a6919b9",
    },
}

CBOTTLE_FORCING_NAME = "pcmdi-amip-1-1-9-sst"
CBOTTLE_WINDOW: tuple[str, str] = ("1940-01-01T00:00:00", "2022-12-16T00:00:00")
CBOTTLE_ERA5_TRAINING_YEARS: tuple[int, int] = (1980, 2017)

CBOTTLE_BATCH_SIZE = 4
CBOTTLE_DEFAULT_CADENCE = "daily_synoptic"
CBOTTLE_ALLOWED_CADENCES: tuple[str, ...] = ("daily_synoptic", "6h")
CBOTTLE_DATE_ORDER = "chronological"
CBOTTLE_SEED_SCHEME = "sha256_uint31(aimip:cbottle-3d:v1:{seed}:{member_index}:{batch_index})"


def cbottle_batch_seed(run_seed: int, member_index: int, batch_index: int) -> int:
    """Derive a disjoint 31-bit non-negative seed from `(run_seed, member_index, batch_index)`.

    Earth2Studio's `CBottle3D.__call__` resets its local batch counter on each invocation and
    passes `self.seed + i` to the diffusion sampler. Deriving a deterministic SHA-256 seed per
    `(run_seed, member_index, batch_index)` guarantees independent diffusion noise across all
    batches and ensemble members regardless of `PYTHONHASHSEED`.
    """
    token = f"aimip:cbottle-3d:v1:{int(run_seed)}:{int(member_index)}:{int(batch_index)}".encode(
        "ascii"
    )
    raw64 = int.from_bytes(hashlib.sha256(token).digest()[:8], "big")
    return int(raw64 % 2_147_483_647)

CBOTTLE_OUTPUT_VARS: tuple[str, ...] = (
    "t2m",
    "msl",
    "tcwv",
    "u10m",
    "v10m",
    "z500",
    "t850",
    "u850",
    "v850",
    "tpf",
)

CBOTTLE_VAR_META: dict[str, dict[str, str]] = {
    "t2m": {"units": "K", "long_name": "2 m air temperature"},
    "msl": {"units": "Pa", "long_name": "Mean sea level pressure"},
    "tcwv": {"units": "kg m**-2", "long_name": "Total column water vapour"},
    "u10m": {"units": "m s**-1", "long_name": "10 m eastward wind"},
    "v10m": {"units": "m s**-1", "long_name": "10 m northward wind"},
    "z500": {"units": "m**2 s**-2", "long_name": "500 hPa geopotential"},
    "t850": {"units": "K", "long_name": "850 hPa air temperature"},
    "u850": {"units": "m s**-1", "long_name": "850 hPa eastward wind"},
    "v850": {"units": "m s**-1", "long_name": "850 hPa northward wind"},
    "tpf": {"units": "kg m**-2 s**-1", "long_name": "Total precipitation flux"},
}

CBOTTLE_REQUIRED_ZIP_MEMBERS: tuple[str, ...] = (
    "net_state.pth",
    "model.json",
    "batch_info.json",
    "metadata.json",
)

CBOTTLE_SECONDS_PER_YEAR = 150.0
CBOTTLE_T2M_BAND_K: tuple[float, float] = (200.0, 330.0)
EXIT_UNSTABLE = 5
TIME_UNITS = "days since 1970-01-01 00:00:00"

_MIN_DT = datetime.datetime(1940, 1, 1, 0, 0, 0)
_MAX_WINDOW_END_DT = datetime.datetime(2022, 12, 16, 0, 0, 0)
_E2S_STRICT_UPPER_DT = datetime.datetime(2022, 12, 16, 12, 0, 0)


def parse_cbottle_iso(iso_text: str) -> datetime.datetime:
    """Parse canonical `YYYY-MM-DDTHH:MM:SS` in proleptic Gregorian calendar."""
    try:
        return datetime.datetime.strptime(iso_text.strip(), "%Y-%m-%dT%H:%M:%S")
    except ValueError as exc:
        raise ConfigError(
            f"Invalid timestamp {iso_text!r} for cbottle-3d (expected YYYY-MM-DDTHH:MM:SS).",
            field="start",
        ) from exc


def cbottle_sample_timestamps(
    start_iso: str,
    steps: int,
    *,
    cadence: str = CBOTTLE_DEFAULT_CADENCE,
) -> list[datetime.datetime]:
    """Build the deterministic chronological list of sample datetimes for `cbottle-3d`.

    - `daily_synoptic` (default): `steps` counts calendar days (`1` sample/day) starting at
      `00:00:00 UTC`, with the synoptic hour rotating `00/06/12/18 UTC`
      (`start + (24*i + 6*(i % 4))h`) so monthly means sample the diurnal cycle evenly.
    - `6h`: `4 * steps` 6-hourly draws (`start + 6*k h` for `k = 0 .. 4*steps - 1`).
    """
    if cadence not in CBOTTLE_ALLOWED_CADENCES:
        raise ConfigError(
            f"Unsupported cbottle-3d cadence {cadence!r} (allowed: {', '.join(CBOTTLE_ALLOWED_CADENCES)}).",
            field="cadence",
            allowed=list(CBOTTLE_ALLOWED_CADENCES),
        )
    if not isinstance(steps, int) or isinstance(steps, bool) or steps < 1:
        raise ConfigError("cbottle-3d --steps must be an integer >= 1.", field="steps")

    start_dt = parse_cbottle_iso(start_iso)
    if start_dt.minute != 0 or start_dt.second != 0:
        raise ConfigError(
            f"cbottle-3d --start {start_iso} must lie on an exact UTC hour.",
            field="start",
        )
    if cadence == "daily_synoptic" and start_dt.hour != 0:
        raise ConfigError(
            f"cbottle-3d with cadence='daily_synoptic' requires --start at 00:00:00 UTC "
            f"(got {start_iso}; e.g. --start {start_dt.strftime('%Y-%m-%d')}T00:00:00).",
            field="start",
        )
    if cadence == "6h" and start_dt.hour not in (0, 6, 12, 18):
        raise ConfigError(
            f"cbottle-3d --start hour must be one of 00, 06, 12, 18 UTC (got {start_iso}).",
            field="start",
        )

    end_dt = start_dt + datetime.timedelta(hours=24 * steps)
    if start_dt < _MIN_DT or end_dt > _MAX_WINDOW_END_DT:
        raise ConfigError(
            f"cbottle-3d valid date range is {CBOTTLE_WINDOW[0]} .. {CBOTTLE_WINDOW[1]} "
            f"(governed by amip_midmonth_sst.nc; requested {start_iso} + {steps}d -> "
            f"{end_dt.strftime('%Y-%m-%dT%H:%M:%S')}).",
            field="start" if start_dt < _MIN_DT else "end",
        )

    if cadence == "daily_synoptic":
        times = [
            start_dt + datetime.timedelta(hours=24 * i + 6 * (i % 4))
            for i in range(steps)
        ]
    else:
        times = [
            start_dt + datetime.timedelta(hours=6 * k)
            for k in range(4 * steps)
        ]

    for dt in times:
        if dt < _MIN_DT or dt >= _E2S_STRICT_UPPER_DT:
            raise ConfigError(
                f"cbottle-3d sample timestamp {dt.strftime('%Y-%m-%dT%H:%M:%S')} falls outside "
                f"supported SST range [{CBOTTLE_WINDOW[0]} .. {CBOTTLE_WINDOW[1]}].",
                field="end",
            )
    return times


def verify_cbottle_checkpoint_zip(
    path: str,
    expected_sha256: str,
    *,
    label: str = "cBottle-3d checkpoint",
) -> dict[str, Any]:
    """Verify `.checkpoint` SHA-256 on an `O_NOFOLLOW` fd and validate `net_state.pth` via `weights_only=True`.

    Upstream `cBottle-3d/training-state-*.checkpoint` files are plain ZIP archives (not PyTorch
    subdirectory archives) containing `net_state.pth`, `model.json`, `batch_info.json`, and
    `metadata.json` (plus `optimizer_state.pth` in `009856000`). This pre-flight gate hashes the
    `.checkpoint` file before parsing ZIP headers and verifies `net_state.pth` loads cleanly with
    `torch.load(..., map_location="cpu", weights_only=True)`; `verified_cbottle_pretrained_loader`
    then enforces the same `O_NOFOLLOW` SHA-256 + `weights_only=True` invariants on the handles
    read by `MixtureOfExpertsDenoiser.from_pretrained`.
    """
    import torch

    with open_verified_hash_stream(path, expected_sha256, label=label) as stream:
        try:
            zf_ctx = zipfile.ZipFile(stream, mode="r")
        except (zipfile.BadZipFile, OSError) as exc:
            raise ValidationError(
                f"Invalid ZIP structure in {label} ({path}): {exc}"
            ) from exc

        with zf_ctx as zf:
            names = zf.namelist()
            for name in names:
                if (
                    not name
                    or name.startswith("/")
                    or "\\" in name
                    or ".." in name.split("/")
                ):
                    raise ValidationError(
                        f"Unsafe member path {name!r} inside {label} ({path})."
                    )
            missing = [req for req in CBOTTLE_REQUIRED_ZIP_MEMBERS if req not in names]
            if missing:
                raise ValidationError(
                    f"{label} ({path}) is missing required archive member(s): {', '.join(missing)}."
                )

            try:
                model_cfg = json.loads(zf.read("model.json").decode("utf-8"))
                batch_info = json.loads(zf.read("batch_info.json").decode("utf-8"))
                metadata = json.loads(zf.read("metadata.json").decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError, KeyError) as exc:
                raise ValidationError(
                    f"Corrupted JSON metadata in {label} ({path}): {exc}"
                ) from exc

            if not isinstance(model_cfg, dict) or int(model_cfg.get("out_channels", 0)) != 45:
                raise ValidationError(
                    f"Unexpected model.json out_channels in {label} ({path}): {model_cfg!r}"
                )
            channels = batch_info.get("channels") if isinstance(batch_info, dict) else None
            if not isinstance(channels, list) or len(channels) != 45:
                raise ValidationError(
                    f"Unexpected batch_info.json channel count in {label} ({path})."
                )

            with zf.open("net_state.pth", mode="r") as net_fh:
                net_bytes = net_fh.read()
            state_dict = torch.load(
                io.BytesIO(net_bytes),
                map_location="cpu",
                weights_only=True,
            )
            if not isinstance(state_dict, Mapping) or len(state_dict) == 0:
                raise ValidationError(
                    f"Expected non-empty tensor mapping in {label} net_state.pth ({path})."
                )
            n_tensors = len(state_dict)
            del state_dict, net_bytes

    return {
        "architecture": model_cfg.get("architecture"),
        "out_channels": 45,
        "metadata_version": metadata.get("version") if isinstance(metadata, dict) else None,
        "n_tensors": n_tensors,
    }


def _extract_cbottle_load_path(target: Any) -> str | None:
    if isinstance(target, (str, bytes, os.PathLike)):
        return os.fsdecode(target)
    if isinstance(target, zipfile.ZipExtFile):
        return None
    fs_path = getattr(target, "path", None)
    if isinstance(fs_path, (str, bytes, os.PathLike)):
        return os.fsdecode(fs_path)
    fh_name = getattr(target, "name", None)
    if isinstance(fh_name, (str, bytes, os.PathLike)):
        return os.fsdecode(fh_name)
    return None


@contextlib.contextmanager
def verified_cbottle_pretrained_loader(
    expected_sha256_by_path: Mapping[str, str],
    *,
    require_all_loaded: bool = True,
) -> Iterator[set[str]]:
    """Ensure `MixtureOfExpertsDenoiser.from_pretrained` reads verified `O_NOFOLLOW` ZIP streams with `weights_only=True`."""
    import torch

    if not expected_sha256_by_path:
        raise ValidationError("cBottle-3d verified loader requires a non-empty checkpoint SHA-256 map.")
    digest_map: dict[str, str] = {}
    expected_by_realpath: dict[str, str] = {}
    for raw_p, raw_sha in expected_sha256_by_path.items():
        safe_p = validate_safe_path(str(raw_p), label="cBottle-3d checkpoint")
        if os.path.islink(safe_p):
            raise ValidationError(f"Refusing symlinked cBottle-3d checkpoint: {safe_p}")
        norm_sha = normalize_sha256(
            str(raw_sha),
            label=f"cBottle-3d {os.path.basename(safe_p)} SHA-256",
        )
        abs_s = os.path.abspath(safe_p)
        real_s = os.path.realpath(safe_p)
        digest_map[abs_s] = norm_sha
        digest_map[real_s] = norm_sha
        expected_by_realpath[real_s] = abs_s

    verified_zip_realpaths: set[str] = set()
    verified_torch_loads = 0

    orig_zipfile = zipfile.ZipFile
    orig_load = getattr(torch, "_aimip_orig_torch_load", torch.load)
    orig_ser_load = getattr(torch.serialization, "_aimip_orig_torch_load", torch.serialization.load)

    with contextlib.ExitStack() as stack:

        def _verified_zipfile(file: Any, mode: str = "r", *args: Any, **kwargs: Any) -> Any:
            cand_path = _extract_cbottle_load_path(file)
            if cand_path is None:
                return orig_zipfile(file, mode, *args, **kwargs)
            if mode != "r":
                raise ValidationError(
                    f"Refusing non-read ZipFile mode {mode!r} in cBottle-3d checkpoint loader: {cand_path}"
                )
            abs_p = os.path.abspath(cand_path)
            want_sha = digest_map.get(abs_p)
            if want_sha is None:
                raise ValidationError(
                    f"Refusing unverified cBottle-3d checkpoint archive: {abs_p}"
                )

            stream_ctx = open_verified_hash_stream(
                abs_p,
                want_sha,
                label=f"cBottle-3d {os.path.basename(abs_p)}",
            )
            stream = stream_ctx.__enter__()
            closed = False

            def _close_stream() -> None:
                nonlocal closed
                if not closed:
                    closed = True
                    stream_ctx.__exit__(None, None, None)

            stack.callback(_close_stream)
            try:
                zf = orig_zipfile(stream, mode, *args, **kwargs)
                names = zf.namelist()
                for member_name in names:
                    if (
                        not member_name
                        or member_name.startswith("/")
                        or "\\" in member_name
                        or ".." in member_name.split("/")
                    ):
                        zf.close()
                        raise ValidationError(
                            f"Unsafe member path {member_name!r} inside cBottle-3d archive ({abs_p})."
                        )
                missing = [req for req in CBOTTLE_REQUIRED_ZIP_MEMBERS if req not in names]
                if missing:
                    zf.close()
                    raise ValidationError(
                        f"cBottle-3d archive ({abs_p}) is missing required member(s): {', '.join(missing)}."
                    )
            except BaseException:
                _close_stream()
                raise

            verified_zip_realpaths.add(os.path.realpath(abs_p))
            orig_zf_close = zf.close

            def _wrapped_close() -> None:
                try:
                    orig_zf_close()
                finally:
                    _close_stream()

            zf.close = _wrapped_close  # type: ignore[method-assign]
            return zf

        def _verified_torch_load(f: Any, *args: Any, **kwargs: Any) -> Any:
            nonlocal verified_torch_loads
            if kwargs.get("weights_only") is False:
                raise ValidationError(
                    "Refusing torch.load(..., weights_only=False) inside cBottle-3d checkpoint loader."
                )
            cand_path = _extract_cbottle_load_path(f)
            if cand_path is not None:
                raise ValidationError(
                    f"Refusing direct path torch.load inside cBottle-3d checkpoint loader: {cand_path}"
                )
            kwargs["weights_only"] = True
            loaded_obj = orig_load(f, *args, **kwargs)
            verified_torch_loads += 1
            return loaded_obj

        zipfile.ZipFile = _verified_zipfile  # type: ignore[misc]
        torch.load = _verified_torch_load  # type: ignore[assignment]
        torch.serialization.load = _verified_torch_load  # type: ignore[assignment]
        try:
            yield verified_zip_realpaths
        finally:
            zipfile.ZipFile = orig_zipfile  # type: ignore[misc]
            torch.load = orig_load  # type: ignore[assignment]
            torch.serialization.load = orig_ser_load  # type: ignore[assignment]

        if require_all_loaded:
            missing_archives = [
                disp for real_p, disp in expected_by_realpath.items()
                if real_p not in verified_zip_realpaths
            ]
            if missing_archives:
                raise ValidationError(
                    "cBottle-3d pretrained loader bypassed verified ZipFile wrapper for checkpoint(s): "
                    f"{', '.join(missing_archives)}"
                )
            if verified_torch_loads < len(expected_by_realpath):
                raise ValidationError(
                    "cBottle-3d pretrained loader bypassed verified torch.load wrapper "
                    f"(expected >= {len(expected_by_realpath)} state loads, got {verified_torch_loads})."
                )


def _write_cbottle_monthly_netcdf(
    out_path: str,
    *,
    var_names: list[str],
    lat: Any,
    lon: Any,
    months: list[str],
    monthly_fields: dict[tuple[int, str], Any],
    monthly_counts: dict[tuple[int, str], int],
    n_members: int,
    attrs: Mapping[str, Any],
) -> None:
    """Write CF-1.8 `<out_dir>/monthly_mean_predictions.nc` in standard `(sample, time, lat, lon)` layout."""
    import netCDF4
    import numpy as np

    lat_arr = np.asarray(lat, dtype=np.float64)
    lon_arr = np.asarray(lon, dtype=np.float64)
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    tmp_path = f"{out_path}.part-{os.getpid()}"
    try:
        with netCDF4.Dataset(tmp_path, "w", format="NETCDF4") as ds:
            ds.createDimension("sample", int(n_members))
            ds.createDimension("time", len(months))
            ds.createDimension("lat", int(lat_arr.size))
            ds.createDimension("lon", int(lon_arr.size))

            v_lat = ds.createVariable("lat", "f8", ("lat",))
            v_lat[:] = lat_arr
            v_lat.units = "degrees_north"
            v_lat.standard_name = "latitude"

            v_lon = ds.createVariable("lon", "f8", ("lon",))
            v_lon[:] = lon_arr
            v_lon.units = "degrees_east"
            v_lon.standard_name = "longitude"

            v_time = ds.createVariable("time", "f8", ("time",))
            v_time.units = TIME_UNITS
            v_time.calendar = "proleptic_gregorian"
            v_time.long_name = "first day of the month"

            v_vt = ds.createVariable("valid_time", "f8", ("sample", "time"))
            v_vt.units = TIME_UNITS
            v_vt.calendar = "proleptic_gregorian"

            v_cnt = ds.createVariable("counts", "i4", ("sample", "time"))
            v_cnt.long_name = "number of independent conditional draws averaged in month"

            nc_vars: dict[str, Any] = {}
            for vname in var_names:
                meta = CBOTTLE_VAR_META.get(vname, {})
                v = ds.createVariable(
                    vname,
                    "f4",
                    ("sample", "time", "lat", "lon"),
                    zlib=True,
                    complevel=1,
                    shuffle=True,
                    chunksizes=(1, 1, int(lat_arr.size), int(lon_arr.size)),
                )
                for mk, mv in meta.items():
                    setattr(v, mk, mv)
                nc_vars[vname] = v

            epoch = datetime.date(1970, 1, 1)
            for t_idx, ym in enumerate(months):
                y_i, m_i = (int(x) for x in ym.split("-"))
                day_offset = float((datetime.date(y_i, m_i, 1) - epoch).days)
                v_time[t_idx] = day_offset
                for m_idx in range(n_members):
                    v_vt[m_idx, t_idx] = day_offset
                    v_cnt[m_idx, t_idx] = int(monthly_counts[(m_idx, ym)])
                    arr = monthly_fields[(m_idx, ym)]
                    for v_i, vname in enumerate(var_names):
                        nc_vars[vname][m_idx, t_idx, :, :] = arr[v_i]

            for ak, av in attrs.items():
                setattr(ds, ak, av)
        os.replace(tmp_path, out_path)
    finally:
        if os.path.exists(tmp_path):
            try:
                os.unlink(tmp_path)
            except OSError:
                pass


def run_cbottle(out_dir: str, job: Mapping[str, Any]) -> dict[str, Any]:
    """Verify all cBottle-3d ZIP checkpoints + SST on `O_NOFOLLOW` fds and sample monthly means."""
    import numpy as np
    import torch
    import xarray as xr

    requested_device = str(job.get("device", "cuda"))
    if requested_device.startswith("cuda"):
        if not torch.cuda.is_available():
            raise SystemExit(
                f"{requested_device} requested for cbottle-3d, but CUDA is unavailable in this job; "
                "refusing silent CPU fallback."
            )
        device = requested_device
    elif requested_device == "cpu":
        device = "cpu"
    else:
        raise SystemExit(f"Unsupported device {requested_device!r} for cbottle-3d.")

    validated_out = validate_safe_path(out_dir, label="cbottle-3d out_dir")
    ckpt_entries = job.get("expert_checkpoints")
    if not isinstance(ckpt_entries, list) or len(ckpt_entries) != 3:
        raise ValidationError("cbottle-3d job config requires 3 expert_checkpoints entries.")

    ckpt_paths: list[str] = []
    ckpt_digest_map: dict[str, str] = {}
    verified_experts: list[dict[str, Any]] = []
    for idx, entry in enumerate(ckpt_entries):
        if not isinstance(entry, Mapping):
            raise ValidationError(f"Invalid expert_checkpoints[{idx}] entry in cbottle-3d job config.")
        c_path = validate_safe_path(str(entry["path"]), label=f"cbottle-3d expert checkpoint {idx}")
        c_sha = normalize_sha256(str(entry["sha256"]), label=f"cbottle-3d expert checkpoint {idx} SHA-256")
        meta = verify_cbottle_checkpoint_zip(
            c_path,
            c_sha,
            label=f"cBottle-3d expert checkpoint {os.path.basename(c_path)}",
        )
        ckpt_paths.append(c_path)
        ckpt_digest_map[c_path] = c_sha
        verified_experts.append(meta)

    sst_path = validate_safe_path(str(job["sst_file"]), label="cbottle-3d amip_midmonth_sst.nc")
    sst_sha = str(job["sst_sha256"])
    with open_verified_hash_stream(sst_path, sst_sha, label="amip_midmonth_sst.nc") as _sst_stream:
        pass

    start_iso = str(job["start"])
    steps = int(job["steps"])
    members = int(job.get("members", 1))
    seed = int(job.get("seed", 0))
    batch_size = int(job.get("batch_size", CBOTTLE_BATCH_SIZE))
    cadence = str(job.get("cadence", CBOTTLE_DEFAULT_CADENCE))
    if batch_size != CBOTTLE_BATCH_SIZE:
        raise ValidationError(
            f"cbottle-3d batch_size is fixed to {CBOTTLE_BATCH_SIZE} for reproducibility (got {batch_size})."
        )

    sample_dts = cbottle_sample_timestamps(start_iso, steps, cadence=cadence)

    from earth2studio.data import CBottle3D
    import earth2studio.data.cbottle as cb_mod

    t0 = time.time()
    with verified_cbottle_pretrained_loader(ckpt_digest_map):
        core = cb_mod.MixtureOfExpertsDenoiser.from_pretrained(
            ckpt_paths,
            CBOTTLE_SIGMA_THRESHOLDS,
        )
    with xr.open_dataset(sst_path, engine="netcdf4") as sst_raw:
        sst_ds = sst_raw.load()

    avail_vars = set(str(v) for v in getattr(CBottle3D, "VARIABLES", CBOTTLE_OUTPUT_VARS))
    var_names = [v for v in CBOTTLE_OUTPUT_VARS if v in avail_vars]
    if not var_names or "t2m" not in var_names:
        raise ValidationError(
            f"CBottle3D.VARIABLES is missing required cbottle-3d variables (got {sorted(avail_vars)})."
        )

    lat_vals: Any = None
    lon_vals: Any = None
    months_order: list[str] = []
    monthly_fields: dict[tuple[int, str], Any] = {}
    monthly_counts: dict[tuple[int, str], int] = {}

    def _check_and_store_month(m_idx: int, ym: str, sum_arr: Any, count: int, lat_a: Any) -> None:
        mean_arr = (sum_arr / float(count)).astype(np.float32)
        if not np.isfinite(mean_arr).all():
            bad = [var_names[i] for i in range(len(var_names)) if not np.isfinite(mean_arr[i]).all()]
            print(
                f"[aimip.cbottle] UNSTABLE member={m_idx} month={ym}: non-finite values in {', '.join(bad[:4])}",
                flush=True,
            )
            raise SystemExit(EXIT_UNSTABLE)
        t2m_idx = var_names.index("t2m")
        w = np.maximum(0.0, np.cos(np.deg2rad(np.asarray(lat_a, dtype=np.float64))))[:, None]
        w2d = np.broadcast_to(w, mean_arr[t2m_idx].shape)
        g_t2m = float(np.sum(mean_arr[t2m_idx] * w2d) / np.sum(w2d))
        if not (CBOTTLE_T2M_BAND_K[0] <= g_t2m <= CBOTTLE_T2M_BAND_K[1]):
            print(
                f"[aimip.cbottle] UNSTABLE member={m_idx} month={ym}: global-mean t2m={g_t2m:.2f} K "
                f"outside [{CBOTTLE_T2M_BAND_K[0]:.0f}, {CBOTTLE_T2M_BAND_K[1]:.0f}] K",
                flush=True,
            )
            raise SystemExit(EXIT_UNSTABLE)
        if ym not in months_order:
            months_order.append(ym)
        monthly_fields[(m_idx, ym)] = mean_arr
        monthly_counts[(m_idx, ym)] = int(count)

    batch_seeds_by_member: list[list[int]] = []
    for m_idx in range(members):
        member_batch_seeds: list[int] = []
        init_seed = cbottle_batch_seed(seed, m_idx, 0)
        generator = CBottle3D(
            core,
            sst_ds,
            lat_lon=True,
            seed=init_seed,
            batch_size=batch_size,
        ).to(device)

        cur_ym: str | None = None
        cur_sum: Any = None
        cur_n = 0

        for batch_idx, b_start in enumerate(range(0, len(sample_dts), batch_size)):
            batch_tt = sample_dts[b_start : b_start + batch_size]
            batch_seed = cbottle_batch_seed(seed, m_idx, batch_idx)
            member_batch_seeds.append(batch_seed)
            generator.seed = batch_seed
            np.random.seed(batch_seed)
            torch.manual_seed(batch_seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(batch_seed)
            da = generator(batch_tt, var_names)
            vals = np.asarray(da.values, dtype=np.float64)
            if vals.ndim != 4 or vals.shape[0] != len(batch_tt) or vals.shape[1] != len(var_names):
                raise ValidationError(
                    f"Unexpected CBottle3D output shape {vals.shape} for batch size {len(batch_tt)} "
                    f"and {len(var_names)} variables."
                )
            if not np.isfinite(vals).all():
                print(
                    f"[aimip.cbottle] UNSTABLE member={m_idx} at {batch_tt[0].isoformat()}: "
                    "non-finite sample returned by CBottle3D",
                    flush=True,
                )
                raise SystemExit(EXIT_UNSTABLE)

            if lat_vals is None:
                lat_vals = np.asarray(da["lat"].values, dtype=np.float64)
                lon_vals = np.asarray(da["lon"].values, dtype=np.float64)

            for j, dt in enumerate(batch_tt):
                ym = f"{dt.year:04d}-{dt.month:02d}"
                if cur_ym is not None and ym != cur_ym:
                    _check_and_store_month(m_idx, cur_ym, cur_sum, cur_n, lat_vals)
                    cur_sum = None
                    cur_n = 0
                if cur_sum is None:
                    cur_ym = ym
                    cur_sum = np.zeros_like(vals[j], dtype=np.float64)
                    cur_n = 0
                cur_sum += vals[j]
                cur_n += 1

        if cur_ym is not None and cur_sum is not None and cur_n > 0:
            _check_and_store_month(m_idx, cur_ym, cur_sum, cur_n, lat_vals)
        batch_seeds_by_member.append(member_batch_seeds)

    if device.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.synchronize()
    infer_s = round(time.time() - t0, 3)

    out_nc = os.path.join(validated_out, "monthly_mean_predictions.nc")
    _write_cbottle_monthly_netcdf(
        out_nc,
        var_names=var_names,
        lat=lat_vals,
        lon=lon_vals,
        months=months_order,
        monthly_fields=monthly_fields,
        monthly_counts=monthly_counts,
        n_members=members,
        attrs={
            "Conventions": "CF-1.8",
            "source": f"cBottle-3d ({CBOTTLE_CODE_REPO}@{CBOTTLE_CODE_COMMIT[:8]}, earth2studio@{CBOTTLE_E2S_COMMIT[:8]})",
            "weights_source": f"hf://{CBOTTLE_HF_REPO}@{CBOTTLE_HF_REVISION}",
            "sst_source_id": "PCMDI-AMIP-1-1-9",
            "sst_licence": CBOTTLE_DATA_LICENCE,
            "generator_semantics": (
                "conditional atmospheric state generator; adjacent days are independent diffusion draws "
                "conditioned only on (day_of_year, hour_of_day) and monthly mid-point SST (tosbcs)"
            ),
            "seed": int(seed),
            "seed_scheme": CBOTTLE_SEED_SCHEME,
            "batch_size": int(batch_size),
            "cadence": cadence,
            "date_order": CBOTTLE_DATE_ORDER,
        },
    )

    result = {
        "family": "cbottle",
        "model": "cbottle-3d",
        "months_written": {"atmosphere": len(months_order)},
        "months": months_order,
        "samples_per_member": len(sample_dts),
        "members": members,
        "seed": seed,
        "seed_scheme": CBOTTLE_SEED_SCHEME,
        "batch_seeds": batch_seeds_by_member,
        "batch_size": batch_size,
        "cadence": cadence,
        "date_order": CBOTTLE_DATE_ORDER,
        "variables": var_names,
        "verified_experts": verified_experts,
        "inference_seconds": infer_s,
    }
    res_path = os.path.join(validated_out, "emu-result.json")
    tmp_res = f"{res_path}.tmp-{os.getpid()}"
    with open(tmp_res, "w", encoding="utf-8") as fh:
        json.dump(result, fh, indent=2)
        fh.write("\n")
    os.replace(tmp_res, res_path)
    return result
