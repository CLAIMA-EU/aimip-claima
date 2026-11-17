"""Deterministic NetCDF builders for AIMIP initial-condition and forcing datasets.

Derived datasets built reproducibly against `aimip.lock`:
- `build_hiro_ic`, `build_hiro_forcing`, `build_eamv3_ic`, `build_eamv3_forcing`:
  re-stamped 1994 initial conditions and historical AMIP forcing files on each model grid.
- `build_co2_table`:
  global-mean CO2 mole fraction table (1979-2024, 6-hourly) derived from `allenai/ACE2-ERA5`
  with a local-file fast path and pinned repository commit metadata.
- `extract_zip_member`: one file of a pinned zip archive, unchanged (ArchesWeather checkpoints and configs).
- `build_oos_forcing` (`shieldplus`, `hiro`, `eamv3`):
  out-of-sample AIMIP SST/SIC forcing (first-order conservative 0.25 deg -> 1 deg Gaussian
  regridding in `sin(lat)` and periodic longitude, nearest-neighbor coastal fill, and linear
  interpolation between month starts).

Determinism guarantees:
- Fixed variable insertion order and coordinate order on every dataset.
- Zero wallclock timestamps (`datetime.now`, `date_created`, dynamic `history`) in netCDF
  global or variable attributes; optional `pinned_attrs` hook can override any attribute.
- Explicit `engine="netcdf4"` (`libnetcdf` sets `H5Pset_obj_track_times(..., 0)` so HDF5
  object headers carry zero creation/modification timestamps).
- Atomic writes via sibling `.tmp` / `.part` files followed by `os.replace`.
"""

from __future__ import annotations

import calendar as py_calendar
import datetime
import os
import shutil
import time
import uuid
from collections.abc import Callable, Iterable, Sequence
from typing import Any

from .errors import ConfigError, ValidationError
from .validators import validate_safe_path

HIST_YEARS_DEFAULT: tuple[int, ...] = tuple(range(1994, 2015))
OOS_YEARS_DEFAULT: tuple[int, ...] = tuple(range(2015, 2025))

CO2_SEPT_REFERENCE: dict[str, float] = {
    "min_ppm": 335.81,
    "max_ppm": 418.20,
    "slope_ppm_yr": 2.33,
    "end_ppm": 422.90,
}


def _atomic_to_netcdf(
    ds: Any,
    out_path: str,
    *,
    units: str,
    calendar: str,
    zlib: bool = False,
    complevel: int = 1,
) -> None:
    """Write *ds* to *out_path* deterministically via a sibling temporary file."""
    validated_out = validate_safe_path(out_path, label="builder output netCDF")
    parent = os.path.dirname(validated_out) or "/"
    os.makedirs(parent, exist_ok=True)
    tmp_path = os.path.join(parent, f".build_{uuid.uuid4().hex}.nc.tmp")

    enc: dict[str, dict[str, Any]] = {}
    for var_name in ds.data_vars:
        if zlib:
            enc[str(var_name)] = {
                "dtype": "float32",
                "zlib": True,
                "complevel": int(complevel),
            }
        else:
            enc[str(var_name)] = {"dtype": "float32"}
    enc["time"] = {
        "units": units,
        "calendar": calendar,
        "dtype": "float64",
    }

    try:
        ds.to_netcdf(tmp_path, engine="netcdf4", encoding=enc)
        os.replace(tmp_path, validated_out)
    finally:
        if os.path.exists(tmp_path):
            try:
                os.unlink(tmp_path)
            except OSError:
                pass


def _link_or_copy_atomic(src_path: str, dst_path: str) -> None:
    """Hardlink *src_path* to *dst_path* when on the same filesystem, else copy atomically."""
    validated_src = validate_safe_path(os.path.realpath(src_path), label="source netCDF")  # never hard-link a symlink
    validated_dst = validate_safe_path(dst_path, label="destination netCDF")
    if not os.path.isfile(validated_src):
        raise ValidationError(f"Builder input file not found: {validated_src}")

    parent = os.path.dirname(validated_dst) or "/"
    os.makedirs(parent, exist_ok=True)
    tmp_path = os.path.join(parent, f".copy_{uuid.uuid4().hex}.nc.tmp")
    try:
        try:
            os.link(validated_src, tmp_path)
        except OSError:
            shutil.copyfile(validated_src, tmp_path)
        os.replace(tmp_path, validated_dst)
    finally:
        if os.path.exists(tmp_path):
            try:
                os.unlink(tmp_path)
            except OSError:
                pass


def _nearest_valid_indices_2d(invalid_mask: Any) -> tuple[Any, Any]:
    """Return `(iy, ix)` of nearest `False` cell for every cell in 2D boolean *invalid_mask*.

    Strictly requires `scipy.ndimage.distance_transform_edt`: never falls back to a
    different Euclidean tie-breaking implementation, ensuring deterministic coastal fills.
    """
    import numpy as np

    try:
        from scipy.ndimage import distance_transform_edt
    except ImportError as exc:
        raise ConfigError(
            "scipy (`scipy.ndimage.distance_transform_edt`) is required to build derived "
            f"forcing datasets, but failed to import: {exc}"
        ) from exc

    mask_bool = np.asarray(invalid_mask, dtype=bool)
    _, (iy, ix) = distance_transform_edt(mask_bool, return_indices=True)
    return iy, ix


def build_hiro_ic(
    raw_ic_2014_path: str,
    out_ic_path: str,
    *,
    force: bool = False,
    pinned_attrs: dict[str, str] | None = None,
) -> str:
    """Re-stamp HiRO `ic_2014.nc` state 0 to `1994-01-01T00:00:00`."""
    import numpy as np
    import xarray as xr

    src = validate_safe_path(raw_ic_2014_path, label="HiRO raw ic_2014.nc")
    dst = validate_safe_path(out_ic_path, label="HiRO ic_1994.nc")
    if os.path.isfile(dst) and not force:
        return dst
    if not os.path.isfile(src):
        raise ValidationError(f"HiRO raw IC file not found: {src}")

    with xr.open_dataset(src, engine="netcdf4") as raw_ds:
        ic = raw_ds.isel(time=[0]).load()
    ic = ic.assign_coords(time=[np.datetime64("1994-01-01T00:00:00", "ns")])
    if pinned_attrs:
        ic.attrs.update(pinned_attrs)
    _atomic_to_netcdf(
        ic,
        dst,
        units="hours since 1979-01-01T00:00:00",
        calendar="proleptic_gregorian",
    )
    return dst


def build_hiro_forcing(
    *,
    raw_hiro_forcing_2014_path: str,
    shield_amip_hist_dir: str,
    out_dir: str,
    years: Iterable[int] = HIST_YEARS_DEFAULT,
    mask_reference_year: int = 1994,
    force: bool = False,
    pinned_attrs: dict[str, str] | None = None,
) -> list[str]:
    """Build HiRO historical forcing files with coastal SST refill and HiRO static fields."""
    import numpy as np
    import xarray as xr

    raw_2014 = validate_safe_path(
        raw_hiro_forcing_2014_path, label="HiRO raw forcing_2014.nc"
    )
    shield_dir = validate_safe_path(shield_amip_hist_dir, label="SHiELD amip-hist dir")
    validated_out_dir = validate_safe_path(out_dir, label="HiRO amip-hist output dir")

    years_list = [int(y) for y in years]
    if not years_list:
        return []
    if not os.path.isfile(raw_2014):
        raise ValidationError(f"HiRO raw forcing_2014.nc not found: {raw_2014}")

    os.makedirs(validated_out_dir, exist_ok=True)
    ref_shield_year = int(mask_reference_year)
    ref_shield_path = os.path.join(shield_dir, f"forcing_{ref_shield_year}.nc")
    if not os.path.isfile(ref_shield_path):
        raise ValidationError(f"SHiELD reference forcing not found: {ref_shield_path}")

    with xr.open_dataset(raw_2014, engine="netcdf4") as pk_full:
        pk = pk_full.isel(time=0).load()
    land = pk["land_fraction"].values
    lat_vals = pk["latitude"].values
    lon_vals = pk["longitude"].values
    stat = {
        str(v): pk[v].drop_vars("time", errors="ignore")
        for v in pk.data_vars
        if str(v).startswith(("ak_", "bk_")) or str(v) == "HGTsfc"
    }

    with xr.open_dataset(ref_shield_path, engine="netcdf4") as s_ref:
        s_land = s_ref["land_fraction"].isel(time=0).values

    bad = (s_land >= 0.5) & (land < 0.5)
    iy, ix = _nearest_valid_indices_2d(s_land >= 0.5)

    written: list[str] = []
    for y in years_list:
        out_file = validate_safe_path(
            f"{validated_out_dir}/forcing_{y}.nc", label=f"HiRO forcing_{y}.nc"
        )
        if os.path.isfile(out_file) and not force:
            written.append(out_file)
            continue

        src_shield = os.path.join(shield_dir, f"forcing_{y}.nc")
        if not os.path.isfile(src_shield):
            raise ValidationError(f"Missing SHiELD forcing input: {src_shield}")

        with xr.open_dataset(src_shield, engine="netcdf4") as s_raw:
            s = s_raw.rename({"grid_yt": "latitude", "grid_xt": "longitude"}).load()

        if not (
            np.allclose(s.latitude.values, lat_vals, atol=1e-5)
            and np.allclose(s.longitude.values, lon_vals, atol=1e-5)
        ):
            raise ConfigError(
                f"Grid mismatch between {src_shield} and HiRO {raw_2014}."
            )

        s = s.assign_coords(latitude=lat_vals, longitude=lon_vals)
        ice = np.minimum(s["sea_ice_fraction"].values, 1.0 - land[None])
        ice = np.clip(ice, 0.0, 1.0)
        nt = int(s.sizes["time"])
        st = s["surface_temperature"].values.copy()
        st[:, bad] = st[:, iy[bad], ix[bad]]

        ds = xr.Dataset(
            {
                "surface_temperature": (("time", "latitude", "longitude"), st),
                "DSWRFtoa": s["DSWRFtoa"],
                "global_mean_co2": s["global_mean_co2"],
                "sea_ice_fraction": (("time", "latitude", "longitude"), ice),
                "land_fraction": (
                    ("time", "latitude", "longitude"),
                    np.broadcast_to(land, (nt,) + land.shape).copy(),
                ),
                "ocean_fraction": (
                    ("time", "latitude", "longitude"),
                    np.clip(1.0 - land[None] - ice, 0.0, 1.0),
                ),
            },
            coords={
                "time": s.time,
                "latitude": lat_vals,
                "longitude": lon_vals,
            },
        )
        for k, v in stat.items():
            ds[k] = v
        ds.attrs = {
            "hist20": (
                "hybrid v2: SHiELD+ AMIP time-varying fields, HiRO statics, "
                "coastal SST from nearest SHiELD+ ocean cell"
            )
        }
        if pinned_attrs:
            ds.attrs.update(pinned_attrs)

        _atomic_to_netcdf(
            ds,
            out_file,
            units="hours since 1979-01-01T00:00:00",
            calendar="proleptic_gregorian",
        )
        written.append(out_file)

    return written


def build_eamv3_ic(
    raw_ic_1971_path: str,
    out_ic_path: str,
    *,
    force: bool = False,
    pinned_attrs: dict[str, str] | None = None,
) -> str:
    """Re-stamp EAMv3 `1971010100.nc` state 0 to noleap `1994-01-01`."""
    import cftime
    import xarray as xr
    from xarray.coding.times import CFDatetimeCoder

    src = validate_safe_path(raw_ic_1971_path, label="EAMv3 raw 1971010100.nc")
    dst = validate_safe_path(out_ic_path, label="EAMv3 ic_1994.nc")
    if os.path.isfile(dst) and not force:
        return dst
    if not os.path.isfile(src):
        raise ValidationError(f"EAMv3 raw IC file not found: {src}")

    with xr.open_dataset(
        src,
        engine="netcdf4",
        decode_times=CFDatetimeCoder(use_cftime=True),
        decode_timedelta=False,
    ) as raw_ds:
        ic = raw_ds.isel(time=[0]).load()
    ic = ic.assign_coords(time=[cftime.DatetimeNoLeap(1994, 1, 1)])
    if pinned_attrs:
        ic.attrs.update(pinned_attrs)
    _atomic_to_netcdf(
        ic,
        dst,
        units="hours since 1970-01-01T00:00:00",
        calendar="noleap",
    )
    return dst


def build_eamv3_forcing(
    *,
    raw_eamv3_1971_path: str,
    era5_amip_hist_dir: str,
    out_dir: str,
    years: Iterable[int] = HIST_YEARS_DEFAULT,
    expected_steps_per_year: int | None = None,
    force: bool = False,
    pinned_attrs: dict[str, str] | None = None,
) -> list[str]:
    """Build EAMv3 noleap 1994-2014 forcing files (`{y}.nc`)."""
    import cftime
    import numpy as np
    import pandas as pd
    import xarray as xr

    raw_1971 = validate_safe_path(raw_eamv3_1971_path, label="EAMv3 raw 1971.nc")
    era5_dir = validate_safe_path(era5_amip_hist_dir, label="ACE2-ERA5 amip-hist dir")
    validated_out_dir = validate_safe_path(out_dir, label="EAMv3 amip-hist output dir")

    years_list = [int(y) for y in years]
    if not years_list:
        return []
    if not os.path.isfile(raw_1971):
        raise ValidationError(f"EAMv3 raw 1971.nc not found: {raw_1971}")

    os.makedirs(validated_out_dir, exist_ok=True)
    with xr.open_dataset(raw_1971, engine="netcdf4") as pk:
        pk.load()
        land = pk["LANDFRAC"].values[0]
        solin = pk["SOLIN"].values.astype("float32")
        lat_vals = pk["lat"].values
        lon_vals = pk["lon"].values
        stat = {
            str(v): (
                pk[v].isel(time=0, drop=True)
                if "time" in pk[v].dims
                else pk[v].copy(deep=True)
            )
            for v in pk.data_vars
            if str(v).startswith(("ak_", "bk_")) or str(v) == "PHIS"
        }

    written: list[str] = []
    for y in years_list:
        out_file = validate_safe_path(
            f"{validated_out_dir}/{y}.nc", label=f"EAMv3 {y}.nc"
        )
        if os.path.isfile(out_file) and not force:
            written.append(out_file)
            continue

        src_era5 = os.path.join(era5_dir, f"forcing_{y}.nc")
        if not os.path.isfile(src_era5):
            raise ValidationError(f"Missing ACE2-ERA5 forcing input: {src_era5}")

        with xr.open_dataset(src_era5, engine="netcdf4") as e_raw:
            t_vals = e_raw.time.values
            dt_idx = pd.DatetimeIndex(t_vals)
            keep = ~((dt_idx.month == 2) & (dt_idx.day == 29))
            e = e_raw.isel(time=np.where(keep)[0]).load()

        nt = int(e.sizes["time"])
        if expected_steps_per_year is not None and nt != expected_steps_per_year:
            raise ConfigError(
                f"Expected {expected_steps_per_year} noleap steps for year {y} in {src_era5}, got {nt}."
            )
        if nt == 0 or solin.shape[0] != nt:
            raise ConfigError(
                f"EAMv3 SOLIN step count {solin.shape[0]} does not match noleap step count {nt} "
                f"for year {y} in {src_era5} (raw {raw_1971})."
            )
        if not (
            np.allclose(e.latitude.values, lat_vals, atol=1e-5)
            and np.allclose(e.longitude.values, lon_vals, atol=1e-5)
        ):
            raise ConfigError(f"Grid mismatch between {src_era5} and EAMv3 {raw_1971}.")

        tt = [
            cftime.DatetimeNoLeap(y, 1, 1) + datetime.timedelta(hours=6 * i)
            for i in range(nt)
        ]
        ice = np.clip(
            np.minimum(e["sea_ice_fraction"].values, 1.0 - land[None]), 0.0, 1.0
        )
        ds = xr.Dataset(
            {
                "TS": (("time", "lat", "lon"), e["surface_temperature"].values),
                "ICEFRAC": (("time", "lat", "lon"), ice),
                "OCNFRAC": (
                    ("time", "lat", "lon"),
                    np.clip(1.0 - land[None] - ice, 0.0, 1.0),
                ),
                "LANDFRAC": (
                    ("time", "lat", "lon"),
                    np.broadcast_to(land, (nt,) + land.shape).copy(),
                ),
                "SOLIN": (("time", "lat", "lon"), solin),
            },
            coords={
                "time": tt,
                "lat": lat_vals,
                "lon": lon_vals,
            },
        )
        for k, v in stat.items():
            ds[k] = v
        ds.attrs = {
            "hist20": (
                "built: ERA5 TS/sea ice (Ai2 ACE2-ERA5 forcing), "
                "EAMv3 statics and 1971 SOLIN cycle, noleap"
            )
        }
        if pinned_attrs:
            ds.attrs.update(pinned_attrs)

        _atomic_to_netcdf(
            ds,
            out_file,
            units="hours since 1970-01-01T00:00:00",
            calendar="noleap",
        )
        written.append(out_file)

    return written


def _resolve_local_mirror_dir(raw_mirror: str | None) -> str | None:
    """Parse an optional local directory or `file:///...` URI into a validated path."""
    if not raw_mirror:
        return None
    val = raw_mirror.strip()
    if val.startswith("file://"):
        val = val[len("file://") :]
    if not val:
        return None
    return validate_safe_path(val, label="local mirror directory")


def _read_remote_co2_year(
    repo: str,
    revision: str,
    year: int,
    *,
    hf_fs: Any = None,
) -> Any:
    """Read `global_mean_co2` from `allenai/ACE2-ERA5` via local mirror or HTTP range requests."""
    import xarray as xr

    mirror_root = _resolve_local_mirror_dir(os.environ.get("AIMIP_HF_MIRROR"))
    if mirror_root is not None:
        for repo_dir in (repo, repo.replace("/", "__")):
            mirror_file = os.path.join(
                mirror_root, repo_dir, "forcing_data", f"forcing_{year}.nc"
            )
            if os.path.isfile(mirror_file):
                with xr.open_dataset(mirror_file, engine="netcdf4") as ds_mir:
                    return ds_mir["global_mean_co2"].load()

    if hf_fs is None:
        from huggingface_hub import HfFileSystem

        hf_fs = HfFileSystem()

    rev_arg = None if revision == "REVISION_TBD" else revision
    last_exc: Exception | None = None
    for attempt in range(5):
        try:
            with hf_fs.open(
                f"{repo}/forcing_data/forcing_{year}.nc",
                "rb",
                revision=rev_arg,
                block_size=1 << 20,
            ) as fh:
                with xr.open_dataset(fh, engine="h5netcdf") as ds:
                    return ds["global_mean_co2"].load()
        except Exception as exc:  # noqa: BLE001
            last_exc = exc
            time.sleep(min(2.0, 0.5 * (attempt + 1)))
    raise ConfigError(
        f"Failed ranged read of {repo}/forcing_data/forcing_{year}.nc@{revision} after 5 tries: {last_exc}"
    )


def build_co2_table(
    out_path: str,
    *,
    repo: str = "allenai/ACE2-ERA5",
    revision: str = "f25123c52d7faa8f3533816a311cf3353ee83daa",
    local_era5_forcing_dir: str | None = None,
    y0: int = 1979,
    y1: int = 2024,
    last_upstream: int = 2022,
    year_reader: Callable[[int], Any] | None = None,
    validate_reference_stats: bool = True,
    force: bool = False,
    pinned_attrs: dict[str, str] | None = None,
) -> str:
    """Build the 6-hourly global-mean CO2 mole fraction table with pinned commit metadata."""
    import numpy as np
    import xarray as xr

    validated_out = validate_safe_path(out_path, label="co2_global_mean.nc")
    if os.path.isfile(validated_out) and not force:
        return validated_out

    local_dir = (
        validate_safe_path(local_era5_forcing_dir, label="local ACE2-ERA5 forcing dir")
        if local_era5_forcing_dir
        else None
    )

    hf_fs: Any = None
    parts: list[Any] = []
    for y in range(y0, last_upstream + 1):
        if year_reader is not None:
            parts.append(year_reader(y))
            continue
        local_cand = (
            os.path.join(local_dir, f"forcing_{y}.nc") if local_dir is not None else None
        )
        if local_cand and os.path.isfile(local_cand):
            with xr.open_dataset(local_cand, engine="netcdf4") as ds_loc:
                parts.append(ds_loc["global_mean_co2"].load())
        else:
            if hf_fs is None:
                from huggingface_hub import HfFileSystem

                hf_fs = HfFileSystem()
            parts.append(_read_remote_co2_year(repo, revision, y, hf_fs=hf_fs))

    s = xr.concat(parts, dim="time").sortby("time")
    t = s.time.values.astype("datetime64[ns]")
    step = np.diff(t).astype("timedelta64[h]").astype(int)
    if not bool((step == 6).all()):
        raise ConfigError(
            f"CO2 series is not uniformly 6-hourly; unique step hours: {np.unique(step).tolist()}."
        )
    last = t[-1]

    tail_start = str(np.datetime64(last, "Y") - 1)
    tail = s.sel(time=slice(tail_start, None))
    x = (tail.time.values - tail.time.values[0]).astype("timedelta64[h]").astype(float)
    slope, icpt = np.polyfit(x, tail.values.astype(float), 1)
    ext_t = np.arange(
        last + np.timedelta64(6, "h"),
        np.datetime64(f"{y1 + 1}-01-01T00", "ns"),
        np.timedelta64(6, "h"),
    ).astype("datetime64[ns]")
    ext_x = (ext_t - tail.time.values[0]).astype("timedelta64[h]").astype(float)
    ext_v = icpt + slope * ext_x

    vals = np.concatenate([s.values.astype(np.float32), ext_v.astype(np.float32)])
    time_ = np.concatenate([t, ext_t])
    want = np.arange(
        np.datetime64(f"{y0}-01-01T00", "ns"),
        np.datetime64(f"{y1 + 1}-01-01T00", "ns"),
        np.timedelta64(6, "h"),
    ).astype("datetime64[ns]")
    if time_.size != want.size or not bool((time_ == want).all()):
        raise ConfigError(
            f"CO2 coverage mismatch vs {y0}-{y1} 6-hourly ({time_.size} vs {want.size})."
        )
    if not (2.5e-4 < float(vals.min()) < 5.0e-4 and 2.5e-4 < float(vals.max()) < 5.0e-4):
        raise ConfigError(
            f"CO2 values outside plausible mole-fraction range [2.5e-4, 5.0e-4]: "
            f"min={float(vals.min()):.6g}, max={float(vals.max()):.6g}."
        )

    if validate_reference_stats and (y0, y1, last_upstream) == (1979, 2024, 2022):
        now_stats = {
            "min_ppm": float(s.min()) * 1e6,
            "max_ppm": float(s.max()) * 1e6,
            "slope_ppm_yr": float(slope * 24 * 365.25 * 1e6),
            "end_ppm": float(ext_v[-1] * 1e6),
        }
        for key, expected in CO2_SEPT_REFERENCE.items():
            if abs(round(now_stats[key], 2) - expected) > 0.011:
                raise ConfigError(
                    f"CO2 reference check failed for {key}: got {now_stats[key]:.3f}, "
                    f"expected {expected:.2f}."
                )

    da = xr.DataArray(vals, dims="time", coords={"time": time_}, name="global_mean_co2")
    da.attrs.update(
        units="1",
        long_name="global mean CO2 mole fraction as used by ACE2-ERA5",
        units_note="dimensionless mole fraction, NOT ppm; multiply by 1e6 for ppm",
        source=(
            f"Ai2 {repo} forcing_data/forcing_YYYY.nc, variable global_mean_co2, "
            f"{y0}-{last_upstream}, repository commit {revision}"
        ),
        extrapolated_from=str(last),
        extrapolation="linear fit over the last 24 months of the Ai2 series",
        extrapolation_slope_ppm_per_year=float(slope * 24 * 365.25 * 1e6),
        note=(
            "Ai2's store has no 2023/2024 forcing files; 2023-2024 values are extrapolated (see attrs). "
            "Rebuilt 2026-10-19 by build_co2_table_ranged.py (range reads of global_mean_co2 only), after the purge; "
            "the series and the extrapolation are build_co2_table_direct.py's."
        ),
    )
    if pinned_attrs:
        da.attrs.update(pinned_attrs)

    parent = os.path.dirname(validated_out) or "/"
    os.makedirs(parent, exist_ok=True)
    part_path = os.path.join(parent, f".co2_{uuid.uuid4().hex}.nc.part")
    try:
        da.to_netcdf(part_path, engine="netcdf4")
        os.replace(part_path, validated_out)
    finally:
        if os.path.exists(part_path):
            try:
                os.unlink(part_path)
            except OSError:
                pass

    return validated_out


def _interval_overlap(src_lo: Any, src_hi: Any, dst_lo: Any, dst_hi: Any) -> Any:
    """Matrix `(dst, src)` of interval overlap lengths."""
    import numpy as np

    return np.clip(
        np.minimum(dst_hi[:, None], src_hi[None])
        - np.maximum(dst_lo[:, None], src_lo[None]),
        0.0,
        None,
    )


def _conservative_weights(
    lat_t: Any,
    lon_t: Any,
    lat_s: Any,
    lon_s: Any,
    *,
    src_half_deg: float = 0.125,
) -> tuple[Any, Any]:
    """First-order conservative weights in `sin(lat)` and periodic longitude."""
    import numpy as np

    sorted_lat = np.sort(lat_t)
    if not np.all(lat_t == sorted_lat):
        raise ConfigError("Target latitude coordinate must be ascending.")
    bt = np.concatenate([[-90.0], (sorted_lat[1:] + sorted_lat[:-1]) / 2.0, [90.0]])
    sl = np.sin(np.deg2rad(np.clip(lat_s - src_half_deg, -90.0, 90.0)))
    sh = np.sin(np.deg2rad(np.clip(lat_s + src_half_deg, -90.0, 90.0)))
    w_lat = _interval_overlap(
        sl,
        sh,
        np.sin(np.deg2rad(bt[:-1])),
        np.sin(np.deg2rad(bt[1:])),
    )
    dl = float(np.diff(lon_t).mean())
    if not np.allclose(np.diff(lon_t), dl):
        raise ConfigError("Target longitude coordinate must be uniformly spaced.")
    tl = lon_t - dl / 2.0
    th = lon_t + dl / 2.0
    w_lon = sum(
        _interval_overlap(lon_s - src_half_deg + k, lon_s + src_half_deg + k, tl, th)
        for k in (-360.0, 0.0, 360.0)
    )
    return w_lat, w_lon


def _fill_nearest(grid_2d: Any) -> Any:
    """Nearest-neighbor NaN fill over the 2-D `(lat, lon)` grid without periodic longitude wrapping.

    Runs `distance_transform_edt` directly on `grid_2d` so derived out-of-sample forcing files
    reproduce the SHA-256 digests pinned in `aimip.lock`.
    """
    import numpy as np

    if not np.isfinite(grid_2d).any():
        raise ConfigError("No finite source value to fill from in AIMIP SST/SIC field.")
    iy, ix = _nearest_valid_indices_2d(~np.isfinite(grid_2d))
    return grid_2d[iy, ix]


def _monthly_aimip_on_grid(
    aimip_nc_path: str,
    lat_t: Any,
    lon_t: Any,
    *,
    t0: str = "2014-12-01",
    t1: str = "2025-01-01",
) -> tuple[Any, Any, Any]:
    """Regrid AIMIP monthly SST & SIC (`t0..t1`) onto `(lat_t, lon_t)`."""
    import numpy as np
    import pandas as pd
    import xarray as xr

    with xr.open_dataset(aimip_nc_path, engine="netcdf4") as raw_a:
        a = raw_a.sel(time=slice(t0, t1)).load()

    lat_s = a.latitude.values.astype(float)
    lon_s = a.longitude.values.astype(float)
    src_half = (
        abs(float(np.diff(lat_s).mean())) / 2.0 if lat_s.size > 1 else 0.125
    )
    w_lat, w_lon = _conservative_weights(
        lat_t, lon_t, lat_s, lon_s, src_half_deg=src_half
    )

    out: dict[str, Any] = {}
    for var_name in ("sea_surface_temperature", "sea_ice_cover"):
        frames = []
        for idx in range(int(a.sizes["time"])):
            f = a[var_name].isel(time=idx).values.astype(np.float64)
            mask = np.isfinite(f)
            num = w_lat @ np.where(mask, f, 0.0) @ w_lon.T
            den = w_lat @ mask.astype(float) @ w_lon.T
            regridded = np.where(den > 0.0, num / np.where(den > 0.0, den, 1.0), np.nan)
            frames.append(_fill_nearest(regridded))
        out[var_name] = np.stack(frames).astype(np.float32)

    return (
        pd.DatetimeIndex(a.time.values),
        out["sea_surface_temperature"],
        out["sea_ice_cover"],
    )


def _interp_linear_in_time(anchor: Any, fields: Any, times: Any) -> Any:
    """Linear time interpolation between month-start values."""
    import numpy as np

    t_sec = times.values.astype("datetime64[s]").astype(np.int64)
    a_sec = anchor.values.astype("datetime64[s]").astype(np.int64)
    idx = np.searchsorted(a_sec, t_sec, side="right") - 1
    if int(idx.min()) < 0 or int(idx.max()) >= len(a_sec) - 1:
        raise ConfigError(
            f"Target timestamps {times[0]}..{times[-1]} fall outside AIMIP monthly anchors "
            f"{anchor[0]}..{anchor[-1]}."
        )
    w = ((t_sec - a_sec[idx]) / (a_sec[idx + 1] - a_sec[idx])).astype(np.float32)[
        :, None, None
    ]
    return (1.0 - w) * fields[idx] + w * fields[idx + 1]


_OOS_FORMATS: dict[str, dict[str, Any]] = {
    "shieldplus": {
        "land": "land_fraction",
        "sst": "surface_temperature",
        "ice": "sea_ice_fraction",
        "ocn": "ocean_fraction",
        "co2": "global_mean_co2",
        "dims": ("grid_yt", "grid_xt"),
        "name": "forcing_{y}.nc",
        "noleap": False,
    },
    "hiro": {
        "land": "land_fraction",
        "sst": "surface_temperature",
        "ice": "sea_ice_fraction",
        "ocn": "ocean_fraction",
        "co2": "global_mean_co2",
        "dims": ("latitude", "longitude"),
        "name": "forcing_{y}.nc",
        "noleap": False,
    },
    "eamv3": {
        "land": "LANDFRAC",
        "sst": "TS",
        "ice": "ICEFRAC",
        "ocn": "OCNFRAC",
        "co2": None,
        "dims": ("lat", "lon"),
        "name": "{y}.nc",
        "noleap": True,
    },
}


def build_oos_forcing(
    fmt: str,
    *,
    hist_forcing_dir: str,
    aimip_monthly_nc_path: str,
    co2_nc_path: str | None,
    out_dir: str,
    years: Sequence[int] = OOS_YEARS_DEFAULT,
    bridge_year: int | None = 2014,
    reference_nonleap_year: int = 2014,
    reference_leap_year: int = 2012,
    anchor_t0: str = "2014-12-01",
    anchor_t1: str = "2025-01-01",
    expected_nonleap_steps: int | None = 1460,
    expected_leap_steps: int | None = 1464,
    force: bool = False,
    pinned_attrs: dict[str, str] | None = None,
) -> list[str]:
    """Build out-of-sample AIMIP forcing files for `shieldplus`, `hiro`, or `eamv3`."""
    import cftime
    import numpy as np
    import pandas as pd
    import xarray as xr
    from xarray.coding.times import CFDatetimeCoder

    if fmt not in _OOS_FORMATS:
        raise ValidationError(
            f"Unknown OOS forcing format {fmt!r}: expected one of {sorted(_OOS_FORMATS)}."
        )
    cfg = _OOS_FORMATS[fmt]
    noleap = bool(cfg["noleap"])
    name_pat = str(cfg["name"])
    ly, lx = cfg["dims"]

    validated_hist_dir = validate_safe_path(
        hist_forcing_dir, label=f"{fmt} historical forcing dir"
    )
    validated_aimip = validate_safe_path(
        aimip_monthly_nc_path, label="Zenodo 17065758 AIMIP monthly forcing"
    )
    validated_out_dir = validate_safe_path(out_dir, label=f"{fmt} amip-oos output dir")
    if not os.path.isfile(validated_aimip):
        raise ValidationError(f"AIMIP monthly forcing file not found: {validated_aimip}")

    co2_da: Any = None
    if cfg["co2"] is not None:
        if not co2_nc_path:
            raise ValidationError(f"OOS format {fmt!r} requires co2_nc_path.")
        validated_co2 = validate_safe_path(co2_nc_path, label="co2_global_mean.nc")
        if not os.path.isfile(validated_co2):
            raise ValidationError(f"CO2 table file not found: {validated_co2}")
        with xr.open_dataset(validated_co2, engine="netcdf4") as co2_ds:
            co2_da = co2_ds["global_mean_co2"].load()

    os.makedirs(validated_out_dir, exist_ok=True)
    written: list[str] = []

    if bridge_year is not None:
        src_bridge = os.path.join(validated_hist_dir, name_pat.format(y=bridge_year))
        dst_bridge = validate_safe_path(
            f"{validated_out_dir}/{name_pat.format(y=bridge_year)}",
            label=f"{fmt} bridge forcing {bridge_year}",
        )
        if not os.path.isfile(dst_bridge) or force:
            _link_or_copy_atomic(src_bridge, dst_bridge)
        written.append(dst_bridge)

    years_list = [int(y) for y in years]
    if not years_list:
        return written

    need_build = [
        y
        for y in years_list
        if force or not os.path.isfile(os.path.join(validated_out_dir, name_pat.format(y=y)))
    ]
    if not need_build:
        for y in years_list:
            written.append(
                validate_safe_path(
                    f"{validated_out_dir}/{name_pat.format(y=y)}",
                    label=f"{fmt} OOS {y}",
                )
            )
        return written

    ref0_path = os.path.join(
        validated_hist_dir, name_pat.format(y=reference_nonleap_year)
    )
    if not os.path.isfile(ref0_path):
        raise ValidationError(f"Reference forcing file not found: {ref0_path}")

    open_kw: dict[str, Any] = {"engine": "netcdf4"}
    if noleap:
        open_kw["decode_times"] = CFDatetimeCoder(use_cftime=True)
        open_kw["decode_timedelta"] = False

    with xr.open_dataset(ref0_path, **open_kw) as r0:
        r0.load()
        lat_t = r0[ly].values.astype(float)
        lon_t = r0[lx].values.astype(float)
        land = (
            r0[cfg["land"]].isel(time=0).values
            if "time" in r0[cfg["land"]].dims
            else r0[cfg["land"]].values
        )
        time_units = str(r0.time.encoding["units"])
        time_calendar = str(r0.time.encoding["calendar"])

    anchor, sst_grid, sic_grid = _monthly_aimip_on_grid(
        validated_aimip,
        lat_t,
        lon_t,
        t0=anchor_t0,
        t1=anchor_t1,
    )

    for y in years_list:
        out_file = validate_safe_path(
            f"{validated_out_dir}/{name_pat.format(y=y)}",
            label=f"{fmt} OOS {y}",
        )
        if os.path.isfile(out_file) and not force:
            written.append(out_file)
            continue

        ry = (
            reference_nonleap_year
            if (noleap or not py_calendar.isleap(y))
            else reference_leap_year
        )
        ref_y_path = os.path.join(validated_hist_dir, name_pat.format(y=ry))
        if not os.path.isfile(ref_y_path):
            raise ValidationError(f"Reference year forcing file not found: {ref_y_path}")

        with xr.open_dataset(ref_y_path, **open_kw) as ref_raw:
            ref = ref_raw.load()

        if noleap:
            tt: Any = [
                cftime.DatetimeNoLeap(y, t.month, t.day, t.hour)
                for t in ref.time.values
            ]
            real = pd.DatetimeIndex(
                [f"{y:04d}-{t.month:02d}-{t.day:02d}T{t.hour:02d}" for t in ref.time.values]
            )
        else:
            rt = pd.DatetimeIndex(ref.time.values)
            real = pd.DatetimeIndex([t.replace(year=y) for t in rt])
            tt = real.values

        # Verify full-year 6-hourly timestep count on the reference forcing dataset.
        want_steps = (
            expected_nonleap_steps
            if (noleap or not py_calendar.isleap(y))
            else expected_leap_steps
        )
        if want_steps is not None and len(real) != want_steps:
            raise ValidationError(
                f"Reference forcing {ref_y_path} for {fmt} year {y} has {len(real)} timesteps, "
                f"expected {want_steps}."
            )

        sst = _interp_linear_in_time(anchor, sst_grid, real)
        ice = _interp_linear_in_time(anchor, sic_grid, real)
        ice = np.clip(np.minimum(ice, 1.0 - land[None]), 0.0, 1.0)

        ds = ref.assign_coords(time=tt)
        dims = ("time", ly, lx)
        ds[cfg["sst"]] = (
            dims,
            np.where(land[None] < 1.0, sst, ref[cfg["sst"]].values).astype(np.float32),
        )
        ds[cfg["ice"]] = (dims, ice.astype(np.float32))
        ds[cfg["ocn"]] = (
            dims,
            np.clip(1.0 - land[None] - ice, 0.0, 1.0).astype(np.float32),
        )
        if co2_da is not None:
            v = co2_da.sel(time=real.values)
            if int(v.sizes["time"]) != len(real) or not bool(np.isfinite(v.values).all()):
                raise ConfigError(
                    f"CO2 lookup failed or non-finite for {fmt} year {y}."
                )
            ds[cfg["co2"]] = (("time",), v.values.astype(np.float32))

        for k in (cfg["sst"], cfg["ice"], cfg["ocn"]):
            ds[k].attrs = dict(ref[k].attrs)

        hist20_attr = (
            f"out-of-sample forcing {y}: AIMIP SST/SIC (Zenodo 17065758) on the model grid, "
            f"reference year {ry} for insolation/statics/land values"
            + (", CO2 from forcing-build/co2_global_mean.nc" if co2_da is not None else "")
        )
        ds.attrs = dict(ref.attrs, hist20=hist20_attr)
        if pinned_attrs:
            ds.attrs.update(pinned_attrs)

        _atomic_to_netcdf(
            ds,
            out_file,
            units=time_units,
            calendar=time_calendar,
            zlib=True,
            complevel=1,
        )
        written.append(out_file)

    return written


def extract_zip_member(zip_path: str, member: str, dst_path: str) -> str:
    """Copy one regular member of *zip_path* unchanged to *dst_path* (atomic; the zip's CRC-32 is checked on read).

    Used for the ArchesWeather checkpoints and configs in Zenodo 22795233 `model_checkpoints.zip`; the result is then
    verified against its SHA-256 in `aimip.lock` like every derived file."""
    import zipfile

    validated_zip = validate_safe_path(os.path.realpath(zip_path), label="zip archive")
    validated_dst = validate_safe_path(dst_path, label="extracted file")
    if not os.path.isfile(validated_zip):
        raise ValidationError(f"Builder input file not found: {validated_zip}")
    parent = os.path.dirname(validated_dst) or "/"
    os.makedirs(parent, exist_ok=True)
    tmp_path = os.path.join(parent, f".extract_{uuid.uuid4().hex}.part")
    try:
        with zipfile.ZipFile(validated_zip) as zf:
            try:
                info = zf.getinfo(member)
            except KeyError as exc:
                raise ValidationError(f"{member!r} is not in {validated_zip}") from exc
            if info.is_dir() or (info.external_attr >> 16) & 0o170000 not in (0, 0o100000):
                raise ValidationError(f"{member!r} in {validated_zip} is not a regular file")
            with zf.open(info) as src, open(tmp_path, "wb") as out:
                shutil.copyfileobj(src, out, 1 << 20)
        if os.path.getsize(tmp_path) != info.file_size:
            raise ValidationError(f"{member!r}: extracted {os.path.getsize(tmp_path)} bytes, expected {info.file_size}")
        os.replace(tmp_path, validated_dst)
    finally:
        if os.path.exists(tmp_path):
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
    return validated_dst
