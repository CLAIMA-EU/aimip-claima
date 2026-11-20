"""CF-style copies of a run's monthly means: time first, real dates, one file per ensemble member, CMIP names.

The model writers keep each model's own layout, `(sample, time, lat, lon)` with the dates in `valid_time`, and their
own variable names (ACE: `TMP2m`, cBottle-3d: `t2m`, ...). Tools such as cdo need time as the first dimension and a time
axis with units. `export_cf` writes, next to the run's own files (which stay untouched) or under `--to <dir>`,
`<run>_<model>_<component>_m<k>.nc` (or `<run>_<component>_m<k>.nc` when no model is recorded) for every member k:
every field with `time` first, `time` taken from `valid_time`, latitude/longitude with CF attributes,
`Conventions = "CF-1.8"`. A variable gets its CMIP short name (`tas`, `pr`, `rlut`, ...) only when the quantity is the
same and its units are one of the listed spellings (or a listed conversion applies); every renamed variable keeps
`original_name` and `original_units`. All other variables keep the model's name. `names="native"` keeps every model name.
"""
from __future__ import annotations

import json
import os
import re
from typing import Any

from .errors import AimipError, ValidationError
from .status import MONTHLY_PREDICTIONS_FILENAME
from .validators import validate_run_path, validate_safe_path

COMPONENTS = ("", "atmosphere", "ocean")
MONTHLY_SOURCES: tuple[tuple[str, str], ...] = (
    ("monthly", MONTHLY_PREDICTIONS_FILENAME),
    ("levels", "monthly_levels.nc"),
    ("atmosphere", os.path.join("atmosphere", MONTHLY_PREDICTIONS_FILENAME)),
    ("atmosphere_levels", os.path.join("atmosphere", "monthly_levels.nc")),
    ("atmosphere_precip", os.path.join("atmosphere", "monthly_precip.nc")),
    ("ocean", os.path.join("ocean", MONTHLY_PREDICTIONS_FILENAME)),
    ("ocean_thetao", os.path.join("ocean", "monthly_thetao.nc")),
)
CF_DIRNAME = "cf"
GRAVITY = 9.80665  # m s-2, WMO standard gravity (geopotential -> geopotential height)

_W_M2 = {"w/m**2", "w/m2", "wm-2", "wm**-2", "w/m^2", "watts/m2"}
_K = {"k", "kelvin", "deg_k"}
_PA = {"pa"}
_MS = {"m/s", "ms-1", "ms**-1"}
_KGKG = {"kg/kg", "kgkg-1", "kgkg**-1", "1"}
_KGM2S = {"kg/m**2/s", "kgm-2s-1", "kg/m2/s", "kgm**-2s**-1", "kg/(m**2s)"}
_KGM2 = {"kg/m**2", "kgm-2", "kg/m2", "kgm**-2"}
_M = {"m", "gpm"}
_GEOPOT = {"m**2s**-2", "m2s-2", "m**2/s**2", "m^2/s^2", "m2/s2"}

# model name -> (CMIP name, CF standard_name, CMIP units, accepted source units, factor or None)
_CMIP: dict[str, tuple[str, str, str, set[str], float | None]] = {
    # ACE (Ai2 fme)
    "TMP2m": ("tas", "air_temperature", "K", _K, None),
    "Q2m": ("huss", "specific_humidity", "1", _KGKG, None),
    "UGRD10m": ("uas", "eastward_wind", "m s-1", _MS, None),
    "VGRD10m": ("vas", "northward_wind", "m s-1", _MS, None),
    "PRESsfc": ("ps", "surface_air_pressure", "Pa", _PA, None),
    "surface_temperature": ("ts", "surface_temperature", "K", _K, None),
    "PRATEsfc": ("pr", "precipitation_flux", "kg m-2 s-1", _KGM2S, None),
    "LHTFLsfc": ("hfls", "surface_upward_latent_heat_flux", "W m-2", _W_M2, None),
    "SHTFLsfc": ("hfss", "surface_upward_sensible_heat_flux", "W m-2", _W_M2, None),
    "DLWRFsfc": ("rlds", "surface_downwelling_longwave_flux_in_air", "W m-2", _W_M2, None),
    "ULWRFsfc": ("rlus", "surface_upwelling_longwave_flux_in_air", "W m-2", _W_M2, None),
    "DSWRFsfc": ("rsds", "surface_downwelling_shortwave_flux_in_air", "W m-2", _W_M2, None),
    "USWRFsfc": ("rsus", "surface_upwelling_shortwave_flux_in_air", "W m-2", _W_M2, None),
    "ULWRFtoa": ("rlut", "toa_outgoing_longwave_flux", "W m-2", _W_M2, None),
    "USWRFtoa": ("rsut", "toa_outgoing_shortwave_flux", "W m-2", _W_M2, None),
    "DSWRFtoa": ("rsdt", "toa_incoming_shortwave_flux", "W m-2", _W_M2, None),
    "TMP850": ("ta850", "air_temperature", "K", _K, None),
    "h500": ("zg500", "geopotential_height", "m", _M, None),
    # ERA5-style names used by cBottle-3d and MD-1.5 (DLESyM files carry no units, so these stay unrenamed there)
    "t2m": ("tas", "air_temperature", "K", _K, None),
    "tcwv": ("prw", "atmosphere_mass_content_of_water_vapor", "kg m-2", _KGM2, None),
    "z500": ("zg500", "geopotential_height", "m", _GEOPOT, 1.0 / GRAVITY),
    "z1000": ("zg1000", "geopotential_height", "m", _GEOPOT, 1.0 / GRAVITY),
    # cBottle-3d names with the units its writer sets (cbottle.py): exact equivalents
    "u10m": ("uas", "eastward_wind", "m s-1", _MS, None),
    "v10m": ("vas", "northward_wind", "m s-1", _MS, None),
    "msl": ("psl", "air_pressure_at_mean_sea_level", "Pa", _PA, None),
    "tpf": ("pr", "precipitation_flux", "kg m-2 s-1", _KGM2S, None),
    "t850": ("ta850", "air_temperature", "K", _K, None),
}

_LAT = {"lat", "latitude", "grid_yt"}
_LON = {"lon", "longitude", "grid_xt"}
_MODEL_SLUG_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$")


def _clean_model_slug(raw: Any) -> str | None:
    if not isinstance(raw, str):
        return None
    s = raw.strip()
    if not s or s.lower() == "unknown" or ".." in s or not _MODEL_SLUG_RE.match(s):
        return None
    return s


def _detect_run_model(run_dir: str) -> str | None:
    """Read the recorded model slug from `aimip-manifest.json`, `plan.json`, or `emu-job.json` if present."""
    for fname in ("aimip-manifest.json", "plan.json", "emu-job.json"):
        fpath = os.path.join(run_dir, fname)
        if not os.path.isfile(fpath) or os.path.islink(fpath):
            continue
        try:
            with open(fpath, encoding="utf-8") as fh:
                doc = json.load(fh)
        except (OSError, ValueError, TypeError):
            continue
        if not isinstance(doc, dict):
            continue
        reg = doc.get("registry_entry")
        cand = (reg.get("name") if isinstance(reg, dict) else None) or doc.get("model")
        slug = _clean_model_slug(cand)
        if slug:
            return slug
    return None


def _detect_run_state(
    run_dir: str,
    *,
    allow_unfinished: bool = False,
    command: str = "export-cf",
) -> str | None:
    """Inspect `<run_dir>/aimip-manifest.json` (if present); refuse active runs and return state string."""
    mpath = os.path.join(run_dir, "aimip-manifest.json")
    if not os.path.isfile(mpath) or os.path.islink(mpath):
        return None
    try:
        with open(mpath, encoding="utf-8") as fh:
            manifest = json.load(fh)
    except (OSError, ValueError, TypeError):
        return "unreadable manifest"
    if not isinstance(manifest, dict):
        return "invalid manifest"
    if "finished" in manifest and "rc" in manifest:
        try:
            return "completed" if int(manifest["rc"]) == 0 else f"failed (rc={manifest['rc']})"
        except (ValueError, TypeError):
            return "incomplete"

    sacct_path = os.path.join(run_dir, "sacct.txt")
    if os.path.isfile(sacct_path) and not os.path.islink(sacct_path):
        try:
            with open(sacct_path, encoding="utf-8") as fh:
                first_line = fh.readline().strip()
        except OSError:
            first_line = ""
        parts = first_line.split("|")
        if len(parts) >= 2 and parts[1].strip():
            st = parts[1].strip().split()[0].upper()
            if st in ("RUNNING", "PENDING", "COMPLETING", "CONFIGURING", "STAGE_OUT"):
                raise ValidationError(
                    f"Run {run_dir} is currently {st.lower()} (aimip status {run_dir} refreshes the state); "
                    f"wait for it to finish before running aimip {command}.",
                    field="run",
                )
            if st == "COMPLETED":
                return "completed"
            return st.lower()

    has_stale_pid = False
    pid_path = os.path.join(run_dir, "local.pid")
    if os.path.isfile(pid_path) and not os.path.islink(pid_path):
        try:
            pid = int(open(pid_path, encoding="utf-8").readline().strip())
        except (OSError, ValueError):
            pid = -1
        if pid > 0:
            try:
                os.kill(pid, 0)
                alive = True
            except PermissionError:
                alive = True
            except OSError:
                alive = False
            if alive:
                raise ValidationError(
                    f"Run {run_dir} is still running (pid {pid}); wait for it to finish before running aimip {command}.",
                    field="run",
                )
        has_stale_pid = True

    if (
        has_stale_pid
        or manifest.get("started_utc")
        or os.path.isfile(os.path.join(run_dir, "slurm.jobid"))
        or os.path.isfile(os.path.join(run_dir, "submitted.txt"))
    ):
        if allow_unfinished:
            return "unfinished"
        raise ValidationError(
            f"Run {run_dir} has not recorded completion yet (aimip status {run_dir} refreshes the state; "
            f"pass --allow-unfinished to run aimip {command} anyway).",
            field="run",
        )
    return "not completed"


def _format_time_endpoint(val: Any, dtype: Any) -> str | None:
    import numpy as np

    if hasattr(val, "year") and hasattr(val, "month") and hasattr(val, "day"):
        return f"{int(val.year):04d}-{int(val.month):02d}-{int(val.day):02d}"
    if np.issubdtype(dtype, np.datetime64) and not bool(np.isnat(val)):
        return str(np.datetime_as_string(val, unit="D"))
    return None


_LAT_UNITS = {"degrees_north", "degree_north", "degrees_n", "degree_n", "degreesn", "degreen"}
_LON_UNITS = {"degrees_east", "degree_east", "degrees_e", "degree_e", "degreese", "degreee"}
_GENERIC_DEGREES = {"degrees", "degree", "deg"}


def _coord_units(src: Any, is_lat: bool) -> str:
    """'ok' (north/east degrees), 'missing', 'generic' (degrees without a direction) or 'bad' (anything else, e.g.
    radians or degrees_west: never relabelled)."""
    if src is None or str(src).strip() == "":
        return "missing"
    u = re.sub(r"\s+", "_", str(src).strip().lower())
    if u in (_LAT_UNITS if is_lat else _LON_UNITS):
        return "ok"
    return "generic" if u in _GENERIC_DEGREES else "bad"


def _entries_protected(dir_fd: int) -> bool:
    """True when only the caller can rename or remove entries of this directory: it is the caller's, and either not
    writable by group or others or sticky (a directory owned by someone else is never trusted: its owner can rename)."""
    st = os.fstat(dir_fd)
    return st.st_uid == os.getuid() and (not st.st_mode & 0o022 or bool(st.st_mode & 0o1000))


def _write_pinned(write: Any, root: str, parts: list[str], name: str) -> str:
    """Write `root/parts.../name` so that nothing can redirect it: every directory level is created and opened with
    O_NOFOLLOW and held by descriptor from *root* down, the temporary file is created exclusively in the held final
    directory, written through /proc/self/fd/<dir>/<tmp> and renamed relative to the same descriptor. A symlink or
    non-directory anywhere below *root*, or a symlinked or non-regular target, is refused. Only the temporary file this
    call created is ever removed. Returns the target path (for reporting)."""
    import secrets
    import stat

    dir_flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0)
    fds: list[int] = []
    try:
        try:
            fds.append(os.open(root, dir_flags))
        except OSError as exc:
            raise ValidationError(f"Output directory {root} is a symlink or not a directory ({exc.strerror})",
                                  field="out") from exc
        for p in parts:
            try:
                os.mkdir(p, dir_fd=fds[-1])
            except FileExistsError:
                pass
            except OSError as exc:
                raise ValidationError(f"Could not create {p} in {root}: {exc.strerror}", field="out") from exc
            try:
                fds.append(os.open(p, dir_flags, dir_fd=fds[-1]))
            except OSError as exc:
                raise ValidationError(f"Refusing {os.path.join(root, *parts)}: {p!r} is a symlink or not a directory "
                                      f"({exc.strerror})", field="out") from exc
        d = fds[-1]
        target = os.path.join(root, *parts, name)
        try:
            st = os.stat(name, dir_fd=d, follow_symlinks=False)
            if not stat.S_ISREG(st.st_mode):
                raise ValidationError(f"Refusing to replace {target}: it is a symlink or not a regular file", field="out")
        except FileNotFoundError:
            pass
        # a private staging directory (0700, ours): nobody else can replace the file while the NetCDF library writes it
        # by name; it is reached through /proc/self/fd of the held descriptor, never through the original path
        stage = f".{name}.{secrets.token_hex(8)}.incomplete"
        os.mkdir(stage, 0o700, dir_fd=d)
        s: int | None = None
        ours = reserved = False
        try:
            s = os.open(stage, dir_flags, dir_fd=d)
            fds.append(s)
            sst = os.fstat(s)
            if sst.st_uid != os.getuid() or sst.st_mode & 0o077:
                raise ValidationError(f"Staging directory in {os.path.join(root, *parts)} is not private", field="out")
            ours = True
            os.close(os.open(name, os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0), 0o666,
                             dir_fd=s))
            reserved = True
            via = f"/proc/self/fd/{s}"
            if not os.path.isdir(via):
                raise ValidationError("Writing exports needs Linux /proc/self/fd (not available here)", field="out")
            write(f"{via}/{name}")
            if not stat.S_ISREG(os.stat(name, dir_fd=s, follow_symlinks=False).st_mode):
                raise ValidationError(f"Refusing to publish {target}: not a regular file", field="out")
            os.replace(name, name, src_dir_fd=s, dst_dir_fd=d)
        finally:  # clean up only what this call created: the reserved file and its own private staging directory
            if reserved and s is not None:
                try:
                    os.unlink(name, dir_fd=s)
                except OSError:
                    pass
            if ours and _entries_protected(d):  # elsewhere another user could have swapped the entry: leave it
                try:
                    os.rmdir(stage, dir_fd=d)
                except OSError:
                    pass
        return target
    finally:
        for fd in reversed(fds):
            os.close(fd)


def _norm_units(u: Any) -> str:
    return re.sub(r"\s+", "", str(u or "")).lower()


def monthly_files(run_dir: str) -> list[tuple[str, str]]:
    """`(component, path)` of the run's monthly-mean files (including levels/precip/thetao), regular files only."""
    found = []
    for comp, rel in MONTHLY_SOURCES:
        path = os.path.join(run_dir, rel)
        parent = os.path.dirname(path)
        if parent != run_dir and os.path.islink(parent):
            continue
        if os.path.isfile(path) and not os.path.islink(path):
            found.append((comp, path))
    return found


def _cmip_rename(ds: Any, notes: list[str]) -> tuple[Any, dict[str, str]]:
    """Rename variables with an exact CMIP equivalent; every change is appended to *notes* in plain words.

    Values are never changed except by a listed physical conversion (geopotential -> geopotential height, divided by
    standard gravity); a variable whose units are missing or not in the accepted list keeps its model name and values.
    """
    renamed: dict[str, str] = {}
    taken = set(ds.variables)
    for name in list(ds.data_vars):
        spec = _CMIP.get(str(name))
        if spec is None:
            continue
        cmip, std, units, accepted, factor = spec
        src_units = ds[name].attrs.get("units")
        if cmip in taken:
            notes.append(f"{name}: kept its model name (a variable called {cmip} already exists)")
            continue
        if _norm_units(src_units) not in accepted:
            if src_units is None or not str(src_units).strip():
                notes.append(f"{name}: kept its model name and values (no units in the file: not renamed to {cmip})")
            else:
                notes.append(f"{name}: kept its model name and values (units {src_units!r} are not a known spelling of "
                             f"{units}; not renamed to {cmip})")
            continue
        da = ds[name]
        attrs = dict(da.attrs)
        if factor is not None:
            da = da * factor
            notes.append(f"CONVERTED {name} ({src_units}, geopotential) -> {cmip} ({units}, geopotential height): "
                         f"values divided by standard gravity g = {GRAVITY} m s-2")
            attrs["aimip_conversion"] = f"{name} [{src_units}] / {GRAVITY} m s-2"
            if "long_name" in attrs:
                attrs["original_long_name"] = str(attrs["long_name"])
            attrs["long_name"] = f"geopotential height at {cmip[2:]} hPa"
        else:
            unit_note = "" if str(src_units) == units else f"; units string {src_units!r} written as {units!r}, same values"
            notes.append(f"renamed {name} -> {cmip} (same values{unit_note})")
        attrs.update(standard_name=std, units=units, original_name=str(name), original_units=str(src_units))
        da.attrs = attrs
        enc = dict(ds[name].encoding)
        if factor is not None:  # new values in new units: no integer or packed storage, no old-unit ranges
            for key in ("dtype", "scale_factor", "add_offset", "_FillValue", "missing_value", "_Unsigned"):
                enc.pop(key, None)
            dropped = [a for a in ("valid_range", "valid_min", "valid_max", "actual_range") if a in da.attrs]
            for a in dropped:
                da.attrs.pop(a)
            if dropped:
                notes.append(f"{cmip}: {', '.join(dropped)} removed (they were in {src_units})")
            da = da.astype("float64") if da.dtype.kind != "f" else da
        da.encoding = enc
        ds = ds.drop_vars(name).assign({cmip: da})
        taken.add(cmip)
        renamed[str(name)] = cmip
    return ds, renamed


def _calendar_of(vt: Any) -> str:
    """The calendar the dates carry; never guessed: cftime objects name theirs, numpy datetimes are proleptic."""
    import numpy as np

    cal = vt.encoding.get("calendar") or vt.attrs.get("calendar")
    if cal:
        return str(cal)
    first = vt.values.flat[0] if vt.size else None
    if getattr(first, "calendar", None):
        return str(first.calendar)
    if np.issubdtype(vt.dtype, np.datetime64):
        return "proleptic_gregorian"
    raise ValidationError("valid_time carries no calendar; cannot write a CF time axis without guessing.",
                          field="valid_time")


def _is_missing_time_value(val: Any) -> bool:
    import numpy as np

    if val is None or val is np.ma.masked:
        return True
    if isinstance(val, (float, np.floating)):
        return bool(np.isnan(val))
    if isinstance(val, np.datetime64):
        return bool(np.isnat(val))
    yr = getattr(val, "year", None)
    if yr is not None:
        try:
            if not (-10000 <= int(yr) <= 99999):
                return True
        except (ValueError, TypeError):
            return True
    return str(val) in ("NaT", "nan", "None", "--", "")


def _check_raw_valid_time(path: str, sel: Any = None) -> None:
    """Check raw numeric on-disk `valid_time` before cftime decoding (independent of xarray/cftime NaN handling)."""
    import numpy as np
    import xarray as xr

    try:
        with xr.open_dataset(path, decode_times=False, mask_and_scale=True, cache=False) as raw_ds:
            if "valid_time" not in raw_ds.variables:
                return
            rvt = raw_ds["valid_time"]
            n = int(raw_ds.sizes.get("sample", 1))
            indices = sel if sel is not None else range(n)
            for k in indices:
                if not 0 <= int(k) < n:
                    continue
                sub = rvt.isel(sample=int(k)) if "sample" in rvt.dims else rvt
                if sub.dims != ("time",):
                    continue
                arr = np.asarray(sub.values)
                if arr.size and arr.dtype.kind in ("f", "c") and not bool(np.isfinite(arr).all()):
                    raise ValidationError(
                        f"valid_time for member {k} contains missing values (incomplete monthly write).",
                        field="valid_time",
                    )
    except AimipError:
        raise
    except (OSError, ValueError, TypeError, KeyError, RuntimeError) as exc:
        raise ValidationError(f"Cannot read {path}: {exc}", field="run") from exc


def _cf_member(
    ds: Any,
    k: int,
    *,
    names: str,
    source_file: str,
    run_warning: str | None = None,
) -> tuple[Any, dict[str, Any]]:
    """One member of *ds* as a CF dataset (time first, `time` from `valid_time`); all changes listed in notes."""
    import numpy as np

    m = ds.isel(sample=k) if "sample" in ds.dims else ds
    notes: list[str] = []
    info: dict[str, Any] = {"time_from": "time" if "time" in m.variables else "none", "notes": notes}
    if "valid_time" in m.variables and m["valid_time"].dims == ("time",):
        vt = m["valid_time"]
        vt_vals = vt.values
        if vt_vals.size:
            has_missing = (
                bool(np.isnat(vt_vals).any()) if np.issubdtype(vt_vals.dtype, np.datetime64)
                else bool(np.isnan(vt_vals).any()) if vt_vals.dtype.kind == "f"
                else any(_is_missing_time_value(x) for x in vt_vals.flat)
            )
            if has_missing:
                raise ValidationError(
                    f"valid_time for member {k} contains missing values (incomplete monthly write).",
                    field="valid_time",
                )
        cal = _calendar_of(vt)
        raw_vt_units = vt.encoding.get("units") or vt.attrs.get("units")
        vt_units = str(raw_vt_units) if raw_vt_units and "since" in str(raw_vt_units).lower() else "days since 1970-01-01 00:00:00"
        drop_time_keys = {"units", "calendar", "_FillValue", "missing_value", "coordinates"}
        kept_time_attrs: dict[str, Any] = {}
        if "time" in m.variables:
            kept_time_attrs.update({ak: av for ak, av in m["time"].attrs.items() if ak not in drop_time_keys})
        kept_time_attrs.update({ak: av for ak, av in vt.attrs.items() if ak not in drop_time_keys})
        m = m.drop_vars("valid_time").assign_coords(time=("time", vt_vals))
        m["time"].attrs = kept_time_attrs
        m["time"].attrs.update(standard_name="time", axis="T")
        m["time"].attrs.setdefault("long_name", "time")
        m["time"].encoding = {"units": vt_units, "calendar": cal}
        info["time_from"] = "valid_time"
        notes.append(f"time axis taken from valid_time (calendar {cal}); the file's time was a step counter")
    else:
        if "valid_time" in m.variables:
            notes.append(f"valid_time has dimensions {m['valid_time'].dims}, expected ('time',): time kept as written")
        elif "time" not in m.variables:
            notes.append("no time or valid_time variable in this file: no time axis written")
        elif not (np.issubdtype(m["time"].dtype, np.datetime64)
                  or (m["time"].size and hasattr(m["time"].values.flat[0], "calendar"))):
            notes.append("time has no dates in this file (no valid_time): kept as written, cdo will not see dates")
        if "time" in m.variables:
            m["time"].attrs.update(standard_name="time", axis="T")
            m["time"].attrs.setdefault("long_name", "time")
    # Scalar or auxiliary coordinates (ACE: init_time) and the writers' `coordinates` attributes, which still name
    # valid_time, make cdo warn ("Coordinates variable ... can't be assigned"): keep them as plain variables.
    aux = [c for c in m.coords if c not in m.dims]
    if aux:
        m = m.reset_coords(aux)
    for v in m.variables:
        m[v].attrs.pop("coordinates", None)
        m[v].encoding.pop("coordinates", None)
    for v in list(m.data_vars):
        dims = m[v].dims
        if "time" in dims and dims[0] != "time":
            m[v] = m[v].transpose("time", ...)
    for c in list(m.coords):
        is_lat, is_lon = str(c) in _LAT, str(c) in _LON
        if not (is_lat or is_lon):
            continue
        vals = np.asarray(m[c].values, dtype="float64")
        lo, hi = (-90.0, 90.0) if is_lat else (-180.0, 360.0)
        units = "degrees_north" if is_lat else "degrees_east"
        src = m[c].attrs.get("units")
        kind = _coord_units(src, is_lat)
        if vals.size and np.nanmin(vals) >= lo and np.nanmax(vals) <= hi and kind != "bad":
            m[c].attrs.update(standard_name="latitude" if is_lat else "longitude", units=units,
                              axis="Y" if is_lat else "X")
            if kind != "ok":
                notes.append(f"{c}: units {src!r} written as {units} (values {np.nanmin(vals):g}..{np.nanmax(vals):g})")
        else:
            notes.append(f"{c}: left as written (units {src!r}, values {np.nanmin(vals):g}..{np.nanmax(vals):g} "
                         f"are not plain degrees)")
    renamed: dict[str, str] = {}
    if names == "cmip":
        m, renamed = _cmip_rename(m, notes)
    n_months = int(m.sizes.get("time", 0)) if "time" in m.dims else 0
    first_date: str | None = None
    last_date: str | None = None
    if "time" in m.variables and m["time"].size > 0:
        t_vals = m["time"].values
        first_date = _format_time_endpoint(t_vals.flat[0], t_vals.dtype)
        last_date = _format_time_endpoint(t_vals.flat[-1], t_vals.dtype)
    info.update(renamed=renamed, months=n_months, first_date=first_date, last_date=last_date)
    post_notes = ([run_warning] if run_warning else []) + list(notes)
    m.attrs = dict(ds.attrs)
    m.attrs.update(
        Conventions="CF-1.8",
        aimip_member=int(k),
        aimip_names=names,
        aimip_source_file=source_file,
        aimip_postprocessing="aimip export-cf: " + "; ".join(post_notes) if post_notes else "aimip export-cf: none",
    )
    return m, info


def _validate_source_run_dir(run_dir: str) -> str:
    """Validate a source run directory for read-only inspection (may reside outside writable run roots)."""
    safe = validate_safe_path(run_dir, label="run directory")
    if os.path.islink(safe):
        raise ValidationError(f"Refusing a symlinked run directory: {safe}", field="run")
    real = os.path.realpath(safe)
    validate_safe_path(real, label="run directory")
    if not os.path.isdir(safe):
        raise ValidationError(f"Run directory does not exist: {safe}", field="run")
    return safe


def export_cf(run_dir: str, *, out_dir: str | None = None, members: list[int] | None = None,
              names: str = "cmip", allow_unfinished: bool = False) -> dict[str, Any]:
    """Write CF copies of the monthly means of *run_dir*; returns what was written."""
    import xarray as xr
    from xarray.coding.times import CFDatetimeCoder

    if names not in ("cmip", "native"):
        raise ValidationError(f"names must be 'cmip' or 'native', not {names!r}.", field="names")
    run = _validate_source_run_dir(run_dir)
    run_state = _detect_run_state(run, allow_unfinished=allow_unfinished, command="export-cf")
    run_warnings: list[str] = []
    run_warn_note: str | None = None
    if run_state is not None and run_state != "completed":
        run_warn_note = f"run status is {run_state} (not completed); exported monthly files may be incomplete"
        run_warnings.append(run_warn_note)
    if out_dir is not None:
        safe_to = validate_safe_path(out_dir, label="--to")
        if os.path.islink(safe_to):
            raise ValidationError(f"Refusing a symlinked output directory: {safe_to}", field="out")
        out = validate_run_path(safe_to, label="--to")
    else:
        default_out = os.path.join(run, CF_DIRNAME)
        if os.path.islink(default_out):
            raise ValidationError(f"Refusing a symlinked output directory: {default_out}", field="out")
        try:
            out = validate_run_path(default_out, label="output directory")
        except ValidationError as exc:
            raise ValidationError(
                f"{exc} Pass --to <dir> under your run root to export a run stored outside it.",
                field="out",
            ) from exc
    files = monthly_files(run)
    if not files:
        raise ValidationError(f"No {MONTHLY_PREDICTIONS_FILENAME} in {run} (the run has not written monthly means).",
                              field="run")
    req_members: list[int] | None = None
    if members is not None:
        req_members = list(dict.fromkeys(int(k) for k in members))
        if not req_members:
            raise ValidationError("--members must specify at least one member index.", field="members")
        for _, path in files:
            try:
                with xr.open_dataset(path, decode_times=False, cache=False) as ds_chk:
                    n_chk = int(ds_chk.sizes.get("sample", 1))
            except AimipError:
                raise
            except (OSError, ValueError, KeyError, RuntimeError) as exc:
                raise ValidationError(f"Cannot read {path}: {exc}", field="run") from exc
            for k in req_members:
                if not 0 <= k < n_chk:
                    raise ValidationError(f"Member {k} outside 0..{n_chk - 1} in {path}.", field="members")
    if os.path.exists(out) and not os.path.isdir(out):
        raise ValidationError(f"Output path exists and is not a directory: {out}", field="out")
    try:
        os.makedirs(out, exist_ok=True)
    except OSError as exc:
        raise ValidationError(f"Cannot create output directory {out}: {exc}", field="out") from exc
    try:
        import dask
        chunks: dict[str, int] | None = {"time": 12}
    except ImportError:
        dask = None  # type: ignore[assignment]
        chunks = None
    run_slug = os.path.basename(run.rstrip("/"))
    run_model = _detect_run_model(run)
    written: list[dict[str, Any]] = []
    for comp, path in files:
        comp_name = comp or "monthly"
        rel_src = os.path.relpath(path, run).replace(os.sep, "/")
        _check_raw_valid_time(path, req_members)
        try:
            ds_ctx = xr.open_dataset(path, decode_times=CFDatetimeCoder(use_cftime=True), decode_timedelta=False,
                                     cache=False, chunks=chunks)
        except AimipError:
            raise
        except (OSError, ValueError, TypeError, KeyError, RuntimeError) as exc:
            raise ValidationError(f"Cannot read {path}: {exc}", field="run") from exc
        with ds_ctx as ds:
            eff_model = run_model or _clean_model_slug(ds.attrs.get("model"))
            if run_model is None and eff_model is not None:
                run_model = eff_model
            stem = f"{run_slug}_{eff_model}_{comp_name}" if eff_model else f"{run_slug}_{comp_name}"
            n = int(ds.sizes.get("sample", 1))
            for k in (req_members if req_members is not None else range(n)):
                if not 0 <= int(k) < n:
                    raise ValidationError(f"Member {k} outside 0..{n - 1} in {path}.", field="members")
                target = os.path.join(out, f"{stem}_m{int(k):03d}.nc")
                if os.path.islink(target):
                    raise ValidationError(f"Refusing symlinked output file: {target}", field="out")
                if os.path.realpath(target) == os.path.realpath(path):
                    raise ValidationError(f"Refusing to overwrite source file {path}.", field="out")
                try:
                    m, info = _cf_member(ds, int(k), names=names, source_file=rel_src, run_warning=run_warn_note)

                    def _to(tmp: str, m: Any = m) -> None:
                        if dask is not None:
                            with dask.config.set(scheduler="synchronous"):
                                m.to_netcdf(tmp)
                        else:
                            m.to_netcdf(tmp)

                    _write_pinned(_to, out, [], os.path.basename(target))
                except AimipError:
                    raise
                except (OSError, ValueError, TypeError, KeyError, RuntimeError) as exc:
                    raise ValidationError(f"Failed writing {target} from {path}: {exc}", field="out") from exc
                written.append({"file": target, "component": comp_name, "member": int(k), **info})
    return {
        "run_dir": run,
        "out_dir": out,
        "model": run_model,
        "run_state": run_state,
        "warnings": run_warnings,
        "names": names,
        "files": written,
    }
