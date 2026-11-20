"""CMIP6-style (CMOR-like) monthly files from a run: one variable per file, Amon attributes, DRS names.

Phase 1, deliberately narrow: only atmosphere variables that map one-to-one to a CMIP6 Amon variable (same quantity;
source units in an accepted spelling; geopotential -> height by standard gravity is the only value change), on the
grid the run wrote (grid label given by the user), and for `ta`/`zg` only the pressure levels the model itself writes.
Nothing is interpolated, regridded, filled or estimated. Months that average fewer model states than a full month
(the partial first or last month of a run) are left out. Every file states what was done (`aimip_postprocessing`)
and which CMIP6 requirements it does not meet (`aimip_missing`). The files are NOT validated by the CMOR library or
PrePARE.

Attribute values follow the CMIP6 data request tables CMIP6_Amon.json and CMIP6_coordinate.json
(PCMDI/cmip6-cmor-tables): long_name, standard_name, units, cell_methods, positive, height 2 m / 10 m, plev19.
"""
from __future__ import annotations

import datetime
import os
import re
from typing import Any

from .cf_export import (
    _CMIP, _LAT, _LON, _PA, GRAVITY, _coord_units, _write_pinned, _calendar_of, _check_raw_valid_time, _clean_model_slug, _detect_run_model,
    _detect_run_state, _is_missing_time_value, _norm_units, _validate_source_run_dir, monthly_files,
)
from .errors import AimipError, ExecutionError, ValidationError
from .validators import validate_run_path

# CMIP6 Amon: name -> (long_name, cell_methods, positive, height_m or None)
_AMON: dict[str, tuple[str, str, str, float | None]] = {
    "tas": ("Near-Surface Air Temperature", "area: time: mean", "", 2.0),
    "huss": ("Near-Surface Specific Humidity", "area: time: mean", "", 2.0),
    "uas": ("Eastward Near-Surface Wind", "area: time: mean", "", 10.0),
    "vas": ("Northward Near-Surface Wind", "area: time: mean", "", 10.0),
    "ts": ("Surface Temperature", "area: time: mean", "", None),
    "ps": ("Surface Air Pressure", "area: time: mean", "", None),
    "pr": ("Precipitation", "area: time: mean", "", None),
    "prw": ("Water Vapor Path", "area: time: mean", "", None),
    "hfls": ("Surface Upward Latent Heat Flux", "area: time: mean", "up", None),
    "hfss": ("Surface Upward Sensible Heat Flux", "area: time: mean", "up", None),
    "rlds": ("Surface Downwelling Longwave Radiation", "area: time: mean", "down", None),
    "rlus": ("Surface Upwelling Longwave Radiation", "area: time: mean", "up", None),
    "rsds": ("Surface Downwelling Shortwave Radiation", "area: time: mean", "down", None),
    "rsus": ("Surface Upwelling Shortwave Radiation", "area: time: mean", "up", None),
    "rlut": ("TOA Outgoing Longwave Radiation", "area: time: mean", "up", None),
    "rsut": ("TOA Outgoing Shortwave Radiation", "area: time: mean", "up", None),
    "rsdt": ("TOA Incident Shortwave Radiation", "area: time: mean", "down", None),
    "psl": ("Sea Level Pressure", "area: time: mean", "", None),
}
# fields on one pressure level -> (Amon 3-D name, long_name, level in Pa); all levels are in plev19
_ONE_LEVEL = {"ta850": ("ta", "Air Temperature", 85000.0), "zg1000": ("zg", "Geopotential Height", 100000.0),
              "zg500": ("zg", "Geopotential Height", 50000.0)}
_PLEV19 = 19
_LEVEL_LONG = {"ta": "Air Temperature", "zg": "Geopotential Height"}
# a variable that already carries the CMIP name: accepted when its units are a known spelling (no value change)
_SAME_NAME: dict[str, tuple[str, str, set[str]]] = {}
for _spec in _CMIP.values():
    if _spec[4] is None and _spec[0] in _AMON:
        _SAME_NAME.setdefault(_spec[0], (_spec[1], _spec[2], set()))[2].update(_spec[3])
for _name, (_std, _units, _acc) in list(_SAME_NAME.items()):
    _acc.add(_norm_units(_units))

_SAME_NAME.setdefault("psl", ("air_pressure_at_mean_sea_level", "Pa", set(_PA) | {"pa"}))
# bookkeeping variables, never climate fields
_NOT_FIELDS = {"counts", "valid_time", "init_time", "time_bnds", "lat_bnds", "lon_bnds"}

_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9.-]{0,63}$")
_GRID_LABELS = ("gn", "gr", "gr1", "gr2")
# CMIP6 required global attributes this command cannot fill truthfully (named in every file)
_NOT_WRITTEN = ("activity_id", "data_specs_version", "experiment", "further_info_url", "grid", "institution",
                "license", "nominal_resolution", "parent_experiment_id", "product", "source", "source_type",
                "sub_experiment", "sub_experiment_id", "tracking_id", "lat_bnds/lon_bnds")


_ATMOS_STEP_H = {  # coupled models: the registry step is the ocean coupling step; the atmosphere steps every 6 h
    "samudrace-cm4": 6,  # SamudrACE (ACE2 atmosphere, Samudra ocean coupled every 120 h); smoke run: 120 atmosphere
    "samudrace-e3smv3": 6,  # counts next to 6 ocean counts over the same 30 days
}


def _registry_step(name: str) -> int | None:
    """timestep_hours of *name* in the registry aimip uses. The full loader also expands the data paths and needs
    $AIMIP_DATA; the step does not, so it is read from the same registry file (or packaged default) when needed."""
    import json

    from .registry import PACKAGE_DEFAULT_REGISTRY_PATH, get_model, resolve_registry_path

    try:
        return int(get_model(name).timestep_hours)
    except Exception:  # noqa: BLE001 - fall back to reading the registry JSON directly
        pass
    for reg_fn in (resolve_registry_path, lambda: PACKAGE_DEFAULT_REGISTRY_PATH):
        try:
            with open(reg_fn(), encoding="utf-8") as fh:
                doc = json.load(fh)
            for entry in doc.get("models", []):
                if isinstance(entry, dict) and entry.get("name") == name:
                    return int(entry["timestep_hours"])
        except Exception:  # noqa: BLE001 - an unreadable registry states nothing
            continue
    return None


def _recorded_step(run: str) -> tuple[str | None, int | None]:
    """(model, timestep_hours) from the registry entry the run was made with (aimip-manifest.json / plan.json /
    emu-job.json), so a later registry change cannot reclassify an old run."""
    import json

    for fname in ("aimip-manifest.json", "plan.json", "emu-job.json"):
        fpath = os.path.join(run, fname)
        if not os.path.isfile(fpath) or os.path.islink(fpath):
            continue
        try:
            with open(fpath, encoding="utf-8") as fh:
                doc = json.load(fh)
        except (OSError, ValueError, TypeError):
            continue
        reg = doc.get("registry_entry") if isinstance(doc, dict) else None
        if isinstance(reg, dict) and reg.get("name"):
            try:
                return _clean_model_slug(reg.get("name")), int(reg["timestep_hours"])
            except (KeyError, TypeError, ValueError):
                return _clean_model_slug(reg.get("name")), None
    return None, None


def _recorded_identities(run: str, ds: Any) -> set[str]:
    """Every model name the run records: registry_entry.name and the top-level model of aimip-manifest.json, plan.json
    and emu-job.json, and the monthly file's model attribute."""
    import json

    ids: set[str] = set()
    for fname in ("aimip-manifest.json", "plan.json", "emu-job.json"):
        fpath = os.path.join(run, fname)
        if not os.path.isfile(fpath) or os.path.islink(fpath):
            continue
        try:
            with open(fpath, encoding="utf-8") as fh:
                doc = json.load(fh)
        except (OSError, ValueError, TypeError):
            continue
        if isinstance(doc, dict):
            reg = doc.get("registry_entry")
            for cand in ((reg.get("name") if isinstance(reg, dict) else None), doc.get("model")):
                slug = _clean_model_slug(cand)
                if slug:
                    ids.add(slug)
    slug = _clean_model_slug(ds.attrs.get("model"))
    if slug:
        ids.add(slug)
    return ids


def _model_step(ds: Any, run: str, model: str | None, comp: str = "") -> tuple[str | None, int | None, str]:
    """(model, time step in hours, where it comes from). Read, never guessed, in this order: the registry entry recorded
    in the run when it was made; the current registry entry of the model (named by --model, the run record or the
    file); the file's own monthly statement. A --model that contradicts the run record is refused."""
    rep = str(ds.attrs.get("monthly_representation", ""))
    rec_name, rec_step = _recorded_step(run)
    recorded = _recorded_identities(run, ds)
    if model and recorded - {model}:
        raise ValidationError(f"--model {model} contradicts the run record ({', '.join(sorted(recorded))}).",
                              field="model")
    if len(recorded) > 1:
        raise ValidationError(f"The run records disagree about the model ({', '.join(sorted(recorded))}); refusing to "
                              f"choose a time step.", field="model")
    name = model or rec_name or (next(iter(recorded)) if recorded else None)
    if name:
        if comp == "atmosphere" and name in _ATMOS_STEP_H:
            return name, _ATMOS_STEP_H[name], f"atmosphere step of {name} (the registry step is the ocean coupling)"
        if name == "cbottle-3d" and str(ds.attrs.get("cadence", "")).strip() == "6h":
            return name, 6, "file attribute cadence: 6h"
        if rec_step is not None and name == rec_name:
            return name, rec_step, f"registry entry of {name} recorded in the run"
        hours = _registry_step(name)
        if hours is not None:
            return name, hours, f"current registry entry of {name} (the run record has no time step)"
    if "single monthly model state" in rep:
        return name, 730, f"file attribute monthly_representation: {rep}"
    return name, None, ""


def _month_table(times: list[Any], counts: Any, cal: str, step_h: int | None, source: str,
                 keep_first_month: bool = False) -> tuple[list[int], list[str]]:
    """Indices of the full months to keep and a note for each decision. A month is full when its count equals the
    number of model steps in that calendar month at the model's stated step; nothing is inferred from the counts."""
    import cftime
    import numpy as np

    ym = [(int(t.year), int(t.month)) for t in times]
    for (y0, m0), (y1, m1) in zip(ym, ym[1:]):
        if (y1, m1) != (y0 + (m0 == 12), 1 if m0 == 12 else m0 + 1):
            raise ValidationError(f"valid_time is not a run of consecutive months ({y0:04d}-{m0:02d} is followed "
                                  f"by {y1:04d}-{m1:02d}); refusing to write a CMIP time axis.", field="valid_time")
    if counts is None:
        return [], ["no counts variable: full months cannot be confirmed: nothing written"]
    if step_h is None:
        return [], ["model time step unknown (no run record naming the model; give --model): full months cannot be "
                    "confirmed: nothing written"]
    raw = np.asarray(counts, dtype="float64").ravel()
    if raw.size != len(times) or not (np.isfinite(raw).all() and (raw >= 0).all() and (raw == np.round(raw)).all()):
        return [], [f"counts are not one whole, non-negative number per month ({raw.tolist()[:6]}): nothing written"]
    cnt = [int(c) for c in raw]
    if step_h >= 28 * 24:  # one model state per calendar month
        if all(c == 1 for c in cnt):
            return list(range(len(times))), [f"monthly-step model ({source}): one model state per month"]
        return [], [f"monthly-step model ({source}) but counts {sorted(set(cnt))} are not all 1: nothing written"]
    if 24 % step_h:
        return [], [f"model step {step_h} h ({source}) does not divide a day: full months cannot be confirmed: "
                    f"nothing written"]
    rate = 24 // step_h

    def days(y: int, mo: int) -> int:
        a = cftime.datetime(y, mo, 1, calendar=cal)
        b = cftime.datetime(y + (mo == 12), 1 if mo == 12 else mo + 1, 1, calendar=cal)
        return int(round((b - a).total_seconds() / 86400))

    keep = []
    notes = [f"full month = {rate} model states per day (time step {step_h} h, {source})"]
    for i, (c, (y, mo)) in enumerate(zip(cnt, ym)):
        full = rate * days(y, mo)
        if c > full:
            return [], [f"{y:04d}-{mo:02d} has {c} model states, more than {full} at a {step_h} h step ({source}): "
                        f"the counts do not match the model step: nothing written"]
        if i == 0 and keep_first_month and c == full - 1 and len(cnt) > 1 and cnt[1] > 0:
            notes.append(f"KEPT {y:04d}-{mo:02d} with {c} of {full} model states (--keep-first-month); which state is "
                         f"missing is not verified (for a run started at 00:00 on the 1st it is expected to be the "
                         f"initial state, which is not model output)")
            keep.append(i)
        elif c < full:
            notes.append(f"LEFT OUT {y:04d}-{mo:02d}: partial month ({c} of {full} model states)")
        else:
            keep.append(i)
    if keep and keep != list(range(keep[0], keep[-1] + 1)):
        raise ValidationError("a partial month inside the run; refusing to write a CMIP time axis with a gap.",
                              field="counts")
    return keep, notes


def _bounds_and_mid(times: list[Any], cal: str) -> tuple[list[tuple[Any, Any]], list[Any], list[str]]:
    import cftime

    bounds, mids, labels = [], [], []
    for t in times:
        y, mo = int(t.year), int(t.month)
        a = cftime.datetime(y, mo, 1, calendar=cal)
        b = cftime.datetime(y + (mo == 12), 1 if mo == 12 else mo + 1, 1, calendar=cal)
        bounds.append((a, b))
        mids.append(a + (b - a) / 2)
        labels.append(f"{y:04d}{mo:02d}")
    return bounds, mids, labels


def _write(dsx: Any, out: str, parts: list[str], fname: str, var: str) -> str:
    """Write atomically. CMIP encoding: _FillValue = missing_value = 1e20 on the variable, none on coordinates and
    bounds, and `coordinates` (the scalar height) only on the variable: cdo refuses it on time_bnds."""
    import netCDF4

    enc = {v: {"_FillValue": None} for v in dsx.variables if v != var}
    enc[var] = {"_FillValue": 1.0e20, "missing_value": 1.0e20, "dtype": "float32"}
    for v in ("time", "time_bnds"):
        enc[v].update({k: dsx[v].encoding[k] for k in ("units", "calendar", "dtype") if k in dsx[v].encoding})

    def _to(tmp: str) -> None:
        dsx.to_netcdf(tmp, unlimited_dims=["time"], encoding=enc)
        with netCDF4.Dataset(tmp, "a") as nc:
            for name, v in nc.variables.items():
                if name != var and "coordinates" in v.ncattrs():
                    v.delncattr("coordinates")

    try:
        return _write_pinned(_to, out, parts, fname)
    except AimipError:
        raise
    except (OSError, ValueError, RuntimeError) as exc:
        raise ExecutionError(f"Could not write {os.path.join(out, *parts, fname)}: {exc}") from exc


def export_cmor(run_dir: str, *, source_id: str, experiment_id: str, grid_label: str, institution_id: str,
                out_dir: str, members: list[int] | None = None, keep_first_month: bool = False,
                model: str | None = None, allow_unfinished: bool = False) -> dict[str, Any]:
    """Write CMIP6-style Amon files (one variable each) for the monthly means of *run_dir*."""
    import numpy as np
    import xarray as xr
    from xarray.coding.times import CFDatetimeCoder

    for label, value in (("source_id", source_id), ("experiment_id", experiment_id),
                         ("institution_id", institution_id)):
        if not _ID.match(str(value or "")):
            raise ValidationError(f"{label} {value!r}: use letters, digits, '.' and '-' only.", field=label)
    if grid_label not in _GRID_LABELS:
        raise ValidationError(f"grid_label {grid_label!r}: one of {', '.join(_GRID_LABELS)} (gn = the model's own "
                              f"grid, gr = regridded, e.g. HEALPix written on lat/lon).", field="grid_label")
    if members is not None and not members:
        raise ValidationError("--members is empty.", field="members")
    run = _validate_source_run_dir(run_dir)
    state = _detect_run_state(run, allow_unfinished=allow_unfinished, command="export-cmor")
    files = [(c, p) for c, p in monthly_files(run) if c in ("monthly", "atmosphere", "ocean")]
    if not files:
        raise ValidationError(f"No monthly_mean_predictions.nc in {run}.", field="run")
    comps = {c for c, _ in files}
    if {"monthly", "atmosphere"} <= comps:
        raise ValidationError(f"{run} has both monthly_mean_predictions.nc and atmosphere/monthly_mean_predictions.nc; "
                              f"they would write the same CMIP6 files. Export a copy of the run with only one of them.",
                              field="run")
    planned: set[str] = set()
    if os.path.islink(out_dir.rstrip("/")):
        raise ValidationError(f"Refusing a symlinked output directory: {out_dir}", field="out")
    out = validate_run_path(out_dir, label="--to")
    if os.path.realpath(out) == os.path.realpath(run):
        raise ValidationError("--to is the run directory itself; give a subdirectory (default <run>/cmor).", field="out")
    try:
        os.makedirs(out, exist_ok=True)
    except OSError as exc:
        raise ValidationError(f"Could not create {out}: {exc}", field="out") from exc
    version = "v" + datetime.date.today().strftime("%Y%m%d")
    written: list[dict[str, Any]] = []
    skipped: dict[str, str] = {}
    state_note = (
        f"run status is {state} (not completed); only full months are written"
        if state is not None and state != "completed"
        else ""
    )
    run_notes: list[str] = [state_note] if state_note else []
    for comp, path in files:
        if comp == "ocean":
            skipped[f"{comp}/monthly_mean_predictions.nc"] = "ocean component: phase 1 writes the Amon table only"
            continue
        try:
            src = xr.open_dataset(path, decode_times=CFDatetimeCoder(use_cftime=True), decode_timedelta=False,
                                  cache=False)
        except (OSError, ValueError, RuntimeError) as exc:
            raise ExecutionError(f"Could not read {path}: {exc}") from exc
        with src as ds:
            if "valid_time" not in ds.variables:
                raise ValidationError(f"{path}: no valid_time, so no dates; refusing to write CMIP time axes.",
                                      field="valid_time")
            n = int(ds.sizes.get("sample", 1))
            sel = members if members is not None else list(range(n))
            for k in sel:
                if not 0 <= int(k) < n:
                    raise ValidationError(f"Member {k} outside 0..{n - 1}.", field="members")
            _check_raw_valid_time(path, sel)
            for k in sel:
                m = ds.isel(sample=int(k)) if "sample" in ds.dims else ds
                vt = m["valid_time"]
                if vt.ndim != 1 or vt.size == 0:
                    raise ValidationError(f"{path}: valid_time is not a non-empty list of months for member {k}.",
                                          field="valid_time")
                cal = _calendar_of(vt)
                times = list(vt.values)
                if any(_is_missing_time_value(x) for x in times):
                    raise ValidationError(
                        f"{path}: valid_time for member {k} contains missing values (incomplete monthly write).",
                        field="valid_time",
                    )
                counts = m["counts"].values if "counts" in m.variables else None
                if counts is not None and m["counts"].dims != vt.dims:
                    raise ValidationError(f"{path}: counts has dimensions {m['counts'].dims}, valid_time {vt.dims}; "
                                          f"counts must be one value per month.", field="counts")
                mdl, step_h, step_src = _model_step(ds, run, model, comp)
                keep, month_notes = _month_table(times, counts, cal, step_h, step_src, keep_first_month)
                for note in month_notes:
                    if note not in run_notes:
                        run_notes.append(note)
                if not keep:
                    skipped[f"member {k}"] = "no full month in the run"
                    continue
                bounds, mids, labels = _bounds_and_mid([times[i] for i in keep], cal)
                y0 = int(times[keep[0]].year)
                epoch = "days since 1850-01-01 00:00:00" if y0 >= 1850 else f"days since {y0:04d}-01-01 00:00:00"
                # descriptive attributes of the source time axes (e.g. Spherical DYffusion's nominal-cycle long_name)
                src_time: dict[str, str] = {}
                for tv in ("time", "valid_time"):
                    if tv in m.variables:
                        for a in ("long_name", "comment", "description", "note"):
                            val = str(m[tv].attrs.get(a, "")).strip()
                            if val and val.lower() not in ("time", "valid time", "valid_time", "month"):
                                src_time.setdefault(a, val)
                time_note = "; ".join(f"source time {a}: {v}" for a, v in src_time.items())
                variant = f"r{int(k) + 1}i1p1f1"
                # collect the fields first, so that two sources of one CMIP variable are caught before any write
                fields: dict[str, list[tuple[float | None, str, Any, str]]] = {}
                for name in list(m.data_vars):
                    spec = _CMIP.get(str(name))
                    src_units = m[name].attrs.get("units")
                    if spec is None and str(name) in _SAME_NAME:
                        std, units, accepted = _SAME_NAME[str(name)]
                        spec = (str(name), std, units, accepted, None)
                    if spec is None:
                        if str(name) not in _NOT_FIELDS and "time" in m[name].dims:
                            skipped[str(name)] = "no CMIP6 Amon equivalent in the mapping table"
                        continue
                    cmip, std, units, accepted, factor = spec
                    if _norm_units(src_units) not in accepted:
                        skipped[str(name)] = f"units {src_units!r} are not a known spelling of {units}"
                        continue
                    if cmip in _AMON:
                        var, plev = cmip, None
                    elif cmip in _ONE_LEVEL:
                        var, _, plev = _ONE_LEVEL[cmip]
                    else:
                        skipped[str(name)] = f"{cmip} is not a CMIP6 Amon variable"
                        continue
                    if any(p == plev for p, *_ in fields.get(var, [])):
                        raise ValidationError(f"Two variables of {path} map to {var}"
                                              f"{'' if plev is None else f' at {plev:g} Pa'}: "
                                              f"{fields[var][0][1]} and {name}; refusing to choose.", field="run")
                    fields.setdefault(var, []).append((plev, str(name), spec, str(src_units)))
                for var, items in fields.items():
                    items.sort(key=lambda it: -(it[0] or 0.0))  # plev descending, as in plev19
                    notes = ["time from valid_time; time is the middle of each calendar month, time_bnds its bounds",
                             *([time_note] if time_note else []),
                             f"grid as written by the run (grid_label {grid_label}), no regridding"]
                    arrays, plevs, origin = [], [], []
                    point_cm = ""
                    std = units = ""
                    for plev, name, spec, src_units in items:
                        _, std, units, _, factor = spec
                        da = m[name].transpose("time", ...).isel(time=keep)
                        if factor is not None:
                            da = da * factor
                            notes.append(f"CONVERTED {name} geopotential [{src_units}] to geopotential height [m]: "
                                         f"divided by standard gravity g = {GRAVITY} m s-2")
                        notes.append(f"{var}{'' if plev is None else f' {plev:g} Pa'} from model variable {name} "
                                     f"[{src_units}]")
                        src_cm = str(m[name].attrs.get("cell_methods", ""))
                        if "time: point" in src_cm:
                            point_cm = src_cm
                        vals = np.asarray(da.values)
                        if vals.dtype != np.float32:
                            v64 = vals.astype("float64")
                            finite = v64[np.isfinite(v64)]
                            if finite.size and np.abs(finite).max() > float(np.finfo("float32").max):
                                raise ValidationError(f"{path}: {name} has values beyond the float32 range; refusing to "
                                                      f"write a CMIP6 float32 field.", field=name)
                            v32 = v64.astype("float32")
                            diff = np.abs(v32.astype("float64") - v64)[np.isfinite(v64)]
                            if diff.size and diff.max() > 0:
                                rel = (diff / np.maximum(np.abs(finite), np.finfo("float64").tiny)).max()
                                notes.append(f"CAST {name} from {vals.dtype} to float32 as CMIP6 Amon requires: largest "
                                             f"change {diff.max():.3g} {units} (relative {rel:.2g})")
                            vals = v32
                        arrays.append(vals)
                        plevs.append(plev)
                        origin.append(name)
                    dims = [d for d in m[items[0][1]].dims if d not in ("time", "sample")]
                    long_name, cell_methods, positive, height = (
                        _AMON[var] if var in _AMON else (_LEVEL_LONG[var], "time: mean", "", None))
                    if plevs[0] is None:
                        data, out_dims = arrays[0], ("time", *dims)
                    else:
                        data, out_dims = np.stack(arrays, axis=1), ("time", "plev", *dims)
                        notes.append(f"{len(plevs)} of the {_PLEV19} plev19 levels: only the levels the model writes; "
                                     f"nothing interpolated")
                    out_da = xr.DataArray(data, dims=out_dims,
                                          coords={d: m[d] for d in dims if d in m.coords})
                    if point_cm:  # the model writes one state per month, not a time mean: keep its own statement
                        cell_methods = point_cm
                        notes.append(f"cell_methods kept from the model ({point_cm}): a single monthly state, not a "
                                     f"time mean; CMIP6 Amon asks for 'area: time: mean'")
                    out_da.attrs = {"standard_name": std, "long_name": long_name, "units": units,
                                    "cell_methods": cell_methods, "original_name": ",".join(origin)}
                    if positive:
                        out_da.attrs["positive"] = positive
                    dsx = xr.Dataset({var: out_da})
                    dsx = dsx.assign_coords(time=("time", mids))
                    dsx["time"].attrs.update(standard_name="time", axis="T", bounds="time_bnds", long_name="time")
                    dsx["time"].encoding = {"units": epoch, "calendar": cal, "dtype": "float64"}
                    dsx["time_bnds"] = (("time", "bnds"), np.array(bounds, dtype=object))
                    dsx["time_bnds"].encoding = {"units": epoch, "calendar": cal, "dtype": "float64"}
                    if height is not None:
                        dsx = dsx.assign_coords(height=height)
                        dsx["height"].attrs.update(standard_name="height", long_name="height", units="m", axis="Z",
                                                   positive="up")
                    if plevs[0] is not None:
                        dsx = dsx.assign_coords(plev=("plev", np.array(plevs, dtype="float64")))
                        dsx["plev"].attrs.update(standard_name="air_pressure", long_name="pressure", units="Pa",
                                                 axis="Z", positive="down")
                    ren = {c: ("lat" if str(c) in _LAT else "lon") for c in list(dsx.dims)
                           if str(c) in (_LAT | _LON) and str(c) not in ("lat", "lon")}
                    if ren:
                        dsx = dsx.rename(ren)
                        notes.append("horizontal coordinates renamed " + ", ".join(f"{a} -> {b}" for a, b in ren.items())
                                     + " (same values)")
                    for c, (std_c, units_c, axis_c, lo, hi) in (("lat", ("latitude", "degrees_north", "Y", -90, 90)),
                                                              ("lon", ("longitude", "degrees_east", "X", -180, 360))):
                        if c in dsx.coords:
                            vals = np.asarray(dsx[c].values, dtype="float64")
                            src_u = dsx[c].attrs.get("units")
                            kind = _coord_units(src_u, c == "lat")
                            if kind == "bad" or not (vals.size and np.isfinite(vals).all() and vals.min() >= lo
                                                     and vals.max() <= hi):
                                raise ValidationError(f"{path}: {c} (units {src_u!r}) is not plain degrees "
                                                      f"{units_c}; refusing to label it.", field=c)
                            if kind != "ok":
                                notes.append(f"{c}: units {src_u!r} written as {units_c} (values {vals.min():g}.."
                                             f"{vals.max():g})")
                            dsx[c].attrs.update(standard_name=std_c, long_name=std_c, units=units_c, axis=axis_c)
                    dsx.attrs = {
                        "Conventions": "CF-1.7", "mip_era": "CMIP6", "table_id": "Amon", "frequency": "mon",
                        "realm": "atmos", "variable_id": var, **({"aimip_model": mdl} if mdl else {}), "source_id": source_id, "experiment_id": experiment_id,
                        "institution_id": institution_id, "variant_label": variant, "realization_index": int(k) + 1,
                        "initialization_index": 1, "physics_index": 1, "forcing_index": 1, "grid_label": grid_label,
                        "creation_date": datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
                        "aimip_postprocessing": "aimip export-cmor: " + "; ".join(
                            notes + month_notes + ([state_note] if state_note else [])
                        ),
                        "aimip_missing": "not validated by the CMOR library or PrePARE; CMIP6 items not written: "
                                         + ", ".join(_NOT_WRITTEN),
                    }
                    fname = (f"{var}_Amon_{source_id}_{experiment_id}_{variant}_{grid_label}_"
                             f"{labels[0]}-{labels[-1]}.nc")
                    parts = [institution_id, source_id, experiment_id, variant, "Amon", var, grid_label, version]
                    key = "/".join([*parts, fname])
                    if key in planned:
                        raise ValidationError(f"Two sources would write {key}; refusing to let one replace the other.",
                                              field="run")
                    planned.add(key)
                    target = os.path.join(out, *parts, fname)
                    _write(dsx, out, parts, fname, var)
                    written.append({"file": target, "variable": var, "from": ",".join(origin), "member": int(k),
                                    "notes": notes + month_notes, "component": comp or "single"})
    if not written and not skipped:
        skipped["(run)"] = ("no variable maps one-to-one to a CMIP6 Amon variable with known units; "
                            "`aimip export-cf` writes all variables")
    return {"run_dir": run, "out_dir": out, "files": written, "skipped": skipped, "notes": run_notes}
