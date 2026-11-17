"""Initial-condition inspection, multi-start spin preparation, and restart merging.

Dimension names are NEVER hardcoded: ACE-family IC datasets use different horizontal
dimension names (`latitude`/`longitude` on `ace2-era5`, `grid_yt`/`grid_xt` on
`ace2-shield` & `ace2s-shield-plus`, `lat`/`lon` on `ace2-eamv3` & `restart.nc`),
mirroring `fme.core.dataset.utils.get_horizontal_coordinates` (`dims[-2:]` at
`fme/core/dataset/utils.py:219-229`).

fme 2026.5.1 references:
- `InitialConditionConfig.get_dataset` & `_subselect_initial_conditions`:
  `fme/ace/inference/inference.py:51-93`
- `get_initial_condition` (accepts duplicate timestamps along `"sample"` dim;
  only checks 3D shape `(n_samples, n_lat, n_lon)` and `len(ds.time) == n_samples`):
  `fme/ace/inference/inference.py:95-150`
- `get_horizontal_coordinates` (detects horizontal dims via `dims[-2:]`):
  `fme/core/dataset/utils.py:211-229`
- `get_forcing_data` (resolves each sample's initial timestamp independently via
  `ExplicitIndices` into the monotonic forcing `CFTimeIndex`):
  `fme/ace/data_loading/getters.py:258-283`
- `restart.nc` layout (`dims = ["sample", "lat", "lon"]`, `time` squeezed to `("sample",)`):
  `fme/core/generics/inference.py:165-166`,
  `fme/ace/inference/data_writer/main.py:216-230, 261-311`.
"""

from __future__ import annotations

import dataclasses
import datetime
import json
import os
import uuid
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

from . import __version__
from .errors import ConfigError, ValidationError
from .registry import ModelSpec
from .time_utils import compute_spin_steps, format_cftime, parse_cftime, select_forcing_set
from .validators import (
    makedirs_no_group_other_write,
    strip_group_other_write,
    validate_iso_timestamp,
    validate_run_path,
    validate_safe_path,
    validate_seed,
    validate_visible_input_path,
)

if TYPE_CHECKING:
    from .runner import SlurmOptions


@dataclasses.dataclass(frozen=True)
class ICFileInfo:
    """Metadata decoded from an fme-compatible initial-condition netCDF file."""

    path: str
    sample_dim: str
    n_samples: int
    calendar: str
    time_units: str
    timestamps: tuple[str, ...]
    variable_names: tuple[str, ...]
    spatial_dims: tuple[str, str] | None
    grid_shape: tuple[int, int] | None
    has_sample_coord: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "path": self.path,
            "sample_dim": self.sample_dim,
            "n_samples": self.n_samples,
            "calendar": self.calendar,
            "time_units": self.time_units,
            "timestamps": list(self.timestamps),
            "variable_names": list(self.variable_names),
            "spatial_dims": (
                list(self.spatial_dims) if self.spatial_dims is not None else None
            ),
            "grid_shape": list(self.grid_shape) if self.grid_shape is not None else None,
            "has_sample_coord": self.has_sample_coord,
        }


def inspect_ic_file(path: str) -> ICFileInfo:
    """Decode `time` (via cftime) and 3D variable shapes/dims of an IC netCDF file."""
    import xarray as xr
    from xarray.coding.times import CFDatetimeCoder

    validated = validate_safe_path(path, label="IC file path")
    if not os.path.isfile(validated):
        raise ValidationError(f"Initial condition file not found: {validated}")

    try:
        with xr.open_dataset(
            validated,
            engine="netcdf4",
            decode_times=CFDatetimeCoder(use_cftime=True),
            decode_timedelta=False,
        ) as ds:
            if "time" not in ds:
                raise ValidationError(
                    f"Initial condition file {validated} has no 'time' variable."
                )
            time_da = ds["time"]
            if time_da.ndim != 1 or time_da.shape[0] < 1:
                raise ValidationError(
                    f"Initial condition file {validated}: 'time' must be 1D with length >= 1, "
                    f"got shape {time_da.shape}."
                )

            sample_dim = str(time_da.dims[0])
            has_sample_coord = sample_dim in ds.coords
            raw_times = time_da.values
            n_samples = int(raw_times.shape[0])

            first_dt = raw_times[0]
            if not hasattr(first_dt, "year"):
                # Older xarray (<2023.12) treats `decode_times=CFDatetimeCoder(...)` as a boolean
                # and returns `np.datetime64`; reopen with legacy `use_cftime=True` only in that case.
                with xr.open_dataset(validated, engine="netcdf4", use_cftime=True) as ds_cf:
                    raw_times = ds_cf["time"].values
                    first_dt = raw_times[0]
            calendar = (
                getattr(first_dt, "calendar", None)
                or time_da.encoding.get("calendar")
                or time_da.attrs.get("calendar")
                or "proleptic_gregorian"
            )
            time_units = str(
                time_da.encoding.get("units")
                or time_da.attrs.get("units")
                or "hours since 1970-01-01 00:00:00"
            )
            timestamps = tuple(format_cftime(dt) for dt in raw_times)

            var_names: list[str] = []
            spatial_dims: tuple[str, str] | None = None
            grid_shape: tuple[int, int] | None = None
            for name, da in ds.data_vars.items():
                if name == "time":
                    continue
                if da.ndim == 3 and da.shape[0] == n_samples:
                    var_names.append(str(name))
                    if spatial_dims is None:
                        spatial_dims = (str(da.dims[1]), str(da.dims[2]))
                        grid_shape = (int(da.shape[1]), int(da.shape[2]))
                elif da.ndim == 4 and "varlev" in da.dims and sample_dim in da.dims[:2]:
                    if "varlev" in ds.coords:
                        var_names.extend(str(v) for v in ds["varlev"].values)
                    else:
                        var_names.append(str(name))
                    if spatial_dims is None:
                        spatial_dims = (str(da.dims[-2]), str(da.dims[-1]))
                        grid_shape = (int(da.shape[-2]), int(da.shape[-1]))

            if spatial_dims is None:
                # Fallback only when a dataset has 2D horizontal coords without 3D vars.
                other_dims = [str(d) for d in ds.dims if str(d) != sample_dim]
                if len(other_dims) == 2:
                    spatial_dims = (other_dims[0], other_dims[1])
                    grid_shape = (
                        int(ds.sizes[other_dims[0]]),
                        int(ds.sizes[other_dims[1]]),
                    )
    except (ValidationError, ConfigError):
        raise
    except Exception as exc:
        raise ValidationError(
            f"Failed to read initial condition netCDF {validated}: {exc}"
        ) from exc

    return ICFileInfo(
        path=validated,
        sample_dim=sample_dim,
        n_samples=n_samples,
        calendar=str(calendar),
        time_units=time_units,
        timestamps=timestamps,
        variable_names=tuple(var_names),
        spatial_dims=spatial_dims,
        grid_shape=grid_shape,
        has_sample_coord=has_sample_coord,
    )


def list_model_ics(model: ModelSpec) -> list[ICFileInfo]:
    """Decode and return `ICFileInfo` for every shipped/staged atmosphere IC file of *model*."""
    paths = list(model.ic_files)
    if not paths and model.family == "md":
        from .registry import resolve_data_dir

        data_root = resolve_data_dir()
        if data_root:
            md_ic_dir = os.path.join(data_root, "inputs", model.name, "ic")
            if os.path.isdir(md_ic_dir) and not os.path.islink(md_ic_dir):
                for fname in sorted(os.listdir(md_ic_dir)):
                    if fname.endswith(".nc"):
                        cand = os.path.join(md_ic_dir, fname)
                        if os.path.isfile(cand) and not os.path.islink(cand):
                            paths.append(cand)
    return [inspect_ic_file(ic_path) for ic_path in paths]


def list_model_ocean_ics(model: ModelSpec) -> list[ICFileInfo]:
    """Decode and return `ICFileInfo` for every shipped ocean IC file of coupled *model*."""
    return [inspect_ic_file(ic_path) for ic_path in model.ocean_ic_files]


def find_shipped_ic_for_time(
    model: ModelSpec,
    target_iso: str,
    *,
    candidate_ic_files: Sequence[str] | None = None,
) -> ICFileInfo:
    """Locate the shipped IC file of *model* (or *candidate_ic_files* subset) containing *target_iso*."""
    norm_target = validate_iso_timestamp(target_iso, label="start timestamp")
    paths = tuple(candidate_ic_files) if candidate_ic_files else model.ic_files
    infos = [inspect_ic_file(p) for p in paths]
    available: list[str] = []
    for info in infos:
        if norm_target in info.timestamps:
            return info
        available.extend(info.timestamps)

    shown = ", ".join(available[:12])
    if len(available) > 12:
        shown += f", ... ({len(available)} total)"
    raise ConfigError(
        f"Timestamp {norm_target} was not found in shipped IC file(s) for model "
        f"{model.name!r} (available: {shown or 'none'}). "
        f"Run `aimip ic list -m {model.name}` to inspect shipped ICs or "
        f"`aimip ic spin -m {model.name} --from <T_shipped> --to {norm_target} --out <spin-dir>` "
        "to spin an initial condition to that date."
    )


def find_shipped_coupled_ic_for_time(
    model: ModelSpec,
    target_iso: str,
    *,
    candidate_ic_files: Sequence[str] | None = None,
    candidate_ocean_ic_files: Sequence[str] | None = None,
) -> tuple[ICFileInfo, ICFileInfo, int]:
    """Locate paired `(atmosphere_ic, ocean_ic, first_sample_index)` for coupled *model* at *target_iso*."""
    if not model.is_coupled:
        raise ConfigError(f"Model {model.name!r} is not a coupled model.")
    norm_target = validate_iso_timestamp(target_iso, label="start timestamp")
    atmos_paths = (
        tuple(candidate_ic_files)
        if (candidate_ic_files and candidate_ocean_ic_files)
        else model.ic_files
    )
    ocean_paths = (
        tuple(candidate_ocean_ic_files)
        if (candidate_ic_files and candidate_ocean_ic_files)
        else model.ocean_ic_files
    )
    if len(atmos_paths) != len(ocean_paths):
        raise ConfigError(
            f"Coupled model {model.name!r} has mismatched atmosphere ({len(atmos_paths)}) "
            f"and ocean ({len(ocean_paths)}) IC file lists."
        )

    available: list[str] = []
    for atmos_path, ocean_path in zip(atmos_paths, ocean_paths):
        atmos_info = inspect_ic_file(atmos_path)
        ocean_info = inspect_ic_file(ocean_path)
        if atmos_info.n_samples != ocean_info.n_samples:
            raise ConfigError(
                f"Coupled IC sample-count mismatch for {model.name!r}: "
                f"atmosphere {atmos_info.path} has {atmos_info.n_samples} sample(s) vs "
                f"ocean {ocean_info.path} with {ocean_info.n_samples}."
            )
        if atmos_info.timestamps != ocean_info.timestamps:
            raise ConfigError(
                f"Coupled IC timestamp mismatch for {model.name!r}: "
                f"atmosphere {atmos_info.path} timestamps {atmos_info.timestamps} != "
                f"ocean {ocean_info.path} timestamps {ocean_info.timestamps}."
            )
        if atmos_info.calendar != ocean_info.calendar:
            raise ConfigError(
                f"Coupled IC calendar mismatch for {model.name!r}: "
                f"atmosphere ({atmos_info.calendar}) != ocean ({ocean_info.calendar})."
            )
        if norm_target in ocean_info.timestamps:
            first_idx = ocean_info.timestamps.index(norm_target)
            return atmos_info, ocean_info, first_idx
        available.extend(ocean_info.timestamps)

    shown = ", ".join(available[:12])
    if len(available) > 12:
        shown += f", ... ({len(available)} total)"
    raise ConfigError(
        f"Timestamp {norm_target} was not found in shipped coupled IC file(s) for model "
        f"{model.name!r} (available: {shown or 'none'}). "
        f"Run `aimip ic list -m {model.name}` to inspect shipped ICs or "
        f"`aimip ic spin -m {model.name} --from <T_shipped> --to {norm_target} --out <spin-dir>` "
        "to spin an initial condition to that date."
    )


def _ensure_outside_playground(
    target_path: str, playground_dir: str | None, label: str
) -> None:
    """Ensure output files are never written inside `$AIMIP_PLAYGROUND`."""
    if not playground_dir:
        return
    pg = playground_dir.rstrip("/")
    if target_path == pg or target_path.startswith(pg + "/"):
        raise ConfigError(
            f"Refusing to write {label} ({target_path}) inside $AIMIP_PLAYGROUND ({pg}); "
            "choose an output path under your configured run root outside the playground."
        )


def prepare_spin(
    *,
    model: ModelSpec,
    from_times: Sequence[str],
    to_time: str,
    spin_dir: str | None = None,
    mode: str = "dry-run",
    seed: int = 0,
    lora_path: str | None = None,
    forcing_set_name: str | None = None,
    slurm_opts: SlurmOptions | None = None,
    site_config: Any = None,
    playground_dir: str | None = None,
    data_dir: str | None = None,
    auto_fetch: bool = True,
    argv: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Prepare `<spin_dir>/0..K-1` fme runs to spin shipped ICs `T1,T2,...` up to `T`.

    Writes both:
    - `<spin_dir>/job.sbatch` (`# aimip-generated v1` calling `"<pg>/bin/aimip" exec-fme "<spin_dir>/<i>"`
      for `0..K-1`, submittable as a single batch job via `aimip submit <spin_dir>`), AND
    - per-member `<spin_dir>/<i>/job.sbatch` (submittable individually in parallel via
      `aimip submit <spin_dir>/<i>`).
    After completion, `aimip collect <spin_dir> --out <ic.nc>` merges all `restart.nc`
    files (or coupled `atmosphere/restart.nc` + `ocean/restart.nc`) into K-member IC netCDF(s).
    """
    from .runner import (
        SlurmOptions,
        default_out_dir,
        ensure_model_files_ready,
        estimate_slurm_walltime,
        exec_fme,
        model_has_missing_files,
        prepare_run,
        render_sbatch_script,
    )
    from .site_config import load_site_config

    if mode not in ("dry-run", "submit", "local"):
        raise ValidationError(f"Invalid spin execution mode {mode!r}.")

    norm_to = validate_iso_timestamp(to_time, label="--to timestamp")
    validated_seed = validate_seed(seed)

    cleaned_from = [t.strip() for t in from_times if t and t.strip()]
    if not cleaned_from:
        raise ValidationError("Spin --from requires at least one shipped IC timestamp.")

    if lora_path is not None:
        validate_visible_input_path(lora_path, label="--lora checkpoint path")

    normalized_from = [
        validate_iso_timestamp(raw_ti, label="--from timestamp") for raw_ti in cleaned_from
    ]
    if not model.stochastic:
        seen_from: set[str] = set()
        duplicate_from: list[str] = []
        for norm_ti in normalized_from:
            if norm_ti in seen_from and norm_ti not in duplicate_from:
                duplicate_from.append(norm_ti)
            seen_from.add(norm_ti)
        if duplicate_from:
            dups_shown = ", ".join(duplicate_from)
            raise ConfigError(
                f"Model {model.name!r} is deterministic (`stochastic: false`), so duplicate --from "
                f"timestamp(s) ({dups_shown}) would produce bit-identical spin trajectories. "
                "Provide distinct shipped IC start times in --from."
            )

    if spin_dir is not None:
        safe_spin_pre = validate_safe_path(spin_dir, label="spin --out directory")
        _ensure_outside_playground(safe_spin_pre, playground_dir, "spin directory")
        resolved_spin_dir = validate_run_path(spin_dir, label="spin --out directory")
    else:
        resolved_spin_dir = default_out_dir(
            f"{model.name}-spin",
            norm_to,
            extra_entropy=",".join(cleaned_from),
        )
        _ensure_outside_playground(resolved_spin_dir, playground_dir, "spin directory")

    manifest_path = validate_run_path(
        f"{resolved_spin_dir}/spin-manifest.json", label="spin-manifest.json"
    )
    if os.path.exists(manifest_path):
        raise ConfigError(
            f"Refusing to overwrite existing spin directory {resolved_spin_dir} "
            "(spin-manifest.json already exists)."
        )

    # Pre-validate [Ti, T] windows against `model.calendar` BEFORE auto-fetching missing
    # artifacts so invalid spin ranges fail fast without triggering large HF downloads.
    if model_has_missing_files(model, playground_dir=playground_dir, data_dir=data_dir):
        for norm_ti in normalized_from:
            pre_steps = compute_spin_steps(
                norm_ti,
                norm_to,
                model.timestep_hours,
                calendar=model.calendar,
            )
            select_forcing_set(
                norm_ti,
                pre_steps,
                model.timestep_hours,
                model.forcing_sets,
                calendar=model.calendar,
                model_name=model.name,
                preferred_set_name=forcing_set_name,
            )
        model = ensure_model_files_ready(
            model,
            playground_dir=playground_dir,
            data_dir=data_dir,
            auto_fetch=auto_fetch,
        )

    # Resolve each Ti against shipped ICs, compute calendar-exact step counts,
    # and verify a forcing set covers [Ti, T].
    plan: list[tuple[str, ICFileInfo, ICFileInfo | None, int, str]] = []
    seen_calendars: set[str] = set()

    for norm_ti in normalized_from:
        # When `--forcing-set` or a window uniquely picks a scenario-specific forcing set
        # (`ace2-som` 1xCO2 vs 3xCO2, `samudrace-cm4` picontrol-0311 vs picontrol-0151),
        # prefer that forcing set's paired IC list if present.
        cand_ics: Sequence[str] | None = None
        cand_ocean_ics: Sequence[str] | None = None
        try:
            pre_steps_i = compute_spin_steps(
                norm_ti, norm_to, model.timestep_hours, calendar=model.calendar
            )
            hint_fs, _ = select_forcing_set(
                norm_ti,
                pre_steps_i,
                model.timestep_hours,
                model.forcing_sets,
                calendar=model.calendar,
                model_name=model.name,
                preferred_set_name=forcing_set_name,
            )
            if hint_fs.ic_files:
                cand_ics = hint_fs.ic_files
            if hint_fs.ocean_ic_files:
                cand_ocean_ics = hint_fs.ocean_ic_files
        except (ConfigError, ValidationError):
            pass

        ocean_info: ICFileInfo | None = None
        if model.is_coupled:
            ic_info, ocean_info, _ = find_shipped_coupled_ic_for_time(
                model,
                norm_ti,
                candidate_ic_files=cand_ics,
                candidate_ocean_ic_files=cand_ocean_ics,
            )
        else:
            ic_info = find_shipped_ic_for_time(
                model, norm_ti, candidate_ic_files=cand_ics
            )

        seen_calendars.add(ic_info.calendar)
        steps_i = compute_spin_steps(
            norm_ti,
            norm_to,
            model.timestep_hours,
            calendar=ic_info.calendar,
        )
        chosen_fs, end_iso = select_forcing_set(
            norm_ti,
            steps_i,
            model.timestep_hours,
            model.forcing_sets,
            calendar=ic_info.calendar,
            model_name=model.name,
            preferred_set_name=forcing_set_name,
        )
        if end_iso != norm_to:
            raise ConfigError(
                f"Internal calendar step mismatch: {norm_ti} + {steps_i}*{model.timestep_hours}h "
                f"= {end_iso} != {norm_to}."
            )
        plan.append((norm_ti, ic_info, ocean_info, steps_i, chosen_fs.name))

    if len(seen_calendars) > 1:
        raise ConfigError(
            f"Shipped IC files for {model.name!r} have inconsistent CF calendars: "
            f"{sorted(seen_calendars)}."
        )

    makedirs_no_group_other_write(resolved_spin_dir)

    member_records: list[dict[str, Any]] = []
    member_dirs: list[str] = []
    sbatch_paths: list[str] = []
    site_cfg = site_config if site_config is not None else load_site_config()
    base_slurm = slurm_opts if slurm_opts is not None else SlurmOptions.from_site(site_cfg)

    for idx, (norm_ti, ic_info, ocean_info, steps_i, fs_name) in enumerate(plan):
        member_dir = validate_run_path(
            f"{resolved_spin_dir}/{idx}", label=f"spin member {idx} dir"
        )
        member_seed = validate_seed(min(validated_seed + idx * 1000, (2**31) - 10))
        m_manifest = prepare_run(
            model=model,
            start_iso=norm_ti,
            steps=steps_i,
            out_dir=member_dir,
            ic_path=None,
            members=1,
            seed=member_seed,
            lora_path=lora_path,
            forcing_set_name=fs_name,
            save_monthly=False,
            save_prediction=False,
            mode="dry-run",
            slurm_opts=base_slurm,
            site_config=site_cfg,
            playground_dir=playground_dir,
            data_dir=data_dir,
            auto_fetch=False,
            argv=list(argv) if argv is not None else ["aimip", "ic", "spin"],
        )
        member_dirs.append(member_dir)
        sbatch_paths.append(str(m_manifest["sbatch_path"]))
        rec: dict[str, Any] = {
            "index": idx,
            "from_time": norm_ti,
            "to_time": norm_to,
            "steps": steps_i,
            "forcing_set": fs_name,
            "seed": member_seed,
            "shipped_ic": ic_info.path,
            "experiment_dir": member_dir,
            "sbatch_path": str(m_manifest["sbatch_path"]),
            "restart_path": (
                f"{member_dir}/atmosphere/restart.nc"
                if model.is_coupled
                else f"{member_dir}/restart.nc"
            ),
        }
        if model.is_coupled and ocean_info is not None:
            rec["shipped_ocean_ic"] = ocean_info.path
            rec["ocean_restart_path"] = f"{member_dir}/ocean/restart.nc"
        member_records.append(rec)

    total_spin_steps = sum(int(m["steps"]) for m in member_records)
    spin_walltime = estimate_slurm_walltime(
        steps=total_spin_steps,
        timestep_hours=model.timestep_hours,
        n_trajectories=1,
        max_walltime=site_cfg.max_time,
    )
    spin_sbatch_script = render_sbatch_script(
        model_name=f"{model.name}-spin",
        out_dir=resolved_spin_dir,
        playground_dir=playground_dir,
        slurm_opts=base_slurm,
        default_walltime=spin_walltime,
        exec_dirs=member_dirs,
    )
    spin_sbatch_path = validate_run_path(
        f"{resolved_spin_dir}/job.sbatch", label="spin job.sbatch"
    )
    with open(spin_sbatch_path, "w", encoding="utf-8") as sbatch_fh:
        sbatch_fh.write(spin_sbatch_script)
    strip_group_other_write(spin_sbatch_path)

    first_ic = plan[0][1]
    first_ocean_ic = plan[0][2]
    manifest: dict[str, Any] = {
        "schema_version": 2,
        "aimip_version": __version__,
        "argv": list(argv) if argv is not None else [],
        "mode": mode,
        "model": model.name,
        "is_coupled": model.is_coupled,
        "target_time": norm_to,
        "calendar": first_ic.calendar,
        "time_units": first_ic.time_units,
        "shipped_ic_reference": first_ic.path,
        "spin_dir": resolved_spin_dir,
        "sbatch_path": spin_sbatch_path,
        "member_sbatch_paths": sbatch_paths,
        "member_count": len(member_records),
        "members": member_records,
        "created_utc": datetime.datetime.now(datetime.timezone.utc).strftime(
            "%Y-%m-%dT%H:%M:%SZ"
        ),
    }
    if model.is_coupled and first_ocean_ic is not None:
        manifest["shipped_ocean_ic_reference"] = first_ocean_ic.path
    with open(manifest_path, "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2, sort_keys=False)
        fh.write("\n")
    strip_group_other_write(manifest_path)

    if mode == "local":
        for m_dir in member_dirs:
            exec_fme(m_dir)

    return manifest


def _default_companion_ocean_ic_path(atmos_out_path: str) -> str:
    """Derive `<stem>_ocean<ext>` next to *atmos_out_path* for coupled spin collection."""
    stem, ext = os.path.splitext(atmos_out_path)
    if ext:
        return f"{stem}_ocean{ext}"
    return f"{atmos_out_path}_ocean.nc"


def _merge_restart_files(
    restart_paths: Sequence[str],
    validated_out: str,
    *,
    expected_iso: str | None,
    ref_ic_path: str | None,
    ref_calendar: str | None,
    ref_units: str | None,
    extra_global_attrs: dict[str, Any] | None = None,
) -> ICFileInfo:
    """Merge K single-sample `restart.nc` files into *validated_out* along leading `'sample'` dim."""
    import numpy as np
    import xarray as xr
    from xarray.coding.times import CFDatetimeCoder

    ref_var_attrs: dict[str, dict[str, Any]] = {}
    ref_spatial_dims: tuple[str, str] | None = None
    ref_y_coord: Any = None
    ref_x_coord: Any = None
    ref_other_coords: dict[str, Any] = {}
    ref_aux_vars: dict[str, Any] = {}
    ref_global_attrs: dict[str, Any] = {}

    if ref_ic_path is not None and os.path.isfile(ref_ic_path):
        with xr.open_dataset(
            ref_ic_path,
            engine="netcdf4",
            decode_times=CFDatetimeCoder(use_cftime=True),
            decode_timedelta=False,
        ) as ref_ds:
            ref_ds.load()
            ref_sample_dim = str(ref_ds["time"].dims[0])
            if ref_calendar is None and ref_ds["time"].shape[0] > 0:
                ref_calendar = getattr(ref_ds["time"].values[0], "calendar", None)
            if ref_units is None:
                ref_units = ref_ds["time"].encoding.get("units") or ref_ds["time"].attrs.get(
                    "units"
                )
            ref_global_attrs = dict(ref_ds.attrs)

            for vname, vda in ref_ds.data_vars.items():
                if vname == "time":
                    continue
                ref_var_attrs[str(vname)] = dict(vda.attrs)
                if (
                    ref_spatial_dims is None
                    and vda.ndim == 3
                    and str(vda.dims[0]) == ref_sample_dim
                ):
                    ref_spatial_dims = (str(vda.dims[1]), str(vda.dims[2]))
                elif ref_sample_dim not in vda.dims and "sample" not in vda.dims and vda.ndim < 2:
                    ref_aux_vars[str(vname)] = vda.copy(deep=True)

            if ref_spatial_dims is not None:
                ry_name, rx_name = ref_spatial_dims
                if ry_name in ref_ds:
                    ref_y_coord = ref_ds[ry_name].copy(deep=True)
                if rx_name in ref_ds:
                    ref_x_coord = ref_ds[rx_name].copy(deep=True)

            for cname, cda in ref_ds.coords.items():
                cname_s = str(cname)
                if (
                    cname_s != "time"
                    and ref_sample_dim not in cda.dims
                    and (ref_spatial_dims is None or cname_s not in ref_spatial_dims)
                ):
                    ref_other_coords[cname_s] = cda.copy(deep=True)

    restarts: list[Any] = []
    detected_calendar: str | None = ref_calendar
    detected_units: str | None = ref_units

    for idx, restart_path in enumerate(restart_paths):
        if not os.path.isfile(restart_path):
            raise ConfigError(
                f"Missing restart file for spin member {idx}: {restart_path} "
                "(ensure all spin jobs have completed)."
            )
        with xr.open_dataset(
            restart_path,
            engine="netcdf4",
            decode_times=CFDatetimeCoder(use_cftime=True),
            decode_timedelta=False,
        ) as ds:
            loaded = ds.load()
            if "time" not in loaded:
                raise ValidationError(f"Restart file {restart_path} is missing 'time' variable.")
            if loaded["time"].ndim != 1 or loaded["time"].shape[0] != 1:
                raise ValidationError(
                    f"Restart file {restart_path}: expected 1 sample in 'time', "
                    f"got shape {loaded['time'].shape}."
                )
            dt_val = loaded["time"].values[0]
            iso_val = format_cftime(dt_val)
            cal_val = getattr(dt_val, "calendar", None) or loaded["time"].encoding.get("calendar")
            if detected_calendar is None and cal_val:
                detected_calendar = str(cal_val)
            if detected_units is None:
                u_val = loaded["time"].encoding.get("units") or loaded["time"].attrs.get("units")
                if u_val:
                    detected_units = str(u_val)

            if expected_iso is None:
                expected_iso = iso_val
            elif iso_val != expected_iso:
                raise ConfigError(
                    f"Restart {restart_path} has final timestamp {iso_val}, "
                    f"expected target time {expected_iso}."
                )
            restarts.append(loaded)

    assert expected_iso is not None
    final_calendar = detected_calendar or "proleptic_gregorian"
    final_units = detected_units or f"hours since {expected_iso.replace('T', ' ')}"

    first_ds = restarts[0]
    first_sample_dim = str(first_ds["time"].dims[0])
    prog_names = [
        str(k)
        for k, da in first_ds.data_vars.items()
        if k != "time" and da.ndim == 3 and str(da.dims[0]) == first_sample_dim
    ]
    if not prog_names:
        raise ValidationError(
            f"Restart file {restart_paths[0]} contains no 3D prognostic variables."
        )

    r_y_dim = str(first_ds[prog_names[0]].dims[1])
    r_x_dim = str(first_ds[prog_names[0]].dims[2])

    # Verify variable sets and 2D horizontal grid values match across all restarts.
    for idx, r_ds in enumerate(restarts[1:], start=1):
        r_sdim = str(r_ds["time"].dims[0])
        r_names = [
            str(k)
            for k, da in r_ds.data_vars.items()
            if k != "time" and da.ndim == 3 and str(da.dims[0]) == r_sdim
        ]
        if set(r_names) != set(prog_names):
            raise ConfigError(
                f"Restart {restart_paths[idx]} has mismatched variable set "
                f"{sorted(r_names)} vs {sorted(prog_names)}."
            )
        cur_y, cur_x = (
            str(r_ds[prog_names[0]].dims[1]),
            str(r_ds[prog_names[0]].dims[2]),
        )
        if r_y_dim in first_ds and cur_y in r_ds:
            if first_ds[r_y_dim].shape != r_ds[cur_y].shape or not np.allclose(
                first_ds[r_y_dim].values, r_ds[cur_y].values
            ):
                raise ConfigError(
                    f"Restart {restart_paths[idx]} has mismatched latitude/y grid."
                )
        if r_x_dim in first_ds and cur_x in r_ds:
            if first_ds[r_x_dim].shape != r_ds[cur_x].shape or not np.allclose(
                first_ds[r_x_dim].values, r_ds[cur_x].values
            ):
                raise ConfigError(
                    f"Restart {restart_paths[idx]} has mismatched longitude/x grid."
                )

    # Verify grid values against shipped IC reference (even when dim names differ, e.g.
    # restart.nc `lat`/`lon` vs shipped IC `latitude`/`longitude` or `grid_yt`/`grid_xt`).
    if ref_y_coord is not None and r_y_dim in first_ds:
        if ref_y_coord.shape != first_ds[r_y_dim].shape or not np.allclose(
            ref_y_coord.values, first_ds[r_y_dim].values
        ):
            raise ConfigError(
                f"Restart y-coordinate ({r_y_dim}) values do not match shipped IC "
                f"({ref_spatial_dims[0] if ref_spatial_dims else 'y'}) in {ref_ic_path}."
            )
    if ref_x_coord is not None and r_x_dim in first_ds:
        if ref_x_coord.shape != first_ds[r_x_dim].shape or not np.allclose(
            ref_x_coord.values, first_ds[r_x_dim].values
        ):
            raise ConfigError(
                f"Restart x-coordinate ({r_x_dim}) values do not match shipped IC "
                f"({ref_spatial_dims[1] if ref_spatial_dims else 'x'}) in {ref_ic_path}."
            )

    # Preserve shipped IC spatial dim names when available, otherwise keep restart spatial dim names.
    out_y_dim, out_x_dim = (
        ref_spatial_dims if ref_spatial_dims is not None else (r_y_dim, r_x_dim)
    )
    out_dims = ("sample", out_y_dim, out_x_dim)

    k_members = len(restarts)
    merged_vars: dict[str, Any] = {}

    for vname in prog_names:
        slices = []
        for r_ds in restarts:
            sdim, yd, xd = (str(d) for d in r_ds[vname].dims)
            arr = r_ds[vname].transpose(sdim, yd, xd).values
            slices.append(arr)
        stacked = np.concatenate(slices, axis=0)
        merged_attrs = dict(first_ds[vname].attrs)
        if vname in ref_var_attrs:
            combined = dict(ref_var_attrs[vname])
            combined.update(merged_attrs)
            merged_attrs = combined
        merged_vars[vname] = xr.DataArray(stacked, dims=out_dims, attrs=merged_attrs)

    for aux_name, aux_da in ref_aux_vars.items():
        if aux_name not in merged_vars:
            merged_vars[aux_name] = aux_da.copy(deep=True)
    for aux_name, aux_da in first_ds.data_vars.items():
        if (
            aux_name not in merged_vars
            and aux_name != "time"
            and first_sample_dim not in aux_da.dims
            and r_y_dim not in aux_da.dims
            and r_x_dim not in aux_da.dims
        ):
            merged_vars[str(aux_name)] = aux_da.copy(deep=True)

    target_dt = parse_cftime(expected_iso, calendar=final_calendar)
    time_array = np.empty((k_members,), dtype=object)
    for i in range(k_members):
        time_array[i] = target_dt
    merged_vars["time"] = xr.DataArray(time_array, dims=("sample",))

    merged_coords: dict[str, Any] = {}
    if ref_y_coord is not None and ref_x_coord is not None:
        merged_coords[out_y_dim] = xr.DataArray(
            ref_y_coord.values, dims=(out_y_dim,), attrs=dict(ref_y_coord.attrs)
        )
        merged_coords[out_x_dim] = xr.DataArray(
            ref_x_coord.values, dims=(out_x_dim,), attrs=dict(ref_x_coord.attrs)
        )
    else:
        if r_y_dim in first_ds:
            merged_coords[out_y_dim] = xr.DataArray(
                first_ds[r_y_dim].values,
                dims=(out_y_dim,),
                attrs=dict(first_ds[r_y_dim].attrs),
            )
        if r_x_dim in first_ds:
            merged_coords[out_x_dim] = xr.DataArray(
                first_ds[r_x_dim].values,
                dims=(out_x_dim,),
                attrs=dict(first_ds[r_x_dim].attrs),
            )

    for cname, cda in ref_other_coords.items():
        if cname not in merged_coords:
            merged_coords[cname] = cda.copy(deep=True)

    global_attrs = dict(ref_global_attrs)
    global_attrs.update(first_ds.attrs)
    global_attrs["aimip_spin_target_time"] = expected_iso
    global_attrs["aimip_spin_members"] = str(k_members)
    if extra_global_attrs:
        global_attrs.update(extra_global_attrs)

    merged_ds = xr.Dataset(data_vars=merged_vars, coords=merged_coords, attrs=global_attrs)

    parent = os.path.dirname(validated_out)
    if parent:
        makedirs_no_group_other_write(parent)

    tmp_path = os.path.join(parent or "/", f".aimip_collect_{uuid.uuid4().hex}.nc.tmp")
    try:
        merged_ds.to_netcdf(
            tmp_path,
            engine="netcdf4",
            encoding={
                "time": {
                    "units": final_units,
                    "calendar": final_calendar,
                    "dtype": "float64",
                }
            },
        )
        strip_group_other_write(tmp_path)
        try:
            os.link(tmp_path, validated_out)
        except FileExistsError as exc:
            raise ConfigError(
                f"Refusing to overwrite existing output IC file: {validated_out}"
            ) from exc
        strip_group_other_write(validated_out)
    finally:
        if os.path.exists(tmp_path):
            try:
                os.unlink(tmp_path)
            except OSError:
                pass

    return inspect_ic_file(validated_out)


def collect_spin(
    spin_dir: str,
    out_ic_path: str,
    *,
    out_ocean_ic_path: str | None = None,
    target_time: str | None = None,
    shipped_ic_reference: str | None = None,
    shipped_ocean_ic_reference: str | None = None,
    playground_dir: str | None = None,
) -> ICFileInfo:
    """Merge `<spin_dir>/0..K-1/restart.nc` into `<out_ic_path>` with leading `sample` dim.

    For coupled models (`fme.coupled.inference`), merges `<member>/atmosphere/restart.nc`
    into `<out_ic_path>` and `<member>/ocean/restart.nc` into `<out_ocean_ic_path>`
    (defaulting to `<stem>_ocean.nc`).
    """
    safe_out_pre = validate_safe_path(out_ic_path, label="output IC path")
    _ensure_outside_playground(safe_out_pre, playground_dir, "collected spin IC")
    validated_spin_dir = validate_run_path(spin_dir, label="spin directory")
    validated_out = validate_run_path(out_ic_path, label="output IC path")

    if os.path.exists(validated_out):
        raise ConfigError(
            f"Refusing to overwrite existing output IC file: {validated_out}"
        )
    if not os.path.isdir(validated_spin_dir):
        raise ConfigError(f"Spin directory not found: {validated_spin_dir}")

    manifest_path = os.path.join(validated_spin_dir, "spin-manifest.json")
    manifest: dict[str, Any] | None = None
    if os.path.isfile(manifest_path):
        try:
            with open(manifest_path, encoding="utf-8") as fh:
                manifest = json.load(fh)
        except (OSError, json.JSONDecodeError) as exc:
            raise ConfigError(f"Failed to read {manifest_path}: {exc}") from exc

    member_dirs: list[str] = []
    if manifest is not None and isinstance(manifest.get("members"), list):
        for item in manifest["members"]:
            m_dir = validate_run_path(
                str(item["experiment_dir"]), label="spin member experiment_dir"
            )
            member_dirs.append(m_dir)
    else:
        numeric_entries: list[int] = []
        for name in os.listdir(validated_spin_dir):
            full = os.path.join(validated_spin_dir, name)
            if name.isdigit() and os.path.isdir(full):
                numeric_entries.append(int(name))
        numeric_entries.sort()
        if not numeric_entries or numeric_entries != list(range(len(numeric_entries))):
            raise ConfigError(
                f"Expected contiguous member directories 0..K-1 inside {validated_spin_dir}, "
                f"found {[str(i) for i in numeric_entries]}."
            )
        for idx in numeric_entries:
            member_dirs.append(
                validate_run_path(
                    f"{validated_spin_dir}/{idx}", label=f"spin member {idx} dir"
                )
            )

    if not member_dirs:
        raise ConfigError(f"No spin member directories found in {validated_spin_dir}.")

    expected_iso: str | None = None
    if target_time is not None:
        expected_iso = validate_iso_timestamp(target_time, label="--to timestamp")
    elif manifest is not None and manifest.get("target_time"):
        expected_iso = validate_iso_timestamp(
            str(manifest["target_time"]), label="manifest target_time"
        )

    ref_ic_path: str | None = shipped_ic_reference
    if ref_ic_path is None and manifest is not None and manifest.get("shipped_ic_reference"):
        cand = str(manifest["shipped_ic_reference"])
        if os.path.isfile(cand):
            ref_ic_path = validate_safe_path(cand, label="shipped_ic_reference")
    elif ref_ic_path is not None:
        ref_ic_path = validate_safe_path(ref_ic_path, label="shipped_ic_reference")

    ref_calendar: str | None = (
        str(manifest["calendar"])
        if manifest is not None and manifest.get("calendar")
        else None
    )
    ref_units: str | None = (
        str(manifest["time_units"])
        if manifest is not None and manifest.get("time_units")
        else None
    )

    is_coupled = bool(manifest and manifest.get("is_coupled")) or (
        not os.path.isfile(os.path.join(member_dirs[0], "restart.nc"))
        and (
            out_ocean_ic_path is not None
            or any(
                os.path.isdir(os.path.join(m_dir, "atmosphere"))
                or os.path.isdir(os.path.join(m_dir, "ocean"))
                for m_dir in member_dirs
            )
        )
    )

    if not is_coupled:
        if out_ocean_ic_path is not None:
            raise ConfigError(
                "--ocean-out is only valid when collecting a coupled model spin."
            )
        single_restarts = [
            validate_run_path(f"{m_dir}/restart.nc", label=f"member {idx} restart.nc")
            for idx, m_dir in enumerate(member_dirs)
        ]
        return _merge_restart_files(
            single_restarts,
            validated_out,
            expected_iso=expected_iso,
            ref_ic_path=ref_ic_path,
            ref_calendar=ref_calendar,
            ref_units=ref_units,
        )

    raw_ocean_out = (
        out_ocean_ic_path
        if out_ocean_ic_path is not None
        else _default_companion_ocean_ic_path(validated_out)
    )
    safe_ocean_pre = validate_safe_path(raw_ocean_out, label="output ocean IC path")
    _ensure_outside_playground(
        safe_ocean_pre, playground_dir, "collected spin ocean IC"
    )
    validated_ocean_out = validate_run_path(
        raw_ocean_out, label="output ocean IC path"
    )
    if validated_ocean_out == validated_out:
        raise ConfigError(
            "Coupled spin atmosphere and ocean output IC paths must be distinct."
        )
    if os.path.exists(validated_ocean_out):
        raise ConfigError(
            f"Refusing to overwrite existing output ocean IC file: {validated_ocean_out}"
        )

    ref_ocean_ic: str | None = shipped_ocean_ic_reference
    if (
        ref_ocean_ic is None
        and manifest is not None
        and manifest.get("shipped_ocean_ic_reference")
    ):
        cand_o = str(manifest["shipped_ocean_ic_reference"])
        if os.path.isfile(cand_o):
            ref_ocean_ic = validate_safe_path(cand_o, label="shipped_ocean_ic_reference")
    elif ref_ocean_ic is not None:
        ref_ocean_ic = validate_safe_path(
            ref_ocean_ic, label="shipped_ocean_ic_reference"
        )

    atmos_restarts = [
        validate_run_path(
            f"{m_dir}/atmosphere/restart.nc",
            label=f"member {idx} atmosphere/restart.nc",
        )
        for idx, m_dir in enumerate(member_dirs)
    ]
    ocean_restarts = [
        validate_run_path(
            f"{m_dir}/ocean/restart.nc",
            label=f"member {idx} ocean/restart.nc",
        )
        for idx, m_dir in enumerate(member_dirs)
    ]

    # Check that every member has both restart files before writing any output so a
    # partial spin never leaves an orphan atmosphere IC file that blocks retries.
    missing_restarts = [
        rpath
        for pair in zip(atmos_restarts, ocean_restarts)
        for rpath in pair
        if not os.path.isfile(rpath)
    ]
    if missing_restarts:
        raise ConfigError(
            "Missing coupled spin restart file(s): " + ", ".join(missing_restarts)
        )

    atmos_info = _merge_restart_files(
        atmos_restarts,
        validated_out,
        expected_iso=expected_iso,
        ref_ic_path=ref_ic_path,
        ref_calendar=ref_calendar,
        ref_units=ref_units,
        extra_global_attrs={"aimip_companion_ocean_ic": validated_ocean_out},
    )
    try:
        _merge_restart_files(
            ocean_restarts,
            validated_ocean_out,
            expected_iso=atmos_info.timestamps[0],
            ref_ic_path=ref_ocean_ic,
            ref_calendar=ref_calendar,
            ref_units=None,
            extra_global_attrs={"aimip_companion_atmosphere_ic": validated_out},
        )
    except BaseException:
        try:
            os.unlink(validated_out)
        except OSError:
            pass
        raise
    return atmos_info


def replicate_ic_for_ensemble(
    source_ic_path: str,
    out_ic_path: str,
    *,
    target_time: str,
    repeats_per_sample: int,
    select_single_timestamp: bool = True,
    sample_slice: tuple[int, int] | None = None,
    playground_dir: str | None = None,
) -> ICFileInfo:
    """Replicate (or slice) IC states `repeats_per_sample` times along `"sample"` into *out_ic_path*.

    Used for stochastic models when `--members K` (`K > 1`) is requested so `fme` runs
    with `n_ensemble_per_ic: 1` and avoids `Stepper.predict_paired`'s double broadcast
    of forcing (`DataShapesNotUniform` at `fme/ace/stepper/single_module.py:1238-1241`
    and `fme/core/packer.py:35`), while `NoiseConditionedModel.forward`
    (`fme/ace/registry/stochastic_sfno.py:110-127`) draws independent stochastic
    noise per batch sample (`x.shape[0] == K`) at every step.

    Preserves all horizontal dimension names (`latitude`/`longitude`, `grid_yt`/`grid_xt`,
    `lat`/`lon`), coordinate values/attrs, scalar auxiliary variables (`ak_*`/`bk_*`),
    and CF calendar/units without writing wallclock timestamps.
    """
    import numpy as np
    import xarray as xr
    from xarray.coding.times import CFDatetimeCoder

    if isinstance(repeats_per_sample, bool) or not isinstance(repeats_per_sample, int) or repeats_per_sample < 1:
        raise ValidationError(
            f"repeats_per_sample must be a positive integer, got {repeats_per_sample!r}."
        )

    validated_src = validate_safe_path(source_ic_path, label="source IC path")
    validated_out = validate_run_path(out_ic_path, label="replicated IC output path")
    _ensure_outside_playground(validated_out, playground_dir, "replicated ensemble IC")
    norm_target = validate_iso_timestamp(target_time, label="target timestamp")

    if not os.path.isfile(validated_src):
        raise ValidationError(f"Initial condition file not found: {validated_src}")

    with xr.open_dataset(
        validated_src,
        engine="netcdf4",
        decode_times=CFDatetimeCoder(use_cftime=True),
        decode_timedelta=False,
    ) as src_ds:
        loaded = src_ds.load()
        if "time" not in loaded:
            raise ValidationError(f"IC file {validated_src} has no 'time' variable.")
        time_da = loaded["time"]
        if time_da.ndim != 1 or time_da.shape[0] < 1:
            raise ValidationError(
                f"IC file {validated_src}: expected 1D 'time', got shape {time_da.shape}."
            )

        src_sample_dim = str(time_da.dims[0])
        formatted_times = [format_cftime(dt) for dt in time_da.values]

        if sample_slice is not None:
            s_first, s_count = sample_slice
            if s_first < 0 or s_count < 1 or (s_first + s_count) > len(formatted_times):
                raise ConfigError(
                    f"IC sample slice ({s_first}, {s_count}) out of bounds for "
                    f"{validated_src} (n_samples={len(formatted_times)})."
                )
            sub_times = formatted_times[s_first : s_first + s_count]
            mismatched = sorted({ts for ts in sub_times if ts != norm_target})
            if mismatched:
                raise ConfigError(
                    f"Selected sample timestamps in {validated_src} must equal {norm_target}, "
                    f"found {mismatched[:8]}."
                )
            sub_ds = loaded.isel({src_sample_dim: slice(s_first, s_first + s_count)})
        elif select_single_timestamp:
            try:
                sel_idx = formatted_times.index(norm_target)
            except ValueError as exc:
                raise ConfigError(
                    f"Timestamp {norm_target} not found in {validated_src}."
                ) from exc
            sub_ds = loaded.isel({src_sample_dim: [sel_idx]})
        else:
            mismatched = sorted({ts for ts in formatted_times if ts != norm_target})
            if mismatched:
                raise ConfigError(
                    f"All timestamps in {validated_src} must equal {norm_target}, "
                    f"found {mismatched[:8]}."
                )
            sub_ds = loaded

        n_src = int(sub_ds.sizes[src_sample_dim])
        total_samples = n_src * repeats_per_sample

        first_dt = sub_ds["time"].values[0]
        final_calendar = str(
            getattr(first_dt, "calendar", None)
            or time_da.encoding.get("calendar")
            or time_da.attrs.get("calendar")
            or "proleptic_gregorian"
        )
        final_units = str(
            time_da.encoding.get("units")
            or time_da.attrs.get("units")
            or f"hours since {norm_target.replace('T', ' ')}"
        )

        prog_names = [
            str(k)
            for k, da in sub_ds.data_vars.items()
            if k != "time" and da.ndim == 3 and str(da.dims[0]) == src_sample_dim
        ]
        if not prog_names:
            raise ValidationError(
                f"IC file {validated_src} contains no 3D prognostic variables."
            )

        out_y_dim = str(sub_ds[prog_names[0]].dims[1])
        out_x_dim = str(sub_ds[prog_names[0]].dims[2])
        out_dims = ("sample", out_y_dim, out_x_dim)

        out_vars: dict[str, Any] = {}
        for vname in prog_names:
            da = sub_ds[vname]
            sdim, yd, xd = (str(d) for d in da.dims)
            arr = da.transpose(sdim, yd, xd).values
            rep = np.repeat(arr, repeats_per_sample, axis=0)
            out_vars[vname] = xr.DataArray(rep, dims=out_dims, attrs=dict(da.attrs))

        for aux_name, aux_da in sub_ds.data_vars.items():
            if (
                aux_name not in out_vars
                and aux_name != "time"
                and src_sample_dim not in aux_da.dims
            ):
                out_vars[str(aux_name)] = aux_da.copy(deep=True)

        target_dt = parse_cftime(norm_target, calendar=final_calendar)
        time_array = np.empty((total_samples,), dtype=object)
        for i in range(total_samples):
            time_array[i] = target_dt
        out_vars["time"] = xr.DataArray(time_array, dims=("sample",))

        out_coords: dict[str, Any] = {}
        if out_y_dim in sub_ds:
            out_coords[out_y_dim] = xr.DataArray(
                sub_ds[out_y_dim].values,
                dims=(out_y_dim,),
                attrs=dict(sub_ds[out_y_dim].attrs),
            )
        if out_x_dim in sub_ds:
            out_coords[out_x_dim] = xr.DataArray(
                sub_ds[out_x_dim].values,
                dims=(out_x_dim,),
                attrs=dict(sub_ds[out_x_dim].attrs),
            )
        for cname, cda in sub_ds.coords.items():
            cname_s = str(cname)
            if cname_s != "time" and src_sample_dim not in cda.dims and cname_s not in out_coords:
                out_coords[cname_s] = cda.copy(deep=True)

        global_attrs = dict(sub_ds.attrs)
        global_attrs["aimip_replicated_target_time"] = norm_target
        global_attrs["aimip_replicated_members"] = str(total_samples)

        out_ds = xr.Dataset(data_vars=out_vars, coords=out_coords, attrs=global_attrs)

    parent = os.path.dirname(validated_out) or "/"
    if parent != "/":
        makedirs_no_group_other_write(parent)
    tmp_path = os.path.join(parent, f".aimip_rep_{uuid.uuid4().hex}.nc.tmp")
    try:
        out_ds.to_netcdf(
            tmp_path,
            engine="netcdf4",
            encoding={
                "time": {
                    "units": final_units,
                    "calendar": final_calendar,
                    "dtype": "float64",
                }
            },
        )
        strip_group_other_write(tmp_path)
        os.replace(tmp_path, validated_out)
        strip_group_other_write(validated_out)
    finally:
        if os.path.exists(tmp_path):
            try:
                os.unlink(tmp_path)
            except OSError:
                pass

    return inspect_ic_file(validated_out)
