"""Installed model-family adapter protocol and allow-listed adapters (`fme`, `e2s-dlesym`, `arches`).

Registry entries may reference ONLY installed adapter identifiers in `INSTALLED_ADAPTERS`;
arbitrary import paths, shell fragments, or external commands in registry JSON are refused.
"""

from __future__ import annotations

import fnmatch
import os
from typing import Any, Protocol

from ..emu import (
    ARCHES_DOMAIN,
    ARCHES_SCENARIOS,
    ARCHES_START,
    _gpu_hours as _emu_gpu_hours,
    _walltime as _emu_walltime,
    precheck_emu_request,
)
from ..errors import ConfigError, RegistryError, ValidationError
from ..lora import inspect_lora, sha256_file
from ..plan_types import AssetRef, RunSpec
from ..registry import SAMUDRACE_CM4_IC_SLUGS, ModelSpec, resolve_data_dir
from ..runner import (
    ACE2_ERA5_GPU_HOURS_PER_MODEL_YEAR,
    HOURS_PER_JULIAN_YEAR,
    SlurmOptions,
    _scenario_for_ic,
    compute_slurm_walltime_details,
    exec_fme,
    model_has_missing_files,
    prepare_run,
    resolve_ic_and_members,
)
from ..setup_data import (
    ARCHES_ARCHIVE_SIZE,
    ARCHES_CODE_REL_DIR,
    ARCHES_TREE_SHA256,
    compute_arches_tree_digest,
    load_lock_file,
    resolve_lock_path,
)
from ..site_config import SiteConfig
from ..status import get_run_status, summarize_run
from ..time_utils import resolve_horizon, select_forcing_set
from ..validators import (
    MAX_SEED,
    validate_int_range,
    validate_iso_timestamp,
    validate_visible_input_path,
)


class FamilyAdapter(Protocol):
    """Uniform lifecycle contract implemented by every installed model-family adapter."""

    family_id: str

    def validate(
        self,
        model: ModelSpec,
        spec: RunSpec,
        *,
        data_dir: str | None = None,
    ) -> dict[str, Any]: ...

    def estimate(
        self,
        model: ModelSpec,
        validated: dict[str, Any],
        site: SiteConfig,
    ) -> dict[str, Any]: ...

    def stage_list(
        self,
        model: ModelSpec,
        validated: dict[str, Any],
        *,
        data_dir: str | None = None,
        lock_path: str | None = None,
    ) -> list[AssetRef]: ...

    def build_config(
        self,
        model: ModelSpec,
        spec: RunSpec,
        validated: dict[str, Any],
        *,
        out_dir: str,
        allow_fetch: bool,
        slurm_opts: SlurmOptions,
        site_config: SiteConfig | None = None,
        playground_dir: str | None = None,
        data_dir: str | None = None,
        manifest_mode: str = "dry-run",
        argv: list[str] | None = None,
    ) -> dict[str, Any]: ...

    def execute(self, out_dir: str) -> dict[str, Any]: ...

    def inspect(self, out_dir: str, *, sacct_json_path: str | None = None) -> dict[str, Any]: ...


def _load_lock_entries(
    *,
    data_dir: str | None = None,
    lock_path: str | None = None,
) -> dict[str, dict[str, Any]]:
    """Load `aimip.lock` entries when available (never failing if a synthetic test omits a lock file)."""
    try:
        resolved_lock = resolve_lock_path(
            lock_path, dest_dir=data_dir, must_exist=False
        )
        if resolved_lock and os.path.isfile(resolved_lock):
            doc = load_lock_file(resolved_lock)
            files = doc.get("files")
            if isinstance(files, dict):
                return files
    except Exception:
        pass
    return {}


def _asset_from_rel(
    rel_path: str,
    abs_path: str,
    role: str,
    lock_files: dict[str, dict[str, Any]],
    *,
    fallback_sha: str | None = None,
) -> AssetRef:
    entry = lock_files.get(rel_path) or {}
    sha = entry.get("sha256") or fallback_sha
    size_val = entry.get("size")
    cached = os.path.isfile(abs_path) and not os.path.islink(abs_path)
    if size_val is None and cached:
        try:
            size_val = os.path.getsize(abs_path)
        except OSError:
            size_val = None
    return AssetRef(
        rel_path=rel_path,
        path=abs_path,
        role=role,
        sha256=str(sha) if sha else None,
        size_bytes=int(size_val) if isinstance(size_val, int) else None,
        cached=cached,
    )


KNOWN_SHIPPED_IC_TIMESTAMPS: dict[str, tuple[str, ...]] = {
    "ace2-era5": tuple(
        f"{y}-{m:02d}-01T00:00:00" for y in (1979, 2001) for m in range(1, 13)
    ),
    "ace2-shield": ("1979-01-01T00:00:00",),
    "ace2s-shield-plus": ("1979-01-01T00:00:00",),
    "ace2-eamv3": ("1994-01-01T00:00:00",),
    "hiro-ace": ("1994-01-01T00:00:00",),
    "ace2-som": ("2032-01-01T00:00:00",),
    "ace2s": tuple(
        f"{y}-{m:02d}-01T00:00:00" for y in (2014, 2023) for m in range(1, 13)
    ),
    "hiro-acev2": tuple(
        f"{y}-{m:02d}-01T00:00:00" for y in (2014, 2023) for m in range(1, 13)
    ),
    "samudrace-cm4": tuple(ts for ts, _ in SAMUDRACE_CM4_IC_SLUGS),
    "samudrace-e3smv3": ("0425-01-03T12:00:00",),
}


def _check_unfetched_shipped_ic_timestamp(model_name: str, norm_start: str) -> None:
    """Reject impossible `--start` timestamps against shipped IC catalogs before downloading."""
    known = KNOWN_SHIPPED_IC_TIMESTAMPS.get(model_name)
    if known is not None and norm_start not in known:
        shown = ", ".join(known[:6]) + (" ..." if len(known) > 6 else "")
        raise ConfigError(
            f"No shipped initial condition for {model_name!r} at {norm_start} "
            f"(shipped timestamps: {shown}; pass `--ic <restart.nc>` or spin up with `aimip ic spin`).",
            field="start",
            allowed=known,
        )


def _hash_custom_forcing_dir_assets(
    fdir: str | None,
    file_pattern: str | None,
    role: str,
    seen: set[str],
) -> list[AssetRef]:
    """Bind custom forcing directories and their matching NetCDF files into `AssetRef`s."""
    if not fdir:
        return []
    out: list[AssetRef] = []
    pat = file_pattern or "*.nc"
    max_inline_hash_bytes = 128 * 1024 * 1024
    if os.path.isdir(fdir) and not os.path.islink(fdir):
        matched_files: list[str] = []
        try:
            for entry in sorted(os.listdir(fdir)):
                if fnmatch.fnmatch(entry, pat):
                    full = os.path.join(fdir, entry)
                    if os.path.isfile(full) and not os.path.islink(full):
                        matched_files.append(full)
        except OSError:
            matched_files = []
        if matched_files:
            for full in matched_files:
                if full in seen:
                    continue
                try:
                    st = os.stat(full)
                    fsize: int | None = int(st.st_size)
                    fsha = (
                        sha256_file(full)
                        if st.st_size <= max_inline_hash_bytes
                        else f"stat:{st.st_size}:{st.st_mtime_ns}"
                    )
                except OSError:
                    fsize = None
                    fsha = None
                out.append(
                    AssetRef(
                        rel_path=f"{role}:{full}",
                        path=full,
                        role=role,
                        sha256=fsha,
                        size_bytes=fsize,
                        cached=True,
                    )
                )
                seen.add(full)
            return out
    if fdir not in seen:
        out.append(
            AssetRef(
                rel_path=f"{role}_dir:{fdir}",
                path=fdir,
                role=role,
                sha256=f"dir:{fdir}:{pat}",
                size_bytes=None,
                cached=os.path.isdir(fdir) and not os.path.islink(fdir),
            )
        )
        seen.add(fdir)
    return out


class FmeFamilyAdapter:
    """Adapter wrapping `aimip.runner` for single-component and coupled `fme` models."""

    family_id = "fme"

    def validate(
        self,
        model: ModelSpec,
        spec: RunSpec,
        *,
        data_dir: str | None = None,
    ) -> dict[str, Any]:
        norm_start = validate_iso_timestamp(spec.start, label="--start")
        steps, end_iso = resolve_horizon(
            start_iso=norm_start,
            steps=spec.steps,
            length=spec.length,
            end_iso=spec.end,
            timestep_hours=model.timestep_hours,
            calendar=model.calendar,
            family=model.family,
            model_name=model.name,
        )
        seed = (
            validate_int_range(spec.seed, "seed", min_value=0, max_value=MAX_SEED)
            if spec.seed is not None
            else 0
        )

        eff_forcing_set = (
            _scenario_for_ic(model, spec.ic, norm_start, steps)
            if (spec.ic is not None and spec.forcing_set is None)
            else spec.forcing_set
        )
        selected_fs, _ = select_forcing_set(
            start_iso=norm_start,
            steps=steps,
            timestep_hours=model.timestep_hours,
            forcing_sets=model.forcing_sets,
            calendar=model.calendar,
            model_name=model.name,
            preferred_set_name=eff_forcing_set,
        )

        lora_path: str | None = None
        lora_sha: str | None = None
        lora_info = None
        if spec.lora is not None:
            if model.is_coupled:
                raise ConfigError(
                    f"--lora single-module checkpoint override is not supported for coupled model {model.name!r}.",
                    field="lora",
                )
            lora_path = validate_visible_input_path(spec.lora, label="--lora checkpoint")
            if not os.path.isfile(lora_path):
                raise ConfigError(
                    f"LoRA checkpoint not found: {lora_path}",
                    field="lora",
                )
            lora_info = inspect_lora(lora_path)
            lora_sha = sha256_file(lora_path)

        if spec.ocean_ic is not None and not model.is_coupled:
            raise ConfigError(
                f"--ocean-ic is only supported for coupled models (`fme.coupled.inference`), not {model.name!r}.",
                field="ocean_ic",
            )

        missing_rels = model_has_missing_files(model, data_dir=data_dir)
        active_ics = selected_fs.ic_files or model.ic_files
        active_ocean_ics = selected_fs.ocean_ic_files or model.ocean_ic_files
        shipped_ics_present = (
            bool(active_ics)
            and all(os.path.isfile(p) for p in active_ics)
            and all(os.path.isfile(p) for p in active_ocean_ics)
        )

        resolved_data = resolve_data_dir(data_dir) or ""
        lock_files = _load_lock_entries(data_dir=resolved_data or None)

        def _lookup_lock_sha(abs_p: str | None) -> str | None:
            if not abs_p:
                return None
            if resolved_data and abs_p.startswith(resolved_data.rstrip("/") + "/"):
                rel = abs_p[len(resolved_data.rstrip("/")) + 1 :]
                entry = lock_files.get(rel)
                if isinstance(entry, dict) and entry.get("sha256"):
                    return str(entry["sha256"])
            return None

        ic_source: str | None = None
        ic_sha: str | None = None
        ocean_ic_source: str | None = None
        ocean_ic_sha: str | None = None
        effective_calendar: str = model.calendar
        ic_checked = bool(spec.ic is not None or shipped_ics_present)

        if spec.ic is not None or shipped_ics_present:
            ic_res = resolve_ic_and_members(
                model=model,
                norm_start=norm_start,
                ic_path=spec.ic,
                ocean_ic_path=spec.ocean_ic,
                members=spec.members,
                candidate_ic_files=selected_fs.ic_files or None,
                candidate_ocean_ic_files=selected_fs.ocean_ic_files or None,
            )
            effective_calendar = ic_res.source_ic.calendar
            if effective_calendar != model.calendar:
                steps, end_iso = resolve_horizon(
                    start_iso=norm_start,
                    steps=spec.steps,
                    length=spec.length,
                    end_iso=spec.end,
                    timestep_hours=model.timestep_hours,
                    calendar=effective_calendar,
                    family=model.family,
                    model_name=model.name,
                )
                selected_fs, _ = select_forcing_set(
                    start_iso=norm_start,
                    steps=steps,
                    timestep_hours=model.timestep_hours,
                    forcing_sets=model.forcing_sets,
                    calendar=effective_calendar,
                    model_name=model.name,
                    preferred_set_name=eff_forcing_set,
                )
            total_members = ic_res.total_members
            ensemble_strategy = ic_res.ensemble_strategy
            ensemble_note = (
                f"{total_members} ensemble member(s) ({ensemble_strategy})."
                if total_members > 1
                else "Single trajectory from initial state."
            )
            ic_source = ic_res.source_ic.path
            if spec.ic is not None or not model.required_files:
                if os.path.isfile(ic_source):
                    ic_sha = sha256_file(ic_source)
            else:
                ic_sha = _lookup_lock_sha(ic_source)
            if ic_res.source_ocean_ic is not None:
                ocean_ic_source = ic_res.source_ocean_ic.path
                if spec.ocean_ic is not None or spec.ic is not None or not model.required_files:
                    if os.path.isfile(ocean_ic_source):
                        ocean_ic_sha = sha256_file(ocean_ic_source)
                else:
                    ocean_ic_sha = _lookup_lock_sha(ocean_ic_source)
            if lora_info is not None and lora_path is not None:
                out_name_set = set(lora_info.out_names)
                ckpt_prognostic = [v for v in lora_info.in_names if v in out_name_set]
                if ckpt_prognostic and ic_res.source_ic.variable_names:
                    ic_var_set = set(ic_res.source_ic.variable_names)
                    missing_prog = sorted(v for v in ckpt_prognostic if v not in ic_var_set)
                    if missing_prog:
                        shown_missing = ", ".join(missing_prog[:8])
                        raise ConfigError(
                            f"--lora checkpoint {lora_path} appears incompatible with model "
                            f"{model.name!r}: checkpoint prognostic variable(s) [{shown_missing}] "
                            f"are missing from initial condition {ic_res.source_ic.path}.",
                            field="lora",
                        )
        else:
            # Side-effect-free planning before model files are fetched locally:
            _check_unfetched_shipped_ic_timestamp(model.name, norm_start)
            if spec.members is not None:
                req_m = validate_int_range(spec.members, "members", min_value=1, max_value=10_000)
                if not model.stochastic and req_m > 1:
                    raise ConfigError(
                        f"Model {model.name!r} is deterministic: --members {req_m} requires a multi-sample IC "
                        "(prepare one with `aimip ic spin` -> `aimip collect`, or use --members 1).",
                        field="members",
                        allowed=[1],
                    )
                total_members = req_m
            else:
                total_members = model.default_n_ensemble_per_ic

            start_year = norm_start[:4]
            start_slug = norm_start.replace(":", "-")
            candidate_ics = list(selected_fs.ic_files or model.ic_files)
            ic_source = next(
                (
                    p
                    for p in candidate_ics
                    if start_slug in p or f"_{start_year}" in p or f"/{start_year}" in p
                ),
                candidate_ics[0] if candidate_ics else None,
            )
            ic_sha = _lookup_lock_sha(ic_source)

            candidate_ocean_ics = list(selected_fs.ocean_ic_files or model.ocean_ic_files)
            ocean_ic_source = next(
                (
                    p
                    for p in candidate_ocean_ics
                    if start_slug in p or f"_{start_year}" in p or f"/{start_year}" in p
                ),
                candidate_ocean_ics[0] if candidate_ocean_ics else None,
            )
            ocean_ic_sha = _lookup_lock_sha(ocean_ic_source)

            if total_members == 1:
                ensemble_strategy = "single_member"
            elif model.name == "samudrace-e3smv3" and total_members <= 3:
                ensemble_strategy = "multi_sample_ic"
            else:
                ensemble_strategy = "replicated_ic_sample_dim"
            ensemble_note = (
                f"{total_members} ensemble member(s) ({ensemble_strategy}; pending fetch)."
                if total_members > 1
                else "Single trajectory from shipped IC (pending fetch)."
            )

        return {
            "norm_start": norm_start,
            "end_time": end_iso,
            "steps": steps,
            "calendar": effective_calendar,
            "members": total_members,
            "seed": seed,
            "forcing_set": selected_fs,
            "ensemble_strategy": ensemble_strategy,
            "ensemble_note": ensemble_note,
            "ic_source": ic_source,
            "ic_sha256": ic_sha,
            "ocean_ic_source": ocean_ic_source,
            "ocean_ic_sha256": ocean_ic_sha,
            "ic_checked": ic_checked,
            "lora_checkpoint": lora_path,
            "lora_sha256": lora_sha,
            "missing_rels": missing_rels,
        }

    def estimate(
        self,
        model: ModelSpec,
        validated: dict[str, Any],
        site: SiteConfig,
    ) -> dict[str, Any]:
        steps = int(validated["steps"])
        members = int(validated["members"])
        req_time = validated.get("requested_walltime")
        max_wt = site.max_time or "12:00:00"
        wt, uncapped, capped = compute_slurm_walltime_details(
            steps=steps,
            timestep_hours=model.timestep_hours,
            n_trajectories=members,
            max_walltime=max_wt,
        )
        effective_time = str(req_time) if req_time else wt
        gpu_hours = round(
            steps
            * members
            * model.timestep_hours
            / HOURS_PER_JULIAN_YEAR
            * ACE2_ERA5_GPU_HOURS_PER_MODEL_YEAR,
            6,
        )
        return {
            "time": effective_time,
            "walltime_uncapped": uncapped,
            "walltime_capped": capped if not req_time else False,
            "gpu_hours_estimate": gpu_hours,
            "estimate_basis": "ACE2-ERA5 1x A100-80GB baseline (~0.064 GPU-h / model year)",
        }

    def stage_list(
        self,
        model: ModelSpec,
        validated: dict[str, Any],
        *,
        data_dir: str | None = None,
        lock_path: str | None = None,
    ) -> list[AssetRef]:
        resolved_data = resolve_data_dir(data_dir) or ""
        lock_files = _load_lock_entries(data_dir=resolved_data or None, lock_path=lock_path)
        assets: list[AssetRef] = []
        seen: set[str] = set()

        if model.required_files and resolved_data:
            for rel in model.required_files:
                abs_p = os.path.join(resolved_data, rel)
                role = (
                    "checkpoint"
                    if rel.startswith("models/") and (rel.endswith(".tar") or rel.endswith(".ckpt"))
                    else ("ic" if "/ic/" in rel or rel.endswith("/ic.nc") else "forcing")
                )
                fallback = model.checkpoint_sha256 if abs_p == model.checkpoint else None
                assets.append(_asset_from_rel(rel, abs_p, role, lock_files, fallback_sha=fallback))
                seen.add(abs_p)
        else:
            if model.checkpoint not in seen:
                assets.append(
                    _asset_from_rel(
                        os.path.basename(model.checkpoint),
                        model.checkpoint,
                        "checkpoint",
                        lock_files,
                        fallback_sha=model.checkpoint_sha256 or None,
                    )
                )
                seen.add(model.checkpoint)
            for ic_p in (*model.ic_files, *model.ocean_ic_files):
                if ic_p not in seen:
                    assets.append(
                        _asset_from_rel(
                            os.path.basename(ic_p),
                            ic_p,
                            "ic",
                            lock_files,
                            fallback_sha=sha256_file(ic_p) if os.path.isfile(ic_p) else None,
                        )
                    )
                    seen.add(ic_p)
            selected_fs = validated.get("forcing_set")
            if selected_fs is not None:
                assets.extend(
                    _hash_custom_forcing_dir_assets(
                        getattr(selected_fs, "dir", None),
                        getattr(selected_fs, "file_pattern", None),
                        "forcing",
                        seen,
                    )
                )
                assets.extend(
                    _hash_custom_forcing_dir_assets(
                        getattr(selected_fs, "ocean_dir", None),
                        getattr(selected_fs, "ocean_file_pattern", None),
                        "ocean_forcing",
                        seen,
                    )
                )

        custom_ic = validated.get("ic_source")
        if isinstance(custom_ic, str) and custom_ic not in seen:
            assets.append(
                AssetRef(
                    rel_path=os.path.basename(custom_ic),
                    path=custom_ic,
                    role="custom_ic",
                    sha256=validated.get("ic_sha256"),
                    size_bytes=os.path.getsize(custom_ic) if os.path.isfile(custom_ic) else None,
                    cached=os.path.isfile(custom_ic),
                )
            )
            seen.add(custom_ic)

        custom_ocean = validated.get("ocean_ic_source")
        if isinstance(custom_ocean, str) and custom_ocean not in seen:
            assets.append(
                AssetRef(
                    rel_path=os.path.basename(custom_ocean),
                    path=custom_ocean,
                    role="custom_ocean_ic",
                    sha256=validated.get("ocean_ic_sha256"),
                    size_bytes=os.path.getsize(custom_ocean) if os.path.isfile(custom_ocean) else None,
                    cached=os.path.isfile(custom_ocean),
                )
            )
            seen.add(custom_ocean)

        lora_p = validated.get("lora_checkpoint")
        if isinstance(lora_p, str) and lora_p not in seen:
            assets.append(
                AssetRef(
                    rel_path=os.path.basename(lora_p),
                    path=lora_p,
                    role="lora",
                    sha256=validated.get("lora_sha256"),
                    size_bytes=os.path.getsize(lora_p) if os.path.isfile(lora_p) else None,
                    cached=os.path.isfile(lora_p),
                )
            )
        return assets

    def build_config(
        self,
        model: ModelSpec,
        spec: RunSpec,
        validated: dict[str, Any],
        *,
        out_dir: str,
        allow_fetch: bool,
        slurm_opts: SlurmOptions,
        site_config: SiteConfig | None = None,
        playground_dir: str | None = None,
        data_dir: str | None = None,
        manifest_mode: str = "dry-run",
        argv: list[str] | None = None,
    ) -> dict[str, Any]:
        return prepare_run(
            model=model,
            start_iso=validated["norm_start"],
            steps=int(validated["steps"]),
            ic_path=spec.ic,
            ocean_ic_path=spec.ocean_ic,
            members=spec.members,
            seed=spec.seed,
            lora_path=spec.lora,
            forcing_set_name=spec.forcing_set,
            out_dir=out_dir,
            save_monthly=spec.monthly,
            save_prediction=spec.prediction,
            mode=manifest_mode,
            slurm_opts=slurm_opts,
            site_config=site_config,
            playground_dir=playground_dir,
            data_dir=data_dir,
            auto_fetch=allow_fetch,
            argv=argv,
        )

    def execute(self, out_dir: str) -> dict[str, Any]:
        from .. import runner

        return runner.exec_fme(out_dir)

    def inspect(self, out_dir: str, *, sacct_json_path: str | None = None) -> dict[str, Any]:
        return {
            "status": get_run_status(out_dir, sacct_json_path=sacct_json_path),
            "summary": summarize_run(out_dir) if self._has_monthly_outputs(out_dir) else None,
        }

    @staticmethod
    def _has_monthly_outputs(out_dir: str) -> bool:
        for rel in (
            "monthly_mean_predictions.nc",
            os.path.join("atmosphere", "monthly_mean_predictions.nc"),
        ):
            if os.path.isfile(os.path.join(out_dir, rel)):
                return True
        return False


class EmuFamilyAdapter:
    """Adapter wrapping `aimip.emu` for `e2s-dlesym` and `arches` families."""

    def __init__(self, family_id: str) -> None:
        self.family_id = family_id

    def validate(
        self,
        model: ModelSpec,
        spec: RunSpec,
        *,
        data_dir: str | None = None,
    ) -> dict[str, Any]:
        norm_start = validate_iso_timestamp(spec.start, label="--start")
        steps, end_iso = resolve_horizon(
            start_iso=norm_start,
            steps=spec.steps,
            length=spec.length,
            end_iso=spec.end,
            timestep_hours=model.timestep_hours,
            calendar=model.calendar,
            family=model.family,
            model_name=model.name,
        )
        seed = (
            validate_int_range(spec.seed, "seed", min_value=0, max_value=MAX_SEED)
            if spec.seed is not None
            else 0
        )

        checked = precheck_emu_request(
            model,
            start_iso=norm_start,
            steps=steps,
            members=spec.members,
            ic_path=spec.ic,
            ocean_ic_path=spec.ocean_ic,
            lora_path=spec.lora,
            forcing_set_name=spec.forcing_set,
        )

        total_members = int(checked["members"])
        scenario: str | None = checked.get("scenario")
        if model.family == "arches":
            selected_name = scenario or ARCHES_SCENARIOS[0]
            if norm_start != ARCHES_START:
                raise ConfigError(
                    f"Model {model.name!r} starts only at {ARCHES_START}.",
                    field="start",
                    allowed=[ARCHES_START],
                )
            ensemble_strategy = "single_scenario_state"
            ensemble_note = (
                f"Deterministic IPSL scenario rollout ({selected_name}, domain={ARCHES_DOMAIN[selected_name]})."
            )
        else:
            selected_name = "era5-initial-state"
            ensemble_strategy = (
                "stochastic_noise_from_era5_ic"
                if (model.stochastic and total_members > 1)
                else "single_era5_ic"
            )
            ensemble_note = (
                f"{total_members} stochastic member(s) from staged ARCO-ERA5 state (seed + member_id)."
                if total_members > 1
                else "Deterministic rollout from staged ARCO-ERA5 state."
            )

        selected_fs = next(
            (fs for fs in model.forcing_sets if fs.name == selected_name),
            model.forcing_sets[0],
        )
        missing_rels = model_has_missing_files(model, data_dir=data_dir)
        return {
            "norm_start": norm_start,
            "end_time": end_iso,
            "steps": steps,
            "calendar": model.calendar,
            "members": total_members,
            "seed": seed,
            "forcing_set": selected_fs,
            "ensemble_strategy": ensemble_strategy,
            "ensemble_note": ensemble_note,
            "ic_source": None if model.family == "arches" else f"arco-era5@{norm_start}",
            "ic_sha256": None,
            "ocean_ic_source": None,
            "ocean_ic_sha256": None,
            "lora_checkpoint": None,
            "lora_sha256": None,
            "missing_rels": missing_rels,
        }

    def estimate(
        self,
        model: ModelSpec,
        validated: dict[str, Any],
        site: SiteConfig,
    ) -> dict[str, Any]:
        steps = int(validated["steps"])
        members = int(validated["members"])
        req_time = validated.get("requested_walltime")
        max_wt = site.max_time or "12:00:00"
        wt, uncapped, capped = _emu_walltime(model.family, steps, members, max_wt)
        gpu_hours = round(_emu_gpu_hours(model.family, steps, members), 6)
        basis = (
            "DLESyM 1x A100-80GB (~40 s / model year / member)"
            if model.family == "e2s-dlesym"
            else "ArchesClimate-SSP 1x A100-80GB (~2 s / simulated month)"
        )
        return {
            "time": str(req_time) if req_time else wt,
            "walltime_uncapped": uncapped,
            "walltime_capped": capped if not req_time else False,
            "gpu_hours_estimate": gpu_hours,
            "estimate_basis": basis,
        }

    def stage_list(
        self,
        model: ModelSpec,
        validated: dict[str, Any],
        *,
        data_dir: str | None = None,
        lock_path: str | None = None,
    ) -> list[AssetRef]:
        resolved_data = resolve_data_dir(data_dir) or ""
        lock_files = _load_lock_entries(data_dir=resolved_data or None, lock_path=lock_path)
        assets: list[AssetRef] = []

        for rel in model.required_files:
            abs_p = os.path.join(resolved_data, rel) if resolved_data else rel
            role = "forcing" if "memmap" in rel or rel.startswith("inputs/") else "package"
            fallback = model.checkpoint_sha256 if abs_p == model.checkpoint else None
            assets.append(_asset_from_rel(rel, abs_p, role, lock_files, fallback_sha=fallback))

        if model.family == "arches":
            override_code = (os.environ.get("AIMIP_ARCHES_CODE") or "").strip()
            if override_code:
                tree_sha, _ = compute_arches_tree_digest(override_code)
                assets.append(
                    AssetRef(
                        rel_path=ARCHES_CODE_REL_DIR,
                        path=override_code,
                        role="source_tree",
                        sha256=tree_sha,
                        size_bytes=None,
                        cached=True,
                    )
                )
            else:
                code_abs = (
                    os.path.join(resolved_data, ARCHES_CODE_REL_DIR)
                    if resolved_data
                    else ARCHES_CODE_REL_DIR
                )
                code_cached = os.path.isdir(code_abs) and not os.path.islink(code_abs)
                assets.append(
                    AssetRef(
                        rel_path=ARCHES_CODE_REL_DIR,
                        path=code_abs,
                        role="source_tree",
                        sha256=ARCHES_TREE_SHA256,
                        size_bytes=ARCHES_ARCHIVE_SIZE,
                        cached=code_cached,
                    )
                )
        else:
            out_cand = validated.get("output_dir")
            staged_ic = os.path.join(str(out_cand), "ic_era5.pt") if out_cand else ""
            if staged_ic and os.path.isfile(staged_ic) and not os.path.islink(staged_ic):
                ic_sha = sha256_file(staged_ic)
                assets.append(
                    AssetRef(
                        rel_path="ic_era5.pt",
                        path=staged_ic,
                        role="dynamic_era5_ic",
                        sha256=ic_sha,
                        size_bytes=os.path.getsize(staged_ic),
                        cached=True,
                        prospective=False,
                    )
                )
            else:
                # DLESyM fetches the 48h initial ERA5 state (~35 MiB) into `<out_dir>/ic_era5.pt` during `prepare`.
                assets.append(
                    AssetRef(
                        rel_path="ic_era5.pt",
                        path=f"arco-era5://{validated['norm_start']}",
                        role="dynamic_era5_ic",
                        sha256=None,
                        size_bytes=36 * 1024 * 1024,
                        cached=False,
                        prospective=True,
                    )
                )
        return assets

    def build_config(
        self,
        model: ModelSpec,
        spec: RunSpec,
        validated: dict[str, Any],
        *,
        out_dir: str,
        allow_fetch: bool,
        slurm_opts: SlurmOptions,
        site_config: SiteConfig | None = None,
        playground_dir: str | None = None,
        data_dir: str | None = None,
        manifest_mode: str = "dry-run",
        argv: list[str] | None = None,
    ) -> dict[str, Any]:
        return prepare_run(
            model=model,
            start_iso=validated["norm_start"],
            steps=int(validated["steps"]),
            ic_path=spec.ic,
            ocean_ic_path=spec.ocean_ic,
            members=spec.members,
            seed=spec.seed,
            lora_path=spec.lora,
            forcing_set_name=spec.forcing_set,
            out_dir=out_dir,
            save_monthly=spec.monthly,
            save_prediction=spec.prediction,
            mode=manifest_mode,
            slurm_opts=slurm_opts,
            site_config=site_config,
            playground_dir=playground_dir,
            data_dir=data_dir,
            auto_fetch=allow_fetch,
            argv=argv,
        )

    def execute(self, out_dir: str) -> dict[str, Any]:
        from .. import runner

        return runner.exec_fme(out_dir)

    def inspect(self, out_dir: str, *, sacct_json_path: str | None = None) -> dict[str, Any]:
        has_monthly = os.path.isfile(
            os.path.join(out_dir, "atmosphere", "monthly_mean_predictions.nc")
        )
        return {
            "status": get_run_status(out_dir, sacct_json_path=sacct_json_path),
            "summary": summarize_run(out_dir) if has_monthly else None,
        }


INSTALLED_ADAPTERS: dict[str, FamilyAdapter] = {
    "fme": FmeFamilyAdapter(),
    "e2s-dlesym": EmuFamilyAdapter("e2s-dlesym"),
    "arches": EmuFamilyAdapter("arches"),
    "emu": EmuFamilyAdapter("e2s-dlesym"),
}


def get_family_adapter(family_id: str) -> FamilyAdapter:
    """Resolve an installed `FamilyAdapter` from the closed allowlist; refuse unknown or import-like strings."""
    if (
        not isinstance(family_id, str)
        or not family_id
        or family_id not in INSTALLED_ADAPTERS
    ):
        raise RegistryError(
            f"Unsupported model family/adapter {family_id!r}; only installed adapters are permitted.",
            field="family",
            allowed=sorted(INSTALLED_ADAPTERS.keys()),
        )
    return INSTALLED_ADAPTERS[family_id]
