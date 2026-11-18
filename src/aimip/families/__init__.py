"""Installed model-family adapter protocol and allow-listed adapters (`fme`, `e2s-dlesym`, `arches`).

Registry entries may reference ONLY installed adapter identifiers in `INSTALLED_ADAPTERS`;
arbitrary import paths, shell fragments, or external commands in registry JSON are refused.
"""

from __future__ import annotations

import fnmatch
import json
import os
from typing import Any, Protocol

from .. import camulator as _camulator_mod
from .. import md15 as _md15_mod
from .. import sdyff as _sdyff_mod
from ..camulator import (
    CAMULATOR_CODE_REL_DIR,
    CAMULATOR_CYCLIC_FORCING_NAME,
    camulator_forcing_rel_path,
    camulator_ic_rel_path,
)
from ..cbottle import (
    CBOTTLE_BATCH_SIZE,
    CBOTTLE_DATE_ORDER,
    CBOTTLE_DEFAULT_CADENCE,
    CBOTTLE_FORCING_NAME,
    CBOTTLE_SEED_SCHEME,
)
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
from ..md15 import MD15_CODE_REL_DIR
from ..plan_types import AssetRef, RunSpec
from ..registry import SAMUDRACE_CM4_IC_SLUGS, ModelSpec, resolve_data_dir
from ..safe_pickle import compute_sha256sum_tree_digest, compute_tree_digest
from ..samudra import SAMUDRA_BYTES_PER_STEP, SAMUDRA_OSN_SOURCE
from ..sdyff import (
    SDYFF_CODE_REL_DIR,
    SDYFF_DATA_REL_DIR,
    SDYFF_FORCING_NAME,
    sdyff_plan_schedule,
)
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
    validate_safe_path,
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


def _load_derived_overrides(
    *,
    data_dir: str | None = None,
    lock_path: str | None = None,
) -> dict[str, str]:
    """Load `<data_dir>/derived-overrides.json` (`rel_path -> actual_sha256`) when present."""
    if not data_dir:
        return {}
    ovr_path = os.path.join(data_dir, "derived-overrides.json")
    if not os.path.isfile(ovr_path) or os.path.islink(ovr_path):
        return {}
    try:
        resolved_lock = resolve_lock_path(lock_path, dest_dir=data_dir, must_exist=False)
        with open(ovr_path, encoding="utf-8") as fh:
            doc = json.load(fh)
        if not isinstance(doc, dict) or not isinstance(doc.get("overrides"), list):
            return {}
        rec_lock = doc.get("lock_path")
        if (
            isinstance(rec_lock, str)
            and rec_lock
            and resolved_lock
            and os.path.normpath(rec_lock) != os.path.normpath(resolved_lock)
        ):
            return {}
        out: dict[str, str] = {}
        for item in doc["overrides"]:
            if not isinstance(item, dict):
                continue
            rel = item.get("rel_path")
            actual = item.get("actual_sha256")
            if isinstance(rel, str) and rel and isinstance(actual, str) and len(actual.strip()) == 64:
                out[rel] = actual.strip().lower()
        return out
    except Exception:
        return {}


def _asset_from_rel(
    rel_path: str,
    abs_path: str,
    role: str,
    lock_files: dict[str, dict[str, Any]],
    *,
    fallback_sha: str | None = None,
    derived_overrides: dict[str, str] | None = None,
    verify_bytes: bool = False,
) -> AssetRef:
    entry = lock_files.get(rel_path) or {}
    expected_sha = (derived_overrides or {}).get(rel_path) or entry.get("sha256") or fallback_sha
    cached = os.path.isfile(abs_path) and not os.path.islink(abs_path)
    if cached and verify_bytes:
        actual_sha = sha256_file(abs_path)
        try:
            size_val: int | None = os.path.getsize(abs_path)
        except OSError:
            size_val = None
        if expected_sha and actual_sha.lower() != str(expected_sha).lower():
            raise ValidationError(
                f"Pinned asset SHA-256 mismatch for {rel_path} ({abs_path}): "
                f"expected {expected_sha}, got {actual_sha}. "
                "Re-run `aimip fetch` with `--force-rebuild`.",
                field="forcing_set" if "forcing" in role else "model",
            )
        sha: str | None = actual_sha
    else:
        sha = str(expected_sha) if expected_sha else None
        raw_size = entry.get("size")
        size_val = int(raw_size) if isinstance(raw_size, int) else None
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
    """Bind custom forcing directories and their matching NetCDF files by SHA-256 content digest."""
    if not fdir:
        return []
    out: list[AssetRef] = []
    pat = file_pattern or "*.nc"
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
                    fsha: str | None = sha256_file(full)
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
        derived_overrides = _load_derived_overrides(data_dir=resolved_data or None, lock_path=lock_path)
        assets: list[AssetRef] = []
        seen: set[str] = set()
        selected_fs = validated.get("forcing_set")
        fs_dir = (getattr(selected_fs, "dir", None) or "").rstrip("/")
        fs_pat = getattr(selected_fs, "file_pattern", None) or "*.nc"
        fs_ocean_dir = (getattr(selected_fs, "ocean_dir", None) or "").rstrip("/")
        fs_ocean_pat = getattr(selected_fs, "ocean_file_pattern", None) or "*.nc"
        active_ics = set(getattr(selected_fs, "ic_files", ()) or model.ic_files) | set(
            getattr(selected_fs, "ocean_ic_files", ()) or model.ocean_ic_files
        )

        if model.required_files and resolved_data:
            for rel in model.required_files:
                abs_p = os.path.join(resolved_data, rel)
                role = (
                    "checkpoint"
                    if rel.startswith("models/") and (rel.endswith(".tar") or rel.endswith(".ckpt"))
                    else ("ic" if "/ic/" in rel or rel.endswith("/ic.nc") else "forcing")
                )
                fallback = model.checkpoint_sha256 if abs_p == model.checkpoint else None
                abs_parent = os.path.dirname(abs_p).rstrip("/")
                abs_name = os.path.basename(abs_p)
                is_selected = (
                    (abs_p == model.checkpoint and not validated.get("lora_checkpoint"))
                    or abs_p in active_ics
                    or (bool(fs_dir) and abs_parent == fs_dir and fnmatch.fnmatch(abs_name, fs_pat))
                    or (bool(fs_ocean_dir) and abs_parent == fs_ocean_dir and fnmatch.fnmatch(abs_name, fs_ocean_pat))
                )
                assets.append(
                    _asset_from_rel(
                        rel,
                        abs_p,
                        role,
                        lock_files,
                        fallback_sha=fallback,
                        derived_overrides=derived_overrides,
                        verify_bytes=is_selected,
                    )
                )
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
                        derived_overrides=derived_overrides,
                        verify_bytes=not bool(validated.get("lora_checkpoint")),
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
                            derived_overrides=derived_overrides,
                            verify_bytes=True,
                        )
                    )
                    seen.add(ic_p)
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
        emu_ic_sha: str | None = None
        emu_ic_custom = False
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
            ic_src: str | None = None
        elif model.family == "samudra":
            selected_name = scenario or "om4-jra55do-replay"
            ensemble_strategy = "single_om4_replay"
            ensemble_note = (
                "Deterministic OM4 replay (seed1 checkpoint; upstream seed1..seed5 are separate networks)."
            )
            ic_src = f"osn://m2lines-pubs/Samudra/OM4@{norm_start}"
        elif model.family == "md":
            selected_name = scenario or (
                model.forcing_sets[0].name if model.forcing_sets else "aimip-era5-monthly"
            )
            ensemble_strategy = "seeded_noise_members" if total_members > 1 else "single_member"
            ensemble_note = (
                f"{total_members} stochastic latent-diffusion member(s) (250 reverse DDPM steps/month, seed={seed})."
                if total_members > 1
                else f"Single latent-diffusion trajectory (250 reverse DDPM steps/month, seed={seed})."
            )
            if spec.ic is not None:
                ic_src = validate_visible_input_path(spec.ic, label="--ic")
                emu_ic_custom = True
                emu_ic_sha = sha256_file(ic_src) if os.path.isfile(ic_src) else None
            else:
                ymd = norm_start[:10]
                matched = [
                    p
                    for p in model.ic_files
                    if ymd in os.path.basename(p) or ymd.replace("-", "") in os.path.basename(p)
                ]
                resolved_data_md = resolve_data_dir(data_dir) or ""
                default_ic_rel = f"inputs/md-1.5/ic/ic_{ymd}.nc"
                ic_src = (
                    matched[0]
                    if matched
                    else (
                        os.path.join(resolved_data_md, default_ic_rel)
                        if resolved_data_md
                        else default_ic_rel
                    )
                )
                if os.path.isfile(ic_src) and not os.path.islink(ic_src):
                    emu_ic_sha = sha256_file(ic_src)
        elif model.family == "camulator":
            selected_name = scenario or CAMULATOR_CYCLIC_FORCING_NAME
            ensemble_strategy = "single_member"
            ensemble_note = (
                f"Deterministic CAMulator rollout (checkpoint.pt00069.pt, forcing={selected_name})."
            )
            ic_rel = camulator_ic_rel_path(norm_start)
            resolved_data = resolve_data_dir(data_dir) or ""
            lock_files = _load_lock_entries(data_dir=resolved_data or None)
            ic_src = os.path.join(resolved_data, ic_rel) if resolved_data else ic_rel
            ic_entry = lock_files.get(ic_rel)
            if isinstance(ic_entry, dict) and ic_entry.get("sha256"):
                emu_ic_sha = str(ic_entry["sha256"])
            elif os.path.isfile(ic_src) and not os.path.islink(ic_src):
                emu_ic_sha = sha256_file(ic_src)
        elif model.family == "cbottle":
            selected_name = scenario or CBOTTLE_FORCING_NAME
            ensemble_strategy = "independent_conditional_draws"
            ensemble_note = (
                f"{total_members} member(s) of independent conditional diffusion draws "
                f"(seed={seed}, batch_size={CBOTTLE_BATCH_SIZE}, cadence={CBOTTLE_DEFAULT_CADENCE}, "
                f"date_order={CBOTTLE_DATE_ORDER}, seed_scheme={CBOTTLE_SEED_SCHEME})."
            )
            ic_src = None
        elif model.family == "sdyff":
            selected_name = scenario or SDYFF_FORCING_NAME
            sdyff_sched = sdyff_plan_schedule(norm_start, steps)
            n_full_m = len(sdyff_sched["complete_months"])
            ensemble_strategy = "seeded_noise_members" if total_members > 1 else "single_member"
            ensemble_note = (
                f"{total_members} probabilistic Spherical DYffusion member(s) "
                f"(seed={seed}, coarsen_factor=4 -> {n_full_m} complete cycle month(s))."
            )
            start_m = int(norm_start[5:7])
            ic_rel_s = f"{SDYFF_DATA_REL_DIR}/2021{start_m:02d}0100.nc"
            resolved_data_s = resolve_data_dir(data_dir) or ""
            lock_files_s = _load_lock_entries(data_dir=resolved_data_s or None)
            ic_src = os.path.join(resolved_data_s, ic_rel_s) if resolved_data_s else ic_rel_s
            ic_entry_s = lock_files_s.get(ic_rel_s)
            if isinstance(ic_entry_s, dict) and ic_entry_s.get("sha256"):
                emu_ic_sha = str(ic_entry_s["sha256"])
            elif os.path.isfile(ic_src) and not os.path.islink(ic_src):
                emu_ic_sha = sha256_file(ic_src)
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
            ic_src = f"arco-era5@{norm_start}"

        selected_fs = next(
            (fs for fs in model.forcing_sets if fs.name == selected_name),
            model.forcing_sets[0],
        )
        missing_rels = model_has_missing_files(model, data_dir=data_dir)
        ic_checked = (
            bool(spec.ic is not None or (ic_src is not None and os.path.isfile(ic_src) and not os.path.islink(ic_src)))
            if model.family == "md"
            else True
        )
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
            "ic_source": ic_src,
            "ic_sha256": emu_ic_sha,
            "ic_is_custom": emu_ic_custom,
            "ic_checked": ic_checked,
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
        if model.family == "e2s-dlesym":
            basis = "DLESyM 1x A100-80GB (~40 s / model year / member)"
        elif model.family == "samudra":
            basis = "Samudra-OM4 1x A100-80GB (~18 s / simulated year)"
        elif model.family == "md":
            basis = "MD-1.5 1x A100-80GB (~20 min for 46.25y x 50 members batched; Hall & Molina, 2026)"
        elif model.family == "camulator":
            basis = "CAMulator 1x A100-80GB (~480 simulated years / day, Chapman et al., 2025)"
        elif model.family == "cbottle":
            basis = (
                f"cBottle-3d 1x A100-80GB (~150 s / sampled year at {CBOTTLE_DEFAULT_CADENCE}, "
                f"batch_size={CBOTTLE_BATCH_SIZE})"
            )
        elif model.family == "sdyff":
            basis = "Spherical DYffusion 1x A100-80GB (~180 s / simulated cycle year)"
        else:
            basis = "ArchesClimate-SSP 1x A100-80GB (~2 s / simulated month)"
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
        derived_overrides = _load_derived_overrides(data_dir=resolved_data or None, lock_path=lock_path)
        assets: list[AssetRef] = []
        selected_fs = validated.get("forcing_set")
        selected_scen = getattr(selected_fs, "name", None)
        scen_tag = f"r1i1p1f1_{selected_scen}_interpolation.memmap/" if selected_scen else None

        for rel in model.required_files:
            if model.family == "camulator" and rel.startswith("inputs/camulator/"):
                continue
            if model.family == "md" and rel.startswith("inputs/md-1.5/ic/"):
                if validated.get("ic_is_custom"):
                    continue
                ymd = str(validated.get("norm_start") or "")[:10]
                if ymd and ymd not in os.path.basename(rel) and ymd.replace("-", "") not in os.path.basename(rel):
                    continue
            abs_p = os.path.join(resolved_data, rel) if resolved_data else rel
            if "/ic/" in rel:
                role = "ic"
            elif "memmap" in rel or rel.startswith(("inputs/", "raw/")):
                role = "forcing"
            else:
                role = "package"
            fallback = model.checkpoint_sha256 if abs_p == model.checkpoint else None
            is_selected = role in ("package", "ic") or (scen_tag is not None and scen_tag in rel)
            assets.append(
                _asset_from_rel(
                    rel,
                    abs_p,
                    role,
                    lock_files,
                    fallback_sha=fallback,
                    derived_overrides=derived_overrides,
                    verify_bytes=is_selected,
                )
            )

        if model.family == "arches":
            override_code = (os.environ.get("AIMIP_ARCHES_CODE") or "").strip()
            if override_code:
                safe_ar_ovr = validate_safe_path(override_code, label="AIMIP_ARCHES_CODE")
                tree_sha, _ = compute_arches_tree_digest(safe_ar_ovr)
                assets.append(
                    AssetRef(
                        rel_path=ARCHES_CODE_REL_DIR,
                        path=safe_ar_ovr,
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
        elif model.family == "samudra":
            out_cand = validated.get("output_dir")
            staged_win = os.path.join(str(out_cand), "window") if out_cand else ""
            if (
                staged_win
                and os.path.isdir(staged_win)
                and not os.path.islink(staged_win)
                and os.path.isfile(os.path.join(staged_win, "window.json"))
            ):
                tree_sha, _, total_bytes = compute_tree_digest(
                    staged_win, label="staged Samudra OM4 window"
                )
                assets.append(
                    AssetRef(
                        rel_path="window",
                        path=staged_win,
                        role="staged_ic",
                        sha256=tree_sha,
                        size_bytes=total_bytes,
                        cached=True,
                        prospective=False,
                    )
                )
            else:
                src_root = (
                    os.environ.get("AIMIP_SAMUDRA_SOURCE") or SAMUDRA_OSN_SOURCE
                ).strip().rstrip("/")
                est_bytes = (int(validated["steps"]) + 2) * SAMUDRA_BYTES_PER_STEP
                assets.append(
                    AssetRef(
                        rel_path="window",
                        path=f"{src_root}/OM4@{validated['norm_start']}+{validated['steps']}steps",
                        role="staged_ic",
                        sha256=None,
                        size_bytes=est_bytes,
                        cached=False,
                        prospective=True,
                    )
                )
        elif model.family == "md":
            if validated.get("ic_is_custom") and validated.get("ic_source"):
                custom_ic_p = str(validated["ic_source"])
                assets.append(
                    AssetRef(
                        rel_path=f"custom_ic:{custom_ic_p}",
                        path=custom_ic_p,
                        role="custom_ic",
                        sha256=validated.get("ic_sha256"),
                        size_bytes=os.path.getsize(custom_ic_p) if os.path.isfile(custom_ic_p) else None,
                        cached=os.path.isfile(custom_ic_p) and not os.path.islink(custom_ic_p),
                    )
                )
            elif validated.get("ic_source"):
                staged_md_ic = str(validated["ic_source"])
                already_listed = any(a.path == staged_md_ic for a in assets)
                if not already_listed and os.path.isfile(staged_md_ic) and not os.path.islink(staged_md_ic):
                    rel_md_ic = (
                        os.path.relpath(staged_md_ic, resolved_data).replace(os.sep, "/")
                        if resolved_data and staged_md_ic.startswith(resolved_data.rstrip("/") + "/")
                        else os.path.basename(staged_md_ic)
                    )
                    assets.append(
                        AssetRef(
                            rel_path=rel_md_ic,
                            path=staged_md_ic,
                            role="ic",
                            sha256=validated.get("ic_sha256") or sha256_file(staged_md_ic),
                            size_bytes=os.path.getsize(staged_md_ic),
                            cached=True,
                        )
                    )
            override_md_code = (os.environ.get("AIMIP_MD15_CODE") or "").strip()
            if override_md_code:
                safe_md_ovr = validate_safe_path(override_md_code, label="AIMIP_MD15_CODE")
                md_tree_sha, _ = compute_sha256sum_tree_digest(safe_md_ovr, label="AIMIP_MD15_CODE")
                assets.append(
                    AssetRef(
                        rel_path=MD15_CODE_REL_DIR,
                        path=safe_md_ovr,
                        role="source_tree",
                        sha256=md_tree_sha,
                        size_bytes=None,
                        cached=True,
                    )
                )
            else:
                md_code_abs = (
                    os.path.join(resolved_data, MD15_CODE_REL_DIR)
                    if resolved_data
                    else MD15_CODE_REL_DIR
                )
                md_code_cached = os.path.isdir(md_code_abs) and not os.path.islink(md_code_abs)
                assets.append(
                    AssetRef(
                        rel_path=MD15_CODE_REL_DIR,
                        path=md_code_abs,
                        role="source_tree",
                        sha256=_md15_mod.MD15_TREE_SHA256,
                        size_bytes=_md15_mod.MD15_GITHUB_ARCHIVE_SIZE,
                        cached=md_code_cached,
                    )
                )
        elif model.family == "camulator":
            norm_start_s = str(validated.get("norm_start") or "")
            eff_scen = str(selected_scen or CAMULATOR_CYCLIC_FORCING_NAME)
            for cam_rel, cam_role in (
                (camulator_ic_rel_path(norm_start_s), "ic"),
                (camulator_forcing_rel_path(eff_scen), "forcing"),
            ):
                cam_abs = os.path.join(resolved_data, cam_rel) if resolved_data else cam_rel
                assets.append(
                    _asset_from_rel(
                        cam_rel,
                        cam_abs,
                        cam_role,
                        lock_files,
                        derived_overrides=derived_overrides,
                        verify_bytes=True,
                    )
                )
            override_cam_code = (os.environ.get("AIMIP_CAMULATOR_CODE") or "").strip()
            if override_cam_code:
                safe_cam_ovr = validate_safe_path(override_cam_code, label="AIMIP_CAMULATOR_CODE")
                cam_tree_sha, _ = compute_sha256sum_tree_digest(
                    safe_cam_ovr, label="AIMIP_CAMULATOR_CODE"
                )
                assets.append(
                    AssetRef(
                        rel_path=CAMULATOR_CODE_REL_DIR,
                        path=safe_cam_ovr,
                        role="source_tree",
                        sha256=cam_tree_sha,
                        size_bytes=None,
                        cached=True,
                    )
                )
            else:
                cam_code_abs = (
                    os.path.join(resolved_data, CAMULATOR_CODE_REL_DIR)
                    if resolved_data
                    else CAMULATOR_CODE_REL_DIR
                )
                cam_code_cached = os.path.isdir(cam_code_abs) and not os.path.islink(cam_code_abs)
                assets.append(
                    AssetRef(
                        rel_path=CAMULATOR_CODE_REL_DIR,
                        path=cam_code_abs,
                        role="source_tree",
                        sha256=_camulator_mod.CAMULATOR_TREE_SHA256,
                        size_bytes=_camulator_mod.CAMULATOR_GITHUB_ARCHIVE_SIZE,
                        cached=cam_code_cached,
                    )
                )
        elif model.family == "cbottle":
            # All 4 cBottle-3d assets (3 expert checkpoints + amip_midmonth_sst.nc) are already in `model.required_files`.
            pass
        elif model.family == "sdyff":
            override_sdyff_code = (os.environ.get("AIMIP_SDYFF_CODE") or "").strip()
            if override_sdyff_code:
                safe_sd_ovr = validate_safe_path(override_sdyff_code, label="AIMIP_SDYFF_CODE")
                sd_tree_sha, _ = compute_sha256sum_tree_digest(
                    safe_sd_ovr, label="AIMIP_SDYFF_CODE"
                )
                assets.append(
                    AssetRef(
                        rel_path=SDYFF_CODE_REL_DIR,
                        path=safe_sd_ovr,
                        role="source_tree",
                        sha256=sd_tree_sha,
                        size_bytes=None,
                        cached=True,
                    )
                )
            else:
                sd_code_abs = (
                    os.path.join(resolved_data, SDYFF_CODE_REL_DIR)
                    if resolved_data
                    else SDYFF_CODE_REL_DIR
                )
                sd_code_cached = os.path.isdir(sd_code_abs) and not os.path.islink(sd_code_abs)
                assets.append(
                    AssetRef(
                        rel_path=SDYFF_CODE_REL_DIR,
                        path=sd_code_abs,
                        role="source_tree",
                        sha256=_sdyff_mod.SDYFF_TREE_SHA256,
                        size_bytes=_sdyff_mod.SDYFF_GITHUB_ARCHIVE_SIZE,
                        cached=sd_code_cached,
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
        has_monthly = os.path.isfile(os.path.join(out_dir, "monthly_mean_predictions.nc")) or any(
            os.path.isfile(os.path.join(out_dir, realm, "monthly_mean_predictions.nc"))
            for realm in ("atmosphere", "ocean")
        )
        return {
            "status": get_run_status(out_dir, sacct_json_path=sacct_json_path),
            "summary": summarize_run(out_dir) if has_monthly else None,
        }


INSTALLED_ADAPTERS: dict[str, FamilyAdapter] = {
    "fme": FmeFamilyAdapter(),
    "e2s-dlesym": EmuFamilyAdapter("e2s-dlesym"),
    "arches": EmuFamilyAdapter("arches"),
    "samudra": EmuFamilyAdapter("samudra"),
    "md": EmuFamilyAdapter("md"),
    "camulator": EmuFamilyAdapter("camulator"),
    "cbottle": EmuFamilyAdapter("cbottle"),
    "sdyff": EmuFamilyAdapter("sdyff"),
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
