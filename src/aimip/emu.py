"""Non-fme model families: DLESyM (Earth2Studio), ArchesClimate-SSP and ArchesWeather / ArchesWeatherGen (geoarches).

All run in a dedicated Python environment (`/opt/aimip/venv-emu` inside the container, or `<prefix>/venv-emu` in
native mode: torch 2.14 cu130, earth2studio @ 8fa7acb, physicsnemo @ 1dacf8d, earth2grid @ 11dcf1b, plus
ArchesClimate-SSP's and geoarches' Python dependencies; ArchesClimate-SSP @ 06ce82dc itself is downloaded into
`$AIMIP_DATA/code/ArchesClimate-SSP` on first use, geoarches @ 4d0c77a1 is unpacked next to the environment at install
time, see `geoarches_src.py`). This module runs in the main environment and only uses the
standard library plus the CLI helpers:

- `prepare_emu_run`: validate the request, fetch the model on first use (same lock/fetch path as fme models), stage
  the ERA5 initial state for DLESyM (ARCO-ERA5, login node), and write `config.yaml` (JSON), `job.sbatch` (the same
  `aimip exec-fme <dir>` line every run uses) and `aimip-manifest.json`.
- `exec_emu`: at job time re-hash the checkpoint, the model package files and the initial state against the manifest,
  then run `python -I -m aimip.emu_job run <dir>` in the emulator environment and record `finished`, `rc`,
  `wall_seconds`.

Model facts:
- DLESyM v1.0 / v1.1: HF nvidia/dlesym-v1-era5 @ b88155c (`checkpoints/v1.0`, `checkpoints/v1.1`); v1.1 samples
  conditional-layer-norm noise every step, so members differ by seed. DLESyMv0-ISCCP: HF nvidia/dlesym-v0-isccp-era5
  @ 924b2d6 (the climate checkpoint Earth2Studio documents for 100-1000-year rollouts; `use_ttr=True` converts the
  initial ERA5 OLR to the model's ISCCP-like OLR). The V1 checkpoints were built for S2S forecasts. Atmosphere outputs every
  6 h, ocean every 48 h, on the 0.25 degree lat-lon grid; no forcing (insolation is computed).
- dlesym-aimip: the v0-ISCCP checkpoints with the SST prescribed from the AIMIP forcing (Zenodo 17065758, linear in time
  between its month-start stamps), as the authors' AIMIP runs (DLESyM_aimip-2026); emu_job.PrescribedSST.
- ArchesClimate-SSP deterministic: HF gclyne/ArchesClimate @ 06a914d, data gclyne/ArchesClimate-data @ b9f5458.
  Monthly steps on 144x144 (1.25 x 2.5 degree), IPSL-CM6A-LR r1i1p1f1 scenario forcing. The authors' rollout starts from
  dataset item 0 (state 2015-02, previous state 2015-01), so the first output is March 2015.
- ArchesWeather-Mx4 / ArchesWeatherGen, AIMIP versions: Zenodo 22795233. Daily steps on the 1-degree grid from the ERA5
  daily-mean state of 1978-09-30 (previous state 1978-09-29), monthly ERA5 SST / sea-ice forcing through 2024; the
  rollout follows arXiv:2605.29976 section 2.3 (see `emu_job.run_geoarches`). Every run writes `restart.nc`;
  `--ic <run-dir>` continues it, and the continuation gives the same result as one longer run (`check_aw_restart`).
"""
from __future__ import annotations

import datetime
import io
import json
import os
import shutil
import subprocess
import sys
import time
from collections.abc import Mapping, Sequence
from typing import Any

from . import __version__
from .camulator import (
    CAMULATOR_BRANCH_FORCING_NAME,
    CAMULATOR_BRANCH_WINDOW,
    CAMULATOR_CONFIG_REL_PATH,
    CAMULATOR_CYCLIC_FORCING_NAME,
    CAMULATOR_CYCLIC_WINDOW,
    CAMULATOR_MEAN_REL_PATH,
    CAMULATOR_METADATA_REL_PATH,
    CAMULATOR_PUBLISHED_STARTS,
    CAMULATOR_SECONDS_PER_YEAR,
    CAMULATOR_STATICS_B_REL_PATH,
    CAMULATOR_STATICS_BE21_REL_PATH,
    CAMULATOR_STATICS_FE21_REL_PATH,
    CAMULATOR_STD_REL_PATH,
    camulator_python,
    resolve_camulator_code_dir,
    verify_camulator_code_dir,
)
from .cbottle import (
    CBOTTLE_BATCH_SIZE,
    CBOTTLE_CHECKPOINT_REL_PATHS,
    CBOTTLE_DATE_ORDER,
    CBOTTLE_DEFAULT_CADENCE,
    CBOTTLE_SECONDS_PER_YEAR,
    CBOTTLE_SEED_SCHEME,
    CBOTTLE_SST_REL_PATH,
    cbottle_batch_seed,
    cbottle_sample_timestamps,
)
from .errors import EXIT_UNSTABLE, ConfigError, MissingAssetsError, ValidationError
from .emu_catalog import (
    AW_FORCING,
    AW_IC,
    AW_LAST_DAY,
    AW_RESTART,
    AW_RESTART_FORMAT,
    AW_START,
    AIMIP_SST_REL,
    DLESYM_AIMIP_FORCING_END,
    DLESYM_AIMIP_WINDOW,
    DLESYM_IC_WINDOW,
    SAMUDRA_WINDOW,
    aw_package_digest,
)
from .lora import sha256_file
from .md15 import (
    MD15_CODE_REL_DIR,
    MD15_CONFIG_REL_PATH,
    MD15_FORCING_WINDOW,
    MD15_PUBLISHED_STARTS,
    MD15_STATS_REL_PATH,
    MD15_WEIGHTS_REL_PATH,
    md15_months_between,
    md15_step_months,
    md15_variant,
    md15_wall_seconds_estimate,
    resolve_md15_code_dir,
    verify_md15_code_dir,
)
from .registry import ModelSpec, resolve_data_dir, resolve_playground_dir
from .safe_pickle import compute_sha256sum_tree_digest, compute_tree_digest
from .samudra import SAMUDRA_OSN_SOURCE, SAMUDRA_SECONDS_PER_YEAR
from .sdyff import (
    SDYFF_COARSEN_FACTOR,
    SDYFF_DATA_REL_DIR,
    SDYFF_FORECASTER_CKPT_REL_PATH,
    SDYFF_INTERPOLATOR_CFG_REL_PATH,
    SDYFF_INTERPOLATOR_CKPT_REL_PATH,
    SDYFF_SECONDS_PER_YEAR,
    SDYFF_ZENODO_FILES,
    resolve_sdyff_code_dir,
    sdyff_plan_schedule,
    sdyff_python,
    verify_sdyff_code_dir,
)
from .site_config import SiteConfig, load_site_config
from .time_utils import format_cftime, parse_cftime
from .validators import (
    makedirs_no_group_other_write,
    strip_group_other_write,
    validate_forcing_set_name,
    validate_int_range,
    validate_iso_timestamp,
    validate_run_path,
    validate_safe_path,
    validate_seed,
    validate_slurm_job_id,
    validate_slurm_time,
    validate_visible_input_path,
)

EMU_FAMILIES: tuple[str, ...] = (
    "e2s-dlesym",
    "arches",
    "geoarches",
    "samudra",
    "md",
    "camulator",
    "cbottle",
    "sdyff",
)
EMU_JOB_MODULE = "aimip.emu_job"
DEFAULT_EMU_PYTHON = "/opt/aimip/venv-emu/bin/python"

# ARCO-ERA5 (public, anonymous GCS) supplies the DLESyM initial state. Inputs reach back 48 h and the v1.1 OLR input
# (ttr03) is a trailing 3-hour sum; with a day of margin the first start is 1940-01-04T00 (emu_catalog.DLESYM_IC_WINDOW).
DLESYM_MODELS: dict[str, dict[str, Any]] = {
    "dlesym-v1.0": {"package": "models/dlesym-v1", "version": "v1.0", "flavor": "v1", "members": False},
    "dlesym-v1.1": {"package": "models/dlesym-v1", "version": "v1.1", "flavor": "v1", "members": True},
    "dlesym-v0-isccp": {"package": "models/dlesym-v0-isccp", "version": "v0", "flavor": "v0", "members": False},
    # AIMIP version: the same checkpoints with the SST prescribed from the AIMIP forcing (emu_job.PrescribedSST)
    "dlesym-aimip": {"package": "models/dlesym-v0-isccp", "version": "v0", "flavor": "v0", "members": False,
                     "prescribed_sst": True},
}
DLESYM_HOURS_PER_ITERATION = 96   # one coupled step of the DLESyM checkpoints (16 atmosphere steps of 6 h)
ARCHES_START = "2015-01-01T00:00:00"
ARCHES_FIRST_OUTPUT_MONTH = (2015, 3)       # item 0 = state 2015-02 with previous state 2015-01 -> first output 2015-03
ARCHES_MAX_STEPS = 1029                     # 1032 monthly states (2015-01..2100-12) minus previous state and offsets
ARCHES_SCENARIOS: tuple[str, ...] = ("ssp434", "ssp585")
ARCHES_DOMAIN = {"ssp434": "val", "ssp585": "test"}   # the checkpoint config's val_filter / test_filter

# Wall-time estimate (measured on one A100 GPU), padded 3x with a 45-minute floor.
DLESYM_SECONDS_PER_YEAR = 40.0
ARCHES_SECONDS_PER_MONTH = 2.0
AW_SECONDS_PER_DAY = {"archesweather": 1.1, "archesweathergen": 4.6}   # per member (A100-80GB, measured)
EMU_FIXED_SECONDS = 900
EMU_SAFETY = 3.0
EMU_MIN_WALLTIME = 2700


def emu_python() -> str:
    """Return the absolute executable path of the DLESyM / ArchesClimate Python environment."""
    raw = os.environ.get("AIMIP_EMU_PYTHON")
    if not raw:
        home = os.environ.get("AIMIP_HOME") or os.environ.get("AIMIP_PLAYGROUND")
        cand = os.path.join(home.rstrip("/"), "venv-emu", "bin", "python") if home else ""
        if cand and (os.path.exists(cand) or not os.path.exists(DEFAULT_EMU_PYTHON)):
            raw = cand
        else:
            raw = DEFAULT_EMU_PYTHON
    path = validate_safe_path(raw, label="emulator python")
    if not os.path.isabs(path) or not os.access(path, os.X_OK):
        raise ConfigError(
            f"The DLESyM/ArchesClimate environment is missing ({path}). "
            "For native installs re-run `install/install-native.sh --with-emu` (or set $AIMIP_EMU_PYTHON); "
            "for containers rebuild the image with `install/install-container.sh`."
        )
    return path


def _runtime_python_for_family(family: str) -> str:
    """Resolve the isolated Python executable for *family* (`venv-camulator`, `venv-sdyff`, vs `venv-emu`)."""
    if family == "camulator":
        return camulator_python()
    if family == "sdyff":
        return sdyff_python()
    return emu_python()


def _now_utc() -> str:
    return datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _write_json_atomic(target: str, payload: dict[str, Any]) -> None:
    tmp = f"{target}.tmp-{os.getpid()}"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)
        fh.write("\n")
    strip_group_other_write(tmp)
    os.replace(tmp, target)


def _format_hms(total_seconds: float) -> str:
    s = max(0, int(total_seconds))
    h, rem = divmod(s, 3600)
    m, sec = divmod(rem, 60)
    return f"{h:02d}:{m:02d}:{sec:02d}"


def _parse_hms(value: str) -> int:
    parts = [int(p) for p in value.split("-")[-1].split(":")]
    days = int(value.split("-")[0]) if "-" in value else 0
    while len(parts) < 3:
        parts.insert(0, 0)
    return days * 86400 + parts[0] * 3600 + parts[1] * 60 + parts[2]


def _work_seconds(family: str, steps: int, members: int, model: str | None) -> float:
    if family == "e2s-dlesym":
        return steps * 6 / (365.25 * 24) * DLESYM_SECONDS_PER_YEAR * members
    if family == "geoarches":
        return steps * AW_SECONDS_PER_DAY.get(str(model), max(AW_SECONDS_PER_DAY.values())) * members
    if family == "samudra":
        return steps * 120 / (365.25 * 24) * SAMUDRA_SECONDS_PER_YEAR * members
    if family == "md":
        return md15_wall_seconds_estimate(steps, members)
    if family == "camulator":
        return steps * 6 / (365.0 * 24) * CAMULATOR_SECONDS_PER_YEAR * members
    if family == "cbottle":
        return steps / 365.25 * CBOTTLE_SECONDS_PER_YEAR * members
    if family == "sdyff":
        return steps * 6 / (365.0 * 24) * SDYFF_SECONDS_PER_YEAR * members
    return steps * ARCHES_SECONDS_PER_MONTH * members


def _walltime(family: str, steps: int, members: int, max_time: str, model: str | None = None) -> tuple[str, str, bool]:
    work = _work_seconds(family, steps, members, model)
    want = max(EMU_MIN_WALLTIME, int(EMU_SAFETY * (EMU_FIXED_SECONDS + work)))
    cap = _parse_hms(max_time) if max_time else 12 * 3600
    return validate_slurm_time(_format_hms(min(want, cap))), _format_hms(want), want > cap


def _gpu_hours(family: str, steps: int, members: int, model: str | None = None) -> float:
    """Expected GPU time (no padding): rollout plus about one minute of loading."""
    return (60 + _work_seconds(family, steps, members, model)) / 3600


# ---------------------------------------------------------------- pinned files per model

def model_package_files(model: ModelSpec, data_dir: str) -> dict[str, str]:
    """Absolute paths of the files the job loads and hashes (checkpoint, package constants, statistics)."""
    out: dict[str, str] = {}
    for rel in model.required_files:
        if model.family == "arches" and rel.startswith("inputs/") and rel.endswith(".memmap"):
            continue      # scenario arrays (GBs): recorded by size, like fme forcing files; side files are hashed
        if model.family == "camulator" and rel.startswith("inputs/camulator/"):
            continue      # selected IC `.pth` and forcing `.nc` are hashed via `ensure_camulator_selected_assets`
        out[rel] = os.path.join(data_dir, rel)
    return out


def _scenario_files(model: ModelSpec, data_dir: str, scenario: str) -> list[dict[str, Any]]:
    tag = f"r1i1p1f1_{scenario}_interpolation.memmap/"
    rows = []
    for rel in model.required_files:
        if rel.startswith("inputs/") and tag in rel and rel.endswith(".memmap"):
            p = os.path.join(data_dir, rel)
            ok_file = os.path.isfile(p) and not os.path.islink(p)
            rows.append(
                {
                    "rel_path": rel,
                    "path": p,
                    "size": os.path.getsize(p) if ok_file else None,
                    "sha256": sha256_file(p) if ok_file else None,
                }
            )
    return rows


def _check_against_lock(name: str, pkg: dict[str, str], pkg_sha: dict[str, str], data_root: str) -> None:
    """Every model file (checkpoints, configs, constants, statistics) must match the active aimip.lock."""
    from .setup_data import load_lock_file, resolve_lock_path

    lock = load_lock_file(resolve_lock_path(None, dest_dir=data_root))["files"]
    bad = []
    for rel, path in pkg.items():
        entry = lock.get(rel)
        if not isinstance(entry, dict) or str(entry.get("sha256", "")).lower() != pkg_sha[path].lower():
            bad.append(rel)
    if bad:
        raise ValidationError(
            f"{name}: {len(bad)} model file(s) do not match aimip.lock ({', '.join(bad[:3])}). "
            f"Ask the install owner to re-fetch: delete them and run `aimip fetch -m {name}`."
        )


def _config_sha(path: str) -> str:
    return sha256_file(path)


# ---------------------------------------------------------------- prepare

def precheck_emu_request(
    model: ModelSpec,
    *,
    start_iso: str,
    steps: int,
    members: int | None,
    forcing_set_name: str | None,
    ic_path: str | None,
    ocean_ic_path: str | None = None,
    lora_path: str | None = None,
) -> dict[str, Any]:
    """Validate a run request for a DLESyM / ArchesClimate model WITHOUT touching disk (runs before any download)."""
    norm_start = validate_iso_timestamp(start_iso, label="--start")
    n_steps = validate_int_range(steps, "--steps", min_value=1, max_value=10_000_000)
    n_members = validate_int_range(1 if members is None else members, "--members", min_value=1, max_value=64)
    for flag, value in (("--ic", ic_path), ("--ocean-ic", ocean_ic_path), ("--lora", lora_path)):
        if value is not None:
            if flag == "--ic" and model.family in ("geoarches", "md"):
                continue
            if model.family == "e2s-dlesym":
                reason = (
                    "DLESyM starts from ERA5 at --start (fetched from ARCO-ERA5) and cannot be continued yet; "
                    "run the whole period in one job (about 40 s of GPU time per simulated year)."
                )
            elif model.family == "geoarches":
                reason = "ArchesWeather takes --ic only to continue a previous run (--ic <run-dir> or its restart.nc)."
            elif model.family == "samudra":
                reason = (
                    "samudra-om4 replays two 5-day OM4 ocean states ending at --start plus OM4 surface boundary "
                    "fluxes from s3://m2lines-pubs/Samudra/OM4 and does not support custom --ic or --lora."
                )
            elif model.family == "md":
                reason = f"{model.name} is an atmosphere-only latent diffusion model and does not support --ocean-ic or --lora."
            elif model.family == "camulator":
                reason = (
                    "camulator starts only from its 69 published noleap initial-condition tensors "
                    "(1980-01-01, Jan 1 & Jul 1 of 1981-2014; training Zarr archive is not public) "
                    "and does not support custom --ic, --ocean-ic, or --lora."
                )
            elif model.family == "cbottle":
                reason = (
                    "cbottle-3d is a conditional atmospheric state generator without state memory "
                    "(conditioned only on date/time and monthly SST tosbcs) and does not accept "
                    "--ic, --ocean-ic, or --lora."
                )
            elif model.family == "sdyff":
                reason = (
                    "spherical-dyffusion initializes from its repeating 365-day FV3GFS reference cycle "
                    "(Zenodo 10791087) and does not support custom --ic, --ocean-ic, or --lora."
                )
            else:
                reason = "ArchesClimate-SSP always starts from the scenario's 2015 state."
            raise ConfigError(
                f"{flag} is not supported for {model.name!r}: {reason}",
                field=flag.lstrip("-").replace("-", "_"),
            )
    if model.family == "e2s-dlesym":
        info = DLESYM_MODELS.get(model.name)
        if info is None:
            raise ConfigError(f"Unknown DLESyM model {model.name!r}.")
        if norm_start[11:] not in ("00:00:00", "06:00:00", "12:00:00", "18:00:00"):
            raise ConfigError(f"--start for {model.name} must be at 00, 06, 12 or 18 UTC, got {norm_start}.")
        if info.get("prescribed_sst"):
            if not (DLESYM_AIMIP_WINDOW[0] <= norm_start <= DLESYM_AIMIP_WINDOW[1]):
                raise ConfigError(
                    f"--start {norm_start} is outside the AIMIP SST forcing this model needs ({DLESYM_AIMIP_WINDOW[0]} "
                    f".. {DLESYM_AIMIP_WINDOW[1]}; the authors' members start 1978-10-03 .. 1978-10-07 at 00 UTC)."
                )
            start_dt = datetime.datetime.fromisoformat(norm_start)
            last = datetime.datetime.fromisoformat(DLESYM_AIMIP_FORCING_END)
            max_steps = int((last - start_dt).total_seconds() // (3600 * DLESYM_HOURS_PER_ITERATION)) * (
                DLESYM_HOURS_PER_ITERATION // 6)
            if n_steps > max_steps:
                raise ConfigError(
                    f"--steps {n_steps} runs past the AIMIP SST forcing: at most {max_steps} 6-h steps from {norm_start} "
                    f"(whole {DLESYM_HOURS_PER_ITERATION}-h model steps ending by {DLESYM_AIMIP_FORCING_END})."
                )
        elif not (DLESYM_IC_WINDOW[0] <= norm_start <= DLESYM_IC_WINDOW[1]):
            raise ConfigError(
                f"--start {norm_start} is outside the ERA5 initial states this model can use "
                f"({DLESYM_IC_WINDOW[0]} .. {DLESYM_IC_WINDOW[1]}, ARCO-ERA5)."
            )
        if n_members > 1 and not info["members"]:
            raise ConfigError(
                f"{model.name} is deterministic: --members must be 1 (dlesym-v1.1 samples noise and supports members)."
            )
        if forcing_set_name is not None and forcing_set_name != model.forcing_sets[0].name:
            raise ConfigError(
                f"{model.name} has one forcing set only ({model.forcing_sets[0].name}: the AIMIP SST)."
                if info.get("prescribed_sst") else
                f"{model.name} has no forcing sets to choose from (it is not forced; insolation is computed).")
        scenario = model.forcing_sets[0].name
        start_dt = datetime.datetime.fromisoformat(norm_start)
        end_iso = (start_dt + datetime.timedelta(hours=6 * n_steps)).strftime("%Y-%m-%dT%H:%M:%S")
    elif model.family == "arches":
        if norm_start != ARCHES_START:
            raise ConfigError(
                f"{model.name} starts from the scenario's 2015 state only: use --start {ARCHES_START} "
                f"(the first output month is {ARCHES_FIRST_OUTPUT_MONTH[0]}-{ARCHES_FIRST_OUTPUT_MONTH[1]:02d})."
            )
        if n_members > 1:
            raise ConfigError(f"{model.name} (deterministic checkpoint) supports one member only.")
        if n_steps > ARCHES_MAX_STEPS:
            raise ConfigError(
                f"--steps {n_steps} exceeds the scenario forcing: at most {ARCHES_MAX_STEPS} monthly steps "
                "(March 2015 .. November 2100)."
            )
        names = [fs.name for fs in model.forcing_sets]
        if forcing_set_name is None:
            scenario = names[0]
        else:
            scenario = validate_forcing_set_name(forcing_set_name)
            if scenario not in names:
                raise ConfigError(f"Unknown --forcing-set {scenario!r} for {model.name}: choose one of {', '.join(names)}.")
        y, m = ARCHES_FIRST_OUTPUT_MONTH
        k = m - 1 + n_steps - 1
        end_iso = f"{y + k // 12:04d}-{k % 12 + 1:02d}-01T00:00:00"
    elif model.family == "geoarches":
        if ic_path is None and norm_start != AW_START:
            raise ConfigError(
                f"{model.name} starts from the ERA5 daily-mean state of 1978-09-30 only: use --start {AW_START} "
                "(the first output day is 1978-10-01), or continue a previous run with --ic <run-dir> --start <its end>."
            )
        if ic_path is not None and not (AW_START < norm_start < AW_LAST_DAY and norm_start.endswith("T00:00:00")):
            raise ConfigError(
                f"A {model.name} continuation starts at 00 UTC on the last day of the run it continues, after "
                f"{AW_START[:10]} and before {AW_LAST_DAY[:10]}; got --start {norm_start}."
            )
        max_steps = (datetime.datetime.fromisoformat(AW_LAST_DAY) - datetime.datetime.fromisoformat(norm_start)).days
        if n_steps > max_steps:
            raise ConfigError(
                f"--steps {n_steps} exceeds the forcing: at most {max_steps} daily steps from {norm_start[:10]} "
                f"(the last output day is {AW_LAST_DAY[:10]})."
            )
        if n_members > 1 and not model.stochastic:
            raise ConfigError(
                f"{model.name} is deterministic: --members must be 1 (archesweathergen samples noise and supports "
                "members)."
            )
        names = [fs.name for fs in model.forcing_sets]
        scenario = names[0]
        if forcing_set_name is not None and validate_forcing_set_name(forcing_set_name) not in names:
            raise ConfigError(f"Unknown --forcing-set {forcing_set_name!r} for {model.name}: choose one of "
                              f"{', '.join(names)}.")
        start_dt = datetime.datetime.fromisoformat(norm_start)
        end_iso = (start_dt + datetime.timedelta(days=n_steps)).strftime("%Y-%m-%dT%H:%M:%S")
    elif model.family == "samudra":
        if norm_start[11:] != "12:00:00":
            raise ConfigError(
                f"--start for {model.name} must be on the OM4 5-day noon grid (YYYY-MM-DDT12:00:00, "
                f"e.g. 2014-10-10T12:00:00), got {norm_start}.",
                field="start",
            )
        if n_steps < 2 or n_steps % 2 != 0:
            raise ConfigError(
                f"--steps for {model.name} must be an even integer >= 2 (each model call predicts two 5-day states), "
                f"got {n_steps}.",
                field="steps",
            )
        if n_members > 1:
            raise ConfigError(
                f"{model.name} is deterministic (--members must be 1): upstream seed1..seed5 are five separately "
                "trained networks, not stochastic ensemble members.",
                field="members",
                allowed=[1],
            )
        names = [fs.name for fs in model.forcing_sets]
        if forcing_set_name is None:
            scenario = names[0]
        else:
            scenario = validate_forcing_set_name(forcing_set_name)
            if scenario not in names:
                raise ConfigError(
                    f"Unknown --forcing-set {scenario!r} for {model.name}: {model.name} replays OM4 surface fluxes "
                    f"({', '.join(names)}) and accepts no custom forcing.",
                    field="forcing_set",
                    allowed=names,
                )
        start_dt = parse_cftime(norm_start, calendar="julian")
        win_first = parse_cftime(SAMUDRA_WINDOW[0], calendar="julian")
        win_last = parse_cftime(SAMUDRA_WINDOW[1], calendar="julian")
        if start_dt < win_first or start_dt > win_last:
            raise ConfigError(
                f"--start {norm_start} is outside the OM4 replay window "
                f"({SAMUDRA_WINDOW[0]} .. {SAMUDRA_WINDOW[1]}, julian calendar).",
                field="start",
            )
        step_td = datetime.timedelta(hours=120)
        end_dt = start_dt + (step_td * n_steps)
        end_iso = format_cftime(end_dt)
        if end_dt > win_last:
            max_even = int(((win_last - start_dt) // step_td) // 2 * 2)
            raise ConfigError(
                f"--steps {n_steps} (end {end_iso}) exceeds the OM4 replay window end {SAMUDRA_WINDOW[1]}; "
                f"at most {max_even} even 5-day steps are available from {norm_start}.",
                field="steps",
            )
    elif model.family == "md":
        md_starts = tuple(md15_variant(model.name)["published_starts"])
        if norm_start[8:] != "01T00:00:00":
            raise ConfigError(
                f"--start for {model.name} must be month-start midnight YYYY-MM-01T00:00:00 "
                f"(published starts: {', '.join(md_starts)}), got {norm_start}.",
                field="start",
                allowed=list(md_starts),
            )
        if ic_path is not None:
            validated_custom_ic = validate_visible_input_path(ic_path, label="--ic")
            if not os.path.isfile(validated_custom_ic) or os.path.islink(validated_custom_ic):
                raise ConfigError(
                    f"Initial condition file not found (or symlinked) at --ic {validated_custom_ic!r}.",
                    field="ic",
                )
        elif norm_start not in md_starts:
            raise ConfigError(
                f"--start {norm_start} has no published prebuilt initial state for {model.name} "
                f"(published starts: {', '.join(md_starts)}; or pass an explicit `--ic <path.nc>`).",
                field="start",
                allowed=list(md_starts),
            )
        names = [fs.name for fs in model.forcing_sets]
        if forcing_set_name is None:
            scenario = names[0]
        else:
            scenario = validate_forcing_set_name(forcing_set_name)
            if scenario not in names:
                raise ConfigError(
                    f"Unknown --forcing-set {scenario!r} for {model.name}: choose one of {', '.join(names)}.",
                    field="forcing_set",
                    allowed=names,
                )
        if not (MD15_FORCING_WINDOW[0] <= norm_start <= MD15_FORCING_WINDOW[1]):
            raise ConfigError(
                f"--start {norm_start} is outside the AIMIP forcing window "
                f"({MD15_FORCING_WINDOW[0]} .. {MD15_FORCING_WINDOW[1]}).",
                field="start",
            )
        ey, em = md15_step_months(int(norm_start[0:4]), int(norm_start[5:7]), n_steps - 1)
        end_iso = f"{ey:04d}-{em:02d}-01T00:00:00"
        if end_iso > MD15_FORCING_WINDOW[1]:
            max_avail = md15_months_between(norm_start, MD15_FORCING_WINDOW[1]) + 1
            raise ConfigError(
                f"--steps {n_steps} (last month {end_iso[:7]}) exceeds AIMIP forcing end {MD15_FORCING_WINDOW[1][:10]}; "
                f"at most {max_avail} monthly steps are available from {norm_start[:10]}.",
                field="steps",
            )
    elif model.family == "camulator":
        if norm_start not in CAMULATOR_PUBLISHED_STARTS:
            raise ConfigError(
                f"--start {norm_start} is not one of the 69 published initial states for {model.name} "
                "(1980-01-01T00:00:00, or YYYY-01-01T00:00:00 / YYYY-07-01T00:00:00 for 1981..2014; "
                "a new initial state requires the training Zarr archive, which is not public).",
                field="start",
                allowed=list(CAMULATOR_PUBLISHED_STARTS),
            )
        if n_members > 1:
            raise ConfigError(
                f"{model.name} (checkpoint.pt00069.pt) is deterministic: --members must be 1.",
                field="members",
                allowed=[1],
            )
        names = [fs.name for fs in model.forcing_sets]
        if forcing_set_name is None:
            scenario = names[0]
        else:
            scenario = validate_forcing_set_name(forcing_set_name)
            if scenario not in names:
                raise ConfigError(
                    f"Unknown --forcing-set {scenario!r} for {model.name}: choose one of {', '.join(names)} "
                    "(upstream b.e21 SST source is undocumented, so it is not named 'amip-hist').",
                    field="forcing_set",
                    allowed=names,
                )
        win = CAMULATOR_BRANCH_WINDOW if scenario == CAMULATOR_BRANCH_FORCING_NAME else CAMULATOR_CYCLIC_WINDOW
        start_dt = parse_cftime(norm_start, calendar="noleap")
        win_first = parse_cftime(win[0], calendar="noleap")
        win_last = parse_cftime(win[1], calendar="noleap")
        step_td = datetime.timedelta(hours=6)
        last_step_dt = start_dt + (step_td * (n_steps - 1))
        end_dt = start_dt + (step_td * n_steps)
        end_iso = format_cftime(end_dt)
        if start_dt < win_first or last_step_dt > win_last:
            max_steps = int((win_last - start_dt) // step_td) + 1
            raise ConfigError(
                f"--steps {n_steps} (end {end_iso}, noleap calendar) exceeds forcing set {scenario!r} "
                f"end {win[1]}; at most {max_steps} 6-hourly steps are available from {norm_start}.",
                field="steps",
            )
    elif model.family == "cbottle":
        names = [fs.name for fs in model.forcing_sets]
        if forcing_set_name is None:
            scenario = names[0]
        else:
            scenario = validate_forcing_set_name(forcing_set_name)
            if scenario not in names:
                raise ConfigError(
                    f"Unknown --forcing-set {scenario!r} for {model.name}: choose one of {', '.join(names)}.",
                    field="forcing_set",
                    allowed=names,
                )
        cbottle_sample_timestamps(norm_start, n_steps, cadence=CBOTTLE_DEFAULT_CADENCE)
        start_dt_g = datetime.datetime.strptime(norm_start, "%Y-%m-%dT%H:%M:%S")
        end_iso = (start_dt_g + datetime.timedelta(hours=24 * n_steps)).strftime("%Y-%m-%dT%H:%M:%S")
    elif model.family == "sdyff":
        names = [fs.name for fs in model.forcing_sets]
        if forcing_set_name is None:
            scenario = names[0]
        else:
            scenario = validate_forcing_set_name(forcing_set_name)
            if scenario not in names:
                raise ConfigError(
                    f"Unknown --forcing-set {scenario!r} for {model.name}: choose one of {', '.join(names)}.",
                    field="forcing_set",
                    allowed=names,
                )
        sdyff_plan_schedule(norm_start, n_steps)
        start_dt_nl = parse_cftime(norm_start, calendar="noleap")
        end_iso = format_cftime(start_dt_nl + datetime.timedelta(hours=6 * n_steps))
    else:
        raise ConfigError(f"Model {model.name!r} is not an installed emulator model.")
    return {"start": norm_start, "steps": n_steps, "members": n_members, "scenario": scenario, "end": end_iso}


def aw_restart_path(ic_path: str) -> str:
    """`--ic` for an ArchesWeather continuation: a run directory (its restart.nc) or the restart.nc itself."""
    path = validate_visible_input_path(ic_path, label="--ic")
    if os.path.isdir(path):
        path = validate_visible_input_path(os.path.join(path, AW_RESTART), label=f"--ic {AW_RESTART}")
    if os.path.islink(path) or not os.path.isfile(path):
        raise ConfigError(f"--ic {path}: no restart file there (a finished ArchesWeather run writes {AW_RESTART}).")
    return path


def aw_restart_info(path: str, model_name: str) -> dict[str, Any]:
    """Attributes and sizes of an ArchesWeather restart of *model_name* (no arrays are read)."""
    import netCDF4

    try:
        with netCDF4.Dataset(path) as ds:
            attrs = {k: ds.getncattr(k) for k in ds.ncattrs()}
            members = len(ds.dimensions["sample"]) if "sample" in ds.dimensions else 0
    except OSError as exc:
        raise ConfigError(f"--ic {path} is not a readable netCDF file: {exc}") from exc
    if attrs.get("aimip_restart_format") != AW_RESTART_FORMAT:
        raise ConfigError(f"--ic {path} is not an ArchesWeather restart written by aimip "
                          f"(aimip_restart_format={attrs.get('aimip_restart_format')!r}, expected {AW_RESTART_FORMAT}).")
    try:
        info = {"path": path, "model": str(attrs["model"]), "valid_time": str(attrs["valid_time"]),
                "step_index": int(attrs["step_index"]), "seed": int(attrs["seed"]), "members": members,
                "package_digest": str(attrs["package_digest"]), "geoarches_commit": str(attrs["geoarches_commit"])}
    except (KeyError, TypeError, ValueError) as exc:
        raise ConfigError(f"--ic {path}: restart attribute missing or malformed ({exc}).") from exc
    if info["model"] != model_name:   # before --members / --seed default to the restart's
        raise ConfigError(f"--ic {path} continues a {info['model']} run, not {model_name}.")
    return info


def check_aw_restart(info: dict[str, Any], *, model_name: str, start: str, members: int, seed: int,
                     package_digest: str | None = None) -> None:
    """A continuation must pick up exactly where the restart left off, so that it matches one longer run: same model,
    --start = the restart's state time, the same members and seed (the noise of step i of member k is drawn with seed
    seed + k + 1000 i, i counted from 1978-09-30), and the same pinned files and geoarches source."""
    from .geoarches_src import GEOARCHES_COMMIT

    path = info["path"]
    if info["model"] != model_name:
        raise ConfigError(f"--ic {path} continues a {info['model']} run, not {model_name}.")
    if start != info["valid_time"]:
        raise ConfigError(f"--ic {path} holds the state of {info['valid_time']}: use --start {info['valid_time']}.")
    if members != info["members"]:
        raise ConfigError(f"--ic {path} holds {info['members']} member(s): use --members {info['members']} "
                          "(or leave it out).")
    if seed != info["seed"]:
        raise ConfigError(f"--ic {path} comes from a run with --seed {info['seed']}: use that seed (or leave it out), so "
                          "that the continuation draws the same noise as one longer run.")
    days = (datetime.datetime.fromisoformat(start) - datetime.datetime.fromisoformat(AW_START)).days
    if info["step_index"] != days:
        raise ValidationError(f"--ic {path}: step_index {info['step_index']} does not match its state time {start}.")
    if info["geoarches_commit"] != GEOARCHES_COMMIT:
        raise ValidationError(f"--ic {path} was written with geoarches {info['geoarches_commit'][:12]}; this install "
                              f"pins {GEOARCHES_COMMIT[:12]}.")
    if package_digest is not None and info["package_digest"] != package_digest:
        raise ValidationError(f"--ic {path} was written with other model or input files than this install pins "
                              "(checkpoints, statistics, initial state or forcing differ).")


def _stage_dlesym_ic(out_dir: str) -> str:
    """Fetch the ERA5 state at --start into <out>/ic_era5.pt (login node, needs internet)."""
    py = emu_python()
    print(f"aimip: fetching the ERA5 initial state from ARCO-ERA5 (public Google Cloud bucket) into {out_dir}", file=sys.stderr)
    t0 = time.monotonic()
    try:
        # Redirect child stdout to stderr so `--json` and `aimip mcp` stdio streams remain pure protocol/JSON.
        try:
            sys.stderr.fileno()
            has_stderr_fd = True
        except (AttributeError, OSError, io.UnsupportedOperation):
            has_stderr_fd = False
        if has_stderr_fd:
            res = subprocess.run(
                [py, "-I", "-m", EMU_JOB_MODULE, "stage", out_dir],
                env=_job_env(out_dir, py),
                stdout=sys.stderr,
                stderr=sys.stderr,
                check=False,
            )
        else:
            res = subprocess.run(
                [py, "-I", "-m", EMU_JOB_MODULE, "stage", out_dir],
                env=_job_env(out_dir, py),
                capture_output=True,
                text=True,
                check=False,
            )
            if res.stdout:
                sys.stderr.write(res.stdout)
            if res.stderr:
                sys.stderr.write(res.stderr)
    finally:
        shutil.rmtree(os.path.join(out_dir, ".aimip-tmp"), ignore_errors=True)
    print(f"aimip: ERA5 state staged in {time.monotonic() - t0:.0f} s", file=sys.stderr)
    ic = os.path.join(out_dir, "ic_era5.pt")
    if res.returncode != 0 or not os.path.isfile(ic):
        raise ConfigError(
            f"Could not fetch the ERA5 initial state (rc={res.returncode}); see the messages above. "
            "This step needs internet access on the node where `aimip run` is called."
        )
    strip_group_other_write(ic)
    return ic


def _stage_samudra_window(out_dir: str) -> tuple[str, str, int, int]:
    """Stage the OM4 evaluation window into `<out>/window` and return `(window_json_path, tree_sha, files, bytes)`."""
    py = emu_python()
    print(
        f"aimip: staging OM4 5-day window from OSN (s3://m2lines-pubs/Samudra/OM4) into {out_dir}/window",
        file=sys.stderr,
    )
    t0 = time.monotonic()
    try:
        try:
            sys.stderr.fileno()
            has_stderr_fd = True
        except (AttributeError, OSError, io.UnsupportedOperation):
            has_stderr_fd = False
        if has_stderr_fd:
            res = subprocess.run(
                [py, "-I", "-m", EMU_JOB_MODULE, "stage", out_dir],
                env=_job_env(out_dir, py),
                stdout=sys.stderr,
                stderr=sys.stderr,
                check=False,
            )
        else:
            res = subprocess.run(
                [py, "-I", "-m", EMU_JOB_MODULE, "stage", out_dir],
                env=_job_env(out_dir, py),
                capture_output=True,
                text=True,
                check=False,
            )
            if res.stdout:
                sys.stderr.write(res.stdout)
            if res.stderr:
                sys.stderr.write(res.stderr)
    finally:
        shutil.rmtree(os.path.join(out_dir, ".aimip-tmp"), ignore_errors=True)
    print(f"aimip: OM4 window staged in {time.monotonic() - t0:.0f} s", file=sys.stderr)
    window_dir = os.path.join(out_dir, "window")
    window_json = os.path.join(window_dir, "window.json")
    if res.returncode != 0 or not os.path.isfile(window_json) or not os.path.isdir(os.path.join(window_dir, "OM4.zarr")):
        raise ConfigError(
            f"Could not stage the Samudra OM4 window (rc={res.returncode}); check --start is on the 5-day OM4 grid "
            "and internet access is available on the staging node."
        )
    tree_sha, tree_files, tree_bytes = compute_tree_digest(window_dir, label="staged Samudra OM4 window")
    return window_json, tree_sha, tree_files, tree_bytes


_STRIPPED_JOB_ENV_KEYS: frozenset[str] = frozenset(
    {
        "AIMIP_MD15_CODE",
        "AIMIP_CAMULATOR_CODE",
        "AIMIP_SDYFF_CODE",
        "AIMIP_ARCHES_CODE",
        "AIMIP_MD15_ARCHIVE_URL",
        "AIMIP_CAMULATOR_ARCHIVE_URL",
        "AIMIP_SDYFF_ARCHIVE_URL",
        "AIMIP_CAMULATOR_ARCHIVE_SHA256",
        "AIMIP_CAMULATOR_ARCHIVE_SIZE",
        "AIMIP_CAMULATOR_TREE_SHA256",
        "AIMIP_CAMULATOR_TREE_FILES",
    }
)


def _job_env(out_dir: str, py_exe: str | None = None) -> dict[str, str]:
    from .submit import sanitized_subprocess_env

    clean_parent = sanitized_subprocess_env()
    env = {
        k: v
        for k, v in clean_parent.items()
        if k.startswith(("AIMIP_", "SLURM_", "CUDA_", "NVIDIA_")) and k not in _STRIPPED_JOB_ENV_KEYS
    }
    if os.environ.get("LD_LIBRARY_PATH"):
        # Apptainer --nv and HPC CUDA modules expose host driver libraries via LD_LIBRARY_PATH.
        env["LD_LIBRARY_PATH"] = os.environ["LD_LIBRARY_PATH"]
    tmp = os.path.join(out_dir, ".aimip-tmp")
    emu_bin = os.path.dirname(py_exe) if py_exe else "/opt/aimip/venv-emu/bin"
    env.update({
        "PATH": f"{emu_bin}:/usr/local/bin:/usr/bin:/bin",
        "HOME": "/tmp",
        "PYTHONNOUSERSITE": "1",
        "AIMIP_EMU_TMP": tmp,
        "EARTH2STUDIO_CACHE": os.path.join(tmp, "e2s-cache"),
        "MPLCONFIGDIR": os.path.join(tmp, "mpl"),
        "MPLBACKEND": "Agg",
        "HF_HUB_OFFLINE": "1",
        "WANDB_MODE": "disabled",
        "WANDB_DISABLED": "true",
        "OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS", "16"),
    })
    return env


def prepare_emu_run(
    *,
    model: ModelSpec,
    start_iso: str,
    steps: int,
    out_dir: str | None,
    ic_path: str | None,
    ocean_ic_path: str | None,
    members: int | None,
    seed: int | None,
    lora_path: str | None,
    forcing_set_name: str | None,
    mode: str,
    slurm_opts: Any,
    site_config: SiteConfig | None,
    playground_dir: str | None,
    data_dir: str | None,
    auto_fetch: bool,
    argv: Sequence[str] | None,
) -> dict[str, Any]:
    from .runner import (   # local import: runner imports this module
        SlurmOptions,
        default_out_dir,
        ensure_model_files_ready,
        model_has_missing_files,
        render_sbatch_script,
        resolve_active_lock_sha256,
    )

    seed_explicit = seed is not None
    aw_restart = None
    if model.family == "geoarches" and ic_path is not None:
        aw_restart = aw_restart_info(aw_restart_path(ic_path), model.name)
        members = aw_restart["members"] if members is None else members
        seed = aw_restart["seed"] if seed is None else seed
    req = precheck_emu_request(
        model, start_iso=start_iso, steps=steps, members=members, forcing_set_name=forcing_set_name,
        ic_path=ic_path, ocean_ic_path=ocean_ic_path, lora_path=lora_path,
    )
    if aw_restart is not None:   # before anything is written; the pinned files are compared once they are verified
        check_aw_restart(aw_restart, model_name=model.name, start=req["start"], members=req["members"],
                         seed=validate_seed(seed))
    if model.family in ("e2s-dlesym", "samudra") and not auto_fetch:
        # --no-fetch forbids network downloads during preparation.
        src_desc = (
            "its ERA5 initial state from ARCO-ERA5"
            if model.family == "e2s-dlesym"
            else "its OM4 ocean window from OSN (s3://m2lines-pubs/Samudra/OM4)"
        )
        raise MissingAssetsError(
            f"{model.name} reads {src_desc} when the run is prepared, and --no-fetch forbids "
            f"downloads. Prepare it on the host: aimip run -m {model.name} --start {req['start']} --steps {req['steps']} "
            "--submit",
            field="model",
            next_command=f"aimip run -m {model.name} --start {req['start']} --steps {req['steps']} --submit",
        )
    resolved_seed = validate_seed(0 if seed is None else seed)
    resolved_playground = resolve_playground_dir(playground_dir)
    active_site = site_config if site_config is not None else load_site_config()

    if out_dir is not None:
        validated_out = validate_run_path(out_dir, label="--out")
    else:
        validated_out = default_out_dir(
            model.name, req["start"], extra_entropy=f"{req['steps']}|{req['members']}|{resolved_seed}|{req['scenario']}"
        )
    if resolved_playground:
        pg = resolved_playground.rstrip("/")
        if validated_out == pg or validated_out.startswith(pg + "/"):
            raise ConfigError(
                f"Refusing to write run outputs ({validated_out}) inside $AIMIP_PLAYGROUND ({pg}); choose another --out."
            )
    manifest_path = validate_run_path(f"{validated_out}/aimip-manifest.json", label="aimip-manifest.json")
    if os.path.isfile(manifest_path):
        try:
            with open(manifest_path, encoding="utf-8") as fh:
                prev = json.load(fh)
            if isinstance(prev, dict) and "finished" in prev:
                raise ConfigError(f"Refusing to overwrite completed run directory {validated_out}; choose a new --out path.")
        except (OSError, json.JSONDecodeError):
            pass

    if model_has_missing_files(model, playground_dir=playground_dir, data_dir=data_dir):
        model = ensure_model_files_ready(model, playground_dir=playground_dir, data_dir=data_dir, auto_fetch=auto_fetch)
    data_root = resolve_data_dir(data_dir, playground_dir=resolved_playground)

    pkg = model_package_files(model, data_root)
    missing = [p for p in pkg.values() if not os.path.isfile(p)]
    if missing:
        raise ConfigError(f"{model.name}: model files missing after fetch: {', '.join(missing[:3])}")
    print(f"aimip: verifying {len(pkg)} model file(s) of {model.name}", file=sys.stderr)
    pkg_sha = {p: sha256_file(p) for p in pkg.values()}
    _check_against_lock(model.name, pkg, pkg_sha, data_root)
    if model.checkpoint not in pkg_sha:
        raise ConfigError(f"Internal error: checkpoint {model.checkpoint} is not among the pinned files of {model.name}.")
    if model.checkpoint_sha256 and pkg_sha[model.checkpoint].lower() != model.checkpoint_sha256.lower():
        raise ValidationError(
            f"Checkpoint SHA-256 mismatch for model {model.name!r}: registry expects {model.checkpoint_sha256}, "
            f"got {pkg_sha[model.checkpoint]}."
        )

    effective_slurm = slurm_opts.validate() if slurm_opts is not None else SlurmOptions.from_site(active_site)
    makedirs_no_group_other_write(validated_out)

    scenario_rows: list[dict[str, Any]] = []
    window_meta: dict[str, Any] = {}
    ic_is_custom = False
    precomputed_ic_file: str | None = None
    if model.family == "e2s-dlesym":
        info = DLESYM_MODELS[model.name]
        package_dir = os.path.join(data_root, info["package"])
        job = {"family": model.family, "model": model.name, "version": info["version"], "flavor": info["flavor"],
               "package_dir": package_dir,
               "sst_forcing": os.path.join(data_root, AIMIP_SST_REL) if info.get("prescribed_sst") else None}
        ic_provenance = f"ERA5 state at {req['start']} from ARCO-ERA5 (gs://gcp-public-data-arco-era5), staged at prepare time"
        if info.get("prescribed_sst"):
            ic_provenance += "; SST prescribed from the AIMIP forcing (Zenodo 17065758) at every step"
    elif model.family == "geoarches":
        from .emu_catalog import AW_DIR, AW_FORCING_STATS, AW_GEN, AW_NORM_STATS, AW_RESIDUAL_STATS, AW_SEEDS
        from .geoarches_src import GEOARCHES_TREE_FILES, GEOARCHES_TREE_SHA256, code_dir, tree_digest

        geo_dir = code_dir(emu_python())
        if not os.path.isdir(geo_dir):
            raise ConfigError(
                f"{model.name}: the geoarches source is not installed at {geo_dir}; this install predates ArchesWeather "
                "support. Reinstall (install/install-container.sh, or install/install-native.sh --with-emu)."
            )
        if not os.environ.get("AIMIP_GEOARCHES_CODE") and tree_digest(geo_dir) != (GEOARCHES_TREE_SHA256,
                                                                                GEOARCHES_TREE_FILES):
            raise ValidationError(f"{model.name}: the geoarches source at {geo_dir} does not match the pinned commit.")
        digest = aw_package_digest(pkg_sha.values())
        if aw_restart is not None:
            check_aw_restart(aw_restart, model_name=model.name, start=req["start"], members=req["members"],
                             seed=resolved_seed, package_digest=digest)
        job = {"family": model.family, "model": model.name, "kind": "gen" if model.name == "archesweathergen" else "mx4",
               "seed_dirs": [os.path.join(data_root, AW_DIR, s) for s in AW_SEEDS],
               "gen_dir": os.path.join(data_root, AW_DIR, AW_GEN) if model.name == "archesweathergen" else None,
               "ic_file": os.path.join(data_root, AW_IC), "forcing_file": os.path.join(data_root, AW_FORCING),
               "forcing_stats_file": os.path.join(data_root, AW_FORCING_STATS),
               "norm_stats_file": os.path.join(data_root, AW_NORM_STATS),
               "residual_stats_file": (os.path.join(data_root, AW_RESIDUAL_STATS)
                                       if model.name == "archesweathergen" else None),
               "package_digest": digest, "restart_file": aw_restart["path"] if aw_restart else None}
        if aw_restart is None:
            ic_provenance = ("ERA5 daily-mean state of 1978-09-30 (previous state 1978-09-29) from Zenodo 22795233 "
                             "era5_1978_00h_sliced.nc; first output day 1978-10-01")
        else:
            first_day = datetime.datetime.fromisoformat(req["start"]) + datetime.timedelta(days=1)
            ic_provenance = (f"continues {aw_restart['path']} (states of {req['start'][:10]} and the day before, step "
                             f"{aw_restart['step_index']} from 1978-09-30); first output day {first_day:%Y-%m-%d}")
    elif model.family == "samudra":
        means_zip = os.path.join(data_root, "models/samudra-om4/OM4_means.zip")
        stds_zip = os.path.join(data_root, "models/samudra-om4/OM4_stds.zip")
        window_dir = os.path.join(validated_out, "window")
        osn_source = (os.environ.get("AIMIP_SAMUDRA_SOURCE") or SAMUDRA_OSN_SOURCE).strip()
        job = {
            "family": model.family,
            "model": model.name,
            "window_dir": window_dir,
            "osn_source": osn_source,
            "checkpoint_sha256": pkg_sha[model.checkpoint],
            "means_zip": means_zip,
            "means_zip_sha256": pkg_sha[means_zip],
            "stds_zip": stds_zip,
            "stds_zip_sha256": pkg_sha[stds_zip],
        }
        ic_provenance = (
            f"Two 5-day OM4 ocean states ending at {req['start']} plus surface boundary fluxes "
            f"staged from {osn_source}/OM4"
        )
    elif model.family == "md":
        if (os.environ.get("AIMIP_MD15_CODE") or "").strip():
            md_code_dir, md_code_ovr = resolve_md15_code_dir(data_root)
            md_tree_sha, md_tree_files = compute_sha256sum_tree_digest(
                md_code_dir, label="AIMIP_MD15_CODE"
            )
        else:
            md_code_info = verify_md15_code_dir(data_root)
            md_code_dir = str(md_code_info["code_dir"])
            md_tree_sha = str(md_code_info["tree_sha256"])
            md_tree_files = int(md_code_info["files"])
            md_code_ovr = False
        md_var = md15_variant(model.name)
        ckpt_dir = os.path.dirname(model.checkpoint)
        config_pkl = os.path.join(ckpt_dir, "config.pkl")
        stats_file = os.path.join(data_root, md_var["stats_rel"])
        for req_p, req_lbl in ((config_pkl, "config.pkl"), (stats_file, "variable_statistics JSON")):
            if req_p not in pkg_sha:
                raise ConfigError(f"{model.name}: {req_lbl} ({req_p}) is missing from required_files.")
        selected_fs = next(fs for fs in model.forcing_sets if fs.name == req["scenario"])
        forcing_file = os.path.join(selected_fs.dir, "ERA5-0.25deg-monthly-mean-forcing-1978-2024.nc")
        forcing_rel = os.path.relpath(forcing_file, data_root).replace(os.sep, "/")
        if forcing_file in pkg_sha:
            scenario_rows = [
                {
                    "rel_path": forcing_rel,
                    "path": forcing_file,
                    "size": os.path.getsize(forcing_file),
                    "sha256": pkg_sha[forcing_file],
                }
            ]
        else:
            if not os.path.isfile(forcing_file) or os.path.islink(forcing_file):
                raise ConfigError(
                    f"{model.name}: AIMIP forcing file is missing ({forcing_file}); run `aimip fetch -m {model.name}`."
                )
            forcing_sha = sha256_file(forcing_file)
            _check_against_lock(model.name, {forcing_rel: forcing_file}, {forcing_file: forcing_sha}, data_root)
            scenario_rows = [
                {
                    "rel_path": forcing_rel,
                    "path": forcing_file,
                    "size": os.path.getsize(forcing_file),
                    "sha256": forcing_sha,
                }
            ]
        if ic_path is not None:
            precomputed_ic_file = validate_visible_input_path(ic_path, label="--ic")
            ic_is_custom = True
            ic_provenance = f"Explicit user initial condition {precomputed_ic_file} at {req['start'][:10]}"
        else:
            ymd_tag = req["start"][:10]
            matched_ics = [p for p in model.ic_files if ymd_tag in os.path.basename(p) or ymd_tag.replace("-", "") in os.path.basename(p)]
            precomputed_ic_file = (
                matched_ics[0]
                if matched_ics
                else os.path.join(data_root, f"inputs/md-1.5/ic/ic_{ymd_tag}.nc")
            )
            if not os.path.isfile(precomputed_ic_file) or os.path.islink(precomputed_ic_file):
                raise ConfigError(
                    f"{model.name}: no initial condition is staged at {precomputed_ic_file} "
                    f"(upstream publishes no initial-state file); pass `--ic <era5_monthly_state.nc>` "
                    f"(under $AIMIP_DATA or your run root) or stage the {ymd_tag} 1.5 deg ERA5 monthly state at {precomputed_ic_file}.",
                    field="ic",
                )
            ic_provenance = (
                f"Staged 1.5 deg ERA5 monthly-mean initial state {precomputed_ic_file} ({ymd_tag}, Copernicus/ECMWF attribution applies)"
            )
        md_ic_sha = sha256_file(precomputed_ic_file)
        job = {
            "family": model.family,
            "model": model.name,
            "scenario": req["scenario"],
            "data_root": data_root,
            "code_dir": md_code_dir,
            "code_tree_sha256": md_tree_sha,
            "code_tree_files": md_tree_files,
            "code_is_override": md_code_ovr,
            "verify_code_tree": True,
            "ckpt_dir": ckpt_dir,
            "config_sha256": pkg_sha[config_pkl],
            "weights_sha256": pkg_sha[model.checkpoint],
            "stats_file": stats_file,
            "stats_sha256": pkg_sha[stats_file],
            "forcing_file": forcing_file,
            "forcing_regrid": md_var["forcing_regrid"],
            "arch": md_var["arch"],
            "ic_file": precomputed_ic_file,
            "ic_sha256": md_ic_sha,
        }
        window_meta = {
            "code_dir": md_code_dir,
            "code_tree_sha256": md_tree_sha,
            "code_tree_files": md_tree_files,
            "code_is_override": md_code_ovr,
            "verify_code_tree": True,
        }
    elif model.family == "camulator":
        from .setup_data import ensure_camulator_selected_assets

        if (os.environ.get("AIMIP_CAMULATOR_CODE") or "").strip():
            cam_code_dir, cam_code_ovr = resolve_camulator_code_dir(data_root)
            cam_tree_sha, cam_tree_files = compute_sha256sum_tree_digest(
                cam_code_dir, label="AIMIP_CAMULATOR_CODE"
            )
        else:
            cam_code_info = verify_camulator_code_dir(data_root)
            cam_code_dir = str(cam_code_info["code_dir"])
            cam_tree_sha = str(cam_code_info["tree_sha256"])
            cam_tree_files = int(cam_code_info["files"])
            cam_code_ovr = False
        selected_assets = ensure_camulator_selected_assets(
            data_root,
            start_iso=req["start"],
            scenario=req["scenario"],
            auto_fetch=auto_fetch,
            progress_stream=sys.stderr,
        )
        cam_ic_info = selected_assets["ic"]
        cam_forcing_info = selected_assets["forcing"]
        precomputed_ic_file = str(cam_ic_info["abs_path"])
        scenario_rows = [
            {
                "rel_path": cam_forcing_info["rel_path"],
                "path": cam_forcing_info["abs_path"],
                "size": int(cam_forcing_info["size"]),
                "sha256": str(cam_forcing_info["sha256"]),
            }
        ]
        base_cfg_p = os.path.join(data_root, CAMULATOR_CONFIG_REL_PATH)
        meta_yaml_p = os.path.join(data_root, CAMULATOR_METADATA_REL_PATH)
        mean_nc_p = os.path.join(data_root, CAMULATOR_MEAN_REL_PATH)
        std_nc_p = os.path.join(data_root, CAMULATOR_STD_REL_PATH)
        statics_b_p = os.path.join(data_root, CAMULATOR_STATICS_B_REL_PATH)
        statics_be21_p = os.path.join(data_root, CAMULATOR_STATICS_BE21_REL_PATH)
        statics_fe21_p = os.path.join(data_root, CAMULATOR_STATICS_FE21_REL_PATH)
        job = {
            "family": model.family,
            "model": model.name,
            "scenario": req["scenario"],
            "data_root": data_root,
            "code_dir": cam_code_dir,
            "code_tree_sha256": cam_tree_sha,
            "code_tree_files": cam_tree_files,
            "code_is_override": cam_code_ovr,
            "verify_code_tree": True,
            "checkpoint_sha256": pkg_sha[model.checkpoint],
            "base_config_file": base_cfg_p,
            "base_config_sha256": pkg_sha[base_cfg_p],
            "metadata_file": meta_yaml_p,
            "metadata_sha256": pkg_sha[meta_yaml_p],
            "mean_file": mean_nc_p,
            "mean_sha256": pkg_sha[mean_nc_p],
            "std_file": std_nc_p,
            "std_sha256": pkg_sha[std_nc_p],
            "statics_b_file": statics_b_p,
            "statics_b_sha256": pkg_sha[statics_b_p],
            "statics_be21_file": statics_be21_p,
            "statics_be21_sha256": pkg_sha[statics_be21_p],
            "statics_fe21_file": statics_fe21_p,
            "statics_fe21_sha256": pkg_sha[statics_fe21_p],
            "forcing_file": str(cam_forcing_info["abs_path"]),
            "forcing_sha256": str(cam_forcing_info["sha256"]),
            "ic_file": precomputed_ic_file,
            "ic_sha256": str(cam_ic_info["sha256"]),
        }
        window_meta = {
            "code_dir": cam_code_dir,
            "code_tree_sha256": cam_tree_sha,
            "code_tree_files": cam_tree_files,
            "code_is_override": cam_code_ovr,
            "verify_code_tree": True,
        }
        ic_provenance = (
            f"Published CAMulator initial-condition tensor ({os.path.basename(precomputed_ic_file)}) "
            f"from hf://willychap/camulator@{model.checkpoint_sha256[:8]} (Apache-2.0)"
        )
    elif model.family == "cbottle":
        sst_p = os.path.join(data_root, CBOTTLE_SST_REL_PATH)
        expert_ckpts = []
        for c_rel in CBOTTLE_CHECKPOINT_REL_PATHS:
            c_abs = os.path.join(data_root, c_rel)
            if c_abs not in pkg_sha:
                raise ConfigError(f"{model.name}: expert checkpoint {c_abs} missing from required_files.")
            expert_ckpts.append({"path": c_abs, "sha256": pkg_sha[c_abs]})
        if sst_p not in pkg_sha:
            raise ConfigError(f"{model.name}: conditioning SST file {sst_p} missing from required_files.")
        n_cb_batches = (int(req["steps"]) + CBOTTLE_BATCH_SIZE - 1) // CBOTTLE_BATCH_SIZE
        cb_batch_seeds = [
            [cbottle_batch_seed(resolved_seed, m_i, b_i) for b_i in range(n_cb_batches)]
            for m_i in range(int(req["members"]))
        ]
        job = {
            "family": model.family,
            "model": model.name,
            "scenario": req["scenario"],
            "expert_checkpoints": expert_ckpts,
            "sst_file": sst_p,
            "sst_sha256": pkg_sha[sst_p],
            "batch_size": CBOTTLE_BATCH_SIZE,
            "cadence": CBOTTLE_DEFAULT_CADENCE,
            "date_order": CBOTTLE_DATE_ORDER,
            "seed_scheme": CBOTTLE_SEED_SCHEME,
        }
        window_meta = {
            "batch_size": CBOTTLE_BATCH_SIZE,
            "cadence": CBOTTLE_DEFAULT_CADENCE,
            "date_order": CBOTTLE_DATE_ORDER,
            "seed_scheme": CBOTTLE_SEED_SCHEME,
            "batch_seeds": cb_batch_seeds,
        }
        ic_provenance = (
            "None (conditional atmospheric state generator without state memory; conditioned on "
            "day_of_year, hour_of_day, and PCMDI-AMIP-1-1-9 monthly mid-point SST tosbcs)"
        )
    elif model.family == "sdyff":
        if (os.environ.get("AIMIP_SDYFF_CODE") or "").strip():
            sd_code_dir, sd_code_ovr = resolve_sdyff_code_dir(data_root)
            if not os.path.isdir(sd_code_dir) or os.path.islink(sd_code_dir):
                raise ValidationError(f"AIMIP_SDYFF_CODE is missing or symlinked: {sd_code_dir}")
            sd_tree_sha, sd_tree_files = compute_sha256sum_tree_digest(
                sd_code_dir, label="AIMIP_SDYFF_CODE"
            )
        else:
            sd_code_info = verify_sdyff_code_dir(data_root)
            sd_code_dir = str(sd_code_info["code_dir"])
            sd_tree_sha = str(sd_code_info["tree_sha256"])
            sd_tree_files = int(sd_code_info["files"])
            sd_code_ovr = False
        f_ckpt_p = os.path.join(data_root, SDYFF_FORECASTER_CKPT_REL_PATH)
        i_ckpt_p = os.path.join(data_root, SDYFF_INTERPOLATOR_CKPT_REL_PATH)
        i_cfg_p = os.path.join(data_root, SDYFF_INTERPOLATOR_CFG_REL_PATH)
        fv3gfs_dir = os.path.join(data_root, SDYFF_DATA_REL_DIR)
        ref_shas: dict[str, str] = {}
        for fname in sorted(SDYFF_ZENODO_FILES):
            f_abs = os.path.join(fv3gfs_dir, fname)
            if f_abs not in pkg_sha:
                raise ConfigError(f"{model.name}: FV3GFS reference file {f_abs} missing from required_files.")
            ref_shas[fname] = pkg_sha[f_abs]
        start_month = int(req["start"][5:7])
        precomputed_ic_file = os.path.join(fv3gfs_dir, f"2021{start_month:02d}0100.nc")
        sdyff_sched = sdyff_plan_schedule(req["start"], req["steps"])
        job = {
            "family": model.family,
            "model": model.name,
            "scenario": req["scenario"],
            "data_dir": data_root,
            "code_dir": sd_code_dir,
            "code_tree_sha256": sd_tree_sha,
            "code_tree_files": sd_tree_files,
            "code_is_override": sd_code_ovr,
            "verify_code_tree": True,
            "forecaster_ckpt": f_ckpt_p,
            "forecaster_sha256": pkg_sha[f_ckpt_p],
            "interpolator_ckpt": i_ckpt_p,
            "interpolator_sha256": pkg_sha[i_ckpt_p],
            "interpolator_cfg": i_cfg_p,
            "interpolator_cfg_sha256": pkg_sha[i_cfg_p],
            "fv3gfs_data_dir": fv3gfs_dir,
            "fv3gfs_file_sha256s": ref_shas,
            "coarsen_factor": SDYFF_COARSEN_FACTOR,
        }
        window_meta = {
            "code_dir": sd_code_dir,
            "code_tree_sha256": sd_tree_sha,
            "code_tree_files": sd_tree_files,
            "code_is_override": sd_code_ovr,
            "verify_code_tree": True,
            "coarsen_factor": SDYFF_COARSEN_FACTOR,
            "nominal_cycle_months": [m["label"] for m in sdyff_sched["complete_months"]],
            "calendar_note": (
                "repeating single-year SST cycle; year numbers are nominal cycle indices, not historical calendar years"
            ),
        }
        ic_provenance = (
            f"1 deg FV3GFS repeating-annual-SST reference state at {req['start']} "
            "(Zenodo 10791087, CC-BY-4.0; nominal year label 2021)"
        )
    else:
        scen = req["scenario"]
        scenario_rows = _scenario_files(model, data_root, scen)
        if not scenario_rows or any(r["size"] is None or not r.get("sha256") for r in scenario_rows):
            raise ConfigError(f"{model.name}: scenario {scen} files are missing; run `aimip fetch -m {model.name}`.")
        _check_against_lock(
            model.name,
            {str(r["rel_path"]): str(r["path"]) for r in scenario_rows},
            {str(r["path"]): str(r["sha256"]) for r in scenario_rows},
            data_root,
        )
        job = {"family": model.family, "model": model.name, "scenario": scen, "domain": ARCHES_DOMAIN[scen],
               "model_dir": os.path.dirname(model.checkpoint),
               "memmap_dir": os.path.join(data_root, "inputs/archesclimate-ssp/memmap_filled_in"),
               "reference_dir": os.path.join(data_root, "models/archesclimate-ssp/reference_data"),
               "stats_file": os.path.join(data_root, "models/archesclimate-ssp/stats/cmip_stats_new_ozone.pt"),
               "first_output_month": "%04d-%02d" % ARCHES_FIRST_OUTPUT_MONTH}
        arches_ovr_raw = (os.environ.get("AIMIP_ARCHES_CODE") or "").strip()
        if arches_ovr_raw:
            from .setup_data import compute_arches_tree_digest

            ar_code_dir = validate_safe_path(arches_ovr_raw, label="AIMIP_ARCHES_CODE")
            ar_tree_sha, ar_tree_files = compute_arches_tree_digest(ar_code_dir)
            ar_override_meta = {
                "code_dir": ar_code_dir,
                "code_tree_sha256": ar_tree_sha,
                "code_tree_files": int(ar_tree_files),
                "code_is_override": True,
            }
            job.update(ar_override_meta)
            window_meta = dict(ar_override_meta)
        ic_provenance = (f"IPSL-CM6A-LR r1i1p1f1 {scen} state of 2015-02 (previous state 2015-01) from "
                         "gclyne/ArchesClimate-data; first output month 2015-03")
    job.update({"out_dir": validated_out, "start": req["start"], "end": req["end"], "steps": req["steps"],
                "members": req["members"], "seed": resolved_seed, "checkpoint": model.checkpoint})

    config_path = validate_run_path(f"{validated_out}/config.yaml", label="config.yaml")
    with open(config_path, "w", encoding="utf-8") as fh:    # JSON is valid YAML; the job reads it with json
        json.dump(job, fh, indent=2)
        fh.write("\n")
    strip_group_other_write(config_path)

    if model.family == "e2s-dlesym":
        ic_file = _stage_dlesym_ic(validated_out)
    elif model.family == "geoarches":
        ic_file = job["restart_file"] or job["ic_file"]
    elif model.family == "samudra":
        ic_file, win_sha, win_files, win_bytes = _stage_samudra_window(validated_out)
        window_meta = {
            "window_dir": os.path.join(validated_out, "window"),
            "window_tree_sha256": win_sha,
            "window_tree_files": win_files,
            "window_tree_bytes": win_bytes,
        }
    elif model.family in ("md", "camulator", "sdyff"):
        assert precomputed_ic_file is not None
        ic_file = precomputed_ic_file
    elif model.family == "cbottle":
        ic_file = None
    else:
        ic_file = os.path.join(data_root, f"inputs/archesclimate-ssp/memmap_filled_in/r1i1p1f1_{req['scenario']}_interpolation.memmap/meta.json")
    ic_sha = sha256_file(ic_file) if ic_file is not None else None

    max_time = getattr(active_site, "max_time", "12:00:00") or "12:00:00"
    default_walltime, unclamped, capped = _walltime(model.family, req["steps"], req["members"], max_time, model.name)
    walltime_capped = bool(effective_slurm.time is None and capped)
    if effective_slurm.scheduler == "slurm" and walltime_capped:
        print(f"aimip warning: estimated walltime {unclamped} exceeds the site limit {max_time}; clamped. "
              "Split the run or pass --time.", file=sys.stderr)
    expected_s = _gpu_hours(model.family, req["steps"], req["members"], model.name) * 3600
    job_time = effective_slurm.time or default_walltime
    if model.family == "geoarches" and effective_slurm.scheduler == "slurm" and expected_s > _parse_hms(job_time):
        print(f"aimip warning: {model.name} is expected to need about {expected_s / 3600:.1f} h on one A100-80GB, "
              f"more than the wall time {job_time}. It writes its monthly files and restart.nc only at the end, so a "
              "job stopped at the limit keeps nothing: use fewer --steps or --members per run.", file=sys.stderr)
    sbatch_path = validate_run_path(f"{validated_out}/job.sbatch", label="job.sbatch")
    with open(sbatch_path, "w", encoding="utf-8") as fh:
        fh.write(render_sbatch_script(model_name=model.name, out_dir=validated_out, playground_dir=resolved_playground,
                                      slurm_opts=effective_slurm, default_walltime=default_walltime))
    strip_group_other_write(sbatch_path)

    manifest: dict[str, Any] = {
        "schema_version": 2,
        "aimip_version": __version__,
        "aimip_lock_sha256": resolve_active_lock_sha256(),
        "argv": list(argv) if argv is not None else [],
        "mode": mode,
        "out_dir": validated_out,
        "config_path": config_path,
        "sbatch_path": sbatch_path,
        "registry_entry": model.to_dict(),
        "family": model.family,
        "checkpoint_path": model.checkpoint,
        "checkpoint_sha256": pkg_sha[model.checkpoint],
        "package_files_sha256": pkg_sha,
        "ic_path": ic_file,
        "ic_sha256": ic_sha,
        "source_ic_path": ic_file,
        "source_ic_sha256": ic_sha,
        "ic_calendar": model.calendar,
        "ic_is_custom": ic_is_custom or (aw_restart is not None),
        "ensemble_strategy": (
            ("independent_conditional_draws" if model.family == "cbottle" else "seeded_noise_members")
            if (req["members"] > 1 or model.family == "cbottle")
            else "single_member"
        ),
        "ensemble_note": (
            (
                f"independent conditional diffusion draws (batch_size={CBOTTLE_BATCH_SIZE}, cadence={CBOTTLE_DEFAULT_CADENCE}, seed_scheme={CBOTTLE_SEED_SCHEME})"
                if model.family == "cbottle"
                else (
                    "members continue their own states from the restart and differ by the sampler noise seed (seed + k)"
                    if aw_restart is not None
                    else (
                        "members share the ERA5 state and differ by the sampler noise seed (seed + k)"
                        if model.family == "geoarches"
                        else (
                            "batched latent-diffusion reverse DDPM noise seeded from --seed"
                            if model.family == "md"
                            else (
                                "probabilistic Spherical DYffusion rollout seeded from --seed"
                                if model.family == "sdyff"
                                else "members share the ERA5 state and differ by the conditional-layer-norm noise seed (seed + k)"
                            )
                        )
                    )
                )
            )
            if (req["members"] > 1 or model.family == "cbottle")
            else None
        ),
        "lora_checkpoint": None,
        "lora_info": None,
        "n_ic_samples": 0 if model.family == "cbottle" else 1,
        "n_ensemble_per_ic": req["members"],
        "gpu_hours_estimate": round(_gpu_hours(model.family, req["steps"], req["members"], model.name), 6),
        "ic_provenance": ic_provenance,
        "start_time": req["start"],
        "end_time": req["end"],
        "steps": req["steps"],
        "step_unit": {
            "e2s-dlesym": "6h",
            "geoarches": "day",
            "camulator": "6h",
            "sdyff": "6h",
            "cbottle": "1d",
            "samudra": "5d",
        }.get(model.family, "month"),
        "members": req["members"],
        "seed": resolved_seed,
        "seed_explicit": seed_explicit,
        "member_seeds": (
            [cbottle_batch_seed(resolved_seed, k, 0) for k in range(req["members"])]
            if model.family == "cbottle"
            else [resolved_seed + k for k in range(req["members"])]
        ),
        "forcing_set": next(fs.to_dict() for fs in model.forcing_sets if fs.name == req["scenario"]),
        "forcing_files": scenario_rows,
        **window_meta,
        "site": {"name": active_site.name, "scheduler": effective_slurm.scheduler, "runtime": active_site.runtime},
        "slurm": {
            "scheduler": effective_slurm.scheduler, "partition": effective_slurm.partition,
            "account": effective_slurm.account, "gres": effective_slurm.gres,
            "cpus_per_task": effective_slurm.cpus_per_task, "mem": effective_slurm.mem,
            "time": effective_slurm.time or default_walltime, "unclamped_estimated_time": unclamped,
            "walltime_capped": walltime_capped,
        },
        "slurm_job_id": None,
        "image_sha": (os.environ.get("AIMIP_IMAGE_SHA") or "")[:256],
        "runtime_env": os.path.dirname(os.path.dirname(_runtime_python_for_family(model.family))),
        "runtime_python": _runtime_python_for_family(model.family),
        "config_sha256": _config_sha(config_path),
        "created_utc": _now_utc(),
    }
    if model.family == "geoarches":
        manifest["package_digest"] = job["package_digest"]
    _write_json_atomic(manifest_path, manifest)
    if mode == "local":
        return exec_emu(validated_out, manifest, manifest_path)
    return manifest


# ---------------------------------------------------------------- job time

def _check_job_matches_manifest(job: dict[str, Any], manifest: dict[str, Any]) -> None:
    """Every field the job uses must agree with the manifest and the pinned files."""
    def fail(what: str) -> None:
        raise ValidationError(f"Integrity check failed: config.yaml {what} does not match aimip-manifest.json.")

    reg = manifest.get("registry_entry") or {}
    for key, mkey in (("model", None), ("start", "start_time"), ("end", "end_time"), ("steps", "steps"),
                      ("members", "members"), ("seed", "seed"), ("family", "family"), ("checkpoint", "checkpoint_path")):
        want = reg.get("name") if mkey is None else manifest.get(mkey)
        if str(job.get(key)) != str(want):
            fail(key)
    pinned = set((manifest.get("package_files_sha256") or {}).keys())
    if job["checkpoint"] not in pinned:
        fail("checkpoint")
    if job["family"] == "e2s-dlesym":
        info = DLESYM_MODELS.get(job["model"])
        if info is None or job.get("version") != info["version"] or job.get("flavor") != info["flavor"]:
            fail("version/flavor")
        pkg = str(job.get("package_dir", "")).rstrip("/") + "/"
        sst = job.get("sst_forcing")
        if bool(info.get("prescribed_sst")) != bool(sst):
            fail("sst_forcing")
        if sst and (sst not in pinned or not sst.endswith("/" + AIMIP_SST_REL)):
            fail("sst_forcing")
        if not job["checkpoint"].startswith(pkg) or not all(p.startswith(pkg) for p in pinned if p != sst):
            fail("package_dir")
        if str(manifest.get("ic_path")) != os.path.join(job["out_dir"], "ic_era5.pt"):
            fail("initial state")
    elif job["family"] == "arches":
        scen = (manifest.get("forcing_set") or {}).get("name")
        if job.get("scenario") != scen or job.get("domain") != ARCHES_DOMAIN.get(str(scen)):
            fail("scenario/domain")
        if job.get("model_dir") != os.path.dirname(job["checkpoint"]) or job.get("stats_file") not in pinned:
            fail("model_dir/stats_file")
        if os.path.join(job["model_dir"], "config.yaml") not in pinned:
            fail("model config")
        if job.get("first_output_month") != "%04d-%02d" % ARCHES_FIRST_OUTPUT_MONTH:
            fail("first_output_month")
        mm = str(job.get("memmap_dir", "")).rstrip("/") + f"/r1i1p1f1_{scen}_interpolation.memmap/"
        rows = manifest.get("forcing_files") or []
        if not rows or not all(str(r.get("path", "")).startswith(mm) for r in rows):
            fail("memmap_dir")
        if os.path.dirname(job.get("reference_dir", "")) != os.path.dirname(os.path.dirname(job["checkpoint"])):
            fail("reference_dir")
        if (
            job.get("code_is_override")
            or manifest.get("code_is_override")
            or job.get("code_dir")
            or manifest.get("code_dir")
        ):
            if (
                bool(job.get("code_is_override", False)) != bool(manifest.get("code_is_override", False))
                or str(job.get("code_dir") or "") != str(manifest.get("code_dir") or "")
                or str(job.get("code_tree_sha256") or "").lower()
                != str(manifest.get("code_tree_sha256") or "").lower()
                or job.get("code_tree_files") != manifest.get("code_tree_files")
            ):
                fail("code_dir/code_tree_sha256")
    elif job["family"] == "geoarches":
        from .emu_catalog import AW_CKPT, AW_SEEDS

        gen = job["model"] == "archesweathergen"
        if job.get("kind") != ("gen" if gen else "mx4"):
            fail("kind")
        seed_dirs = job.get("seed_dirs") or []
        if [os.path.basename(d) for d in seed_dirs] != list(AW_SEEDS):
            fail("seed_dirs")
        dirs = list(seed_dirs) + ([job.get("gen_dir")] if gen else [])
        if not gen and job.get("gen_dir") is not None:
            fail("gen_dir")
        if job["checkpoint"] != os.path.join(str(dirs[-1] if gen else dirs[0]), AW_CKPT):
            fail("checkpoint")
        for d in dirs:
            if not isinstance(d, str) or os.path.join(d, AW_CKPT) not in pinned or os.path.join(d, "config.yaml") not in pinned:
                fail("model directories")
        for key in ("ic_file", "forcing_file", "forcing_stats_file", "norm_stats_file") + (("residual_stats_file",) if gen else ()):
            if job.get(key) not in pinned:
                fail(key)
        if not gen and job.get("residual_stats_file") is not None:
            fail("residual_stats_file")
        restart = job.get("restart_file")
        if restart is None:
            if str(manifest.get("ic_path")) != job["ic_file"] or manifest.get("ic_is_custom"):
                fail("initial state")
        elif str(manifest.get("ic_path")) != restart or not manifest.get("ic_is_custom"):
            fail("restart_file")
        digest = job.get("package_digest")
        if digest != manifest.get("package_digest") or digest != aw_package_digest(
                (manifest.get("package_files_sha256") or {}).values()):
            fail("package_digest")
    elif job["family"] == "samudra":
        exp_win = os.path.join(job["out_dir"], "window")
        if job.get("window_dir") != exp_win or manifest.get("window_dir") != exp_win:
            fail("window_dir")
        if str(manifest.get("ic_path")) != os.path.join(exp_win, "window.json"):
            fail("window.json")
        pkg_map = manifest.get("package_files_sha256") or {}
        if str(job.get("checkpoint_sha256", "")).lower() != str(manifest.get("checkpoint_sha256", "")).lower():
            fail("checkpoint_sha256")
        for zip_key, sha_key in (("means_zip", "means_zip_sha256"), ("stds_zip", "stds_zip_sha256")):
            zpath = str(job.get(zip_key) or "")
            if zpath not in pinned or str(job.get(sha_key, "")).lower() != str(pkg_map.get(zpath, "")).lower():
                fail(zip_key)
    elif job["family"] == "md":
        pkg_map = manifest.get("package_files_sha256") or {}
        if (
            str(job.get("code_dir") or "") != str(manifest.get("code_dir") or "")
            or str(job.get("code_tree_sha256") or "").lower() != str(manifest.get("code_tree_sha256") or "").lower()
            or job.get("code_tree_files") != manifest.get("code_tree_files")
            or bool(job.get("code_is_override", False)) != bool(manifest.get("code_is_override", False))
            or job.get("verify_code_tree") is not True
            or manifest.get("verify_code_tree") is not True
        ):
            fail("code_dir/code_tree_sha256")
        if job.get("ckpt_dir") != os.path.dirname(job["checkpoint"]):
            fail("ckpt_dir")
        cfg_pkl = os.path.join(str(job.get("ckpt_dir") or ""), "config.pkl")
        if cfg_pkl not in pinned or str(job.get("config_sha256", "")).lower() != str(pkg_map.get(cfg_pkl, "")).lower():
            fail("config_sha256")
        if str(job.get("weights_sha256", "")).lower() != str(manifest.get("checkpoint_sha256", "")).lower():
            fail("weights_sha256")
        st_path = str(job.get("stats_file") or "")
        if st_path not in pinned or str(job.get("stats_sha256", "")).lower() != str(pkg_map.get(st_path, "")).lower():
            fail("stats_file")
        md_var = md15_variant(str(job.get("model") or ""))
        if os.path.basename(st_path) != os.path.basename(md_var["stats_rel"]):
            fail("stats_file")
        if job.get("arch") != md_var["arch"] or job.get("forcing_regrid") != md_var["forcing_regrid"]:
            fail("arch/forcing_regrid")
        if str(job.get("ic_file")) != str(manifest.get("ic_path")) or str(job.get("ic_sha256", "")).lower() != str(
            manifest.get("ic_sha256", "")
        ).lower():
            fail("ic_file")
        f_path = str(job.get("forcing_file") or "")
        f_rows = manifest.get("forcing_files") or []
        if f_path not in pinned and not any(str(r.get("path")) == f_path for r in f_rows):
            fail("forcing_file")
    elif job["family"] == "camulator":
        pkg_map = manifest.get("package_files_sha256") or {}
        if (
            str(job.get("code_dir") or "") != str(manifest.get("code_dir") or "")
            or str(job.get("code_tree_sha256") or "").lower() != str(manifest.get("code_tree_sha256") or "").lower()
            or job.get("code_tree_files") != manifest.get("code_tree_files")
            or bool(job.get("code_is_override", False)) != bool(manifest.get("code_is_override", False))
            or job.get("verify_code_tree") is not True
            or manifest.get("verify_code_tree") is not True
        ):
            fail("code_dir/code_tree_sha256")
        if str(job.get("checkpoint_sha256", "")).lower() != str(manifest.get("checkpoint_sha256", "")).lower():
            fail("checkpoint_sha256")
        for f_key, sha_key in (
            ("base_config_file", "base_config_sha256"),
            ("metadata_file", "metadata_sha256"),
            ("mean_file", "mean_sha256"),
            ("std_file", "std_sha256"),
            ("statics_b_file", "statics_b_sha256"),
            ("statics_be21_file", "statics_be21_sha256"),
            ("statics_fe21_file", "statics_fe21_sha256"),
        ):
            p_val = str(job.get(f_key) or "")
            if p_val not in pinned or str(job.get(sha_key, "")).lower() != str(pkg_map.get(p_val, "")).lower():
                fail(f_key)
        if str(job.get("ic_file")) != str(manifest.get("ic_path")) or str(job.get("ic_sha256", "")).lower() != str(
            manifest.get("ic_sha256", "")
        ).lower():
            fail("ic_file")
        f_rows = manifest.get("forcing_files") or []
        f_path = str(job.get("forcing_file") or "")
        f_sha = str(job.get("forcing_sha256") or "").lower()
        if len(f_rows) != 1 or str(f_rows[0].get("path")) != f_path or str(f_rows[0].get("sha256", "")).lower() != f_sha:
            fail("forcing_file")
    elif job["family"] == "cbottle":
        pkg_map = manifest.get("package_files_sha256") or {}
        experts = job.get("expert_checkpoints")
        if not isinstance(experts, list) or len(experts) != 3:
            fail("expert_checkpoints")
        for ex in experts:
            if not isinstance(ex, dict):
                fail("expert_checkpoints")
            ex_p = str(ex.get("path") or "")
            if ex_p not in pinned or str(ex.get("sha256", "")).lower() != str(pkg_map.get(ex_p, "")).lower():
                fail("expert_checkpoints")
        sst_p = str(job.get("sst_file") or "")
        if sst_p not in pinned or str(job.get("sst_sha256", "")).lower() != str(pkg_map.get(sst_p, "")).lower():
            fail("sst_file")
        if job.get("batch_size") != manifest.get("batch_size") or job.get("cadence") != manifest.get("cadence"):
            fail("batch_size/cadence")
        if (
            job.get("seed_scheme") != manifest.get("seed_scheme")
            or job.get("seed_scheme") != CBOTTLE_SEED_SCHEME
        ):
            fail("seed_scheme")
    elif job["family"] == "sdyff":
        pkg_map = manifest.get("package_files_sha256") or {}
        if (
            str(job.get("code_dir") or "") != str(manifest.get("code_dir") or "")
            or str(job.get("code_tree_sha256") or "").lower() != str(manifest.get("code_tree_sha256") or "").lower()
            or job.get("code_tree_files") != manifest.get("code_tree_files")
            or bool(job.get("code_is_override", False)) != bool(manifest.get("code_is_override", False))
            or job.get("verify_code_tree") is not True
            or manifest.get("verify_code_tree") is not True
        ):
            fail("code_dir/code_tree_sha256")
        for f_key, sha_key in (
            ("forecaster_ckpt", "forecaster_sha256"),
            ("interpolator_ckpt", "interpolator_sha256"),
            ("interpolator_cfg", "interpolator_cfg_sha256"),
        ):
            p_val = str(job.get(f_key) or "")
            if p_val not in pinned or str(job.get(sha_key, "")).lower() != str(pkg_map.get(p_val, "")).lower():
                fail(f_key)
        fv3_dir = str(job.get("fv3gfs_data_dir") or "")
        ref_shas = job.get("fv3gfs_file_sha256s") or {}
        if not isinstance(ref_shas, dict) or len(ref_shas) != len(SDYFF_ZENODO_FILES):
            fail("fv3gfs_file_sha256s")
        for fname in SDYFF_ZENODO_FILES:
            f_abs = os.path.join(fv3_dir, fname)
            if f_abs not in pinned or str(ref_shas.get(fname, "")).lower() != str(pkg_map.get(f_abs, "")).lower():
                fail("fv3gfs_file_sha256s")
    else:
        fail("family")


_FAMILY_SOURCE_OVERRIDE_ENV: dict[str, str] = {
    "md": "AIMIP_MD15_CODE",
    "camulator": "AIMIP_CAMULATOR_CODE",
    "sdyff": "AIMIP_SDYFF_CODE",
    "arches": "AIMIP_ARCHES_CODE",
}


def _verify_bound_source_tree_at_exec(job: Mapping[str, Any], manifest: Mapping[str, Any]) -> None:
    """Re-verify the manifest-bound source tree at job time and refuse unapproved ambient overrides."""
    fam = str(job["family"])
    env_var = _FAMILY_SOURCE_OVERRIDE_ENV.get(fam)
    if env_var is None:
        return
    ambient_ovr = (os.environ.get(env_var) or "").strip()
    is_ovr = bool(manifest.get("code_is_override", False))
    if is_ovr:
        approved_dir = validate_safe_path(
            str(manifest.get("code_dir") or ""),
            label=f"approved {env_var}",
        )
        if ambient_ovr and validate_safe_path(ambient_ovr, label=env_var) != approved_dir:
            raise ValidationError(
                f"Source override {env_var}={ambient_ovr!r} differs from approved plan directory {approved_dir!r}."
            )
        want_sha = str(manifest.get("code_tree_sha256") or "").strip().lower()
        want_files = manifest.get("code_tree_files")
        if len(want_sha) != 64 or not isinstance(want_files, int):
            raise ValidationError(
                f"Manifest does not pin tree digest/count for source override {approved_dir}."
            )
        if not os.path.isdir(approved_dir) or os.path.islink(approved_dir):
            raise ValidationError(
                f"Approved source override directory is missing or symlinked at job time: {approved_dir}"
            )
        if fam == "arches":
            from .setup_data import compute_arches_tree_digest

            got_sha, got_files = compute_arches_tree_digest(approved_dir)
        else:
            got_sha, got_files = compute_sha256sum_tree_digest(
                approved_dir,
                label=f"approved {env_var}",
            )
        if got_files != want_files or got_sha.lower() != want_sha:
            raise ValidationError(
                f"Approved source override tree changed at job time ({approved_dir}): "
                f"manifest expects {want_files} files / sha256={want_sha}, "
                f"got {got_files} files / sha256={got_sha}."
            )
        return

    if ambient_ovr:
        raise ValidationError(
            f"Refusing unapproved source override {env_var}={ambient_ovr!r} at execution time; "
            f"re-plan and re-prepare with {env_var} set."
        )
    if fam == "md":
        verify_md15_code_dir(str(job.get("data_root") or ""))
    elif fam == "camulator":
        verify_camulator_code_dir(str(job.get("data_root") or ""))
    elif fam == "sdyff":
        verify_sdyff_code_dir(
            str(job.get("data_dir") or ""),
            expected_tree_sha256=(
                str(manifest["code_tree_sha256"]) if manifest.get("code_tree_sha256") else None
            ),
            expected_tree_files=(
                int(manifest["code_tree_files"]) if manifest.get("code_tree_files") is not None else None
            ),
        )


def exec_emu(validated_out: str, manifest: dict[str, Any], manifest_path: str) -> dict[str, Any]:
    """Re-verify the pinned files, run the job in the second environment, record the outcome."""
    config_path = validate_run_path(f"{validated_out}/config.yaml", label="config.yaml")
    try:
        with open(config_path, encoding="utf-8") as fh:
            job = json.load(fh)
    except (OSError, json.JSONDecodeError) as exc:
        raise ConfigError(f"Failed to read {config_path}: {exc}") from exc
    if not isinstance(job, dict) or job.get("out_dir") != validated_out:
        raise ValidationError(f"Integrity check failed in {config_path}: out_dir does not match {validated_out}.")
    if job.get("checkpoint") != manifest.get("checkpoint_path") or job.get("family") != manifest.get("family"):
        raise ValidationError(f"Integrity check failed in {config_path}: checkpoint/family differ from the manifest.")
    want_cfg = str(manifest.get("config_sha256") or "")
    if len(want_cfg) != 64 or _config_sha(config_path).lower() != want_cfg.lower():
        raise ValidationError(f"Integrity check failed: {config_path} changed after `aimip run` prepared it.")
    _check_job_matches_manifest(job, manifest)

    pinned = dict(manifest.get("package_files_sha256") or {})
    if manifest.get("ic_path") is not None:
        pinned[str(manifest.get("ic_path"))] = str(manifest.get("ic_sha256"))
    if str(manifest.get("checkpoint_path")) not in pinned:
        raise ValidationError("Manifest does not pin the checkpoint.")
    for path, want in pinned.items():
        p = validate_safe_path(path, label="pinned file")
        if not os.path.isfile(p):
            raise ValidationError(f"Pinned file not found at job time: {p}")
        got = sha256_file(p).lower()
        if got != str(want).lower():
            raise ValidationError(f"SHA-256 mismatch at job time for {p}: manifest expects {want}, got {got}.")
    for row in manifest.get("forcing_files") or []:
        p = validate_safe_path(str(row.get("path")), label="scenario file")
        if not os.path.isfile(p) or os.path.islink(p) or os.path.getsize(p) != row.get("size"):
            raise ValidationError(f"Scenario file changed or missing at job time: {p}")
        want_scen_sha = str(row.get("sha256") or "").strip().lower()
        if len(want_scen_sha) != 64:
            raise ValidationError(f"Manifest does not pin SHA-256 for scenario file: {p}")
        got_scen_sha = sha256_file(p).lower()
        if got_scen_sha != want_scen_sha:
            raise ValidationError(
                f"SHA-256 mismatch at job time for scenario file {p}: "
                f"manifest expects {want_scen_sha}, got {got_scen_sha}."
            )
    if job["family"] == "samudra":
        win_dir = validate_safe_path(str(manifest.get("window_dir") or ""), label="staged Samudra OM4 window")
        want_win_sha = str(manifest.get("window_tree_sha256") or "").strip().lower()
        if len(want_win_sha) != 64:
            raise ValidationError("Manifest does not pin window_tree_sha256 for samudra-om4.")
        got_win_sha, got_win_files, got_win_bytes = compute_tree_digest(
            win_dir, label="staged Samudra OM4 window"
        )
        if (
            got_win_sha.lower() != want_win_sha
            or got_win_files != manifest.get("window_tree_files")
            or got_win_bytes != manifest.get("window_tree_bytes")
        ):
            raise ValidationError(
                f"Staged Samudra OM4 window changed at job time ({win_dir}): "
                f"manifest expects tree SHA-256 {want_win_sha}, got {got_win_sha}."
            )
    elif job["family"] in _FAMILY_SOURCE_OVERRIDE_ENV:
        _verify_bound_source_tree_at_exec(job, manifest)

    raw_job_id = os.environ.get("SLURM_JOB_ID")
    if raw_job_id:
        try:
            manifest["slurm_job_id"] = validate_slurm_job_id(raw_job_id.strip())
        except ValidationError:
            pass
    manifest["started_utc"] = _now_utc()
    _write_json_atomic(manifest_path, manifest)

    tmp_dir = os.path.join(validated_out, ".aimip-tmp")
    if os.path.islink(tmp_dir):
        raise ConfigError(f"Refusing a symlinked temporary directory: {tmp_dir}")
    os.makedirs(tmp_dir, mode=0o700, exist_ok=True)
    t0 = time.monotonic()
    rc = 1
    err: str | None = None
    try:
        py = _runtime_python_for_family(str(job["family"]))
        res = subprocess.run([py, "-I", "-m", EMU_JOB_MODULE, "run", validated_out],
                             env=_job_env(validated_out, py), check=False)
        rc = int(res.returncode)
        if rc == EXIT_UNSTABLE:
            err = "the rollout became unstable (non-finite or out-of-band month): see result.unstable_from"
        elif rc != 0:
            err = f"emulator job exited with rc={rc} (see slurm-*.err)"
    except BaseException as exc:   # noqa: BLE001 - recorded, then re-raised below
        err = f"{type(exc).__name__}: {exc}"
        rc = 1
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)
        manifest["finished"] = _now_utc()
        manifest["rc"] = rc
        manifest["wall_seconds"] = round(time.monotonic() - t0, 3)
        if err:
            manifest["error"] = err
        result_file = os.path.join(validated_out, "emu-result.json")
        if os.path.isfile(result_file) and not os.path.islink(result_file):
            try:
                with open(result_file, encoding="utf-8") as fh:
                    manifest["result"] = json.load(fh)
            except (OSError, json.JSONDecodeError):
                pass
        _write_json_atomic(manifest_path, manifest)
    if rc != 0:
        raise SystemExit(rc)
    return manifest
