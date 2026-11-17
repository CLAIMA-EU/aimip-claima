"""Model registry schema (`forcing_sets`, `fme_version`, coupled ICs), loader, and `${DATA}` / `${PLAYGROUND}` expansion."""

from __future__ import annotations

import dataclasses
import json
import os
import re
from collections.abc import Sequence
from typing import Any

from .errors import RegistryError, ValidationError
from .validators import (
    validate_forcing_set_name,
    validate_int_range,
    validate_iso_timestamp,
    validate_model_name,
    validate_python_module,
    validate_safe_path,
    validate_sha256_or_empty,
)

DEFAULT_REGISTRY_PATH = "/opt/aimip/registry.json"
PACKAGE_DEFAULT_REGISTRY_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "registry.default.json"
)
PLAYGROUND_PLACEHOLDER = "${PLAYGROUND}"
DATA_PLACEHOLDER = "${DATA}"

SUPPORTED_FAMILIES: tuple[str, ...] = ("fme", "e2s-dlesym", "arches", "geoarches")
ALLOWED_FME_MODULES: tuple[str, ...] = (
    "fme.ace.inference",
    "fme.coupled.inference",
    "aimip.emu_job",
)
SUPPORTED_FME_VERSIONS: tuple[str, ...] = ("2026.5.1", "2026.10.0")
SUPPORTED_CALENDARS: tuple[str, ...] = (
    "proleptic_gregorian",
    "julian",
    "noleap",
    "365_day",
    "360_day",
    "gregorian",
    "standard",
)

_SAFE_FILE_PATTERN_RE = re.compile(r"^[A-Za-z0-9._*+-]+$")

REQUIRED_MODEL_KEYS = frozenset(
    {
        "name",
        "fme_module",
        "checkpoint",
        "checkpoint_sha256",
        "ic_files",
        "forcing_sets",
        "timestep_hours",
        "stochastic",
        "default_n_ensemble_per_ic",
        "notes",
    }
)

REQUIRED_FORCING_SET_KEYS = frozenset({"name", "dir", "window"})

# Provenance of shipped initial-condition files per canonical model.
MODEL_IC_PROVENANCE: dict[str, str] = {
    "ace2-era5": (
        "Shipped ERA5 initial conditions: 12 monthly timestamps in 1979 (ic_1979.nc) "
        "and 12 monthly timestamps in 2001 (ic_2001.nc) from allenai/ACE2-ERA5."
    ),
    "ace2-shield": (
        "Single shipped IC (1979-01-01T00:00:00) shared from allenai/ACE2S-SHiELD-plus "
        "initial_conditions/AMIP/ic.nc (allenai/ACE2-SHiELD ships checkpoint only)."
    ),
    "ace2s-shield-plus": (
        "Single shipped IC (1979-01-01T00:00:00) from allenai/ACE2S-SHiELD-plus "
        "initial_conditions/AMIP/ic.nc."
    ),
    "ace2-eamv3": (
        "Shipped ic_1994.nc is allenai/ACE2-EAMv3 initial_conditions/1971010100.nc "
        "(1971-01-01 E3SM state) relabelled to noleap 1994-01-01T00:00:00; "
        "discard the first simulated year as spin-up."
    ),
    "hiro-ace": (
        "Shipped ic_1994.nc is allenai/HiRO-ACE initial_conditions/ic_2014.nc state 0 "
        "(2014-01-01 state) relabelled to 1994-01-01T00:00:00 for AIMIP's experimental "
        "1994-2014 extension outside the 2014-2022 training period; "
        "discard the first simulated year as spin-up."
    ),
    "ace2-som": (
        "Scenario-specific slab-ocean equilibrium ICs at 2032-01-01T00:00:00 from "
        "allenai/ACE2-SOM: initial_conditions/1xCO2/ic.nc (paired with 1xCO2 forcing) "
        "and initial_conditions/3xCO2/ic.nc (paired with 3xCO2 forcing)."
    ),
    "ace2s": (
        "HF-native global ACE2S ICs at 2014-01-01T00:00:00 (ic_2014.nc) and "
        "2023-01-01T00:00:00 (ic_2023.nc) from allenai/HiRO-ACE."
    ),
    "hiro-acev2": (
        "HF-native global ACE2S-X-SHiELD ICs at 2014-01-01T00:00:00 (ic_2014.nc) and "
        "2023-01-01T00:00:00 (ic_2023.nc) from allenai/HiRO-ACEv2."
    ),
    "samudrace-cm4": (
        "Paired coupled atmosphere + ocean piControl ICs from allenai/SamudrACE-CM4-piControl "
        "(0151-01-06T00:00:00 and 0311/0313/0315/0317/0319-01-01T00:00:00, staged at colon-free local paths)."
    ),
    "samudrace-e3smv3": (
        "Paired 3-member coupled atmosphere + ocean preindustrial control ICs "
        "(SamudrACE-E3SMv3-ICx3-train_{atmosphere,ocean}_ic.nc at 0425-01-03T12:00:00) "
        "from allenai/SamudrACE-E3SMv3."
    ),
    "dlesym-v1.0": "ERA5 state at --start fetched from ARCO-ERA5 at prepare time (no shipped IC).",
    "dlesym-v1.1": "ERA5 state at --start fetched from ARCO-ERA5 at prepare time (no shipped IC).",
    "dlesym-v0-isccp": "ERA5 state at --start fetched from ARCO-ERA5 at prepare time (no shipped IC).",
    "dlesym-aimip": "ERA5 state at --start (1978-10-03..07 00 UTC for the authors' five AIMIP members) fetched from "
                    "ARCO-ERA5 at prepare time (no shipped IC).",
    "archesclimate-ssp": "IPSL-CM6A-LR r1i1p1f1 scenario state of 2015-02 (previous state 2015-01) from gclyne/ArchesClimate-data.",
    "archesweather": "ERA5 daily-mean state of 1978-09-30 (previous state 1978-09-29) from Zenodo 22795233 era5_1978_00h_sliced.nc.",
    "archesweathergen": "ERA5 daily-mean state of 1978-09-30 (previous state 1978-09-29) from Zenodo 22795233 era5_1978_00h_sliced.nc.",
}

SAMUDRACE_CM4_IC_SLUGS: tuple[tuple[str, str], ...] = (
    ("0311-01-01T00:00:00", "0311-01-01T000000"),
    ("0313-01-01T00:00:00", "0313-01-01T000000"),
    ("0315-01-01T00:00:00", "0315-01-01T000000"),
    ("0317-01-01T00:00:00", "0317-01-01T000000"),
    ("0319-01-01T00:00:00", "0319-01-01T000000"),
    ("0151-01-06T00:00:00", "0151-01-06T000000"),
)


def get_model_ic_provenance(model_name: str, notes: str = "") -> str:
    """Return scientific provenance note for a model's shipped initial conditions."""
    return MODEL_IC_PROVENANCE.get(model_name, notes)


def _validate_file_pattern(value: str, *, label: str) -> str:
    """Validate a safe NetCDF filename/glob pattern (no directory traversal or shell chars)."""
    if not isinstance(value, str) or not value or ".." in value or "/" in value:
        raise RegistryError(
            f"{label}: file pattern must be a non-empty filename/glob without '/' or '..'."
        )
    if not _SAFE_FILE_PATTERN_RE.match(value):
        raise RegistryError(
            f"{label}: invalid characters in file pattern {value!r}."
        )
    return value


def _validate_rel_lock_path(value: str, *, label: str) -> str:
    """Validate a canonical relative lock path under `$AIMIP_DATA`."""
    if not isinstance(value, str) or not value or value.startswith("/"):
        raise RegistryError(f"{label}: expected relative path under $AIMIP_DATA, got {value!r}.")
    parts = value.split("/")
    if any(p in ("", ".", "..") for p in parts):
        raise RegistryError(f"{label}: invalid relative path {value!r}.")
    try:
        validate_safe_path("/" + value, label=label)
    except ValidationError as exc:
        raise RegistryError(str(exc)) from exc
    return value


@dataclasses.dataclass(frozen=True)
class ForcingSet:
    """A contiguous forcing dataset directory covering `window = (first, last)`.

    Optional coupled / scenario fields (`file_pattern`, `ocean_dir`, `ocean_file_pattern`,
    `ic_files`, `ocean_ic_files`, `repeat_period_hours`, `repeat_base_start`) default to
    `None`/`()` so single-component registries serialize and parse unchanged.
    """

    name: str
    dir: str
    window: tuple[str, str]
    file_pattern: str | None = None
    ocean_dir: str | None = None
    ocean_file_pattern: str | None = None
    ic_files: tuple[str, ...] = ()
    ocean_ic_files: tuple[str, ...] = ()
    repeat_period_hours: int | None = None
    repeat_base_start: str | None = None

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "name": self.name,
            "dir": self.dir,
            "window": [self.window[0], self.window[1]],
        }
        if self.file_pattern is not None:
            out["file_pattern"] = self.file_pattern
        if self.ocean_dir is not None:
            out["ocean_dir"] = self.ocean_dir
        if self.ocean_file_pattern is not None:
            out["ocean_file_pattern"] = self.ocean_file_pattern
        if self.ic_files:
            out["ic_files"] = list(self.ic_files)
        if self.ocean_ic_files:
            out["ocean_ic_files"] = list(self.ocean_ic_files)
        if self.repeat_period_hours is not None:
            out["repeat_period_hours"] = self.repeat_period_hours
        if self.repeat_base_start is not None:
            out["repeat_base_start"] = self.repeat_base_start
        return out


@dataclasses.dataclass(frozen=True)
class ModelSpec:
    """Validated specification for an fme-family model in the playground registry."""

    name: str
    fme_module: str
    checkpoint: str
    checkpoint_sha256: str
    ic_files: tuple[str, ...]
    forcing_sets: tuple[ForcingSet, ...]
    timestep_hours: int
    stochastic: bool
    default_n_ensemble_per_ic: int
    notes: str
    family: str = "fme"
    fme_version: str = "2026.5.1"
    calendar: str = "proleptic_gregorian"
    ocean_ic_files: tuple[str, ...] = ()
    required_files: tuple[str, ...] = ()

    @property
    def is_coupled(self) -> bool:
        """Return True when the model executes via `fme.coupled.inference`."""
        return self.fme_module == "fme.coupled.inference"

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a JSON-compatible dictionary."""
        out: dict[str, Any] = {
            "name": self.name,
            "family": self.family,
            "fme_version": self.fme_version,
            "fme_module": self.fme_module,
            "calendar": self.calendar,
            "checkpoint": self.checkpoint,
            "checkpoint_sha256": self.checkpoint_sha256,
            "ic_files": list(self.ic_files),
            "forcing_sets": [fs.to_dict() for fs in self.forcing_sets],
            "timestep_hours": self.timestep_hours,
            "stochastic": self.stochastic,
            "default_n_ensemble_per_ic": self.default_n_ensemble_per_ic,
            "notes": self.notes,
        }
        if self.ocean_ic_files:
            out["ocean_ic_files"] = list(self.ocean_ic_files)
        if self.required_files:
            out["required_files"] = list(self.required_files)
        return out


def resolve_playground_dir(explicit_playground: str | None = None) -> str | None:
    """Resolve and validate `$AIMIP_PLAYGROUND` (or `$AIMIP_HOME`) if set."""
    raw = (
        explicit_playground
        if explicit_playground is not None
        else (os.environ.get("AIMIP_PLAYGROUND") or os.environ.get("AIMIP_HOME"))
    )
    if raw is None or raw == "":
        return None
    return validate_safe_path(raw, label="AIMIP_PLAYGROUND")


def resolve_data_dir(
    explicit_data: str | None = None,
    playground_dir: str | None = None,
) -> str | None:
    """Resolve and validate `$AIMIP_DATA`, falling back to `playground_dir` when unset."""
    raw = explicit_data if explicit_data is not None else os.environ.get("AIMIP_DATA")
    if raw is not None and raw != "":
        return validate_safe_path(raw, label="AIMIP_DATA")
    return resolve_playground_dir(playground_dir)


def resolve_registry_path(
    explicit_path: str | None = None,
    data_dir: str | None = None,
) -> str:
    """Resolve registry path from `--registry`, `$AIMIP_REGISTRY`, `$AIMIP_DATA/registry.json`, or packaged `registry.default.json`."""
    if explicit_path:
        return validate_safe_path(explicit_path, label="registry path")

    env_reg = os.environ.get("AIMIP_REGISTRY")
    if env_reg:
        return validate_safe_path(env_reg, label="AIMIP_REGISTRY")

    raw_data = data_dir if data_dir is not None else os.environ.get("AIMIP_DATA")
    if raw_data:
        validated_data = validate_safe_path(raw_data, label="AIMIP_DATA").rstrip("/")
        data_reg = validate_safe_path(
            f"{validated_data}/registry.json", label="AIMIP_DATA/registry.json"
        )
        if os.path.isfile(data_reg):
            return data_reg

    for env_var in ("AIMIP_HOME", "AIMIP_PLAYGROUND"):
        env_root = os.environ.get(env_var)
        if env_root:
            validated_root = validate_safe_path(env_root, label=env_var).rstrip("/")
            cand = validate_safe_path(
                f"{validated_root}/registry.json", label=f"{env_var}/registry.json"
            )
            if os.path.isfile(cand):
                return cand

    if os.path.isfile(DEFAULT_REGISTRY_PATH):
        return validate_safe_path(DEFAULT_REGISTRY_PATH, label="registry path")

    if os.path.isfile(PACKAGE_DEFAULT_REGISTRY_PATH):
        return validate_safe_path(
            PACKAGE_DEFAULT_REGISTRY_PATH, label="packaged registry.default.json"
        )

    return validate_safe_path(DEFAULT_REGISTRY_PATH, label="registry path")


def _expand_registry_path(
    raw_path: str,
    *,
    playground_dir: str | None,
    data_dir: str | None,
    label: str,
) -> str:
    """Expand `${DATA}` and `${PLAYGROUND}` in *raw_path* and strictly validate the resulting path."""
    if not isinstance(raw_path, str) or not raw_path:
        raise RegistryError(f"Registry field {label} must be a non-empty path string.")
    expanded = raw_path
    if DATA_PLACEHOLDER in expanded:
        if not data_dir:
            raise RegistryError(
                f"Registry field {label} ({raw_path!r}) contains {DATA_PLACEHOLDER}, "
                "but neither $AIMIP_DATA nor $AIMIP_PLAYGROUND is set."
            )
        base_data = data_dir.rstrip("/") or "/"
        expanded = expanded.replace(DATA_PLACEHOLDER, base_data)
    if PLAYGROUND_PLACEHOLDER in expanded:
        if not playground_dir:
            raise RegistryError(
                f"Registry field {label} ({raw_path!r}) contains {PLAYGROUND_PLACEHOLDER}, "
                "but $AIMIP_PLAYGROUND is not set."
            )
        base_pg = playground_dir.rstrip("/") or "/"
        expanded = expanded.replace(PLAYGROUND_PLACEHOLDER, base_pg)
    try:
        return validate_safe_path(expanded, label=label)
    except ValidationError as exc:
        raise RegistryError(str(exc)) from exc


def _parse_forcing_set(
    raw_fs: Any,
    model_name: str,
    idx: int,
    playground_dir: str | None,
    data_dir: str | None,
    *,
    is_coupled: bool = False,
) -> ForcingSet:
    label_prefix = f"{model_name}.forcing_sets[{idx}]"
    if not isinstance(raw_fs, dict):
        raise RegistryError(f"{label_prefix} must be an object.")
    missing = REQUIRED_FORCING_SET_KEYS - set(raw_fs.keys())
    if missing:
        raise RegistryError(
            f"{label_prefix} missing required field(s): {', '.join(sorted(missing))}."
        )
    try:
        fs_name = validate_forcing_set_name(raw_fs["name"])
    except ValidationError as exc:
        raise RegistryError(f"{label_prefix}.name: {exc}") from exc

    fs_dir = _expand_registry_path(
        raw_fs["dir"],
        playground_dir=playground_dir,
        data_dir=data_dir,
        label=f"{label_prefix}.dir",
    )
    raw_win = raw_fs["window"]
    if not isinstance(raw_win, (list, tuple)) or len(raw_win) != 2:
        raise RegistryError(
            f"{label_prefix}.window must be a 2-element [first, last] ISO list."
        )
    try:
        w_start = validate_iso_timestamp(raw_win[0], label=f"{label_prefix}.window[0]")
        w_end = validate_iso_timestamp(raw_win[1], label=f"{label_prefix}.window[1]")
    except ValidationError as exc:
        raise RegistryError(str(exc)) from exc
    if w_start > w_end:
        raise RegistryError(
            f"{label_prefix}: window start {w_start} is after window end {w_end}."
        )

    file_pattern: str | None = None
    if raw_fs.get("file_pattern") is not None:
        file_pattern = _validate_file_pattern(
            raw_fs["file_pattern"], label=f"{label_prefix}.file_pattern"
        )

    ocean_dir: str | None = None
    if raw_fs.get("ocean_dir") is not None:
        ocean_dir = _expand_registry_path(
            raw_fs["ocean_dir"],
            playground_dir=playground_dir,
            data_dir=data_dir,
            label=f"{label_prefix}.ocean_dir",
        )

    ocean_file_pattern: str | None = None
    if raw_fs.get("ocean_file_pattern") is not None:
        ocean_file_pattern = _validate_file_pattern(
            raw_fs["ocean_file_pattern"], label=f"{label_prefix}.ocean_file_pattern"
        )

    fs_ic_files: tuple[str, ...] = ()
    if raw_fs.get("ic_files") is not None:
        raw_fs_ics = raw_fs["ic_files"]
        if not isinstance(raw_fs_ics, list) or len(raw_fs_ics) == 0:
            raise RegistryError(
                f"{label_prefix}.ic_files must be a non-empty list when provided."
            )
        fs_ic_files = tuple(
            _expand_registry_path(
                p,
                playground_dir=playground_dir,
                data_dir=data_dir,
                label=f"{label_prefix}.ic_files[{i}]",
            )
            for i, p in enumerate(raw_fs_ics)
        )

    fs_ocean_ic_files: tuple[str, ...] = ()
    if raw_fs.get("ocean_ic_files") is not None:
        raw_fs_oics = raw_fs["ocean_ic_files"]
        if not isinstance(raw_fs_oics, list) or len(raw_fs_oics) == 0:
            raise RegistryError(
                f"{label_prefix}.ocean_ic_files must be a non-empty list when provided."
            )
        fs_ocean_ic_files = tuple(
            _expand_registry_path(
                p,
                playground_dir=playground_dir,
                data_dir=data_dir,
                label=f"{label_prefix}.ocean_ic_files[{i}]",
            )
            for i, p in enumerate(raw_fs_oics)
        )

    if is_coupled and fs_ic_files and len(fs_ocean_ic_files) != len(fs_ic_files):
        raise RegistryError(
            f"{label_prefix}: coupled forcing_set with ic_files ({len(fs_ic_files)}) "
            f"must provide matching ocean_ic_files ({len(fs_ocean_ic_files)})."
        )

    repeat_period_hours: int | None = None
    if raw_fs.get("repeat_period_hours") is not None:
        try:
            repeat_period_hours = validate_int_range(
                raw_fs["repeat_period_hours"],
                f"{label_prefix}.repeat_period_hours",
                min_value=6,
                max_value=24 * 366 * 100,
            )
        except ValidationError as exc:
            raise RegistryError(str(exc)) from exc

    repeat_base_start: str | None = None
    if raw_fs.get("repeat_base_start") is not None:
        try:
            repeat_base_start = validate_iso_timestamp(
                raw_fs["repeat_base_start"], label=f"{label_prefix}.repeat_base_start"
            )
        except ValidationError as exc:
            raise RegistryError(str(exc)) from exc

    return ForcingSet(
        name=fs_name,
        dir=fs_dir,
        window=(w_start, w_end),
        file_pattern=file_pattern,
        ocean_dir=ocean_dir,
        ocean_file_pattern=ocean_file_pattern,
        ic_files=fs_ic_files,
        ocean_ic_files=fs_ocean_ic_files,
        repeat_period_hours=repeat_period_hours,
        repeat_base_start=repeat_base_start,
    )


def _parse_model_entry(
    entry: dict[str, Any],
    playground_dir: str | None,
    data_dir: str | None,
) -> ModelSpec:
    """Validate a single raw model dictionary into a `ModelSpec`."""
    if not isinstance(entry, dict):
        raise RegistryError(f"Model entry must be an object, got {type(entry).__name__}.")

    missing = REQUIRED_MODEL_KEYS - set(entry.keys())
    if missing:
        shown = ", ".join(sorted(missing))
        raise RegistryError(
            f"Model entry {entry.get('name', '<unnamed>')!r} missing required field(s): {shown}."
        )

    try:
        name = validate_model_name(entry["name"])
        fme_module = validate_python_module(entry["fme_module"], label=f"{name}.fme_module")
        checkpoint_sha256 = validate_sha256_or_empty(
            entry["checkpoint_sha256"], label=f"{name}.checkpoint_sha256"
        )
        timestep_hours = validate_int_range(
            entry["timestep_hours"],
            f"{name}.timestep_hours",
            min_value=1,
            max_value=24 * 366,
        )
        default_n_ens = validate_int_range(
            entry["default_n_ensemble_per_ic"],
            f"{name}.default_n_ensemble_per_ic",
            min_value=1,
            max_value=10_000,
        )
    except ValidationError as exc:
        raise RegistryError(str(exc)) from exc

    family = entry.get("family", "fme")
    if not isinstance(family, str) or family not in SUPPORTED_FAMILIES:
        raise RegistryError(
            f"Model {name!r}: unsupported 'family' {family!r} "
            f"(supported: {', '.join(SUPPORTED_FAMILIES)}).",
            field="family",
            allowed=SUPPORTED_FAMILIES,
        )
    raw_adapter = entry.get("adapter")
    if raw_adapter is not None and (
        not isinstance(raw_adapter, str) or raw_adapter not in (*SUPPORTED_FAMILIES, "emu")
    ):
        raise RegistryError(
            f"Model {name!r}: unsupported 'adapter' {raw_adapter!r}; registry JSON may name only installed adapter IDs.",
            field="adapter",
            allowed=(*SUPPORTED_FAMILIES, "emu"),
        )
    if fme_module not in ALLOWED_FME_MODULES:
        raise RegistryError(
            f"Model {name!r}: disallowed 'fme_module' {fme_module!r}; arbitrary module imports are refused.",
            field="fme_module",
            allowed=ALLOWED_FME_MODULES,
        )
    if family == "fme" and fme_module not in ("fme.ace.inference", "fme.coupled.inference"):
        raise RegistryError(
            f"Model {name!r}: family 'fme' must use 'fme.ace.inference' or 'fme.coupled.inference', got {fme_module!r}.",
            field="fme_module",
            allowed=("fme.ace.inference", "fme.coupled.inference"),
        )

    fme_version = entry.get("fme_version", "2026.5.1")
    if not isinstance(fme_version, str) or fme_version not in SUPPORTED_FME_VERSIONS:
        raise RegistryError(
            f"Model {name!r}: unsupported 'fme_version' {fme_version!r} "
            f"(supported: {', '.join(SUPPORTED_FME_VERSIONS)})."
        )

    calendar = entry.get("calendar", "proleptic_gregorian")
    if not isinstance(calendar, str) or calendar not in SUPPORTED_CALENDARS:
        raise RegistryError(
            f"Model {name!r}: unsupported 'calendar' {calendar!r} "
            f"(supported: {', '.join(SUPPORTED_CALENDARS)})."
        )

    stochastic = entry["stochastic"]
    if not isinstance(stochastic, bool):
        raise RegistryError(f"Model {name!r}: 'stochastic' must be a boolean.")
    if not stochastic and default_n_ens != 1:
        raise RegistryError(
            f"Model {name!r}: deterministic models must have default_n_ensemble_per_ic == 1."
        )

    raw_ics = entry["ic_files"]
    if not isinstance(raw_ics, list) or (len(raw_ics) == 0 and family == "fme"):
        raise RegistryError(f"Model {name!r}: 'ic_files' must be a non-empty list of paths.")
    ic_files = tuple(
        _expand_registry_path(
            p,
            playground_dir=playground_dir,
            data_dir=data_dir,
            label=f"{name}.ic_files[{idx}]",
        )
        for idx, p in enumerate(raw_ics)
    )

    is_coupled = fme_module == "fme.coupled.inference"
    raw_ocean_ics = entry.get("ocean_ic_files")
    ocean_ic_files: tuple[str, ...] = ()
    if raw_ocean_ics is not None:
        if not isinstance(raw_ocean_ics, list):
            raise RegistryError(f"Model {name!r}: 'ocean_ic_files' must be a list of paths.")
        ocean_ic_files = tuple(
            _expand_registry_path(
                p,
                playground_dir=playground_dir,
                data_dir=data_dir,
                label=f"{name}.ocean_ic_files[{idx}]",
            )
            for idx, p in enumerate(raw_ocean_ics)
        )
    if family != "fme" and (is_coupled or fme_module != "aimip.emu_job"):
        raise RegistryError(f"Model {name!r}: family {family!r} must use fme_module 'aimip.emu_job'.")
    if is_coupled and len(ocean_ic_files) != len(ic_files):
        raise RegistryError(
            f"Model {name!r}: coupled model (`fme.coupled.inference`) must define "
            f"'ocean_ic_files' matching 'ic_files' length ({len(ic_files)}), "
            f"got {len(ocean_ic_files)}."
        )
    if not is_coupled and ocean_ic_files:
        raise RegistryError(
            f"Model {name!r}: 'ocean_ic_files' is only valid for coupled models."
        )

    raw_req_files = entry.get("required_files")
    required_files: tuple[str, ...] = ()
    if raw_req_files is not None:
        if not isinstance(raw_req_files, list):
            raise RegistryError(f"Model {name!r}: 'required_files' must be a list of relative paths.")
        required_files = tuple(
            _validate_rel_lock_path(p, label=f"{name}.required_files[{idx}]")
            for idx, p in enumerate(raw_req_files)
        )

    checkpoint = _expand_registry_path(
        entry["checkpoint"],
        playground_dir=playground_dir,
        data_dir=data_dir,
        label=f"{name}.checkpoint",
    )

    raw_sets = entry["forcing_sets"]
    if not isinstance(raw_sets, list) or len(raw_sets) == 0:
        raise RegistryError(
            f"Model {name!r}: 'forcing_sets' must be a non-empty list of forcing set objects."
        )
    parsed_sets: list[ForcingSet] = []
    seen_set_names: set[str] = set()
    for idx, raw_fs in enumerate(raw_sets):
        fs = _parse_forcing_set(
            raw_fs, name, idx, playground_dir, data_dir, is_coupled=is_coupled
        )
        if fme_version == "2026.10.0" and fs.repeat_period_hours is not None:
            raise RegistryError(
                f"Model {name!r}: forcing_set {fs.name!r} cannot use repeat_period_hours "
                "on fme 2026.10.0 (XarrayDataConfig.n_repeats was removed in 2026.10.0)."
            )
        if fs.name in seen_set_names:
            raise RegistryError(
                f"Model {name!r}: duplicate forcing_set name {fs.name!r}."
            )
        seen_set_names.add(fs.name)
        parsed_sets.append(fs)

    notes = entry["notes"]
    if not isinstance(notes, str):
        raise RegistryError(f"Model {name!r}: 'notes' must be a string.")

    return ModelSpec(
        name=name,
        fme_module=fme_module,
        checkpoint=checkpoint,
        checkpoint_sha256=checkpoint_sha256,
        ic_files=ic_files,
        forcing_sets=tuple(parsed_sets),
        timestep_hours=timestep_hours,
        stochastic=stochastic,
        default_n_ensemble_per_ic=default_n_ens,
        notes=notes,
        family=family,
        fme_version=fme_version,
        calendar=calendar,
        ocean_ic_files=ocean_ic_files,
        required_files=required_files,
    )


def load_registry(
    registry_path: str | None = None,
    playground_dir: str | None = None,
    data_dir: str | None = None,
) -> dict[str, ModelSpec]:
    """Load and validate all models from the JSON registry."""
    resolved_path = resolve_registry_path(registry_path, data_dir=data_dir)
    resolved_playground = resolve_playground_dir(playground_dir)
    resolved_data = resolve_data_dir(data_dir, playground_dir=resolved_playground)

    if not os.path.isfile(resolved_path):
        raise RegistryError(f"Registry file not found: {resolved_path}")

    try:
        with open(resolved_path, encoding="utf-8") as fh:
            payload = json.load(fh)
    except (OSError, json.JSONDecodeError) as exc:
        raise RegistryError(f"Failed to parse registry {resolved_path}: {exc}") from exc

    raw_models: list[dict[str, Any]]
    if isinstance(payload, list):
        raw_models = payload
    elif isinstance(payload, dict) and "models" in payload:
        models_field = payload["models"]
        if isinstance(models_field, list):
            raw_models = models_field
        elif isinstance(models_field, dict):
            raw_models = [
                ({"name": k, **v} if isinstance(v, dict) and "name" not in v else v)
                for k, v in models_field.items()
            ]
        else:
            raise RegistryError("Registry 'models' field must be a list or mapping.")
    else:
        raise RegistryError(
            f"Invalid registry top-level structure in {resolved_path}: expected object with 'models' list."
        )

    # An explicitly empty `"models": []` registry is valid and occurs when `aimip setup`
    # invalidates all previously published models; keeping the file on disk prevents
    # `resolve_registry_path` from falling back to the packaged `registry.default.json`.
    catalog: dict[str, ModelSpec] = {}
    for entry in raw_models:
        spec = _parse_model_entry(entry, resolved_playground, resolved_data)
        if spec.name in catalog:
            raise RegistryError(f"Duplicate model name {spec.name!r} in {resolved_path}.")
        catalog[spec.name] = spec

    return catalog


def get_model(
    model_name: str,
    registry_path: str | None = None,
    playground_dir: str | None = None,
    data_dir: str | None = None,
) -> ModelSpec:
    """Validate *model_name* and fetch its `ModelSpec` from the registry."""
    catalog = load_registry(
        registry_path=registry_path,
        playground_dir=playground_dir,
        data_dir=data_dir,
    )
    validated = validate_model_name(model_name, allowed=set(catalog.keys()))
    return catalog[validated]


def build_default_registry_payload(
    selected_models: Sequence[str] | None = None,
    *,
    root_placeholder: str = DATA_PLACEHOLDER,
) -> dict[str, Any]:
    """Construct the canonical registry JSON payload (10 fme models + 7 DLESyM/ArchesClimate/ArchesWeather models) with `${DATA}/...` paths and SHA-256s."""
    p = root_placeholder.rstrip("/")

    era5_hist_rels = [f"inputs/ace2-era5/amip-hist/forcing_{y}.nc" for y in range(1979, 2015)]
    shield_hist_rels = [f"inputs/shield/amip-hist/forcing_{y}.nc" for y in range(1979, 2015)]
    shield_oos_rels = [f"inputs/shield/amip-oos/forcing_{y}.nc" for y in range(2014, 2025)]
    eam_hist_rels = [f"inputs/ace2-eamv3/amip-hist/{y}.nc" for y in range(1994, 2015)]
    hiro_hist_rels = [f"inputs/hiro-ace/amip-hist/forcing_{y}.nc" for y in range(1994, 2015)]
    hiro_oos_rels = [f"inputs/hiro-ace/amip-oos/forcing_{y}.nc" for y in range(2014, 2025)]

    som_1x_rels = [f"inputs/ace2-som/1xCO2/forcing_{y}.nc" for y in range(2032, 2036)]
    som_3x_rels = [f"inputs/ace2-som/3xCO2/forcing_{y}.nc" for y in range(2032, 2036)]
    ace2s_forcing_rels = [f"inputs/ace2s/amip-2014-2023/forcing_{y}.nc" for y in range(2014, 2024)]
    hiro2_forcing_rels = [
        f"inputs/hiro-acev2/amip-2014-2023/forcing_{y}.nc" for y in range(2014, 2024)
    ]

    cm4_atmos_ics_rel = [
        f"inputs/samudrace-cm4/ic/{slug}/atmosphere/initial_condition.nc"
        for _, slug in SAMUDRACE_CM4_IC_SLUGS
    ]
    cm4_ocean_ics_rel = [
        f"inputs/samudrace-cm4/ic/{slug}/ocean/initial_condition.nc"
        for _, slug in SAMUDRACE_CM4_IC_SLUGS
    ]
    cm4_0311_slugs = [slug for ts, slug in SAMUDRACE_CM4_IC_SLUGS if not ts.startswith("0151")]

    all_entries: list[dict[str, Any]] = [
        {
            "name": "ace2-era5",
            "family": "fme",
            "fme_version": "2026.5.1",
            "fme_module": "fme.ace.inference",
            "calendar": "proleptic_gregorian",
            "checkpoint": f"{p}/models/ace2-era5/ace2_era5_ckpt.tar",
            "checkpoint_sha256": "0e2522df5a8db087f18bcebea6e88277cea7c3b71892b868c427aca27673edc2",
            "ic_files": [
                f"{p}/inputs/ace2-era5/ic/ic_1979.nc",
                f"{p}/inputs/ace2-era5/ic/ic_2001.nc",
            ],
            "ic_sha256": {
                f"{p}/inputs/ace2-era5/ic/ic_1979.nc": "a4e8c13f281bcefd759a18fd5aac21b7af3524fa19ed9b5b53e22f32598e8793",
                f"{p}/inputs/ace2-era5/ic/ic_2001.nc": "e80ba139af21ad77bf3e5818a6ae54088a5cdc1a09cd1c1d668dee9eed7636bf",
            },
            "forcing_sets": [
                {
                    "name": "amip-hist",
                    "dir": f"{p}/inputs/ace2-era5/amip-hist",
                    "window": ["1979-01-01T00:00:00", "2014-12-31T18:00:00"],
                }
            ],
            "required_files": [
                "models/ace2-era5/ace2_era5_ckpt.tar",
                "inputs/ace2-era5/ic/ic_1979.nc",
                "inputs/ace2-era5/ic/ic_2001.nc",
                *era5_hist_rels,
            ],
            "timestep_hours": 6,
            "stochastic": False,
            "default_n_ensemble_per_ic": 1,
            "notes": "ACE2-ERA5 (Ai2), deterministic, trained on ERA5 1979-2008/2012-2019; shipped ICs: 12 monthly times in 1979 and 2001.",
        },
        {
            "name": "ace2-shield",
            "family": "fme",
            "fme_version": "2026.5.1",
            "fme_module": "fme.ace.inference",
            "calendar": "julian",
            "checkpoint": f"{p}/models/ace2-shield/ace2_shield_ckpt.tar",
            "checkpoint_sha256": "2debfee21a4cee81f34631c81539a7f312b9ea67a73cebd0f9b00cbe3cc23482",
            "ic_files": [f"{p}/models/ace2s-shield-plus/ic.nc"],
            "ic_sha256": {
                f"{p}/models/ace2s-shield-plus/ic.nc": "3263356a1ff67726a96a5c42e93dc0579d49dc48a35328ea231910b95ef70120"
            },
            "forcing_sets": [
                {
                    "name": "amip-hist",
                    "dir": f"{p}/inputs/shield/amip-hist",
                    "window": ["1979-01-01T00:00:00", "2014-12-31T18:00:00"],
                },
                {
                    "name": "amip-oos",
                    "dir": f"{p}/inputs/shield/amip-oos",
                    "window": ["2014-01-01T00:00:00", "2024-12-31T18:00:00"],
                },
            ],
            "required_files": [
                "models/ace2-shield/ace2_shield_ckpt.tar",
                "models/ace2s-shield-plus/ic.nc",
                *shield_hist_rels,
                *shield_oos_rels,
            ],
            "timestep_hours": 6,
            "stochastic": False,
            "default_n_ensemble_per_ic": 1,
            "notes": "ACE2-SHiELD (Ai2), deterministic, trained on SHiELD; single shipped IC (1979-01-01) shared from ACE2S-SHiELD-plus AMIP package; amip-oos = AIMIP SST/SIC 2015-2024 (Zenodo 17065758).",
        },
        {
            "name": "ace2s-shield-plus",
            "family": "fme",
            "fme_version": "2026.5.1",
            "fme_module": "fme.ace.inference",
            "calendar": "julian",
            "checkpoint": f"{p}/models/ace2s-shield-plus/ace2s_shield_plus_ckpt.tar",
            "checkpoint_sha256": "4d11bf3ceb714d2c0ca971a5a7c1304efeab01f70edfe8a318e7816af0fbab25",
            "ic_files": [f"{p}/models/ace2s-shield-plus/ic.nc"],
            "ic_sha256": {
                f"{p}/models/ace2s-shield-plus/ic.nc": "3263356a1ff67726a96a5c42e93dc0579d49dc48a35328ea231910b95ef70120"
            },
            "forcing_sets": [
                {
                    "name": "amip-hist",
                    "dir": f"{p}/inputs/shield/amip-hist",
                    "window": ["1979-01-01T00:00:00", "2014-12-31T18:00:00"],
                },
                {
                    "name": "amip-oos",
                    "dir": f"{p}/inputs/shield/amip-oos",
                    "window": ["2014-01-01T00:00:00", "2024-12-31T18:00:00"],
                },
            ],
            "required_files": [
                "models/ace2s-shield-plus/ace2s_shield_plus_ckpt.tar",
                "models/ace2s-shield-plus/ic.nc",
                *shield_hist_rels,
                *shield_oos_rels,
            ],
            "timestep_hours": 6,
            "stochastic": True,
            "default_n_ensemble_per_ic": 1,
            "notes": "ACE2S-SHiELD-plus (Ai2), stochastic (ensemble members via IC replication along sample + model noise); shipped IC: 1979-01-01 from AMIP package.",
        },
        {
            "name": "ace2-eamv3",
            "family": "fme",
            "fme_version": "2026.5.1",
            "fme_module": "fme.ace.inference",
            "calendar": "noleap",
            "checkpoint": f"{p}/models/ace2-eamv3/ace2_EAMv3_ckpt.tar",
            "checkpoint_sha256": "99c0bc44966394aab2a7bc91241893a115c163b57965af1717c5ac0e9bbf0e06",
            "ic_files": [f"{p}/inputs/ace2-eamv3/ic/ic_1994.nc"],
            "ic_sha256": {
                f"{p}/inputs/ace2-eamv3/ic/ic_1994.nc": "b4f7214a0069d911ed941c3c415b0931db72f22e47b316ea6e80846cff34c875"
            },
            "forcing_sets": [
                {
                    "name": "amip-hist",
                    "dir": f"{p}/inputs/ace2-eamv3/amip-hist",
                    "window": ["1994-01-01T00:00:00", "2014-12-31T18:00:00"],
                }
            ],
            "required_files": [
                "models/ace2-eamv3/ace2_EAMv3_ckpt.tar",
                "inputs/ace2-eamv3/ic/ic_1994.nc",
                *eam_hist_rels,
            ],
            "timestep_hours": 6,
            "stochastic": False,
            "default_n_ensemble_per_ic": 1,
            "notes": "ACE2-EAMv3 (Ai2), deterministic, trained on E3SMv3 EAM; noleap calendar; window 1994-2014 only. Shipped ic_1994.nc is an E3SM 1971-01-01 state relabelled to 1994-01-01; discard the first simulated year as spin-up.",
        },
        {
            "name": "hiro-ace",
            "family": "fme",
            "fme_version": "2026.5.1",
            "fme_module": "fme.ace.inference",
            "calendar": "julian",
            "checkpoint": f"{p}/models/hiro-ace/ACE2S.ckpt",
            "checkpoint_sha256": "0b39d182074c02c92d6700088dfe5f6f1ae9488d50eb4a260564121ec2dcbf0e",
            "ic_files": [f"{p}/inputs/hiro-ace/ic/ic_1994.nc"],
            "ic_sha256": {
                f"{p}/inputs/hiro-ace/ic/ic_1994.nc": "4ed27e044c0a7d9c653655853adce573b41dd45e09d6d68b544e573b9778a7be"
            },
            "forcing_sets": [
                {
                    "name": "amip-hist",
                    "dir": f"{p}/inputs/hiro-ace/amip-hist",
                    "window": ["1994-01-01T00:00:00", "2014-12-31T18:00:00"],
                },
                {
                    "name": "amip-oos",
                    "dir": f"{p}/inputs/hiro-ace/amip-oos",
                    "window": ["2014-01-01T00:00:00", "2024-12-31T18:00:00"],
                },
            ],
            "required_files": [
                "models/hiro-ace/ACE2S.ckpt",
                "inputs/hiro-ace/ic/ic_1994.nc",
                *hiro_hist_rels,
                *hiro_oos_rels,
            ],
            "timestep_hours": 6,
            "stochastic": True,
            "default_n_ensemble_per_ic": 1,
            "notes": "HiRO-ACE ACE2S stage (Ai2), stochastic; trained only on 9 years of X-SHiELD (2014-2022) and not expected by the authors to generalize outside training forcing. Here 2014-2022 (in amip-oos 2014-2024) is the training period, and amip-hist (1994-2014) is AIMIP's experimental extension outside training. Shipped ic_1994.nc is a HiRO 2014-01-01 state (ic_2014.nc[0]) relabelled to 1994-01-01; discard the first simulated year as spin-up.",
        },
        {
            "name": "ace2-som",
            "family": "fme",
            "fme_version": "2026.5.1",
            "fme_module": "fme.ace.inference",
            "calendar": "proleptic_gregorian",
            "checkpoint": f"{p}/models/ace2-som/ace2_som_ckpt.tar",
            "checkpoint_sha256": "b191174e45ac16d86f3a5ccfa99b8952866ec6482b19e08edbd0f05114f5f9ab",
            "ic_files": [
                f"{p}/inputs/ace2-som/ic/1xCO2/ic.nc",
                f"{p}/inputs/ace2-som/ic/3xCO2/ic.nc",
            ],
            "ic_sha256": {
                f"{p}/inputs/ace2-som/ic/1xCO2/ic.nc": "f481537cc80e4dd4c41f4e629563b2f62b4c3b1d9ede3db67f429ed3669fab13",
                f"{p}/inputs/ace2-som/ic/3xCO2/ic.nc": "449b50d6b97d360d70bb52f90ee604e933a84d1168cab7658c2fba20774ef751",
            },
            "forcing_sets": [
                {
                    "name": "1xCO2",
                    "dir": f"{p}/inputs/ace2-som/1xCO2",
                    "window": ["2032-01-01T00:00:00", "2035-12-31T18:00:00"],
                    "ic_files": [f"{p}/inputs/ace2-som/ic/1xCO2/ic.nc"],
                },
                {
                    "name": "3xCO2",
                    "dir": f"{p}/inputs/ace2-som/3xCO2",
                    "window": ["2032-01-01T00:00:00", "2035-12-31T18:00:00"],
                    "ic_files": [f"{p}/inputs/ace2-som/ic/3xCO2/ic.nc"],
                },
            ],
            "required_files": [
                "models/ace2-som/ace2_som_ckpt.tar",
                "inputs/ace2-som/ic/1xCO2/ic.nc",
                "inputs/ace2-som/ic/3xCO2/ic.nc",
                *som_1x_rels,
                *som_3x_rels,
            ],
            "timestep_hours": 6,
            "stochastic": False,
            "default_n_ensemble_per_ic": 1,
            "notes": "ACE2-SOM (Ai2 / GFDL SHiELD slab-ocean), deterministic, proleptic_gregorian; scenario forcing sets 1xCO2 (default) and 3xCO2 over 2032-2035, each paired with its equilibrated mixed-layer IC.",
        },
        {
            "name": "ace2s",
            "family": "fme",
            "fme_version": "2026.5.1",
            "fme_module": "fme.ace.inference",
            "calendar": "julian",
            "checkpoint": f"{p}/models/hiro-ace/ACE2S.ckpt",
            "checkpoint_sha256": "0b39d182074c02c92d6700088dfe5f6f1ae9488d50eb4a260564121ec2dcbf0e",
            "ic_files": [
                f"{p}/inputs/ace2s/ic/ic_2014.nc",
                f"{p}/inputs/ace2s/ic/ic_2023.nc",
            ],
            "ic_sha256": {
                f"{p}/inputs/ace2s/ic/ic_2014.nc": "81633b64252b2bcf0b8101935ed01373af26e31f25bc7bc610cd16e12f4d5d9a",
                f"{p}/inputs/ace2s/ic/ic_2023.nc": "55d199efe0d9979a2daf378d2bd53790f539fa1a44ce3815104b77ac3adb2d78",
            },
            "forcing_sets": [
                {
                    "name": "amip-2014-2023",
                    "dir": f"{p}/inputs/ace2s/amip-2014-2023",
                    "window": ["2014-01-01T00:00:00", "2023-12-31T18:00:00"],
                }
            ],
            "required_files": [
                "models/hiro-ace/ACE2S.ckpt",
                "inputs/ace2s/ic/ic_2014.nc",
                "inputs/ace2s/ic/ic_2023.nc",
                *ace2s_forcing_rels,
            ],
            "timestep_hours": 6,
            "stochastic": True,
            "default_n_ensemble_per_ic": 1,
            "notes": "ACE2S standalone global stochastic atmosphere emulator (allenai/HiRO-ACE ACE2S.ckpt, trained on X-SHiELD 2014-2022; HF-native 2014-2023 training-period forcing and ic_2014/ic_2023; julian calendar).",
        },
        {
            "name": "hiro-acev2",
            "family": "fme",
            "fme_version": "2026.10.0",
            "fme_module": "fme.ace.inference",
            "calendar": "julian",
            "checkpoint": f"{p}/models/hiro-acev2/ACE2S-X-SHiELD.ckpt",
            "checkpoint_sha256": "121bcce0d03de915c7964aff6619e11e32fd6d0780fc9ca0b654f88c6535729f",
            "ic_files": [
                f"{p}/inputs/hiro-acev2/ic/ic_2014.nc",
                f"{p}/inputs/hiro-acev2/ic/ic_2023.nc",
            ],
            "ic_sha256": {
                f"{p}/inputs/hiro-acev2/ic/ic_2014.nc": "81633b64252b2bcf0b8101935ed01373af26e31f25bc7bc610cd16e12f4d5d9a",
                f"{p}/inputs/hiro-acev2/ic/ic_2023.nc": "55d199efe0d9979a2daf378d2bd53790f539fa1a44ce3815104b77ac3adb2d78",
            },
            "forcing_sets": [
                {
                    "name": "amip-2014-2023",
                    "dir": f"{p}/inputs/hiro-acev2/amip-2014-2023",
                    "window": ["2014-01-01T00:00:00", "2023-12-31T18:00:00"],
                }
            ],
            "required_files": [
                "models/hiro-acev2/ACE2S-X-SHiELD.ckpt",
                "inputs/hiro-acev2/ic/ic_2014.nc",
                "inputs/hiro-acev2/ic/ic_2023.nc",
                *hiro2_forcing_rels,
            ],
            "timestep_hours": 6,
            "stochastic": True,
            "default_n_ensemble_per_ic": 1,
            "notes": "HiRO-ACEv2 global ACE2S-X-SHiELD stage (allenai/HiRO-ACEv2, trained on X-SHiELD 2014-2022, fme 2026.10.0 overlay), stochastic, julian calendar; HF-native 2014-2023 training-period forcing and ic_2014/ic_2023 (3 km downscaler deferred).",
        },
        {
            "name": "samudrace-cm4",
            "family": "fme",
            "fme_version": "2026.5.1",
            "fme_module": "fme.coupled.inference",
            "calendar": "noleap",
            "checkpoint": f"{p}/models/samudrace-cm4/samudrACE_CM4_piControl_ckpt.tar",
            "checkpoint_sha256": "cc20dcc8ec4254729f42acad7c2d55efc1b61a118f4370f730e537cdae8c2b79",
            "ic_files": [f"{p}/{rel}" for rel in cm4_atmos_ics_rel],
            "ocean_ic_files": [f"{p}/{rel}" for rel in cm4_ocean_ics_rel],
            "ic_sha256": {
                f"{p}/inputs/samudrace-cm4/ic/0311-01-01T000000/atmosphere/initial_condition.nc": "48401621295aa4dbe168c3c4c636657a68ec3363dfb0f7e0bfbd5bff0210c264",
                f"{p}/inputs/samudrace-cm4/ic/0313-01-01T000000/atmosphere/initial_condition.nc": "061a82f8c101a46f073ea5c9588b005b07e32ff34ebc6cc69030ccff423eab50",
                f"{p}/inputs/samudrace-cm4/ic/0315-01-01T000000/atmosphere/initial_condition.nc": "1b60da411a6b1644fa7ff456a704874a495a5ea087716aaafa58a42f42fec4e2",
                f"{p}/inputs/samudrace-cm4/ic/0317-01-01T000000/atmosphere/initial_condition.nc": "2d05e0e73238862dea46cb70baa82ec03f27f2a3129f940705b85cdcce5c457e",
                f"{p}/inputs/samudrace-cm4/ic/0319-01-01T000000/atmosphere/initial_condition.nc": "bdd50e6dbae686022e4a13b87b8843a0b77f9cbe112c8ea5953d2ddedbefa649",
                f"{p}/inputs/samudrace-cm4/ic/0151-01-06T000000/atmosphere/initial_condition.nc": "5552c031c74c14d3a82e985ab5ab3de9d00157d716db576cb431b800909487af",
            },
            "ocean_ic_sha256": {
                f"{p}/inputs/samudrace-cm4/ic/0311-01-01T000000/ocean/initial_condition.nc": "6a84dbfae2da51a6de2dd3f55a5b1218f167e2631d6c7a3011639ccd2a3b200e",
                f"{p}/inputs/samudrace-cm4/ic/0313-01-01T000000/ocean/initial_condition.nc": "f6f2876c73b4e53d561c3da55c4842bb8012c8b894fb67e5689b04515894d2ff",
                f"{p}/inputs/samudrace-cm4/ic/0315-01-01T000000/ocean/initial_condition.nc": "a8179b6331e27b66a4e3b679afbad1c60a8332615ed2dc2e12ce36f039e0c95b",
                f"{p}/inputs/samudrace-cm4/ic/0317-01-01T000000/ocean/initial_condition.nc": "c8923fd85e9f0068f656f6ec0611b8f6113e5e23619d21671339c6faf32fee09",
                f"{p}/inputs/samudrace-cm4/ic/0319-01-01T000000/ocean/initial_condition.nc": "be4bd002e6a5416296114fa849cb22524a6500b276145fde5ae96746e7cb7f85",
                f"{p}/inputs/samudrace-cm4/ic/0151-01-06T000000/ocean/initial_condition.nc": "19e993bd1138a59464acbd704af1cc3ac05ac60cdd8e857b837226339bf0b4f3",
            },
            "forcing_sets": [
                {
                    "name": "picontrol-0311",
                    "dir": f"{p}/inputs/samudrace-cm4/forcing",
                    "window": ["0311-01-01T00:00:00", "0410-12-27T00:00:00"],
                    "file_pattern": "forcing_0311.nc",
                    "ic_files": [
                        f"{p}/inputs/samudrace-cm4/ic/{slug}/atmosphere/initial_condition.nc"
                        for slug in cm4_0311_slugs
                    ],
                    "ocean_ic_files": [
                        f"{p}/inputs/samudrace-cm4/ic/{slug}/ocean/initial_condition.nc"
                        for slug in cm4_0311_slugs
                    ],
                    "repeat_period_hours": 8760,
                    "repeat_base_start": "0311-01-01T00:00:00",
                },
                {
                    "name": "picontrol-0151",
                    "dir": f"{p}/inputs/samudrace-cm4/forcing",
                    "window": ["0151-01-06T00:00:00", "0250-12-27T00:00:00"],
                    "file_pattern": "forcing_0151.nc",
                    "ic_files": [
                        f"{p}/inputs/samudrace-cm4/ic/0151-01-06T000000/atmosphere/initial_condition.nc"
                    ],
                    "ocean_ic_files": [
                        f"{p}/inputs/samudrace-cm4/ic/0151-01-06T000000/ocean/initial_condition.nc"
                    ],
                    "repeat_period_hours": 8760,
                    "repeat_base_start": "0151-01-01T00:00:00",
                },
            ],
            "required_files": [
                "models/samudrace-cm4/samudrACE_CM4_piControl_ckpt.tar",
                "inputs/samudrace-cm4/forcing/forcing_0151.nc",
                "inputs/samudrace-cm4/forcing/forcing_0311.nc",
                *cm4_atmos_ics_rel,
                *cm4_ocean_ics_rel,
            ],
            "timestep_hours": 120,
            "stochastic": False,
            "default_n_ensemble_per_ic": 1,
            "notes": "SamudrACE-CM4 coupled atmosphere-ocean piControl emulator (Ai2/M2LInES/GFDL, fme 2026.5.1, fme.coupled.inference); 120h (5-day) coupled step = 20x6h atmosphere steps, noleap calendar; 1-year forcing_0311.nc / forcing_0151.nc extended via n_repeats when multi-year.",
        },
        {
            "name": "samudrace-e3smv3",
            "family": "fme",
            "fme_version": "2026.10.0",
            "fme_module": "fme.coupled.inference",
            "calendar": "noleap",
            "checkpoint": f"{p}/models/samudrace-e3smv3/SamudrACE-E3SMv3.tar",
            "checkpoint_sha256": "dd5f9245e7d05f8e220ea7ed74b7a4ebd5621a9bfe5f540ddb136fc4625f0be4",
            "ic_files": [
                f"{p}/inputs/samudrace-e3smv3/ic/SamudrACE-E3SMv3-ICx3-train_atmosphere_ic.nc"
            ],
            "ocean_ic_files": [
                f"{p}/inputs/samudrace-e3smv3/ic/SamudrACE-E3SMv3-ICx3-train_ocean_ic.nc"
            ],
            "ic_sha256": {
                f"{p}/inputs/samudrace-e3smv3/ic/SamudrACE-E3SMv3-ICx3-train_atmosphere_ic.nc": "9f97310b98bfa29a4c21c8a316014ede221d20b9b37d9e5cfda39a4e3bed0161"
            },
            "ocean_ic_sha256": {
                f"{p}/inputs/samudrace-e3smv3/ic/SamudrACE-E3SMv3-ICx3-train_ocean_ic.nc": "b21bf023757ad5d80c387dfaf3d71a8bc50b168258cfc555e9e195759d582609"
            },
            "forcing_sets": [
                {
                    "name": "train-1yr",
                    "dir": f"{p}/inputs/samudrace-e3smv3/forcing",
                    "window": ["0425-01-03T12:00:00", "0425-12-29T12:00:00"],
                    "file_pattern": "atmosphere-forcing-1yr.nc",
                    "ocean_dir": f"{p}/inputs/samudrace-e3smv3/forcing",
                    "ocean_file_pattern": "ocean-forcing-1yr.nc",
                }
            ],
            "required_files": [
                "models/samudrace-e3smv3/SamudrACE-E3SMv3.tar",
                "inputs/samudrace-e3smv3/ic/SamudrACE-E3SMv3-ICx3-train_atmosphere_ic.nc",
                "inputs/samudrace-e3smv3/ic/SamudrACE-E3SMv3-ICx3-train_ocean_ic.nc",
                "inputs/samudrace-e3smv3/forcing/atmosphere-forcing-1yr.nc",
                "inputs/samudrace-e3smv3/forcing/ocean-forcing-1yr.nc",
            ],
            "timestep_hours": 120,
            "stochastic": True,
            "default_n_ensemble_per_ic": 1,
            "notes": "SamudrACE-E3SMv3 stochastic coupled atmosphere-ocean piControl emulator (Ai2, fme 2026.10.0 overlay, fme.coupled.inference); 120h (5-day) coupled step = 20x6h atmosphere steps, noleap calendar; 1-year native forcing (max 72 coupled steps; fme 2026.10.0 has no n_repeats) + 3-state ICx3 at 0425-01-03T12:00:00.",
        },
    ]

    from .emu_catalog import registry_entries as _emu_entries

    all_entries.extend(_emu_entries(p))

    if selected_models is not None:
        wanted = set(selected_models)
        entries = [m for m in all_entries if m["name"] in wanted]
    else:
        entries = all_entries

    return {
        "schema_version": 3,
        "generated_by": "aimip setup",
        "models": entries,
    }
