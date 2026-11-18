"""Portable HPC data installer (`aimip setup`), lock verifier (`aimip.lock`), and lock generator.

Workflow:
- Downloads ACE-family, DLESyM, and ArchesClimate-SSP checkpoints, ICs, and historical forcings from pinned
  Hugging Face repositories via `huggingface_hub` (`HF_HOME` under `--dest`, resumable, pinned commit `revision=`).
- Downloads the AIMIP monthly SST/SIC boundary dataset (`ERA5-0.25deg-monthly-mean-forcing-1978-2024.nc`)
  from Zenodo record `17065758` via resumable HTTP `Range` requests (or local `file://` mirror).
- Runs the deterministic netCDF builders (`aimip.builders`) to produce `co2_global_mean.nc`,
  `ace2-eamv3` 1994 IC & 1994-2014 noleap forcing, `hiro-ace` 1994 IC & 1994-2014 forcing,
  and 2014-2024 out-of-sample (`amip-oos`) forcing for SHiELD and HiRO-ACE.
- Verifies every downloaded and derived file against `aimip.lock` (size, SHA-256, and
  Zenodo MD5 when pinned), raising `ValidationError` naming the exact file on mismatch.
- Generates `<dest>/registry.json` using `${DATA}/...` paths and provides `--write-lock`
  to generate/refresh `aimip.lock` from an existing staged tree.
"""

from __future__ import annotations

import calendar as py_calendar
import contextlib
import dataclasses
import fcntl
import hashlib
import json
import os
import shutil
import sys
import tarfile
import urllib.error
import urllib.request
import uuid
from collections.abc import Callable, Iterator, Mapping, Sequence
from typing import Any

from . import __version__
from .builders import (
    HIST_YEARS_DEFAULT,
    OOS_YEARS_DEFAULT,
    build_co2_table,
    build_eamv3_forcing,
    build_eamv3_ic,
    build_hiro_forcing,
    build_hiro_ic,
    build_oos_forcing,
    extract_zip_member,
)
from .errors import ConfigError, ValidationError
from .lora import sha256_file
from .registry import (
    DATA_PLACEHOLDER,
    SAMUDRACE_CM4_IC_SLUGS,
    build_default_registry_payload,
    resolve_data_dir,
)
from .validators import (
    strip_group_other_write,
    validate_model_name,
    validate_safe_path,
)

CANONICAL_MODELS: tuple[str, ...] = (
    "ace2-era5",
    "ace2-shield",
    "ace2s-shield-plus",
    "ace2-eamv3",
    "hiro-ace",
    "ace2-som",
    "ace2s",
    "hiro-acev2",
    "samudrace-cm4",
    "samudrace-e3smv3",
    # Emulator environment models (Earth2Studio DLESyM, ArchesClimate-SSP, ArchesWeather, Samudra-OM4, MD-1.5, CAMulator, cBottle-3d, Spherical DYffusion)
    "dlesym-v1.0",
    "dlesym-v1.1",
    "dlesym-v0-isccp",
    "dlesym-aimip",
    "archesclimate-ssp",
    "archesweather",
    "archesweathergen",
    "samudra-om4",
    "md-1.5",
    "md-1.5-aimip",
    "camulator",
    "cbottle-3d",
    "spherical-dyffusion",
)

# Default pinned Hugging Face commit revisions (overrideable per file in `aimip.lock`).
HF_REVISIONS: dict[str, str] = {
    "allenai/ACE2-ERA5": "f25123c52d7faa8f3533816a311cf3353ee83daa",
    "allenai/ACE2-SHiELD": "13978196694ef4bb20d6f9ed94a83e697adda328",
    "allenai/ACE2S-SHiELD-plus": "85ac7b78b13cf166d3b86c23c0894f41c97dbddb",
    "allenai/ACE2-EAMv3": "5f00316f10c1435db21b2a1dade7620f9a6de51c",
    "allenai/HiRO-ACE": "286264358ea097b794a776375eef992b08b98668",
    "allenai/ACE2-SOM": "47c5b54424093d8607b6b5014699a1fc9b30f6b0",
    "allenai/HiRO-ACEv2": "16bd51b8da79d85c138cc166c93196501616c3bf",
    "allenai/SamudrACE-CM4-piControl": "48567c80676de7db51d270c8c0c5dd61f08c9f39",
    "allenai/SamudrACE-E3SMv3": "b9abeb489154a9f7a4f776180122d64fdcf66cb5",
    "nvidia/cbottle": "e60ce82b236b953d79d3ead7f43445a3bb8e37d3",
    "salv47/spherical-dyffusion": "56fcb5d995717e91ba770f5ee41e8c3a8cfb00f6",
}

ZENODO_AIMIP_RECORD = "17065758"
ZENODO_AIMIP_FILENAME = "ERA5-0.25deg-monthly-mean-forcing-1978-2024.nc"
ZENODO_AIMIP_URL = (
    f"https://zenodo.org/records/{ZENODO_AIMIP_RECORD}/files/"
    f"{ZENODO_AIMIP_FILENAME}?download=1"
)
ZENODO_AIMIP_REL_PATH = f"raw/zenodo-{ZENODO_AIMIP_RECORD}/{ZENODO_AIMIP_FILENAME}"
CO2_TABLE_REL_PATH = "raw/co2_global_mean.nc"

SHA256_TBD = "SHA256_TBD"
MD5_TBD = "MD5_TBD"
REVISION_TBD = "REVISION_TBD"

INSTALLED_CONTAINER_LOCK_PATH = "/opt/aimip/src/src/aimip/aimip.lock"

# ArchesClimate-SSP has no upstream licence file and is fetched on first use into
# `$AIMIP_DATA/code/ArchesClimate-SSP` (never redistributed in the image or repository).
ARCHES_COMMIT = "06ce82dca2414db4f1b3d6940b61c5cd723f4620"
ARCHES_ARCHIVE_URL = (
    f"https://github.com/grahamclyne/ArchesClimate-SSP/archive/{ARCHES_COMMIT}.tar.gz"
)
ARCHES_ARCHIVE_SHA256 = (
    "e56e082040f375e851a9e1928f959de0e2d5a1b97e145424be4e6d1fba5735e4"
)
ARCHES_ARCHIVE_SIZE = 13_436_543
ARCHES_TREE_SHA256 = (
    "5cd1d90a0be28d65088e8162764c4ddc57a697f09eceddc89b88c429275d9006"
)
ARCHES_TREE_FILES = 242
ARCHES_CODE_REL_DIR = "code/ArchesClimate-SSP"


@dataclasses.dataclass(frozen=True)
class SetupYearConfig:
    """Configurable year ranges for production setup (1979-2024) and fast synthetic unit tests."""

    era5_hist_years: tuple[int, ...] = tuple(range(1979, 2015))
    shield_hist_years: tuple[int, ...] = tuple(range(1979, 2015))
    derived_hist_years: tuple[int, ...] = HIST_YEARS_DEFAULT
    oos_years: tuple[int, ...] = OOS_YEARS_DEFAULT
    oos_bridge_year: int | None = 2014
    oos_ref_nonleap_year: int = 2014
    oos_ref_leap_year: int = 2012
    oos_anchor_t0: str = "2014-12-01"
    oos_anchor_t1: str = "2025-01-01"
    co2_y0: int = 1979
    co2_y1: int = 2024
    co2_last_upstream: int = 2022
    co2_validate_reference_stats: bool = True
    hiro_mask_reference_year: int = 1994
    oos_expected_nonleap_steps: int | None = 1460
    oos_expected_leap_steps: int | None = 1464
    som_years: tuple[int, ...] = (2032, 2033, 2034, 2035)
    hiro_native_years: tuple[int, ...] = tuple(range(2014, 2024))


@dataclasses.dataclass(frozen=True)
class ArtifactSpec:
    """Catalog specification for a single downloaded or derived file under `--dest`."""

    rel_path: str
    role: str  # "output" | "raw"
    models_output: tuple[str, ...]
    models_needed: tuple[str, ...]
    source_kind: str  # "hf" | "zenodo" | "url" | "github_archive" | "derived"
    hf_repo: str | None = None
    hf_path: str | None = None
    hf_revision: str | None = None
    hf_repo_type: str | None = None   # None/"model" or "dataset" (gclyne/ArchesClimate-data)
    zenodo_record: str | None = None
    zenodo_file: str | None = None
    zenodo_url: str | None = None
    url: str | None = None
    archive_sha256: str | None = None
    builder: str | None = None
    input_refs: tuple[str, ...] = ()
    pin_after_first_fetch: bool = False

    def default_source_dict(
        self,
        *,
        known_shas: Mapping[str, str] | None = None,
        zenodo_md5: str = MD5_TBD,
    ) -> dict[str, Any]:
        """Construct the `source` dictionary recorded in `aimip.lock`."""
        if self.source_kind == "hf":
            out_src: dict[str, Any] = {
                "kind": "hf",
                "repo": self.hf_repo,
                "path": self.hf_path,
                "revision": self.hf_revision or REVISION_TBD,
            }
            if self.hf_repo_type and self.hf_repo_type != "model":
                out_src["repo_type"] = self.hf_repo_type
            return out_src
        if self.source_kind == "zenodo":
            return {
                "kind": "zenodo",
                "record": self.zenodo_record,
                "file": self.zenodo_file,
                "url": self.zenodo_url,
                "md5": zenodo_md5,
            }
        if self.source_kind == "url":
            return {
                "kind": "url",
                "url": self.url,
            }
        if self.source_kind == "github_archive":
            return {
                "kind": "github_archive",
                "repo": self.hf_repo,
                "path": self.hf_path,
                "revision": self.hf_revision or REVISION_TBD,
                "url": self.url,
                "archive_sha256": self.archive_sha256 or SHA256_TBD,
            }
        if self.source_kind == "derived":
            shas = known_shas or {}
            inputs_map = {
                ref: shas.get(ref, SHA256_TBD) for ref in self.input_refs
            }
            return {
                "kind": "derived",
                "builder": self.builder,
                "inputs": inputs_map,
            }
        raise ValidationError(f"Unsupported artifact source_kind {self.source_kind!r}.")


def md5_file(path: str, chunk_size: int = 8 * 1024 * 1024) -> str:
    """Compute lowercase 32-hex MD5 digest of *path*."""
    validated = validate_safe_path(path, label="file for MD5")
    digest = hashlib.md5()  # noqa: S324 - used solely for Zenodo MD5 content verification
    with open(validated, "rb") as fh:
        while True:
            block = fh.read(chunk_size)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def parse_selected_models(models_arg: str) -> tuple[str, ...]:
    """Parse `--models all|m1,m2` into a deduplicated tuple in canonical order."""
    if not isinstance(models_arg, str) or not models_arg.strip():
        raise ValidationError("`--models` must be 'all' or a comma-separated list of model names.")
    cleaned = models_arg.strip()
    if cleaned.lower() == "all":
        return CANONICAL_MODELS

    tokens = [t.strip() for t in cleaned.split(",") if t.strip()]
    if not tokens:
        raise ValidationError("`--models` list is empty.")

    allowed = set(CANONICAL_MODELS)
    requested: set[str] = set()
    for tok in tokens:
        if tok.lower() == "all":
            return CANONICAL_MODELS
        validated = validate_model_name(tok, allowed=allowed)
        requested.add(validated)

    return tuple(m for m in CANONICAL_MODELS if m in requested)


def resolve_setup_dest_dir(
    explicit_dest: str | None = None,
    playground_dir: str | None = None,
) -> str:
    """Resolve target data directory from `--dest` or `$AIMIP_DATA`."""
    resolved = resolve_data_dir(explicit_dest, playground_dir=playground_dir)
    if not resolved:
        raise ConfigError(
            "No destination directory specified: pass `--dest DIR` or set `$AIMIP_DATA`."
        )
    return resolved


def resolve_lock_path(
    explicit_lock: str | None = None,
    *,
    dest_dir: str | None = None,
    must_exist: bool = True,
) -> str:
    """Locate `aimip.lock` from `--lock`, `$AIMIP_LOCK`, packaged `aimip/aimip.lock`, or fallback paths."""
    if explicit_lock:
        path = validate_safe_path(explicit_lock, label="--lock path")
        if must_exist and not os.path.isfile(path):
            raise ConfigError(f"Lock file not found: {path}")
        return path

    env_lock = os.environ.get("AIMIP_LOCK")
    if env_lock:
        path = validate_safe_path(env_lock, label="AIMIP_LOCK")
        if must_exist and not os.path.isfile(path):
            raise ConfigError(f"Lock file from $AIMIP_LOCK not found: {path}")
        return path

    pkg_dir = os.path.dirname(os.path.abspath(__file__))
    pkg_parent = os.path.abspath(os.path.join(pkg_dir, ".."))
    candidates = [
        os.path.join(pkg_dir, "aimip.lock"),
        os.path.join(pkg_parent, "aimip.lock"),
        INSTALLED_CONTAINER_LOCK_PATH,
        "/opt/aimip/src/aimip.lock",
    ]
    if dest_dir:
        candidates.append(os.path.join(dest_dir, "aimip.lock"))

    for cand in candidates:
        try:
            validated = validate_safe_path(cand, label="default aimip.lock")
        except ValidationError:
            continue
        if os.path.isfile(validated):
            return validated

    default_target = validate_safe_path(
        candidates[0], label="default aimip.lock path"
    )
    if must_exist:
        raise ConfigError(
            f"Lock file not found at {default_target} (pass `--lock FILE` or run `aimip setup --write-lock`)."
        )
    return default_target


def build_artifact_catalog(
    years: SetupYearConfig | None = None,
    revisions: Mapping[str, str] | None = None,
) -> dict[str, ArtifactSpec]:
    """Build the complete ordered catalog of raw and output artifacts across all registered models."""
    yc = years or SetupYearConfig()
    rev = dict(HF_REVISIONS)
    if revisions:
        rev.update(revisions)

    items: list[ArtifactSpec] = []

    def _add(spec: ArtifactSpec) -> None:
        items.append(spec)

    # 1. Model checkpoints (`models/<model>/<ckpt>`)
    ckpt_defs: list[tuple[str, tuple[str, ...], str, str]] = [
        ("ace2-era5", ("ace2-era5",), "allenai/ACE2-ERA5", "ace2_era5_ckpt.tar"),
        ("ace2-shield", ("ace2-shield",), "allenai/ACE2-SHiELD", "ace2_shield_ckpt.tar"),
        (
            "ace2s-shield-plus",
            ("ace2s-shield-plus",),
            "allenai/ACE2S-SHiELD-plus",
            "ace2s_shield_plus_ckpt.tar",
        ),
        ("ace2-eamv3", ("ace2-eamv3",), "allenai/ACE2-EAMv3", "ace2_EAMv3_ckpt.tar"),
        ("hiro-ace", ("hiro-ace", "ace2s"), "allenai/HiRO-ACE", "ACE2S.ckpt"),
    ]
    for model_dir, model_names, repo, fname in ckpt_defs:
        _add(
            ArtifactSpec(
                rel_path=f"models/{model_dir}/{fname}",
                role="output",
                models_output=model_names,
                models_needed=model_names,
                source_kind="hf",
                hf_repo=repo,
                hf_path=fname,
                hf_revision=rev.get(repo, REVISION_TBD),
            )
        )

    # 2. Direct HF initial conditions
    for ic_name in ("ic_1979.nc", "ic_2001.nc"):
        _add(
            ArtifactSpec(
                rel_path=f"inputs/ace2-era5/ic/{ic_name}",
                role="output",
                models_output=("ace2-era5",),
                models_needed=("ace2-era5",),
                source_kind="hf",
                hf_repo="allenai/ACE2-ERA5",
                hf_path=f"initial_conditions/{ic_name}",
                hf_revision=rev.get("allenai/ACE2-ERA5", REVISION_TBD),
            )
        )

    _add(
        ArtifactSpec(
            rel_path="models/ace2s-shield-plus/ic.nc",
            role="output",
            models_output=("ace2-shield", "ace2s-shield-plus"),
            models_needed=("ace2-shield", "ace2s-shield-plus"),
            source_kind="hf",
            hf_repo="allenai/ACE2S-SHiELD-plus",
            hf_path="initial_conditions/AMIP/ic.nc",
            hf_revision=rev.get("allenai/ACE2S-SHiELD-plus", REVISION_TBD),
        )
    )

    # 3. Direct HF AMIP historical forcings (`ace2-era5` and `shield`)
    derived_year_set = set(yc.derived_hist_years)
    for y in yc.era5_hist_years:
        needed_by = ["ace2-era5"]
        if y in derived_year_set:
            needed_by.append("ace2-eamv3")
        _add(
            ArtifactSpec(
                rel_path=f"inputs/ace2-era5/amip-hist/forcing_{y}.nc",
                role="output",
                models_output=("ace2-era5",),
                models_needed=tuple(needed_by),
                source_kind="hf",
                hf_repo="allenai/ACE2-ERA5",
                hf_path=f"forcing_data/forcing_{y}.nc",
                hf_revision=rev.get("allenai/ACE2-ERA5", REVISION_TBD),
            )
        )

    hiro_shield_years = set(yc.derived_hist_years)
    if yc.derived_hist_years:
        hiro_shield_years.add(1994 if 1994 in hiro_shield_years else yc.derived_hist_years[0])
    for y in yc.shield_hist_years:
        needed_by = ["ace2-shield", "ace2s-shield-plus"]
        if y in hiro_shield_years:
            needed_by.append("hiro-ace")
        _add(
            ArtifactSpec(
                rel_path=f"inputs/shield/amip-hist/forcing_{y}.nc",
                role="output",
                models_output=("ace2-shield", "ace2s-shield-plus"),
                models_needed=tuple(needed_by),
                source_kind="hf",
                hf_repo="allenai/ACE2S-SHiELD-plus",
                hf_path=f"forcing_data/AMIP/forcing_{y}.nc",
                hf_revision=rev.get("allenai/ACE2S-SHiELD-plus", REVISION_TBD),
            )
        )

    # 4. Raw upstream build inputs (`raw/...`)
    _add(
        ArtifactSpec(
            rel_path="raw/ace2-eamv3/initial_conditions/1971010100.nc",
            role="raw",
            models_output=(),
            models_needed=("ace2-eamv3",),
            source_kind="hf",
            hf_repo="allenai/ACE2-EAMv3",
            hf_path="initial_conditions/1971010100.nc",
            hf_revision=rev.get("allenai/ACE2-EAMv3", REVISION_TBD),
        )
    )
    _add(
        ArtifactSpec(
            rel_path="raw/ace2-eamv3/forcing_data/1971.nc",
            role="raw",
            models_output=(),
            models_needed=("ace2-eamv3",),
            source_kind="hf",
            hf_repo="allenai/ACE2-EAMv3",
            hf_path="forcing_data/1971.nc",
            hf_revision=rev.get("allenai/ACE2-EAMv3", REVISION_TBD),
        )
    )
    _add(
        ArtifactSpec(
            rel_path="raw/hiro-ace/initial_conditions/ic_2014.nc",
            role="raw",
            models_output=(),
            models_needed=("hiro-ace",),
            source_kind="hf",
            hf_repo="allenai/HiRO-ACE",
            hf_path="initial_conditions/ic_2014.nc",
            hf_revision=rev.get("allenai/HiRO-ACE", REVISION_TBD),
        )
    )
    _add(
        ArtifactSpec(
            rel_path="raw/hiro-ace/forcing_data/forcing_2014.nc",
            role="raw",
            models_output=(),
            models_needed=("hiro-ace",),
            source_kind="hf",
            hf_repo="allenai/HiRO-ACE",
            hf_path="forcing_data/forcing_2014.nc",
            hf_revision=rev.get("allenai/HiRO-ACE", REVISION_TBD),
        )
    )

    oos_models = ("ace2-shield", "ace2s-shield-plus", "hiro-ace")
    _add(
        ArtifactSpec(
            rel_path=ZENODO_AIMIP_REL_PATH,
            role="raw",
            models_output=("dlesym-aimip", "md-1.5", "md-1.5-aimip"),   # read directly by these models
            models_needed=oos_models + ("dlesym-aimip", "md-1.5", "md-1.5-aimip"),
            source_kind="zenodo",
            zenodo_record=ZENODO_AIMIP_RECORD,
            zenodo_file=ZENODO_AIMIP_FILENAME,
            zenodo_url=ZENODO_AIMIP_URL,
        )
    )

    era5_hist_set = set(yc.era5_hist_years)
    co2_refs: list[str] = []
    for y in range(yc.co2_y0, yc.co2_last_upstream + 1):
        if y in era5_hist_set:
            co2_refs.append(f"inputs/ace2-era5/amip-hist/forcing_{y}.nc")
        else:
            co2_refs.append(f"allenai/ACE2-ERA5:forcing_data/forcing_{y}.nc")

    _add(
        ArtifactSpec(
            rel_path=CO2_TABLE_REL_PATH,
            role="raw",
            models_output=(),
            models_needed=oos_models,
            source_kind="derived",
            builder="build_co2_table",
            input_refs=tuple(co2_refs),
        )
    )

    # 5. Derived EAMv3 IC & historical forcing
    _add(
        ArtifactSpec(
            rel_path="inputs/ace2-eamv3/ic/ic_1994.nc",
            role="output",
            models_output=("ace2-eamv3",),
            models_needed=("ace2-eamv3",),
            source_kind="derived",
            builder="build_eamv3_ic",
            input_refs=("raw/ace2-eamv3/initial_conditions/1971010100.nc",),
        )
    )
    for y in yc.derived_hist_years:
        _add(
            ArtifactSpec(
                rel_path=f"inputs/ace2-eamv3/amip-hist/{y}.nc",
                role="output",
                models_output=("ace2-eamv3",),
                models_needed=("ace2-eamv3",),
                source_kind="derived",
                builder="build_eamv3_forcing",
                input_refs=(
                    "raw/ace2-eamv3/forcing_data/1971.nc",
                    f"inputs/ace2-era5/amip-hist/forcing_{y}.nc",
                ),
            )
        )

    # 6. Derived HiRO-ACE IC & historical forcing
    _add(
        ArtifactSpec(
            rel_path="inputs/hiro-ace/ic/ic_1994.nc",
            role="output",
            models_output=("hiro-ace",),
            models_needed=("hiro-ace",),
            source_kind="derived",
            builder="build_hiro_ic",
            input_refs=("raw/hiro-ace/initial_conditions/ic_2014.nc",),
        )
    )
    ref_shield_mask_y = (
        1994
        if 1994 in yc.derived_hist_years
        else (yc.derived_hist_years[0] if yc.derived_hist_years else 1994)
    )
    for y in yc.derived_hist_years:
        hiro_refs = [
            "raw/hiro-ace/forcing_data/forcing_2014.nc",
            f"inputs/shield/amip-hist/forcing_{ref_shield_mask_y}.nc",
        ]
        cand_y = f"inputs/shield/amip-hist/forcing_{y}.nc"
        if cand_y not in hiro_refs:
            hiro_refs.append(cand_y)
        _add(
            ArtifactSpec(
                rel_path=f"inputs/hiro-ace/amip-hist/forcing_{y}.nc",
                role="output",
                models_output=("hiro-ace",),
                models_needed=("hiro-ace",),
                source_kind="derived",
                builder="build_hiro_forcing",
                input_refs=tuple(hiro_refs),
            )
        )

    # 7. Derived OOS forcings (`shield/amip-oos` and `hiro-ace/amip-oos`)
    oos_targets = [
        ("shieldplus", "inputs/shield", ("ace2-shield", "ace2s-shield-plus")),
        ("hiro", "inputs/hiro-ace", ("hiro-ace",)),
    ]
    for fmt, prefix, target_models in oos_targets:
        if yc.oos_bridge_year is not None:
            by = yc.oos_bridge_year
            _add(
                ArtifactSpec(
                    rel_path=f"{prefix}/amip-oos/forcing_{by}.nc",
                    role="output",
                    models_output=target_models,
                    models_needed=target_models,
                    source_kind="derived",
                    builder=f"build_oos_forcing:{fmt}:bridge",
                    input_refs=(f"{prefix}/amip-hist/forcing_{by}.nc",),
                )
            )
        for y in yc.oos_years:
            ry = (
                yc.oos_ref_leap_year
                if py_calendar.isleap(y)
                else yc.oos_ref_nonleap_year
            )
            refs = [f"{prefix}/amip-hist/forcing_{yc.oos_ref_nonleap_year}.nc"]
            ref_y_str = f"{prefix}/amip-hist/forcing_{ry}.nc"
            if ref_y_str not in refs:
                refs.append(ref_y_str)
            refs.extend([ZENODO_AIMIP_REL_PATH, CO2_TABLE_REL_PATH])
            _add(
                ArtifactSpec(
                    rel_path=f"{prefix}/amip-oos/forcing_{y}.nc",
                    role="output",
                    models_output=target_models,
                    models_needed=target_models,
                    source_kind="derived",
                    builder=f"build_oos_forcing:{fmt}",
                    input_refs=tuple(refs),
                )
            )

    # 8. Additional fme-family models (`ace2-som`, `ace2s`, `hiro-acev2`, `samudrace-cm4`, `samudrace-e3smv3`)
    extra_fme_ckpt_defs = [
        ("ace2-som", "allenai/ACE2-SOM", "ace2_som_ckpt.tar"),
        ("hiro-acev2", "allenai/HiRO-ACEv2", "ACE2S-X-SHiELD.ckpt"),
        (
            "samudrace-cm4",
            "allenai/SamudrACE-CM4-piControl",
            "samudrACE_CM4_piControl_ckpt.tar",
        ),
        ("samudrace-e3smv3", "allenai/SamudrACE-E3SMv3", "SamudrACE-E3SMv3.tar"),
    ]
    for model_name, repo, fname in extra_fme_ckpt_defs:
        _add(
            ArtifactSpec(
                rel_path=f"models/{model_name}/{fname}",
                role="output",
                models_output=(model_name,),
                models_needed=(model_name,),
                source_kind="hf",
                hf_repo=repo,
                hf_path=fname,
                hf_revision=rev.get(repo, REVISION_TBD),
            )
        )

    # Non-LFS upstream reference inference config YAMLs (`pin_after_first_fetch`)
    extra_fme_ref_yaml_defs = [
        ("ace2-som", "allenai/ACE2-SOM", "inference_config.yaml"),
        ("hiro-acev2", "allenai/HiRO-ACEv2", "ace2s_inference_config_global.yaml"),
        ("samudrace-cm4", "allenai/SamudrACE-CM4-piControl", "inference_config.yaml"),
        ("samudrace-e3smv3", "allenai/SamudrACE-E3SMv3", "inference-config.yaml"),
    ]
    for model_name, repo, fname in extra_fme_ref_yaml_defs:
        _add(
            ArtifactSpec(
                rel_path=f"models/{model_name}/{fname}",
                role="output",
                models_output=(model_name,),
                models_needed=(model_name,),
                source_kind="hf",
                hf_repo=repo,
                hf_path=fname,
                hf_revision=rev.get(repo, REVISION_TBD),
                pin_after_first_fetch=True,
            )
        )

    # 8a. `ace2-som`: 1xCO2 and 3xCO2 scenario ICs + 2032-2035 forcings
    for scen in ("1xCO2", "3xCO2"):
        _add(
            ArtifactSpec(
                rel_path=f"inputs/ace2-som/ic/{scen}/ic.nc",
                role="output",
                models_output=("ace2-som",),
                models_needed=("ace2-som",),
                source_kind="hf",
                hf_repo="allenai/ACE2-SOM",
                hf_path=f"initial_conditions/{scen}/ic.nc",
                hf_revision=rev.get("allenai/ACE2-SOM", REVISION_TBD),
            )
        )
        for y in yc.som_years:
            _add(
                ArtifactSpec(
                    rel_path=f"inputs/ace2-som/{scen}/forcing_{y}.nc",
                    role="output",
                    models_output=("ace2-som",),
                    models_needed=("ace2-som",),
                    source_kind="hf",
                    hf_repo="allenai/ACE2-SOM",
                    hf_path=f"forcing_data/{scen}/forcing_{y}.nc",
                    hf_revision=rev.get("allenai/ACE2-SOM", REVISION_TBD),
                )
            )

    # 8b. `ace2s` (`allenai/HiRO-ACE`) & `hiro-acev2` (`allenai/HiRO-ACEv2`) HF-native 2014-2023 ICs + forcings
    for model_name, repo in (
        ("ace2s", "allenai/HiRO-ACE"),
        ("hiro-acev2", "allenai/HiRO-ACEv2"),
    ):
        for ic_name in ("ic_2014.nc", "ic_2023.nc"):
            _add(
                ArtifactSpec(
                    rel_path=f"inputs/{model_name}/ic/{ic_name}",
                    role="output",
                    models_output=(model_name,),
                    models_needed=(model_name,),
                    source_kind="hf",
                    hf_repo=repo,
                    hf_path=f"initial_conditions/{ic_name}",
                    hf_revision=rev.get(repo, REVISION_TBD),
                )
            )
        for y in yc.hiro_native_years:
            _add(
                ArtifactSpec(
                    rel_path=f"inputs/{model_name}/amip-2014-2023/forcing_{y}.nc",
                    role="output",
                    models_output=(model_name,),
                    models_needed=(model_name,),
                    source_kind="hf",
                    hf_repo=repo,
                    hf_path=f"forcing_data/forcing_{y}.nc",
                    hf_revision=rev.get(repo, REVISION_TBD),
                )
            )

    # 8c. `samudrace-cm4` coupled atmosphere + ocean ICs (upstream colon paths -> colon-free local slugs) & forcings
    for fname in ("forcing_0151.nc", "forcing_0311.nc"):
        _add(
            ArtifactSpec(
                rel_path=f"inputs/samudrace-cm4/forcing/{fname}",
                role="output",
                models_output=("samudrace-cm4",),
                models_needed=("samudrace-cm4",),
                source_kind="hf",
                hf_repo="allenai/SamudrACE-CM4-piControl",
                hf_path=f"forcing_data/{fname}",
                hf_revision=rev.get("allenai/SamudrACE-CM4-piControl", REVISION_TBD),
            )
        )
    for upstream_ts, local_slug in SAMUDRACE_CM4_IC_SLUGS:
        for realm in ("atmosphere", "ocean"):
            _add(
                ArtifactSpec(
                    rel_path=f"inputs/samudrace-cm4/ic/{local_slug}/{realm}/initial_condition.nc",
                    role="output",
                    models_output=("samudrace-cm4",),
                    models_needed=("samudrace-cm4",),
                    source_kind="hf",
                    hf_repo="allenai/SamudrACE-CM4-piControl",
                    hf_path=f"initial_conditions/{upstream_ts}/{realm}/initial_condition.nc",
                    hf_revision=rev.get("allenai/SamudrACE-CM4-piControl", REVISION_TBD),
                )
            )

    # 8d. `samudrace-e3smv3` coupled 3-state ICx3 + 1-year atmosphere & ocean forcings
    for ic_fname in (
        "SamudrACE-E3SMv3-ICx3-train_atmosphere_ic.nc",
        "SamudrACE-E3SMv3-ICx3-train_ocean_ic.nc",
    ):
        _add(
            ArtifactSpec(
                rel_path=f"inputs/samudrace-e3smv3/ic/{ic_fname}",
                role="output",
                models_output=("samudrace-e3smv3",),
                models_needed=("samudrace-e3smv3",),
                source_kind="hf",
                hf_repo="allenai/SamudrACE-E3SMv3",
                hf_path=f"initial_conditions/{ic_fname}",
                hf_revision=rev.get("allenai/SamudrACE-E3SMv3", REVISION_TBD),
            )
        )
    for forcing_fname in ("atmosphere-forcing-1yr.nc", "ocean-forcing-1yr.nc"):
        _add(
            ArtifactSpec(
                rel_path=f"inputs/samudrace-e3smv3/forcing/{forcing_fname}",
                role="output",
                models_output=("samudrace-e3smv3",),
                models_needed=("samudrace-e3smv3",),
                source_kind="hf",
                hf_repo="allenai/SamudrACE-E3SMv3",
                hf_path=f"forcing_data/{forcing_fname}",
                hf_revision=rev.get("allenai/SamudrACE-E3SMv3", REVISION_TBD),
            )
        )

    # 9. Emulator families: pinned HF, Zenodo, and GitHub-archive files (`emu_sources.json`)
    from .emu_catalog import artifacts as _emu_artifacts

    for a in _emu_artifacts():
        if a.get("source_kind") == "zenodo":
            _add(
                ArtifactSpec(
                    rel_path=a["rel_path"],
                    role="output",
                    models_output=tuple(a["models"]),
                    models_needed=tuple(a["models"]),
                    source_kind="zenodo",
                    zenodo_record=a["record"],
                    zenodo_file=a["path"],
                    zenodo_url=a["url"],
                )
            )
        elif a.get("source_kind") == "github_archive":
            _add(
                ArtifactSpec(
                    rel_path=a["rel_path"],
                    role="output",
                    models_output=tuple(a["models"]),
                    models_needed=tuple(a["models"]),
                    source_kind="github_archive",
                    hf_repo=a["repo"],
                    hf_path=a["path"],
                    hf_revision=a["revision"],
                    url=a["archive_url"],
                    archive_sha256=a["archive_sha256"],
                )
            )
        else:
            _add(
                ArtifactSpec(
                    rel_path=a["rel_path"],
                    role="output",
                    models_output=tuple(a["models"]),
                    models_needed=tuple(a["models"]),
                    source_kind="hf",
                    hf_repo=a["repo"],
                    hf_path=a["path"],
                    hf_revision=a["revision"],
                    hf_repo_type=a["repo_type"],
                )
            )

    # 10. ArchesWeather / ArchesWeatherGen: Zenodo 22795233 files, checkpoints and configs extracted from its zip
    from .emu_catalog import AW_RECORD, AW_ZENODO_FILES, AW_ZIP_MEMBERS, AW_ZIP_REL

    for rel, (fname, _size, _md5, _sha, models) in AW_ZENODO_FILES.items():
        _add(
            ArtifactSpec(
                rel_path=rel,
                role="raw" if rel == AW_ZIP_REL else "output",
                models_output=() if rel == AW_ZIP_REL else tuple(models),
                models_needed=tuple(models),
                source_kind="zenodo",
                zenodo_record=AW_RECORD,
                zenodo_file=fname,
                zenodo_url=f"https://zenodo.org/records/{AW_RECORD}/files/{fname}?download=1",
            )
        )
    for rel, (_member, _size, _sha, models) in AW_ZIP_MEMBERS.items():
        _add(
            ArtifactSpec(
                rel_path=rel,
                role="output",
                models_output=tuple(models),
                models_needed=tuple(models),
                source_kind="derived",
                builder="extract_zip_member",
                input_refs=(AW_ZIP_REL,),
            )
        )

    return {spec.rel_path: spec for spec in items}


def load_lock_file(lock_path: str) -> dict[str, Any]:
    """Load and structurally validate `aimip.lock`."""
    validated = validate_safe_path(lock_path, label="aimip.lock")
    if not os.path.isfile(validated):
        raise ConfigError(f"Lock file not found: {validated}")
    try:
        with open(validated, encoding="utf-8") as fh:
            payload = json.load(fh)
    except (OSError, json.JSONDecodeError) as exc:
        raise ConfigError(f"Failed to parse lock file {validated}: {exc}") from exc

    if not isinstance(payload, dict) or not isinstance(payload.get("files"), dict):
        raise ConfigError(
            f"Invalid lock schema in {validated}: expected top-level object with 'files' mapping."
        )
    return payload


def verify_file_against_lock(
    dest_dir: str,
    rel_path: str,
    lock_entry: Mapping[str, Any],
    *,
    role: str = "output",
    strict_unpinned: bool = True,
) -> dict[str, Any]:
    """Verify a single file's existence, byte size, SHA-256, and optional MD5 against *lock_entry*.

    Raises `ValidationError` naming *rel_path* on any missing file or checksum/size mismatch.
    """
    abs_path = validate_safe_path(
        os.path.join(dest_dir, rel_path), label=f"locked file {rel_path}"
    )
    if not os.path.isfile(abs_path):
        raise ValidationError(
            f"Lock verification failed for {rel_path}: file is missing ({abs_path})."
        )

    actual_size = os.path.getsize(abs_path)
    expected_size = lock_entry.get("size")
    if expected_size is not None:
        if not isinstance(expected_size, int) or expected_size < 0:
            raise ConfigError(
                f"Invalid 'size' {expected_size!r} in aimip.lock for {rel_path}."
            )
        if actual_size != expected_size:
            raise ValidationError(
                f"Lock verification failed for {rel_path}: size mismatch "
                f"(expected {expected_size} bytes, got {actual_size} bytes)."
            )

    expected_sha = str(lock_entry.get("sha256") or "")
    pin_after = bool(lock_entry.get("pin_after_first_fetch"))
    computed_unpinned_pin_after = False
    if not expected_sha or expected_sha == SHA256_TBD:
        if strict_unpinned and not pin_after:
            raise ValidationError(
                f"Lock verification failed for {rel_path}: lock entry has unpinned checksum "
                f"{expected_sha!r} (run `aimip setup --write-lock` to pin)."
            )
        if pin_after and (not isinstance(expected_size, int) or expected_size < 0):
            raise ValidationError(
                f"Lock verification failed for {rel_path}: 'pin_after_first_fetch' entry "
                "must specify a non-negative integer 'size' in aimip.lock."
            )
        actual_sha = sha256_file(abs_path)
        computed_unpinned_pin_after = pin_after
    else:
        actual_sha = sha256_file(abs_path)
        if actual_sha.lower() != expected_sha.lower():
            raise ValidationError(
                f"Lock verification failed for {rel_path}: sha256 mismatch "
                f"(expected {expected_sha.lower()}, got {actual_sha})."
            )

    source_info = lock_entry.get("source")
    if isinstance(source_info, Mapping) and source_info.get("kind") == "zenodo":
        expected_md5 = source_info.get("md5")
        if (
            not isinstance(expected_md5, str)
            or not expected_md5
            or expected_md5 == MD5_TBD
        ):
            if strict_unpinned:
                raise ValidationError(
                    f"Lock verification failed for {rel_path}: Zenodo lock entry has unpinned md5 "
                    f"{expected_md5!r} (run `aimip setup --write-lock` to pin)."
                )
        else:
            actual_md5 = md5_file(abs_path)
            if actual_md5.lower() != expected_md5.lower():
                raise ValidationError(
                    f"Lock verification failed for {rel_path}: Zenodo md5 mismatch "
                    f"(expected {expected_md5.lower()}, got {actual_md5})."
                )

    out_info: dict[str, Any] = {
        "rel_path": rel_path,
        "abs_path": abs_path,
        "size": actual_size,
        "sha256": actual_sha,
    }
    if computed_unpinned_pin_after:
        out_info["pin_after_first_fetch"] = True
    return out_info


def _file_matches_lock_quiet(
    dest_dir: str,
    rel_path: str,
    lock_entry: Mapping[str, Any] | None,
) -> bool:
    """Return True if `<dest_dir>/<rel_path>` already exists and satisfies *lock_entry*."""
    abs_path = os.path.join(dest_dir, rel_path)
    if not os.path.isfile(abs_path):
        return False
    if lock_entry is None:
        return False
    expected_sha = lock_entry.get("sha256")
    expected_size = lock_entry.get("size")
    if expected_size is not None and os.path.getsize(abs_path) != expected_size:
        return False
    if isinstance(expected_sha, str) and expected_sha and expected_sha != SHA256_TBD:
        try:
            verify_file_against_lock(
                dest_dir, rel_path, lock_entry, role="output", strict_unpinned=True
            )
            return True
        except ValidationError:
            return False
    if bool(lock_entry.get("pin_after_first_fetch")):
        return isinstance(expected_size, int) and os.path.getsize(abs_path) == expected_size
    # Unpinned raw intermediate already present with non-zero bytes
    return os.path.getsize(abs_path) > 0


def _validate_source_file_path(raw_path: str, *, label: str) -> str:
    """Validate an upstream downloaded/mirror source file path after `realpath`.

    Upstream Hugging Face repos (e.g. `allenai/SamudrACE-CM4-piControl`) contain ISO
    timestamps with `:` in directory names (`initial_conditions/0311-01-01T00:00:00/...`).
    Source files are only read by Python `os.link`/`shutil.copyfile`/hashing and never
    interpolated into shell or Apptainer `--bind` strings; staged destinations under
    `$AIMIP_DATA` remain strictly colon-free (`validate_safe_path`).
    """
    real = os.path.realpath(raw_path)
    if not real.startswith("/") or ".." in real.split("/"):
        raise ValidationError(f"Invalid {label} {raw_path!r}: unsafe resolved path {real!r}.")
    # Validate every character except `:` against `validate_safe_path`.
    validate_safe_path(real.replace(":", "_"), label=label)
    return real


def _sha256_source_file(path: str, chunk_size: int = 8 * 1024 * 1024) -> str:
    """Compute SHA-256 of an upstream/staged source file (supporting `:` in upstream HF paths)."""
    validated = _validate_source_file_path(path, label="source file for SHA-256")
    digest = hashlib.sha256()
    with open(validated, "rb") as fh:
        while True:
            block = fh.read(chunk_size)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def _parse_mirror_uri(raw_mirror: str | None) -> str | None:
    """Resolve an optional directory path or `file:///...` URI."""
    if not raw_mirror:
        return None
    cleaned = raw_mirror.strip()
    if cleaned.startswith("file://"):
        cleaned = cleaned[len("file://") :]
    if not cleaned:
        return None
    return validate_safe_path(cleaned, label="mirror path")


def _atomic_stage_file(src_path: str, dst_path: str) -> None:
    """Stage *src_path* to *dst_path* atomically via hardlink or copy."""
    # huggingface_hub returns a snapshot path that is a RELATIVE symlink into ../../blobs/; os.link() would hard-link
    # the symlink itself (Linux linkat without AT_SYMLINK_FOLLOW), leaving a dangling link at the destination.
    # Resolve and stage the real file.
    validated_src = _validate_source_file_path(src_path, label="downloaded source")
    validated_dst = validate_safe_path(dst_path, label="staged destination")
    if not os.path.isfile(validated_src):
        raise ConfigError(f"Downloaded source is not a regular file: {src_path} -> {validated_src}")
    parent = os.path.dirname(validated_dst) or "/"
    os.makedirs(parent, exist_ok=True)
    tmp_path = os.path.join(parent, f".stage_{uuid.uuid4().hex}.tmp")
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


def download_hf_artifact(
    *,
    repo: str,
    path_in_repo: str,
    revision: str,
    dst_path: str,
    hf_home: str,
    repo_type: str | None = None,
) -> str:
    """Download a single file from Hugging Face Hub (or `$AIMIP_HF_MIRROR`) with resume support."""
    validated_dst = validate_safe_path(dst_path, label=f"HF target {path_in_repo}")

    mirror_root = _parse_mirror_uri(os.environ.get("AIMIP_HF_MIRROR"))
    if mirror_root is not None:
        rel_variants = (
            (path_in_repo, path_in_repo.replace(":", ""))
            if ":" in path_in_repo
            else (path_in_repo,)
        )
        for repo_sub in (repo, repo.replace("/", "__")):
            for rel_v in rel_variants:
                cand = os.path.join(mirror_root, repo_sub, rel_v)
                if os.path.isfile(cand):
                    _atomic_stage_file(cand, validated_dst)
                    return validated_dst
        raise ValidationError(
            f"HF mirror {mirror_root} does not contain {repo}/{path_in_repo}."
        )

    from huggingface_hub import hf_hub_download

    os.makedirs(hf_home, exist_ok=True)
    cache_dir = os.path.join(hf_home, "hub")
    rev_arg = None if (not revision or revision == REVISION_TBD) else revision
    try:
        cached_blob = hf_hub_download(
            repo_id=repo,
            filename=path_in_repo,
            revision=rev_arg,
            cache_dir=cache_dir,
            **({"repo_type": repo_type} if repo_type and repo_type != "model" else {}),
        )
    except Exception as exc:  # noqa: BLE001
        raise ConfigError(
            f"Failed to download {repo}/{path_in_repo} (revision={revision}): {exc}"
        ) from exc

    _atomic_stage_file(cached_blob, validated_dst)
    return validated_dst


def download_zenodo_artifact(
    *,
    record: str,
    filename: str,
    url: str,
    dst_path: str,
    expected_size: int | None = None,
    expected_md5: str | None = None,
    timeout_seconds: float = 120.0,
) -> str:
    """Download a Zenodo record file with HTTP `Range` resume (or from `$AIMIP_ZENODO_MIRROR`)."""
    validated_dst = validate_safe_path(dst_path, label=f"Zenodo target {filename}")

    mirror_root = _parse_mirror_uri(os.environ.get("AIMIP_ZENODO_MIRROR"))
    if mirror_root is not None:
        for cand in (
            os.path.join(mirror_root, record, filename),
            os.path.join(mirror_root, filename),
        ):
            if os.path.isfile(cand):
                _atomic_stage_file(cand, validated_dst)
                return validated_dst
        raise ValidationError(
            f"Zenodo mirror {mirror_root} does not contain record {record} file {filename}."
        )

    if url.startswith("file://"):
        local_src = validate_safe_path(url[len("file://") :], label="Zenodo file:// URL")
        if not os.path.isfile(local_src):
            raise ValidationError(f"Zenodo local source not found: {local_src}")
        _atomic_stage_file(local_src, validated_dst)
        return validated_dst

    parent = os.path.dirname(validated_dst) or "/"
    os.makedirs(parent, exist_ok=True)
    part_path = f"{validated_dst}.part"
    offset = os.path.getsize(part_path) if os.path.isfile(part_path) else 0
    if expected_size is not None and offset > expected_size:
        offset = 0

    headers: dict[str, str] = {"User-Agent": f"aimip-setup/{__version__}"}
    if offset > 0:
        headers["Range"] = f"bytes={offset}-"

    req = urllib.request.Request(url, headers=headers)  # noqa: S310 - validated HTTPS Zenodo URL
    try:
        with urllib.request.urlopen(req, timeout=timeout_seconds) as resp:  # noqa: S310
            status = getattr(resp, "status", 200)
            mode = "ab" if (offset > 0 and status == 206) else "wb"
            with open(part_path, mode) as out_fh:
                while True:
                    chunk = resp.read(8 * 1024 * 1024)
                    if not chunk:
                        break
                    out_fh.write(chunk)
    except urllib.error.HTTPError as exc:
        if exc.code == 416 and os.path.isfile(part_path):
            # 416 Range Not Satisfiable: `.part` is already at full file length
            pass
        else:
            raise ConfigError(
                f"Failed to download Zenodo {record}/{filename} from {url}: HTTP {exc.code}"
            ) from exc
    except OSError as exc:
        raise ConfigError(
            f"Failed to download Zenodo {record}/{filename} from {url}: {exc}"
        ) from exc

    if expected_size is not None:
        got_size = os.path.getsize(part_path)
        if got_size != expected_size:
            raise ValidationError(
                f"Zenodo download {filename} size mismatch: expected {expected_size}, got {got_size}."
            )
    if expected_md5 and expected_md5 != MD5_TBD:
        got_md5 = md5_file(part_path)
        if got_md5.lower() != expected_md5.lower():
            raise ValidationError(
                f"Zenodo download {filename} MD5 mismatch: expected {expected_md5.lower()}, got {got_md5}."
            )

    os.replace(part_path, validated_dst)
    return validated_dst


def compute_arches_tree_digest(tree_dir: str) -> tuple[str, int]:
    """Compute `(sha256_hex, file_count)` of an extracted `ArchesClimate-SSP` directory.

    Digest = SHA-256 over sorted POSIX relative paths of:
      `relpath + "\\0" + (sha256(file) | "LINK:" + readlink(symlink)) + "\\n"`
    walking directories in sorted order without following symlinks.
    """
    root = validate_safe_path(tree_dir, label="ArchesClimate-SSP code directory")
    if not os.path.isdir(root) or os.path.islink(root):
        raise ValidationError(
            f"ArchesClimate-SSP source directory missing or symlinked: {root}"
        )

    records: list[tuple[str, str]] = []
    for dirpath, dirs, files in os.walk(root, followlinks=False):
        dir_links = [d for d in dirs if os.path.islink(os.path.join(dirpath, d))]
        dirs[:] = sorted(d for d in dirs if not os.path.islink(os.path.join(dirpath, d)))
        for name in (*files, *dir_links):
            full = os.path.join(dirpath, name)
            rel = os.path.relpath(full, root).replace(os.sep, "/")
            if os.path.islink(full):
                val = f"LINK:{os.readlink(full)}"
            elif os.path.isfile(full):
                digest = hashlib.sha256()
                with open(full, "rb") as fh:
                    while True:
                        chunk = fh.read(1024 * 1024)
                        if not chunk:
                            break
                        digest.update(chunk)
                val = digest.hexdigest()
            else:
                raise ValidationError(
                    f"Unexpected non-regular entry in ArchesClimate-SSP tree: {full}"
                )
            records.append((rel, val))

    records.sort(key=lambda item: item[0])
    overall = hashlib.sha256()
    for rel, val in records:
        overall.update(f"{rel}\0{val}\n".encode("utf-8"))
    return overall.hexdigest(), len(records)


def verify_arches_code_dir(
    dest_dir: str,
    *,
    expected_tree_sha256: str = ARCHES_TREE_SHA256,
    expected_tree_files: int = ARCHES_TREE_FILES,
) -> dict[str, Any]:
    """Verify `<dest_dir>/code/ArchesClimate-SSP` matches the pinned tree digest and file count."""
    code_dir = validate_safe_path(
        os.path.join(dest_dir, ARCHES_CODE_REL_DIR),
        label="ArchesClimate-SSP code directory",
    )
    if not os.path.isdir(code_dir) or os.path.islink(code_dir):
        raise ValidationError(
            f"Lock verification failed for {ARCHES_CODE_REL_DIR}: directory is missing ({code_dir})."
        )
    tree_sha, n_files = compute_arches_tree_digest(code_dir)
    if n_files != expected_tree_files or tree_sha.lower() != expected_tree_sha256.lower():
        raise ValidationError(
            f"Lock verification failed for {ARCHES_CODE_REL_DIR}: tree digest/count mismatch "
            f"(expected {expected_tree_files} files / sha256={expected_tree_sha256.lower()}, "
            f"got {n_files} files / sha256={tree_sha})."
        )
    return {
        "code_dir": code_dir,
        "tree_sha256": tree_sha,
        "files": n_files,
    }


def extract_and_verify_arches_archive(
    archive_path: str,
    target_dir: str,
    *,
    expected_archive_sha256: str = ARCHES_ARCHIVE_SHA256,
    expected_archive_size: int = ARCHES_ARCHIVE_SIZE,
    expected_tree_sha256: str = ARCHES_TREE_SHA256,
    expected_tree_files: int = ARCHES_TREE_FILES,
) -> dict[str, Any]:
    """Safely extract *archive_path* into *target_dir* after extracted-tree digest verification.

    Always requires `(extracted_tree_sha256, file_count) == (expected_tree_sha256, expected_tree_files)`
    before publishing *target_dir*, so neither regenerated GitHub tarballs with altered content nor
    truncated/corrupted writes can be published.
    """
    validated_archive = _validate_source_file_path(
        archive_path, label="ArchesClimate-SSP archive"
    )
    validated_target = validate_safe_path(
        target_dir, label="ArchesClimate-SSP target directory"
    )
    if not os.path.isfile(validated_archive):
        raise ValidationError(f"ArchesClimate-SSP archive not found: {validated_archive}")

    archive_size = os.path.getsize(validated_archive)
    archive_sha = _sha256_source_file(validated_archive)

    parent_dir = os.path.dirname(validated_target) or "/"
    os.makedirs(parent_dir, exist_ok=True)
    strip_group_other_write(parent_dir)
    stage_dir = os.path.join(parent_dir, f".arches_stage_{uuid.uuid4().hex}")
    old_backup: str | None = None

    try:
        os.makedirs(stage_dir, mode=0o755, exist_ok=False)
        strip_group_other_write(stage_dir)

        try:
            tf_ctx = tarfile.open(validated_archive, mode="r:*")
        except (tarfile.TarError, OSError) as exc:
            raise ValidationError(
                f"Invalid ArchesClimate-SSP archive {validated_archive}: {exc}"
            ) from exc

        with tf_ctx as tf:
            members = tf.getmembers()
            if not members:
                raise ValidationError(
                    f"ArchesClimate-SSP archive {validated_archive} is empty."
                )

            parsed_parts: list[tuple[tarfile.TarInfo, list[str]]] = []
            for m in members:
                if not (m.isdir() or m.isfile() or m.issym()):
                    raise ValidationError(
                        f"Refusing unsupported tar entry type in ArchesClimate-SSP archive: {m.name!r}"
                    )
                if (
                    not m.name
                    or m.name.startswith("/")
                    or "\x00" in m.name
                    or "\\" in m.name
                ):
                    raise ValidationError(
                        f"Refusing unsafe path in ArchesClimate-SSP archive: {m.name!r}"
                    )
                stripped = m.name.rstrip("/")
                if not stripped:
                    raise ValidationError(
                        f"Refusing root path in ArchesClimate-SSP archive: {m.name!r}"
                    )
                parts = stripped.split("/")
                if any(seg in ("", ".", "..") for seg in parts):
                    raise ValidationError(
                        f"Refusing traversal segment in ArchesClimate-SSP archive: {m.name!r}"
                    )
                if m.issym() and (not m.linkname or "\x00" in m.linkname):
                    raise ValidationError(
                        f"Refusing invalid symlink target in ArchesClimate-SSP archive: {m.name!r}"
                    )
                parsed_parts.append((m, parts))

            first_top = parsed_parts[0][1][0]
            strip_top = all(p[0] == first_top for _, p in parsed_parts) and any(
                len(p) > 1 for _, p in parsed_parts
            )

            rel_entries: list[tuple[tarfile.TarInfo, str]] = []
            symlink_rels: set[str] = set()
            seen_leaf_rels: set[str] = set()
            for m, parts in parsed_parts:
                rel_parts = parts[1:] if strip_top else parts
                if not rel_parts:
                    continue
                rel = "/".join(rel_parts)
                if m.issym():
                    symlink_rels.add(rel)
                if m.isfile() or m.issym():
                    if rel in seen_leaf_rels:
                        raise ValidationError(
                            f"Refusing duplicate tar entry in ArchesClimate-SSP archive: {rel!r}"
                        )
                    seen_leaf_rels.add(rel)
                rel_entries.append((m, rel))

            for _, rel in rel_entries:
                segs = rel.split("/")
                for depth in range(1, len(segs)):
                    anc = "/".join(segs[:depth])
                    if anc in symlink_rels:
                        raise ValidationError(
                            f"Refusing tar entry {rel!r} beneath symlinked directory {anc!r}"
                        )

            # Pass 1: directories and regular files (O_CREAT | O_EXCL, group/other write stripped).
            for m, rel in rel_entries:
                dst = os.path.join(stage_dir, *rel.split("/"))
                if m.isdir():
                    os.makedirs(dst, mode=0o755, exist_ok=True)
                    strip_group_other_write(dst)
                elif m.isfile():
                    parent = os.path.dirname(dst)
                    os.makedirs(parent, mode=0o755, exist_ok=True)
                    strip_group_other_write(parent)
                    src_fh = tf.extractfile(m)
                    if src_fh is None:
                        raise ValidationError(f"Cannot read tar member {rel!r}")
                    file_mode = 0o755 if (m.mode & 0o100) else 0o644
                    fd = os.open(dst, os.O_WRONLY | os.O_CREAT | os.O_EXCL, file_mode)
                    try:
                        with src_fh:
                            while True:
                                chunk = src_fh.read(1024 * 1024)
                                if not chunk:
                                    break
                                view = memoryview(chunk)
                                while view:
                                    written = os.write(fd, view)
                                    if written <= 0:
                                        raise OSError(
                                            f"Short write while extracting ArchesClimate-SSP member {rel!r}"
                                        )
                                    view = view[written:]
                        os.fchmod(fd, file_mode)
                    finally:
                        os.close(fd)

            # Pass 2: symbolic links (preserved as relative/raw link targets; never dereferenced).
            for m, rel in rel_entries:
                if m.issym():
                    dst = os.path.join(stage_dir, *rel.split("/"))
                    parent = os.path.dirname(dst)
                    os.makedirs(parent, mode=0o755, exist_ok=True)
                    strip_group_other_write(parent)
                    os.symlink(m.linkname, dst)

        for walk_root, walk_dirs, walk_files in os.walk(stage_dir, followlinks=False):
            if not os.path.islink(walk_root):
                strip_group_other_write(walk_root)
            for fname in walk_files:
                fpath = os.path.join(walk_root, fname)
                if not os.path.islink(fpath):
                    strip_group_other_write(fpath)

        tree_sha, n_files = compute_arches_tree_digest(stage_dir)
        archive_ok = (
            archive_sha.lower() == expected_archive_sha256.lower()
            and archive_size == expected_archive_size
        )
        tree_ok = (
            tree_sha.lower() == expected_tree_sha256.lower()
            and n_files == expected_tree_files
        )
        if not tree_ok:
            raise ValidationError(
                f"ArchesClimate-SSP extracted tree verification failed: archive sha256={archive_sha} "
                f"(size={archive_size}, expected {expected_archive_sha256} / {expected_archive_size} B) "
                f"and extracted tree sha256={tree_sha} ({n_files} files, expected "
                f"{expected_tree_sha256} / {expected_tree_files} files)."
            )

        if os.path.lexists(validated_target):
            old_backup = os.path.join(parent_dir, f".arches_old_{uuid.uuid4().hex}")
            os.replace(validated_target, old_backup)
        os.replace(stage_dir, validated_target)
        if old_backup and os.path.lexists(old_backup):
            shutil.rmtree(old_backup, ignore_errors=True)

        return {
            "code_dir": validated_target,
            "archive_sha256": archive_sha,
            "archive_size": archive_size,
            "tree_sha256": tree_sha,
            "files": n_files,
            "matched_by": "archive+tree" if archive_ok else "tree",
        }
    except BaseException:
        if os.path.lexists(stage_dir):
            shutil.rmtree(stage_dir, ignore_errors=True)
        if old_backup and os.path.lexists(old_backup) and not os.path.lexists(validated_target):
            try:
                os.replace(old_backup, validated_target)
            except OSError:
                pass
        raise


def ensure_arches_code(
    dest_dir: str,
    *,
    archive_url: str = ARCHES_ARCHIVE_URL,
    timeout_seconds: float = 120.0,
    progress_stream: Any = None,
) -> dict[str, Any]:
    """Ensure `<dest_dir>/code/ArchesClimate-SSP` is present and matches the pinned digest."""
    validated_dest = validate_safe_path(dest_dir, label="setup destination dir")
    target_dir = validate_safe_path(
        os.path.join(validated_dest, ARCHES_CODE_REL_DIR),
        label="ArchesClimate-SSP code directory",
    )
    if os.path.isdir(target_dir) and not os.path.islink(target_dir):
        try:
            info = verify_arches_code_dir(validated_dest)
            info["downloaded"] = False
            return info
        except ValidationError:
            pass

    if progress_stream is not None:
        print(
            f"  [code] {ARCHES_CODE_REL_DIR} (~{_format_bytes(ARCHES_ARCHIVE_SIZE)}) <- {archive_url}",
            file=progress_stream,
            flush=True,
        )

    code_parent = os.path.dirname(target_dir) or "/"
    os.makedirs(code_parent, exist_ok=True)
    tmp_archive = os.path.join(code_parent, f".arches_{ARCHES_COMMIT[:12]}_{uuid.uuid4().hex}.tar.gz")
    try:
        if archive_url.startswith("file://"):
            local_src = validate_safe_path(
                archive_url[len("file://") :], label="ArchesClimate-SSP file:// archive"
            )
            if not os.path.isfile(local_src):
                raise ConfigError(f"ArchesClimate-SSP local archive not found: {local_src}")
            shutil.copyfile(local_src, tmp_archive)
        else:
            req = urllib.request.Request(
                archive_url, headers={"User-Agent": f"aimip-setup/{__version__}"}
            )  # noqa: S310 - pinned HTTPS GitHub archive URL
            try:
                with urllib.request.urlopen(req, timeout=timeout_seconds) as resp:  # noqa: S310
                    with open(tmp_archive, "wb") as out_fh:
                        while True:
                            chunk = resp.read(1024 * 1024)
                            if not chunk:
                                break
                            out_fh.write(chunk)
            except urllib.error.HTTPError as exc:
                raise ConfigError(
                    f"Failed to download ArchesClimate-SSP archive from {archive_url}: HTTP {exc.code}"
                ) from exc
            except OSError as exc:
                raise ConfigError(
                    f"Failed to download ArchesClimate-SSP archive from {archive_url}: {exc}"
                ) from exc

        res = extract_and_verify_arches_archive(tmp_archive, target_dir)
        res["downloaded"] = True
        return res
    finally:
        if os.path.exists(tmp_archive):
            try:
                os.unlink(tmp_archive)
            except OSError:
                pass


def download_url_artifact(
    *,
    url: str,
    dst_path: str,
    expected_size: int | None = None,
    timeout_seconds: float = 120.0,
) -> str:
    """Download a pinned HTTPS / `file://` asset (or from `$AIMIP_URL_MIRROR`) with HTTP `Range` resume."""
    validated_dst = validate_safe_path(dst_path, label="URL artifact target")
    fname = os.path.basename(validated_dst)

    mirror_root = _parse_mirror_uri(os.environ.get("AIMIP_URL_MIRROR"))
    if mirror_root is not None:
        cand = os.path.join(mirror_root, fname)
        if os.path.isfile(cand):
            _atomic_stage_file(cand, validated_dst)
            return validated_dst
        raise ValidationError(f"URL mirror {mirror_root} does not contain {fname}.")

    if url.startswith("file://"):
        local_src = validate_safe_path(url[len("file://") :], label="file:// asset URL")
        if not os.path.isfile(local_src):
            raise ValidationError(f"Local asset source not found: {local_src}")
        _atomic_stage_file(local_src, validated_dst)
        return validated_dst

    parent = os.path.dirname(validated_dst) or "/"
    os.makedirs(parent, exist_ok=True)
    part_path = f"{validated_dst}.part"
    offset = os.path.getsize(part_path) if os.path.isfile(part_path) else 0
    if expected_size is not None and offset > expected_size:
        offset = 0

    headers: dict[str, str] = {"User-Agent": f"aimip-setup/{__version__}"}
    if offset > 0:
        headers["Range"] = f"bytes={offset}-"

    req = urllib.request.Request(url, headers=headers)  # noqa: S310 - pinned HTTPS release URL
    try:
        with urllib.request.urlopen(req, timeout=timeout_seconds) as resp:  # noqa: S310
            status = getattr(resp, "status", 200)
            mode = "ab" if (offset > 0 and status == 206) else "wb"
            with open(part_path, mode) as out_fh:
                while True:
                    chunk = resp.read(8 * 1024 * 1024)
                    if not chunk:
                        break
                    out_fh.write(chunk)
    except urllib.error.HTTPError as exc:
        if exc.code == 416 and os.path.isfile(part_path):
            pass
        else:
            raise ConfigError(f"Failed to download {fname} from {url}: HTTP {exc.code}") from exc
    except OSError as exc:
        raise ConfigError(f"Failed to download {fname} from {url}: {exc}") from exc

    if expected_size is not None:
        got_size = os.path.getsize(part_path)
        if got_size != expected_size:
            raise ValidationError(
                f"Download {fname} size mismatch: expected {expected_size}, got {got_size}."
            )
    strip_group_other_write(part_path)
    os.replace(part_path, validated_dst)
    return validated_dst


def ensure_md15_code_and_weights(
    dest_dir: str,
    *,
    archive_url: str | None = None,
    expected_archive_sha256: str | None = None,
    expected_archive_size: int | None = None,
    expected_tree_sha256: str | None = None,
    expected_tree_files: int | None = None,
    timeout_seconds: float = 120.0,
    progress_stream: Any = None,
) -> dict[str, Any]:
    """Ensure `<dest_dir>/code/monthly-diffusion` and `<dest_dir>/models/md-1.5/` match pinned SHA-256 digests."""
    from . import md15 as _md15_mod

    eff_ar_sha = (
        _md15_mod.MD15_GITHUB_ARCHIVE_SHA256
        if expected_archive_sha256 is None
        else expected_archive_sha256
    )
    eff_ar_size = (
        _md15_mod.MD15_GITHUB_ARCHIVE_SIZE
        if expected_archive_size is None
        else int(expected_archive_size)
    )

    validated_dest = validate_safe_path(dest_dir, label="setup destination dir")
    code_target = validate_safe_path(
        os.path.join(validated_dest, _md15_mod.MD15_CODE_REL_DIR),
        label="MD-1.5 code directory",
    )
    model_target = validate_safe_path(
        os.path.join(validated_dest, "models", "md-1.5"),
        label="MD-1.5 model directory",
    )

    three_files_ok = all(
        os.path.isfile(os.path.join(validated_dest, rel_p))
        and not os.path.islink(os.path.join(validated_dest, rel_p))
        and sha256_file(os.path.join(validated_dest, rel_p)).lower() == want_s.lower()
        for rel_p, want_s in (
            (_md15_mod.MD15_CONFIG_REL_PATH, _md15_mod.MD15_CONFIG_SHA256),
            (_md15_mod.MD15_WEIGHTS_REL_PATH, _md15_mod.MD15_WEIGHTS_SHA256),
            (_md15_mod.MD15_STATS_REL_PATH, _md15_mod.MD15_STATS_SHA256),
            (_md15_mod.MD15_AIMIP_CONFIG_REL_PATH, _md15_mod.MD15_AIMIP_CONFIG_SHA256),
            (_md15_mod.MD15_AIMIP_WEIGHTS_REL_PATH, _md15_mod.MD15_AIMIP_WEIGHTS_SHA256),
        )
    )
    override_md15 = os.environ.get("AIMIP_MD15_CODE")
    if override_md15 and three_files_ok:
        safe_ovr = validate_safe_path(override_md15, label="AIMIP_MD15_CODE")
        if not os.path.isdir(safe_ovr) or os.path.islink(safe_ovr):
            raise ValidationError(f"AIMIP_MD15_CODE is missing or symlinked: {safe_ovr}")
        return {"code_dir": safe_ovr, "downloaded": False, "override": True}
    if three_files_ok and os.path.isdir(code_target) and not os.path.islink(code_target):
        try:
            info = _md15_mod.verify_md15_code_dir(
                validated_dest,
                expected_tree_sha256=expected_tree_sha256,
                expected_tree_files=expected_tree_files,
            )
            info["downloaded"] = False
            return info
        except ValidationError:
            pass

    eff_url = (
        archive_url
        or os.environ.get("AIMIP_MD15_ARCHIVE_URL")
        or _md15_mod.MD15_GITHUB_ARCHIVE_URL
    )
    if progress_stream is not None:
        print(
            f"  [code+weights] {_md15_mod.MD15_CODE_REL_DIR} (~{_format_bytes(eff_ar_size)}) <- {eff_url}",
            file=progress_stream,
            flush=True,
        )

    code_parent = os.path.dirname(code_target) or "/"
    os.makedirs(code_parent, exist_ok=True)
    tmp_archive = os.path.join(
        code_parent, f".md15_{_md15_mod.MD15_COMMIT[:12]}_{uuid.uuid4().hex}.tar.gz"
    )
    try:
        download_url_artifact(
            url=eff_url,
            dst_path=tmp_archive,
            expected_size=eff_ar_size,
            timeout_seconds=timeout_seconds,
        )
        res = _md15_mod.extract_and_verify_md15_archive(
            tmp_archive,
            code_target,
            model_target_dir=model_target,
            expected_archive_sha256=eff_ar_sha,
            expected_archive_size=eff_ar_size,
            expected_tree_sha256=expected_tree_sha256,
            expected_tree_files=expected_tree_files,
        )
        res["downloaded"] = True
        return res
    finally:
        if os.path.exists(tmp_archive):
            try:
                os.unlink(tmp_archive)
            except OSError:
                pass


def ensure_camulator_code(
    dest_dir: str,
    *,
    archive_url: str | None = None,
    expected_archive_sha256: str | None = None,
    expected_archive_size: int | None = None,
    expected_tree_sha256: str | None = None,
    expected_tree_files: int | None = None,
    timeout_seconds: float = 120.0,
    progress_stream: Any = None,
) -> dict[str, Any]:
    """Ensure `<dest_dir>/code/miles-credit-camulator` matches the pinned 380-file Apache-2.0 tree."""
    from . import camulator as _camulator_mod

    override = os.environ.get("AIMIP_CAMULATOR_CODE")
    if override:
        safe_ovr = validate_safe_path(override, label="AIMIP_CAMULATOR_CODE")
        if not os.path.isdir(safe_ovr) or os.path.islink(safe_ovr):
            raise ValidationError(f"AIMIP_CAMULATOR_CODE is missing or symlinked: {safe_ovr}")
        return {"code_dir": safe_ovr, "downloaded": False, "override": True}

    eff_ar_sha = (
        _camulator_mod.CAMULATOR_GITHUB_ARCHIVE_SHA256
        if expected_archive_sha256 is None
        else expected_archive_sha256
    )
    eff_ar_size = (
        _camulator_mod.CAMULATOR_GITHUB_ARCHIVE_SIZE
        if expected_archive_size is None
        else int(expected_archive_size)
    )

    validated_dest = validate_safe_path(dest_dir, label="setup destination dir")
    code_target = validate_safe_path(
        os.path.join(validated_dest, _camulator_mod.CAMULATOR_CODE_REL_DIR),
        label="CAMulator code directory",
    )
    if os.path.isdir(code_target) and not os.path.islink(code_target):
        try:
            info = _camulator_mod.verify_camulator_code_dir(
                validated_dest,
                expected_tree_sha256=expected_tree_sha256,
                expected_tree_files=expected_tree_files,
            )
            info["downloaded"] = False
            return info
        except ValidationError:
            pass

    eff_url = (
        archive_url
        or os.environ.get("AIMIP_CAMULATOR_ARCHIVE_URL")
        or _camulator_mod.CAMULATOR_GITHUB_ARCHIVE_URL
    )
    if progress_stream is not None:
        print(
            f"  [code] {_camulator_mod.CAMULATOR_CODE_REL_DIR} (~{_format_bytes(eff_ar_size)}) <- {eff_url}",
            file=progress_stream,
            flush=True,
        )

    code_parent = os.path.dirname(code_target) or "/"
    os.makedirs(code_parent, exist_ok=True)
    tmp_archive = os.path.join(
        code_parent,
        f".camulator_{_camulator_mod.CAMULATOR_CODE_COMMIT[:12]}_{uuid.uuid4().hex}.tar.gz",
    )
    try:
        download_url_artifact(
            url=eff_url,
            dst_path=tmp_archive,
            expected_size=eff_ar_size,
            timeout_seconds=timeout_seconds,
        )
        res = _camulator_mod.extract_and_verify_camulator_archive(
            tmp_archive,
            code_target,
            expected_archive_sha256=eff_ar_sha,
            expected_archive_size=eff_ar_size,
            expected_tree_sha256=expected_tree_sha256,
            expected_tree_files=expected_tree_files,
        )
        res["downloaded"] = True
        return res
    finally:
        if os.path.exists(tmp_archive):
            try:
                os.unlink(tmp_archive)
            except OSError:
                pass


def ensure_sdyff_code(
    dest_dir: str,
    *,
    archive_url: str | None = None,
    expected_archive_sha256: str | None = None,
    expected_archive_size: int | None = None,
    expected_tree_sha256: str | None = None,
    expected_tree_files: int | None = None,
    timeout_seconds: float = 120.0,
    progress_stream: Any = None,
) -> dict[str, Any]:
    """Ensure `<dest_dir>/code/spherical-dyffusion` matches the pinned 190-file Apache-2.0 tree."""
    from . import sdyff as _sdyff_mod

    override = os.environ.get("AIMIP_SDYFF_CODE")
    if override:
        safe_ovr = validate_safe_path(override, label="AIMIP_SDYFF_CODE")
        if not os.path.isdir(safe_ovr) or os.path.islink(safe_ovr):
            raise ValidationError(f"AIMIP_SDYFF_CODE is missing or symlinked: {safe_ovr}")
        return {"code_dir": safe_ovr, "downloaded": False, "override": True}

    eff_ar_sha = (
        _sdyff_mod.SDYFF_GITHUB_ARCHIVE_SHA256
        if expected_archive_sha256 is None
        else expected_archive_sha256
    )
    eff_ar_size = (
        _sdyff_mod.SDYFF_GITHUB_ARCHIVE_SIZE
        if expected_archive_size is None
        else int(expected_archive_size)
    )

    validated_dest = validate_safe_path(dest_dir, label="setup destination dir")
    code_target = validate_safe_path(
        os.path.join(validated_dest, _sdyff_mod.SDYFF_CODE_REL_DIR),
        label="Spherical DYffusion code directory",
    )
    if os.path.isdir(code_target) and not os.path.islink(code_target):
        try:
            info = _sdyff_mod.verify_sdyff_code_dir(
                validated_dest,
                expected_tree_sha256=expected_tree_sha256,
                expected_tree_files=expected_tree_files,
            )
            info["downloaded"] = False
            return info
        except ValidationError:
            pass

    eff_url = (
        archive_url
        or os.environ.get("AIMIP_SDYFF_ARCHIVE_URL")
        or _sdyff_mod.SDYFF_GITHUB_ARCHIVE_URL
    )
    if progress_stream is not None:
        print(
            f"  [code] {_sdyff_mod.SDYFF_CODE_REL_DIR} (~{_format_bytes(eff_ar_size)}) <- {eff_url}",
            file=progress_stream,
            flush=True,
        )

    code_parent = os.path.dirname(code_target) or "/"
    os.makedirs(code_parent, exist_ok=True)
    tmp_archive = os.path.join(
        code_parent,
        f".sdyff_{_sdyff_mod.SDYFF_CODE_COMMIT[:12]}_{uuid.uuid4().hex}.tar.gz",
    )
    try:
        download_url_artifact(
            url=eff_url,
            dst_path=tmp_archive,
            expected_size=eff_ar_size,
            timeout_seconds=timeout_seconds,
        )
        res = _sdyff_mod.extract_and_verify_sdyff_archive(
            tmp_archive,
            code_target,
            expected_archive_sha256=eff_ar_sha,
            expected_archive_size=eff_ar_size,
            expected_tree_sha256=expected_tree_sha256,
            expected_tree_files=expected_tree_files,
        )
        res["downloaded"] = True
        return res
    finally:
        if os.path.exists(tmp_archive):
            try:
                os.unlink(tmp_archive)
            except OSError:
                pass


def ensure_camulator_selected_assets(
    dest_dir: str,
    *,
    start_iso: str,
    scenario: str,
    auto_fetch: bool = True,
    lock_path: str | None = None,
    hf_downloader: Callable[..., str] | None = None,
    progress_stream: Any = None,
) -> dict[str, dict[str, Any]]:
    """Ensure the requested CAMulator IC `.pth` and forcing `.nc` are staged & lock-verified under *dest_dir*.

    Default `aimip fetch -m camulator` downloads only the 11 core/default files (~6.3 GB).
    Any of the 67 other published start-date ICs or the 10.46 GB `camulator-branch-1980-2014`
    forcing is fetched on demand here and verified against `aimip.lock`.
    """
    from .camulator import (
        CAMULATOR_HF_REPO,
        CAMULATOR_HF_REVISION,
        camulator_forcing_rel_path,
        camulator_ic_rel_path,
    )

    validated_dest = validate_safe_path(dest_dir, label="CAMulator data root")
    resolved_lock = resolve_lock_path(lock_path, dest_dir=validated_dest, must_exist=True)
    lock_payload = load_lock_file(resolved_lock)
    lock_files = lock_payload["files"]

    ic_rel = camulator_ic_rel_path(start_iso)
    forcing_rel = camulator_forcing_rel_path(scenario)
    hf_dl = hf_downloader or download_hf_artifact

    out: dict[str, dict[str, Any]] = {}
    for key, rel_path in (("ic", ic_rel), ("forcing", forcing_rel)):
        lock_entry = lock_files.get(rel_path)
        if not isinstance(lock_entry, Mapping):
            raise ValidationError(
                f"CAMulator asset {rel_path} is missing from lock file {resolved_lock}."
            )
        if _file_matches_lock_quiet(validated_dest, rel_path, lock_entry):
            out[key] = verify_file_against_lock(
                validated_dest, rel_path, lock_entry, role="output"
            )
            continue

        if not auto_fetch:
            raise ConfigError(
                f"CAMulator on-demand asset {rel_path} is not yet staged under {validated_dest}; "
                "re-run without `--no-auto-fetch` so aimip can download and lock-verify it."
            )

        with _exclusive_fetch_lock(validated_dest, progress_stream=progress_stream):
            if _file_matches_lock_quiet(validated_dest, rel_path, lock_entry):
                out[key] = verify_file_against_lock(
                    validated_dest, rel_path, lock_entry, role="output"
                )
                continue
            src_meta = lock_entry.get("source", {})
            repo = str(src_meta.get("repo") or CAMULATOR_HF_REPO)
            path_in_repo = str(src_meta.get("path") or rel_path.split("/", 2)[-1])
            rev = str(src_meta.get("revision") or CAMULATOR_HF_REVISION)
            exp_sz = lock_entry.get("size") if isinstance(lock_entry.get("size"), int) else None
            if progress_stream is not None:
                print(
                    f"  [on-demand] {rel_path} ({_format_bytes(exp_sz)}) <- {repo}/{path_in_repo}",
                    file=progress_stream,
                    flush=True,
                )
            hf_home = validate_safe_path(
                os.environ.get("HF_HOME") or os.path.join(validated_dest, ".hf"),
                label="HF_HOME",
            )
            os.environ.setdefault("HF_HOME", hf_home)
            dst_abs = os.path.join(validated_dest, rel_path)
            hf_dl(
                repo=repo,
                path_in_repo=path_in_repo,
                revision=rev,
                dst_path=dst_abs,
                hf_home=hf_home,
            )
            out[key] = verify_file_against_lock(
                validated_dest, rel_path, lock_entry, role="output"
            )
    return out


def verify_tree_against_lock(
    dest_dir: str,
    lock_payload: Mapping[str, Any],
    *,
    selected_models: Sequence[str] = CANONICAL_MODELS,
    years: SetupYearConfig | None = None,
    only_downloaded: bool = False,
    strict_unpinned: bool = True,
) -> list[dict[str, Any]]:
    """Verify every required file for *selected_models* under *dest_dir* against *lock_payload*."""
    validated_dest = validate_safe_path(dest_dir, label="setup destination dir")
    lock_files = lock_payload.get("files")
    if not isinstance(lock_files, Mapping):
        raise ConfigError("Lock payload missing 'files' mapping.")

    catalog = build_artifact_catalog(years=years)
    wanted = set(selected_models)
    verified: list[dict[str, Any]] = []

    for rel_path, spec in catalog.items():
        if only_downloaded:
            if not (set(spec.models_needed) & wanted):
                continue
            if spec.source_kind == "derived":
                continue
        else:
            is_model_output = bool(set(spec.models_output) & wanted)
            is_present_raw = (
                spec.role == "raw"
                and bool(set(spec.models_needed) & wanted)
                and os.path.isfile(os.path.join(validated_dest, rel_path))
            )
            if not (is_model_output or is_present_raw):
                continue

        if rel_path not in lock_files:
            raise ValidationError(
                f"Lock verification failed for {rel_path}: entry missing from aimip.lock."
            )
        entry = lock_files[rel_path]
        if not isinstance(entry, Mapping):
            raise ConfigError(f"Lock entry for {rel_path} must be an object.")

        info = verify_file_against_lock(
            validated_dest,
            rel_path,
            entry,
            role=spec.role,
            strict_unpinned=strict_unpinned,
        )
        verified.append(info)

    return verified


def _build_registry_path_lookup(dest_dir: str) -> dict[str, str]:
    """Map canonical relative paths (`models/...`, `inputs/...`) to external paths in `<dest>/registry.json`."""
    reg_file = os.path.join(dest_dir, "registry.json")
    if not os.path.isfile(reg_file):
        return {}
    try:
        with open(reg_file, encoding="utf-8") as fh:
            reg = json.load(fh)
    except (OSError, json.JSONDecodeError):
        return {}

    models = reg.get("models", [])
    if not isinstance(models, list):
        return {}

    ckpt_rel_by_model = {
        "ace2-era5": "models/ace2-era5/ace2_era5_ckpt.tar",
        "ace2-shield": "models/ace2-shield/ace2_shield_ckpt.tar",
        "ace2s-shield-plus": "models/ace2s-shield-plus/ace2s_shield_plus_ckpt.tar",
        "ace2-eamv3": "models/ace2-eamv3/ace2_EAMv3_ckpt.tar",
        "hiro-ace": "models/hiro-ace/ACE2S.ckpt",
    }
    ic_rel_by_model = {
        "ace2-shield": ["models/ace2s-shield-plus/ic.nc"],
        "ace2s-shield-plus": ["models/ace2s-shield-plus/ic.nc"],
        "ace2-eamv3": ["inputs/ace2-eamv3/ic/ic_1994.nc"],
        "hiro-ace": ["inputs/hiro-ace/ic/ic_1994.nc"],
        "ace2-era5": [
            "inputs/ace2-era5/ic/ic_1979.nc",
            "inputs/ace2-era5/ic/ic_2001.nc",
        ],
    }

    mapping: dict[str, str] = {}
    for m in models:
        if not isinstance(m, dict):
            continue
        name = str(m.get("name") or "")
        ckpt = m.get("checkpoint")
        if name in ckpt_rel_by_model and isinstance(ckpt, str) and not ckpt.startswith("$"):
            if os.path.isfile(ckpt):
                mapping[ckpt_rel_by_model[name]] = ckpt
        ics = m.get("ic_files")
        if name in ic_rel_by_model and isinstance(ics, list):
            for rel_ic, raw_ic in zip(ic_rel_by_model[name], ics):
                if isinstance(raw_ic, str) and not raw_ic.startswith("$") and os.path.isfile(raw_ic):
                    mapping[rel_ic] = raw_ic
    return mapping


def _locate_existing_artifact(
    dest_dir: str,
    spec: ArtifactSpec,
    reg_lookup: Mapping[str, str],
    search_dirs: Sequence[str],
) -> str | None:
    """Locate *spec* in *dest_dir*, `<dest>/registry.json` external paths, or `--search-dir` roots."""
    direct = os.path.join(dest_dir, spec.rel_path)
    if os.path.isfile(direct):
        return direct

    if spec.rel_path in reg_lookup and os.path.isfile(reg_lookup[spec.rel_path]):
        return reg_lookup[spec.rel_path]

    basename = os.path.basename(spec.rel_path)
    for root in search_dirs:
        candidates = [os.path.join(root, spec.rel_path), os.path.join(root, basename)]
        if spec.hf_repo and spec.hf_path:
            for hf_rel in (
                (spec.hf_path, spec.hf_path.replace(":", ""))
                if ":" in spec.hf_path
                else (spec.hf_path,)
            ):
                candidates.append(os.path.join(root, spec.hf_repo, hf_rel))
                candidates.append(
                    os.path.join(root, spec.hf_repo.replace("/", "__"), hf_rel)
                )
        for cand in candidates:
            if os.path.isfile(cand):
                return cand
    return None


def write_lock_from_tree(
    dest_dir: str,
    lock_out_path: str,
    *,
    selected_models: Sequence[str] = CANONICAL_MODELS,
    years: SetupYearConfig | None = None,
    baseline_lock: Mapping[str, Any] | None = None,
    search_dirs: Sequence[str] = (),
) -> dict[str, Any]:
    """Generate or refresh `aimip.lock` by hashing files in *dest_dir* (`--write-lock`)."""
    validated_dest = validate_safe_path(dest_dir, label="setup --dest dir")
    validated_lock_out = validate_safe_path(lock_out_path, label="lock output path")
    validated_search = [
        validate_safe_path(d, label="--search-dir") for d in search_dirs
    ]

    catalog = build_artifact_catalog(years=years)
    wanted = set(selected_models)
    base_files: Mapping[str, Any] = (
        baseline_lock.get("files", {})
        if isinstance(baseline_lock, Mapping) and isinstance(baseline_lock.get("files"), Mapping)
        else {}
    )
    reg_lookup = _build_registry_path_lookup(validated_dest)

    computed_shas: dict[str, str] = {}
    # Seed `computed_shas` with any remote CO2 input hashes already recorded in baseline lock
    for existing_entry in base_files.values():
        if isinstance(existing_entry, Mapping):
            src_blk = existing_entry.get("source")
            if isinstance(src_blk, Mapping) and isinstance(src_blk.get("inputs"), Mapping):
                for k_in, v_sha in src_blk["inputs"].items():
                    if isinstance(v_sha, str) and v_sha and v_sha != SHA256_TBD:
                        computed_shas[k_in] = v_sha

    new_files: dict[str, dict[str, Any]] = {}
    hashed_count = 0
    preserved_count = 0

    # Pass 1: hash all present files (raw + output) so Pass 2 can populate `derived.inputs` shas
    resolved_paths: dict[str, str | None] = {}
    for rel_path, spec in catalog.items():
        if not (set(spec.models_needed) & wanted or set(spec.models_output) & wanted):
            continue
        found = _locate_existing_artifact(
            validated_dest, spec, reg_lookup, validated_search
        )
        resolved_paths[rel_path] = found
        if found is not None:
            computed_shas[rel_path] = _sha256_source_file(found)
        elif rel_path in base_files and isinstance(base_files[rel_path], Mapping):
            prev_sha = base_files[rel_path].get("sha256")
            if isinstance(prev_sha, str) and prev_sha:
                computed_shas[rel_path] = prev_sha

    # Pass 2: assemble ordered lock entries
    for rel_path, spec in catalog.items():
        if not (set(spec.models_needed) & wanted or set(spec.models_output) & wanted):
            continue
        found = resolved_paths[rel_path]
        prev_entry = (
            base_files[rel_path]
            if rel_path in base_files and isinstance(base_files[rel_path], Mapping)
            else None
        )

        if found is not None:
            f_size = os.path.getsize(found)
            f_sha = computed_shas[rel_path]
            z_md5 = md5_file(found) if spec.source_kind == "zenodo" else MD5_TBD
            src_dict = spec.default_source_dict(
                known_shas=computed_shas, zenodo_md5=z_md5
            )
            # Preserve custom HF revision from baseline lock if present
            if (
                spec.source_kind == "hf"
                and prev_entry is not None
                and isinstance(prev_entry.get("source"), Mapping)
                and prev_entry["source"].get("revision")
            ):
                src_dict["revision"] = prev_entry["source"]["revision"]
            new_files[rel_path] = {
                "size": f_size,
                "sha256": f_sha,
                "source": src_dict,
            }
            hashed_count += 1
        elif prev_entry is not None:
            copied = json.loads(json.dumps(prev_entry))
            if (
                spec.source_kind == "derived"
                and isinstance(copied.get("source"), dict)
                and isinstance(copied["source"].get("inputs"), dict)
            ):
                for in_ref in spec.input_refs:
                    if in_ref in computed_shas:
                        copied["source"]["inputs"][in_ref] = computed_shas[in_ref]
            new_files[rel_path] = copied
            preserved_count += 1
        elif spec.role == "raw":
            new_files[rel_path] = {
                "size": None,
                "sha256": SHA256_TBD,
                "source": spec.default_source_dict(known_shas=computed_shas),
            }
        else:
            raise ValidationError(
                f"Cannot write lock: required output file {rel_path} not found under {validated_dest}."
            )

    # Preserve optional/auxiliary locked entries (e.g. standalone uncoupled tars or HiROv2.ckpt)
    # from `baseline_lock` when generating a full all-model production lock.
    if wanted == set(CANONICAL_MODELS) and (years is None or years == SetupYearConfig()):
        for extra_rel, extra_entry in base_files.items():
            if extra_rel not in new_files and isinstance(extra_entry, Mapping):
                new_files[extra_rel] = json.loads(json.dumps(extra_entry))
                preserved_count += 1

    lock_doc: dict[str, Any] = {
        "schema_version": 1,
        "generated_by": "aimip setup --write-lock",
        "files": new_files,
    }

    parent = os.path.dirname(validated_lock_out) or "/"
    os.makedirs(parent, exist_ok=True)
    tmp_path = os.path.join(parent, f".lock_{uuid.uuid4().hex}.json.tmp")
    try:
        with open(tmp_path, "w", encoding="utf-8") as fh:
            json.dump(lock_doc, fh, indent=2)
            fh.write("\n")
        os.replace(tmp_path, validated_lock_out)
    finally:
        if os.path.exists(tmp_path):
            try:
                os.unlink(tmp_path)
            except OSError:
                pass

    return {
        "lock_path": validated_lock_out,
        "dest_dir": validated_dest,
        "hashed_files": hashed_count,
        "preserved_baseline_files": preserved_count,
        "total_entries": len(new_files),
    }


def _write_data_registry(
    dest_dir: str,
    verified_models: Sequence[str],
    *,
    failed_models: Sequence[str] = (),
    catalog: Mapping[str, ArtifactSpec] | None = None,
) -> str:
    """Write `<dest_dir>/registry.json` containing verified models (preserving prior verified entries).

    Refreshes `checkpoint_sha256`, `ic_sha256`, `ocean_ic_sha256`, and `required_files` from the
    verified staged tree in *dest_dir* so custom `--lock` mirrors and `--accept-derived` builds
    record exact on-disk checksums (matching `registry.default.json` 100% on production `aimip.lock`).
    """
    reg_path = validate_safe_path(
        os.path.join(dest_dir, "registry.json"), label="data registry.json"
    )
    existing_by_name: dict[str, dict[str, Any]] = {}
    if os.path.isfile(reg_path):
        try:
            with open(reg_path, encoding="utf-8") as fh:
                prev_doc = json.load(fh)
            if isinstance(prev_doc, Mapping) and isinstance(prev_doc.get("models"), list):
                for item in prev_doc["models"]:
                    if isinstance(item, Mapping) and item.get("name") in CANONICAL_MODELS:
                        existing_by_name[str(item["name"])] = dict(item)
        except (OSError, json.JSONDecodeError):
            existing_by_name = {}

    verified_set = set(verified_models)
    merged_set = (set(existing_by_name.keys()) | verified_set) - set(failed_models)
    ordered_models = tuple(m for m in CANONICAL_MODELS if m in merged_set)
    payload = build_default_registry_payload(
        selected_models=ordered_models, root_placeholder=DATA_PLACEHOLDER
    )

    placeholder_prefix = f"{DATA_PLACEHOLDER}/"
    reconciled_models: list[dict[str, Any]] = []
    for m_entry in payload["models"]:
        m_name = str(m_entry["name"])
        if m_name not in verified_set and m_name in existing_by_name:
            reconciled_models.append(existing_by_name[m_name])
            continue

        ckpt_placeholder = str(m_entry.get("checkpoint") or "")
        if ckpt_placeholder.startswith(placeholder_prefix):
            ckpt_abs = os.path.join(dest_dir, ckpt_placeholder[len(placeholder_prefix) :])
            if os.path.isfile(ckpt_abs):
                m_entry["checkpoint_sha256"] = sha256_file(ckpt_abs)

        for sha_map_key in ("ic_sha256", "ocean_ic_sha256"):
            sha_map = m_entry.get(sha_map_key)
            if isinstance(sha_map, dict):
                for p_key in list(sha_map.keys()):
                    if isinstance(p_key, str) and p_key.startswith(placeholder_prefix):
                        ic_abs = os.path.join(dest_dir, p_key[len(placeholder_prefix) :])
                        if os.path.isfile(ic_abs):
                            sha_map[p_key] = sha256_file(ic_abs)

        if catalog is not None:
            m_entry["required_files"] = [
                rel
                for rel, sp in catalog.items()
                if m_name in sp.models_output and not sp.pin_after_first_fetch
            ]
        reconciled_models.append(m_entry)

    payload["models"] = reconciled_models

    parent = os.path.dirname(reg_path) or "/"
    os.makedirs(parent, exist_ok=True)
    tmp_path = os.path.join(parent, f".reg_{uuid.uuid4().hex}.json.tmp")
    try:
        with open(tmp_path, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2)
            fh.write("\n")
        strip_group_other_write(tmp_path)
        os.replace(tmp_path, reg_path)
    finally:
        if os.path.exists(tmp_path):
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
    return reg_path


def _write_derived_overrides(
    dest_dir: str,
    *,
    lock_path: str,
    overrides: Sequence[Mapping[str, Any]],
    selected_models: Sequence[str] = (),
) -> str | None:
    """Reconcile `<dest_dir>/derived-overrides.json` across `--accept-derived` and strict retries.

    Preserves overrides belonging to unselected models, replaces entries for *selected_models*
    with *overrides*, and removes stale entries when a strict retry re-evaluates *selected_models*.
    """
    overrides_path = validate_safe_path(
        os.path.join(dest_dir, "derived-overrides.json"),
        label="derived-overrides.json",
    )
    selected_set = set(selected_models)
    retained_prior: list[dict[str, Any]] = []
    file_existed = os.path.isfile(overrides_path)
    if file_existed:
        try:
            with open(overrides_path, encoding="utf-8") as fh:
                prev_doc = json.load(fh)
            if isinstance(prev_doc, Mapping) and isinstance(prev_doc.get("overrides"), list):
                for item in prev_doc["overrides"]:
                    if not isinstance(item, Mapping):
                        continue
                    item_models = item.get("models")
                    if isinstance(item_models, list):
                        remaining_models = [
                            str(m) for m in item_models if str(m) not in selected_set
                        ]
                        if remaining_models:
                            updated = dict(item)
                            updated["models"] = remaining_models
                            retained_prior.append(updated)
        except (OSError, json.JSONDecodeError):
            retained_prior = []

    merged_overrides = retained_prior + [dict(o) for o in overrides]
    if not merged_overrides and not file_existed:
        return None

    payload = {
        "schema_version": 1,
        "generated_by": "aimip setup --accept-derived",
        "lock_path": lock_path,
        "overrides": merged_overrides,
    }
    parent = os.path.dirname(overrides_path) or "/"
    os.makedirs(parent, exist_ok=True)
    tmp_path = os.path.join(parent, f".ovr_{uuid.uuid4().hex}.json.tmp")
    try:
        with open(tmp_path, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2)
            fh.write("\n")
        strip_group_other_write(tmp_path)
        os.replace(tmp_path, overrides_path)
    finally:
        if os.path.exists(tmp_path):
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
    return overrides_path if overrides else None


def _format_bytes(num_bytes: int | None) -> str:
    """Format a byte count in human-readable IEC/decimal units (e.g. '1.70 GiB (1,823,641,466 B)')."""
    if num_bytes is None or num_bytes < 0:
        return "unknown size"
    if num_bytes < 1024:
        return f"{num_bytes} B"
    if num_bytes < 1024 * 1024:
        return f"{num_bytes / 1024.0:.1f} KiB"
    if num_bytes < 1024 * 1024 * 1024:
        return f"{num_bytes / (1024.0 ** 2):.1f} MiB"
    return f"{num_bytes / (1024.0 ** 3):.2f} GiB"


@contextlib.contextmanager
def _exclusive_fetch_lock(
    dest_dir: str,
    *,
    progress_stream: Any = None,
) -> Iterator[str]:
    """Acquire an exclusive advisory `fcntl.flock` on `<dest_dir>/.aimip-fetch.lock`.

    Concurrency-safe across multiple processes or users sharing `$AIMIP_DATA`:
    uses `O_CREAT | O_RDONLY` so `flock(LOCK_EX)` succeeds even when another user in
    the shared group created `.aimip-fetch.lock` with `0644`.
    """
    os.makedirs(dest_dir, exist_ok=True)
    flock_path = validate_safe_path(
        os.path.join(dest_dir, ".aimip-fetch.lock"), label="fetch lock file"
    )
    try:
        fd = os.open(flock_path, os.O_CREAT | os.O_RDONLY, 0o664)
    except PermissionError:
        fd = os.open(flock_path, os.O_RDONLY)

    try:
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            if progress_stream is not None:
                print(
                    f"aimip fetch: waiting for concurrent fetch lock ({flock_path})...",
                    file=progress_stream,
                    flush=True,
                )
            fcntl.flock(fd, fcntl.LOCK_EX)
        yield flock_path
    finally:
        try:
            fcntl.flock(fd, fcntl.LOCK_UN)
        finally:
            os.close(fd)


def run_setup(
    *,
    models: str | Sequence[str] = "all",
    dest_dir: str | None = None,
    playground_dir: str | None = None,
    lock_path: str | None = None,
    only_download: bool = False,
    verify_only: bool = False,
    write_lock: bool = False,
    search_dirs: Sequence[str] = (),
    force_rebuild: bool = False,
    accept_derived: bool = False,
    strict_unpinned: bool = True,
    years: SetupYearConfig | None = None,
    hf_downloader: Callable[..., str] | None = None,
    zenodo_downloader: Callable[..., str] | None = None,
    arches_code_fetcher: Callable[..., dict[str, Any]] | None = None,
    md15_code_fetcher: Callable[..., dict[str, Any]] | None = None,
    camulator_code_fetcher: Callable[..., dict[str, Any]] | None = None,
    sdyff_code_fetcher: Callable[..., dict[str, Any]] | None = None,
    co2_year_reader: Callable[[int], Any] | None = None,
    progress_stream: Any = None,
) -> dict[str, Any]:
    """Execute `aimip setup` across download, deterministic derivation, and lock verification."""
    selected = (
        parse_selected_models(models)
        if isinstance(models, str)
        else tuple(models)
    )
    mode_flags = sum(bool(x) for x in (only_download, verify_only, write_lock))
    if mode_flags > 1:
        raise ConfigError(
            "Choose at most one of `--only-download`, `--verify-only`, or `--write-lock`."
        )
    if accept_derived and mode_flags > 0:
        raise ConfigError(
            "`--accept-derived` is only valid during full `aimip setup` "
            "(cannot be combined with `--only-download`, `--verify-only`, or `--write-lock`)."
        )

    resolved_dest = resolve_setup_dest_dir(dest_dir, playground_dir=playground_dir)
    yc = years or SetupYearConfig()

    if write_lock:
        target_lock = resolve_lock_path(
            lock_path, dest_dir=resolved_dest, must_exist=False
        )
        baseline: dict[str, Any] | None = None
        if os.path.isfile(target_lock):
            baseline = load_lock_file(target_lock)
        else:
            try:
                default_shipped = resolve_lock_path(
                    None, dest_dir=resolved_dest, must_exist=True
                )
                baseline = load_lock_file(default_shipped)
            except ConfigError:
                baseline = None

        res = write_lock_from_tree(
            resolved_dest,
            target_lock,
            selected_models=selected,
            years=yc,
            baseline_lock=baseline,
            search_dirs=search_dirs,
        )
        res["mode"] = "write-lock"
        res["models"] = list(selected)
        return res

    resolved_lock = resolve_lock_path(
        lock_path, dest_dir=resolved_dest, must_exist=True
    )
    lock_payload = load_lock_file(resolved_lock)
    lock_files = lock_payload["files"]

    if verify_only:
        verified = verify_tree_against_lock(
            resolved_dest,
            lock_payload,
            selected_models=selected,
            years=yc,
            only_downloaded=False,
            strict_unpinned=strict_unpinned,
        )
        if "archesclimate-ssp" in selected and arches_code_fetcher is None:
            verify_arches_code_dir(resolved_dest)
        if (
            {"md-1.5", "md-1.5-aimip"} & set(selected)
            and md15_code_fetcher is None
            and not os.environ.get("AIMIP_MD15_CODE")
        ):
            from .md15 import verify_md15_code_dir

            verify_md15_code_dir(resolved_dest)
        if (
            "camulator" in selected
            and camulator_code_fetcher is None
            and not os.environ.get("AIMIP_CAMULATOR_CODE")
        ):
            from .camulator import verify_camulator_code_dir

            verify_camulator_code_dir(resolved_dest)
        if (
            "spherical-dyffusion" in selected
            and sdyff_code_fetcher is None
            and not os.environ.get("AIMIP_SDYFF_CODE")
        ):
            from .sdyff import verify_sdyff_code_dir

            verify_sdyff_code_dir(resolved_dest)
        return {
            "mode": "verify-only",
            "dest_dir": resolved_dest,
            "lock_path": resolved_lock,
            "models": list(selected),
            "verified_files": len(verified),
        }

    with _exclusive_fetch_lock(resolved_dest, progress_stream=progress_stream):
        # Reload lock payload inside the exclusive lock so concurrent callers see fresh state.
        lock_payload = load_lock_file(resolved_lock)
        lock_files = lock_payload["files"]

        hf_home = validate_safe_path(
            os.environ.get("HF_HOME") or os.path.join(resolved_dest, ".hf"),
            label="HF_HOME",
        )
        os.environ.setdefault("HF_HOME", hf_home)

        hf_dl = hf_downloader or download_hf_artifact
        zen_dl = zenodo_downloader or download_zenodo_artifact

        catalog = build_artifact_catalog(years=yc)
        wanted = set(selected)

        downloaded_count = 0
        downloaded_bytes = 0
        skipped_download_count = 0

        if progress_stream is not None:
            pending_downloads: list[tuple[str, int | None]] = []
            present_sources = 0
            for rel_p, sp in catalog.items():
                if not (set(sp.models_needed) & wanted) or sp.source_kind not in ("hf", "zenodo"):
                    continue
                le = lock_files.get(rel_p)
                if isinstance(le, Mapping) and _file_matches_lock_quiet(resolved_dest, rel_p, le):
                    present_sources += 1
                else:
                    sz = le.get("size") if isinstance(le, Mapping) and isinstance(le.get("size"), int) else None
                    pending_downloads.append((rel_p, sz))
            known_total = sum(sz for _, sz in pending_downloads if isinstance(sz, int))
            models_label = ", ".join(selected)
            if pending_downloads:
                print(
                    f"aimip fetch: [{models_label}] downloading {len(pending_downloads)} missing file(s) "
                    f"({_format_bytes(known_total)}, {known_total} B) into {resolved_dest}...",
                    file=progress_stream,
                    flush=True,
                )
            else:
                print(
                    f"aimip fetch: [{models_label}] all {present_sources} source file(s) already staged "
                    f"in {resolved_dest}; verifying against lock...",
                    file=progress_stream,
                    flush=True,
                )
            pending_total_count = len(pending_downloads)
        else:
            pending_total_count = 0

        md15_code_info: dict[str, Any] | None = None
        md15_code_err: str | None = None
        md15_wanted = tuple(m for m in ("md-1.5", "md-1.5-aimip") if m in wanted)
        if md15_wanted:
            md15_fn = md15_code_fetcher or ensure_md15_code_and_weights
            try:
                md15_code_info = md15_fn(resolved_dest, progress_stream=progress_stream)
            except (ValidationError, ConfigError) as exc:
                if only_download:
                    if os.path.isfile(os.path.join(resolved_dest, "registry.json")):
                        _write_data_registry(resolved_dest, (), failed_models=md15_wanted)
                    raise
                md15_code_err = str(exc)

        # Step 1: Fetch and verify all direct HF, Zenodo, and HTTPS URL inputs needed for `selected`
        for rel_path, spec in catalog.items():
            if not (set(spec.models_needed) & wanted):
                continue
            if spec.source_kind not in ("hf", "zenodo", "url"):
                continue
            affected_step1 = tuple(m for m in selected if m in spec.models_needed)
            if rel_path not in lock_files:
                if not only_download or os.path.isfile(os.path.join(resolved_dest, "registry.json")):
                    _write_data_registry(resolved_dest, (), failed_models=affected_step1)
                raise ValidationError(
                    f"Setup aborted: {rel_path} is missing from lock file {resolved_lock}."
                )
            lock_entry = lock_files[rel_path]
            dst_abs = os.path.join(resolved_dest, rel_path)

            if _file_matches_lock_quiet(resolved_dest, rel_path, lock_entry):
                skipped_download_count += 1
                continue

            exp_entry_size = (
                int(lock_entry["size"])
                if isinstance(lock_entry.get("size"), int)
                else None
            )
            if progress_stream is not None:
                if spec.source_kind == "hf":
                    src_label = f"{spec.hf_repo}/{spec.hf_path}"
                elif spec.source_kind == "zenodo":
                    src_label = f"zenodo:{spec.zenodo_record}/{spec.zenodo_file}"
                else:
                    src_label = str(spec.url)
                print(
                    f"  [{downloaded_count + 1}/{max(1, pending_total_count)}] {rel_path} "
                    f"({_format_bytes(exp_entry_size)}) <- {src_label}",
                    file=progress_stream,
                    flush=True,
                )

            try:
                if spec.source_kind == "hf":
                    src_meta = lock_entry.get("source", {})
                    rev = (
                        src_meta.get("revision")
                        if isinstance(src_meta, Mapping) and src_meta.get("revision")
                        else (spec.hf_revision or REVISION_TBD)
                    )
                    assert spec.hf_repo is not None and spec.hf_path is not None
                    # repo_type only for dataset repos, so injected test downloaders keep their signature
                    hf_dl(
                        repo=spec.hf_repo,
                        path_in_repo=spec.hf_path,
                        revision=str(rev),
                        dst_path=dst_abs,
                        hf_home=hf_home,
                        **(
                            {"repo_type": spec.hf_repo_type}
                            if spec.hf_repo_type and spec.hf_repo_type != "model"
                            else {}
                        ),
                    )
                elif spec.source_kind == "zenodo":
                    src_meta = lock_entry.get("source", {})
                    exp_md5 = (
                        str(src_meta.get("md5"))
                        if isinstance(src_meta, Mapping) and src_meta.get("md5")
                        else None
                    )
                    assert (
                        spec.zenodo_record is not None
                        and spec.zenodo_file is not None
                        and spec.zenodo_url is not None
                    )
                    zen_dl(
                        record=spec.zenodo_record,
                        filename=spec.zenodo_file,
                        url=str(
                            src_meta.get("url")
                            if isinstance(src_meta, Mapping) and src_meta.get("url")
                            else spec.zenodo_url
                        ),
                        dst_path=dst_abs,
                        expected_size=exp_entry_size,
                        expected_md5=exp_md5,
                    )
                else:
                    src_meta = lock_entry.get("source", {})
                    target_url = str(
                        src_meta.get("url")
                        if isinstance(src_meta, Mapping) and src_meta.get("url")
                        else spec.url
                    )
                    download_url_artifact(
                        url=target_url,
                        dst_path=dst_abs,
                        expected_size=exp_entry_size,
                    )

                ver_step1 = verify_file_against_lock(
                    resolved_dest,
                    rel_path,
                    lock_entry,
                    role=spec.role,
                    strict_unpinned=strict_unpinned,
                )
            except ValidationError:
                if not only_download or os.path.isfile(os.path.join(resolved_dest, "registry.json")):
                    _write_data_registry(resolved_dest, (), failed_models=affected_step1)
                raise
            downloaded_count += 1
            downloaded_bytes += int(ver_step1["size"])

        arches_code_info: dict[str, Any] | None = None
        arches_code_err: str | None = None
        if "archesclimate-ssp" in wanted:
            arches_fn = arches_code_fetcher or ensure_arches_code
            try:
                arches_code_info = arches_fn(
                    resolved_dest, progress_stream=progress_stream
                )
            except (ValidationError, ConfigError) as exc:
                if only_download:
                    if os.path.isfile(os.path.join(resolved_dest, "registry.json")):
                        _write_data_registry(
                            resolved_dest, (), failed_models=("archesclimate-ssp",)
                        )
                    raise
                arches_code_err = str(exc)

        camulator_code_info: dict[str, Any] | None = None
        camulator_code_err: str | None = None
        if "camulator" in wanted:
            cam_fn = camulator_code_fetcher or ensure_camulator_code
            try:
                camulator_code_info = cam_fn(
                    resolved_dest, progress_stream=progress_stream
                )
            except (ValidationError, ConfigError) as exc:
                if only_download:
                    if os.path.isfile(os.path.join(resolved_dest, "registry.json")):
                        _write_data_registry(
                            resolved_dest, (), failed_models=("camulator",)
                        )
                    raise
                camulator_code_err = str(exc)

        sdyff_code_info: dict[str, Any] | None = None
        sdyff_code_err: str | None = None
        if "spherical-dyffusion" in wanted:
            sdyff_fn = sdyff_code_fetcher or ensure_sdyff_code
            try:
                sdyff_code_info = sdyff_fn(
                    resolved_dest, progress_stream=progress_stream
                )
            except (ValidationError, ConfigError) as exc:
                if only_download:
                    if os.path.isfile(os.path.join(resolved_dest, "registry.json")):
                        _write_data_registry(
                            resolved_dest, (), failed_models=("spherical-dyffusion",)
                        )
                    raise
                sdyff_code_err = str(exc)

        if only_download:
            verified = verify_tree_against_lock(
                resolved_dest,
                lock_payload,
                selected_models=selected,
                years=yc,
                only_downloaded=True,
                strict_unpinned=strict_unpinned,
            )
            return {
                "mode": "only-download",
                "dest_dir": resolved_dest,
                "lock_path": resolved_lock,
                "models": list(selected),
                "downloaded_files": downloaded_count,
                "downloaded_bytes": downloaded_bytes,
                "skipped_existing_files": skipped_download_count,
                "verified_files": len(verified),
            }

        # Step 2: Run deterministic builders in dependency order, rebuilding whenever missing
        # or not matching the lock (or when `force_rebuild=True`).
        built_count = 0

        def _needs_build(rel_p: str) -> bool:
            if force_rebuild:
                return True
            return not _file_matches_lock_quiet(
                resolved_dest, rel_p, lock_files.get(rel_p)
            )

        from .emu_catalog import AW_ZIP_MEMBERS, AW_ZIP_REL

        for aw_rel, (aw_member, _size, _sha, aw_models) in AW_ZIP_MEMBERS.items():
            if set(aw_models) & wanted and _needs_build(aw_rel):
                extract_zip_member(
                    os.path.join(resolved_dest, AW_ZIP_REL),
                    aw_member,
                    os.path.join(resolved_dest, aw_rel),
                )
                built_count += 1

        needs_oos = bool({"ace2-shield", "ace2s-shield-plus", "hiro-ace"} & wanted)
        if needs_oos and _needs_build(CO2_TABLE_REL_PATH):
            era5_rev = HF_REVISIONS["allenai/ACE2-ERA5"]
            era5_ckpt_entry = lock_files.get("models/ace2-era5/ace2_era5_ckpt.tar")
            if isinstance(era5_ckpt_entry, Mapping) and isinstance(
                era5_ckpt_entry.get("source"), Mapping
            ):
                era5_rev = str(era5_ckpt_entry["source"].get("revision") or era5_rev)
            build_co2_table(
                os.path.join(resolved_dest, CO2_TABLE_REL_PATH),
                revision=era5_rev,
                local_era5_forcing_dir=os.path.join(
                    resolved_dest, "inputs/ace2-era5/amip-hist"
                ),
                y0=yc.co2_y0,
                y1=yc.co2_y1,
                last_upstream=yc.co2_last_upstream,
                year_reader=co2_year_reader,
                validate_reference_stats=yc.co2_validate_reference_stats,
                force=True,
            )
            built_count += 1

        if "ace2-eamv3" in wanted:
            ic_rel = "inputs/ace2-eamv3/ic/ic_1994.nc"
            if _needs_build(ic_rel):
                build_eamv3_ic(
                    os.path.join(
                        resolved_dest, "raw/ace2-eamv3/initial_conditions/1971010100.nc"
                    ),
                    os.path.join(resolved_dest, ic_rel),
                    force=True,
                )
                built_count += 1

            eam_years_to_build = [
                y
                for y in yc.derived_hist_years
                if _needs_build(f"inputs/ace2-eamv3/amip-hist/{y}.nc")
            ]
            if eam_years_to_build:
                build_eamv3_forcing(
                    raw_eamv3_1971_path=os.path.join(
                        resolved_dest, "raw/ace2-eamv3/forcing_data/1971.nc"
                    ),
                    era5_amip_hist_dir=os.path.join(
                        resolved_dest, "inputs/ace2-era5/amip-hist"
                    ),
                    out_dir=os.path.join(resolved_dest, "inputs/ace2-eamv3/amip-hist"),
                    years=eam_years_to_build,
                    force=True,
                )
                built_count += len(eam_years_to_build)

        if "hiro-ace" in wanted:
            hiro_ic_rel = "inputs/hiro-ace/ic/ic_1994.nc"
            if _needs_build(hiro_ic_rel):
                build_hiro_ic(
                    os.path.join(
                        resolved_dest, "raw/hiro-ace/initial_conditions/ic_2014.nc"
                    ),
                    os.path.join(resolved_dest, hiro_ic_rel),
                    force=True,
                )
                built_count += 1

            hiro_years_to_build = [
                y
                for y in yc.derived_hist_years
                if _needs_build(f"inputs/hiro-ace/amip-hist/forcing_{y}.nc")
            ]
            if hiro_years_to_build:
                build_hiro_forcing(
                    raw_hiro_forcing_2014_path=os.path.join(
                        resolved_dest, "raw/hiro-ace/forcing_data/forcing_2014.nc"
                    ),
                    shield_amip_hist_dir=os.path.join(
                        resolved_dest, "inputs/shield/amip-hist"
                    ),
                    out_dir=os.path.join(resolved_dest, "inputs/hiro-ace/amip-hist"),
                    years=hiro_years_to_build,
                    mask_reference_year=yc.hiro_mask_reference_year,
                    force=True,
                )
                built_count += len(hiro_years_to_build)

        if {"ace2-shield", "ace2s-shield-plus"} & wanted:
            shield_oos_years = [
                y
                for y in yc.oos_years
                if _needs_build(f"inputs/shield/amip-oos/forcing_{y}.nc")
            ]
            need_shield_bridge = (
                yc.oos_bridge_year is not None
                and _needs_build(f"inputs/shield/amip-oos/forcing_{yc.oos_bridge_year}.nc")
            )
            if shield_oos_years or need_shield_bridge:
                build_oos_forcing(
                    "shieldplus",
                    hist_forcing_dir=os.path.join(
                        resolved_dest, "inputs/shield/amip-hist"
                    ),
                    aimip_monthly_nc_path=os.path.join(resolved_dest, ZENODO_AIMIP_REL_PATH),
                    co2_nc_path=os.path.join(resolved_dest, CO2_TABLE_REL_PATH),
                    out_dir=os.path.join(resolved_dest, "inputs/shield/amip-oos"),
                    years=shield_oos_years,
                    bridge_year=yc.oos_bridge_year if need_shield_bridge else None,
                    reference_nonleap_year=yc.oos_ref_nonleap_year,
                    reference_leap_year=yc.oos_ref_leap_year,
                    anchor_t0=yc.oos_anchor_t0,
                    anchor_t1=yc.oos_anchor_t1,
                    expected_nonleap_steps=yc.oos_expected_nonleap_steps,
                    expected_leap_steps=yc.oos_expected_leap_steps,
                    force=True,
                )
                built_count += len(shield_oos_years) + (1 if need_shield_bridge else 0)

        if "hiro-ace" in wanted:
            hiro_oos_years = [
                y
                for y in yc.oos_years
                if _needs_build(f"inputs/hiro-ace/amip-oos/forcing_{y}.nc")
            ]
            need_hiro_bridge = (
                yc.oos_bridge_year is not None
                and _needs_build(
                    f"inputs/hiro-ace/amip-oos/forcing_{yc.oos_bridge_year}.nc"
                )
            )
            if hiro_oos_years or need_hiro_bridge:
                build_oos_forcing(
                    "hiro",
                    hist_forcing_dir=os.path.join(
                        resolved_dest, "inputs/hiro-ace/amip-hist"
                    ),
                    aimip_monthly_nc_path=os.path.join(resolved_dest, ZENODO_AIMIP_REL_PATH),
                    co2_nc_path=os.path.join(resolved_dest, CO2_TABLE_REL_PATH),
                    out_dir=os.path.join(resolved_dest, "inputs/hiro-ace/amip-oos"),
                    years=hiro_oos_years,
                    bridge_year=yc.oos_bridge_year if need_hiro_bridge else None,
                    reference_nonleap_year=yc.oos_ref_nonleap_year,
                    reference_leap_year=yc.oos_ref_leap_year,
                    anchor_t0=yc.oos_anchor_t0,
                    anchor_t1=yc.oos_anchor_t1,
                    expected_nonleap_steps=yc.oos_expected_nonleap_steps,
                    expected_leap_steps=yc.oos_expected_leap_steps,
                    force=True,
                )
                built_count += len(hiro_oos_years) + (1 if need_hiro_bridge else 0)

        # Step 3: Verify every output & present raw file per model against `aimip.lock`.
        # Verification is strict by default (`accept_derived=False`); any verified models are persisted
        # to `<dest>/registry.json` BEFORE raising `ValidationError` on failed models.
        verified: list[dict[str, Any]] = []
        derived_overrides: list[dict[str, Any]] = []
        pin_after_computed: dict[str, dict[str, Any]] = {}
        model_failures: dict[str, list[str]] = {m: [] for m in selected}
        if arches_code_err is not None and "archesclimate-ssp" in model_failures:
            model_failures["archesclimate-ssp"].append(arches_code_err)
        if md15_code_err is not None:
            for md15_name in ("md-1.5", "md-1.5-aimip"):
                if md15_name in model_failures:
                    model_failures[md15_name].append(md15_code_err)
        if camulator_code_err is not None and "camulator" in model_failures:
            model_failures["camulator"].append(camulator_code_err)
        if sdyff_code_err is not None and "spherical-dyffusion" in model_failures:
            model_failures["spherical-dyffusion"].append(sdyff_code_err)

        for rel_path, spec in catalog.items():
            affected_models = [
                m
                for m in selected
                if m in spec.models_output
                or (
                    spec.role == "raw"
                    and m in spec.models_needed
                    and os.path.isfile(os.path.join(resolved_dest, rel_path))
                )
            ]
            if not affected_models:
                continue

            if rel_path not in lock_files:
                msg = f"Lock verification failed for {rel_path}: entry missing from aimip.lock."
                for m in affected_models:
                    model_failures[m].append(msg)
                continue

            entry = lock_files[rel_path]
            if not isinstance(entry, Mapping):
                raise ConfigError(f"Lock entry for {rel_path} must be an object.")

            try:
                info = verify_file_against_lock(
                    resolved_dest,
                    rel_path,
                    entry,
                    role=spec.role,
                    strict_unpinned=strict_unpinned,
                )
                verified.append(info)
                if info.get("pin_after_first_fetch"):
                    pin_after_computed[rel_path] = {
                        "size": info["size"],
                        "sha256": info["sha256"],
                    }
                    if progress_stream is not None:
                        print(
                            f"  pin_after_first_fetch: {rel_path} sha256={info['sha256']} (size={info['size']})",
                            file=progress_stream,
                            flush=True,
                        )
            except ValidationError as exc:
                err_str = str(exc)
                abs_candidate = os.path.join(resolved_dest, rel_path)
                can_accept_derived = (
                    accept_derived
                    and spec.source_kind == "derived"
                    and os.path.isfile(abs_candidate)
                    and os.path.getsize(abs_candidate) > 0
                    and ("size mismatch" in err_str or "sha256 mismatch" in err_str)
                )
                if can_accept_derived:
                    actual_size = os.path.getsize(abs_candidate)
                    actual_sha = sha256_file(abs_candidate)
                    expected_sha = str(entry.get("sha256") or "")
                    override_item = {
                        "rel_path": rel_path,
                        "models": affected_models,
                        "builder": spec.builder,
                        "expected_size": entry.get("size"),
                        "actual_size": actual_size,
                        "expected_sha256": expected_sha,
                        "actual_sha256": actual_sha,
                    }
                    derived_overrides.append(override_item)
                    verified.append(
                        {
                            "rel_path": rel_path,
                            "abs_path": abs_candidate,
                            "size": actual_size,
                            "sha256": actual_sha,
                            "accepted_derived_override": True,
                        }
                    )
                    print(
                        f"aimip warning: accepted non-locked derived artifact {rel_path} "
                        f"(expected sha256={expected_sha[:12]}..., got {actual_sha[:12]}...) "
                        "via --accept-derived.",
                        file=sys.stderr,
                    )
                else:
                    for m in affected_models:
                        model_failures[m].append(err_str)

        verified_models = [m for m in selected if not model_failures[m]]
        failed_models = {m: errs for m, errs in model_failures.items() if errs}

        # Always reconcile `<dest>/registry.json` even when `verified_models` is empty
        # so a failed-only rerun drops previously published models that no longer verify (and writes
        # an explicitly empty `"models": []` registry rather than falling back to `registry.default.json`).
        registry_out = _write_data_registry(
            resolved_dest,
            verified_models,
            failed_models=tuple(failed_models.keys()),
            catalog=catalog,
        )

        overrides_out = _write_derived_overrides(
            resolved_dest,
            lock_path=resolved_lock,
            overrides=derived_overrides,
            selected_models=selected,
        )

        if failed_models:
            lines = [
                f"Setup lock verification failed for {len(failed_models)}/{len(selected)} model(s) "
                f"({', '.join(failed_models.keys())}):"
            ]
            for m, errs in failed_models.items():
                lines.append(f"  - {m}: {errs[0]}")
                if len(errs) > 1:
                    lines.append(f"    (+{len(errs) - 1} additional file mismatch(es) for {m})")
            if verified_models:
                lines.append(
                    f"Verified models ({', '.join(verified_models)}) were written to {registry_out} "
                    "and are ready to run."
                )
            else:
                lines.append(
                    f"Failed model(s) were excluded from {registry_out} (0 models newly verified)."
                )
            lines.append(
                "Remediation: re-run with `--force-rebuild`, or if running on a non-x86_64/non-locked "
                "libm/netCDF4 platform (e.g. experimental aarch64/arm64), pass `--accept-derived` "
                "to record non-locked derived netCDF checksums in `<dest>/derived-overrides.json`."
            )
            raise ValidationError("\n".join(lines))

        if progress_stream is not None:
            print(
                f"aimip fetch: [{', '.join(selected)}] ready — verified {len(verified)} file(s) "
                f"(downloaded={downloaded_count} / {_format_bytes(downloaded_bytes)}, "
                f"built={built_count}, skipped={skipped_download_count}).",
                file=progress_stream,
                flush=True,
            )

    out_payload: dict[str, Any] = {
        "mode": "full",
        "dest_dir": resolved_dest,
        "lock_path": resolved_lock,
        "registry_path": registry_out,
        "models": list(selected),
        "verified_models": verified_models,
        "downloaded_files": downloaded_count,
        "downloaded_bytes": downloaded_bytes,
        "skipped_existing_files": skipped_download_count,
        "built_files": built_count,
        "verified_files": len(verified),
        "pin_after_first_fetch": pin_after_computed,
        "derived_overrides": derived_overrides,
        "derived_overrides_path": overrides_out,
    }
    if arches_code_info is not None:
        out_payload["arches_code"] = arches_code_info
    if md15_code_info is not None:
        out_payload["md15_code"] = md15_code_info
    if camulator_code_info is not None:
        out_payload["camulator_code"] = camulator_code_info
    if sdyff_code_info is not None:
        out_payload["sdyff_code"] = sdyff_code_info
    return out_payload


def fetch_model(
    models: str | Sequence[str],
    *,
    dest_dir: str | None = None,
    playground_dir: str | None = None,
    lock_path: str | None = None,
    accept_derived: bool = False,
    years: SetupYearConfig | None = None,
    hf_downloader: Callable[..., str] | None = None,
    zenodo_downloader: Callable[..., str] | None = None,
    arches_code_fetcher: Callable[..., dict[str, Any]] | None = None,
    md15_code_fetcher: Callable[..., dict[str, Any]] | None = None,
    camulator_code_fetcher: Callable[..., dict[str, Any]] | None = None,
    sdyff_code_fetcher: Callable[..., dict[str, Any]] | None = None,
    co2_year_reader: Callable[[int], Any] | None = None,
    progress_stream: Any = False,
) -> dict[str, Any]:
    """Fetch, build (if derived), and lock-verify only *models* into `$AIMIP_DATA` (`aimip fetch -m X`).

    Concurrency-safe via `<dest>/.aimip-fetch.lock`. Streams download counts and byte sizes to
    *progress_stream* (`sys.stderr` by default so CLI `--json` / `--print-sbatch-path` stdout
    stays clean; pass `progress_stream=None` to silence).
    """
    eff_stream = sys.stderr if progress_stream is False else progress_stream
    res = run_setup(
        models=models,
        dest_dir=dest_dir,
        playground_dir=playground_dir,
        lock_path=lock_path,
        accept_derived=accept_derived,
        years=years,
        hf_downloader=hf_downloader,
        zenodo_downloader=zenodo_downloader,
        arches_code_fetcher=arches_code_fetcher,
        md15_code_fetcher=md15_code_fetcher,
        camulator_code_fetcher=camulator_code_fetcher,
        sdyff_code_fetcher=sdyff_code_fetcher,
        co2_year_reader=co2_year_reader,
        progress_stream=eff_stream,
    )
    res["mode"] = "fetch"
    return res
