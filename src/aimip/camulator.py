"""CAMulator (`camulator`, 6-hourly CAM6 / ERA5-scaled atmosphere emulator) adapter and runner.

Upstream provenance & licences:
- Code (Apache-2.0): `WillyChap/miles-credit` commit `d0eb9add439f09bd25883f4393296f3266b5876d`
  (branch `camulator_huggingface`, fork of `NCAR/miles-credit`; Chapman et al., 2025, `arXiv:2504.06007`).
  Fetched from the pinned commit archive (`88,976,237` B, SHA-256 `be2a070c...`; `380` regular files,
  `LC_ALL=C` `find -type f` tree SHA-256 `c560d3bf...`, with the 14 in-tree relative symlinks
  `credit/applications/*.py -> ../../applications/*.py` verified to resolve strictly inside the
  archive tree and omitted on extraction so zero symlinks exist on disk), never from the mutable
  branch name.
- Weights & inputs (Apache-2.0): `hf://willychap/camulator@01afc272553fedd7e21d8046149ec41d2a5d90b8`
  (`checkpoint.pt00069.pt`, `inference_config.yaml`, `metadata/camulator_metadata.yaml`,
  `mean_6h_*.nc`, `std_6h_*.nc`, the 3 `normalization/*statics*.nc` files, both upstream forcing
  files, and the 69 `init_camulator_condition_tensor_<date>T00Z.pth` initial-state tensors).

Scientific & runtime invariants:
- Published `checkpoint.pt00069.pt` takes 4 dynamic forcings (`SOLIN`, `SST`, `ICEFRAC`, `co2vmr_3d`)
  trained on `"CAM6 / ERA5-scaled (1980-2014)"` (not the 2-forcing `1979-2010` model of `arXiv:2504.06007`).
- Upstream forcing sets are named honestly (`camulator-cyclic-1yr` and `camulator-branch-1980-2014`),
  never `"amip-hist"`, because the SST source in `b.e21.CREDIT_climate_*` is undocumented upstream.
- Runs in an isolated Python environment (`venv-camulator`: `torch==2.4.1`, `torch_harmonics==0.7.2`,
  `numpy<2`) with `multiprocessing.set_start_method("spawn", force=True)` set BEFORE importing
  `torch` or `credit`, zero symlinks in `<run_dir>`, `torch.load` wrapped inside the child bootstrap
  so `checkpoint.pt00069.pt` and the initial-state `.pth` are hashed on `O_NOFOLLOW` descriptors and
  loaded from those same open handles with `weights_only=True` (SHA-256 is the primary gate;
  `weights_only=True` is defense-in-depth on `torch==2.4.1`), and all conservation post-blocks
  (`global_energy_fixer_updown`, `global_mass_fixer`, `global_water_fixer`, `tracer_fixer`) + `WindPP`
  kept active.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator, Mapping, Sequence
import contextlib
import copy
import datetime
import glob
import json
import multiprocessing as mp
import os
import posixpath
import re
import runpy
import shutil
import stat
import subprocess
import sys
import tarfile
import time
import uuid
from typing import Any
import zipfile

from .errors import ConfigError, ValidationError
from .safe_pickle import (
    compute_sha256sum_tree_digest,
    normalize_sha256,
    open_verified_hash_stream,
    stage_verified_code_snapshot,
    torch_load_verified,
)
from .validators import makedirs_no_group_other_write, strip_group_other_write, validate_safe_path

CAMULATOR_CODE_COMMIT = "d0eb9add439f09bd25883f4393296f3266b5876d"
CAMULATOR_CODE_LICENCE = "Apache-2.0"
CAMULATOR_WEIGHTS_LICENCE = "Apache-2.0"
CAMULATOR_GITHUB_ARCHIVE_URL = (
    f"https://github.com/WillyChap/miles-credit/archive/{CAMULATOR_CODE_COMMIT}.tar.gz"
)
CAMULATOR_GITHUB_ARCHIVE_SIZE = 88_976_237
CAMULATOR_GITHUB_ARCHIVE_SHA256 = (
    "be2a070ca9233a24ebd2baa5a644d10ef4780726394f6a7f37d2577f47106abb"
)
# `LC_ALL=C` `find -type f` tree digest over the 380 regular files in `WillyChap/miles-credit@d0eb9add`
# (the 14 in-tree relative symlinks `credit/applications/*.py -> ../../applications/*.py` are verified
# to stay strictly within the archive root and omitted on extraction so zero symlinks exist on disk).
CAMULATOR_TREE_FILES = 380
CAMULATOR_TREE_SHA256 = (
    "c560d3bf679f5651b7e47341ca564caac9225ed701ec020a57b1680253ff577b"
)
CAMULATOR_CODE_REL_DIR = "code/miles-credit-camulator"

CAMULATOR_HF_REPO = "willychap/camulator"
CAMULATOR_HF_REVISION = "01afc272553fedd7e21d8046149ec41d2a5d90b8"

CAMULATOR_CKPT_NAME = "checkpoint.pt00069.pt"
CAMULATOR_CKPT_REL_PATH = f"models/camulator/{CAMULATOR_CKPT_NAME}"
CAMULATOR_CKPT_SIZE = 4_805_192_874
CAMULATOR_CKPT_SHA256 = (
    "3c0b5f89784c1e0456cfd930f416e49b96adb0eb55d9063c11c2a3e6eb30a547"
)

CAMULATOR_CONFIG_REL_PATH = "models/camulator/inference_config.yaml"
CAMULATOR_CONFIG_SHA256 = (
    "b8b0e53b26bfa8b013468bfac22bae2c028a18e4a28fd3e10ebdf4154ab26743"
)

CAMULATOR_METADATA_REL_PATH = "models/camulator/metadata/camulator_metadata.yaml"
CAMULATOR_METADATA_SHA256 = (
    "d9b935b47396a612125712e397042de70b5ae92fd3b02508c3d641feb33a0457"
)

CAMULATOR_MEAN_REL_PATH = (
    "models/camulator/normalization/"
    "mean_6h_Coupled_1980_2014_32lev_1.0deg_ERA5scaled_F32_Qtot_Mixed_Modal.nc"
)
CAMULATOR_STD_REL_PATH = (
    "models/camulator/normalization/"
    "std_6h_Coupled_1980_2014_32lev_1.0deg_ERA5scaled_F32_Qtot_Mixed_Modal.nc"
)
CAMULATOR_STATICS_B_REL_PATH = "models/camulator/normalization/statics_b_credit_runs_f32_02.nc"
CAMULATOR_STATICS_BE21_REL_PATH = (
    "models/camulator/normalization/b.e21.CREDIT_climate.statics_1.0deg_32levs_latlon_F32_hyai_fixed.nc"
)
CAMULATOR_STATICS_FE21_REL_PATH = (
    "models/camulator/normalization/f.e21.CREDIT_climate.statics_1.0deg_32levs_latlon_F32_hyai_fixed.nc"
)

CAMULATOR_CYCLIC_FORCING_NAME = "camulator-cyclic-1yr"
CAMULATOR_CYCLIC_FORCING_REL_PATH = (
    "inputs/camulator/forcing_data/b.e21.CREDIT_climate_cyclic_1yr_f32coords.nc"
)
CAMULATOR_CYCLIC_WINDOW: tuple[str, str] = ("1980-01-01T00:00:00", "2100-12-31T18:00:00")

CAMULATOR_BRANCH_FORCING_NAME = "camulator-branch-1980-2014"
CAMULATOR_BRANCH_FORCING_REL_PATH = (
    "inputs/camulator/forcing_data/b.e21.CREDIT_climate_branch_1980_2014.nc"
)
CAMULATOR_BRANCH_WINDOW: tuple[str, str] = ("1980-01-01T00:00:00", "2014-12-31T18:00:00")

# Verified from both upstream forcing headers (`min SST = 273.0 K`).
CAMULATOR_MIN_OCEAN_SST_K = 273.0

# Strictly the 69 published initial-condition tensors (`1980-01-01`, then Jan 1 & Jul 1 for `1981..2014`).
CAMULATOR_PUBLISHED_STARTS: tuple[str, ...] = (
    "1980-01-01T00:00:00",
    *(
        f"{year:04d}-{month:02d}-01T00:00:00"
        for year in range(1981, 2015)
        for month in (1, 7)
    ),
)
CAMULATOR_DEFAULT_FETCH_STARTS: tuple[str, ...] = (
    "1980-01-01T00:00:00",
    "1981-01-01T00:00:00",
)

CAMULATOR_IC_SHAPE: tuple[int, int, int, int, int] = (1, 136, 1, 192, 288)
CAMULATOR_IC_SIZE_BYTES = 30_082_527
CAMULATOR_SECONDS_PER_YEAR = 180.0   # ~480 simulated years / day on 1x A100-80GB
CAMULATOR_T2M_BAND_K: tuple[float, float] = (200.0, 330.0)
EXIT_UNSTABLE = 5
DEFAULT_CAMULATOR_PYTHON = "/opt/aimip/venv-camulator/bin/python"

_NOLEAP_MONTH_DAYS: tuple[int, ...] = (31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31)
_PRED_MONTH_RE = re.compile(r"_([0-9]{4})([0-9]{2})\.nc\Z")

CAMULATOR_UPPER_AIR_VARS: tuple[str, ...] = ("U", "V", "T", "Qtot")
CAMULATOR_SURFACE_DIAG_VARS: tuple[str, ...] = (
    "TREFHT",
    "TS",
    "PS",
    "PRECT",
    "CLDHGH",
    "CLDLOW",
    "CLDMED",
    "TAUX",
    "TAUY",
    "U10",
    "QFLX",
    "SHFLX",
    "LHFLX",
    "FSDS_J",
    "FLDS_J",
    "FSUS",
    "FLUS",
    "FSUTOA",
    "FLUT",
)


def camulator_ic_rel_path(start_iso: str) -> str:
    """Return `$AIMIP_DATA`-relative path for the published `.pth` IC tensor at `start_iso`."""
    ymd = start_iso[:10]
    return f"inputs/camulator/initial_conditions/init_camulator_condition_tensor_{ymd}T00Z.pth"


def camulator_forcing_rel_path(scenario: str) -> str:
    """Return `$AIMIP_DATA`-relative path for a named CAMulator forcing set."""
    if scenario == CAMULATOR_CYCLIC_FORCING_NAME:
        return CAMULATOR_CYCLIC_FORCING_REL_PATH
    if scenario == CAMULATOR_BRANCH_FORCING_NAME:
        return CAMULATOR_BRANCH_FORCING_REL_PATH
    raise ConfigError(
        f"Unknown --forcing-set {scenario!r} for camulator: choose "
        f"{CAMULATOR_CYCLIC_FORCING_NAME!r} or {CAMULATOR_BRANCH_FORCING_NAME!r}.",
        field="forcing_set",
        allowed=[CAMULATOR_CYCLIC_FORCING_NAME, CAMULATOR_BRANCH_FORCING_NAME],
    )


def camulator_python() -> str:
    """Return the absolute executable path of the isolated CAMulator Python environment (`numpy<2`, `torch==2.4.1`)."""
    raw = os.environ.get("AIMIP_CAMULATOR_PYTHON")
    if not raw:
        home = os.environ.get("AIMIP_HOME") or os.environ.get("AIMIP_PLAYGROUND")
        cand = os.path.join(home.rstrip("/"), "venv-camulator", "bin", "python") if home else ""
        if cand and (os.path.exists(cand) or not os.path.exists(DEFAULT_CAMULATOR_PYTHON)):
            raw = cand
        else:
            raw = DEFAULT_CAMULATOR_PYTHON
    path = validate_safe_path(raw, label="CAMulator python")
    if not os.path.isabs(path) or not os.access(path, os.X_OK):
        raise ConfigError(
            f"The isolated CAMulator environment is missing ({path}; CAMulator requires torch 2.4.1, "
            "torch_harmonics 0.7.2, and numpy<2). Re-run `install/install-native.sh --with-camulator` "
            "or set $AIMIP_CAMULATOR_PYTHON."
        )
    return path


def noleap_days_since_1970(year: int, month: int, day: int) -> int:
    """Compute integer days relative to `1970-01-01` in the 365-day `noleap` calendar."""
    return (int(year) - 1970) * 365 + sum(_NOLEAP_MONTH_DAYS[: int(month) - 1]) + (int(day) - 1)


def noleap_expected_6h_counts(start_iso: str, steps: int) -> dict[str, int]:
    """Compute exact number of 6-hourly output steps falling in each `YYYY-MM` over a CAMulator rollout.

    `Quick_Climate.py` stamps step `k` (`0 .. steps - 1`) at `start_dt + 6h * k` (starting with step 0 at `start_dt`).
    """
    y, m, d = int(start_iso[0:4]), int(start_iso[5:7]), int(start_iso[8:10])
    h = int(start_iso[11:13])
    counts: dict[str, int] = {}
    for _ in range(int(steps)):
        ym = f"{y:04d}-{m:02d}"
        counts[ym] = counts.get(ym, 0) + 1
        h += 6
        if h >= 24:
            h -= 24
            d += 1
            if d > _NOLEAP_MONTH_DAYS[m - 1]:
                d = 1
                m += 1
                if m > 12:
                    m = 1
                    y += 1
    return counts


_UNSET_SIZE: Any = object()


def extract_and_verify_camulator_archive(
    archive_path: str,
    code_target_dir: str,
    *,
    expected_archive_sha256: str | None = None,
    expected_archive_size: Any = _UNSET_SIZE,
    expected_tree_sha256: str | None = None,
    expected_tree_files: int | None = None,
) -> dict[str, Any]:
    """Verify the `WillyChap/miles-credit` commit archive on an `O_NOFOLLOW` fd, extract, and verify the 380-file tree."""
    eff_archive_sha = (
        CAMULATOR_GITHUB_ARCHIVE_SHA256
        if expected_archive_sha256 is None
        else expected_archive_sha256
    )
    eff_archive_size = (
        CAMULATOR_GITHUB_ARCHIVE_SIZE
        if expected_archive_size is _UNSET_SIZE
        else expected_archive_size
    )
    eff_tree_sha = (
        CAMULATOR_TREE_SHA256 if expected_tree_sha256 is None else expected_tree_sha256
    )
    eff_tree_files = (
        CAMULATOR_TREE_FILES if expected_tree_files is None else int(expected_tree_files)
    )

    safe_target = validate_safe_path(code_target_dir, label="CAMulator code target directory")
    parent_dir = os.path.dirname(safe_target) or "/"
    makedirs_no_group_other_write(parent_dir)
    stage_root = os.path.join(parent_dir, f".camulator_stage_{uuid.uuid4().hex}")
    makedirs_no_group_other_write(stage_root)
    real_stage = os.path.realpath(stage_root)

    try:
        with open_verified_hash_stream(
            archive_path,
            eff_archive_sha,
            label="CAMulator GitHub archive",
        ) as stream:
            if eff_archive_size is not None:
                st = os.fstat(stream.fileno())
                if int(st.st_size) != int(eff_archive_size):
                    raise ValidationError(
                        f"CAMulator archive size mismatch ({archive_path}): "
                        f"expected {eff_archive_size}, got {st.st_size}."
                    )
            try:
                tf = tarfile.open(fileobj=stream, mode="r:gz")
            except tarfile.TarError as exc:
                raise ValidationError(f"Corrupted CAMulator tar.gz archive ({archive_path}): {exc}") from exc

            with tf:
                members = tf.getmembers()
                if not members:
                    raise ValidationError(f"Empty CAMulator archive: {archive_path}")
                top_prefixes = {m.name.split("/", 1)[0] for m in members if m.name and m.name != "."}
                strip_prefix = ""
                if len(top_prefixes) == 1:
                    only_top = next(iter(top_prefixes))
                    if all(m.name == only_top or m.name.startswith(only_top + "/") for m in members if m.name):
                        strip_prefix = only_top + "/"

                regular_rels: set[str] = set()
                symlink_rels: set[str] = set()
                pending_symlinks: list[tuple[str, str, str]] = []
                extract_plan: list[tuple[tarfile.TarInfo, list[str], str]] = []

                for member in members:
                    raw_name = member.name
                    if (
                        not raw_name
                        or "\x00" in raw_name
                        or "\\" in raw_name
                        or raw_name.startswith("/")
                    ):
                        raise ValidationError(f"Unsafe archive entry {raw_name!r} in {archive_path}")
                    if member.islnk() or member.isdev() or member.isfifo():
                        raise ValidationError(
                            f"Refusing special/link tar member {raw_name!r} in {archive_path}"
                        )
                    rel_name = (
                        raw_name[len(strip_prefix) :]
                        if strip_prefix and raw_name.startswith(strip_prefix)
                        else raw_name
                    )
                    if strip_prefix and raw_name.rstrip("/") == strip_prefix.rstrip("/"):
                        continue
                    rel_stripped = rel_name.rstrip("/")
                    if not rel_stripped:
                        continue
                    parts = rel_stripped.split("/")
                    if any(p in ("", ".", "..") for p in parts):
                        raise ValidationError(f"Path traversal tar member {raw_name!r} in {archive_path}")
                    rel_norm = "/".join(parts)

                    if member.issym():
                        link_target = member.linkname
                        if (
                            not link_target
                            or "\x00" in link_target
                            or "\\" in link_target
                            or link_target.startswith("/")
                        ):
                            raise ValidationError(
                                f"Refusing unsafe symlink {raw_name!r} -> {link_target!r} in {archive_path}"
                            )
                        link_dir = posixpath.dirname(rel_norm)
                        joined = f"{link_dir}/{link_target}" if link_dir else link_target
                        norm_target = posixpath.normpath(joined)
                        if (
                            norm_target in ("", ".")
                            or norm_target == ".."
                            or norm_target.startswith("../")
                        ):
                            raise ValidationError(
                                f"Refusing escaping symlink {raw_name!r} -> {link_target!r} in {archive_path}"
                            )
                        symlink_rels.add(rel_norm)
                        pending_symlinks.append((raw_name, rel_norm, norm_target))
                        continue

                    if member.isdir():
                        extract_plan.append((member, parts, rel_norm))
                    elif member.isfile():
                        if rel_norm in regular_rels:
                            raise ValidationError(
                                f"Refusing duplicate archive member {raw_name!r} in {archive_path}"
                            )
                        regular_rels.add(rel_norm)
                        extract_plan.append((member, parts, rel_norm))
                    else:
                        raise ValidationError(f"Refusing non-regular tar member {raw_name!r} in {archive_path}")

                for _, _, rel_norm in extract_plan:
                    segs = rel_norm.split("/")
                    for depth in range(1, len(segs) + 1):
                        anc = "/".join(segs[:depth])
                        if anc in symlink_rels:
                            raise ValidationError(
                                f"Refusing tar entry {rel_norm!r} colliding with or beneath symlink {anc!r} in {archive_path}"
                            )

                for raw_name, rel_norm, norm_target in pending_symlinks:
                    t_segs = norm_target.split("/")
                    for depth in range(1, len(t_segs)):
                        anc = "/".join(t_segs[:depth])
                        if anc in symlink_rels:
                            raise ValidationError(
                                f"Refusing symlink {raw_name!r} traversing symlinked ancestor {anc!r} in {archive_path}"
                            )
                    if norm_target not in regular_rels:
                        raise ValidationError(
                            f"Refusing symlink {raw_name!r} -> {norm_target!r} with missing regular-file target in {archive_path}"
                        )

                extracted = 0
                for member, parts, _ in extract_plan:
                    dest_path = os.path.realpath(os.path.join(real_stage, *parts))
                    if dest_path != real_stage and not dest_path.startswith(real_stage + os.sep):
                        raise ValidationError(
                            f"Archive member {member.name!r} escapes staging directory in {archive_path}"
                        )
                    if member.isdir():
                        makedirs_no_group_other_write(dest_path)
                        continue
                    makedirs_no_group_other_write(os.path.dirname(dest_path))
                    src_fh = tf.extractfile(member)
                    if src_fh is None:
                        raise ValidationError(f"Readable stream missing for {member.name!r} in {archive_path}")
                    with src_fh, open(dest_path, "wb") as out_fh:
                        shutil.copyfileobj(src_fh, out_fh, length=1024 * 1024)
                    strip_group_other_write(dest_path)
                    extracted += 1
                if extracted == 0:
                    raise ValidationError(f"No regular files extracted from {archive_path}")

        tree_sha, n_files = compute_sha256sum_tree_digest(stage_root, label="CAMulator extracted tree")
        if n_files != int(eff_tree_files) or tree_sha.lower() != eff_tree_sha.lower():
            raise ValidationError(
                f"CAMulator extracted tree verification failed: expected {eff_tree_files} files "
                f"with sha256={eff_tree_sha.lower()}, got {n_files} files with sha256={tree_sha}."
            )

        if os.path.lexists(safe_target):
            if os.path.islink(safe_target) or os.path.isfile(safe_target):
                os.unlink(safe_target)
            else:
                shutil.rmtree(safe_target)
        os.replace(stage_root, safe_target)
    finally:
        if os.path.lexists(stage_root):
            shutil.rmtree(stage_root, ignore_errors=True)

    return {"code_dir": safe_target, "tree_sha256": tree_sha, "files": n_files}


def install_camulator_source_tree(
    venv_dir: str,
    *,
    site_packages_dir: str | None = None,
    archive_url: str | None = None,
    expected_archive_sha256: str | None = None,
    expected_archive_size: Any = _UNSET_SIZE,
    expected_tree_sha256: str | None = None,
    expected_tree_files: int | None = None,
    timeout_seconds: int = 300,
) -> dict[str, Any]:
    """Stage the SHA-256- and tree-verified `WillyChap/miles-credit` source tree into `<venv>/src/miles-credit` and expose it via `.pth`.

    Upstream `WillyChap/miles-credit@d0eb9add` `pyproject.toml` `[tool.setuptools] packages` lists only 11
    subpackages and omits 5 subdirectories present in the repository (`credit.transforms.deprecated`,
    `credit.preblock`, `credit.ensemble`, `credit.trainers.deprecated`, `credit.verification`), causing a
    wheel build to fail on `from credit.transforms.deprecated._transforms import ToTensor`. Installing the
    digest-verified 380-file source tree non-packaged under `<venv>/src/miles-credit` and wiring it onto
    `sys.path` via `<site-packages>/miles_credit_camulator.pth` matches upstream's editable checkout while
    keeping archive SHA-256 and 380-file tree SHA-256 verification mandatory.
    """
    import aimip as _aimip_pkg
    from .setup_data import download_url_artifact

    safe_venv = validate_safe_path(venv_dir, label="venv-camulator directory")
    real_venv = os.path.realpath(safe_venv)
    if site_packages_dir:
        safe_site = validate_safe_path(site_packages_dir, label="venv-camulator site-packages")
    else:
        inferred_site = os.path.dirname(os.path.dirname(os.path.realpath(_aimip_pkg.__file__)))
        safe_site = validate_safe_path(inferred_site, label="venv-camulator site-packages")
    real_site = os.path.realpath(safe_site)
    if not real_site.startswith(real_venv + os.sep):
        raise ValidationError(
            f"venv-camulator site-packages ({real_site}) does not reside inside {real_venv}"
        )
    if not os.path.isdir(safe_site) or os.path.islink(safe_site):
        raise ValidationError(
            f"venv-camulator site-packages directory is missing or symlinked: {safe_site}"
        )

    eff_url = archive_url or CAMULATOR_GITHUB_ARCHIVE_URL
    eff_ar_sha = (
        CAMULATOR_GITHUB_ARCHIVE_SHA256
        if expected_archive_sha256 is None
        else expected_archive_sha256
    )
    eff_ar_size = (
        CAMULATOR_GITHUB_ARCHIVE_SIZE
        if expected_archive_size is _UNSET_SIZE
        else expected_archive_size
    )
    eff_tree_sha = (
        CAMULATOR_TREE_SHA256
        if expected_tree_sha256 is None
        else expected_tree_sha256
    )
    eff_tree_files = (
        CAMULATOR_TREE_FILES
        if expected_tree_files is None
        else int(expected_tree_files)
    )

    src_parent = os.path.join(safe_venv, "src")
    makedirs_no_group_other_write(src_parent)
    code_target = os.path.join(src_parent, "miles-credit")
    tmp_archive = os.path.join(
        src_parent,
        f".miles_credit_{CAMULATOR_CODE_COMMIT[:12]}_{uuid.uuid4().hex}.tar.gz",
    )
    try:
        download_url_artifact(
            url=eff_url,
            dst_path=tmp_archive,
            expected_size=eff_ar_size,
            timeout_seconds=timeout_seconds,
        )
        extracted = extract_and_verify_camulator_archive(
            tmp_archive,
            code_target,
            expected_archive_sha256=eff_ar_sha,
            expected_archive_size=eff_ar_size,
            expected_tree_sha256=eff_tree_sha,
            expected_tree_files=eff_tree_files,
        )
    finally:
        if os.path.exists(tmp_archive):
            try:
                os.unlink(tmp_archive)
            except OSError:
                pass

    real_src = os.path.realpath(extracted["code_dir"])
    pth_path = os.path.join(safe_site, "miles_credit_camulator.pth")
    tmp_pth = f"{pth_path}.tmp-{os.getpid()}"
    with open(tmp_pth, "w", encoding="utf-8") as pth_fh:
        pth_fh.write(real_src + "\n")
    strip_group_other_write(tmp_pth)
    os.replace(tmp_pth, pth_path)
    return {
        "code_dir": real_src,
        "pth_path": pth_path,
        "tree_sha256": extracted["tree_sha256"],
        "files": extracted["files"],
    }


def verify_camulator_code_dir(
    data_root: str,
    *,
    expected_tree_sha256: str | None = None,
    expected_tree_files: int | None = None,
) -> dict[str, Any]:
    """Verify `<data_root>/code/miles-credit-camulator` matches the pinned 380-file Apache-2.0 tree."""
    eff_tree_sha = (
        CAMULATOR_TREE_SHA256 if expected_tree_sha256 is None else expected_tree_sha256
    )
    eff_tree_files = (
        CAMULATOR_TREE_FILES if expected_tree_files is None else int(expected_tree_files)
    )
    code_dir = validate_safe_path(
        os.path.join(data_root, CAMULATOR_CODE_REL_DIR),
        label="CAMulator source directory",
    )
    if not os.path.isdir(code_dir) or os.path.islink(code_dir):
        raise ValidationError(
            f"Lock verification failed for {CAMULATOR_CODE_REL_DIR}: directory is missing ({code_dir})."
        )
    tree_sha, n_files = compute_sha256sum_tree_digest(code_dir, label="CAMulator source directory")
    if n_files != int(eff_tree_files) or tree_sha.lower() != eff_tree_sha.lower():
        raise ValidationError(
            f"Lock verification failed for {CAMULATOR_CODE_REL_DIR}: expected {eff_tree_files} files "
            f"/ sha256={eff_tree_sha.lower()}, got {n_files} files / sha256={tree_sha}."
        )
    return {"code_dir": code_dir, "tree_sha256": tree_sha, "files": n_files}


def resolve_camulator_code_dir(data_root: str | None = None) -> tuple[str, bool]:
    """Resolve `(code_dir, is_override)` for `$AIMIP_DATA/code/miles-credit-camulator`."""
    override = os.environ.get("AIMIP_CAMULATOR_CODE")
    if override:
        safe_ovr = validate_safe_path(override, label="AIMIP_CAMULATOR_CODE")
        if not os.path.isdir(safe_ovr) or os.path.islink(safe_ovr):
            raise ValidationError(f"AIMIP_CAMULATOR_CODE is missing or symlinked: {safe_ovr}")
        return safe_ovr, True

    candidates: list[str] = []
    if data_root:
        candidates.append(os.path.join(data_root, CAMULATOR_CODE_REL_DIR))
    env_data = os.environ.get("AIMIP_DATA")
    if env_data:
        candidates.append(os.path.join(env_data, CAMULATOR_CODE_REL_DIR))
    for home_var in ("AIMIP_HOME", "AIMIP_PLAYGROUND"):
        root = os.environ.get(home_var)
        if root:
            candidates.append(os.path.join(root, "data", CAMULATOR_CODE_REL_DIR))
    for cand in candidates:
        if os.path.isdir(cand) and not os.path.islink(cand):
            return validate_safe_path(cand, label="CAMulator code directory"), False

    target = candidates[0] if candidates else f"$AIMIP_DATA/{CAMULATOR_CODE_REL_DIR}"
    raise ConfigError(
        f"CAMulator source directory not found at {target}; run `aimip fetch -m camulator` first."
    )


def render_camulator_runtime_config(
    base_config_path: str,
    base_config_sha256: str,
    *,
    checkpoint_dir: str,
    mean_nc: str,
    std_nc: str,
    statics_b_nc: str,
    statics_be21_nc: str,
    statics_fe21_nc: str,
    metadata_yaml: str,
    forcing_nc: str,
    ic_pth: str,
    start_iso: str,
    steps: int,
    raw_save_dir: str,
    dst_config_path: str,
) -> dict[str, Any]:
    """Render a self-contained CAMulator `config.yaml` with ZERO symlinks and all conservation post-blocks active."""
    import yaml

    with open_verified_hash_stream(
        base_config_path,
        base_config_sha256,
        label="CAMulator inference_config.yaml",
    ) as fh:
        conf = yaml.safe_load(fh.read().decode("utf-8"))

    if not isinstance(conf, dict):
        raise ValidationError("CAMulator inference_config.yaml did not parse to a mapping.")
    rendered = copy.deepcopy(conf)

    # Point checkpoint root and all normalization/static files directly to verified absolute paths (no ./assets symlinks).
    rendered["save_loc"] = checkpoint_dir.rstrip("/") + "/"
    data_cfg = rendered.setdefault("data", {})
    data_cfg["save_loc_static"] = statics_b_nc
    data_cfg["save_loc_physics"] = statics_b_nc
    data_cfg["mean_path"] = mean_nc
    data_cfg["std_path"] = std_nc
    if data_cfg.get("scaler_type") != "std_new" or data_cfg.get("static_first") is not True:
        raise ValidationError("CAMulator config must keep scaler_type='std_new' and static_first=True.")

    # Enforce active conservation post-blocks (`global_energy_fixer_updown`, mass, water, tracer) + `WindPP`.
    post_conf = rendered.get("model", {}).get("post_conf", {})
    if not post_conf.get("activate"):
        raise ValidationError("CAMulator post_conf must remain active (disabling it is a train/inference mismatch).")
    for req_fixer in (
        "tracer_fixer",
        "global_mass_fixer",
        "global_water_fixer",
        "global_energy_fixer_updown",
    ):
        f_block = post_conf.get(req_fixer)
        if not isinstance(f_block, dict) or not f_block.get("activate"):
            raise ValidationError(
                f"CAMulator conservation fixer {req_fixer!r} must remain active at inference."
            )
        f_block["mean_path"] = mean_nc
        f_block["std_path"] = std_nc
        if "save_loc_physics" in f_block:
            f_block["save_loc_physics"] = statics_be21_nc

    wind_pp = rendered.get("postprocessing", {}).get("wind_artifact_filter", {})
    if not isinstance(wind_pp, dict) or not wind_pp.get("activate"):
        raise ValidationError("CAMulator wind_artifact_filter (WindPP) must remain active.")

    rendered.setdefault("loss", {})["latitude_weights"] = statics_fe21_nc

    y = int(start_iso[0:4])
    m = int(start_iso[5:7])
    d = int(start_iso[8:10])
    h = int(start_iso[11:13])

    pred_cfg = rendered.setdefault("predict", {})
    pred_cfg["mode"] = "none"
    pred_cfg["batch_size"] = 1
    pred_cfg["forecasts"] = {
        "type": "custom",
        "start_year": y,
        "start_month": m,
        "start_day": d,
        "start_hours": [h],
        "duration": 1,
        "days": 365,
    }
    pred_cfg["start_datetime"] = f"{y:04d}-{m:02d}-{d:02d} {h:02d}:00:00"
    pred_cfg["climate_rescale_output"] = True
    pred_cfg["metadata"] = metadata_yaml
    pred_cfg["save_forecast"] = raw_save_dir.rstrip("/") + "/"
    pred_cfg["forcing_file"] = forcing_nc
    pred_cfg["run_fast_climate"] = False
    pred_cfg["init_cond_fast_climate"] = ic_pth
    pred_cfg["timesteps_fast_climate"] = int(steps)

    tmp_cfg = f"{dst_config_path}.tmp-{os.getpid()}"
    with open(tmp_cfg, "w", encoding="utf-8") as out_fh:
        yaml.safe_dump(rendered, out_fh, sort_keys=False)
    strip_group_other_write(tmp_cfg)
    os.replace(tmp_cfg, dst_config_path)
    return rendered


def assert_no_symlinks_under(root_dir: str, *, label: str = "CAMulator run directory") -> None:
    """Fail closed if any symbolic link exists at or beneath *root_dir*."""
    if os.path.islink(root_dir):
        raise ValidationError(f"Refusing symlinked {label}: {root_dir}")
    for walk_root, dirs, files in os.walk(root_dir, followlinks=False):
        if os.path.islink(walk_root):
            raise ValidationError(f"Refusing symlink inside {label}: {walk_root}")
        for entry in (*dirs, *files):
            cand = os.path.join(walk_root, entry)
            if os.path.islink(cand):
                raise ValidationError(f"Refusing symlink inside {label}: {cand}")


def consolidate_camulator_monthly_outputs(
    raw_save_dir: str,
    *,
    metadata_yaml_path: str,
    metadata_sha256: str,
    start_iso: str,
    steps: int,
    scenario: str,
    out_dir: str,
    complete_months_only: bool = True,
) -> dict[str, Any]:
    """Consolidate `Quick_Climate.py --monthly_mean` `pred_*_<YYYYMM>.nc` files into CF-1.8 AIMIP NetCDFs."""
    import netCDF4
    import numpy as np
    import yaml

    with open_verified_hash_stream(
        metadata_yaml_path,
        metadata_sha256,
        label="CAMulator metadata YAML",
    ) as mfh:
        meta_doc = yaml.safe_load(mfh.read().decode("utf-8")) or {}

    pattern = os.path.join(raw_save_dir, "**", "pred_*.nc")
    pred_files = sorted(glob.glob(pattern, recursive=True))
    if not pred_files:
        raise ValidationError(
            f"CAMulator produced zero monthly NetCDF files under {raw_save_dir} "
            "(ensure multiprocessing start method is 'spawn')."
        )

    step_counts_by_ym = noleap_expected_6h_counts(start_iso, steps)
    parsed_files: list[tuple[str, int, int, int, str]] = []
    for fpath in pred_files:
        m_obj = _PRED_MONTH_RE.search(os.path.basename(fpath))
        if not m_obj:
            continue
        yy, mm = int(m_obj.group(1)), int(m_obj.group(2))
        ym = f"{yy:04d}-{mm:02d}"
        full_month_steps = 4 * _NOLEAP_MONTH_DAYS[mm - 1]
        cov = step_counts_by_ym.get(ym, full_month_steps)
        if complete_months_only and cov < full_month_steps:
            continue
        parsed_files.append((ym, yy, mm, cov, fpath))

    parsed_files.sort(key=lambda item: item[0])
    if not parsed_files:
        # Short smoke runs (< 1 full calendar month): retain partial month(s) with exact `cov` step count.
        for fpath in pred_files:
            m_obj = _PRED_MONTH_RE.search(os.path.basename(fpath))
            if not m_obj:
                continue
            yy, mm = int(m_obj.group(1)), int(m_obj.group(2))
            ym = f"{yy:04d}-{mm:02d}"
            cov = step_counts_by_ym.get(ym, 1)
            parsed_files.append((ym, yy, mm, cov, fpath))
        parsed_files.sort(key=lambda item: item[0])

    if not parsed_files:
        raise ValidationError(f"No `pred_*_<YYYYMM>.nc` monthly files matched under {raw_save_dir}.")

    n_months = len(parsed_files)
    with netCDF4.Dataset(parsed_files[0][4], "r") as ds0:
        lat_name = "latitude" if "latitude" in ds0.variables else "lat"
        lon_name = "longitude" if "longitude" in ds0.variables else "lon"
        lat = np.asarray(ds0.variables[lat_name][:], dtype=np.float32)
        lon = np.asarray(ds0.variables[lon_name][:], dtype=np.float32)
        levels = (
            np.asarray(ds0.variables["level"][:], dtype=np.float32)
            if "level" in ds0.variables
            else None
        )
        present_sfc = [v for v in CAMULATOR_SURFACE_DIAG_VARS if v in ds0.variables]
        present_ua = [v for v in CAMULATOR_UPPER_AIR_VARS if v in ds0.variables]

    if "TREFHT" not in present_sfc:
        raise ValidationError(f"CAMulator monthly file {parsed_files[0][4]} is missing `TREFHT`.")

    cos_w = np.cos(np.deg2rad(lat.astype(np.float64)))[:, np.newaxis]
    sfc_data: dict[str, Any] = {
        v: np.empty((1, n_months, lat.size, lon.size), dtype=np.float32) for v in present_sfc
    }
    ua_data: dict[str, Any] = (
        {
            v: np.empty((n_months, levels.size, lat.size, lon.size), dtype=np.float32)
            for v in present_ua
        }
        if levels is not None and present_ua
        else {}
    )
    valid_days = np.empty((1, n_months), dtype=np.int64)
    counts_arr = np.empty((1, n_months), dtype=np.int32)

    for idx, (ym, yy, mm, cov, fpath) in enumerate(parsed_files):
        valid_days[0, idx] = noleap_days_since_1970(yy, mm, 15)
        counts_arr[0, idx] = int(cov)
        with netCDF4.Dataset(fpath, "r") as ds_m:
            for v in present_sfc:
                slab = np.ma.filled(ds_m.variables[v][0], np.nan).astype(np.float32)
                if not np.isfinite(slab).all():
                    raise SystemExit(EXIT_UNSTABLE)
                sfc_data[v][0, idx] = slab
            gm_trefht = float(
                np.sum(sfc_data["TREFHT"][0, idx].astype(np.float64) * cos_w)
                / (np.sum(cos_w) * lon.size)
            )
            if not (CAMULATOR_T2M_BAND_K[0] <= gm_trefht <= CAMULATOR_T2M_BAND_K[1]):
                raise SystemExit(EXIT_UNSTABLE)
            for v in ua_data:
                ua_slab = np.ma.filled(ds_m.variables[v][0], np.nan).astype(np.float32)
                if not np.isfinite(ua_slab).all():
                    raise SystemExit(EXIT_UNSTABLE)
                ua_data[v][idx] = ua_slab

    monthly_nc = os.path.join(out_dir, "monthly_mean_predictions.nc")
    tmp_monthly = f"{monthly_nc}.tmp-{os.getpid()}"
    with netCDF4.Dataset(tmp_monthly, "w", format="NETCDF4") as out_ds:
        out_ds.Conventions = "CF-1.8"
        out_ds.model = "camulator"
        out_ds.checkpoint = CAMULATOR_CKPT_NAME
        out_ds.upstream_code_commit = CAMULATOR_CODE_COMMIT
        out_ds.upstream_hf_revision = CAMULATOR_HF_REVISION
        out_ds.forcing_set = scenario
        out_ds.target_attribution = (
            "Emulates CAM6 / ERA5-scaled climate target (1980-2014); compare with that target, "
            "not directly with observations. Published 4-forcing checkpoint differs from the 2-forcing "
            "1979-2010 model in arXiv:2504.06007."
        )

        out_ds.createDimension("sample", 1)
        out_ds.createDimension("time", n_months)
        out_ds.createDimension("lat", lat.size)
        out_ds.createDimension("lon", lon.size)

        v_lat = out_ds.createVariable("lat", "f4", ("lat",))
        v_lat.units = "degrees_north"
        v_lat.standard_name = "latitude"
        v_lat[:] = lat

        v_lon = out_ds.createVariable("lon", "f4", ("lon",))
        v_lon.units = "degrees_east"
        v_lon.standard_name = "longitude"
        v_lon[:] = lon

        v_time = out_ds.createVariable("time", "i8", ("time",))
        v_time.units = "months"
        v_time[:] = np.arange(n_months, dtype=np.int64)

        v_valid = out_ds.createVariable("valid_time", "i8", ("sample", "time"))
        v_valid.units = "days since 1970-01-01 00:00:00"
        v_valid.calendar = "noleap"
        v_valid[:] = valid_days

        v_counts = out_ds.createVariable("counts", "i4", ("sample", "time"))
        v_counts.long_name = "Number of 6-hourly noleap steps averaged into each calendar month"
        v_counts[:] = counts_arr

        for v_name in present_sfc:
            v_meta = meta_doc.get(v_name) if isinstance(meta_doc.get(v_name), dict) else {}
            nc_var = out_ds.createVariable(v_name, "f4", ("sample", "time", "lat", "lon"))
            if v_meta.get("units"):
                nc_var.units = str(v_meta["units"])
            if v_meta.get("long_name"):
                nc_var.long_name = str(v_meta["long_name"])
            nc_var[:] = sfc_data[v_name]

    strip_group_other_write(tmp_monthly)
    os.replace(tmp_monthly, monthly_nc)

    levels_nc: str | None = None
    if ua_data and levels is not None:
        levels_nc = os.path.join(out_dir, "monthly_levels.nc")
        tmp_levels = f"{levels_nc}.tmp-{os.getpid()}"
        with netCDF4.Dataset(tmp_levels, "w", format="NETCDF4") as lds:
            lds.Conventions = "CF-1.8"
            lds.model = "camulator"
            lds.checkpoint = CAMULATOR_CKPT_NAME
            lds.forcing_set = scenario
            lds.createDimension("time", n_months)
            lds.createDimension("level", levels.size)
            lds.createDimension("lat", lat.size)
            lds.createDimension("lon", lon.size)

            lt = lds.createVariable("time", "i8", ("time",))
            lt.units = "days since 1970-01-01 00:00:00"
            lt.calendar = "noleap"
            lt[:] = [noleap_days_since_1970(yy, mm, 1) for _, yy, mm, _, _ in parsed_files]

            ll = lds.createVariable("level", "f4", ("level",))
            ll.units = "hPa"
            ll.long_name = "hybrid_sigma_pressure_reference_level"
            ll[:] = levels

            llat = lds.createVariable("lat", "f4", ("lat",))
            llat.units = "degrees_north"
            llat[:] = lat

            llon = lds.createVariable("lon", "f4", ("lon",))
            llon.units = "degrees_east"
            llon[:] = lon

            for v_name, arr in ua_data.items():
                v_meta = meta_doc.get(v_name) if isinstance(meta_doc.get(v_name), dict) else {}
                lvar = lds.createVariable(
                    v_name,
                    "f4",
                    ("time", "level", "lat", "lon"),
                    zlib=True,
                    complevel=1,
                    shuffle=True,
                    chunksizes=(1, levels.size, lat.size, lon.size),
                )
                if v_meta.get("units"):
                    lvar.units = str(v_meta["units"])
                if v_meta.get("long_name"):
                    lvar.long_name = str(v_meta["long_name"])
                lvar[:] = arr

        strip_group_other_write(tmp_levels)
        os.replace(tmp_levels, levels_nc)

    return {
        "monthly_nc": monthly_nc,
        "levels_nc": levels_nc,
        "months_written": {"atmosphere": n_months},
        "first_month": parsed_files[0][0],
        "last_month": parsed_files[-1][0],
    }


def _extract_camulator_load_path(target: Any) -> str | None:
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


def _append_camulator_audit_record(audit_log_path: str, payload: Mapping[str, Any]) -> None:
    safe_log = validate_safe_path(audit_log_path, label="CAMulator verified-load audit log")
    if os.path.islink(safe_log):
        raise ValidationError(f"Refusing symlinked CAMulator verified-load audit log: {safe_log}")
    flags = os.O_WRONLY | os.O_APPEND | getattr(os, "O_NOFOLLOW", 0)
    try:
        fd = os.open(safe_log, flags)
    except OSError as exc:
        raise ValidationError(
            f"Cannot open CAMulator verified-load audit log {safe_log}: {exc}"
        ) from exc
    try:
        st = os.fstat(fd)
        if not stat.S_ISREG(st.st_mode):
            raise ValidationError(f"Refusing non-regular CAMulator verified-load audit log: {safe_log}")
        os.write(fd, (json.dumps(dict(payload)) + "\n").encode("utf-8"))
    finally:
        os.close(fd)


def _read_camulator_audit_realpaths(audit_log_path: str) -> set[str]:
    safe_log = validate_safe_path(audit_log_path, label="CAMulator verified-load audit log")
    if os.path.islink(safe_log):
        raise ValidationError(f"Refusing symlinked CAMulator verified-load audit log: {safe_log}")
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    try:
        fd = os.open(safe_log, flags)
    except OSError as exc:
        raise ValidationError(
            f"Cannot read CAMulator verified-load audit log {safe_log}: {exc}"
        ) from exc
    fh = None
    try:
        st = os.fstat(fd)
        if not stat.S_ISREG(st.st_mode) or st.st_uid != os.geteuid():
            raise ValidationError(
                f"Refusing untrusted CAMulator verified-load audit log: {safe_log}"
            )
        fh = os.fdopen(fd, "rb")
        with fh:
            raw_lines = fh.read().decode("utf-8").splitlines()
    except BaseException:
        if fh is None:
            try:
                os.close(fd)
            except OSError:
                pass
        raise

    realpaths: set[str] = set()
    for line in raw_lines:
        if not line.strip():
            continue
        rec = json.loads(line)
        if isinstance(rec, Mapping) and isinstance(rec.get("realpath"), str):
            realpaths.add(rec["realpath"])
    return realpaths


def install_camulator_verified_torch_load(
    expected_sha256_by_path: Mapping[str, str],
    *,
    audit_log_path: str | None = None,
    recorded_realpaths: set[str] | None = None,
) -> Callable[[], None]:
    """Wrap `torch.load` so every path load hashes an `O_NOFOLLOW` descriptor and loads that same stream with `weights_only=True`."""
    import torch

    if not expected_sha256_by_path:
        raise ValidationError("CAMulator verified torch.load requires a non-empty SHA-256 digest map.")
    digest_map: dict[str, str] = {}
    for raw_p, raw_sha in expected_sha256_by_path.items():
        safe_p = validate_safe_path(str(raw_p), label="CAMulator verified torch artifact")
        if os.path.islink(safe_p):
            raise ValidationError(f"Refusing symlinked CAMulator torch artifact: {safe_p}")
        norm_sha = normalize_sha256(
            str(raw_sha),
            label=f"CAMulator {os.path.basename(safe_p)} SHA-256",
        )
        digest_map[os.path.abspath(safe_p)] = norm_sha
        digest_map[os.path.realpath(safe_p)] = norm_sha

    orig_load = getattr(torch, "_aimip_orig_torch_load", torch.load)
    orig_ser_load = getattr(torch.serialization, "_aimip_orig_torch_load", torch.serialization.load)
    setattr(torch, "_aimip_orig_torch_load", orig_load)
    setattr(torch.serialization, "_aimip_orig_torch_load", orig_ser_load)

    def _verified_camulator_torch_load(f: Any, *args: Any, **kwargs: Any) -> Any:
        if kwargs.get("weights_only") is False:
            raise ValidationError(
                "Refusing torch.load(..., weights_only=False) inside CAMulator runtime."
            )
        kwargs["weights_only"] = True
        cand_path = _extract_camulator_load_path(f)
        if cand_path is not None:
            abs_p = os.path.abspath(cand_path)
            want_sha = digest_map.get(abs_p)
            if want_sha is None:
                raise ValidationError(
                    f"Refusing unverified CAMulator torch.load path: {abs_p}"
                )
            with open_verified_hash_stream(
                abs_p,
                want_sha,
                label=f"CAMulator {os.path.basename(abs_p)}",
            ) as stream:
                loaded_obj = orig_load(stream, *args, **kwargs)
            real_p = os.path.realpath(abs_p)
            if recorded_realpaths is not None:
                recorded_realpaths.add(real_p)
            if audit_log_path:
                _append_camulator_audit_record(
                    audit_log_path,
                    {"abspath": abs_p, "realpath": real_p},
                )
            return loaded_obj
        return orig_load(f, *args, **kwargs)

    torch.load = _verified_camulator_torch_load  # type: ignore[assignment]
    torch.serialization.load = _verified_camulator_torch_load  # type: ignore[assignment]

    def _restore() -> None:
        torch.load = orig_load  # type: ignore[assignment]
        torch.serialization.load = orig_ser_load  # type: ignore[assignment]

    return _restore


@contextlib.contextmanager
def verified_camulator_torch_load(
    expected_sha256_by_path: Mapping[str, str],
) -> Iterator[set[str]]:
    """Context-managed `install_camulator_verified_torch_load` for tests and in-process checks."""
    loaded_set: set[str] = set()
    restore = install_camulator_verified_torch_load(
        expected_sha256_by_path,
        recorded_realpaths=loaded_set,
    )
    try:
        yield loaded_set
    finally:
        restore()


_SPAWN_BOOTSTRAP = (
    "import json, multiprocessing as mp, os, runpy, sys\n"
    "sys.dont_write_bytecode = True\n"
    "os.environ['PYTHONDONTWRITEBYTECODE'] = '1'\n"
    "mp.set_start_method('spawn', force=True)\n"
    "_pkg_root = os.environ.get('AIMIP_BOOTSTRAP_PKG_ROOT', '')\n"
    "if _pkg_root and _pkg_root not in sys.path:\n"
    "    sys.path.insert(0, _pkg_root)\n"
    "from aimip.camulator import install_camulator_verified_torch_load\n"
    "install_camulator_verified_torch_load(\n"
    "    json.loads(os.environ['AIMIP_CAMULATOR_TORCH_DIGESTS']),\n"
    "    audit_log_path=os.environ.get('AIMIP_CAMULATOR_VERIFIED_LOG'),\n"
    ")\n"
    "climate_dir, quick_climate_py = sys.argv[1], sys.argv[2]\n"
    "code_parent = os.path.dirname(climate_dir)\n"
    "for p in (climate_dir, code_parent):\n"
    "    if p not in sys.path:\n"
    "        sys.path.insert(0, p)\n"
    "os.chdir(climate_dir)\n"
    "if os.path.isfile(os.path.join(climate_dir, 'Model_State.py')):\n"
    "    import Model_State\n"
    "    for _flag in ('CREDIT_AVAILABLE', 'POSTBLOCK_AVAILABLE', 'WINDPP_AVAILABLE'):\n"
    "        if not getattr(Model_State, _flag, True):\n"
    "            raise SystemExit(f'CAMulator Model_State.{_flag} is False; refusing run with degraded physics/imports')\n"
    "sys.argv = ['Quick_Climate.py', *sys.argv[3:]]\n"
    "runpy.run_path(quick_climate_py, run_name='__main__')\n"
)


def run_camulator(out_dir: str, job: Mapping[str, Any]) -> dict[str, Any]:
    """Verify CAMulator assets, enforce `O_NOFOLLOW` fd-verified `torch.load(..., weights_only=True)` in child, and consolidate."""
    import torch
    import yaml

    requested_device = str(job.get("device", "cuda"))
    if requested_device.startswith("cuda"):
        if not torch.cuda.is_available():
            raise SystemExit(
                f"{requested_device} requested for camulator, but CUDA is unavailable in this job; refusing silent CPU fallback."
            )
        qc_device = "cuda"
    elif requested_device == "cpu":
        qc_device = "cpu"
    else:
        raise SystemExit(f"Unsupported device {requested_device!r} for camulator.")

    verify_tree = bool(job.get("verify_code_tree", True))
    code_dir_raw = job.get("code_dir")
    if code_dir_raw:
        code_dir = validate_safe_path(str(code_dir_raw), label="CAMulator code_dir")
        is_ovr = bool(job.get("code_is_override", False))
    else:
        code_dir, is_ovr = resolve_camulator_code_dir(job.get("data_root"))

    exp_tree_sha = str(job.get("code_tree_sha256") or CAMULATOR_TREE_SHA256)
    exp_tree_files = int(job.get("code_tree_files") or CAMULATOR_TREE_FILES)
    if verify_tree:
        if is_ovr and (
            job.get("code_tree_sha256") is None or job.get("code_tree_files") is None
        ):
            raise ValidationError(
                f"Refusing unverified AIMIP_CAMULATOR_CODE override ({code_dir}): "
                "code_tree_sha256 and code_tree_files must be pinned in the job manifest."
            )
        tree_sha, n_files = compute_sha256sum_tree_digest(code_dir, label="CAMulator code directory")
        if n_files != exp_tree_files or tree_sha.lower() != exp_tree_sha.lower():
            raise ValidationError(
                f"Refusing tampered CAMulator code tree ({code_dir}): expected "
                f"{exp_tree_files} files / {exp_tree_sha}, got {n_files} / {tree_sha}."
            )

    # Pre-flight `O_NOFOLLOW` SHA-256 check in parent; the spawned child also wraps `torch.load`
    # via `install_camulator_verified_torch_load` so upstream loads directly from the verified fd.
    ckpt_path = validate_safe_path(str(job["checkpoint"]), label="CAMulator checkpoint")
    ckpt_sha = normalize_sha256(str(job["checkpoint_sha256"]), label="CAMulator checkpoint SHA-256")
    with open_verified_hash_stream(ckpt_path, ckpt_sha, label="CAMulator checkpoint.pt00069.pt"):
        pass

    ic_pth = validate_safe_path(str(job["ic_file"]), label="CAMulator initial-condition tensor")
    ic_sha = normalize_sha256(str(job["ic_sha256"]), label="CAMulator IC SHA-256")
    ic_tensor = torch_load_verified(
        ic_pth,
        ic_sha,
        map_location="cpu",
        weights_only=True,
        label="CAMulator initial-condition .pth",
    )
    if not isinstance(ic_tensor, torch.Tensor) or tuple(ic_tensor.shape) != CAMULATOR_IC_SHAPE:
        raise ValidationError(
            f"CAMulator IC tensor {ic_pth} has unexpected type/shape "
            f"({type(ic_tensor).__name__} { getattr(ic_tensor, 'shape', None) }, expected float32 {CAMULATOR_IC_SHAPE})."
        )
    del ic_tensor

    for asset_key, sha_key, lbl in (
        ("mean_file", "mean_sha256", "CAMulator mean NetCDF"),
        ("std_file", "std_sha256", "CAMulator std NetCDF"),
        ("statics_b_file", "statics_b_sha256", "CAMulator statics_b NetCDF"),
        ("statics_be21_file", "statics_be21_sha256", "CAMulator b.e21 statics NetCDF"),
        ("statics_fe21_file", "statics_fe21_sha256", "CAMulator f.e21 statics NetCDF"),
        ("forcing_file", "forcing_sha256", "CAMulator forcing NetCDF"),
    ):
        with open_verified_hash_stream(str(job[asset_key]), str(job[sha_key]), label=lbl):
            pass

    meta_src_path = validate_safe_path(str(job["metadata_file"]), label="CAMulator metadata YAML")
    meta_sha = normalize_sha256(str(job["metadata_sha256"]), label="CAMulator metadata YAML SHA-256")
    with open_verified_hash_stream(meta_src_path, meta_sha, label="CAMulator metadata YAML") as meta_fh:
        meta_bytes = meta_fh.read()
    meta_parsed = yaml.safe_load(meta_bytes.decode("utf-8"))
    if not isinstance(meta_parsed, dict):
        raise ValidationError(f"CAMulator metadata YAML {meta_src_path} did not parse to a mapping.")

    makedirs_no_group_other_write(out_dir)
    verified_meta_path = os.path.join(out_dir, "camulator_metadata_verified.yaml")
    tmp_meta_path = f"{verified_meta_path}.tmp-{os.getpid()}"
    with open(tmp_meta_path, "wb") as m_out:
        m_out.write(meta_bytes)
    strip_group_other_write(tmp_meta_path)
    os.replace(tmp_meta_path, verified_meta_path)

    raw_save_dir = os.path.join(out_dir, "_raw_pred")
    if os.path.lexists(raw_save_dir):
        shutil.rmtree(raw_save_dir)
    makedirs_no_group_other_write(raw_save_dir)

    runtime_cfg_path = os.path.join(out_dir, "camulator_runtime_config.yaml")
    render_camulator_runtime_config(
        str(job["base_config_file"]),
        str(job["base_config_sha256"]),
        checkpoint_dir=os.path.dirname(ckpt_path),
        mean_nc=str(job["mean_file"]),
        std_nc=str(job["std_file"]),
        statics_b_nc=str(job["statics_b_file"]),
        statics_be21_nc=str(job["statics_be21_file"]),
        statics_fe21_nc=str(job["statics_fe21_file"]),
        metadata_yaml=verified_meta_path,
        forcing_nc=str(job["forcing_file"]),
        ic_pth=ic_pth,
        start_iso=str(job["start"]),
        steps=int(job["steps"]),
        raw_save_dir=raw_save_dir,
        dst_config_path=runtime_cfg_path,
    )
    assert_no_symlinks_under(out_dir)

    snap_code_dir: str | None = None
    exec_code_dir = code_dir
    if verify_tree:
        snap_code_dir = os.path.join(out_dir, ".aimip-camulator-code")
        stage_verified_code_snapshot(
            code_dir,
            snap_code_dir,
            expected_tree_sha256=exp_tree_sha,
            expected_tree_files=exp_tree_files,
            label="CAMulator code directory",
        )
        exec_code_dir = snap_code_dir

    climate_dir = os.path.join(exec_code_dir, "climate")
    quick_climate_py = os.path.join(climate_dir, "Quick_Climate.py")
    if not os.path.isfile(quick_climate_py) or os.path.islink(quick_climate_py):
        if snap_code_dir and os.path.lexists(snap_code_dir):
            shutil.rmtree(snap_code_dir, ignore_errors=True)
        raise ValidationError(f"CAMulator Quick_Climate.py not found at {quick_climate_py}")

    cli_args = [
        "--config",
        runtime_cfg_path,
        "--model_name",
        os.path.basename(ckpt_path),
        "--device",
        qc_device,
        "--save_append",
        "aimip",
        "--monthly_mean",
    ]
    audit_path = os.path.join(out_dir, ".camulator_verified_loads.jsonl")
    if os.path.lexists(audit_path):
        if os.path.isdir(audit_path) and not os.path.islink(audit_path):
            shutil.rmtree(audit_path)
        else:
            os.unlink(audit_path)
    os.close(
        os.open(
            audit_path,
            os.O_CREAT | os.O_EXCL | os.O_WRONLY | getattr(os, "O_NOFOLLOW", 0),
            0o600,
        )
    )
    assert_no_symlinks_under(out_dir)

    child_env = dict(os.environ)
    child_env["PYTHONDONTWRITEBYTECODE"] = "1"
    child_env["AIMIP_BOOTSTRAP_PKG_ROOT"] = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    child_env["AIMIP_CAMULATOR_VERIFIED_LOG"] = audit_path
    child_env["AIMIP_CAMULATOR_TORCH_DIGESTS"] = json.dumps(
        {
            os.path.abspath(ckpt_path): ckpt_sha,
            os.path.abspath(ic_pth): ic_sha,
        }
    )
    t0 = time.time()
    try:
        child_res = subprocess.run(
            [sys.executable, "-B", "-c", _SPAWN_BOOTSTRAP, climate_dir, quick_climate_py, *cli_args],
            cwd=climate_dir,
            env=child_env,
            check=False,
        )
        if child_res.returncode != 0:
            raise SystemExit(int(child_res.returncode))
        infer_s = round(time.time() - t0, 3)

        recorded_realpaths = _read_camulator_audit_realpaths(audit_path)
        missing_verified = [
            req_p for req_p in (ckpt_path, ic_pth)
            if os.path.realpath(req_p) not in recorded_realpaths
        ]
        if missing_verified:
            raise ValidationError(
                "CAMulator child process completed without loading required artifact(s) "
                f"through verified torch.load wrapper: {', '.join(missing_verified)}"
            )
    finally:
        if os.path.lexists(audit_path):
            try:
                os.unlink(audit_path)
            except OSError:
                pass
        if snap_code_dir and os.path.lexists(snap_code_dir):
            shutil.rmtree(snap_code_dir, ignore_errors=True)

    consolidated = consolidate_camulator_monthly_outputs(
        raw_save_dir,
        metadata_yaml_path=verified_meta_path,
        metadata_sha256=meta_sha,
        start_iso=str(job["start"]),
        steps=int(job["steps"]),
        scenario=str(job["scenario"]),
        out_dir=out_dir,
    )
    shutil.rmtree(raw_save_dir, ignore_errors=True)
    assert_no_symlinks_under(out_dir)

    return {
        "months_written": consolidated["months_written"],
        "first_month": consolidated["first_month"],
        "last_month": consolidated["last_month"],
        "scenario": str(job["scenario"]),
        "checkpoint": os.path.basename(ckpt_path),
        "inference_seconds": infer_s,
    }
