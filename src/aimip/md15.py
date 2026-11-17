"""Monthly Diffusion v0.9 at 1.5 deg (`md-1.5`) adapter, verified loader, and runner.

Upstream provenance & licences:
- Code & weights (MIT, Copyright (c) 2026 Kyle Hall):
  `kjhall01/monthly-diffusion@b077d3bcbf602a6b363d5a9da28cf1dc3583c4fb`
  (tag `MDv0.9-1.5-licensed`, Zenodo record `21430292`, `arXiv:2604.13481`).
  Uses `models/MD-1p5-latest.pth/{config.pkl,weights.pt}` and
  `scripts/variable_statistics_197901-201412.json` (legacy `models/MD-1p5.pth` is never used).
- AIMIP Phase 1 forcing (CC-BY-4.0, Zenodo record `17065758`):
  `ERA5-0.25deg-monthly-mean-forcing-1978-2024.nc` (`556` month-start-centred points,
  `1978-10-01 .. 2025-01-01`). Every point of the `1.5 deg` target grid (`121 x 240`,
  `lat = -90..90`, `lon = 0..358.5`) coincides with an exact `0.25 deg` source point
  (`1.5 / 0.25 = 6`), so periodic bilinear regridding equals exact coincident subsampling
  (`max|diff| = 0.0` with identical land `NaN` masks). Non-coincident grids fail closed.
"""

from __future__ import annotations

import calendar as py_calendar
from collections.abc import Mapping, Sequence
import datetime
import importlib.util
import json
import os
import shutil
import sys
import tarfile
import time
import types
import uuid
from typing import Any

from .errors import ConfigError, ValidationError
from .safe_pickle import (
    compute_sha256sum_tree_digest,
    open_verified_hash_stream,
    pickle_load_verified,
    torch_load_verified,
)
from .validators import makedirs_no_group_other_write, strip_group_other_write, validate_safe_path

MD15_COMMIT = "b077d3bcbf602a6b363d5a9da28cf1dc3583c4fb"
MD15_TAG = "MDv0.9-1.5-licensed"
MD15_CODE_LICENCE = "MIT"
MD15_WEIGHTS_LICENCE = "MIT"
MD15_FORCING_LICENCE = "CC-BY-4.0"

MD15_GITHUB_ARCHIVE_URL = (
    f"https://github.com/kjhall01/monthly-diffusion/archive/{MD15_COMMIT}.tar.gz"
)
MD15_GITHUB_ARCHIVE_SIZE = 63_744_574
MD15_GITHUB_ARCHIVE_SHA256 = (
    "a11ce4d8c5434653d19426a0a45346b2cf0aa43338167514ed55c8fec09fdded"
)
MD15_TREE_FILES = 37
MD15_TREE_SHA256 = (
    "3fb285946e40a72b2dfd1e806601c95cda6cf285392834726c5bf03ec0a96f28"
)
MD15_CODE_REL_DIR = "code/monthly-diffusion"

MD15_ZENODO_RECORD = "21430292"
MD15_ZENODO_ZIP_SIZE = 63_775_851
MD15_ZENODO_ZIP_MD5 = "3c568f95e6151a0704e84ae7a8d2cd5c"
MD15_ZENODO_ZIP_SHA256 = (
    "73f2d88e43ceb6f3e9a2a164ce3649ebaef0c9abbe7a359cf4ad2a805476ed58"
)

MD15_CONFIG_REL_PATH = "models/md-1.5/MD-1p5-latest.pth/config.pkl"
MD15_CONFIG_SIZE = 582_803
MD15_CONFIG_SHA256 = (
    "52c60da6176378889cecf5d6656ebcfbbd8d818215d8d43af47b9d39e89d5a9f"
)

MD15_WEIGHTS_REL_PATH = "models/md-1.5/MD-1p5-latest.pth/weights.pt"
MD15_WEIGHTS_SIZE = 26_319_522
MD15_WEIGHTS_SHA256 = (
    "dac25623a523c9d05e1a788a8258b68f72d55f4643a3a0034c6a1ec32424f022"
)

MD15_STATS_REL_PATH = "models/md-1.5/variable_statistics_197901-201412.json"
MD15_STATS_SIZE = 3_645
MD15_STATS_SHA256 = (
    "443e684601519cb4c29e20e2e00dce359e7e4b8f476a8af2b3ad8f623b2cafc4"
)

MD15_FORCING_WINDOW: tuple[str, str] = ("1978-10-01T00:00:00", "2025-01-01T00:00:00")
MD15_FORCING_STEPS_TOTAL = 556
MD15_TRAIN_PERIOD: tuple[str, str] = ("1985-01-01", "2014-12-01")
MD15_TEST_PERIOD_START = "2015-01-01"
MD15_PUBLISHED_STARTS: tuple[str, ...] = (
    "1978-10-01T00:00:00",
    "1979-01-01T00:00:00",
    "1985-01-01T00:00:00",
    "2015-01-01T00:00:00",
)

MD15_LAT_COUNT = 121
MD15_LON_COUNT = 240
MD15_PLEVELS_HPA: tuple[int, ...] = (1000, 850, 700, 500, 250, 100, 50)

# Exact 48-entry `variable_names` order serialized in `models/MD-1p5-latest.pth/config.pkl`:
# 37 prognostic channels (`data_dimensions = (37, 121, 240)`), 6 conditioning labels
# (`forcing_dim + seasonality_dim = 3 + 3`), and 5 static subgrid-orography / land-sea-mask
# channels (`n_statics = 5`, embedded in `config.pkl["statics"]` with shape `(5, 121, 240)`).
MD15_CONFIG_VARIABLE_NAMES: tuple[str, ...] = (
    "SPSFC",
    "SKTSFC",
    "VAR_2TSFC",
    "VAR_2DSFC",
    "VAR_10USFC",
    "VAR_10VSFC",
    "MTPRSFC",
    "MSLSFC",
    *(f"t{float(p)}" for p in MD15_PLEVELS_HPA),
    *(f"q{float(p)}" for p in MD15_PLEVELS_HPA),
    *(f"u{float(p)}" for p in MD15_PLEVELS_HPA),
    *(f"v{float(p)}" for p in MD15_PLEVELS_HPA),
    "z500.0",
    "SSTKSFC",
    "CISFC",
    "nanmaskSFC",
    "time0SFC",
    "time1SFC",
    "time2SFC",
    "SDORSFC",
    "ANORSFC",
    "ISORSFC",
    "SLORSFC",
    "LSMSFC",
)
MD15_PROGNOSTIC_VARS: tuple[str, ...] = MD15_CONFIG_VARIABLE_NAMES[:37]
MD15_CONDITIONING_VARS: tuple[str, ...] = MD15_CONFIG_VARIABLE_NAMES[37:43]
MD15_STATIC_VARS: tuple[str, ...] = MD15_CONFIG_VARIABLE_NAMES[43:48]
MD15_SQRT_VARS: frozenset[str] = frozenset(
    {"MTPRSFC", *(f"q{float(p)}" for p in MD15_PLEVELS_HPA)}
)

# Mapping from all 37 prognostic channels to `(netcdf_name, units, long_name)` in
# `<run_dir>/monthly_mean_predictions.nc` (`VAR_2TSFC -> TMP2m`, `SKTSFC -> surface_temperature`).
MD15_OUTPUT_CHANNELS: tuple[tuple[str, str, str, str], ...] = (
    ("SPSFC", "sp", "Pa", "Surface pressure"),
    ("SKTSFC", "surface_temperature", "K", "Skin temperature"),
    ("VAR_2TSFC", "TMP2m", "K", "2 metre temperature"),
    ("VAR_2DSFC", "d2m", "K", "2 metre dewpoint temperature"),
    ("VAR_10USFC", "u10", "m s-1", "10 metre U wind component"),
    ("VAR_10VSFC", "v10", "m s-1", "10 metre V wind component"),
    ("MTPRSFC", "mtpr", "kg m-2 s-1", "Mean total precipitation rate"),
    ("MSLSFC", "msl", "Pa", "Mean sea level pressure"),
    *((f"t{float(p)}", f"t_{p}", "K", f"Air temperature at {p} hPa") for p in MD15_PLEVELS_HPA),
    *((f"q{float(p)}", f"q_{p}", "kg kg-1", f"Specific humidity at {p} hPa") for p in MD15_PLEVELS_HPA),
    *((f"u{float(p)}", f"u_{p}", "m s-1", f"Eastward wind at {p} hPa") for p in MD15_PLEVELS_HPA),
    *((f"v{float(p)}", f"v_{p}", "m s-1", f"Northward wind at {p} hPa") for p in MD15_PLEVELS_HPA),
    ("z500.0", "z500", "m2 s-2", "Geopotential at 500 hPa"),
)

MD15_MONTHLY_REPRESENTATION = (
    "single monthly model state per step (not an average of sub-monthly steps)"
)
MD15_SECONDS_PER_STEP_BASE = 0.25
MD15_SECONDS_PER_MEMBER_MONTH = 0.08
MD15_T2M_BAND_K: tuple[float, float] = (200.0, 330.0)
EXIT_UNSTABLE = 5

_EPO_1970 = datetime.date(1970, 1, 1)


def md15_step_months(year: int, month: int, n_months: int) -> tuple[int, int]:
    """Advance `(year, month)` by `n_months` calendar months (`1 <= month <= 12`)."""
    total = (int(year) * 12 + (int(month) - 1)) + int(n_months)
    new_y, new_m0 = divmod(total, 12)
    return int(new_y), int(new_m0 + 1)


def md15_months_between(start_iso: str, end_iso: str) -> int:
    """Return integer month offset `(y2 - y1) * 12 + (m2 - m1)` between two `YYYY-MM-01` ISO timestamps."""
    y1, m1 = int(start_iso[0:4]), int(start_iso[5:7])
    y2, m2 = int(end_iso[0:4]), int(end_iso[5:7])
    return (y2 - y1) * 12 + (m2 - m1)


def md15_wall_seconds_estimate(steps: int, members: int) -> float:
    """Estimate GPU rollout seconds on 1x A100-80GB (batched across `members`, 15 reverse DDPM steps/month)."""
    n_steps = max(1, int(steps))
    n_members = max(1, int(members))
    return n_steps * (MD15_SECONDS_PER_STEP_BASE + MD15_SECONDS_PER_MEMBER_MONTH * n_members)


def _log(msg: str) -> None:
    print(f"[aimip.md15 {time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}] {msg}", flush=True)


def ensure_torch_harmonics() -> Any:
    """Import `torch_harmonics`, falling back to a trusted sibling `overlay-fme-2026.10.0` install if needed."""
    try:
        import torch_harmonics

        return torch_harmonics
    except ModuleNotFoundError:
        pass

    exe_dir = os.path.dirname(os.path.realpath(sys.executable))
    venv_root = os.path.dirname(exe_dir)
    install_prefix = os.path.dirname(venv_root)
    candidates: list[str] = [
        os.path.join(install_prefix, "overlay-fme-2026.10.0"),
        "/opt/aimip/overlay-fme-2026.10.0",
    ]
    for home_var in ("AIMIP_HOME", "AIMIP_PLAYGROUND"):
        raw_home = os.environ.get(home_var)
        if raw_home:
            candidates.append(os.path.join(raw_home.rstrip("/"), "overlay-fme-2026.10.0"))

    for ov_dir in candidates:
        if not os.path.isdir(ov_dir):
            continue
        try:
            safe_ov = validate_safe_path(ov_dir, label="torch_harmonics overlay")
        except ValidationError:
            continue
        pkg_dir = os.path.join(safe_ov, "torch_harmonics")
        init_py = os.path.join(pkg_dir, "__init__.py")
        if os.path.isfile(init_py) and not os.path.islink(init_py):
            spec = importlib.util.spec_from_file_location(
                "torch_harmonics",
                init_py,
                submodule_search_locations=[pkg_dir],
            )
            if spec is not None and spec.loader is not None:
                mod = importlib.util.module_from_spec(spec)
                sys.modules["torch_harmonics"] = mod
                spec.loader.exec_module(mod)
                return mod

    raise ConfigError(
        "md-1.5 requires `torch_harmonics` in the emulator environment, and neither `venv-emu` nor "
        "`overlay-fme-2026.10.0` provides it. Re-run `install/install-native.sh --with-emu` or rebuild the image."
    )


_UNSET_MD15_SIZE: Any = object()


def extract_and_verify_md15_archive(
    archive_path: str,
    code_target_dir: str,
    model_target_dir: str | None = None,
    *,
    expected_archive_sha256: str | None = None,
    expected_archive_size: Any = _UNSET_MD15_SIZE,
    expected_tree_sha256: str | None = None,
    expected_tree_files: int | None = None,
) -> dict[str, Any]:
    """Verify the `monthly-diffusion` tarball on an `O_NOFOLLOW` fd, extract safely, verify 37-file tree SHA-256, and stage weights."""
    eff_archive_sha = (
        MD15_GITHUB_ARCHIVE_SHA256
        if expected_archive_sha256 is None
        else expected_archive_sha256
    )
    eff_archive_size = (
        MD15_GITHUB_ARCHIVE_SIZE
        if expected_archive_size is _UNSET_MD15_SIZE
        else expected_archive_size
    )
    eff_tree_sha = MD15_TREE_SHA256 if expected_tree_sha256 is None else expected_tree_sha256
    eff_tree_files = (
        MD15_TREE_FILES if expected_tree_files is None else int(expected_tree_files)
    )

    safe_code_target = validate_safe_path(code_target_dir, label="MD-1.5 code target directory")
    parent_dir = os.path.dirname(safe_code_target) or "/"
    makedirs_no_group_other_write(parent_dir)
    stage_root = os.path.join(parent_dir, f".md15_stage_{uuid.uuid4().hex}")
    makedirs_no_group_other_write(stage_root)
    real_stage = os.path.realpath(stage_root)

    try:
        with open_verified_hash_stream(
            archive_path,
            eff_archive_sha,
            label="MD-1.5 GitHub archive",
        ) as stream:
            if eff_archive_size is not None:
                st = os.fstat(stream.fileno())
                if int(st.st_size) != int(eff_archive_size):
                    raise ValidationError(
                        f"MD-1.5 archive size mismatch ({archive_path}): "
                        f"expected {eff_archive_size}, got {st.st_size}."
                    )
            try:
                tf = tarfile.open(fileobj=stream, mode="r:gz")
            except tarfile.TarError as exc:
                raise ValidationError(f"Corrupted MD-1.5 tar.gz archive ({archive_path}): {exc}") from exc

            with tf:
                members = tf.getmembers()
                if not members:
                    raise ValidationError(f"Empty MD-1.5 archive: {archive_path}")
                top_prefixes = {m.name.split("/", 1)[0] for m in members if m.name and m.name != "."}
                strip_prefix = ""
                if len(top_prefixes) == 1:
                    only_top = next(iter(top_prefixes))
                    if all(m.name == only_top or m.name.startswith(only_top + "/") for m in members if m.name):
                        strip_prefix = only_top + "/"

                extracted = 0
                for member in members:
                    raw_name = member.name
                    if not raw_name or "\\" in raw_name or raw_name.startswith("/"):
                        raise ValidationError(f"Unsafe archive entry {raw_name!r} in {archive_path}")
                    if member.issym() or member.islnk() or member.isdev() or member.isfifo():
                        raise ValidationError(
                            f"Refusing special/link tar member {raw_name!r} in {archive_path}"
                        )
                    rel_name = raw_name[len(strip_prefix) :] if strip_prefix and raw_name.startswith(strip_prefix) else raw_name
                    if strip_prefix and raw_name == strip_prefix.rstrip("/"):
                        continue
                    parts = [p for p in rel_name.split("/") if p]
                    if any(p in (".", "..") for p in parts):
                        raise ValidationError(f"Path traversal tar member {raw_name!r} in {archive_path}")
                    if not parts:
                        continue
                    dest_path = os.path.realpath(os.path.join(real_stage, *parts))
                    if dest_path != real_stage and not dest_path.startswith(real_stage + os.sep):
                        raise ValidationError(
                            f"Archive member {raw_name!r} escapes staging directory in {archive_path}"
                        )
                    if member.isdir():
                        makedirs_no_group_other_write(dest_path)
                        continue
                    if not member.isfile():
                        raise ValidationError(f"Refusing non-regular tar member {raw_name!r} in {archive_path}")
                    makedirs_no_group_other_write(os.path.dirname(dest_path))
                    src_fh = tf.extractfile(member)
                    if src_fh is None:
                        raise ValidationError(f"Readable stream missing for {raw_name!r} in {archive_path}")
                    with src_fh, open(dest_path, "wb") as out_fh:
                        shutil.copyfileobj(src_fh, out_fh, length=1024 * 1024)
                    strip_group_other_write(dest_path)
                    extracted += 1
                if extracted == 0:
                    raise ValidationError(f"No regular files extracted from {archive_path}")

        tree_sha, n_files = compute_sha256sum_tree_digest(stage_root, label="MD-1.5 extracted tree")
        if n_files != int(eff_tree_files) or tree_sha.lower() != eff_tree_sha.lower():
            raise ValidationError(
                f"MD-1.5 extracted tree verification failed: expected {eff_tree_files} files "
                f"with sha256={eff_tree_sha.lower()}, got {n_files} files with sha256={tree_sha}."
            )

        if model_target_dir is not None:
            safe_model_dir = validate_safe_path(model_target_dir, label="MD-1.5 model directory")
            ckpt_out = os.path.join(safe_model_dir, "MD-1p5-latest.pth")
            makedirs_no_group_other_write(ckpt_out)
            copies = (
                (
                    os.path.join(stage_root, "models", "MD-1p5-latest.pth", "config.pkl"),
                    os.path.join(ckpt_out, "config.pkl"),
                    MD15_CONFIG_SHA256,
                ),
                (
                    os.path.join(stage_root, "models", "MD-1p5-latest.pth", "weights.pt"),
                    os.path.join(ckpt_out, "weights.pt"),
                    MD15_WEIGHTS_SHA256,
                ),
                (
                    os.path.join(stage_root, "scripts", "variable_statistics_197901-201412.json"),
                    os.path.join(safe_model_dir, "variable_statistics_197901-201412.json"),
                    MD15_STATS_SHA256,
                ),
            )
            for src_file, dst_file, want_sha in copies:
                with open_verified_hash_stream(src_file, want_sha, label="MD-1.5 model artifact") as vfh:
                    tmp_dst = f"{dst_file}.tmp-{uuid.uuid4().hex}"
                    with open(tmp_dst, "wb") as out_fh:
                        shutil.copyfileobj(vfh, out_fh, length=1024 * 1024)
                    strip_group_other_write(tmp_dst)
                    os.replace(tmp_dst, dst_file)

        if os.path.lexists(safe_code_target):
            if os.path.islink(safe_code_target) or os.path.isfile(safe_code_target):
                os.unlink(safe_code_target)
            else:
                shutil.rmtree(safe_code_target)
        os.replace(stage_root, safe_code_target)
    finally:
        if os.path.lexists(stage_root):
            shutil.rmtree(stage_root, ignore_errors=True)

    return {
        "code_dir": safe_code_target,
        "tree_sha256": tree_sha,
        "files": n_files,
    }


def verify_md15_code_dir(
    data_root: str,
    *,
    expected_tree_sha256: str | None = None,
    expected_tree_files: int | None = None,
) -> dict[str, Any]:
    """Verify `<data_root>/code/monthly-diffusion` matches the pinned 37-file tree digest."""
    eff_tree_sha = MD15_TREE_SHA256 if expected_tree_sha256 is None else expected_tree_sha256
    eff_tree_files = (
        MD15_TREE_FILES if expected_tree_files is None else int(expected_tree_files)
    )
    code_dir = validate_safe_path(
        os.path.join(data_root, MD15_CODE_REL_DIR),
        label="MD-1.5 source directory",
    )
    if not os.path.isdir(code_dir) or os.path.islink(code_dir):
        raise ValidationError(
            f"Lock verification failed for {MD15_CODE_REL_DIR}: directory is missing ({code_dir})."
        )
    tree_sha, n_files = compute_sha256sum_tree_digest(code_dir, label="MD-1.5 source directory")
    if n_files != int(eff_tree_files) or tree_sha.lower() != eff_tree_sha.lower():
        raise ValidationError(
            f"Lock verification failed for {MD15_CODE_REL_DIR}: expected {eff_tree_files} files "
            f"/ sha256={eff_tree_sha.lower()}, got {n_files} files / sha256={tree_sha}."
        )
    return {"code_dir": code_dir, "tree_sha256": tree_sha, "files": n_files}


def resolve_md15_code_dir(data_root: str | None = None) -> tuple[str, bool]:
    """Resolve `(code_dir, is_override)` for `$AIMIP_DATA/code/monthly-diffusion`."""
    override = os.environ.get("AIMIP_MD15_CODE")
    if override:
        safe_ovr = validate_safe_path(override, label="AIMIP_MD15_CODE")
        if not os.path.isdir(safe_ovr) or os.path.islink(safe_ovr):
            raise ValidationError(f"AIMIP_MD15_CODE is missing or symlinked: {safe_ovr}")
        return safe_ovr, True

    candidates: list[str] = []
    if data_root:
        candidates.append(os.path.join(data_root, MD15_CODE_REL_DIR))
    env_data = os.environ.get("AIMIP_DATA")
    if env_data:
        candidates.append(os.path.join(env_data, MD15_CODE_REL_DIR))
    for home_var in ("AIMIP_HOME", "AIMIP_PLAYGROUND"):
        root = os.environ.get(home_var)
        if root:
            candidates.append(os.path.join(root, "data", MD15_CODE_REL_DIR))
    for cand in candidates:
        if os.path.isdir(cand) and not os.path.islink(cand):
            return validate_safe_path(cand, label="MD-1.5 code directory"), False

    target = candidates[0] if candidates else f"$AIMIP_DATA/{MD15_CODE_REL_DIR}"
    raise ConfigError(
        f"MD-1.5 source directory not found at {target}; run `aimip fetch -m md-1.5` first."
    )


def load_verified_md_package(
    code_dir: str,
    *,
    verify_tree: bool = True,
    expected_tree_sha256: str | None = None,
    expected_tree_files: int | None = None,
) -> Any:
    """Verify *code_dir* against the pinned tree SHA-256, apply the `src` shim, and import `md`."""
    eff_tree_sha = MD15_TREE_SHA256 if expected_tree_sha256 is None else expected_tree_sha256
    eff_tree_files = (
        MD15_TREE_FILES if expected_tree_files is None else int(expected_tree_files)
    )
    safe_code = validate_safe_path(code_dir, label="MD-1.5 code directory")
    if verify_tree:
        tree_sha, n_files = compute_sha256sum_tree_digest(safe_code, label="MD-1.5 code directory")
        if n_files != int(eff_tree_files) or tree_sha.lower() != eff_tree_sha.lower():
            raise ValidationError(
                f"Refusing to import tampered MD-1.5 code ({safe_code}): expected "
                f"{eff_tree_files} files / {eff_tree_sha.lower()}, got {n_files} / {tree_sha}."
            )

    ensure_torch_harmonics()
    # Keep `$AIMIP_DATA/code/monthly-diffusion` free of `md/__pycache__/*.pyc` so repeated runs pass tree-hash checks.
    sys.dont_write_bytecode = True
    # Upstream `md/prep_data.py:3` executes `import src`; provide an inert module shim.
    sys.modules.setdefault("src", types.ModuleType("src"))

    real_code = os.path.realpath(safe_code)
    existing = sys.modules.get("md")
    if existing is not None:
        existing_file = os.path.realpath(getattr(existing, "__file__", "") or "")
        if existing_file.startswith(real_code + os.sep):
            return existing
        for mod_key in [k for k in list(sys.modules) if k == "md" or k.startswith("md.")]:
            sys.modules.pop(mod_key, None)

    if real_code not in sys.path:
        sys.path.insert(0, real_code)
    import md  # type: ignore[import-not-found]

    loaded_file = os.path.realpath(getattr(md, "__file__", "") or "")
    if not loaded_file.startswith(real_code + os.sep):
        raise ValidationError(
            f"`md` imported from unexpected location {loaded_file!r} (expected under {real_code!r})."
        )
    return md


def select_md15_device(requested: str, cuda_available: bool) -> Any:
    """Resolve active `torch.device`, refusing silent CPU fallback and preserving `x.device == 'cuda'` for GPU 0."""
    import torch

    if requested.startswith("cuda"):
        if not cuda_available:
            raise SystemExit(
                f"{requested} requested for md-1.5, but CUDA is unavailable in this job; refusing silent CPU fallback."
            )
        idx = 0
        if ":" in requested:
            idx = int(requested.split(":", 1)[1])
        torch.cuda.set_device(idx)
        # `md/spectral_conv.py:73,156,171` compares `if x.device == 'cuda':`, which is True in PyTorch
        # only when `device` has no explicit `:0` index (`torch.device('cuda')`).
        return torch.device("cuda") if idx == 0 else torch.device(f"cuda:{idx}")
    if requested == "cpu":
        return torch.device("cpu")
    raise SystemExit(f"Unsupported torch device {requested!r} for md-1.5.")


def load_md15_cvae(
    ckpt_dir: str,
    *,
    config_sha256: str,
    weights_sha256: str,
    device: Any,
    md_module: Any,
    expected_variable_names: Sequence[str] | None = MD15_CONFIG_VARIABLE_NAMES,
) -> tuple[Any, dict[str, Any]]:
    """Verify `config.pkl` and `weights.pt` on `O_NOFOLLOW` fds before restricted unpickling and `strict=True` load."""
    import torch

    safe_dir = validate_safe_path(ckpt_dir, label="MD-1.5 checkpoint directory")
    config_path = os.path.join(safe_dir, "config.pkl")
    weights_path = os.path.join(safe_dir, "weights.pt")

    cfg = pickle_load_verified(
        config_path,
        config_sha256,
        map_location="cpu",
        allow_torch_config=True,
        label="MD-1.5 config.pkl",
    )
    if not isinstance(cfg, dict):
        raise ValidationError(f"MD-1.5 config.pkl did not deserialize to a dict: {type(cfg).__name__}")

    raw_var_names = cfg.get("variable_names")
    if not isinstance(raw_var_names, (list, tuple)):
        raise ValidationError("MD-1.5 config.pkl is missing `variable_names`.")
    var_names = tuple(str(v) for v in raw_var_names)
    if expected_variable_names is not None and var_names != tuple(expected_variable_names):
        raise ValidationError(
            f"MD-1.5 config.pkl variable_names mismatch: got {len(var_names)} entries ({var_names[:5]}...)."
        )

    n_statics = int(cfg.get("n_statics", 0))
    statics_raw = cfg.get("statics")
    if n_statics > 0:
        if not isinstance(statics_raw, torch.Tensor):
            raise ValidationError(
                f"MD-1.5 config.pkl declares n_statics={n_statics} but `statics` is {type(statics_raw).__name__}."
            )
        data_dims = cfg.get("data_dimensions")
        want_hw = (
            (int(data_dims[1]), int(data_dims[2]))
            if isinstance(data_dims, (list, tuple)) and len(data_dims) == 3
            else (MD15_LAT_COUNT, MD15_LON_COUNT)
        )
        if tuple(statics_raw.shape) != (n_statics, want_hw[0], want_hw[1]):
            raise ValidationError(
                f"MD-1.5 config.pkl `statics` shape mismatch: expected "
                f"{(n_statics, want_hw[0], want_hw[1])}, got {tuple(statics_raw.shape)}."
            )
        if not bool(torch.isfinite(statics_raw).all()):
            raise ValidationError("MD-1.5 config.pkl `statics` tensor contains non-finite values.")

    cfg_runtime = dict(cfg)
    cfg_runtime["device"] = device
    if isinstance(cfg_runtime.get("statics"), torch.Tensor):
        cfg_runtime["statics"] = cfg_runtime["statics"].to(device=device, dtype=torch.float32)

    model = md_module.CVAE(**cfg_runtime).to(device)
    # In `md/diffusion.py:177-178`, `nn.Parameter(torch.zeros(1)).to(self.device)` turns `latent_mu` and
    # `latent_log_sigma` into non-leaf plain Tensors on CUDA but keeps `nn.Parameter` on CPU. Normalize
    # both attributes to plain Tensors on `device` so `load_state_dict(..., strict=True)` succeeds on CPU & GPU.
    predictor = getattr(model, "predictor", None)
    if predictor is not None and hasattr(predictor, "_parameters"):
        for attr in ("latent_mu", "latent_log_sigma"):
            if attr in predictor._parameters and predictor._parameters[attr] is not None:
                tensor_val = predictor._parameters.pop(attr).detach().to(device)
                setattr(predictor, attr, tensor_val)

    state = torch_load_verified(
        weights_path,
        weights_sha256,
        map_location=device,
        weights_only=True,
        label="MD-1.5 weights.pt",
    )
    model.load_state_dict(state, strict=True)
    model.eval()
    return model, cfg


def find_coincident_1p5_indices(
    src_lat: Any,
    src_lon: Any,
    *,
    tol_deg: float = 1e-4,
) -> dict[str, Any]:
    """Locate exact source indices coinciding with the 1.5 deg `(121 x 240)` target grid (`fail closed` otherwise).

    Target grid (`data-pipeline/regrid_forcings.ipynb`):
    - `lat_out = np.linspace(-90.0, 90.0, 121)` (`-90.0, -88.5, ..., 90.0`)
    - `lon_out = np.arange(0.0, 360.0, 1.5)` (`0.0, 1.5, ..., 358.5`)
    """
    import numpy as np

    lat_arr = np.asarray(src_lat, dtype=np.float64)
    lon_arr = np.asarray(src_lon, dtype=np.float64) % 360.0
    if lat_arr.ndim != 1 or lon_arr.ndim != 1:
        raise ValidationError(
            f"MD-1.5 requires 1D lat/lon coordinates for coincident 1.5 deg subsampling, "
            f"got shapes {lat_arr.shape} and {lon_arr.shape}."
        )

    target_lat = np.linspace(-90.0, 90.0, MD15_LAT_COUNT, dtype=np.float64)
    target_lon = np.arange(0.0, 360.0, 1.5, dtype=np.float64)

    lat_order = np.argsort(lat_arr)
    lat_sorted = lat_arr[lat_order]
    lat_pos = np.searchsorted(lat_sorted, target_lat)
    lat_idx: list[int] = []
    for k, want_lat in enumerate(target_lat):
        best_j = -1
        best_err = float("inf")
        for cand in (int(lat_pos[k]) - 1, int(lat_pos[k])):
            if 0 <= cand < lat_sorted.size:
                err = abs(float(lat_sorted[cand]) - float(want_lat))
                if err < best_err:
                    best_err = err
                    best_j = cand
        if best_j < 0 or best_err > tol_deg:
            raise ValidationError(
                f"Source latitude grid does not coincide with MD-1.5 1.5 deg target at lat={want_lat:.2f} "
                f"(closest error={best_err:.6g} deg > {tol_deg} deg); refusing non-coincident regrid."
            )
        lat_idx.append(int(lat_order[best_j]))

    lon_order = np.argsort(lon_arr)
    lon_sorted = lon_arr[lon_order]
    lon_pos = np.searchsorted(lon_sorted, target_lon)
    lon_idx: list[int] = []
    for m, want_lon in enumerate(target_lon):
        best_j = -1
        best_err = float("inf")
        for cand in (int(lon_pos[m]) - 1, int(lon_pos[m])):
            if 0 <= cand < lon_sorted.size:
                err = abs(float(lon_sorted[cand]) - float(want_lon))
                if err < best_err:
                    best_err = err
                    best_j = cand
        if best_j < 0 or best_err > tol_deg:
            raise ValidationError(
                f"Source longitude grid does not coincide with MD-1.5 1.5 deg target at lon={want_lon:.2f} "
                f"(closest error={best_err:.6g} deg > {tol_deg} deg); refusing non-coincident regrid."
            )
        lon_idx.append(int(lon_order[best_j]))

    return {
        "lat_indices": np.asarray(lat_idx, dtype=np.int64),
        "lon_indices": np.asarray(lon_idx, dtype=np.int64),
        "target_lat": target_lat,
        "target_lon": target_lon,
    }


def _slice_2d_or_3d_coincident(var_obj: Any, lat_idx: Any, lon_idx: Any) -> Any:
    """Read `(..., 121, 240)` from a `netCDF4.Variable` or `np.ndarray` using fast strided slicing when monotonic."""
    import numpy as np

    lat_i = np.asarray(lat_idx, dtype=np.int64)
    lon_i = np.asarray(lon_idx, dtype=np.int64)

    lat_diffs = np.diff(lat_i)
    lon_diffs = np.diff(lon_i)
    lat_uniform = lat_diffs.size > 0 and bool(np.all(lat_diffs == lat_diffs[0])) and int(lat_diffs[0]) != 0
    lon_uniform = lon_diffs.size > 0 and bool(np.all(lon_diffs == lon_diffs[0])) and int(lon_diffs[0]) != 0

    if lat_uniform and lon_uniform:
        d_lat = int(lat_diffs[0])
        d_lon = int(lon_diffs[0])
        # Read forward on disk for HDF5 chunk efficiency, then reverse if descending (e.g. ERA5 90 -> -90).
        if d_lat > 0:
            lat_sl = slice(int(lat_i[0]), int(lat_i[-1]) + 1, d_lat)
            flip_lat = False
        else:
            lat_sl = slice(int(lat_i[-1]), int(lat_i[0]) + 1, -d_lat)
            flip_lat = True

        if d_lon > 0:
            lon_sl = slice(int(lon_i[0]), int(lon_i[-1]) + 1, d_lon)
            flip_lon = False
        else:
            lon_sl = slice(int(lon_i[-1]), int(lon_i[0]) + 1, -d_lon)
            flip_lon = True

        if var_obj.ndim == 3:
            raw = np.ma.filled(var_obj[:, lat_sl, lon_sl], np.nan).astype(np.float32, copy=False)
            if flip_lat:
                raw = raw[:, ::-1, :]
            if flip_lon:
                raw = raw[:, :, ::-1]
            return np.ascontiguousarray(raw)
        if var_obj.ndim == 2:
            raw = np.ma.filled(var_obj[lat_sl, lon_sl], np.nan).astype(np.float32, copy=False)
            if flip_lat:
                raw = raw[::-1, :]
            if flip_lon:
                raw = raw[:, ::-1]
            return np.ascontiguousarray(raw)

    full = np.ma.filled(var_obj[...], np.nan).astype(np.float32, copy=False)
    if full.ndim == 3:
        return np.ascontiguousarray(full[:, lat_i, :][:, :, lon_i])
    if full.ndim == 2:
        return np.ascontiguousarray(full[lat_i, :][:, lon_i])
    raise ValidationError(f"Expected 2D or 3D forcing variable, got shape {full.shape}.")


def _decode_nc_string_vector(var: Any) -> list[str]:
    """Decode a NetCDF string/character coordinate (`VLEN str`, `|Sn` bytes, or 2D `|S1` char array) to Python `str`."""
    import netCDF4
    import numpy as np

    raw = var[:]
    if hasattr(raw, "dtype") and raw.dtype.kind == "S" and raw.ndim == 2:
        raw = netCDF4.chartostring(raw)
    out: list[str] = []
    for item in raw:
        if isinstance(item, (bytes, np.bytes_)):
            out.append(item.decode("utf-8", errors="replace").rstrip("\x00 "))
        else:
            out.append(str(item).rstrip("\x00 "))
    return out


def _decode_nc_time_ym_strings(nc_ds: Any) -> list[str]:
    """Decode `time` coordinate from an open `netCDF4.Dataset` to `YYYY-MM-DD` strings."""
    import cftime
    import numpy as np

    if "time" not in nc_ds.variables:
        raise ValidationError("Forcing / IC NetCDF is missing a `time` coordinate variable.")
    t_var = nc_ds.variables["time"]
    raw_vals = np.asarray(t_var[:])
    units = getattr(t_var, "units", None)
    cal = getattr(t_var, "calendar", "proleptic_gregorian")
    if units:
        decoded = cftime.num2date(raw_vals, units=units, calendar=cal)
        return [f"{int(dt.year):04d}-{int(dt.month):02d}-{int(dt.day):02d}" for dt in decoded]
    return [str(v)[:10] for v in raw_vals]


def subsample_and_normalize_aimip_forcing(
    forcing_nc_path: str,
    *,
    start_iso: str,
    steps: int,
    train_period: tuple[str, str] = MD15_TRAIN_PERIOD,
    sst_offset_k: float = 0.0,
    allow_short_train_fallback: bool = False,
) -> dict[str, Any]:
    """Subsample `ERA5-0.25deg-monthly-mean-forcing-1978-2024.nc` onto `121 x 240` and standardize per `run_aimip_runs.py`.

    Preserves the exact index pairing between forcing timestamp and predicted month, and reproduces
    `md.preprocess_by_variables(..., group_levels=False)` over `1985-01-01 .. 2014-12-01` (including
    ocean-only NaN masking on `SSTKSFC` and `da.where(da > 1e-3, 1e-3)` logit scaling on `CISFC`).
    """
    import netCDF4
    import numpy as np
    from scipy.special import logit

    safe_nc = validate_safe_path(forcing_nc_path, label="AIMIP monthly forcing NetCDF")
    if not os.path.isfile(safe_nc) or os.path.islink(safe_nc):
        raise ValidationError(f"AIMIP monthly forcing NetCDF is missing or symlinked: {safe_nc}")

    with netCDF4.Dataset(safe_nc, "r") as ds:
        lat_name = "latitude" if "latitude" in ds.variables else ("lat" if "lat" in ds.variables else "")
        lon_name = "longitude" if "longitude" in ds.variables else ("lon" if "lon" in ds.variables else "")
        if not lat_name or not lon_name:
            raise ValidationError(f"Forcing NetCDF {safe_nc} is missing latitude/longitude coordinates.")

        grid_map = find_coincident_1p5_indices(ds.variables[lat_name][:], ds.variables[lon_name][:])
        lat_idx = grid_map["lat_indices"]
        lon_idx = grid_map["lon_indices"]
        date_strs = _decode_nc_time_ym_strings(ds)

        if "sea_surface_temperature" in ds.variables:
            sst = _slice_2d_or_3d_coincident(ds.variables["sea_surface_temperature"], lat_idx, lon_idx)
            sic = _slice_2d_or_3d_coincident(ds.variables["sea_ice_cover"], lat_idx, lon_idx)
            lsm = _slice_2d_or_3d_coincident(ds.variables["land_sea_mask"], lat_idx, lon_idx)
            if lsm.ndim == 2:
                lsm = np.broadcast_to(lsm[np.newaxis, :, :], sst.shape).copy()
        elif "da" in ds.variables and "varlev" in ds.variables:
            varlevs = _decode_nc_string_vector(ds.variables["varlev"])
            da_var = ds.variables["da"]
            raw_da = np.ma.filled(da_var[...], np.nan).astype(np.float32, copy=False)
            # Support `(time, varlev, lat, lon)` or `(varlev, time, lat, lon)`
            dims = tuple(str(d) for d in da_var.dimensions)
            if dims[:2] == ("varlev", "time"):
                raw_da = np.transpose(raw_da, (1, 0, 2, 3))
            raw_da = raw_da[:, :, lat_idx, :][:, :, :, lon_idx]
            sst = raw_da[:, varlevs.index("SSTKSFC"), :, :]
            sic = raw_da[:, varlevs.index("CISFC"), :, :]
            lsm = raw_da[:, varlevs.index("LSMSFC"), :, :]
        else:
            raise ValidationError(
                f"Forcing NetCDF {safe_nc} must provide (`sea_surface_temperature`, `sea_ice_cover`, `land_sea_mask`) "
                "or flat `da(time, varlev, lat, lon)`."
            )

    n_total = len(date_strs)
    start_ymd = start_iso[:10]
    if start_ymd not in date_strs:
        raise ValidationError(
            f"Requested start {start_ymd} is not present in AIMIP forcing time axis "
            f"({date_strs[0]} .. {date_strs[-1]}, {n_total} monthly points)."
        )
    i0 = date_strs.index(start_ymd)
    if i0 + int(steps) > n_total:
        raise ValidationError(
            f"Requested {steps} monthly steps from {start_ymd} exceeds AIMIP forcing end "
            f"{date_strs[-1]} (available from {start_ymd}: {n_total - i0} steps)."
        )

    # Exact `md/preprocessing.py:155-174` mask & per-variable standardization on `train_period`:
    # any cell that is NaN at ANY timestep in the training slice is masked across all timesteps.
    train_indices = [
        idx for idx, d_str in enumerate(date_strs) if train_period[0] <= d_str <= train_period[1]
    ]
    if not train_indices:
        if not allow_short_train_fallback:
            raise ValidationError(
                f"AIMIP forcing {safe_nc} contains no timestamps inside training period "
                f"{train_period[0]} .. {train_period[1]}."
            )
        train_indices = list(range(n_total))

    sst_train_raw = sst[train_indices]
    sic_train_raw = sic[train_indices]
    sst_wet = ~np.isnan(np.min(sst_train_raw, axis=0))
    sic_wet = ~np.isnan(np.min(sic_train_raw, axis=0))

    sst_mask = np.where(sst_wet, np.float32(1.0), np.float32(np.nan))
    sic_mask = np.where(sic_wet, np.float32(1.0), np.float32(np.nan))

    sst_train_masked = sst_train_raw * sst_mask[np.newaxis, :, :]
    sst_mu = float(np.nanmean(sst_train_masked))
    sst_sigma = float(np.nanstd(sst_train_masked))
    if not np.isfinite(sst_sigma) or sst_sigma <= 0.0:
        raise ValidationError(f"Degenerate SST standard deviation ({sst_sigma}) in {safe_nc}.")

    # Exact xarray `da.where(da > eps, other=eps)` semantics (`NaN > eps` is False -> `NaN` becomes `eps`).
    eps = np.float32(1e-3)
    one_minus_eps = np.float32(1.0 - 1e-3)
    sic_train_masked = sic_train_raw * sic_mask[np.newaxis, :, :]
    sic_train_clamped = np.where(sic_train_masked > eps, sic_train_masked, eps)
    sic_train_clamped = np.where(sic_train_clamped < one_minus_eps, sic_train_clamped, one_minus_eps)
    sic_train_logit = logit(sic_train_clamped.astype(np.float64)).astype(np.float32)
    sic_mu = float(np.mean(sic_train_logit))
    sic_sigma = float(np.std(sic_train_logit))
    if not np.isfinite(sic_sigma) or sic_sigma <= 0.0:
        raise ValidationError(f"Degenerate sea-ice logit standard deviation ({sic_sigma}) in {safe_nc}.")

    # Transform requested slice `[i0 : i0 + steps]` using full-series mask (`md/preprocessing.py:155-162`).
    sst_full_wet = ~np.isnan(np.min(sst, axis=0))
    sic_full_wet = ~np.isnan(np.min(sic, axis=0))
    sst_full_mask = np.where(sst_full_wet, np.float32(1.0), np.float32(np.nan))
    sic_full_mask = np.where(sic_full_wet, np.float32(1.0), np.float32(np.nan))

    sst_slice = (sst[i0 : i0 + steps] + np.float32(sst_offset_k)) * sst_full_mask[np.newaxis, :, :]
    sst_norm = np.nan_to_num((sst_slice - np.float32(sst_mu)) / np.float32(sst_sigma), nan=0.0).astype(np.float32)

    sic_slice = sic[i0 : i0 + steps] * sic_full_mask[np.newaxis, :, :]
    sic_clamped = np.where(sic_slice > eps, sic_slice, eps)
    sic_clamped = np.where(sic_clamped < one_minus_eps, sic_clamped, one_minus_eps)
    sic_logit = logit(sic_clamped.astype(np.float64)).astype(np.float32)
    sic_norm = np.nan_to_num((sic_logit - np.float32(sic_mu)) / np.float32(sic_sigma), nan=0.0).astype(np.float32)

    lsm_slice = np.nan_to_num(lsm[i0 : i0 + steps], nan=0.0).astype(np.float32)

    forcings_f = np.stack([sst_norm, sic_norm, lsm_slice], axis=1)  # (steps, 3, 121, 240)
    slice_dates = date_strs[i0 : i0 + steps]
    months = np.asarray([[int(d[5:7])] for d in slice_dates], dtype=np.float32)  # (steps, 1)

    return {
        "forcings": forcings_f,
        "months": months,
        "dates": slice_dates,
        "target_lat": grid_map["target_lat"],
        "target_lon": grid_map["target_lon"],
        "forcing_total_steps": n_total,
        "forcing_first": date_strs[0],
        "forcing_last": date_strs[-1],
        "land_nan_cells_per_step": int(np.sum(~sst_full_wet)),
        "sst_mu": sst_mu,
        "sst_sigma": sst_sigma,
        "sic_mu": sic_mu,
        "sic_sigma": sic_sigma,
    }


def normalize_md15_prognostic_state(
    raw_state: Any,
    stats_dict: Mapping[str, Mapping[str, float]],
) -> Any:
    """Standardize physical `(B, 37, 121, 240)` prognostic channels using `variable_statistics_197901-201412.json`."""
    import numpy as np

    arr = np.asarray(raw_state, dtype=np.float32)
    if arr.ndim != 4 or arr.shape[1:] != (len(MD15_PROGNOSTIC_VARS), MD15_LAT_COUNT, MD15_LON_COUNT):
        raise ValidationError(
            f"Expected MD-1.5 state shape (B, 37, 121, 240), got {arr.shape}."
        )
    if not np.isfinite(arr).all():
        bad_ch = [
            MD15_PROGNOSTIC_VARS[c]
            for c in range(arr.shape[1])
            if not np.isfinite(arr[:, c, :, :]).all()
        ]
        raise ValidationError(
            f"MD-1.5 initial condition contains non-finite values in prognostic channel(s): {bad_ch[:5]}"
        )

    out = np.empty_like(arr, dtype=np.float32)
    for idx, ch_name in enumerate(MD15_PROGNOSTIC_VARS):
        params = stats_dict.get(ch_name)
        if not isinstance(params, Mapping) or "mu" not in params or "sigma" not in params:
            raise ValidationError(f"MD-1.5 statistics JSON is missing mu/sigma for {ch_name!r}.")
        mu = np.float32(params["mu"])
        sigma = np.float32(params["sigma"])
        slab = arr[:, idx, :, :]
        if ch_name in MD15_SQRT_VARS:
            slab = np.sqrt(np.maximum(slab, np.float32(0.0)))
        out[:, idx, :, :] = (slab - mu) / sigma
    return out


def denormalize_md15_prognostic_state(
    norm_state: Any,
    stats_dict: Mapping[str, Mapping[str, float]],
) -> Any:
    """Undo standardization (`undo_preprocess_by_variables(..., group_levels=False)`) on `(B, 37, 121, 240)`."""
    import numpy as np

    arr = np.asarray(norm_state, dtype=np.float32)
    out = np.empty_like(arr, dtype=np.float32)
    for idx, ch_name in enumerate(MD15_PROGNOSTIC_VARS):
        params = stats_dict[ch_name]
        mu = np.float32(params["mu"])
        sigma = np.float32(params["sigma"])
        slab = arr[:, idx, :, :] * sigma + mu
        if ch_name in MD15_SQRT_VARS:
            slab = slab * slab
        out[:, idx, :, :] = slab
    return out


_CF_SURFACE_ALIASES: dict[str, tuple[str, ...]] = {
    "SPSFC": ("SPSFC", "sp", "surface_pressure", "PRESsfc"),
    "SKTSFC": ("SKTSFC", "skt", "surface_temperature", "skin_temperature"),
    "VAR_2TSFC": ("VAR_2TSFC", "TMP2m", "t2m", "2t", "2m_temperature"),
    "VAR_2DSFC": ("VAR_2DSFC", "d2m", "2d", "2m_dewpoint_temperature"),
    "VAR_10USFC": ("VAR_10USFC", "u10", "10u", "10m_u_component_of_wind", "UGRD10m"),
    "VAR_10VSFC": ("VAR_10VSFC", "v10", "10v", "10m_v_component_of_wind", "VGRD10m"),
    "MTPRSFC": ("MTPRSFC", "mtpr", "avg_tprate", "mean_total_precipitation_rate"),
    "MSLSFC": ("MSLSFC", "msl", "mean_sea_level_pressure", "PRMSLmsl"),
    "z500.0": ("z500.0", "z500", "geopotential_500"),
}

_CF_PLEVEL_ALIASES: dict[str, tuple[str, ...]] = {
    "t": ("temperature", "t", "ta"),
    "q": ("specific_humidity", "q", "hus"),
    "u": ("u_component_of_wind", "u", "ua"),
    "v": ("v_component_of_wind", "v", "va"),
}


def load_md15_initial_state(
    ic_nc_path: str,
    *,
    start_iso: str,
    members: int,
    stats_dict: Mapping[str, Mapping[str, float]],
) -> tuple[Any, dict[str, Any]]:
    """Load and standardize a `1.5 deg` `(121 x 240)` initial state at `start_iso` into `(members, 37, 121, 240)`.

    Supports:
    1. 3D per-variable NetCDFs (`SPSFC..z500.0` or `TMP2m..z500` on `(time|sample, lat, lon)`, incl. `restart.nc`),
    2. 4D flat NetCDFs (`da(time, varlev, lat, lon)` matching `era5-flat-1p5x1p5.nc`), and
    3. CF/ERA5 NetCDFs combining surface variables with 4D `(time, level, lat, lon)` pressure-level arrays.
    """
    import netCDF4
    import numpy as np

    safe_ic = validate_safe_path(ic_nc_path, label="MD-1.5 initial condition NetCDF")
    if not os.path.isfile(safe_ic) or os.path.islink(safe_ic):
        raise ValidationError(f"MD-1.5 initial condition NetCDF is missing or symlinked: {safe_ic}")

    want_ymd = start_iso[:10]
    with netCDF4.Dataset(safe_ic, "r") as ds:
        lat_name = "lat" if "lat" in ds.variables else ("latitude" if "latitude" in ds.variables else "")
        lon_name = "lon" if "lon" in ds.variables else ("longitude" if "longitude" in ds.variables else "")
        if not lat_name or not lon_name:
            raise ValidationError(f"MD-1.5 initial condition {safe_ic} is missing lat/lon coordinates.")

        grid_map = find_coincident_1p5_indices(ds.variables[lat_name][:], ds.variables[lon_name][:])
        lat_idx = grid_map["lat_indices"]
        lon_idx = grid_map["lon_indices"]

        date_strs = _decode_nc_time_ym_strings(ds)
        t_var = ds.variables["time"]
        sample_dim = str(t_var.dimensions[0]) if t_var.dimensions else "time"

        matching_indices = [i for i, d_str in enumerate(date_strs) if d_str == want_ymd]
        if not matching_indices:
            raise ValidationError(
                f"Initial condition {safe_ic} does not contain timestamp {want_ymd} "
                f"(available: {date_strs[:5]})."
            )
        if sample_dim == "sample" and len(matching_indices) == int(members):
            sel_indices = matching_indices
        else:
            sel_indices = [matching_indices[0]]
        n_in = len(sel_indices)

        raw_channels = np.empty((n_in, len(MD15_PROGNOSTIC_VARS), MD15_LAT_COUNT, MD15_LON_COUNT), dtype=np.float32)

        if "da" in ds.variables and "varlev" in ds.variables:
            varlevs = _decode_nc_string_vector(ds.variables["varlev"])
            da_var = ds.variables["da"]
            dims = tuple(str(d) for d in da_var.dimensions)
            arr_all = np.ma.filled(da_var[...], np.nan).astype(np.float32, copy=False)
            if dims[:2] == ("varlev", sample_dim):
                arr_all = np.transpose(arr_all, (1, 0, 2, 3))
            arr_sel = arr_all[sel_indices, :, :, :][:, :, lat_idx, :][:, :, :, lon_idx]
            for c_idx, ch_name in enumerate(MD15_PROGNOSTIC_VARS):
                if ch_name not in varlevs:
                    raise ValidationError(
                        f"Flat initial condition {safe_ic} is missing required channel {ch_name!r}."
                    )
                raw_channels[:, c_idx, :, :] = arr_sel[:, varlevs.index(ch_name), :, :]
        else:
            lev_vals: list[int] = []
            lev_name = ""
            for cand_lev in ("level", "pressure_level", "plev", "isobaricInhPa"):
                if cand_lev in ds.variables:
                    lev_name = cand_lev
                    raw_lev = np.asarray(ds.variables[cand_lev][:], dtype=np.float64)
                    # Convert Pa -> hPa if stored in Pa (`100000..5000`)
                    if raw_lev.size > 0 and float(np.max(raw_lev)) > 2000.0:
                        raw_lev = raw_lev / 100.0
                    lev_vals = [int(round(float(v))) for v in raw_lev]
                    break

            for c_idx, ch_name in enumerate(MD15_PROGNOSTIC_VARS):
                found_slab = None
                # 1. Exact or alias 3D variable (`(time|sample, lat, lon)`)
                candidates: list[str] = [ch_name]
                if ch_name in _CF_SURFACE_ALIASES:
                    candidates.extend(_CF_SURFACE_ALIASES[ch_name])
                elif ch_name[0] in ("t", "q", "u", "v") and ch_name.endswith(".0"):
                    p_int = int(float(ch_name[1:]))
                    candidates.append(f"{ch_name[0]}_{p_int}")

                for cand in candidates:
                    if cand in ds.variables and ds.variables[cand].ndim == 3:
                        slab_all = _slice_2d_or_3d_coincident(ds.variables[cand], lat_idx, lon_idx)
                        found_slab = slab_all[sel_indices, :, :]
                        break

                # 2. Fallback to 4D pressure-level variable `(time|sample, level, lat, lon)`
                if found_slab is None and lev_name:
                    if ch_name == "z500.0":
                        for z_cand in ("z", "geopotential"):
                            if z_cand in ds.variables and ds.variables[z_cand].ndim == 4 and 500 in lev_vals:
                                z_lev_i = lev_vals.index(500)
                                z_var = np.ma.filled(ds.variables[z_cand][:, z_lev_i, :, :], np.nan).astype(np.float32)
                                found_slab = z_var[sel_indices, :, :][:, lat_idx, :][:, :, lon_idx]
                                break
                    elif ch_name[0] in _CF_PLEVEL_ALIASES and ch_name.endswith(".0"):
                        prefix = ch_name[0]
                        p_int = int(float(ch_name[1:]))
                        if p_int in lev_vals:
                            p_idx = lev_vals.index(p_int)
                            for pl_cand in _CF_PLEVEL_ALIASES[prefix]:
                                if pl_cand in ds.variables and ds.variables[pl_cand].ndim == 4:
                                    pl_var = np.ma.filled(ds.variables[pl_cand][:, p_idx, :, :], np.nan).astype(np.float32)
                                    found_slab = pl_var[sel_indices, :, :][:, lat_idx, :][:, :, lon_idx]
                                    break

                if found_slab is None:
                    raise ValidationError(
                        f"Initial condition {safe_ic} is missing prognostic variable {ch_name!r}."
                    )
                raw_channels[:, c_idx, :, :] = found_slab

    norm_ic = normalize_md15_prognostic_state(raw_channels, stats_dict)
    if norm_ic.shape[0] == 1 and int(members) > 1:
        norm_ic = np.repeat(norm_ic, int(members), axis=0)
    return norm_ic, {"ic_samples_loaded": n_in, "start": want_ymd}


def _write_md15_outputs(
    out_dir: str,
    *,
    all_months_phys: Any,
    month_dates: Sequence[str],
    lat: Any,
    lon: Any,
    seed: int,
) -> dict[str, str]:
    """Write `<out_dir>/monthly_mean_predictions.nc` (all 37 channels, `counts=1`) and `<out_dir>/restart.nc`."""
    import netCDF4
    import numpy as np

    # `all_months_phys`: shape `(steps, members, 37, 121, 240)`
    n_steps, n_members, _, n_lat, n_lon = all_months_phys.shape
    monthly_nc = os.path.join(out_dir, "monthly_mean_predictions.nc")
    restart_nc = os.path.join(out_dir, "restart.nc")

    valid_days = np.empty((n_members, n_steps), dtype=np.int64)
    for s_idx, d_str in enumerate(month_dates):
        y, m = int(d_str[0:4]), int(d_str[5:7])
        mid_day = (datetime.date(y, m, 15) - _EPO_1970).days
        valid_days[:, s_idx] = mid_day

    tmp_monthly = f"{monthly_nc}.tmp-{os.getpid()}"
    with netCDF4.Dataset(tmp_monthly, "w", format="NETCDF4") as ds:
        ds.Conventions = "CF-1.8"
        ds.model = "md-1.5"
        ds.upstream_commit = MD15_COMMIT
        ds.monthly_representation = MD15_MONTHLY_REPRESENTATION
        ds.forcing_time_convention = (
            "AIMIP Zenodo 17065758 forcing values are centred on the start of each month; "
            "index pairing matches scripts/run_aimip_runs.py."
        )
        ds.seed = int(seed)

        ds.createDimension("sample", n_members)
        ds.createDimension("time", n_steps)
        ds.createDimension("lat", n_lat)
        ds.createDimension("lon", n_lon)

        v_lat = ds.createVariable("lat", "f4", ("lat",))
        v_lat.units = "degrees_north"
        v_lat.standard_name = "latitude"
        v_lat[:] = np.asarray(lat, dtype=np.float32)

        v_lon = ds.createVariable("lon", "f4", ("lon",))
        v_lon.units = "degrees_east"
        v_lon.standard_name = "longitude"
        v_lon[:] = np.asarray(lon, dtype=np.float32)

        v_time = ds.createVariable("time", "i8", ("time",))
        v_time.units = "months"
        v_time[:] = np.arange(n_steps, dtype=np.int64)

        v_valid = ds.createVariable("valid_time", "i8", ("sample", "time"))
        v_valid.units = "days since 1970-01-01 00:00:00"
        v_valid.calendar = "proleptic_gregorian"
        v_valid[:] = valid_days

        v_counts = ds.createVariable("counts", "i4", ("sample", "time"))
        v_counts.long_name = "Single monthly model state per step (counts = 1)"
        v_counts[:] = np.ones((n_members, n_steps), dtype=np.int32)

        # Transpose `(steps, members, 37, lat, lon)` -> `(members, steps, 37, lat, lon)`
        by_sample = np.transpose(all_months_phys, (1, 0, 2, 3, 4))
        for c_idx, (md_key, out_name, units, long_name) in enumerate(MD15_OUTPUT_CHANNELS):
            var = ds.createVariable(out_name, "f4", ("sample", "time", "lat", "lon"))
            var.units = units
            var.long_name = long_name
            var.md15_channel = md_key
            var.cell_methods = "time: point (single monthly latent-diffusion state)"
            var[:] = by_sample[:, :, c_idx, :, :]

    strip_group_other_write(tmp_monthly)
    os.replace(tmp_monthly, monthly_nc)

    last_date_str = month_dates[-1]
    ly, lm = int(last_date_str[0:4]), int(last_date_str[5:7])
    last_days = (datetime.date(ly, lm, 1) - _EPO_1970).days

    tmp_restart = f"{restart_nc}.tmp-{os.getpid()}"
    with netCDF4.Dataset(tmp_restart, "w", format="NETCDF4") as rds:
        rds.Conventions = "CF-1.8"
        rds.model = "md-1.5"
        rds.upstream_commit = MD15_COMMIT
        rds.restart_timestamp = f"{ly:04d}-{lm:02d}-01T00:00:00"

        rds.createDimension("sample", n_members)
        rds.createDimension("lat", n_lat)
        rds.createDimension("lon", n_lon)

        rlat = rds.createVariable("lat", "f4", ("lat",))
        rlat.units = "degrees_north"
        rlat[:] = np.asarray(lat, dtype=np.float32)

        rlon = rds.createVariable("lon", "f4", ("lon",))
        rlon.units = "degrees_east"
        rlon[:] = np.asarray(lon, dtype=np.float32)

        rtime = rds.createVariable("time", "i8", ("sample",))
        rtime.units = "days since 1970-01-01 00:00:00"
        rtime.calendar = "proleptic_gregorian"
        rtime[:] = np.full((n_members,), last_days, dtype=np.int64)

        final_step = all_months_phys[-1]  # (members, 37, lat, lon)
        for c_idx, (md_key, _out_name, units, long_name) in enumerate(MD15_OUTPUT_CHANNELS):
            rvar = rds.createVariable(md_key, "f4", ("sample", "lat", "lon"))
            rvar.units = units
            rvar.long_name = long_name
            rvar[:] = final_step[:, c_idx, :, :]

    strip_group_other_write(tmp_restart)
    os.replace(tmp_restart, restart_nc)
    return {"monthly_nc": monthly_nc, "restart_nc": restart_nc}


def run_md15(out_dir: str, job: Mapping[str, Any]) -> dict[str, Any]:
    """Execute batched `md-1.5` rollout (`T = 15` reverse DDPM steps/month) and write CF outputs."""
    import numpy as np
    import torch

    steps = int(job["steps"])
    members = int(job["members"])
    seed = int(job["seed"])
    start_iso = str(job["start"])
    requested_device = str(job.get("device", "cuda:0"))
    device = select_md15_device(requested_device, torch.cuda.is_available())

    code_dir_raw = job.get("code_dir")
    if code_dir_raw:
        code_dir = validate_safe_path(str(code_dir_raw), label="MD-1.5 code_dir")
        verify_tree = bool(job.get("verify_code_tree", True))
    else:
        code_dir, is_ovr = resolve_md15_code_dir(job.get("data_root"))
        verify_tree = not is_ovr

    md_mod = load_verified_md_package(
        code_dir,
        verify_tree=verify_tree,
        expected_tree_sha256=(
            str(job["code_tree_sha256"]) if job.get("code_tree_sha256") else None
        ),
        expected_tree_files=(
            int(job["code_tree_files"]) if job.get("code_tree_files") else None
        ),
    )

    stats_path = validate_safe_path(str(job["stats_file"]), label="MD-1.5 statistics JSON")
    stats_sha = str(job["stats_sha256"])
    with open_verified_hash_stream(stats_path, stats_sha, label="MD-1.5 statistics JSON") as sfh:
        stats_dict = json.loads(sfh.read().decode("utf-8"))

    forcing_info = subsample_and_normalize_aimip_forcing(
        str(job["forcing_file"]),
        start_iso=start_iso,
        steps=steps,
        sst_offset_k=float(job.get("sst_offset_k", 0.0)),
        allow_short_train_fallback=bool(job.get("allow_short_train_fallback", False)),
    )

    ic_path = str(job["ic_file"])
    ic_sha = job.get("ic_sha256")
    if ic_sha:
        with open_verified_hash_stream(ic_path, str(ic_sha), label="MD-1.5 initial condition"):
            pass
    norm_ic, ic_meta = load_md15_initial_state(
        ic_path,
        start_iso=start_iso,
        members=members,
        stats_dict=stats_dict,
    )

    torch.manual_seed(seed)
    np.random.seed(seed % (2**32 - 1))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    model, _cfg = load_md15_cvae(
        str(job["ckpt_dir"]),
        config_sha256=str(job["config_sha256"]),
        weights_sha256=str(job["weights_sha256"]),
        device=device,
        md_module=md_mod,
        expected_variable_names=job.get("expected_variable_names", MD15_CONFIG_VARIABLE_NAMES),
    )

    lat = forcing_info["target_lat"]
    lon = forcing_info["target_lon"]
    cos_w = np.cos(np.deg2rad(lat)).astype(np.float64)[:, np.newaxis]
    t2m_idx = MD15_PROGNOSTIC_VARS.index("VAR_2TSFC")

    forcings_t = torch.as_tensor(forcing_info["forcings"], dtype=torch.float32, device=device)
    months_t = torch.as_tensor(forcing_info["months"], dtype=torch.float32, device=device)
    phys_t = torch.as_tensor(norm_ic, dtype=torch.float32, device=device)

    all_months_phys = np.empty(
        (steps, members, len(MD15_PROGNOSTIC_VARS), MD15_LAT_COUNT, MD15_LON_COUNT),
        dtype=np.float32,
    )

    def _check_and_store(step_i: int, decoded_norm: Any) -> None:
        phys_step = denormalize_md15_prognostic_state(decoded_norm, stats_dict)
        t2m_slab = phys_step[:, t2m_idx, :, :].astype(np.float64)
        if not np.isfinite(phys_step).all():
            _log(f"UNSTABLE at step {step_i} ({forcing_info['dates'][step_i]}): non-finite prognostic output")
            raise SystemExit(EXIT_UNSTABLE)
        for m_i in range(members):
            gm = float(np.sum(t2m_slab[m_i] * cos_w) / (np.sum(cos_w) * MD15_LON_COUNT))
            if not (MD15_T2M_BAND_K[0] <= gm <= MD15_T2M_BAND_K[1]):
                _log(
                    f"UNSTABLE at step {step_i} ({forcing_info['dates'][step_i]}), member {m_i}: "
                    f"global-mean TMP2m={gm:.2f} K outside {MD15_T2M_BAND_K}"
                )
                raise SystemExit(EXIT_UNSTABLE)
        all_months_phys[step_i] = phys_step

    t_run0 = time.time()
    _log(
        f"md-1.5 rollout start={start_iso[:10]} steps={steps} members={members} seed={seed} device={device}"
    )
    with torch.no_grad():
        f0 = forcings_t[0].unsqueeze(0).expand(members, -1, -1, -1)
        m0 = model.seasonality_embedding(months_t[0].unsqueeze(0)).expand(members, -1, -1, -1)
        cond0 = torch.cat([f0, m0], dim=1)
        mu_state, _logvar, cs = model.encoder(phys_t, cond0)
        dec0 = model.decoder(mu_state, cs).detach().cpu().numpy()
        _check_and_store(0, dec0)

        for step_idx in range(1, steps):
            mu_state = model.predictor(mu_state, cs[-1])
            f_step = forcings_t[step_idx].unsqueeze(0).expand(members, -1, -1, -1)
            m_step = model.seasonality_embedding(months_t[step_idx].unsqueeze(0)).expand(members, -1, -1, -1)
            cond_step = torch.cat([f_step, m_step], dim=1)
            cs = model.encoder.embed_conditions(cond_step)
            dec_step = model.decoder(mu_state, cs).detach().cpu().numpy()
            _check_and_store(step_idx, dec_step)

    if device.type == "cuda":
        torch.cuda.synchronize()
    wall_infer = round(time.time() - t_run0, 3)

    paths = _write_md15_outputs(
        out_dir,
        all_months_phys=all_months_phys,
        month_dates=forcing_info["dates"],
        lat=lat,
        lon=lon,
        seed=seed,
    )
    _log(f"md-1.5 rollout completed in {wall_infer:.2f}s -> {paths['monthly_nc']}")

    return {
        "months_written": {"atmosphere": steps},
        "first_month": forcing_info["dates"][0][:7],
        "last_month": forcing_info["dates"][-1][:7],
        "members": members,
        "seed": seed,
        "monthly_representation": MD15_MONTHLY_REPRESENTATION,
        "spinup_warning": (
            steps < 120
            and "Requires >= 10-year (120-month) spin-up (Hall & Molina, 2026); early years have strong QBO spin-up bias."
        )
        or None,
        "in_training_or_val_period": forcing_info["dates"][0] < MD15_TEST_PERIOD_START,
        "inference_seconds": wall_infer,
        **ic_meta,
    }
