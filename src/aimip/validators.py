"""Strict input validators preventing shell/SLURM/YAML injection and path traversal.

Every user-supplied or environment-derived value interpolated into configs,
manifests, or SLURM batch scripts MUST pass through a validator in this module.
Character allowlists match `container/aimip-wrapper.in` and `aimip.submit` (`safe_path`).
"""

from __future__ import annotations

import dataclasses
import errno
import math
import os
import re
import stat
from collections.abc import Sequence

from .errors import ValidationError

# Matches `container/aimip-wrapper.in` safe_path(): absolute path, only [A-Za-z0-9._/+-], no `..`.
_SAFE_PATH_RE = re.compile(r"^/[A-Za-z0-9._/+-]+$")

# Default standalone run / output directories when `$AIMIP_HOME` is unset.
# Under the installed launcher (`<prefix>/bin/aimip`), `AIMIP_HOME` is always set and
# writable output roots come strictly from `$AIMIP_DEFAULT_OUT_ROOT`, `$AIMIP_RUN_ROOTS`, or `site.json`.
ALLOWED_RUN_ROOTS: tuple[str, ...] = ("/scratch/", "/work/")
_FORBIDDEN_RUN_ROOT_PREFIXES: tuple[str, ...] = (
    "/bin/",
    "/sbin/",
    "/usr/",
    "/etc/",
    "/lib/",
    "/lib64/",
    "/boot/",
    "/proc/",
    "/sys/",
    "/dev/",
)
_FORBIDDEN_EXACT_RUN_ROOTS: tuple[str, ...] = ("/", "/tmp/", "/var/tmp/")

_MODEL_NAME_RE = re.compile(r"^[a-z0-9][a-z0-9._-]{0,63}$")
_FORCING_SET_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$")
_MODULE_NAME_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*(\.[A-Za-z_][A-Za-z0-9_]*)*$")
_ISO_DATETIME_RE = re.compile(
    r"^(?P<year>\d{4})-(?P<month>\d{2})-(?P<day>\d{2})"
    r"(?:T(?P<hour>\d{2}):(?P<minute>\d{2}):(?P<second>\d{2}))?$"
)
_SLURM_TOKEN_RE = re.compile(r"^[A-Za-z0-9_.+:-]+$")
_SLURM_MEM_RE = re.compile(r"^[1-9]\d{0,7}[KMGTP]?$")
_SLURM_TIME_RE = re.compile(
    r"^(?:(?:\d{1,2}-)?\d{1,3}:[0-5]\d:[0-5]\d|[1-9]\d{0,4})$"
)
_SLURM_JOB_ID_RE = re.compile(r"^\d+(?:_\d+)?$")
_SHA256_RE = re.compile(r"^[0-9a-fA-F]{64}$")

# fme.core.set_seed adds up to +5 and passes (seed + 1) to np.random.seed (uint32)
# and alternate_seed uses 2**31; keep seeds within [0, 2**31 - 10].
MAX_SEED = (2**31) - 10


def _reject_control_chars(value: str, label: str) -> None:
    for ch in value:
        code = ord(ch)
        if code < 32 or code == 127:
            raise ValidationError(
                f"Invalid {label}: contains control character (ord={code})."
            )


def validate_safe_path(value: str, label: str = "path") -> str:
    """Validate that *value* is a clean, absolute POSIX path matching `safe_path`.

    Rules enforced:
    - Non-empty string starting with '/'
    - Only characters in `[A-Za-z0-9._/+-]` (no spaces, quotes, `$`, backticks, `:`, `;`, globs, or control chars)
    - Must not contain `..` anywhere
    """
    if not isinstance(value, str) or not value:
        raise ValidationError(f"Invalid {label}: path must be a non-empty string.")

    _reject_control_chars(value, label)

    if not value.startswith("/") or not os.path.isabs(value):
        raise ValidationError(
            f"Invalid {label} {value!r}: path must be absolute (start with '/')."
        )

    if ".." in value:
        raise ValidationError(
            f"Invalid {label} {value!r}: '..' is not allowed in paths."
        )

    if not _SAFE_PATH_RE.match(value):
        bad = sorted({ch for ch in value if not re.match(r"[A-Za-z0-9._/+-]", ch)})
        shown = ", ".join(repr(c) for c in bad)
        raise ValidationError(
            f"Invalid {label} {value!r}: contains forbidden character(s): {shown} "
            "(allowed: A-Z, a-z, 0-9, '.', '_', '/', '+', '-')."
        )

    normalized = os.path.normpath(value)
    if normalized.startswith("//"):
        normalized = "/" + normalized.lstrip("/")
    if normalized == "/" or ".." in normalized.split("/"):
        raise ValidationError(
            f"Invalid {label} {value!r}: root or parent traversal path is forbidden."
        )
    return normalized


def _append_candidate_run_root(
    roots: list[str],
    raw_root: str | None,
    *,
    forbid_exact: tuple[str, ...] = _FORBIDDEN_EXACT_RUN_ROOTS,
) -> None:
    """Validate *raw_root* and append `<norm>/` if not already present or a forbidden system prefix."""
    if not raw_root:
        return
    try:
        norm = validate_safe_path(raw_root, label="configured root")
    except ValidationError:
        return
    with_slash = norm.rstrip("/") + "/"
    if with_slash in forbid_exact:
        return
    if any(
        with_slash == bad or with_slash.startswith(bad)
        for bad in _FORBIDDEN_RUN_ROOT_PREFIXES
    ):
        return
    if with_slash not in roots:
        roots.append(with_slash)


def _path_under_root(normalized: str, root: str, *, disallow_exact: bool = False) -> bool:
    """Return True if *normalized* equals (unless *disallow_exact*) or resides strictly inside *root*."""
    clean_root = root.rstrip("/")
    if not disallow_exact and normalized == clean_root:
        return True
    return normalized.startswith(root) and len(normalized) > len(root)


def get_allowed_run_roots() -> tuple[str, ...]:
    """Return active writable output/run directory prefixes.

    When `$AIMIP_HOME` is set (installed launcher mode), writable roots
    come strictly from `$AIMIP_DEFAULT_OUT_ROOT`, `$AIMIP_RUN_ROOTS`, and `site.json`
    (`run_root` / `allowed_run_roots`) so unmounted host paths and read-only binds are refused.
    When `$AIMIP_HOME` is unset (standalone CLI mode), `ALLOWED_RUN_ROOTS` also applies.
    """
    roots: list[str] = []
    if not os.environ.get("AIMIP_HOME", "").strip():
        for r in ALLOWED_RUN_ROOTS:
            with_slash = r if r.endswith("/") else f"{r}/"
            if with_slash not in roots:
                roots.append(with_slash)

    # 1. Installed launcher (`<prefix>/bin/aimip`) sets $AIMIP_DEFAULT_OUT_ROOT on every site.
    env_default_out = os.environ.get("AIMIP_DEFAULT_OUT_ROOT")
    if env_default_out:
        _append_candidate_run_root(roots, env_default_out)

    # 2. Optional colon-separated $AIMIP_RUN_ROOTS override for portable HPC sites.
    env_run_roots = os.environ.get("AIMIP_RUN_ROOTS")
    if env_run_roots:
        for part in env_run_roots.split(":"):
            if part.strip():
                _append_candidate_run_root(roots, part.strip())

    # 3. `site.json` `run_root` and `allowed_run_roots` if present.
    try:
        from .site_config import expand_run_root_template, load_site_config, resolve_site_config_path

        cfg_path, _ = resolve_site_config_path(None, require_target=False)
        if cfg_path is not None and os.path.isfile(cfg_path):
            site_cfg = load_site_config(cfg_path)
            expanded_rr = expand_run_root_template(site_cfg.run_root)
            if expanded_rr:
                _append_candidate_run_root(roots, expanded_rr)
            for extra_rr in site_cfg.allowed_run_roots:
                _append_candidate_run_root(roots, extra_rr)
    except Exception:
        pass

    return tuple(roots)


def get_readonly_mount_roots() -> tuple[str, ...]:
    """Return read-only container mount prefixes (`AIMIP_HOME`, `AIMIP_PLAYGROUND`, `AIMIP_DATA`, `AIMIP_EXTRA_RO_BINDS`)."""
    ro_roots: list[str] = []
    for env_key in ("AIMIP_HOME", "AIMIP_PLAYGROUND", "AIMIP_DATA"):
        val = os.environ.get(env_key, "").strip()
        if val:
            _append_candidate_run_root(ro_roots, val, forbid_exact=("/",))

    extra_ro_raw = os.environ.get("AIMIP_EXTRA_RO_BINDS", "").strip()
    if extra_ro_raw:
        for token in re.split(r"[\s:]+", extra_ro_raw):
            if token:
                _append_candidate_run_root(ro_roots, token, forbid_exact=("/",))

    try:
        from .site_config import load_site_config, resolve_site_config_path

        cfg_path, _ = resolve_site_config_path(None, require_target=False)
        if cfg_path is not None and os.path.isfile(cfg_path):
            site_cfg = load_site_config(cfg_path)
            for bind_root in getattr(site_cfg, "extra_ro_binds", ()):
                _append_candidate_run_root(ro_roots, bind_root, forbid_exact=("/",))
    except Exception:
        pass

    return tuple(ro_roots)


def get_visible_input_roots() -> tuple[str, ...]:
    """Return roots visible at job (`exec-fme`) time.

    Allowed input roots are the union of writable run roots (`AIMIP_DEFAULT_OUT_ROOT`,
    `AIMIP_RUN_ROOTS`, `site.json` run roots) and read-only container binds (`AIMIP_HOME`,
    `AIMIP_PLAYGROUND`, `AIMIP_DATA`, `AIMIP_EXTRA_RO_BINDS`).
    """
    combined: list[str] = list(get_allowed_run_roots())
    for ro in get_readonly_mount_roots():
        if ro not in combined:
            combined.append(ro)
    return tuple(combined)


@dataclasses.dataclass(frozen=True)
class _ResolvedRoot:
    """Configured root paired with its physical (`realpath`) directory prefix."""

    display_root: str
    real_root: str
    disallow_exact: bool


def _has_symlink_loop(path: str) -> str | None:
    """Return the deepest component triggering `ELOOP` along *path*, or `None` if loop-free."""
    probe = path
    while probe and probe != "/":
        if os.path.lexists(probe):
            try:
                os.stat(probe)
            except OSError as exc:
                if exc.errno == errno.ELOOP:
                    return probe
            return None
        parent = os.path.dirname(probe)
        if parent == probe:
            break
        probe = parent
    return None


def _resolve_configured_roots(
    roots: Sequence[str],
    *,
    check_forbidden_prefixes: bool,
) -> tuple[_ResolvedRoot, ...]:
    """Resolve every configured root via `os.path.realpath` while preserving root aliases (e.g. `/var`)."""
    resolved: list[_ResolvedRoot] = []
    for root in roots:
        stripped = root.rstrip("/") or "/"
        if stripped == "/" or _has_symlink_loop(stripped) is not None:
            continue
        real_raw = os.path.realpath(stripped)
        if not real_raw.startswith("/") or real_raw == "/":
            continue
        real_with_slash = real_raw.rstrip("/") + "/"
        if check_forbidden_prefixes:
            if real_with_slash in _FORBIDDEN_EXACT_RUN_ROOTS:
                continue
            if any(
                real_with_slash == bad or real_with_slash.startswith(bad)
                for bad in _FORBIDDEN_RUN_ROOT_PREFIXES
            ):
                continue
        resolved.append(
            _ResolvedRoot(
                display_root=root,
                real_root=real_with_slash,
                disallow_exact=root in ("/scratch/", "/work/"),
            )
        )
    return tuple(resolved)


def _resolve_candidate_realpath(normalized: str, *, label: str) -> str:
    """Resolve existing ancestor/leaf symlinks for *normalized* (even if leaf does not yet exist)."""
    loop_at = _has_symlink_loop(normalized)
    if loop_at is not None:
        raise ValidationError(
            f"Invalid {label} {normalized!r}: symlink loop detected at {loop_at!r}."
        )
    real_candidate = os.path.realpath(normalized)
    if not real_candidate.startswith("/") or real_candidate == "/":
        raise ValidationError(
            f"Invalid {label} {normalized!r}: path resolves to filesystem root '/'."
        )
    return real_candidate


def validate_run_path(value: str, label: str = "run directory") -> str:
    """Validate a writable output/run path and require its resolved path to reside under a writable run root."""
    normalized = validate_safe_path(value, label=label)
    allowed_roots = get_allowed_run_roots()
    ro_roots = get_readonly_mount_roots()

    if not allowed_roots:
        raise ValidationError(
            f"Invalid {label} {normalized!r}: no writable run root is configured "
            "(set AIMIP_DEFAULT_OUT_ROOT, AIMIP_RUN_ROOTS, or site.json run_root)."
        )

    # Resolve the entire existing ancestor chain (including when the output leaf
    # does not yet exist) and enforce writable vs read-only precedence on physical paths so an
    # intermediate or leaf symlink cannot escape outside writable roots or into a `:ro` mount.
    resolved_allowed = _resolve_configured_roots(
        allowed_roots, check_forbidden_prefixes=True
    )
    resolved_ro = _resolve_configured_roots(
        ro_roots, check_forbidden_prefixes=False
    )
    real_candidate = _resolve_candidate_realpath(normalized, label=label)

    matching_writable = [
        entry
        for entry in resolved_allowed
        if _path_under_root(
            real_candidate,
            entry.real_root,
            disallow_exact=entry.disallow_exact,
        )
    ]
    matching_ro = [
        entry
        for entry in resolved_ro
        if _path_under_root(real_candidate, entry.real_root, disallow_exact=False)
    ]

    if not matching_writable:
        roots_shown = " or ".join(r.rstrip("/") for r in allowed_roots)
        if matching_ro:
            ro_shown = ", ".join(
                dict.fromkeys(e.display_root.rstrip("/") for e in matching_ro)
            )
            raise ValidationError(
                f"Invalid {label} {normalized!r}: {ro_shown} is mounted read-only inside the container "
                f"(AIMIP_HOME / AIMIP_DATA / AIMIP_EXTRA_RO_BINDS); writable run/output paths must live under {roots_shown}."
            )
        lexical_matched = any(
            _path_under_root(
                normalized,
                r,
                disallow_exact=r in ("/scratch/", "/work/"),
            )
            for r in allowed_roots
        )
        if lexical_matched and real_candidate != normalized:
            raise ValidationError(
                f"Invalid {label} {normalized!r}: symlink resolves to {real_candidate!r} outside writable roots; "
                f"run/output paths must live under {roots_shown}."
            )
        raise ValidationError(
            f"Invalid {label} {normalized!r}: run/output paths must live under {roots_shown}."
        )

    if matching_ro:
        best_w = max(matching_writable, key=lambda e: len(e.real_root))
        best_r = max(matching_ro, key=lambda e: len(e.real_root))
        if len(best_r.real_root) >= len(best_w.real_root):
            writable_non_ro = [
                e.display_root.rstrip("/")
                for e in resolved_allowed
                if not any(
                    e.real_root.startswith(ro.real_root)
                    and len(ro.real_root) >= len(e.real_root)
                    for ro in resolved_ro
                )
            ] or [r.rstrip("/") for r in allowed_roots]
            roots_shown = " or ".join(dict.fromkeys(writable_non_ro))
            raise ValidationError(
                f"Invalid {label} {normalized!r}: {best_r.display_root.rstrip('/')} is a read-only installation/data/bind root; "
                f"writable run/output paths must live under {roots_shown}."
            )

    return normalized


def validate_visible_input_path(value: str, label: str = "input path") -> str:
    """Validate an input file path (`--ic`, `--lora`, `lora scale` input) for job-time visibility.

    Refuses paths whose resolved ancestor/target location lies outside the roots visible during
    `exec-fme`: caller run root (`AIMIP_DEFAULT_OUT_ROOT` / `AIMIP_RUN_ROOTS`), `AIMIP_HOME`,
    `AIMIP_DATA`, and each root in `AIMIP_EXTRA_RO_BINDS`.
    """
    normalized = validate_safe_path(value, label=label)
    visible_roots = get_visible_input_roots()
    if not visible_roots:
        raise ValidationError(
            f"Invalid {label} {normalized!r}: no container-visible roots are configured "
            "(expected AIMIP_DEFAULT_OUT_ROOT, AIMIP_HOME, AIMIP_DATA, or AIMIP_EXTRA_RO_BINDS)."
        )

    resolved_visible = _resolve_configured_roots(
        visible_roots, check_forbidden_prefixes=False
    )
    real_candidate = _resolve_candidate_realpath(normalized, label=label)

    matched = any(
        _path_under_root(
            real_candidate,
            entry.real_root,
            disallow_exact=entry.disallow_exact,
        )
        for entry in resolved_visible
    )
    if not matched:
        roots_shown = ", ".join(r.rstrip("/") for r in visible_roots)
        lexical_matched = any(
            _path_under_root(
                normalized,
                r,
                disallow_exact=r in ("/scratch/", "/work/"),
            )
            for r in visible_roots
        )
        if lexical_matched and real_candidate != normalized:
            raise ValidationError(
                f"Invalid {label} {normalized!r}: symlink resolves to {real_candidate!r} outside container-visible roots "
                f"(AIMIP_DEFAULT_OUT_ROOT, AIMIP_HOME, AIMIP_DATA, AIMIP_EXTRA_RO_BINDS: {roots_shown})."
            )
        raise ValidationError(
            f"Invalid {label} {normalized!r}: path will not be visible inside the container at job time. "
            f"Allowed roots (AIMIP_DEFAULT_OUT_ROOT, AIMIP_HOME, AIMIP_DATA, AIMIP_EXTRA_RO_BINDS): {roots_shown}."
        )

    return normalized


def strip_group_other_write(path: str | os.PathLike[str]) -> None:
    """Remove group-write and world-write bits (`chmod go-w`) without widening stricter modes like `0700`/`0600`.

    Under `umask 002` or `022`, directories become `0755` and files become `0644` (matching
    `trusted_dir` in `container/aimip-wrapper.in` and `aimip.submit`); under `umask 077`, `0700` / `0600` is preserved.
    """
    p = os.fspath(path)
    st = os.lstat(p)
    if stat.S_ISLNK(st.st_mode):
        raise ValidationError(f"Refusing to modify permissions on symlink {p!r}.")
    current = stat.S_IMODE(st.st_mode)
    hardened = current & ~(stat.S_IWGRP | stat.S_IWOTH)
    if hardened != current:
        os.chmod(p, hardened)


def makedirs_no_group_other_write(target_dir: str) -> None:
    """Create *target_dir* (and any missing parents) and strip `g+w`/`o+w` on all created directories."""
    norm = os.path.normpath(target_dir)
    missing_ancestors: list[str] = []
    probe = norm
    while probe and probe != "/" and not os.path.lexists(probe):
        missing_ancestors.append(probe)
        parent = os.path.dirname(probe)
        if parent == probe:
            break
        probe = parent

    os.makedirs(norm, exist_ok=True)
    for created_dir in reversed(missing_ancestors):
        strip_group_other_write(created_dir)
    strip_group_other_write(norm)


def validate_model_name(name: str, allowed: set[str] | None = None) -> str:
    """Validate model identifier format and optional membership."""
    if not isinstance(name, str) or not name:
        raise ValidationError("Identifier name must be a non-empty string.")
    _reject_control_chars(name, "identifier")
    if not _MODEL_NAME_RE.match(name):
        raise ValidationError(
            f"Invalid identifier {name!r}: expected lowercase alphanumeric, '.', '_', or '-'."
        )
    if allowed is not None and name not in allowed:
        known = ", ".join(sorted(allowed)) if allowed else "(none)"
        raise ValidationError(
            f"Unknown model {name!r}. Available models in registry: {known}."
        )
    return name


def validate_forcing_set_name(name: str) -> str:
    """Validate forcing-set / scenario identifier (e.g. 'amip-hist', '1xCO2', '3xCO2')."""
    if not isinstance(name, str) or not name:
        raise ValidationError("Forcing set name must be a non-empty string.")
    _reject_control_chars(name, "forcing_set")
    if not _FORCING_SET_NAME_RE.match(name):
        raise ValidationError(
            f"Invalid forcing set name {name!r}: expected alphanumeric, '.', '_', or '-'."
        )
    return name


def validate_python_module(module_name: str, label: str = "fme_module") -> str:
    """Validate a dotted Python module path such as 'fme.ace.inference'."""
    if not isinstance(module_name, str) or not _MODULE_NAME_RE.match(module_name):
        raise ValidationError(
            f"Invalid {label} {module_name!r}: expected dotted Python module name."
        )
    return module_name


def validate_iso_timestamp(value: str, label: str = "timestamp") -> str:
    """Validate an ISO timestamp ('YYYY-MM-DDTHH:MM:SS' or 'YYYY-MM-DD') and normalize to '%Y-%m-%dT%H:%M:%S'."""
    if not isinstance(value, str) or not value:
        raise ValidationError(f"Invalid {label}: must be a non-empty ISO timestamp.")
    _reject_control_chars(value, label)
    match = _ISO_DATETIME_RE.match(value)
    if not match:
        raise ValidationError(
            f"Invalid {label} {value!r}: expected 'YYYY-MM-DDTHH:MM:SS' (or 'YYYY-MM-DD')."
        )
    year = int(match.group("year"))
    month = int(match.group("month"))
    day = int(match.group("day"))
    hour = int(match.group("hour") or "00")
    minute = int(match.group("minute") or "00")
    second = int(match.group("second") or "00")

    if year < 1:
        raise ValidationError(f"Invalid {label} {value!r}: year must be >= 1.")
    if not (1 <= month <= 12):
        raise ValidationError(f"Invalid {label} {value!r}: month {month} out of range 1..12.")
    if not (1 <= day <= 31):
        raise ValidationError(f"Invalid {label} {value!r}: day {day} out of range 1..31.")
    if month in (4, 6, 9, 11) and day > 30:
        raise ValidationError(f"Invalid {label} {value!r}: month {month:02d} has at most 30 days.")
    if month == 2 and day > 30:
        raise ValidationError(f"Invalid {label} {value!r}: February cannot have {day} days.")
    if not (0 <= hour <= 23):
        raise ValidationError(f"Invalid {label} {value!r}: hour {hour} out of range 0..23.")
    if not (0 <= minute <= 59):
        raise ValidationError(f"Invalid {label} {value!r}: minute {minute} out of range 0..59.")
    if not (0 <= second <= 59):
        raise ValidationError(f"Invalid {label} {value!r}: second {second} out of range 0..59.")

    return f"{year:04d}-{month:02d}-{day:02d}T{hour:02d}:{minute:02d}:{second:02d}"


def validate_int_range(
    value: int,
    label: str,
    *,
    min_value: int = 1,
    max_value: int = 10_000_000,
) -> int:
    """Validate that *value* is an integer (not bool) within [min_value, max_value]."""
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValidationError(f"Invalid {label} {value!r}: expected integer.")
    if value < min_value or value > max_value:
        raise ValidationError(
            f"Invalid {label}={value}: must be in range [{min_value}, {max_value}]."
        )
    return value


def validate_seed(seed: int) -> int:
    """Validate RNG seed compatibility with fme.core.set_seed."""
    return validate_int_range(seed, "seed", min_value=0, max_value=MAX_SEED)


def validate_finite_float(value: float, label: str = "float") -> float:
    """Validate that *value* is a finite float/int (not bool, NaN, or Inf)."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValidationError(f"Invalid {label} {value!r}: expected number.")
    fval = float(value)
    if not math.isfinite(fval):
        raise ValidationError(f"Invalid {label} {value!r}: value must be finite.")
    return fval


def validate_slurm_token(value: str, label: str) -> str:
    """Validate SLURM token options such as partition, account, job-name, or gres."""
    if not isinstance(value, str) or not value:
        raise ValidationError(f"Invalid {label}: must be a non-empty string.")
    _reject_control_chars(value, label)
    if not _SLURM_TOKEN_RE.match(value):
        raise ValidationError(
            f"Invalid {label} {value!r}: only [A-Za-z0-9_.:+-] characters are allowed."
        )
    return value


def validate_slurm_mem(value: str) -> str:
    """Validate SLURM --mem specification such as '64G' or '65536M'."""
    if not isinstance(value, str) or not _SLURM_MEM_RE.match(value):
        raise ValidationError(
            f"Invalid SLURM --mem {value!r}: expected format like '64G' or '65536M'."
        )
    return value


def validate_slurm_time(value: str) -> str:
    """Validate SLURM --time specification ('HH:MM:SS', 'D-HH:MM:SS', or minutes)."""
    if not isinstance(value, str) or not _SLURM_TIME_RE.match(value):
        raise ValidationError(
            f"Invalid SLURM --time {value!r}: expected 'HH:MM:SS' (e.g. '02:00:00')."
        )
    return value


def validate_slurm_job_id(value: str) -> str:
    """Validate a SLURM job ID before passing it to `sacct -j`."""
    if not isinstance(value, str) or not _SLURM_JOB_ID_RE.match(value):
        raise ValidationError(f"Invalid SLURM job ID {value!r}.")
    return value


def validate_sha256_or_empty(value: str | None, label: str = "sha256") -> str:
    """Validate optional 64-hex-character SHA256 string."""
    if value is None or value == "":
        return ""
    if not isinstance(value, str) or not _SHA256_RE.match(value):
        raise ValidationError(
            f"Invalid {label} {value!r}: expected 64 hexadecimal characters."
        )
    return value.lower()
