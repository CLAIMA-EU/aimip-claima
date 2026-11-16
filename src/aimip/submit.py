"""Host-side `aimip submit` gate and run-directory user/permission checks.

Implements the exact same path, UID, ancestry, private-copy, and `job.sbatch`
validation rules enforced by `container/aimip-wrapper.in`, so native installs and
direct `aimip submit <run-dir>` calls have identical submission guarantees.
"""

from __future__ import annotations

import os
import pwd
import re
import shutil
import stat
import subprocess
import tempfile
from typing import Any

from .errors import ConfigError, ValidationError
from .site_config import load_site_config

_WRAPPER_SAFE_PATH_RE = re.compile(r"\A/[A-Za-z0-9._/+-]*\Z")
_SBATCH_VAL_RE = re.compile(r"\A[A-Za-z0-9._/:%+-]+\Z")

ALLOWED_SBATCH_KEYS: frozenset[str] = frozenset(
    {
        "job-name",
        "partition",
        "account",
        "gres",
        "cpus-per-task",
        "mem",
        "time",
        "output",
        "error",
    }
)

TRUSTED_RUN_FILES: tuple[str, ...] = (
    "job.sbatch",
    "config.yaml",
    "aimip-manifest.json",
    "spin-manifest.json",
    "slurm.jobid",
    "sacct.txt",
    "local.pid",
    "local-run.log",
    "ic_members.nc",
    "ic_members_atmosphere.nc",
    "ic_members_ocean.nc",
)


def check_safe_path(path: str) -> str:
    """Validate *path* against the wrapper's strict absolute path allowlist `[A-Za-z0-9._/+-]`."""
    if not isinstance(path, str) or not path.startswith("/"):
        raise ValidationError(f"path must be absolute: {path}")
    if ".." in path or not _WRAPPER_SAFE_PATH_RE.match(path):
        raise ValidationError(f"refusing path: {path}")
    return path


def _uid_name(uid: int) -> str:
    try:
        return pwd.getpwuid(uid).pw_name
    except KeyError:
        return str(uid)


def check_ancestry(path: str, *, caller_uid: int | None = None) -> None:
    """Verify no parent directory above *path* allows unprivileged rename/replacement."""
    euid = os.geteuid() if caller_uid is None else caller_uid
    cur = path
    while True:
        parent = os.path.dirname(cur)
        if not parent or parent == cur or parent == "/":
            break
        cur = parent
        try:
            ast = os.lstat(cur)
        except OSError as exc:
            raise ValidationError(f"cannot inspect parent directory {cur}: {exc}") from exc

        if stat.S_ISLNK(ast.st_mode):
            # Root-owned system links (e.g. /var -> /private/var) and caller-owned links are allowed;
            # physical target ancestry is checked separately after realpath resolution.
            if ast.st_uid not in (0, euid):
                uname = _uid_name(ast.st_uid)
                raise ValidationError(
                    f"a symlink owned by {uname} is in the run directory path: {cur}"
                )
            continue

        if (ast.st_mode & (stat.S_IWGRP | stat.S_IWOTH)) and not (
            ast.st_mode & stat.S_ISVTX
        ):
            mode_str = stat.filemode(ast.st_mode)
            raise ValidationError(
                f"a parent directory is writable by others without sticky bit ({mode_str}): {cur}"
            )


def check_trusted_run_dir(run_dir: str) -> str:
    """Verify *run_dir* and its control files are caller-owned, non-symlinked, and non-group/world-writable."""
    check_safe_path(run_dir)
    d = run_dir.rstrip("/") or "/"
    euid = os.geteuid()

    try:
        st = os.lstat(d)
    except FileNotFoundError as exc:
        raise ValidationError(f"not a directory: {d}") from exc
    except OSError as exc:
        raise ValidationError(f"cannot inspect {d}: {exc}") from exc

    if stat.S_ISLNK(st.st_mode):
        raise ValidationError(f"refusing a symlinked run directory: {d}")
    if not stat.S_ISDIR(st.st_mode):
        raise ValidationError(f"not a directory: {d}")
    if st.st_uid != euid:
        raise ValidationError(f"run directory is not owned by you: {d}")
    if st.st_mode & (stat.S_IWGRP | stat.S_IWOTH):
        mode_str = stat.filemode(st.st_mode)
        raise ValidationError(
            f"run directory is group- or world-writable ({mode_str}): {d} — fix with: chmod go-w {d}"
        )

    for fname in TRUSTED_RUN_FILES:
        fpath = f"{d}/{fname}"
        try:
            fst = os.lstat(fpath)
        except FileNotFoundError:
            continue
        except OSError as exc:
            raise ValidationError(f"cannot inspect {fpath}: {exc}") from exc

        if stat.S_ISLNK(fst.st_mode):
            raise ValidationError(f"refusing symlinked {fpath}")
        if fst.st_uid != euid:
            raise ValidationError(f"{fpath} is not owned by you")
        if fst.st_mode & (stat.S_IWGRP | stat.S_IWOTH):
            mode_str = stat.filemode(fst.st_mode)
            raise ValidationError(
                f"{fpath} is group- or world-writable ({mode_str}) — fix with: chmod go-w {fpath}"
            )

    check_ancestry(d, caller_uid=euid)
    try:
        phys = os.path.realpath(d)
    except OSError as exc:
        raise ValidationError(f"cannot resolve {d}: {exc}") from exc
    if not os.path.isdir(phys):
        raise ValidationError(f"cannot resolve {d}")
    if phys != d:
        check_ancestry(phys, caller_uid=euid)

    return d


def is_trusted_run_dir(run_dir: str) -> bool:
    """Return True when *run_dir* passes `check_trusted_run_dir` without raising."""
    try:
        check_trusted_run_dir(run_dir)
        return True
    except (ValidationError, ConfigError):
        return False


def write_in_trusted_dir(trusted_dir: str, name: str, content: str) -> str:
    """Atomically write *content* (with trailing newline) to `<trusted_dir>/<name>`, never following symlinks."""
    target = os.path.join(trusted_dir, name)
    if os.path.islink(target):
        raise ValidationError(f"refusing symlinked {target}")
    fd, tmp_path = tempfile.mkstemp(prefix=f".{name}.", dir=trusted_dir)
    try:
        payload = content if content.endswith("\n") else f"{content}\n"
        os.write(fd, payload.encode("utf-8"))
        os.fchmod(fd, 0o644)
        os.close(fd)
        fd = -1
        os.replace(tmp_path, target)
    except BaseException:
        if fd >= 0:
            try:
                os.close(fd)
            except OSError:
                pass
        if os.path.exists(tmp_path):
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
        raise
    return target


def check_job_script(script_path: str, run_dir: str, prefix: str) -> None:
    """Validate *script_path* against the exact `aimip-generated v1` job script grammar."""
    d = run_dir.rstrip("/") or "/"
    norm_prefix = check_safe_path(prefix).rstrip("/")
    want_exec = f'"{norm_prefix}/bin/aimip" exec-fme "'

    try:
        with open(script_path, "rb") as fh:
            raw_bytes = fh.read()
    except OSError as exc:
        raise ConfigError(f"cannot read {script_path}: {exc}") from exc

    if b"\x00" in raw_bytes:
        raise ValidationError(f"{script_path} contains NUL bytes")
    try:
        text = raw_bytes.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ValidationError(f"{script_path} is not valid UTF-8") from exc

    # Match bash `while IFS= read -r line || [ -n "$line" ]` strictly on '\n' boundaries
    # so CR/FF/Unicode line breaks are never silently normalized away.
    if not text:
        lines: list[str] = []
    elif text.endswith("\n"):
        lines = text[:-1].split("\n")
    else:
        lines = text.split("\n")

    execs = 0
    for idx, line in enumerate(lines, start=1):
        if idx == 1:
            if line != "#!/bin/bash":
                raise ValidationError("line 1 must be #!/bin/bash")
            continue
        if idx == 2:
            if line != "# aimip-generated v1":
                raise ValidationError(f"{script_path} is not an aimip-generated script")
            continue

        if line in ("", "set -euo pipefail", "#SBATCH --hint=nomultithread"):
            continue

        if line.startswith("#SBATCH --"):
            rest = line[len("#SBATCH --") :]
            if "=" not in rest:
                raise ValidationError(f"unexpected #SBATCH line: {line}")
            key, val = rest.split("=", 1)
            if key not in ALLOWED_SBATCH_KEYS:
                raise ValidationError(f"unexpected #SBATCH line: {line}")
            if not val or ".." in val or not _SBATCH_VAL_RE.match(val):
                raise ValidationError(f"unexpected #SBATCH value: {line}")
            if key in ("output", "error"):
                if not val.startswith(f"{d}/"):
                    raise ValidationError(f"slurm log outside the run directory: {line}")
                log_rel = val[len(d) + 1 :]
                if not log_rel or log_rel == "." or "/" in log_rel:
                    raise ValidationError(
                        f"slurm log must be a file directly in the run directory: {line}"
                    )
                log_dst = f"{d}/{log_rel}"
                if os.path.islink(log_dst):
                    raise ValidationError(f"refusing symlinked slurm log: {log_dst}")
                if os.path.isdir(log_dst):
                    raise ValidationError(f"slurm log destination is a directory: {log_dst}")
                if "%" in log_rel:
                    parts = log_rel.split("%j")
                    if len(parts) != 2 or "%" in parts[0] or "%" in parts[1]:
                        raise ValidationError(f"unsupported slurm log substitution: {line}")
                    pfx, sfx = parts
                    try:
                        entries = os.listdir(d)
                    except OSError as exc:
                        raise ValidationError(f"cannot list run directory {d}: {exc}") from exc
                    for entry in entries:
                        if (
                            len(entry) > len(pfx) + len(sfx)
                            and entry.startswith(pfx)
                            and (not sfx or entry.endswith(sfx))
                        ):
                            mid = entry[len(pfx) : len(entry) - len(sfx) if sfx else len(entry)]
                            if mid.isascii() and mid.isdigit():
                                cand = f"{d}/{entry}"
                                if os.path.islink(cand):
                                    raise ValidationError(f"refusing symlinked slurm log: {cand}")
                                if os.path.isdir(cand):
                                    raise ValidationError(
                                        f"slurm log destination is a directory: {cand}"
                                    )
            continue

        if line.startswith(want_exec):
            rest = line[len(want_exec) :]
            if not (rest.startswith("/") and rest.endswith('"')):
                raise ValidationError(f"unexpected exec line: {line}")
            target = rest[:-1]
            check_safe_path(target)
            if target != d and not target.startswith(f"{d}/"):
                raise ValidationError(f"exec target outside the run directory: {target}")
            execs += 1
            continue

        raise ValidationError(f"{script_path} has an unexpected line: {line}")

    if execs < 1:
        raise ValidationError(f"{script_path} has no exec-fme line")


def _resolve_submit_prefix(playground_dir: str | None = None) -> str:
    raw = (
        playground_dir
        or os.environ.get("AIMIP_HOME")
        or os.environ.get("AIMIP_PLAYGROUND")
    )
    if not raw:
        raise ConfigError(
            "Cannot determine install prefix for `aimip submit`: set $AIMIP_HOME "
            "(or invoke `<prefix>/bin/aimip submit <run-dir>`)."
        )
    return check_safe_path(raw).rstrip("/")


def submit_run(
    run_dir: str,
    *,
    playground_dir: str | None = None,
    site_config_path: str | None = None,
) -> dict[str, Any]:
    """Validate `<run_dir>/job.sbatch` in a caller-private 0700 copy and submit via `sbatch` or detached `bash`."""
    if not run_dir:
        raise ConfigError("usage: aimip submit <run-dir>")
    check_safe_path(run_dir)
    d = check_trusted_run_dir(run_dir)

    sbatch_src = f"{d}/job.sbatch"
    if os.path.islink(sbatch_src) or not os.path.isfile(sbatch_src):
        raise ConfigError(
            f"{d} has no job.sbatch (prepare it with: aimip run ... --out {d})"
        )
    if os.path.lexists(f"{d}/slurm.jobid") or os.path.lexists(f"{d}/local.pid"):
        raise ConfigError("already submitted")

    prefix = _resolve_submit_prefix(playground_dir)
    site_cfg = load_site_config(site_config_path)
    scheduler = site_cfg.scheduler

    try:
        priv_dir = tempfile.mkdtemp(prefix="aimip-submit.")
        os.chmod(priv_dir, 0o700)
    except OSError as exc:
        raise ConfigError(f"cannot create a private directory: {exc}") from exc

    keep_priv_dir = False
    try:
        priv_script = os.path.join(priv_dir, "job.sbatch")
        read_flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
        try:
            src_fd = os.open(sbatch_src, read_flags)
            try:
                with os.fdopen(src_fd, "rb") as rfh:
                    script_bytes = rfh.read()
            except BaseException:
                try:
                    os.close(src_fd)
                except OSError:
                    pass
                raise
            dst_fd = os.open(
                priv_script,
                os.O_WRONLY | os.O_CREAT | os.O_EXCL,
                0o400,
            )
            try:
                os.write(dst_fd, script_bytes)
            finally:
                os.close(dst_fd)
        except OSError as exc:
            raise ConfigError(f"cannot read {sbatch_src}: {exc}") from exc

        check_job_script(priv_script, d, prefix)

        if scheduler == "slurm":
            try:
                proc = subprocess.run(
                    ["sbatch", "--parsable", priv_script],
                    capture_output=True,
                    text=True,
                    check=False,
                )
            except FileNotFoundError as exc:
                raise ConfigError("sbatch not found on PATH") from exc
            except OSError as exc:
                raise ConfigError(f"sbatch failed: {exc}") from exc

            if proc.returncode != 0:
                detail = (proc.stderr or proc.stdout or "").strip()
                raise ConfigError(f"sbatch failed: {detail}" if detail else "sbatch failed")

            first_line = (
                proc.stdout.strip().splitlines()[0] if proc.stdout.strip() else ""
            )
            job_id = first_line.split(";", 1)[0].strip()
            if not job_id:
                raise ConfigError("sbatch failed: empty job ID")

            write_in_trusted_dir(d, "slurm.jobid", job_id)
            return {
                "out_dir": d,
                "scheduler": "slurm",
                "slurm_job_id": job_id,
                "local_pid": None,
                "log_path": None,
            }

        # scheduler == "none": run the validated private copy in a detached background bash session.
        log_path = f"{d}/local-run.log"
        if os.path.islink(log_path):
            raise ValidationError(f"refusing symlinked {log_path}")
        log_flags = (
            os.O_WRONLY
            | os.O_CREAT
            | os.O_TRUNC
            | getattr(os, "O_NOFOLLOW", 0)
        )
        try:
            log_fd = os.open(log_path, log_flags, 0o644)
        except OSError as exc:
            raise ConfigError(f"cannot open {log_path}: {exc}") from exc

        with os.fdopen(log_fd, "wb") as log_fh:
            try:
                bg_proc = subprocess.Popen(
                    ["bash", priv_script],
                    stdin=subprocess.DEVNULL,
                    stdout=log_fh,
                    stderr=subprocess.STDOUT,
                    start_new_session=True,
                    close_fds=True,
                )
            except OSError as exc:
                raise ConfigError(f"failed to launch background bash job: {exc}") from exc

        keep_priv_dir = True
        write_in_trusted_dir(d, "local.pid", str(bg_proc.pid))
        return {
            "out_dir": d,
            "scheduler": "none",
            "slurm_job_id": None,
            "local_pid": int(bg_proc.pid),
            "log_path": log_path,
        }
    finally:
        if not keep_priv_dir:
            shutil.rmtree(priv_dir, ignore_errors=True)
