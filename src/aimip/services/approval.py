"""Human-in-the-loop `/dev/tty` approval prompt, in-memory single-use broker, and `$AIMIP_STATE/approvals/` store."""

from __future__ import annotations

import getpass
import json
import os
from pathlib import Path
import re
import secrets
import stat
import time
from typing import Any

from ..errors import AimipError, ApprovalError, ConfigError, ValidationError
from ..registry import resolve_data_dir
from ..site_config import expand_run_root_template, load_site_config, resolve_site_config_path
from ..submit import check_ancestry
from ..validators import validate_safe_path
from .types import Authorization

APPROVAL_TTL_SECONDS = 600.0
_HEX_PREFIX_RE = re.compile(r"\A[0-9a-fA-F]{8,64}\Z")
_HEX64_RE = re.compile(r"\A[0-9a-f]{64}\Z")


def current_user() -> str:
    """Resolve current POSIX username consistently across CLI, chat, and MCP."""
    for env_key in ("AIMIP_USER", "USER", "LOGNAME"):
        val = os.environ.get(env_key, "").strip()
        if val:
            return val
    try:
        return getpass.getuser() or str(os.geteuid())
    except Exception:
        return str(os.geteuid())


def format_bytes(num_bytes: int | None) -> str:
    """Format byte counts clearly for download prompts and plan cards."""
    if num_bytes is None or num_bytes <= 0:
        return "0 B"
    val = float(num_bytes)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if val < 1024.0 or unit == "TiB":
            return f"{int(val)} B" if unit == "B" else f"{val:.2f} {unit}"
        val /= 1024.0
    return f"{num_bytes} B"


def resolve_state_dir(
    state_dir: str | None = None,
    *,
    site_config_path: str | None = None,
    fallback_run_dir: str | None = None,
) -> Path:
    """Resolve `$AIMIP_STATE` (defaults to `<run_root>/.aimip-state`)."""
    raw = state_dir or os.environ.get("AIMIP_STATE")
    if not raw:
        env_root = os.environ.get("AIMIP_DEFAULT_OUT_ROOT")
        if env_root:
            base = validate_safe_path(env_root, label="AIMIP_DEFAULT_OUT_ROOT")
        else:
            site_path, is_explicit = resolve_site_config_path(
                site_config_path, require_target=False
            )
            has_configured_site = bool(
                is_explicit or (site_path and os.path.isfile(site_path))
            )
            cfg = load_site_config(site_config_path)
            expanded = expand_run_root_template(cfg.run_root)
            if not has_configured_site and fallback_run_dir:
                parent = os.path.dirname(fallback_run_dir.rstrip("/")) or "/"
                if parent != "/":
                    probe = expanded or ""
                    while probe and probe != "/" and not os.path.exists(probe):
                        probe = os.path.dirname(probe) or "/"
                    if not probe or not os.access(probe, os.W_OK):
                        return Path(
                            validate_safe_path(
                                os.path.join(parent, ".aimip-state"),
                                label="AIMIP_STATE",
                            )
                        )
            if not expanded:
                raise ConfigError(
                    f"Cannot expand site.json run_root {cfg.run_root!r}; set $AIMIP_STATE explicitly.",
                    field="AIMIP_STATE",
                )
            base = validate_safe_path(expanded, label="site run_root")
        raw = os.path.join(base, ".aimip-state")
    validated = validate_safe_path(raw, label="AIMIP_STATE")
    return Path(validated)


def _open_verified_private_subdir(base_state: Path, subdir_name: str) -> tuple[Path, int]:
    """Create `<base_state>/<subdir_name>` with mode `0700` and return `(path, dir_fd)` after ancestry/mode checks."""
    base_state.mkdir(parents=True, exist_ok=True, mode=0o700)
    if base_state.is_symlink():
        raise ApprovalError(f"Refusing symlinked state directory: {base_state}")
    os.chmod(base_state, 0o700)

    sub = base_state / subdir_name
    if sub.is_symlink():
        raise ApprovalError(f"Refusing symlinked {subdir_name} directory: {sub}")
    sub.mkdir(parents=True, exist_ok=True, mode=0o700)

    try:
        check_ancestry(str(sub))
        phys = os.path.realpath(str(sub))
        if phys != str(sub):
            check_ancestry(phys)
    except ValidationError as exc:
        raise ApprovalError(f"Untrusted state directory ancestry ({sub}): {exc}") from exc

    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        dfd = os.open(sub, flags)
    except OSError as exc:
        raise ApprovalError(f"Cannot open state directory {sub}: {exc}") from exc

    try:
        info = os.fstat(dfd)
        if not stat.S_ISDIR(info.st_mode):
            raise ApprovalError(f"State path is not a directory: {sub}")
        if info.st_uid != os.geteuid():
            raise ApprovalError(f"State directory {sub} is not owned by the current user")
        if stat.S_IMODE(info.st_mode) & 0o077:
            os.fchmod(dfd, 0o700)
            info = os.fstat(dfd)
            if stat.S_IMODE(info.st_mode) & 0o077:
                raise ApprovalError(f"State directory {sub} must have mode 0700")
        return sub, dfd
    except BaseException:
        os.close(dfd)
        raise


def _write_private_json_in_dir(dfd: int, filename: str, payload: dict[str, Any]) -> None:
    """Atomically write `0600` JSON inside `dfd` without following symlinks."""
    try:
        lst = os.lstat(filename, dir_fd=dfd)
        if stat.S_ISLNK(lst.st_mode):
            raise ApprovalError(f"Refusing symlinked state file: {filename}")
    except FileNotFoundError:
        pass

    tmp_name = f".tmp.{filename}.{secrets.token_hex(8)}"
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0)
    fd = os.open(tmp_name, flags, 0o600, dir_fd=dfd)
    try:
        data = (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode("utf-8")
        view = memoryview(data)
        while view:
            written = os.write(fd, view)
            if written <= 0:
                raise OSError("short write to state file")
            view = view[written:]
        os.fchmod(fd, 0o600)
    except BaseException:
        os.close(fd)
        try:
            os.unlink(tmp_name, dir_fd=dfd)
        except OSError:
            pass
        raise
    os.close(fd)
    os.replace(tmp_name, filename, src_dir_fd=dfd, dst_dir_fd=dfd)


def _read_and_verify_regular_0600(
    dfd: int,
    filename: str,
    *,
    expected_stat: os.stat_result | None = None,
) -> dict[str, Any]:
    """Read a `0600` caller-owned non-symlink JSON file from `dfd`."""
    lst = os.lstat(filename, dir_fd=dfd)
    if stat.S_ISLNK(lst.st_mode):
        raise ApprovalError(f"Refusing symlinked approval/proposal file: {filename}")
    if not stat.S_ISREG(lst.st_mode):
        raise ApprovalError(f"Approval/proposal entry is not a regular file: {filename}")
    if lst.st_uid != os.geteuid():
        raise ApprovalError(f"Approval/proposal file {filename} is not owned by the current user")
    if stat.S_IMODE(lst.st_mode) != 0o600:
        raise ApprovalError(
            f"Approval/proposal file {filename} has insecure mode {oct(stat.S_IMODE(lst.st_mode))} (expected 0o600)"
        )
    if expected_stat is not None and (lst.st_dev, lst.st_ino) != (
        expected_stat.st_dev,
        expected_stat.st_ino,
    ):
        raise ApprovalError(f"Approval file {filename} changed concurrently")

    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    fd = os.open(filename, flags, dir_fd=dfd)
    try:
        fst = os.fstat(fd)
        if (fst.st_dev, fst.st_ino) != (lst.st_dev, lst.st_ino):
            raise ApprovalError(f"Approval file {filename} swapped during open")
        if not stat.S_ISREG(fst.st_mode) or fst.st_uid != os.geteuid() or stat.S_IMODE(fst.st_mode) != 0o600:
            raise ApprovalError(f"Approval file {filename} failed open-descriptor ownership/mode check")
        with os.fdopen(fd, "r", encoding="utf-8") as fh:
            fd = -1
            raw = fh.read(256 * 1024)
    finally:
        if fd >= 0:
            os.close(fd)

    try:
        doc = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ApprovalError(f"Corrupted approval/proposal JSON in {filename}: {exc}") from exc
    if not isinstance(doc, dict):
        raise ApprovalError(f"Malformed approval/proposal JSON in {filename}")
    return doc


def can_download_into_data_dir(
    *,
    data_dir: str | None = None,
    playground_dir: str | None = None,
    model_name: str | None = None,
) -> tuple[bool, str]:
    """Return `(ok, reason)` indicating whether the current session can fetch missing model files into `$AIMIP_DATA`."""
    target_hint = f"aimip fetch -m {model_name}" if model_name else "aimip fetch -m <model>"
    if os.environ.get("AIMIP_DATA_MODE") == "ro":
        return (
            False,
            f"This session has read-only access to $AIMIP_DATA (AIMIP_DATA_MODE=ro); "
            f"ask the install owner to run `{target_hint}` on the host first.",
        )
    try:
        resolved_data = resolve_data_dir(data_dir, playground_dir=playground_dir)
    except ValidationError as exc:
        return (False, str(exc))
    if not resolved_data:
        return (
            False,
            f"$AIMIP_DATA is not configured; set $AIMIP_DATA or run `{target_hint}` on the host first.",
        )
    probe = resolved_data
    while probe != "/" and not os.path.exists(probe):
        probe = os.path.dirname(probe) or "/"
    if not os.access(probe, os.W_OK):
        return (
            False,
            f"Data directory {resolved_data} is not writable by you; "
            f"ask the install owner to run `{target_hint}` on the host first.",
        )
    return (True, "")


def prompt_tty_approval(
    *,
    digest: str,
    card_text: str,
    action: str = "submit",
    missing_bytes: int = 0,
    missing_count: int = 0,
    requires_data_dir_write: bool = True,
    model_name: str | None = None,
    data_dir: str | None = None,
    playground_dir: str | None = None,
    tty_path: str = "/dev/tty",
    now_fn: Any = time.time,
) -> Authorization:
    """Display *card_text* on the controlling TTY and require typing `digest[:8]` (plus download yes/no if needed).

    Refuses immediately if `tty_path` cannot be opened, is a symlink, or is not a real TTY (`os.isatty(fd)`),
    or if missing `$AIMIP_DATA` files are required in a read-only session.
    """
    norm_digest = digest.strip().lower()
    if not _HEX64_RE.match(norm_digest):
        raise ValidationError(f"Invalid plan digest {digest!r}", field="digest")

    if (missing_bytes > 0 or missing_count > 0) and requires_data_dir_write:
        dl_ok, dl_reason = can_download_into_data_dir(
            data_dir=data_dir,
            playground_dir=playground_dir,
            model_name=model_name,
        )
        if not dl_ok:
            raise ApprovalError(
                dl_reason,
                field="download",
                next_command=f"aimip fetch -m {model_name}" if model_name else None,
            )

    open_flags = os.O_RDWR | getattr(os, "O_NOCTTY", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        tty_fd = os.open(tty_path, open_flags)
    except OSError as exc:
        raise ApprovalError(
            f"Human approval requires an interactive controlling terminal ({tty_path} unavailable: {exc}).",
            field="approval",
        ) from exc

    try:
        if not os.isatty(tty_fd):
            raise ApprovalError(
                f"Refusing non-TTY approval channel ({tty_path}); approval must come from an interactive terminal.",
                field="approval",
            )

        r_fd = os.dup(tty_fd)
        r_stream = os.fdopen(r_fd, "r", encoding="utf-8", errors="replace")
        try:
            def _tty_write(text: str) -> None:
                data = text.encode("utf-8")
                view = memoryview(data)
                while view:
                    n = os.write(tty_fd, view)
                    if n <= 0:
                        raise OSError("TTY write failed")
                    view = view[n:]

            prefix8 = norm_digest[:8]
            _tty_write(f"\n{card_text.rstrip()}\n")
            _tty_write(
                f"\nTo approve {action}, type the first 8 hex characters of digest [{prefix8}] (or anything else to abort): "
            )
            typed = r_stream.readline()
            if not typed:
                raise ApprovalError("Approval aborted (EOF on controlling terminal).", field="approval")
            cleaned = typed.strip().lower()
            if cleaned != prefix8:
                raise ApprovalError(
                    f"Approval refused: typed {typed.strip()!r} does not match expected digest prefix {prefix8!r}.",
                    field="approval",
                )

            allow_download = False
            if missing_bytes > 0 or missing_count > 0:
                shown_size = format_bytes(missing_bytes)
                _tty_write(
                    f"Download required: {missing_count} missing asset(s) (~{shown_size}). Allow download now? [yes/no]: "
                )
                dl_ans = r_stream.readline()
                if dl_ans.strip().lower() != "yes":
                    raise ApprovalError(
                        "Download not approved; run `aimip fetch` on the host first or answer 'yes' at the download prompt.",
                        field="download",
                    )
                allow_download = True

            now_ts = float(now_fn())
            return Authorization(
                action=action,
                digest=norm_digest,
                user=current_user(),
                issued_at=now_ts,
                expires_at=now_ts + APPROVAL_TTL_SECONDS,
                nonce=secrets.token_hex(16),
                allow_download=allow_download,
            )
        finally:
            r_stream.close()
    finally:
        os.close(tty_fd)


class MemoryApprovalBroker:
    """In-memory single-use authorization registry for `aimip chat` (never exposed to LLM tools)."""

    def __init__(self, *, now_fn: Any = time.time) -> None:
        self._now_fn = now_fn
        self._pending: dict[str, Any] | None = None
        self._tokens: dict[str, Authorization] = {}

    @property
    def pending(self) -> dict[str, Any] | None:
        return dict(self._pending) if self._pending is not None else None

    def clear_pending(self) -> None:
        self._pending = None

    def revoke_all(self) -> None:
        """Revoke both any unapproved proposal and any unconsumed in-memory `Authorization` tokens."""
        self._pending = None
        self._tokens.clear()

    def burn_token(self, auth: Authorization | None) -> None:
        """Unconditionally burn *auth* so a failed submission attempt cannot reuse a single-use token."""
        if isinstance(auth, Authorization):
            self._tokens.pop(auth.nonce, None)

    def register_proposal(
        self,
        *,
        action: str,
        digest: str,
        card_text: str,
        card_data: dict[str, Any],
        missing_bytes: int = 0,
        missing_count: int = 0,
    ) -> dict[str, Any]:
        self._pending = {
            "action": action,
            "digest": digest.lower(),
            "card_text": card_text,
            "card_data": json.loads(json.dumps(card_data)),
            "missing_bytes": int(missing_bytes),
            "missing_count": int(missing_count),
        }
        return dict(self._pending)

    def approve_pending_on_tty(
        self,
        *,
        tty_path: str = "/dev/tty",
        data_dir: str | None = None,
        playground_dir: str | None = None,
    ) -> tuple[Authorization, dict[str, Any]]:
        if self._pending is None:
            raise ApprovalError("No pending proposal to approve.")
        proposal = self._pending
        self._pending = None
        card_data = proposal.get("card_data") if isinstance(proposal.get("card_data"), dict) else {}
        missing_assets = card_data.get("missing_assets") or []
        req_data_write = any(
            isinstance(a, dict) and not a.get("prospective", False)
            for a in missing_assets
        ) if missing_assets else True
        auth = prompt_tty_approval(
            digest=proposal["digest"],
            card_text=proposal["card_text"],
            action=proposal["action"],
            missing_bytes=proposal["missing_bytes"],
            missing_count=proposal["missing_count"],
            requires_data_dir_write=req_data_write,
            model_name=str(card_data.get("model") or "") or None,
            data_dir=data_dir,
            playground_dir=playground_dir,
            tty_path=tty_path,
            now_fn=self._now_fn,
        )
        self._tokens[auth.nonce] = auth
        return auth, proposal

    def store_minted(self, auth: Authorization) -> None:
        self._tokens[auth.nonce] = auth

    def consume(self, auth: Authorization, *, expected_action: str, expected_digest: str) -> Authorization:
        """Validate and atomically consume a single-use in-memory `Authorization`."""
        if not isinstance(auth, Authorization):
            raise ApprovalError("Missing valid human Authorization object.")
        stored = self._tokens.pop(auth.nonce, None)
        if stored is None or stored != auth:
            raise ApprovalError("Authorization was already used, revoked, or not minted by this host broker.")
        self.verify_authorization_fields(
            stored,
            expected_action=expected_action,
            expected_digest=expected_digest,
            now_ts=float(self._now_fn()),
        )
        return stored

    @staticmethod
    def verify_authorization_fields(
        auth: Authorization,
        *,
        expected_action: str,
        expected_digest: str,
        now_ts: float | None = None,
    ) -> None:
        ts = float(time.time() if now_ts is None else now_ts)
        if auth.action != expected_action:
            raise ApprovalError(
                f"Authorization action mismatch: expected {expected_action!r}, got {auth.action!r}."
            )
        if auth.digest.lower() != expected_digest.lower():
            raise ApprovalError("Plan changed, approve again (digest mismatch).")
        if auth.user != current_user():
            raise ApprovalError(
                f"Authorization user mismatch: issued for {auth.user!r}, current user is {current_user()!r}."
            )
        if ts > auth.expires_at:
            raise ApprovalError("Authorization expired (10-minute limit); approve again.")


def record_state_proposal(
    *,
    action: str,
    digest: str,
    card_text: str,
    payload: dict[str, Any],
    missing_bytes: int = 0,
    missing_count: int = 0,
    requires_data_dir_write: bool = True,
    state_dir: str | None = None,
    now_fn: Any = time.time,
) -> Path:
    """Persist a pending MCP proposal in `$AIMIP_STATE/proposals/<digest>.json` (`0600`)."""
    norm_digest = digest.strip().lower()
    if not _HEX64_RE.match(norm_digest):
        raise ValidationError(f"Invalid digest {digest!r}", field="digest")
    base = resolve_state_dir(state_dir)
    prop_dir, dfd = _open_verified_private_subdir(base, "proposals")
    try:
        doc = {
            "action": action,
            "digest": norm_digest,
            "user": current_user(),
            "created_at": float(now_fn()),
            "card_text": card_text,
            "missing_bytes": int(missing_bytes),
            "missing_count": int(missing_count),
            "requires_data_dir_write": bool(requires_data_dir_write),
            "payload": payload,
        }
        fname = f"{norm_digest}.json"
        _write_private_json_in_dir(dfd, fname, doc)
        return prop_dir / fname
    finally:
        os.close(dfd)


def read_state_proposal(
    digest: str,
    *,
    state_dir: str | None = None,
) -> dict[str, Any]:
    """Read and verify a `0600` caller-owned proposal record from `$AIMIP_STATE/proposals/<digest>.json`."""
    norm_digest = digest.strip().lower()
    if not _HEX64_RE.match(norm_digest):
        raise ValidationError(f"Invalid digest {digest!r}", field="digest")
    base = resolve_state_dir(state_dir)
    _, dfd = _open_verified_private_subdir(base, "proposals")
    try:
        try:
            return _read_and_verify_regular_0600(dfd, f"{norm_digest}.json")
        except FileNotFoundError as exc:
            for entry_name in sorted(os.listdir(dfd)):
                if not entry_name.endswith(".json") or entry_name.startswith("."):
                    continue
                try:
                    cand_doc = _read_and_verify_regular_0600(dfd, entry_name)
                except AimipError:
                    continue
                cand_payload = cand_doc.get("payload") if isinstance(cand_doc.get("payload"), dict) else {}
                if cand_payload.get("previous_prepare_digest") == norm_digest:
                    final_dig = str(cand_doc.get("digest") or "")
                    if _HEX64_RE.match(final_dig):
                        next_cmd = f"aimip approve {final_dig[:8]}"
                        raise ApprovalError(
                            f"Prepare digest {norm_digest[:12]} was already consumed during staging; "
                            f"no human approval found yet for final submit digest {final_dig[:12]} — "
                            f"run `{next_cmd}` in your terminal, then call `submit_approved` with digest={final_dig!r}.",
                            field="digest",
                            next_command=next_cmd,
                        ) from exc
            raise ApprovalError(
                f"No human approval or active proposal found for digest {norm_digest[:12]} "
                "(already consumed or not proposed); call `propose_run` to create a fresh proposal, "
                "then run `aimip approve <prefix>`.",
                field="digest",
                next_command="propose_run",
            ) from exc
    finally:
        os.close(dfd)


def has_state_approval(
    digest: str,
    *,
    state_dir: str | None = None,
) -> bool:
    """Return True if an entry `<digest>.json` exists in `$AIMIP_STATE/approvals/` (even if tampered, so `consume_state_approval` inspects and rejects tamper)."""
    norm_digest = digest.strip().lower()
    if not _HEX64_RE.match(norm_digest):
        return False
    base = resolve_state_dir(state_dir)
    app_dir = base / "approvals"
    if not app_dir.exists() or app_dir.is_symlink():
        return False
    entry = app_dir / f"{norm_digest}.json"
    return os.path.lexists(entry)


def delete_state_proposal(
    digest: str,
    *,
    state_dir: str | None = None,
) -> None:
    """Remove a consumed proposal `<digest>.json` from `$AIMIP_STATE/proposals/`, if present."""
    norm_digest = digest.strip().lower()
    if not _HEX64_RE.match(norm_digest):
        return
    base = resolve_state_dir(state_dir)
    prop_dir = base / "proposals"
    if not prop_dir.exists() or prop_dir.is_symlink():
        return
    _, dfd = _open_verified_private_subdir(base, "proposals")
    try:
        try:
            os.unlink(f"{norm_digest}.json", dir_fd=dfd)
        except OSError:
            pass
    finally:
        os.close(dfd)


def approve_state_proposal(
    digest_prefix: str,
    *,
    state_dir: str | None = None,
    tty_path: str = "/dev/tty",
    now_fn: Any = time.time,
) -> tuple[Authorization, dict[str, Any]]:
    """Interactive `aimip approve <digest-prefix>` (TTY-only): verify human on `/dev/tty` and write `0600` approval."""
    if not isinstance(digest_prefix, str) or not _HEX_PREFIX_RE.match(digest_prefix.strip()):
        raise ValidationError(
            f"Invalid digest prefix {digest_prefix!r}: provide at least the first 8 hex characters of the plan digest.",
            field="digest_prefix",
        )
    prefix = digest_prefix.strip().lower()
    base = resolve_state_dir(state_dir)
    _, prop_dfd = _open_verified_private_subdir(base, "proposals")
    try:
        names = [
            n
            for n in os.listdir(prop_dfd)
            if n.endswith(".json") and not n.startswith(".") and n[:-5].startswith(prefix)
        ]
        if not names:
            for entry_name in sorted(os.listdir(prop_dfd)):
                if not entry_name.endswith(".json") or entry_name.startswith("."):
                    continue
                try:
                    cand_doc = _read_and_verify_regular_0600(prop_dfd, entry_name)
                except AimipError:
                    continue
                cand_payload = cand_doc.get("payload") if isinstance(cand_doc.get("payload"), dict) else {}
                prev_prep = str(cand_payload.get("previous_prepare_digest") or "")
                if prev_prep.startswith(prefix):
                    final_dig = str(cand_doc.get("digest") or "")
                    if _HEX64_RE.match(final_dig):
                        next_cmd = f"aimip approve {final_dig[:8]}"
                        raise ApprovalError(
                            f"Prepare proposal matching prefix {prefix!r} was already consumed during staging; "
                            f"run `{next_cmd}` to approve the final submit card.",
                            field="digest_prefix",
                            next_command=next_cmd,
                        )
            raise ApprovalError(
                f"No pending proposal matches prefix {prefix!r} (already consumed or not proposed); "
                "call `propose_run` first, then run `aimip approve <prefix>`.",
                field="digest_prefix",
                next_command="propose_run",
            )
        if len(names) > 1:
            raise ApprovalError(
                f"Digest prefix {prefix!r} is ambiguous ({len(names)} proposals match); pass more hex characters.",
                field="digest_prefix",
            )
        proposal = _read_and_verify_regular_0600(prop_dfd, names[0])
    finally:
        os.close(prop_dfd)

    full_digest = str(proposal.get("digest") or "")
    if not full_digest.startswith(prefix):
        raise ApprovalError("Proposal digest mismatch inside state file.")

    prop_payload = proposal.get("payload") if isinstance(proposal.get("payload"), dict) else {}
    prop_spec = prop_payload.get("spec") if isinstance(prop_payload.get("spec"), dict) else {}
    prop_model = str(prop_spec.get("model") or "") or None
    req_data_write = bool(proposal.get("requires_data_dir_write", True))

    auth = prompt_tty_approval(
        digest=full_digest,
        card_text=str(proposal.get("card_text") or ""),
        action=str(proposal.get("action") or "submit"),
        missing_bytes=int(proposal.get("missing_bytes") or 0),
        missing_count=int(proposal.get("missing_count") or 0),
        requires_data_dir_write=req_data_write,
        model_name=prop_model,
        tty_path=tty_path,
        now_fn=now_fn,
    )

    _, app_dfd = _open_verified_private_subdir(base, "approvals")
    try:
        _write_private_json_in_dir(app_dfd, f"{full_digest}.json", auth.to_dict())
    finally:
        os.close(app_dfd)

    return auth, proposal


def consume_state_approval(
    digest: str,
    *,
    expected_action: str = "submit",
    state_dir: str | None = None,
    now_fn: Any = time.time,
) -> Authorization:
    """Atomically consume `<$AIMIP_STATE>/approvals/<digest>.json` via rename after strict symlink/owner/0600 checks."""
    norm_digest = digest.strip().lower()
    if not _HEX64_RE.match(norm_digest):
        raise ValidationError(f"Invalid digest {digest!r}", field="digest")

    base = resolve_state_dir(state_dir)
    _, dfd = _open_verified_private_subdir(base, "approvals")
    fname = f"{norm_digest}.json"
    consumed_name = f".consumed.{norm_digest}.{secrets.token_hex(8)}"
    try:
        try:
            lst = os.lstat(fname, dir_fd=dfd)
        except FileNotFoundError as exc:
            raise ApprovalError(
                f"No human approval found for digest {norm_digest[:12]}; run `aimip approve {norm_digest[:8]}` in your terminal.",
                field="digest",
                next_command=f"aimip approve {norm_digest[:8]}",
            ) from exc

        if stat.S_ISLNK(lst.st_mode):
            raise ApprovalError(f"Refusing symlinked approval file: {fname}")
        if not stat.S_ISREG(lst.st_mode):
            raise ApprovalError(f"Approval entry is not a regular file: {fname}")
        if lst.st_uid != os.geteuid():
            raise ApprovalError(f"Approval file {fname} is not owned by the current user")
        if stat.S_IMODE(lst.st_mode) != 0o600:
            raise ApprovalError(
                f"Approval file {fname} has insecure mode {oct(stat.S_IMODE(lst.st_mode))} (expected 0o600)"
            )

        try:
            os.replace(fname, consumed_name, src_dir_fd=dfd, dst_dir_fd=dfd)
        except FileNotFoundError as exc:
            raise ApprovalError("Approval was already consumed.") from exc

        try:
            raw_doc = _read_and_verify_regular_0600(dfd, consumed_name, expected_stat=lst)
        finally:
            try:
                os.unlink(consumed_name, dir_fd=dfd)
            except OSError:
                pass
    finally:
        os.close(dfd)

    auth = Authorization.from_dict(raw_doc)
    MemoryApprovalBroker.verify_authorization_fields(
        auth,
        expected_action=expected_action,
        expected_digest=norm_digest,
        now_ts=float(now_fn()),
    )
    return auth


def record_sealed_prepared(
    *,
    digest: str,
    run_dir: str,
    artifact_hashes: dict[str, str],
    state_dir: str | None = None,
    site_config_path: str | None = None,
    now_fn: Any = time.time,
) -> Path:
    """Seal authoritative `artifact_hashes` outside `<run_dir>` in `$AIMIP_STATE/prepared/<digest>.json` (`0600`).

    Fails closed (`ApprovalError` / `ValidationError`) if the private `0700` state directory or `0600`
    seal record cannot be created.
    """
    norm_digest = digest.strip().lower()
    if not _HEX64_RE.match(norm_digest):
        raise ValidationError(f"Invalid digest {digest!r}", field="digest")
    try:
        base = resolve_state_dir(
            state_dir,
            site_config_path=site_config_path,
            fallback_run_dir=run_dir,
        )
        prep_dir, dfd = _open_verified_private_subdir(base, "prepared")
    except OSError as exc:
        raise ApprovalError(
            f"Cannot create private prepared-run seal directory: {exc}",
            field="AIMIP_STATE",
        ) from exc
    try:
        doc = {
            "digest": norm_digest,
            "run_dir": run_dir,
            "user": current_user(),
            "created_at": float(now_fn()),
            "artifact_hashes": dict(artifact_hashes),
        }
        fname = f"{norm_digest}.json"
        _write_private_json_in_dir(dfd, fname, doc)
        return prep_dir / fname
    except OSError as exc:
        raise ApprovalError(
            f"Cannot write prepared-run seal in {prep_dir}: {exc}",
            field="AIMIP_STATE",
        ) from exc
    finally:
        os.close(dfd)


def read_sealed_prepared(
    digest: str,
    *,
    run_dir: str | None = None,
    state_dir: str | None = None,
    site_config_path: str | None = None,
) -> dict[str, Any] | None:
    """Read and verify `0600` sealed prepared-artifact hashes for *digest* from `$AIMIP_STATE/prepared/<digest>.json`."""
    norm_digest = digest.strip().lower()
    if not _HEX64_RE.match(norm_digest):
        return None
    base = resolve_state_dir(
        state_dir,
        site_config_path=site_config_path,
        fallback_run_dir=run_dir,
    )
    prep_sub = base / "prepared"
    if prep_sub.is_symlink():
        raise ApprovalError(f"Refusing symlinked prepared state directory: {prep_sub}")
    if not prep_sub.exists():
        return None
    _, dfd = _open_verified_private_subdir(base, "prepared")
    try:
        try:
            return _read_and_verify_regular_0600(dfd, f"{norm_digest}.json")
        except FileNotFoundError:
            return None
    finally:
        os.close(dfd)
