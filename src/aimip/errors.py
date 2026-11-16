"""Structured exception hierarchy and stable exit codes for AIMIP."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

EXIT_OK = 0
EXIT_EXECUTION = 1
EXIT_USAGE = 2
EXIT_MISSING_DATA = 3
EXIT_APPROVAL = 4
EXIT_DEADLINE = 75

EXIT_CODES: dict[str, int] = {
    "ok": EXIT_OK,
    "doctor_failed": EXIT_EXECUTION,
    "execution_error": EXIT_EXECUTION,
    "validation_error": EXIT_USAGE,
    "config_error": EXIT_USAGE,
    "registry_error": EXIT_USAGE,
    "missing_assets": EXIT_MISSING_DATA,
    "approval_error": EXIT_APPROVAL,
    "deadline": EXIT_DEADLINE,
}


class AimipError(Exception):
    """Base exception for expected AIMIP errors with structured CLI/JSON fields."""

    _default_code = "validation_error"

    def __init__(
        self,
        message: str,
        *,
        field: str | None = None,
        allowed: Sequence[Any] | None = None,
        next_command: str | None = None,
        code: str | None = None,
        exit_code: int | None = None,
    ) -> None:
        super().__init__(message)
        self.message = str(message)
        self.field = field
        self.allowed = list(allowed) if allowed is not None else None
        self.next_command = next_command
        self.code = code or self._default_code
        self.exit_code = (
            int(exit_code)
            if exit_code is not None
            else EXIT_CODES.get(self.code, EXIT_USAGE)
        )

    def to_dict(self) -> dict[str, Any]:
        """Serialize structured error details for `--json` envelopes and tool responses."""
        return {
            "code": self.code,
            "exit_code": self.exit_code,
            "message": self.message,
            "field": self.field,
            "allowed": self.allowed,
            "next_command": self.next_command,
        }

    to_error_dict = to_dict


class ValidationError(AimipError):
    """Raised when CLI input, environment values, or file contents fail validation."""

    _default_code = "validation_error"


class RegistryError(AimipError):
    """Raised when the model registry cannot be loaded or is malformed."""

    _default_code = "registry_error"


class ConfigError(AimipError):
    """Raised when run/spin configuration constraints are violated."""

    _default_code = "config_error"


class MissingAssetsError(ConfigError):
    """Raised when required model weights, forcings, or source code are not staged locally."""

    _default_code = "missing_assets"


class ApprovalError(AimipError):
    """Raised when human TTY/MCP approval is missing, mismatched, expired, replayed, or tampered."""

    _default_code = "approval_error"


class ExecutionError(AimipError):
    """Raised when scheduler submission, job execution, or cancellation fails at runtime."""

    _default_code = "execution_error"


def enrich_error(
    exc: AimipError,
    *,
    model: str | None = None,
    allowed_models: Sequence[str] | None = None,
) -> AimipError:
    """Populate `field`, `allowed`, `next_command`, and `exit_code` when inferred from legacy validators."""
    msg = exc.message
    low = msg.lower()

    if exc.field is None:
        if "unknown model" in low or "model name" in low:
            exc.field = "model"
        elif "forcing set" in low or "forcing_sets" in low or "crosses forcing sets" in low:
            exc.field = "forcing_set"
        elif "start time" in low or "start=" in low or "--start" in low:
            exc.field = "start"
        elif "end time" in low or "--end" in low:
            exc.field = "end"
        elif "--length" in low or "length " in low:
            exc.field = "length"
        elif "steps" in low:
            exc.field = "steps"
        elif "members" in low:
            exc.field = "members"
        elif "seed" in low:
            exc.field = "seed"
        elif "account" in low:
            exc.field = "account"
        elif "partition" in low:
            exc.field = "partition"
        elif "lora" in low:
            exc.field = "lora"
        elif "--ic" in low or "initial condition" in low or "ic file" in low:
            exc.field = "ic"
        elif "run directory" in low or "--out" in low or "run name" in low:
            exc.field = "out"

    if exc.allowed is None and exc.field == "model" and allowed_models is not None:
        exc.allowed = list(allowed_models)

    if exc.next_command is None:
        if "aimip fetch -m " in msg:
            idx = msg.find("aimip fetch -m ")
            tail = msg[idx:].split("`")[0].split("\n")[0].strip().rstrip(".")
            exc.next_command = tail
        elif "missing required" in low and model:
            exc.next_command = f"aimip fetch -m {model}"
        elif "change_me" in low or "aimip site init" in low:
            exc.next_command = "aimip site show"
        elif exc.field == "model":
            exc.next_command = "aimip models"
        elif exc.field in ("start", "end", "forcing_set") and model:
            exc.next_command = f"aimip show {model}"

    if ("missing required" in low or "forbids downloads" in low) and exc.code == "config_error":
        exc.code = "missing_assets"
        exc.exit_code = EXIT_MISSING_DATA

    return exc
