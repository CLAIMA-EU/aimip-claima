"""Pure leaf dataclasses for the shared AIMIP service pipeline:
`RunSpec -> ResolvedPlan -> PreparedRun -> Attempt -> Result` and `Authorization`.

Kept free of `aimip.services` and `aimip.families` imports so `import aimip.families`
and `import aimip.services` succeed in any order without circular imports.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
from typing import Any

from .errors import ValidationError


def canonical_json(payload: dict[str, Any]) -> str:
    """Serialize *payload* into canonical ASCII JSON for deterministic SHA-256 plan digests."""
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def compute_plan_digest(payload: dict[str, Any]) -> str:
    """Compute the SHA-256 hex digest of *payload*'s canonical JSON."""
    return hashlib.sha256(canonical_json(payload).encode("ascii")).hexdigest()


def coerce_optional_int(value: Any, field: str) -> int | None:
    """Strictly validate an optional integer field (refusing booleans, floats, and non-numeric strings)."""
    if value is None:
        return None
    if isinstance(value, bool):
        raise ValidationError(f"Invalid integer for {field!r}: got boolean {value!r}.", field=field)
    if isinstance(value, int):
        return value
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            raise ValidationError(f"Invalid integer for {field!r}: empty string.", field=field)
        if stripped.lstrip("+-").isdigit():
            return int(stripped)
        raise ValidationError(f"Invalid integer for {field!r}: {value!r}.", field=field)
    raise ValidationError(f"Invalid integer for {field!r}: {value!r}.", field=field)


def coerce_bool(value: Any, field: str, *, default: bool) -> bool:
    """Strictly validate a boolean field."""
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    raise ValidationError(
        f"Invalid boolean for {field!r}: expected true or false, got {value!r}.",
        field=field,
        allowed=[True, False],
    )


@dataclasses.dataclass(frozen=True)
class AssetRef:
    """One model, forcing, IC, source-tree, or dynamic input asset bound by a plan."""

    rel_path: str
    path: str
    role: str
    sha256: str | None
    size_bytes: int | None
    cached: bool
    prospective: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "rel_path": self.rel_path,
            "path": self.path,
            "role": self.role,
            "sha256": self.sha256,
            "size_bytes": self.size_bytes,
            "cached": self.cached,
            "prospective": self.prospective,
        }


@dataclasses.dataclass(frozen=True)
class RunSpec:
    """Normalized request for planning or preparing a model run."""

    model: str
    start: str
    steps: int | None = None
    length: str | None = None
    end: str | None = None
    members: int | None = None
    seed: int | None = None
    ic: str | None = None
    ocean_ic: str | None = None
    lora: str | None = None
    forcing_set: str | None = None
    name: str | None = None
    out: str | None = None
    monthly: bool = True
    prediction: bool = False
    resources: dict[str, Any] = dataclasses.field(default_factory=dict)
    mode: str = "slurm"
    config_file: str | None = None
    cli_overrides: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "model": self.model,
            "start": self.start,
            "steps": self.steps,
            "length": self.length,
            "end": self.end,
            "members": self.members,
            "seed": self.seed,
            "ic": self.ic,
            "ocean_ic": self.ocean_ic,
            "lora": self.lora,
            "forcing_set": self.forcing_set,
            "name": self.name,
            "out": self.out,
            "monthly": self.monthly,
            "prediction": self.prediction,
            "resources": dict(self.resources),
            "mode": self.mode,
            "config_file": self.config_file,
            "cli_overrides": list(self.cli_overrides),
        }

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> RunSpec:
        if not isinstance(raw, dict):
            raise ValidationError("RunSpec payload must be a mapping.", field="spec")
        if not raw.get("model"):
            raise ValidationError("RunSpec is missing 'model'.", field="model")
        if not raw.get("start"):
            raise ValidationError("RunSpec is missing 'start'.", field="start")
        res_raw = raw.get("resources") or {}
        if not isinstance(res_raw, dict):
            raise ValidationError("RunSpec 'resources' must be a mapping.", field="resources")
        res_clean = dict(res_raw)
        if res_clean.get("cpus_per_task") is not None:
            res_clean["cpus_per_task"] = coerce_optional_int(
                res_clean["cpus_per_task"], "resources.cpus_per_task"
            )
        return cls(
            model=str(raw["model"]),
            start=str(raw["start"]),
            steps=coerce_optional_int(raw.get("steps"), "steps"),
            length=str(raw["length"]) if raw.get("length") is not None else None,
            end=str(raw["end"]) if raw.get("end") is not None else None,
            members=coerce_optional_int(raw.get("members"), "members"),
            seed=coerce_optional_int(raw.get("seed"), "seed"),
            ic=str(raw["ic"]) if raw.get("ic") is not None else None,
            ocean_ic=str(raw["ocean_ic"]) if raw.get("ocean_ic") is not None else None,
            lora=str(raw["lora"]) if raw.get("lora") is not None else None,
            forcing_set=str(raw["forcing_set"]) if raw.get("forcing_set") is not None else None,
            name=str(raw["name"]) if raw.get("name") is not None else None,
            out=str(raw["out"]) if raw.get("out") is not None else None,
            monthly=coerce_bool(raw.get("monthly"), "monthly", default=True),
            prediction=coerce_bool(raw.get("prediction"), "prediction", default=False),
            resources=res_clean,
            mode=str(raw.get("mode", "slurm")),
            config_file=str(raw["config_file"]) if raw.get("config_file") is not None else None,
            cli_overrides=tuple(str(x) for x in (raw.get("cli_overrides") or ())),
        )


@dataclasses.dataclass(frozen=True)
class ResolvedPlan:
    """Side-effect-free resolved execution plan and its deterministic SHA-256 digest."""

    spec: RunSpec
    model: str
    family: str
    status_tier: str
    task: str
    calendar: str
    timestep_hours: int
    start_time: str
    end_time: str
    steps: int
    members: int
    seed: int
    ensemble_strategy: str
    ensemble_note: str
    forcing_set: str | None
    ic_source: str | None
    ocean_ic_source: str | None
    lora_checkpoint: str | None
    resources: dict[str, Any]
    runtime_identity: dict[str, Any]
    assets: tuple[AssetRef, ...]
    missing_assets: tuple[AssetRef, ...]
    missing_bytes: int
    output_dir: str
    run_name: str | None
    cli_overrides: tuple[str, ...]
    canonical_payload: dict[str, Any]
    digest: str
    ic_checked: bool = True

    @property
    def short_digest(self) -> str:
        return self.digest[:12]

    @property
    def approval_prefix(self) -> str:
        return self.digest[:8]

    @property
    def requires_staging(self) -> bool:
        """True when preparation materializes dynamic/unpinned assets that change the final submit digest."""
        return any(a.prospective for a in self.missing_assets)

    @property
    def required_approval_action(self) -> str:
        """Return `'prepare'` for prospective staging plans, or `'submit'` when prospective digest == final digest."""
        return "prepare" if self.requires_staging else "submit"

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": "aimip-plan-v1",
            "digest": self.digest,
            "short_digest": self.short_digest,
            "requires_staging": self.requires_staging,
            "required_approval_action": self.required_approval_action,
            "model": self.model,
            "family": self.family,
            "status_tier": self.status_tier,
            "task": self.task,
            "calendar": self.calendar,
            "timestep_hours": self.timestep_hours,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "steps": self.steps,
            "members": self.members,
            "seed": self.seed,
            "ensemble_strategy": self.ensemble_strategy,
            "ensemble_note": self.ensemble_note,
            "forcing_set": self.forcing_set,
            "ic_source": self.ic_source,
            "ocean_ic_source": self.ocean_ic_source,
            "ic_checked": self.ic_checked,
            "lora_checkpoint": self.lora_checkpoint,
            "resources": dict(self.resources),
            "runtime_identity": dict(self.runtime_identity),
            "assets": [a.to_dict() for a in self.assets],
            "missing_assets": [a.to_dict() for a in self.missing_assets],
            "missing_bytes": self.missing_bytes,
            "cached_assets_count": len(self.assets) - len(self.missing_assets),
            "output_dir": self.output_dir,
            "run_name": self.run_name,
            "cli_overrides": list(self.cli_overrides),
            "spec": self.spec.to_dict(),
            "canonical": dict(self.canonical_payload),
        }


@dataclasses.dataclass(frozen=True)
class PreparedRun:
    """Materialized run directory on disk with sealed `artifact_hashes`."""

    plan: ResolvedPlan
    run_dir: str
    plan_path: str
    manifest_path: str
    config_path: str
    sbatch_path: str
    manifest: dict[str, Any]
    artifact_hashes: dict[str, str] = dataclasses.field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "digest": self.plan.digest,
            "short_digest": self.plan.short_digest,
            "run_dir": self.run_dir,
            "run_name": self.plan.run_name,
            "plan_path": self.plan_path,
            "manifest_path": self.manifest_path,
            "config_path": self.config_path,
            "sbatch_path": self.sbatch_path,
            "artifact_hashes": dict(self.artifact_hashes),
            "plan": self.plan.to_dict(),
            "manifest": dict(self.manifest),
        }


@dataclasses.dataclass(frozen=True)
class Attempt:
    """Submission or execution state of one prepared run."""

    run_dir: str
    run_name: str
    digest: str | None
    scheduler: str
    attempt_id: str | None
    state: str
    elapsed: str | None = None
    exit_code: str | None = None
    reconciled: bool = False
    host_command: str | None = None

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "run_dir": self.run_dir,
            "out_dir": self.run_dir,
            "run_name": self.run_name,
            "digest": self.digest,
            "scheduler": self.scheduler,
            "attempt_id": self.attempt_id,
            "slurm_job_id": self.attempt_id if self.scheduler == "slurm" else None,
            "local_pid": (
                int(self.attempt_id)
                if self.scheduler == "none" and self.attempt_id is not None and self.attempt_id.isdigit()
                else None
            ),
            "state": self.state,
            "elapsed": self.elapsed,
            "exit_code": self.exit_code,
            "reconciled": self.reconciled,
            "host_command": self.host_command,
        }
        if self.scheduler == "slurm" and self.attempt_id is not None:
            out["job_id"] = self.attempt_id
        return out


@dataclasses.dataclass(frozen=True)
class Result:
    """Area-weighted global-mean diagnostic summary of a run directory."""

    run_dir: str
    model: str | None
    state: str
    variables: dict[str, Any]
    raw_summary: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        out = dict(self.raw_summary)
        out.setdefault("run_dir", self.run_dir)
        if self.model is not None:
            out.setdefault("model", self.model)
        out.setdefault("state", self.state)
        return out


@dataclasses.dataclass(frozen=True)
class Authorization:
    """Short-lived, single-use human approval token bound to an exact digest and user."""

    action: str
    digest: str
    user: str
    issued_at: float
    expires_at: float
    nonce: str
    allow_download: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "action": self.action,
            "digest": self.digest,
            "user": self.user,
            "issued_at": self.issued_at,
            "expires_at": self.expires_at,
            "nonce": self.nonce,
            "allow_download": self.allow_download,
        }

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> Authorization:
        return cls(
            action=str(raw["action"]),
            digest=str(raw["digest"]),
            user=str(raw["user"]),
            issued_at=float(raw["issued_at"]),
            expires_at=float(raw["expires_at"]),
            nonce=str(raw["nonce"]),
            allow_download=bool(raw.get("allow_download", False)),
        )
