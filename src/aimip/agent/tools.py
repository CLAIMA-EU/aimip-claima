"""In-process service adapters and host-side `/dev/tty` digest approval for `aimip chat`.

In production (`cli=None`), `AgentTools` calls `aimip.services` directly in-process and never
shells out to an `aimip` subprocess. Standalone `cli_json` is retained as a secret-scrubbing
isolated-Python helper, and unit tests may pass `cli=...` to stub catalogue responses.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
import tempfile
import time
from typing import Any

from ..errors import AimipError, ApprovalError
from ..services import (
    MemoryApprovalBroker,
    ResolvedPlan,
    RunSpec,
    cite as svc_cite,
    execute_tty_plan_flow,
    logs as svc_logs,
    ls as svc_ls,
    models as svc_models,
    plan as svc_plan,
    prepare as svc_prepare,
    render_plan_card,
    resolve_configured_run_root,
    show as svc_show,
    status as svc_status,
    submit as svc_submit,
    summary as svc_summary,
)
from ..services.core import _load_catalog_for_discovery
from .security import trusted_lora_path

MAX_MEMBERS = 64
MAX_SPIN_MEMBERS = 24
MAX_REPLICATION_BYTES = 2 * 1024**3


def cli_json(args: list[str]) -> Any:
    """Invoke isolated `python -I -B -m aimip ... --json` with API secrets scrubbed and `AIMIP_DATA_MODE=ro`."""
    env = {
        k: v
        for k, v in os.environ.items()
        if k
        not in {
            "GEMINI_API_KEY",
            "GOOGLE_API_KEY",
            "AIMIP_GEMINI_KEY_FILE",
            "AIMIP_GEMINI_VERTEX_TOKEN_FILE",
            "GOOGLE_APPLICATION_CREDENTIALS",
        }
        and not k.startswith(
            (
                "LANGCHAIN_",
                "LANGSMITH_",
                "APPTAINERENV_GEMINI",
                "APPTAINERENV_GOOGLE",
                "SINGULARITYENV_GEMINI",
                "SINGULARITYENV_GOOGLE",
            )
        )
    }
    env["AIMIP_DATA_MODE"] = "ro"
    result = subprocess.run(
        [sys.executable, "-I", "-B", "-m", "aimip", *args, "--json"],
        capture_output=True,
        text=True,
        env=env,
        timeout=300,
        check=False,
    )
    if result.returncode:
        err_text = result.stderr.strip()
        if not err_text and result.stdout.strip():
            try:
                err_doc = json.loads(result.stdout)
                if isinstance(err_doc, dict) and isinstance(err_doc.get("error"), dict):
                    err_text = str(err_doc["error"].get("message") or "")
            except json.JSONDecodeError:
                pass
        raise ValueError(err_text or f"CLI exited {result.returncode}")
    doc = json.loads(result.stdout)
    if isinstance(doc, dict) and doc.get("aimip") == "1" and "result" in doc:
        return doc["result"]
    return doc


def _with_fetch_hint(model: str, text: str, *, custom_ic: bool = False) -> str:
    """Append an `aimip fetch -m <model>` hint when a canonical model is not staged yet."""
    from ..setup_data import CANONICAL_MODELS

    low = text.lower()
    if "aimip fetch" in low or model not in CANONICAL_MODELS:
        return text
    missing_model_or_data = (
        "unknown model" in low
        or "missing required" in low
        or "missing " in low
        or "not yet fetched" in low
        or (not custom_ic and "file not found" in low)
    )
    if missing_model_or_data:
        return f"{text} If model {model} is not downloaded yet, run `aimip fetch -m {model}` on the host first."
    return text


def fingerprint(files: list[Path]) -> str:
    digest = hashlib.sha256()
    for path in files:
        if path.is_symlink() or not path.is_file():
            raise ValueError(f"Plan artifact missing or symlinked: {path}")
        digest.update(str(path.resolve()).encode())
        with path.open("rb") as stream:
            digest.update(hashlib.file_digest(stream, "sha256").digest())
    return digest.hexdigest()


def _planned_outputs(model_spec: Any, *, monthly: bool, prediction: bool) -> dict[str, Any]:
    if model_spec.family == "samudra":
        return {
            "monthly": True,
            "prediction": False,
            "files": [
                "ocean/monthly_mean_predictions.nc",
                "ocean/monthly_thetao.nc",
            ],
        }
    if model_spec.family == "camulator":
        return {
            "monthly": True,
            "prediction": False,
            "files": [
                "monthly_mean_predictions.nc",
                "monthly_levels.nc",
            ],
        }
    if model_spec.family == "md":
        return {
            "monthly": True,
            "prediction": False,
            "files": [
                "monthly_mean_predictions.nc",
                "restart.nc",
            ],
        }
    if model_spec.family in ("cbottle", "sdyff"):
        return {
            "monthly": True,
            "prediction": False,
            "files": ["monthly_mean_predictions.nc"],
        }
    if model_spec.family in ("e2s-dlesym", "arches"):
        return {
            "monthly": True,
            "prediction": False,
            "files": [
                "atmosphere/monthly_mean_predictions.nc",
                "ocean/monthly_mean_predictions.nc",
            ],
        }
    if model_spec.is_coupled:
        return {
            "ocean": {"monthly": bool(monthly), "prediction": bool(prediction)},
            "atmosphere": {"monthly": bool(monthly), "prediction": bool(prediction)},
        }
    return {"monthly": bool(monthly), "prediction": bool(prediction)}


def _build_run_detail(resolved: ResolvedPlan) -> dict[str, Any]:
    from ..lora import inspect_lora
    from ..registry import get_model_ic_provenance

    catalog = _load_catalog_for_discovery()
    model_spec = catalog[resolved.model]
    ic_prov = (
        f"Custom IC ({resolved.ic_source})"
        if resolved.spec.ic is not None
        else get_model_ic_provenance(model_spec.name, model_spec.notes)
    )
    lora_meta = (
        inspect_lora(resolved.lora_checkpoint).to_dict()
        if resolved.lora_checkpoint
        else None
    )
    return {
        "model": resolved.model,
        "start": resolved.start_time,
        "end": resolved.end_time,
        "steps": resolved.steps,
        "members": resolved.members,
        "ic_source": resolved.ic_source,
        "forcing_set": {"name": resolved.forcing_set} if resolved.forcing_set else None,
        "seed": resolved.seed,
        "seed_explicit": resolved.spec.seed is not None,
        "sbatch_resources": dict(resolved.resources),
        "stochastic": model_spec.stochastic,
        "calendar": resolved.calendar,
        "ic_provenance": ic_prov,
        "model_notes": model_spec.notes,
        "ensemble_strategy": resolved.ensemble_strategy,
        "ensemble_note": resolved.ensemble_note,
        "lora_checkpoint": resolved.lora_checkpoint,
        "lora_info": lora_meta,
        "outputs": _planned_outputs(
            model_spec,
            monthly=resolved.spec.monthly,
            prediction=resolved.spec.prediction,
        ),
        "walltime_capped": bool(resolved.resources.get("walltime_capped", False)),
    }


def _manifest_outputs(manifest: dict[str, Any]) -> dict[str, Any]:
    import yaml

    if manifest.get("family") == "samudra":
        return {
            "monthly": True,
            "prediction": False,
            "files": [
                "ocean/monthly_mean_predictions.nc",
                "ocean/monthly_thetao.nc",
            ],
        }
    if manifest.get("family") == "camulator":
        return {
            "monthly": True,
            "prediction": False,
            "files": [
                "monthly_mean_predictions.nc",
                "monthly_levels.nc",
            ],
        }
    if manifest.get("family") == "md":
        return {
            "monthly": True,
            "prediction": False,
            "files": [
                "monthly_mean_predictions.nc",
                "restart.nc",
            ],
        }
    if manifest.get("family") in ("cbottle", "sdyff"):
        return {
            "monthly": True,
            "prediction": False,
            "files": ["monthly_mean_predictions.nc"],
        }
    if manifest.get("family") in ("e2s-dlesym", "arches"):
        return {
            "monthly": True,
            "prediction": False,
            "files": [
                "atmosphere/monthly_mean_predictions.nc",
                "ocean/monthly_mean_predictions.nc",
            ],
        }
    writer = yaml.safe_load(Path(manifest["config_path"]).read_text())["data_writer"]
    if "save_monthly_files" not in writer:
        return {
            part: {
                "monthly": w.get("save_monthly_files"),
                "prediction": w.get("save_prediction_files"),
            }
            for part, w in writer.items()
            if isinstance(w, dict)
        }
    return {
        "monthly": writer["save_monthly_files"],
        "prediction": writer["save_prediction_files"],
    }


class AgentTools:
    """Service-backed tools for `aimip chat`.

    Only the host REPL outside the LLM graph can prompt `/dev/tty`, mint a single-use
    `Authorization`, and invoke `approve_and_execute_on_tty(...)`.
    """

    def __init__(self, cli: Any = None, *, now_fn: Any = time.time) -> None:
        self.cli = cli
        self._now_fn = now_fn
        self.broker = MemoryApprovalBroker(now_fn=now_fn)
        self._proposed_spec: RunSpec | None = None
        self._legacy_pending: dict[str, Any] | None = None

    @property
    def pending(self) -> dict[str, Any] | None:
        if self.broker.pending is not None:
            return json.loads(json.dumps(self.broker.pending))
        return json.loads(json.dumps(self._legacy_pending)) if self._legacy_pending is not None else None

    def revoke(self) -> None:
        self.broker.revoke_all()
        self._proposed_spec = None
        self._legacy_pending = None

    def list_models(
        self,
        task: str | None = None,
        family: str | None = None,
        status: str | None = None,
    ) -> list[dict[str, Any]]:
        """List registered models, forcing windows, status tiers, and IC availability."""
        if self.cli is not None and task is None and family is None and status is None:
            return self.cli(["models"])
        return svc_models(task=task, family=family, status=status)

    def show_model(self, model: str, include_examples: bool = False) -> dict[str, Any]:
        """Return detailed model card (grid, step, calendar, forcing sets, licence, limits)."""
        return svc_show(model, include_examples=include_examples)

    def cite_model(self, model: str) -> dict[str, Any]:
        """Return BibTeX citation entries and licence attribution for a model."""
        return svc_cite(model)

    def list_ics(self, model: str) -> dict[str, Any]:
        """List shipped initial-condition files and timestamps for an `fme` model."""
        from ..emu_catalog import EMU_MODELS
        from ..ic import list_model_ics, list_model_ocean_ics
        from ..registry import get_model, get_model_ic_provenance

        if model in EMU_MODELS and model != "md-1.5":
            if model == "archesclimate-ssp":
                ic_origin = "the scenario's 2015 state."
            elif model == "samudra-om4":
                ic_origin = "the OM4 ocean window at --start (staged from OSN)."
            elif model == "camulator":
                ic_origin = "its 69 published .pth initial-condition tensors (1980-01-01, Jan 1 & Jul 1 of 1981-2014)."
            elif model == "cbottle-3d":
                ic_origin = "conditional draws without an initial state (conditioned on date/time and monthly SST)."
            elif model == "spherical-dyffusion":
                ic_origin = "its 365-day FV3GFS reference cycle (Zenodo 10791087)."
            else:
                ic_origin = "ERA5 at --start."
            raise ValueError(
                f"`list_ics` applies to fme models only (plus md-1.5 NetCDF ICs); {model} starts from {ic_origin}"
            )
        try:
            spec = get_model(model)
            infos = list_model_ics(spec)
            ocean_infos = list_model_ocean_ics(spec) if spec.is_coupled else []
            payload: dict[str, Any] = {
                "model": spec.name,
                "ic_provenance": get_model_ic_provenance(spec.name, spec.notes),
                "ic_files": [info.to_dict() for info in infos],
            }
            if spec.is_coupled:
                payload["ocean_ic_files"] = [info.to_dict() for info in ocean_infos]
            return payload
        except AimipError as exc:
            raise ValueError(_with_fetch_hint(model, str(exc))) from exc

    def plan(
        self,
        model: str,
        start: str,
        steps: int | None = None,
        length: str | None = None,
        end: str | None = None,
        members: int | None = None,
        ic: str | None = None,
        ocean_ic: str | None = None,
        seed: int | None = None,
        forcing_set: str | None = None,
        lora: str | None = None,
        name: str | None = None,
        out: str | None = None,
    ) -> dict[str, Any]:
        """Resolve and validate a run specification with zero side effects (does not queue approval)."""
        if lora is not None:
            lora = trusted_lora_path(lora)
        self._check_run_size(model, start, members, ic)
        spec = RunSpec(
            model=model,
            start=start,
            steps=steps,
            length=length,
            end=end,
            members=members,
            seed=seed,
            ic=ic,
            ocean_ic=ocean_ic,
            lora=lora,
            forcing_set=forcing_set,
            name=name,
            out=out,
        )
        try:
            resolved = svc_plan(spec)
        except AimipError as exc:
            raise ValueError(_with_fetch_hint(model, str(exc), custom_ic=ic is not None)) from exc
        run_detail = _build_run_detail(resolved)
        result = {
            "kind": "run",
            "run_dir": resolved.output_dir,
            "digest": resolved.digest,
            "short_digest": resolved.short_digest,
            "approval_prefix": resolved.approval_prefix,
            "card": render_plan_card(resolved, is_tty=False),
            "plan": resolved.to_dict(),
            "runs": [run_detail],
            "gpu_hours_estimate": resolved.resources["gpu_hours_estimate"],
            "walltime_capped": bool(resolved.resources.get("walltime_capped", False)),
            "missing_bytes": resolved.missing_bytes,
            "missing_count": len(resolved.missing_assets),
        }
        return json.loads(json.dumps(result))

    def propose_run(
        self,
        model: str,
        start: str,
        steps: int | None = None,
        length: str | None = None,
        end: str | None = None,
        members: int | None = None,
        ic: str | None = None,
        ocean_ic: str | None = None,
        seed: int | None = None,
        forcing_set: str | None = None,
        lora: str | None = None,
        name: str | None = None,
        out: str | None = None,
    ) -> dict[str, Any]:
        """Propose a run for human approval on `/dev/tty`.

        Side-effect-free during tool execution: returns the plan card and 64-hex `digest`.
        The host loop (never the LLM) prompts the human on `/dev/tty` for `digest[:8]`
        (and a separate download yes/no when assets are missing), then prepares and submits.
        """
        self.revoke()
        fresh_stub_dir: Path | None = None
        try:
            if lora is not None:
                lora = trusted_lora_path(lora)
            self._check_run_size(model, start, members, ic)
            if self.cli is not None:
                fresh_stub_dir = self._fresh_dir()
                self.cli(["run", "--no-fetch", f"--model={model}", f"--start={start}", f"--out={fresh_stub_dir}", "--dry-run"])
                shutil.rmtree(fresh_stub_dir, ignore_errors=True)
                fresh_stub_dir = None

            spec = RunSpec(
                model=model,
                start=start,
                steps=steps,
                length=length,
                end=end,
                members=members,
                seed=seed,
                ic=ic,
                ocean_ic=ocean_ic,
                lora=lora,
                forcing_set=forcing_set,
                name=name,
                out=out,
            )
            resolved = svc_plan(spec)
        except AimipError as exc:
            if fresh_stub_dir is not None:
                shutil.rmtree(fresh_stub_dir, ignore_errors=True)
            self.revoke()
            raise ValueError(_with_fetch_hint(model, str(exc), custom_ic=ic is not None)) from exc
        except BaseException:
            if fresh_stub_dir is not None:
                shutil.rmtree(fresh_stub_dir, ignore_errors=True)
            self.revoke()
            raise

        card_text = render_plan_card(resolved, is_tty=False)
        card_data = resolved.to_dict()
        run_detail = _build_run_detail(resolved)
        req_action = resolved.required_approval_action
        self._proposed_spec = spec
        self.broker.register_proposal(
            action=req_action,
            digest=resolved.digest,
            card_text=card_text,
            card_data=card_data,
            missing_bytes=resolved.missing_bytes,
            missing_count=len(resolved.missing_assets),
        )
        result = {
            "kind": "run",
            "action": req_action,
            "requires_staging": resolved.requires_staging,
            "run_dir": resolved.output_dir,
            "digest": resolved.digest,
            "short_digest": resolved.short_digest,
            "approval_prefix": resolved.approval_prefix,
            "card": card_text,
            "plan": card_data,
            "runs": [run_detail],
            "gpu_hours_estimate": resolved.resources["gpu_hours_estimate"],
            "walltime_capped": bool(resolved.resources.get("walltime_capped", False)),
            "missing_bytes": resolved.missing_bytes,
            "missing_count": len(resolved.missing_assets),
            "approval_required": (
                f"The host terminal will ask the human on /dev/tty to type '{resolved.approval_prefix}' "
                "to approve staging/preparation, then confirm the final post-stage digest before submitting."
                if resolved.requires_staging
                else f"The host terminal will ask the human on /dev/tty to type '{resolved.approval_prefix}' "
                "before preparing or submitting."
            ),
        }
        return json.loads(json.dumps(result))

    def plan_run(
        self,
        model: str,
        start: str,
        steps: int | None = None,
        members: int | None = None,
        ic: str | None = None,
        seed: int | None = None,
        forcing_set: str | None = None,
        lora: str | None = None,
        *,
        length: str | None = None,
        end: str | None = None,
        ocean_ic: str | None = None,
        name: str | None = None,
        out: str | None = None,
    ) -> dict[str, Any]:
        """Prepare a side-effect-free run proposal requiring `/dev/tty` 8-hex digest approval."""
        return self.propose_run(
            model=model,
            start=start,
            steps=steps,
            length=length,
            end=end,
            members=members,
            ic=ic,
            ocean_ic=ocean_ic,
            seed=seed,
            forcing_set=forcing_set,
            lora=lora,
            name=name,
            out=out,
        )

    def approve_and_execute_on_tty(self, *, tty_path: str = "/dev/tty") -> dict[str, Any]:
        """Host-only entrypoint: prompt human on `/dev/tty` for `digest[:8]`, mint single-use `Authorization`, and execute."""
        if self._legacy_pending is not None and self.broker.pending is None:
            return self._approve_legacy_spin_or_collect_on_tty(tty_path=tty_path)

        spec = self._proposed_spec
        if spec is None or self.broker.pending is None:
            raise ApprovalError("No pending run proposal to approve.")

        try:
            attempt = execute_tty_plan_flow(
                spec,
                broker=self.broker,
                tty_path=tty_path,
                now_fn=self._now_fn,
            )
            return attempt.to_dict()
        finally:
            self._proposed_spec = None

    def _approve_legacy_spin_or_collect_on_tty(self, *, tty_path: str = "/dev/tty") -> dict[str, Any]:
        """Host `/dev/tty` 8-hex digest approval for multi-member spin / spin-collect cards."""
        from ..services import prompt_tty_approval

        pending = self._legacy_pending
        self._legacy_pending = None
        if pending is None:
            raise ApprovalError("No pending spin/collect proposal to approve.")
        files: list[Path] = pending["files"]
        expected = pending["digest"]
        try:
            current_fp = fingerprint(files)
        except ValueError as exc:
            raise ApprovalError(f"Plan changed, approve again ({exc}).") from exc
        if current_fp != expected:
            raise ApprovalError("Plan changed, approve again.")
        card = pending["card"]
        auth = prompt_tty_approval(
            digest=expected,
            card_text=json.dumps(card, indent=2),
            action="collect" if card["kind"] == "collect" else "submit",
            tty_path=tty_path,
            now_fn=self._now_fn,
        )
        if fingerprint(files) != auth.digest:
            raise ApprovalError("Plan changed, approve again.")
        if card["kind"] == "collect":
            targets = [Path(card["output"])] + (
                [Path(card["ocean_output"])] if card.get("ocean_output") else []
            )
            if any(t.exists() or t.is_symlink() for t in targets):
                raise ApprovalError("Collection output already exists; prepare a new plan.")
            cmd = shlex.join(["aimip", "collect", card["spin_dir"], "--out", card["output"]])
            return {
                "submitted": False,
                "run_dir": card["run_dir"],
                "digest": auth.digest,
                "state": "PREPARED_COLLECT_COMMAND",
                "host_command": cmd,
            }
        attempt = svc_submit(card["run_dir"])
        return attempt.to_dict()

    @staticmethod
    def _check_run_size(
        model: str,
        start: str,
        members: int | None,
        ic: str | None,
    ) -> None:
        import netCDF4

        from ..emu_catalog import EMU_MODELS
        from ..registry import get_model
        from ..runner import model_has_missing_files, resolve_ic_and_members
        from ..validators import validate_iso_timestamp

        if members is not None and not 1 <= members <= MAX_MEMBERS:
            raise ValueError(f"Chat plans support at most {MAX_MEMBERS} members")
        if model in EMU_MODELS:
            return
        try:
            spec = get_model(model)
        except AimipError:
            # Unknown or unfetched model: `svc_plan` will validate or report missing assets cleanly.
            return
        if ic is None and model_has_missing_files(spec):
            # Shipped ICs are not staged yet; `svc_plan` computes missing bytes without opening NetCDF files.
            return
        try:
            resolved = resolve_ic_and_members(
                model=spec,
                norm_start=validate_iso_timestamp(start),
                ic_path=ic,
                members=members,
            )
        except AimipError as exc:
            raise ValueError(_with_fetch_hint(model, str(exc), custom_ic=ic is not None)) from exc
        if resolved.total_members > MAX_MEMBERS:
            raise ValueError(
                f"IC implies {resolved.total_members} members; chat limit is {MAX_MEMBERS}"
            )
        if resolved.replicate_repeats > 1:
            ic_paths = [resolved.source_ic.path]
            if resolved.source_ocean_ic is not None:
                ic_paths.append(resolved.source_ocean_ic.path)
            size = 0
            for ic_file in ic_paths:
                if os.path.isfile(ic_file):
                    with netCDF4.Dataset(ic_file) as dataset:
                        size += sum(
                            v.size * getattr(v.dtype, "itemsize", 8)
                            for v in dataset.variables.values()
                        )
            estimate = size * resolved.replicate_repeats
            if estimate > MAX_REPLICATION_BYTES:
                raise ValueError(
                    f"IC replication estimate {estimate} bytes exceeds chat limit {MAX_REPLICATION_BYTES}; use the CLI deliberately"
                )

    def plan_spin(
        self,
        model: str,
        from_times: list[str],
        to_time: str,
        seed: int = 0,
        forcing_set: str | None = None,
        lora: str | None = None,
    ) -> dict[str, Any]:
        """Prepare spin members from shipped IC timestamps up to *to_time* (requires `/dev/tty` digest approval to submit)."""
        from ..ic import prepare_spin
        from ..registry import get_model, resolve_playground_dir
        from ..runner import (
            ACE2_ERA5_GPU_HOURS_PER_MODEL_YEAR,
            HOURS_PER_JULIAN_YEAR,
            SlurmOptions,
            compute_slurm_walltime_details,
        )
        from ..site_config import load_site_config

        self.revoke()
        if not 1 <= len(from_times) <= MAX_SPIN_MEMBERS:
            raise ValueError(f"Chat spin plans require 1–{MAX_SPIN_MEMBERS} starts")
        if lora is not None:
            lora = trusted_lora_path(lora)

        out = self._fresh_dir()
        try:
            spec = get_model(model)
            site_cfg = load_site_config()
            slurm_opts = SlurmOptions.from_site(site_cfg)
            manifest = prepare_spin(
                model=spec,
                from_times=from_times,
                to_time=to_time,
                spin_dir=str(out),
                mode="dry-run",
                seed=seed,
                lora_path=lora,
                forcing_set_name=forcing_set,
                slurm_opts=slurm_opts,
                site_config=site_cfg,
                playground_dir=resolve_playground_dir(None),
                auto_fetch=False,
            )
            files = [out / "job.sbatch", out / "spin-manifest.json"]
            runs = []
            for member in manifest["members"]:
                directory = Path(member["experiment_dir"])
                if directory.parent != out:
                    raise ValueError("Spin member escaped its fresh plan directory")
                runs.append(json.loads((directory / "aimip-manifest.json").read_text()))
                files.extend(
                    directory / fname
                    for fname in ("aimip-manifest.json", "config.yaml", "job.sbatch")
                )
            gpu_hours = sum(
                m["steps"]
                * m["members"]
                * m["registry_entry"]["timestep_hours"]
                / HOURS_PER_JULIAN_YEAR
                * ACE2_ERA5_GPU_HOURS_PER_MODEL_YEAR
                for m in runs
            )
            _, _, capped = compute_slurm_walltime_details(
                steps=sum(m["steps"] for m in runs),
                timestep_hours=runs[0]["registry_entry"]["timestep_hours"],
                n_trajectories=1,
                max_walltime=site_cfg.max_time,
            )
            digest = fingerprint(files)
            card = {
                "kind": "spin",
                "run_dir": str(out),
                "digest": digest,
                "approval_prefix": digest[:8],
                "runs": [
                    {
                        "model": m["registry_entry"]["name"],
                        "start": m["start_time"],
                        "end": m["end_time"],
                        "steps": m["steps"],
                        "members": m["members"],
                        "forcing_set": m["forcing_set"],
                        "sbatch_resources": m["slurm"],
                        "stochastic": m["registry_entry"]["stochastic"],
                        "calendar": m["ic_calendar"],
                        "ic_provenance": m["ic_provenance"],
                        "outputs": _manifest_outputs(m),
                        "walltime_capped": m["slurm"].get("walltime_capped", False),
                    }
                    for m in runs
                ],
                "gpu_hours_estimate": round(gpu_hours, 6),
                "walltime_capped": capped,
            }
            self._legacy_pending = {"card": card, "files": files, "digest": digest}
            return json.loads(json.dumps(card))
        except AimipError as exc:
            shutil.rmtree(out, ignore_errors=True)
            self.revoke()
            raise ValueError(_with_fetch_hint(model, str(exc))) from exc
        except BaseException:
            shutil.rmtree(out, ignore_errors=True)
            self.revoke()
            raise

    def plan_collect(self, spin_dir: str) -> dict[str, Any]:
        """Plan collection of completed spin restarts."""
        from ..validators import validate_run_path, validate_visible_input_path

        self.revoke()
        source = Path(validate_visible_input_path(spin_dir, label="spin directory"))
        manifest_path = source / "spin-manifest.json"
        if not manifest_path.is_file():
            raise ValueError(f"Spin manifest not found: {manifest_path}")
        manifest = json.loads(manifest_path.read_text())
        members = manifest["members"]
        if not 1 <= len(members) <= MAX_SPIN_MEMBERS:
            raise ValueError(f"Chat collection supports 1–{MAX_SPIN_MEMBERS} spin members")
        coupled = bool(manifest.get("is_coupled"))
        files = [manifest_path]
        for member in members:
            directory = Path(member["experiment_dir"])
            if directory.parent.resolve() != source.resolve():
                raise ValueError("Spin member escaped its directory")
            restarts = (
                [directory / "atmosphere" / "restart.nc", directory / "ocean" / "restart.nc"]
                if coupled
                else [directory / "restart.nc"]
            )
            for restart in restarts:
                if not restart.is_file():
                    raise ValueError(f"Spin restart not ready: {restart}")
                files.append(restart)
        if sum(p.stat().st_size for p in files) > MAX_REPLICATION_BYTES:
            raise ValueError("Collection exceeds chat size limit; use the CLI deliberately")
        output_dir = self._fresh_dir()
        output = validate_run_path(str(output_dir / "ic.nc"), label="collected IC output")
        card: dict[str, Any] = {
            "kind": "collect",
            "run_dir": str(output_dir),
            "spin_dir": str(source),
            "model": manifest["model"],
            "members": len(members),
            "target_time": manifest["target_time"],
            "calendar": manifest["calendar"],
            "output": output,
            "gpu_hours_estimate": 0,
        }
        if coupled:
            card["ocean_output"] = validate_run_path(
                str(output_dir / "ic_ocean.nc"), label="collected ocean IC output"
            )
        try:
            plan_file = output_dir / "collect-plan.json"
            plan_file.write_text(json.dumps(card))
            files.append(plan_file)
            digest = fingerprint(files)
            card["digest"] = digest
            card["approval_prefix"] = digest[:8]
            self._legacy_pending = {"card": card, "files": files, "digest": digest}
            return json.loads(json.dumps(card))
        except BaseException:
            shutil.rmtree(output_dir, ignore_errors=True)
            raise

    @staticmethod
    def _fresh_dir() -> Path:
        root = Path(resolve_configured_run_root())
        root.mkdir(parents=True, exist_ok=True)
        return Path(tempfile.mkdtemp(prefix="chat-", dir=root))

    def status(self, run: str) -> dict[str, Any]:
        """Inspect run status by run name or path."""
        return svc_status(run)

    def logs(self, run: str, max_bytes: int = 32768) -> dict[str, Any]:
        """Read bounded, control-character-stripped logs for a run."""
        bounded = min(max(64, int(max_bytes)), 65536)
        return svc_logs(run, follow=False, max_bytes=bounded)

    def ls(self, run_root: str | None = None) -> list[dict[str, Any]]:
        """List runs in the configured run root."""
        return svc_ls(run_root)

    def summary(self, run: str) -> dict[str, Any]:
        """Summarize area-weighted monthly output for a completed run."""
        return svc_summary(run).to_dict()

    def lora_info(self, ckpt: str) -> dict[str, Any]:
        """Inspect LoRA rank/alpha and Frobenius norms in a trusted checkpoint."""
        from ..lora import inspect_lora

        return inspect_lora(trusted_lora_path(ckpt)).to_dict()
