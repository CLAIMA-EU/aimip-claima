"""Stdio Model Context Protocol (`aimip mcp`) server backed by `aimip.services`.

Exposes ONLY allow-listed, input-validated service tools:
`models`, `show`, `plan`, `propose_run`, `submit_approved`, `status`, `logs`, `ls`, `cancel`, `summary`.
No arbitrary shell/command/path execution tools exist. Any state-changing action (`submit_approved`
or `cancel`) requires out-of-band human confirmation via `aimip approve <digest-prefix>` on `/dev/tty`,
stored as a `0600` caller-owned file in `$AIMIP_STATE/approvals/` and consumed atomically by rename.

The `mcp` Python SDK is optional (`pip install 'aimip[mcp]'`).
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence
import sys
import time
from typing import Any

from .agent.security import trusted_lora_path
from .agent.tools import MAX_MEMBERS
from .errors import AimipError, ApprovalError, ConfigError, ValidationError, enrich_error
from .services import (
    TERMINAL_RUN_STATES,
    RunSpec,
    cancel as svc_cancel,
    cancel_digest_for,
    consume_state_approval,
    delete_state_proposal,
    has_state_approval,
    logs as svc_logs,
    ls as svc_ls,
    models as svc_models,
    plan as svc_plan,
    prepare as svc_prepare,
    read_state_proposal,
    record_state_proposal,
    render_plan_card,
    resolve_run_target,
    show as svc_show,
    status as svc_status,
    submit as svc_submit,
    summary as svc_summary,
)
from .status import get_run_status
from .submit import check_trusted_run_dir

MCP_INSTALL_HINT = (
    "aimip mcp requires the optional MCP SDK; install it with: pip install 'aimip[mcp]'"
)


class McpToolset:
    """Pure-Python tool implementation shared with the `mcp` stdio transport and offline unit tests."""

    def __init__(
        self,
        *,
        state_dir: str | None = None,
        now_fn: Any = time.time,
    ) -> None:
        self.state_dir = state_dir
        self._now_fn = now_fn

    def models(
        self,
        task: str | None = None,
        family: str | None = None,
        status: str | None = None,
        installed: bool = False,
    ) -> list[dict[str, Any]]:
        """List registered models with optional task, family, status, and installed filters."""
        return svc_models(
            task=task,
            family=family,
            status=status,
            installed_only=installed,
        )

    def show(self, model: str, include_examples: bool = False) -> dict[str, Any]:
        """Return the declarative model card for *model*."""
        return svc_show(model, include_examples=include_examples)

    def plan(
        self,
        model: str,
        start: str,
        steps: int | None = None,
        length: str | None = None,
        end: str | None = None,
        members: int | None = None,
        seed: int | None = None,
        ic: str | None = None,
        ocean_ic: str | None = None,
        lora: str | None = None,
        forcing_set: str | None = None,
        name: str | None = None,
        out: str | None = None,
    ) -> dict[str, Any]:
        """Resolve and validate a run specification with zero side effects."""
        spec = self._validated_spec(
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
        return {
            "digest": resolved.digest,
            "short_digest": resolved.short_digest,
            "card": render_plan_card(resolved, is_tty=False),
            "plan": resolved.to_dict(),
        }

    def propose_run(
        self,
        model: str,
        start: str,
        steps: int | None = None,
        length: str | None = None,
        end: str | None = None,
        members: int | None = None,
        seed: int | None = None,
        ic: str | None = None,
        ocean_ic: str | None = None,
        lora: str | None = None,
        forcing_set: str | None = None,
        name: str | None = None,
        out: str | None = None,
    ) -> dict[str, Any]:
        """Record a run proposal in `$AIMIP_STATE/proposals/` and return the human approval command."""
        spec = self._validated_spec(
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
        card_text = render_plan_card(resolved, is_tty=False)
        req_action = resolved.required_approval_action
        req_data_write = any(not a.prospective for a in resolved.missing_assets) if resolved.missing_assets else True
        record_state_proposal(
            action=req_action,
            digest=resolved.digest,
            card_text=card_text,
            payload={"spec": spec.to_dict()},
            missing_bytes=resolved.missing_bytes,
            missing_count=len(resolved.missing_assets),
            requires_data_dir_write=req_data_write,
            state_dir=self.state_dir,
            now_fn=self._now_fn,
        )
        approve_cmd = f"aimip approve {resolved.approval_prefix}"
        return {
            "action": req_action,
            "requires_staging": resolved.requires_staging,
            "run_dir": resolved.output_dir,
            "digest": resolved.digest,
            "short_digest": resolved.short_digest,
            "approval_prefix": resolved.approval_prefix,
            "card": card_text,
            "missing_bytes": resolved.missing_bytes,
            "missing_count": len(resolved.missing_assets),
            "approve_command": approve_cmd,
            "message": (
                f"Run `{approve_cmd}` in your terminal to approve preparation/staging, then call "
                f"`submit_approved` with digest={resolved.digest!r} to stage inputs and obtain the final submit digest."
                if resolved.requires_staging
                else f"Run `{approve_cmd}` in your terminal, then call `submit_approved` with digest={resolved.digest!r}."
            ),
        }

    def submit_approved(self, digest: str) -> dict[str, Any]:
        """Consume a single-use human approval created by `aimip approve <prefix>`, enforce prepare vs submit action, and execute.

        - Step 1 (`proposal.action == 'prepare'` for staging models): consumes the prepare-only approval,
          stages/prepares the run directory (NEVER submitting), records the final `action='submit'` proposal
          bound to the staged bytes, and returns the final card + `approve_command`.
        - Step 2 (`proposal.action == 'submit'`): consumes the submit approval and submits the verified run.
        """
        proposal = read_state_proposal(digest, state_dir=self.state_dir)
        prop_action = proposal.get("action")
        if prop_action not in ("prepare", "submit"):
            raise ApprovalError(
                f"Proposal {digest[:12]} is for action {prop_action!r}, not 'prepare' or 'submit'.",
                field="digest",
            )
        payload = proposal.get("payload") or {}
        raw_spec = payload.get("spec")
        if not isinstance(raw_spec, dict):
            raise ApprovalError("Stored proposal is missing a valid RunSpec.", field="digest")
        spec = RunSpec.from_dict(raw_spec)

        if prop_action == "prepare":
            auth = consume_state_approval(
                digest,
                expected_action="prepare",
                state_dir=self.state_dir,
                now_fn=self._now_fn,
            )
            delete_state_proposal(digest, state_dir=self.state_dir)
            prepared = svc_prepare(
                spec,
                authorization=auth,
                require_authorization=True,
            )
            post_card = render_plan_card(prepared.plan, is_tty=False)
            record_state_proposal(
                action="submit",
                digest=prepared.plan.digest,
                card_text=post_card,
                payload={
                    "spec": prepared.plan.spec.to_dict(),
                    "prepared_run_dir": prepared.run_dir,
                    "previous_prepare_digest": digest.strip().lower(),
                },
                missing_bytes=0,
                missing_count=0,
                requires_data_dir_write=False,
                state_dir=self.state_dir,
                now_fn=self._now_fn,
            )
            post_cmd = f"aimip approve {prepared.plan.approval_prefix}"
            return {
                "state": "PREPARED_AWAITING_SUBMIT_APPROVAL",
                "submitted": False,
                "action": "submit",
                "run_dir": prepared.run_dir,
                "run_name": prepared.plan.run_name,
                "digest": prepared.plan.digest,
                "short_digest": prepared.plan.short_digest,
                "approval_prefix": prepared.plan.approval_prefix,
                "card": post_card,
                "approve_command": post_cmd,
                "message": (
                    f"Inputs staged and hashed into final digest {prepared.plan.short_digest}. "
                    f"Run `{post_cmd}` in your terminal, then call `submit_approved` with digest={prepared.plan.digest!r}."
                ),
            }

        auth = consume_state_approval(
            digest,
            expected_action="submit",
            state_dir=self.state_dir,
            now_fn=self._now_fn,
        )
        delete_state_proposal(digest, state_dir=self.state_dir)

        prepared_run_dir = payload.get("prepared_run_dir")
        if isinstance(prepared_run_dir, str) and prepared_run_dir:
            attempt = svc_submit(
                prepared_run_dir,
                authorization=auth,
                require_authorization=True,
            )
            return attempt.to_dict()

        prepared = svc_prepare(
            spec,
            authorization=auth,
            require_authorization=True,
        )
        attempt = svc_submit(
            prepared,
            authorization=auth,
            require_authorization=True,
        )
        return attempt.to_dict()

    def status(self, run: str) -> dict[str, Any]:
        """Inspect status of a run by name or path."""
        return svc_status(run)

    def logs(self, run: str, max_bytes: int = 32768) -> dict[str, Any]:
        """Return bounded, control-character-stripped logs for *run*."""
        bounded = min(max(64, int(max_bytes)), 65536)
        return svc_logs(run, follow=False, max_bytes=bounded)

    def ls(self, run_root: str | None = None) -> list[dict[str, Any]]:
        """List runs in *run_root* (or the configured site `run_root`)."""
        return svc_ls(run_root)

    def cancel(self, run: str) -> dict[str, Any]:
        """Cancel a running Slurm or local job ONLY after human approval via `aimip approve <prefix>`.

        If no approval exists yet for this `(run_dir, attempt_id)`, writes a cancel proposal and
        returns an `ApprovalError` instructing the user to run `aimip approve <prefix>` in their terminal.
        """
        run_dir, run_name = resolve_run_target(run)
        trusted = check_trusted_run_dir(run_dir)
        snap = get_run_status(trusted)
        job_id = snap.get("slurm_job_id") or snap.get("job_id")
        local_pid = snap.get("local_pid")
        if not job_id and not local_pid:
            raise ConfigError(
                f"Run {run_name!r} ({trusted}) has no active Slurm job ID or local PID to cancel.",
                field="run",
            )
        cur_state = str(snap.get("state") or "unknown").strip()
        if (cur_state.split() or [""])[0].rstrip("+").upper() in TERMINAL_RUN_STATES or cur_state.upper() in TERMINAL_RUN_STATES:
            raise ConfigError(
                f"Run {run_name!r} ({trusted}) is already in terminal state {cur_state!r} and cannot be cancelled.",
                field="run",
            )
        attempt_id = str(job_id) if job_id else str(local_pid)
        digest = cancel_digest_for(trusted, attempt_id)

        if not has_state_approval(digest, state_dir=self.state_dir):
            card_text = (
                f"CANCEL {run_name}  [digest {digest[:12]}]\n"
                f"  run_dir:     {trusted}\n"
                f"  attempt_id:  {attempt_id}\n"
                f"  model:       {snap.get('model')}\n"
                f"  digest:      {digest[:12]}  (approval prefix: {digest[:8]})"
            )
            record_state_proposal(
                action="cancel",
                digest=digest,
                card_text=card_text,
                payload={"run_dir": trusted, "attempt_id": attempt_id},
                state_dir=self.state_dir,
                now_fn=self._now_fn,
            )
            raise ApprovalError(
                f"Cancelling {run_name!r} (attempt {attempt_id}) requires human approval: "
                f"run `aimip approve {digest[:8]}` in your terminal, then call `cancel` again.",
                field="authorization",
                next_command=f"aimip approve {digest[:8]}",
            )

        auth = consume_state_approval(
            digest,
            expected_action="cancel",
            state_dir=self.state_dir,
            now_fn=self._now_fn,
        )
        return svc_cancel(
            trusted,
            authorization=auth,
            require_authorization=True,
        )

    def summary(self, run: str) -> dict[str, Any]:
        """Return area-weighted global-mean monthly summary series for *run*."""
        return svc_summary(run).to_dict()

    @staticmethod
    def _validated_spec(
        *,
        model: str,
        start: str,
        steps: int | None,
        length: str | None,
        end: str | None,
        members: int | None,
        seed: int | None,
        ic: str | None,
        ocean_ic: str | None,
        lora: str | None,
        forcing_set: str | None,
        name: str | None,
        out: str | None,
    ) -> RunSpec:
        if members is not None and not 1 <= int(members) <= MAX_MEMBERS:
            raise ValidationError(
                f"MCP plans support between 1 and {MAX_MEMBERS} members.",
                field="members",
            )
        checked_lora: str | None = None
        if lora is not None:
            try:
                checked_lora = trusted_lora_path(lora)
            except (OSError, ValueError) as exc:
                raise ValidationError(str(exc), field="lora") from exc
        return RunSpec(
            model=model,
            start=start,
            steps=steps,
            length=length,
            end=end,
            members=members,
            seed=seed,
            ic=ic,
            ocean_ic=ocean_ic,
            lora=checked_lora,
            forcing_set=forcing_set,
            name=name,
            out=out,
        )


def _wrap_mcp_call(fn: Any) -> Any:
    """Wrap a `McpToolset` method so `AimipError` serializes cleanly as structured JSON for MCP clients."""
    import functools

    @functools.wraps(fn)
    def _wrapped(*args: Any, **kwargs: Any) -> dict[str, Any]:
        try:
            res = fn(*args, **kwargs)
            return {"aimip": "1", "ok": True, "result": res}
        except AimipError as exc:
            enrich_error(exc)
            return {"aimip": "1", "ok": False, "error": exc.to_dict()}

    return _wrapped


def build_fastmcp_server(toolset: McpToolset | None = None) -> Any:
    """Build a `FastMCP` server instance with the 10 allow-listed aimip-claima tools."""
    from mcp.server.fastmcp import FastMCP

    ts = toolset or McpToolset()
    server = FastMCP("aimip")
    for method in (
        ts.models,
        ts.show,
        ts.plan,
        ts.propose_run,
        ts.submit_approved,
        ts.status,
        ts.logs,
        ts.ls,
        ts.cancel,
        ts.summary,
    ):
        server.tool(name=method.__name__)(_wrap_mcp_call(method))
    return server


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entrypoint for `aimip mcp` (stdio transport)."""
    parser = argparse.ArgumentParser(
        prog="aimip mcp",
        description="Start the aimip-claima Model Context Protocol (MCP) server over stdio.",
    )
    parser.add_argument(
        "--state-dir",
        dest="state_dir",
        default=None,
        help="Optional override for $AIMIP_STATE proposal/approval directory.",
    )
    args = parser.parse_args(list(argv) if argv is not None else [])

    try:
        server = build_fastmcp_server(McpToolset(state_dir=args.state_dir))
    except ImportError:
        print(MCP_INSTALL_HINT, file=sys.stderr)
        return 2

    server.run(transport="stdio")
    return 0
