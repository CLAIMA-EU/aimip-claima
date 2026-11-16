"""Phase 2a tests for `aimip mcp` (stdio MCP server) and `aimip approve <digest-prefix>`:
- Optional `mcp` SDK import handling (`pip install 'aimip[mcp]'`)
- Allow-listed tools (`models`, `show`, `plan`, `propose_run`, `submit_approved`, `status`, `logs`, `ls`, `cancel`, `summary`)
- End-to-end `propose_run` -> `aimip approve <8-hex>` on real PTY -> `submit_approved`
- Out-of-band approval required for `cancel` as well
- `$AIMIP_STATE/approvals/` tamper resistance: wrong mode (`0644`), wrong owner (`st_uid`), symlinked file, symlinked dir, replay after atomic rename, expiry, and plan drift
"""

from __future__ import annotations

import builtins
import json
import os
from pathlib import Path
import sys
import types

import pytest

from aimip.cli import main
from aimip.errors import ApprovalError
from aimip.mcp_server import MCP_INSTALL_HINT, McpToolset, build_fastmcp_server
from aimip.mcp_server import main as mcp_main
from conftest import parse_cli_json, pty_responder, write_fake_ic_netcdf


@pytest.fixture
def mcp_env(playground_env, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict[str, Path]:
    fbin = tmp_path / "mcp_slurm_bin"
    fbin.mkdir()
    scancel_log = tmp_path / "mcp_scancel.log"

    (fbin / "sbatch").write_text("#!/bin/bash\necho 909090\n", encoding="utf-8")
    (fbin / "sbatch").chmod(0o755)
    (fbin / "squeue").write_text("#!/bin/bash\nexit 0\n", encoding="utf-8")
    (fbin / "squeue").chmod(0o755)
    (fbin / "sacct").write_text("#!/bin/bash\necho '909090|RUNNING|00:00:30|0:0'\n", encoding="utf-8")
    (fbin / "sacct").chmod(0o755)
    (fbin / "scancel").write_text(f'#!/bin/bash\necho "$@" >> "{scancel_log}"\n', encoding="utf-8")
    (fbin / "scancel").chmod(0o755)

    state_dir = tmp_path / "aimip_state"
    monkeypatch.setenv("PATH", f"{fbin}:{os.environ.get('PATH', '')}")
    monkeypatch.setenv("AIMIP_STATE", str(state_dir))
    return {"state_dir": state_dir, "scancel_log": scancel_log}


def test_mcp_without_sdk_prints_install_hint_and_with_fake_sdk_binds_exact_ten_tools(
    mcp_env, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    orig_import = builtins.__import__

    def _block_mcp(name, *args, **kwargs):
        if name == "mcp" or name.startswith("mcp."):
            raise ImportError("No module named 'mcp'")
        return orig_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", _block_mcp)
    assert main(["mcp"]) == 2
    err = capsys.readouterr().err
    assert "pip install 'aimip[mcp]'" in err and err.strip() == MCP_INSTALL_HINT
    monkeypatch.setattr(builtins, "__import__", orig_import)

    # Install tiny in-memory fake `mcp.server.fastmcp` module and verify exact tool allowlist + stdio transport
    registered: dict[str, object] = {}
    transports: list[str] = []

    class FakeFastMCP:
        def __init__(self, name: str) -> None:
            self.name = name

        def tool(self, *, name: str):
            def _dec(fn):
                registered[name] = fn
                return fn

            return _dec

        def run(self, *, transport: str = "stdio") -> None:
            transports.append(transport)

    mcp_pkg = types.ModuleType("mcp")
    mcp_srv = types.ModuleType("mcp.server")
    mcp_fast = types.ModuleType("mcp.server.fastmcp")
    mcp_fast.FastMCP = FakeFastMCP  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "mcp", mcp_pkg)
    monkeypatch.setitem(sys.modules, "mcp.server", mcp_srv)
    monkeypatch.setitem(sys.modules, "mcp.server.fastmcp", mcp_fast)

    assert mcp_main(["--state-dir", str(mcp_env["state_dir"])]) == 0
    assert transports == ["stdio"]
    assert set(registered.keys()) == {
        "models",
        "show",
        "plan",
        "propose_run",
        "submit_approved",
        "status",
        "logs",
        "ls",
        "cancel",
        "summary",
    }
    # Verify wrapped MCP error envelope
    bad_resp = registered["show"]("unknown-model")  # type: ignore[operator]
    assert bad_resp["aimip"] == "1" and bad_resp["ok"] is False
    assert bad_resp["error"]["field"] == "model"


def test_mcp_propose_approve_submit_and_cancel_workflow(
    mcp_env, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    ts = McpToolset(state_dir=str(mcp_env["state_dir"]))
    assert len(ts.models(task="amip")) == 3
    assert ts.show("ace2-era5")["status"] == "certified"
    assert ts.plan("ace2-era5", "1979-01-01T00:00:00", length="5d", name="mcp-preview")["plan"]["steps"] == 20

    prop = ts.propose_run("ace2-era5", "1979-01-01T00:00:00", steps=4, name="mcp-job")
    digest = prop["digest"]
    prefix = prop["approval_prefix"]
    assert f"aimip approve {prefix}" in prop["message"]

    # Calling `submit_approved` BEFORE `aimip approve` is refused!
    with pytest.raises(ApprovalError, match="No human approval found"):
        ts.submit_approved(digest)

    # Human runs `aimip approve <prefix>` on a real controlling PTY
    with pty_responder(prefix) as (tty_path, _):
        monkeypatch.setenv("AIMIP_TTY_PATH", tty_path)
        assert main(["approve", prefix, "--json"]) == 0
    app_out = parse_cli_json(capsys.readouterr().out)
    assert app_out["approved"] is True and app_out["digest"] == digest

    # Now `submit_approved` consumes the `0600` approval via atomic rename, prepares, and submits!
    attempt = ts.submit_approved(digest)
    assert attempt["attempt_id"] == "909090"
    assert ts.status("mcp-job")["slurm_job_id"] == "909090"
    assert any(r["name"] == "mcp-job" for r in ts.ls())

    # Replay of `submit_approved` AND re-running `aimip approve <prefix>` both fail because both approval and proposal were consumed!
    with pytest.raises(ApprovalError, match=r"No human approval or active proposal found.*propose_run"):
        ts.submit_approved(digest)
    with pty_responder(prefix) as (tty_path, _):
        monkeypatch.setenv("AIMIP_TTY_PATH", tty_path)
        assert main(["approve", prefix]) == 4
    capsys.readouterr()

    # `cancel` ALSO requires `aimip approve <prefix>`!
    with pytest.raises(ApprovalError, match="aimip approve") as cancel_exc:
        ts.cancel("mcp-job")
    cancel_cmd = cancel_exc.value.next_command
    assert cancel_cmd and cancel_cmd.startswith("aimip approve ")
    cancel_prefix = cancel_cmd.split()[-1]

    with pty_responder(cancel_prefix) as (tty_path, _):
        monkeypatch.setenv("AIMIP_TTY_PATH", tty_path)
        assert main(["approve", cancel_prefix]) == 0
    capsys.readouterr()

    cancelled = ts.cancel("mcp-job")
    assert cancelled["cancelled"] is True and cancelled["attempt_id"] == "909090"
    assert "909090" in mcp_env["scancel_log"].read_text(encoding="utf-8")


def test_mcp_approval_file_tamper_mode_owner_symlink_expiry_and_drift(
    playground_env, mcp_env, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    clock = [5000.0]
    state_dir: Path = mcp_env["state_dir"]
    ts = McpToolset(state_dir=str(state_dir), now_fn=lambda: clock[0])

    def _propose_and_approve(run_name: str, *, ic: str | None = None) -> tuple[str, Path]:
        p = ts.propose_run("ace2-era5", "1979-01-01T00:00:00", steps=4, name=run_name, ic=ic)
        with pty_responder(p["approval_prefix"]) as (tty_path, _):
            from aimip.services import approve_state_proposal

            approve_state_proposal(
                p["approval_prefix"],
                state_dir=str(state_dir),
                tty_path=tty_path,
                now_fn=lambda: clock[0],
            )
        return p["digest"], state_dir / "approvals" / f"{p['digest']}.json"

    # 1. File mode tampered (`0644` instead of `0600`) -> refused!
    d_mode, f_mode = _propose_and_approve("tamper-mode")
    f_mode.chmod(0o644)
    with pytest.raises(ApprovalError, match="insecure mode"):
        ts.submit_approved(d_mode)
    f_mode.unlink()

    # 2. Wrong file owner (`st_uid != os.geteuid()`) -> refused!
    d_uid, f_uid = _propose_and_approve("tamper-uid")
    real_geteuid = os.geteuid
    with monkeypatch.context() as mp:
        mp.setattr(os, "geteuid", lambda: real_geteuid() + 10000)
        with pytest.raises(ApprovalError, match="not owned by the current user"):
            ts.submit_approved(d_uid)
    f_uid.unlink()

    # 3. Symlinked approval file -> refused!
    d_sym, f_sym = _propose_and_approve("tamper-sym")
    moved = tmp_path / "stolen_approval.json"
    f_sym.rename(moved)
    moved.chmod(0o600)
    f_sym.symlink_to(moved)
    with pytest.raises(ApprovalError, match="symlinked approval file"):
        ts.submit_approved(d_sym)
    f_sym.unlink()

    # 4. Expired approval (> 600s) -> refused!
    d_exp, _ = _propose_and_approve("tamper-exp")
    clock[0] += 605.0
    with pytest.raises(ApprovalError, match="expired"):
        ts.submit_approved(d_exp)

    # 5. Plan changed between `aimip approve` and `submit_approved` -> refused!
    custom_ic = write_fake_ic_netcdf(
        playground_env["runs_dir"] / "mcp_drift.nc",
        ["1979-01-01T00:00:00"],
        spatial_dims=("latitude", "longitude"),
        seed=1,
    )
    d_drift, _ = _propose_and_approve("tamper-drift", ic=str(custom_ic))
    write_fake_ic_netcdf(
        custom_ic,
        ["1979-01-01T00:00:00"],
        spatial_dims=("latitude", "longitude"),
        seed=2,
    )
    with pytest.raises(ApprovalError, match="Plan changed, approve again"):
        ts.submit_approved(d_drift)
