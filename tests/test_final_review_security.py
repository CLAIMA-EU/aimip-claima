"""Regression tests for final public-main security, execution-gate, metadata, and CLI parity fixes."""

from __future__ import annotations

import errno
import hashlib
import importlib
import json
import os
from pathlib import Path
import subprocess
import sys
import textwrap
import time
import tomllib

import pytest

from aimip import emu_job, services as svc
from aimip.cli import build_parser, main
from aimip.completion import render_completion
from aimip.errors import (
    EXIT_APPROVAL,
    EXIT_UNSTABLE,
    ApprovalError,
    ConfigError,
    ExecutionError,
    ValidationError,
)
from aimip.families import _hash_custom_forcing_dir_assets
from aimip.mcp_server import McpToolset, build_fastmcp_server
from aimip.services import approval
from aimip.services.metadata import MODEL_CARDS, TIER_DEFINITIONS
from aimip.services.types import Authorization, RunSpec
from aimip.status import get_run_status
from aimip.submit import read_pid_start_id
from conftest import parse_cli_json, pty_responder, write_fake_ic_netcdf


# ---------------------------------------------------------------------------
# 1. Read-only sealed-state and proposal verification (R1)
# ---------------------------------------------------------------------------


def test_read_prepared_seal_needs_no_metadata_write(tmp_path, monkeypatch):
    state = tmp_path.resolve() / "state"
    run = tmp_path.resolve() / "run"
    run.mkdir()
    digest = "a" * 64
    approval.record_sealed_prepared(
        digest=digest,
        run_dir=str(run),
        artifact_hashes={"job.sbatch": "b" * 64},
        state_dir=str(state),
    )
    real_chmod = os.chmod

    def readonly_chmod(path, mode, *args, **kwargs):
        if not isinstance(path, int) and Path(path).is_relative_to(state):
            raise OSError(errno.EROFS, "read-only verification mount", str(path))
        return real_chmod(path, mode, *args, **kwargs)

    # A read-only bind rejects chmod even if the requested mode already matches.
    monkeypatch.setattr(os, "chmod", readonly_chmod)
    doc = approval.read_sealed_prepared(digest, run_dir=str(run), state_dir=str(state))
    assert doc["digest"] == digest


def test_readonly_seal_and_proposal_readers_reject_loose_modes_and_symlinks_without_chmod(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    base = tmp_path.resolve()
    state = base / "state"
    run = base / "run"
    run.mkdir()
    digest = "c" * 64
    prop_digest = "d" * 64

    approval.record_sealed_prepared(
        digest=digest,
        run_dir=str(run),
        artifact_hashes={"job.sbatch": "e" * 64},
        state_dir=str(state),
    )
    approval.record_state_proposal(
        action="submit",
        digest=prop_digest,
        card_text="PLAN demo",
        payload={"spec": {"model": "ace2-era5", "start": "1979-01-01T00:00:00", "steps": 4}},
        state_dir=str(state),
    )

    real_chmod = os.chmod
    real_fchmod = getattr(os, "fchmod", None)
    real_mkdir = Path.mkdir

    def forbid_state_chmod(path, mode, *args, **kwargs):
        if isinstance(path, int) or Path(path).is_relative_to(state):
            raise OSError(errno.EROFS, "read-only verification mount", str(path))
        return real_chmod(path, mode, *args, **kwargs)

    def forbid_state_fchmod(fd, mode):
        raise OSError(errno.EROFS, "read-only verification mount")

    def forbid_state_mkdir(self_path: Path, *args, **kwargs):
        if self_path == state or self_path.is_relative_to(state):
            raise OSError(errno.EROFS, "read-only verification mount", str(self_path))
        return real_mkdir(self_path, *args, **kwargs)

    with monkeypatch.context() as mp:
        mp.setattr(os, "chmod", forbid_state_chmod)
        if real_fchmod is not None:
            mp.setattr(os, "fchmod", forbid_state_fchmod)
        mp.setattr(Path, "mkdir", forbid_state_mkdir)

        assert approval.read_sealed_prepared(digest, run_dir=str(run), state_dir=str(state))["digest"] == digest
        assert approval.read_state_proposal(prop_digest, state_dir=str(state))["digest"] == prop_digest
        assert approval.read_sealed_prepared("f" * 64, run_dir=str(run), state_dir=str(state)) is None

    # Loose 0755 mode on `prepared` must fail closed under read-only mount without attempting fchmod.
    real_chmod(state / "prepared", 0o755)
    with monkeypatch.context() as mp:
        mp.setattr(os, "chmod", forbid_state_chmod)
        if real_fchmod is not None:
            mp.setattr(os, "fchmod", forbid_state_fchmod)
        mp.setattr(Path, "mkdir", forbid_state_mkdir)
        with pytest.raises(ApprovalError, match="0700"):
            approval.read_sealed_prepared(digest, run_dir=str(run), state_dir=str(state))
    real_chmod(state / "prepared", 0o700)

    # Loose 0644 seal file must also fail closed.
    seal_file = state / "prepared" / f"{digest}.json"
    real_chmod(seal_file, 0o644)
    with pytest.raises(ApprovalError, match="0o?600"):
        approval.read_sealed_prepared(digest, run_dir=str(run), state_dir=str(state))
    real_chmod(seal_file, 0o600)

    # Symlinked `base_state` is rejected on read.
    sym_state = base / "state_link"
    sym_state.symlink_to(state)
    with pytest.raises(ApprovalError, match="symlink"):
        approval.read_sealed_prepared(digest, run_dir=str(run), state_dir=str(sym_state))


# ---------------------------------------------------------------------------
# 2. Native submission private-script verification inside `.submit.lock` (R2)
# ---------------------------------------------------------------------------


def test_native_submit_verifies_the_private_script_not_an_earlier_source(playground_env, monkeypatch):
    gate = importlib.import_module("aimip.submit")
    prepared = svc.prepare(
        RunSpec(
            model="ace2-era5",
            start="1979-01-01T00:00:00",
            steps=4,
            name="private-copy-race",
            resources={"time": "00:10:00"},
        )
    )
    broker = approval.MemoryApprovalBroker()
    now = time.time()
    auth = Authorization(
        action="submit",
        digest=prepared.plan.digest,
        user=approval.current_user(),
        issued_at=now,
        expires_at=now + 600,
        nonce="private-copy-race",
    )
    broker.store_minted(auth)
    real_lock = gate._acquire_run_submit_lock
    submitted = []

    def replace_while_entering_gate(run_dir):
        fd = real_lock(run_dir)
        script = Path(prepared.sbatch_path)
        before = script.read_text()
        assert "--time=00:10:00" in before
        script.write_text(before.replace("--time=00:10:00", "--time=00:20:00"))
        return fd

    def fake_scheduler(argv, **kwargs):
        assert Path(argv[0]).name == "sbatch", argv
        submitted.append(Path(argv[-1]).read_text())
        return subprocess.CompletedProcess(argv, 0, stdout="424242\n", stderr="")

    monkeypatch.setenv("AIMIP_IMAGE_SHA", "native")
    monkeypatch.setattr(gate, "_acquire_run_submit_lock", replace_while_entering_gate)
    monkeypatch.setattr(gate.subprocess, "run", fake_scheduler)
    with pytest.raises(ApprovalError):
        svc.submit(prepared, authorization=auth, require_authorization=True, broker=broker)
    assert not submitted


def test_prepare_serializes_with_submit_lock_on_existing_run_dir(playground_env, monkeypatch) -> None:
    gate = importlib.import_module("aimip.submit")
    lock_calls: list[str] = []
    real_lock = gate._acquire_run_submit_lock

    def spy_lock(run_dir: str) -> int:
        lock_calls.append(run_dir)
        return real_lock(run_dir)

    monkeypatch.setattr(gate, "_acquire_run_submit_lock", spy_lock)
    prepared = svc.prepare(
        RunSpec(model="ace2-era5", start="1979-01-01T00:00:00", steps=4, name="locked-prep")
    )
    assert prepared.run_dir in lock_calls


# ---------------------------------------------------------------------------
# 3. Container-local detached submission routes to host wrapper (R3)
# ---------------------------------------------------------------------------


def test_container_local_submission_returns_host_command_without_popen_across_surfaces(
    playground_env, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from aimip.site_config import init_site_config
    from aimip.ui.server import UiApp, _plan_decision_context, _ui_runtime_capabilities

    local_site = tmp_path / "local_container_site.json"
    init_site_config(
        preset="local",
        out_path=str(local_site),
        runtime="docker",
        run_root=str(playground_env["default_out_root"]),
        overwrite=True,
    )
    monkeypatch.setenv("AIMIP_SITE_CONFIG", str(local_site))
    monkeypatch.setenv("AIMIP_SCHEDULER", "none")
    monkeypatch.setenv("AIMIP_IMAGE_SHA", "b" * 64)

    popen_calls: list[object] = []

    def forbid_popen(*args, **kwargs):
        popen_calls.append((args, kwargs))
        raise AssertionError("Popen must not be called inside a container for detached local submission")

    monkeypatch.setattr(subprocess, "Popen", forbid_popen)

    # 1. Direct `services.submit` on `scheduler="none"` in container returns `PREPARED_AWAITING_HOST_SUBMIT`.
    prep = svc.prepare(
        RunSpec(model="ace2-era5", start="1979-01-01T00:00:00", steps=4, name="cnt-local-svc", mode="local")
    )
    att = svc.submit(prep)
    assert att.state == "PREPARED_AWAITING_HOST_SUBMIT"
    assert att.scheduler == "none"
    assert att.attempt_id is None
    assert att.host_command is not None
    assert f"--expect-digest={prep.plan.digest}" in att.host_command
    assert prep.run_dir in att.host_command
    assert not (Path(prep.run_dir) / "local.pid").exists()
    assert popen_calls == []

    # 2. UI runtime capabilities & plan decision context accurately report prepare-only in container-local mode.
    caps = _ui_runtime_capabilities()
    assert caps["in_container"] is True
    assert caps["scheduler"] == "none"
    assert caps["can_submit_directly"] is False

    ctx = _plan_decision_context(prep.plan)
    assert ctx["will_prepare_only_in_container"] is True

    app = UiApp(token="tok-r3")
    ui_spec = RunSpec(
        model="ace2-era5", start="1979-01-01T00:00:00", steps=4, name="cnt-local-ui", mode="local"
    )
    ui_resolved = svc.plan(ui_spec)
    prop = app.approvals.register_submit_proposal(
        spec=ui_spec,
        resolved=ui_resolved,
        card_text=svc.render_plan_card(ui_resolved, is_tty=False),
    )
    assert prop["will_prepare_only_in_container"] is True
    minted = app.approvals.approve(
        digest_prefix=prop["approval_prefix"],
        expected_digest=prop["digest"],
        action="submit",
    )
    ui_att = app.approvals.execute_submit(nonce=minted["authorization_nonce"])
    assert ui_att["state"] == "PREPARED_AWAITING_HOST_SUBMIT"
    assert ui_att["scheduler"] == "none"
    assert f"--expect-digest={prop['digest']}" in ui_att["host_command"]
    assert popen_calls == []

    # 3. TTY approval flow (`aimip run --approve` / `aimip chat`) also returns `PREPARED_AWAITING_HOST_SUBMIT`.
    tty_spec = RunSpec(
        model="ace2-era5", start="1979-01-01T00:00:00", steps=4, name="cnt-local-tty", mode="local"
    )
    tty_plan = svc.plan(tty_spec)
    with pty_responder(tty_plan.approval_prefix) as (tty_path, _):
        tty_att = svc.execute_tty_plan_flow(tty_spec, tty_path=tty_path)
    assert tty_att.state == "PREPARED_AWAITING_HOST_SUBMIT"
    assert tty_att.scheduler == "none"
    assert tty_att.host_command and f"--expect-digest={tty_plan.digest}" in tty_att.host_command
    assert popen_calls == []

    # 4. MCP `propose_run` -> `approve` -> `submit_approved` returns `PREPARED_AWAITING_HOST_SUBMIT`.
    mcp_state = tmp_path / "mcp_r3_state"
    ts = McpToolset(state_dir=str(mcp_state))
    mcp_prop = ts.propose_run("ace2-era5", "1979-01-01T00:00:00", steps=4, name="cnt-local-mcp")
    with pty_responder(mcp_prop["approval_prefix"]) as (tty_path, _):
        monkeypatch.setenv("AIMIP_TTY_PATH", tty_path)
        assert main(["approve", mcp_prop["approval_prefix"], "--state-dir", str(mcp_state), "--json"]) == 0
    mcp_att = ts.submit_approved(mcp_prop["digest"])
    assert mcp_att["state"] == "PREPARED_AWAITING_HOST_SUBMIT"
    assert mcp_att["scheduler"] == "none"
    assert f"--expect-digest={mcp_prop['digest']}" in mcp_att["host_command"]
    assert popen_calls == []

    # 5. Container-mode `cancel` on a Slurm run without `scancel` or local PID carries `next_command` on the host.
    host_local_dir = Path(prep.run_dir)
    (host_local_dir / "local.pid").write_text("99999\n", encoding="utf-8")
    with pytest.raises(ExecutionError, match="aimip cancel") as cancel_exc:
        svc.cancel("cnt-local-svc")
    assert cancel_exc.value.next_command == "aimip cancel cnt-local-svc"


# ---------------------------------------------------------------------------
# 4. Forcing byte binding and pre-execution verification (R4)
# ---------------------------------------------------------------------------


def test_large_custom_forcing_binds_bytes_even_if_size_and_mtime_are_preserved(tmp_path):
    directory = tmp_path.resolve() / "forcing"
    directory.mkdir()
    forcing = directory / "forcing.nc"
    # Sparse fixture: do not allocate a 129-MiB Python buffer or load NetCDF.
    with forcing.open("wb") as stream:
        stream.seek(128 * 1024 * 1024)
        stream.write(b"A")
    previous_stat = forcing.stat()
    before = _hash_custom_forcing_dir_assets(str(directory), "*.nc", "forcing", set())
    with forcing.open("r+b") as stream:
        stream.seek(128 * 1024 * 1024)
        stream.write(b"B")
    os.utime(forcing, ns=(previous_stat.st_atime_ns, previous_stat.st_mtime_ns))
    after = _hash_custom_forcing_dir_assets(str(directory), "*.nc", "forcing", set())
    assert before[0].sha256 != after[0].sha256
    assert len(before[0].sha256) == 64 and not before[0].sha256.startswith("stat:")


def test_packaged_forcing_corruption_with_unchanged_lock_is_refused_before_submit(
    tmp_path: Path, allow_tmp_run_root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    data_dir = tmp_path / "pkg_data"
    ckpt = data_dir / "models" / "ace2-era5" / "ace2_era5_ckpt.tar"
    ckpt.parent.mkdir(parents=True)
    ckpt.write_bytes(b"pinned-ckpt-bytes")
    ckpt_sha = hashlib.sha256(ckpt.read_bytes()).hexdigest()

    ic_1979 = write_fake_ic_netcdf(
        data_dir / "inputs" / "ace2-era5" / "ic" / "ic_1979.nc",
        ["1979-01-01T00:00:00"],
        calendar="proleptic_gregorian",
    )
    ic_sha = hashlib.sha256(ic_1979.read_bytes()).hexdigest()

    fdir = data_dir / "inputs" / "ace2-era5" / "amip-hist"
    fdir.mkdir(parents=True)
    f1979 = fdir / "forcing_1979.nc"
    f1979.write_bytes(b"PACKAGED_FORCING_ORIGINAL_BYTES_1979")
    f1979_sha = hashlib.sha256(f1979.read_bytes()).hexdigest()

    lock_doc = {
        "schema_version": 2,
        "files": {
            "models/ace2-era5/ace2_era5_ckpt.tar": {
                "size": ckpt.stat().st_size,
                "sha256": ckpt_sha,
                "source": {"kind": "hf", "repo": "allenai/ACE2-ERA5", "path": "ace2_era5_ckpt.tar", "revision": "main"},
            },
            "inputs/ace2-era5/ic/ic_1979.nc": {
                "size": ic_1979.stat().st_size,
                "sha256": ic_sha,
                "source": {"kind": "hf", "repo": "allenai/ACE2-ERA5", "path": "ic_1979.nc", "revision": "main"},
            },
            "inputs/ace2-era5/amip-hist/forcing_1979.nc": {
                "size": f1979.stat().st_size,
                "sha256": f1979_sha,
                "source": {"kind": "hf", "repo": "allenai/ACE2-ERA5", "path": "forcing_1979.nc", "revision": "main"},
            },
        },
    }
    lock_path = tmp_path / "pkg.lock"
    lock_path.write_text(json.dumps(lock_doc, indent=2), encoding="utf-8")

    reg_doc = {
        "schema_version": 3,
        "models": [
            {
                "name": "ace2-era5",
                "family": "fme",
                "fme_version": "2026.5.1",
                "fme_module": "fme.ace.inference",
                "calendar": "proleptic_gregorian",
                "checkpoint": "${DATA}/models/ace2-era5/ace2_era5_ckpt.tar",
                "checkpoint_sha256": ckpt_sha,
                "ic_files": ["${DATA}/inputs/ace2-era5/ic/ic_1979.nc"],
                "forcing_sets": [
                    {
                        "name": "amip-hist",
                        "dir": "${DATA}/inputs/ace2-era5/amip-hist",
                        "window": ["1979-01-01T00:00:00", "2014-12-31T18:00:00"],
                    }
                ],
                "required_files": [
                    "models/ace2-era5/ace2_era5_ckpt.tar",
                    "inputs/ace2-era5/ic/ic_1979.nc",
                    "inputs/ace2-era5/amip-hist/forcing_1979.nc",
                ],
                "timestep_hours": 6,
                "stochastic": False,
                "default_n_ensemble_per_ic": 1,
                "notes": "Pinned lock test model.",
            }
        ],
    }
    (data_dir / "registry.json").write_text(json.dumps(reg_doc, indent=2), encoding="utf-8")

    pg_dir = tmp_path / "pkg_prefix"
    (pg_dir / "bin").mkdir(parents=True)
    (pg_dir / "bin" / "aimip").write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    (pg_dir / "bin" / "aimip").chmod(0o755)
    runs_root = allow_tmp_run_root / "pkg_runs"
    runs_root.mkdir()
    (pg_dir / "site.json").write_text(
        json.dumps(
            {
                "name": "test",
                "scheduler": "slurm",
                "runtime": "native",
                "sbatch": {"partition": "gpu", "account": "acct", "gres": "gpu:1", "cpus_per_task": 4, "mem": "16G", "max_time": "02:00:00"},
                "run_root": str(runs_root),
            }
        ),
        encoding="utf-8",
    )

    monkeypatch.setenv("AIMIP_HOME", str(pg_dir))
    monkeypatch.setenv("AIMIP_PLAYGROUND", str(pg_dir))
    monkeypatch.setenv("AIMIP_DATA", str(data_dir))
    monkeypatch.setenv("AIMIP_LOCK", str(lock_path))
    monkeypatch.setenv("AIMIP_SITE_CONFIG", str(pg_dir / "site.json"))
    monkeypatch.setenv("AIMIP_IMAGE_SHA", "native")
    monkeypatch.delenv("AIMIP_REGISTRY", raising=False)

    # Valid intact forcing prepares cleanly.
    prep_ok = svc.prepare(RunSpec(model="ace2-era5", start="1979-01-01T00:00:00", steps=4, name="pkg-ok"))
    assert Path(prep_ok.plan_path).is_file()

    # Corrupt 1 byte of `forcing_1979.nc` in place while keeping `aimip.lock`, file size, and mtime identical!
    st_before = f1979.stat()
    f1979.write_bytes(b"PACKAGED_FORCING_TAMPERED_BYTES_1979")
    os.utime(f1979, ns=(st_before.st_atime_ns, st_before.st_mtime_ns))

    with pytest.raises(ValidationError, match="forcing_1979.nc"):
        svc.plan(RunSpec(model="ace2-era5", start="1979-01-01T00:00:00", steps=4, name="pkg-bad-plan"))

    with pytest.raises(ValidationError, match="forcing_1979.nc"):
        svc.prepare(RunSpec(model="ace2-era5", start="1979-01-01T00:00:00", steps=4, name="pkg-bad-prep"))

    with pytest.raises((ValidationError, ApprovalError), match="forcing_1979.nc"):
        svc.submit(prep_ok)

    # An explicit `derived-overrides.json` matching the rebuilt bytes is honored, and subsequent drift is still caught.
    tampered_sha = hashlib.sha256(f1979.read_bytes()).hexdigest()
    (data_dir / "derived-overrides.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "lock_path": str(lock_path),
                "overrides": [
                    {
                        "rel_path": "inputs/ace2-era5/amip-hist/forcing_1979.nc",
                        "models": ["ace2-era5"],
                        "expected_sha256": f1979_sha,
                        "actual_sha256": tampered_sha,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    overridden_plan = svc.plan(RunSpec(model="ace2-era5", start="1979-01-01T00:00:00", steps=4, name="pkg-ovr"))
    assert overridden_plan.digest != prep_ok.plan.digest


def test_post_submit_forcing_mutation_is_refused_at_exec_fme_and_exec_emu_before_inference(
    playground_env, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import aimip.emu as emu_mod
    from aimip.runner import exec_fme

    fbin = tmp_path / "r4_bin"
    fbin.mkdir()
    (fbin / "sbatch").write_text("#!/bin/bash\necho 777111\n", encoding="utf-8")
    (fbin / "sbatch").chmod(0o755)
    (fbin / "squeue").write_text("#!/bin/bash\nexit 0\n", encoding="utf-8")
    (fbin / "squeue").chmod(0o755)
    monkeypatch.setenv("PATH", f"{fbin}:{os.environ.get('PATH', '')}")
    monkeypatch.setenv("AIMIP_IMAGE_SHA", "native")

    # 1. `exec_fme`: prepare + submit, then mutate 1 byte of forcing (preserving size & mtime) before job starts.
    prep_fme = svc.prepare(RunSpec(model="ace2-era5", start="1979-01-01T00:00:00", steps=4, name="r4-fme-job"))
    att_fme = svc.submit(prep_fme)
    assert att_fme.attempt_id == "777111"

    f_file = playground_env["playground_dir"] / "inputs" / "ace2-era5" / "amip-hist" / "forcing_1979.nc"
    orig_bytes = f_file.read_bytes()
    orig_st = f_file.stat()
    f_file.write_bytes(b"Z" + orig_bytes[1:])
    os.utime(f_file, ns=(orig_st.st_atime_ns, orig_st.st_mtime_ns))

    fme_invoked = False

    def fail_if_fme_runs(*args, **kwargs):
        nonlocal fme_invoked
        fme_invoked = True
        raise AssertionError("fme inference must not start when forcing bytes changed after submit")

    monkeypatch.setattr("aimip.runner._activate_and_verify_fme_runtime", fail_if_fme_runs)
    with pytest.raises(ValidationError, match="forcing"):
        exec_fme(prep_fme.run_dir)
    assert fme_invoked is False

    # Restore original forcing bytes for cleanliness.
    f_file.write_bytes(orig_bytes)
    os.utime(f_file, ns=(orig_st.st_atime_ns, orig_st.st_mtime_ns))

    # 2. `exec_emu` (`archesclimate-ssp`): mutate 1 byte of scenario `.memmap` (preserving size & mtime) before job starts.
    from aimip import emu_catalog

    emu_data = tmp_path / "emu_r4_data"
    arches_files = [a for a in emu_catalog.artifacts() if "archesclimate-ssp" in a["models"]]
    lock_entries: dict[str, dict[str, object]] = {}
    for item in arches_files:
        fp = emu_data / item["rel_path"]
        fp.parent.mkdir(parents=True, exist_ok=True)
        payload = f"content:{item['rel_path']}".encode("utf-8")
        fp.write_bytes(payload)
        lock_entries[item["rel_path"]] = {
            "size": len(payload),
            "sha256": hashlib.sha256(payload).hexdigest(),
            "source": {"kind": "hf", "repo": item["repo"], "path": item["path"], "revision": item["revision"]},
        }
    emu_lock = tmp_path / "emu_r4.lock"
    emu_lock.write_text(json.dumps({"schema_version": 2, "files": lock_entries}, indent=2), encoding="utf-8")

    arches_code = emu_data / "code" / "ArchesClimate-SSP"
    (arches_code / "src").mkdir(parents=True)
    (arches_code / "src" / "__init__.py").write_text("", encoding="utf-8")

    arches_entries = emu_catalog.registry_entries("${DATA}")
    arches_entry = next(e for e in arches_entries if e["name"] == "archesclimate-ssp")
    ckpt_rel = "models/archesclimate-ssp/archesclimate-ssp-deterministic/model.ckpt"
    arches_entry["checkpoint_sha256"] = str(lock_entries[ckpt_rel]["sha256"])
    (emu_data / "registry.json").write_text(
        json.dumps({"schema_version": 3, "models": [arches_entry]}, indent=2),
        encoding="utf-8",
    )

    fake_emu_py = tmp_path / "fake_emu_py"
    emu_ran = tmp_path / "emu_ran.flag"
    fake_emu_py.write_text(f"#!/bin/sh\ntouch '{emu_ran}'\nexit 0\n", encoding="utf-8")
    fake_emu_py.chmod(0o755)

    with monkeypatch.context() as mp:
        mp.delenv("AIMIP_REGISTRY", raising=False)
        mp.setenv("AIMIP_DATA", str(emu_data))
        mp.setenv("AIMIP_LOCK", str(emu_lock))
        mp.setenv("AIMIP_EMU_PYTHON", str(fake_emu_py))
        mp.setenv("AIMIP_ARCHES_CODE", str(arches_code))

        prep_emu = svc.prepare(
            RunSpec(
                model="archesclimate-ssp",
                start="2015-01-01T00:00:00",
                steps=3,
                forcing_set="ssp585",
                name="r4-arches-job",
            )
        )
        att_emu = svc.submit(prep_emu)
        assert att_emu.attempt_id == "777111"

        memmap_file = next(
            Path(r["path"]) for r in prep_emu.manifest["forcing_files"] if r["path"].endswith(".memmap")
        )
        mm_orig = memmap_file.read_bytes()
        mm_st = memmap_file.stat()
        memmap_file.write_bytes(b"X" + mm_orig[1:])
        os.utime(memmap_file, ns=(mm_st.st_atime_ns, mm_st.st_mtime_ns))

        manifest_doc = json.loads(Path(prep_emu.manifest_path).read_text(encoding="utf-8"))
        with pytest.raises(ValidationError, match="SHA-256|Scenario file"):
            emu_mod.exec_emu(prep_emu.run_dir, manifest_doc, prep_emu.manifest_path)
        assert not emu_ran.exists()


# ---------------------------------------------------------------------------
# 5. Local process-tree cancellation & PID-reuse protection (R5)
# ---------------------------------------------------------------------------


def _pid_is_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    try:
        # On Linux `/proc/<pid>/stat` state `Z` means zombie (already exited, awaiting wait).
        stat_p = Path(f"/proc/{pid}/stat")
        if stat_p.is_file():
            raw = stat_p.read_text(encoding="utf-8", errors="replace")
            after = raw.rsplit(")", 1)[1].split()
            if after and after[0] == "Z":
                return False
    except OSError:
        pass
    return True


def test_local_submit_cancels_process_group_descendants_and_refuses_reused_pid(
    tmp_path: Path, allow_tmp_run_root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from aimip.site_config import init_site_config

    repo_root = Path(__file__).resolve().parents[1]
    runs_root = allow_tmp_run_root / "r5_runs"
    runs_root.mkdir()
    prefix = tmp_path / "r5_prefix"
    (prefix / "bin").mkdir(parents=True)
    site_path = prefix / "site.json"
    init_site_config(preset="local", out_path=str(site_path), runtime="native", run_root=str(runs_root), overwrite=True)

    tree_timeout = float(os.environ.get("AIMIP_TEST_SUBPROCESS_TIMEOUT", "90"))

    # Helper script invoked as `"$PREFIX/bin/aimip" exec-fme "<run_dir>"`: spawns child -> grandchild and writes PIDs atomically.
    tree_helper = tmp_path / "spawn_tree.py"
    tree_helper.write_text(
        textwrap.dedent(
            f"""\
            import json, os, pathlib, subprocess, sys, time
            run_dir = pathlib.Path(sys.argv[1])
            mode = sys.argv[2]
            if mode == "grandchild":
                time.sleep(300)
                sys.exit(0)
            if mode == "child":
                gc = subprocess.Popen([sys.executable, "-B", "-S", __file__, str(run_dir), "grandchild"])
                gc_tmp = run_dir / ".grandchild.pid.tmp"
                gc_tmp.write_text(f"{{gc.pid}}\\n", encoding="utf-8")
                os.replace(gc_tmp, run_dir / "grandchild.pid")
                gc.wait()
                sys.exit(0)
            ch = subprocess.Popen([sys.executable, "-B", "-S", __file__, str(run_dir), "child"])
            deadline = time.monotonic() + {tree_timeout!r}
            while time.monotonic() < deadline and not (run_dir / "grandchild.pid").is_file():
                if ch.poll() is not None and not (run_dir / "grandchild.pid").is_file():
                    break
                time.sleep(0.05)
            gc_pid = int((run_dir / "grandchild.pid").read_text(encoding="utf-8").strip())
            pids_tmp = run_dir / ".tree_pids.json.tmp"
            pids_tmp.write_text(
                json.dumps({{"parent": os.getpid(), "child": ch.pid, "grandchild": gc_pid}}),
                encoding="utf-8",
            )
            os.replace(pids_tmp, run_dir / "tree_pids.json")
            ch.wait()
            """
        ),
        encoding="utf-8",
    )

    def _wait_for_started_tree(run_dir: Path, leader_pid: int) -> dict[str, int]:
        pids_file = run_dir / "tree_pids.json"
        deadline = time.monotonic() + tree_timeout
        while time.monotonic() < deadline:
            if pids_file.is_file():
                break
            if not _pid_is_alive(leader_pid) and not pids_file.is_file():
                time.sleep(0.1)
                if not pids_file.is_file():
                    break
            time.sleep(0.05)
        if not pids_file.is_file():
            log_file = run_dir / "local-run.log"
            log_txt = log_file.read_text(encoding="utf-8", errors="replace") if log_file.is_file() else "<no local-run.log>"
            pytest.fail(
                f"Timed out waiting for {pids_file} (leader_pid={leader_pid}, alive={_pid_is_alive(leader_pid)}); "
                f"local-run.log:\n{log_txt}"
            )
        parsed = json.loads(pids_file.read_text(encoding="utf-8"))
        out_pids = {role: int(pval) for role, pval in parsed.items()}
        assert set(out_pids.keys()) == {"parent", "child", "grandchild"}
        for role, pval in out_pids.items():
            assert _pid_is_alive(pval), f"{run_dir.name}: {role} ({pval}) did not start or already exited"
        return out_pids

    launcher = prefix / "bin" / "aimip"
    launcher.write_text(
        f'#!/bin/bash\nif [ "${{1:-}}" = "exec-fme" ]; then exec "{sys.executable}" -B -S "{tree_helper}" "$2" parent; fi\n'
        f'PYTHONPATH="{repo_root / "src"}" exec "{sys.executable}" -B -m aimip "$@"\n',
        encoding="utf-8",
    )
    launcher.chmod(0o755)

    monkeypatch.setenv("AIMIP_HOME", str(prefix))
    monkeypatch.setenv("AIMIP_PLAYGROUND", str(prefix))
    monkeypatch.setenv("AIMIP_SITE_CONFIG", str(site_path))
    monkeypatch.setenv("AIMIP_DEFAULT_OUT_ROOT", str(runs_root))
    monkeypatch.setenv("AIMIP_SCHEDULER", "none")
    monkeypatch.setenv("AIMIP_IMAGE_SHA", "native")

    # 1. Native Python `submit` + `services.cancel`: kills parent, child, AND grandchild, then reports `cancelled`.
    py_run = runs_root / "py_tree_run"
    py_run.mkdir()
    (py_run / "job.sbatch").write_text(
        f'#!/bin/bash\n# aimip-generated v1\n\nset -euo pipefail\n"{prefix}/bin/aimip" exec-fme "{py_run}"\n',
        encoding="utf-8",
    )

    att = svc.submit(str(py_run))
    assert att.scheduler == "none" and att.attempt_id is not None
    # `local.pid` contract: plain numeric PID line.
    py_lpid_str = (py_run / "local.pid").read_text(encoding="utf-8").strip()
    assert py_lpid_str.isdigit()

    (py_run / "aimip-manifest.json").write_text(
        json.dumps(
            {
                "schema_version": 2,
                "mode": "local",
                "out_dir": str(py_run),
                "registry_entry": {"name": "ace2-era5"},
                "forcing_set": {"name": "amip-hist"},
                "start_time": "1979-01-01T00:00:00",
                "end_time": "1979-01-02T00:00:00",
                "steps": 4,
                "members": 1,
            }
        ),
        encoding="utf-8",
    )

    pids: dict[str, int] = {}
    try:
        pids = _wait_for_started_tree(py_run, int(py_lpid_str))
        cres = svc.cancel(str(py_run))
        assert cres["cancelled"] is True

        deadline = time.monotonic() + 30.0
        while time.monotonic() < deadline:
            if not any(_pid_is_alive(int(p)) for p in pids.values()) and not _pid_is_alive(int(att.attempt_id)):
                break
            time.sleep(0.05)

        for role, pval in pids.items():
            assert not _pid_is_alive(int(pval)), f"{role} ({pval}) survived cancellation!"
        st_after = get_run_status(str(py_run))
        assert st_after["state"] == "cancelled"
    finally:
        for pval in (*pids.values(), int(py_lpid_str)):
            try:
                os.kill(int(pval), 9)
            except OSError:
                pass

    # 2. Shell wrapper `submit` + `cancel` (`SCHEDULER=none`): also kills parent, child, AND grandchild.
    sh_prefix = tmp_path / "r5_sh_prefix"
    (sh_prefix / "bin").mkdir(parents=True)
    sh_sif = sh_prefix / "img.sif"
    sh_sif.write_bytes(b"R5_SHELL_WRAPPER_SIF_IMAGE")
    (sh_prefix / "image.sha256").write_text(
        f"{hashlib.sha256(sh_sif.read_bytes()).hexdigest()}\n",
        encoding="utf-8",
    )
    fakert = tmp_path / "r5_fakert"
    fakert.write_text(
        f'#!/bin/bash\nexec "{sys.executable}" -B -S "{tree_helper}" "${{@: -1}}" parent\n',
        encoding="utf-8",
    )
    fakert.chmod(0o755)
    tpl = (repo_root / "container" / "aimip-wrapper.in").read_text(encoding="utf-8")
    sh_wrapper = sh_prefix / "bin" / "aimip"
    sh_wrapper.write_text(
        tpl.replace("@PREFIX@", str(sh_prefix))
        .replace("@RUNTIME@", "apptainer")
        .replace("@RUNTIME_BIN@", str(fakert))
        .replace("@IMAGE@", str(sh_prefix / "img.sif"))
        .replace("@SCHEDULER@", "none")
        .replace("@RUN_ROOT@", str(runs_root))
        .replace("@EXTRA_RO@", ""),
        encoding="utf-8",
    )
    sh_wrapper.chmod(0o755)

    sh_run = runs_root / "sh_tree_run"
    sh_run.mkdir()
    (sh_run / "job.sbatch").write_text(
        f'#!/bin/bash\n# aimip-generated v1\n\nset -euo pipefail\n"{sh_prefix}/bin/aimip" exec-fme "{sh_run}"\n',
        encoding="utf-8",
    )
    sub_sh = subprocess.run(["/bin/bash", str(sh_wrapper), "submit", str(sh_run)], capture_output=True, text=True)
    assert sub_sh.returncode == 0, sub_sh.stderr
    sh_lpid_str = (sh_run / "local.pid").read_text(encoding="utf-8").strip()
    assert sh_lpid_str.isdigit()

    sh_pids: dict[str, int] = {}
    try:
        sh_pids = _wait_for_started_tree(sh_run, int(sh_lpid_str))
        can_sh = subprocess.run(["/bin/bash", str(sh_wrapper), "cancel", str(sh_run)], capture_output=True, text=True)
        assert can_sh.returncode == 0, can_sh.stderr
        deadline = time.monotonic() + 30.0
        while time.monotonic() < deadline:
            if not any(_pid_is_alive(int(p)) for p in sh_pids.values()) and not _pid_is_alive(int(sh_lpid_str)):
                break
            time.sleep(0.05)
        for role, pval in sh_pids.items():
            assert not _pid_is_alive(int(pval)), f"shell {role} ({pval}) survived cancellation!"
    finally:
        for pval in (*sh_pids.values(), int(sh_lpid_str)):
            try:
                os.kill(int(pval), 9)
            except OSError:
                pass

    # 3. PID-reuse protection: spawn an unrelated live process, write its PID with a stale `local_start_id`,
    #    and verify `status` does NOT report `"running"` and `cancel` (Python & shell) REFUSES to kill it!
    bystander = subprocess.Popen(
        [sys.executable, "-B", "-S", "-c", "import time; time.sleep(300)"], start_new_session=True
    )
    try:
        stale_run = runs_root / "reused_pid_run"
        stale_run.mkdir()
        (stale_run / "aimip-manifest.json").write_text(
            json.dumps(
                {
                    "schema_version": 2,
                    "mode": "local",
                    "out_dir": str(stale_run),
                    "registry_entry": {"name": "ace2-era5"},
                    "forcing_set": {"name": "amip-hist"},
                    "start_time": "1979-01-01T00:00:00",
                    "end_time": "1979-01-02T00:00:00",
                    "steps": 4,
                    "members": 1,
                }
            ),
            encoding="utf-8",
        )
        (stale_run / "local.pid").write_text(f"{bystander.pid}\n", encoding="utf-8")
        (stale_run / "submit-intent.json").write_text(
            json.dumps(
                {
                    "state": "submitted_local",
                    "local_pid": bystander.pid,
                    "local_pgid": bystander.pid,
                    "local_start_id": "stale-start-identity-from-1999",
                }
            ),
            encoding="utf-8",
        )

        st_reused = get_run_status(str(stale_run))
        assert st_reused["state"] != "running"
        with pytest.raises(ConfigError, match="terminal state"):
            svc.cancel(str(stale_run))
        with monkeypatch.context() as mp_race:
            mp_race.setattr(
                "aimip.services.core.get_run_status",
                lambda _d: {"state": "running", "local_pid": bystander.pid},
            )
            with pytest.raises(ExecutionError, match="reused"):
                svc.cancel(str(stale_run))
        assert bystander.poll() is None, "Python cancel signalled a reused bystander PID!"

        sh_reused = subprocess.run(
            ["/bin/bash", str(sh_wrapper), "cancel", str(stale_run)], capture_output=True, text=True
        )
        assert sh_reused.returncode != 0
        assert bystander.poll() is None, "Shell cancel signalled a reused bystander PID!"

        # 3b. Absent `submit-intent.json` (only `local.pid`): `status` sees running, `cancel` (Python & shell) REFUSES without signalling!
        no_intent_run = runs_root / "absent_intent_run"
        no_intent_run.mkdir()
        (no_intent_run / "aimip-manifest.json").write_text(
            (stale_run / "aimip-manifest.json").read_text(encoding="utf-8"),
            encoding="utf-8",
        )
        (no_intent_run / "local.pid").write_text(f"{bystander.pid}\n", encoding="utf-8")
        assert get_run_status(str(no_intent_run))["state"] == "running"
        with pytest.raises(ExecutionError, match="cannot verify"):
            svc.cancel(str(no_intent_run))
        assert bystander.poll() is None, "Python cancel signalled a PID with absent submit-intent.json!"

        sh_no_intent = subprocess.run(
            ["/bin/bash", str(sh_wrapper), "cancel", str(no_intent_run)], capture_output=True, text=True
        )
        assert sh_no_intent.returncode != 0
        assert "cannot verify" in sh_no_intent.stderr
        assert bystander.poll() is None, "Shell cancel signalled a PID with absent submit-intent.json!"

        # 3c. Failed live-identity lookup: recorded `submit-intent.json` matches `bystander`, but live start lookup fails -> REFUSED!
        live_fail_run = runs_root / "failed_live_lookup_run"
        live_fail_run.mkdir()
        (live_fail_run / "aimip-manifest.json").write_text(
            (stale_run / "aimip-manifest.json").read_text(encoding="utf-8"),
            encoding="utf-8",
        )
        real_bystander_start = read_pid_start_id(bystander.pid) or "proc:123456"
        real_bystander_pgid = int(os.getpgid(bystander.pid))
        (live_fail_run / "local.pid").write_text(f"{bystander.pid}\n", encoding="utf-8")
        (live_fail_run / "submit-intent.json").write_text(
            json.dumps(
                {
                    "state": "submitted_local",
                    "local_pid": bystander.pid,
                    "local_pgid": real_bystander_pgid,
                    "local_start_id": real_bystander_start,
                }
            ),
            encoding="utf-8",
        )
        with monkeypatch.context() as mp_nolid:
            mp_nolid.setattr("aimip.submit.read_pid_start_id", lambda _p: None)
            with pytest.raises(ExecutionError, match="cannot verify"):
                svc.cancel(str(live_fail_run))
        assert bystander.poll() is None, "Python cancel signalled a PID when live start lookup failed!"

        nolid_bin = tmp_path / "r5_nolid_bin"
        nolid_bin.mkdir()
        import shutil as _shutil

        real_awk = _shutil.which("awk") or "/usr/bin/awk"
        real_ps = _shutil.which("ps") or "/bin/ps"
        (nolid_bin / "awk").write_text(
            f'#!/bin/sh\ncase "$1" in *"proc:"*) exit 0;; esac\nexec "{real_awk}" "$@"\n',
            encoding="utf-8",
        )
        (nolid_bin / "awk").chmod(0o755)
        (nolid_bin / "ps").write_text(
            f'#!/bin/sh\ncase "$*" in *lstart=*) exit 1;; esac\nexec "{real_ps}" "$@"\n',
            encoding="utf-8",
        )
        (nolid_bin / "ps").chmod(0o755)
        sh_nolid = subprocess.run(
            ["/bin/bash", str(sh_wrapper), "cancel", str(live_fail_run)],
            env={**os.environ, "PATH": f"{nolid_bin}:{os.environ.get('PATH', '')}"},
            capture_output=True,
            text=True,
        )
        assert sh_nolid.returncode != 0
        assert "cannot verify" in sh_nolid.stderr
        assert bystander.poll() is None, "Shell cancel signalled a PID when live start lookup failed!"

        # 3d. Mismatched live PGID: recorded `local_start_id` matches, but `local_pgid` differs from live `os.getpgid` -> REFUSED!
        bad_pgid_run = runs_root / "mismatched_pgid_run"
        bad_pgid_run.mkdir()
        (bad_pgid_run / "aimip-manifest.json").write_text(
            (stale_run / "aimip-manifest.json").read_text(encoding="utf-8"),
            encoding="utf-8",
        )
        (bad_pgid_run / "local.pid").write_text(f"{bystander.pid}\n", encoding="utf-8")
        (bad_pgid_run / "submit-intent.json").write_text(
            json.dumps(
                {
                    "state": "submitted_local",
                    "local_pid": bystander.pid,
                    "local_pgid": real_bystander_pgid + 1000,
                    "local_start_id": real_bystander_start,
                }
            ),
            encoding="utf-8",
        )
        with pytest.raises(ExecutionError, match="cannot verify"):
            svc.cancel(str(bad_pgid_run))
        sh_bad_pgid = subprocess.run(
            ["/bin/bash", str(sh_wrapper), "cancel", str(bad_pgid_run)], capture_output=True, text=True
        )
        assert sh_bad_pgid.returncode != 0
        assert "cannot verify" in sh_bad_pgid.stderr
        assert bystander.poll() is None, "Cancel signalled a PID with mismatched live PGID!"
    finally:
        bystander.kill()
        bystander.wait()

    # 4. When BOTH `setsid` and `python3` are unavailable, shell local `submit` fails closed without launching or writing `local.pid`.
    import shutil as _shutil

    no_session_bin = tmp_path / "r5_no_session_bin"
    no_session_bin.mkdir()
    for tool in (
        "bash",
        "id",
        "mkdir",
        "head",
        "tr",
        "cut",
        "dirname",
        "ls",
        "awk",
        "sed",
        "sha256sum",
        "shasum",
        "mktemp",
        "cat",
        "chmod",
        "mv",
        "rm",
        "rmdir",
        "basename",
        "sleep",
    ):
        found_tool = _shutil.which(tool)
        if found_tool:
            os.symlink(found_tool, no_session_bin / tool)
    noses_marker = tmp_path / "noses_should_not_run.flag"
    noses_fakert = tmp_path / "r5_noses_fakert"
    noses_fakert.write_text(f"#!/bin/sh\ntouch '{noses_marker}'\nexit 0\n", encoding="utf-8")
    noses_fakert.chmod(0o755)
    noses_wrapper = sh_prefix / "bin" / "aimip-noses"
    noses_wrapper.write_text(
        tpl.replace("@PREFIX@", str(sh_prefix))
        .replace("@RUNTIME@", "apptainer")
        .replace("@RUNTIME_BIN@", str(noses_fakert))
        .replace("@IMAGE@", str(sh_prefix / "img.sif"))
        .replace("@SCHEDULER@", "none")
        .replace("@RUN_ROOT@", str(runs_root))
        .replace("@EXTRA_RO@", ""),
        encoding="utf-8",
    )
    noses_wrapper.chmod(0o755)
    noses_run = runs_root / "no_session_helpers_run"
    noses_run.mkdir()
    (noses_run / "job.sbatch").write_text(
        f'#!/bin/bash\n# aimip-generated v1\n\nset -euo pipefail\n"{sh_prefix}/bin/aimip" exec-fme "{noses_run}"\n',
        encoding="utf-8",
    )
    sub_noses = subprocess.run(
        ["/bin/bash", str(noses_wrapper), "submit", str(noses_run)],
        env={"PATH": str(no_session_bin), "HOME": str(tmp_path), "TMPDIR": str(tmp_path)},
        capture_output=True,
        text=True,
    )
    assert sub_noses.returncode != 0
    assert "setsid or python3" in sub_noses.stderr
    assert not (noses_run / "local.pid").exists()
    assert not (noses_run / "submit-intent.json").exists()
    assert not noses_marker.exists()


# ---------------------------------------------------------------------------
# 6. Immutable container image execution & SIF SHA-256 verification (R6)
# ---------------------------------------------------------------------------


def test_container_wrapper_enforces_immutable_docker_image_id_and_verifies_sif_sha256(
    tmp_path: Path, allow_tmp_run_root: Path
) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    real_version = (repo_root / "container" / "VERSION").read_text(encoding="utf-8").strip()
    runs_root = allow_tmp_run_root / "r6_runs"
    runs_root.mkdir()

    fakebin = tmp_path / "r6_fakebin"
    fakebin.mkdir()
    (fakebin / "uname").write_text("#!/bin/sh\necho x86_64\n", encoding="utf-8")
    (fakebin / "uname").chmod(0o755)

    img_a_id = "1111111111111111111111111111111111111111111111111111111111111111"
    img_b_id = "2222222222222222222222222222222222222222222222222222222222222222"
    docker_state = tmp_path / "docker_tag_id.txt"
    docker_state.write_text(img_a_id, encoding="utf-8")
    docker_run_log = tmp_path / "docker_run_images.log"

    fake_docker = fakebin / "docker"
    fake_docker.write_text(
        textwrap.dedent(
            f"""\
            #!{sys.executable}
            import pathlib, sys
            args = sys.argv[1:]
            tag_id = pathlib.Path({str(docker_state)!r}).read_text().strip()
            if args[:1] == ["pull"]:
                sys.exit(0)
            if args[:2] == ["image", "inspect"]:
                target = args[-1]
                fmt = args[args.index("--format") + 1] if "--format" in args else ""
                if "RepoDigests" in fmt:
                    print("registry.example.org/org/aimip@sha256:" + tag_id)
                elif target.startswith("sha256:"):
                    # Only `img_a_id` exists in local image store unless overwritten
                    live = pathlib.Path({str(docker_state)!r} + ".byid").read_text().strip() if pathlib.Path({str(docker_state)!r} + ".byid").exists() else {img_a_id!r}
                    print("sha256:" + live)
                else:
                    print("sha256:" + tag_id)
                sys.exit(0)
            if args[:1] == ["run"]:
                # Record which image ref was passed to `docker run`
                for idx, tok in enumerate(args):
                    if tok.startswith("sha256:") or tok.startswith("registry.example.org/"):
                        with pathlib.Path({str(docker_run_log)!r}).open("a") as fh:
                            fh.write(tok + "\\n")
                        break
                if "--version" in args:
                    print({real_version!r})
                    sys.exit(0)
                if "--entrypoint" in args and "python" in args:
                    c_idx = args.index("-c")
                    code = args[c_idx + 1]
                    rest = args[c_idx + 2:]
                    import types
                    sys.modules["aimip"] = types.SimpleNamespace(__version__={real_version!r})
                    sys.argv = ["-c", *rest]
                    exec(compile(code, "<string>", "exec"), {{"__name__": "__main__"}})
                sys.exit(0)
            sys.exit(0)
            """
        ),
        encoding="utf-8",
    )
    fake_docker.chmod(0o755)

    env = {**os.environ, "PATH": f"{fakebin}:{os.environ.get('PATH', '')}", "TMPDIR": "/tmp"}
    doc_pfx = tmp_path / "r6_doc_pfx"
    inst = subprocess.run(
        [
            "/bin/bash",
            str(repo_root / "install" / "install-container.sh"),
            "--prefix",
            str(doc_pfx),
            "--site",
            "local",
            "--run-root",
            str(runs_root),
            "--runtime",
            "docker",
            "--image",
            f"registry.example.org/org/aimip:{real_version}",
        ],
        env=env,
        capture_output=True,
        text=True,
    )
    assert inst.returncode == 0, inst.stderr + inst.stdout
    wrapper_txt = (doc_pfx / "bin" / "aimip").read_text(encoding="utf-8")
    assert f"IMAGE=sha256:{img_a_id}" in wrapper_txt
    assert f"registry.example.org/org/aimip:{real_version}" in (doc_pfx / "image.ref").read_text(encoding="utf-8")

    # Retagging the mutable tag to `img_b_id` still executes pinned `sha256:<img_a_id>`!
    docker_state.write_text(img_b_id, encoding="utf-8")
    docker_run_log.write_text("", encoding="utf-8")
    r_ver = subprocess.run([str(doc_pfx / "bin" / "aimip"), "--version"], env=env, capture_output=True, text=True)
    assert r_ver.returncode == 0, r_ver.stderr
    assert f"sha256:{img_a_id}" in docker_run_log.read_text(encoding="utf-8")

    # If live inspect of the configured image returns a mismatched ID, wrapper refuses before running!
    Path(str(docker_state) + ".byid").write_text(img_b_id, encoding="utf-8")
    r_tamper_doc = subprocess.run([str(doc_pfx / "bin" / "aimip"), "models"], env=env, capture_output=True, text=True)
    assert r_tamper_doc.returncode != 0
    assert "image" in r_tamper_doc.stderr.lower()

    # SIF replacement check: replacing SIF bytes (even preserving exact file size, mtime, AND matching `image.stat`!)
    # refuses before `exec-fme`, `submit`, AND inline `run ... --local`, and missing `image.sha256` also fails closed!
    sif_pfx = tmp_path / "r6_sif_pfx"
    (sif_pfx / "bin").mkdir(parents=True)
    sif_file = sif_pfx / "aimip.sif"
    sif_file.write_bytes(b"ORIGINAL_SIF_IMAGE_BYTES_V1")
    sif_sha = hashlib.sha256(sif_file.read_bytes()).hexdigest()
    (sif_pfx / "image.sha256").write_text(f"{sif_sha}\n", encoding="utf-8")

    def _ls_stat_tuple(path: Path) -> str:
        res = subprocess.run(
            ["/bin/sh", "-c", 'LC_ALL=C ls -ln "$1" | awk \'{print $5"|"$6"|"$7"|"$8}\'', "sh", str(path)],
            capture_output=True,
            text=True,
            check=True,
        )
        return res.stdout.strip()

    expected_stat_tuple = _ls_stat_tuple(sif_file)
    (sif_pfx / "image.stat").write_text(f"{expected_stat_tuple}\n", encoding="utf-8")

    apptainer_ran = tmp_path / "apptainer_ran.flag"
    fake_app = fakebin / "apptainer"
    fake_app.write_text(f"#!/bin/sh\ntouch '{apptainer_ran}'\nexit 0\n", encoding="utf-8")
    fake_app.chmod(0o755)

    tpl = (repo_root / "container" / "aimip-wrapper.in").read_text(encoding="utf-8")
    sif_wrapper = sif_pfx / "bin" / "aimip"
    sif_wrapper.write_text(
        tpl.replace("@PREFIX@", str(sif_pfx))
        .replace("@RUNTIME@", "apptainer")
        .replace("@RUNTIME_BIN@", str(fake_app))
        .replace("@IMAGE@", str(sif_file))
        .replace("@SCHEDULER@", "slurm")
        .replace("@RUN_ROOT@", str(runs_root))
        .replace("@EXTRA_RO@", ""),
        encoding="utf-8",
    )
    sif_wrapper.chmod(0o755)

    exec_dir = runs_root / "sif_exec_check"
    exec_dir.mkdir()
    ok_exec = subprocess.run([str(sif_wrapper), "exec-fme", str(exec_dir)], env=env, capture_output=True, text=True)
    assert ok_exec.returncode == 0, ok_exec.stderr
    assert apptainer_ran.exists()
    apptainer_ran.unlink()

    ok_inline_local = subprocess.run(
        [str(sif_wrapper), "run", "ace2-era5", "--start", "1979-01-01", "--steps", "1", "--local"],
        env=env,
        capture_output=True,
        text=True,
    )
    assert ok_inline_local.returncode == 0, ok_inline_local.stderr
    assert apptainer_ran.exists()
    apptainer_ran.unlink()

    # Tamper SIF bytes while preserving exact size & mtime (so `image.stat` still matches!) -> refused on BOTH `exec-fme` and `run --local`!
    st_sif = sif_file.stat()
    sif_file.write_bytes(b"REPLACED_SIF_IMAGE_BYTES_V2")
    os.utime(sif_file, ns=(st_sif.st_atime_ns, st_sif.st_mtime_ns))
    assert _ls_stat_tuple(sif_file) == expected_stat_tuple

    bad_exec = subprocess.run([str(sif_wrapper), "exec-fme", str(exec_dir)], env=env, capture_output=True, text=True)
    assert bad_exec.returncode != 0
    assert not apptainer_ran.exists()

    bad_inline_local = subprocess.run(
        [str(sif_wrapper), "run", "ace2-era5", "--start", "1979-01-01", "--steps", "1", "--local"],
        env=env,
        capture_output=True,
        text=True,
    )
    assert bad_inline_local.returncode != 0
    assert "sha-256" in bad_inline_local.stderr.lower()
    assert not apptainer_ran.exists()

    # Restore valid SIF bytes, remove `image.sha256` -> missing checksum MUST fail closed without calling container runtime!
    sif_file.write_bytes(b"ORIGINAL_SIF_IMAGE_BYTES_V1")
    (sif_pfx / "image.sha256").unlink()
    for missing_sha_cmd in (
        [str(sif_wrapper), "run", "ace2-era5", "--start", "1979-01-01", "--steps", "1", "--local"],
        [str(sif_wrapper), "models"],
        [str(doc_pfx / "bin" / "aimip"), "models"],
    ):
        if missing_sha_cmd[0] == str(doc_pfx / "bin" / "aimip"):
            (doc_pfx / "image.sha256").unlink()
        res_missing_sha = subprocess.run(missing_sha_cmd, env=env, capture_output=True, text=True)
        assert res_missing_sha.returncode != 0
        assert "image.sha256" in res_missing_sha.stderr
        assert not apptainer_ran.exists()


# ---------------------------------------------------------------------------
# 7–10. Model metadata honesty, HiRO-ACE period/citation, tiers, IC provenance, MCP
# ---------------------------------------------------------------------------


def test_hiro_ace_periods_tier_citation_honest_tiers_and_ic_provenance(
    tmp_path: Path, allow_tmp_run_root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from aimip.ui.server import _plan_decision_context

    data_dir = tmp_path / "meta_data"
    data_dir.mkdir()
    monkeypatch.delenv("AIMIP_REGISTRY", raising=False)
    monkeypatch.setenv("AIMIP_DATA", str(data_dir))
    monkeypatch.setenv("AIMIP_DEFAULT_OUT_ROOT", str(allow_tmp_run_root))

    # Item 7: `hiro-ace` tier, title, training period (2014-2022), out-of-training 1994-2014 caveat, and arXiv:2512.18224 citation.
    hiro_card = svc.show("hiro-ace")
    assert hiro_card["title"] == "HiRO-ACE (Ai2, experimental 1994-2014 extension)"
    assert "1994-2024" not in hiro_card["title"]
    assert hiro_card["status"] == "experimental"
    assert hiro_card["verified_on"] is None
    assert "2014-2022" in hiro_card["limitations"] or "2014–2022" in hiro_card["limitations"]
    assert "outside" in hiro_card["limitations"].lower()
    assert "not expected to generalize" in hiro_card["limitations"].lower()

    for hiro_model in ("hiro-ace", "ace2s", "hiro-acev2"):
        cite_doc = svc.cite(hiro_model)
        assert "2512.18224" in cite_doc["bibtex"]
        assert "Perkins" in cite_doc["bibtex"]

    # `hiro-acev2` cites its own paper (Kwa et al. 2026, arXiv:2610.17693) and names it in `limitations`;
    # `samudrace-e3smv3` cites both Wu et al. (arXiv:2608.10277) and SamudrACE GRL (doi:10.1029/2025GL119340).
    hiro2_cite = svc.cite("hiro-acev2")
    assert "2610.17693" in hiro2_cite["bibtex"]
    assert "Extreme events and statistics" in hiro2_cite["bibtex"]
    assert "2610.17693" in MODEL_CARDS["hiro-acev2"]["limitations"]

    e3sm_cite = svc.cite("samudrace-e3smv3")
    assert "10.1029/2025GL119340" in e3sm_cite["bibtex"]
    assert "2608.10277" in e3sm_cite["bibtex"]

    # Verbatim quoted `"Authors' limits"` vs `"aimip-claima:"` caveats:
    shieldp_lim = MODEL_CARDS["ace2s-shield-plus"]["limitations"]
    assert '"inherits the biases relative to observations"' in shieldp_lim
    assert '"generalization ability to unseen sea ice coverage is limited"' in shieldp_lim
    assert '"abrupt regime shifts in stratospheric temperature and moisture can occur in inference runs"' in shieldp_lim
    assert "aimip-claima:" in shieldp_lim

    som_lim = MODEL_CARDS["ace2-som"]["limitations"]
    assert "idealized" not in som_lim.lower()
    assert '"Sea-ice coverage, ocean heat transport, and ocean mixed layer depth are prescribed' in som_lim
    assert '"Abrupt regime shifts in stratospheric temperature and moisture' in som_lim
    assert '"Adjustment to an abrupt change in CO2 concentration occurs too quickly' in som_lim
    assert "aimip-claima:" in som_lim

    cm4_lim = MODEL_CARDS["samudrace-cm4"]["limitations"]
    assert "aimip-claima:" in cm4_lim and "~1 K colder" in cm4_lim
    assert cm4_lim.index("aimip-claima:") < cm4_lim.index("~1 K colder")

    eam_lim = MODEL_CARDS["ace2-eamv3"]["limitations"]
    assert "1970-2020" in eam_lim
    assert eam_lim.index("aimip-claima:") < eam_lim.index("no CO2 input")

    # Dedicated exit code 5 (`EXIT_UNSTABLE`) for unstable DLESyM rollouts (distinct from `EXIT_APPROVAL == 4`).
    assert EXIT_APPROVAL == 4
    assert EXIT_UNSTABLE == 5
    assert emu_job.EXIT_UNSTABLE == 5
    for dlesym_name in ("dlesym-v1.0", "dlesym-v1.1", "dlesym-v0-isccp"):
        dlim = MODEL_CARDS[dlesym_name]["limitations"]
        assert "exit code 5" in dlim
        assert "exit code 4" not in dlim
    repo_root = Path(__file__).resolve().parents[1]
    assert "5 an unstable rollout" in (repo_root / "docs" / "console.md").read_text(encoding="utf-8")
    assert "exit code 5" in (repo_root / "docs" / "models.md").read_text(encoding="utf-8")
    assert "exit code 5" in (repo_root / "docs" / "troubleshooting.md").read_text(encoding="utf-8")

    # Item 9: Honest tiers — CI does NOT run model weights; only the 3 bit-identical fme models are `certified`.
    assert "AIMIP CI" not in TIER_DEFINITIONS["certified"]
    certified_models = {name for name, c in MODEL_CARDS.items() if c["status"] == "certified"}
    assert certified_models == {"ace2-era5", "hiro-acev2", "samudrace-cm4"}
    assert MODEL_CARDS["samudrace-e3smv3"]["status"] == "runs"
    assert MODEL_CARDS["samudrace-e3smv3"]["verified_on"] is None
    assert MODEL_CARDS["dlesym-v1.1"]["verified_on"] is not None
    assert MODEL_CARDS["dlesym-v1.0"]["verified_on"] is None

    # Item 8: Authors' limitations + `ic_provenance` shown in `render_show_card`, `render_plan_card`, and UI decision context.
    era5_show = svc.show("ace2-era5")
    assert "doubling of CO2" in era5_show["limitations"] or "perturbations" in era5_show["limitations"]
    assert "operational" in era5_show["limitations"].lower()

    for relabel_model, orig_year in (("ace2-eamv3", "1971-01-01"), ("hiro-ace", "2014-01-01")):
        show_doc = svc.show(relabel_model)
        show_txt = svc.render_show_card(show_doc)
        assert orig_year in show_txt
        assert "discard the first simulated year" in show_txt

        resolved = svc.plan(
            RunSpec(
                model=relabel_model,
                start="1994-01-01T00:00:00",
                steps=4,
                name=f"card-{relabel_model}",
                resources={"partition": "gpu", "account": "test_acct"},
            )
        )
        plan_txt = svc.render_plan_card(resolved, is_tty=False)
        assert orig_year in plan_txt
        assert "discard the first simulated year" in plan_txt
        assert "limits:" in plan_txt

        ui_ctx = _plan_decision_context(resolved)
        assert orig_year in ui_ctx["model_card"]["ic_provenance"]

    app_js = (Path(__file__).resolve().parents[1] / "src" / "aimip" / "ui" / "static" / "app.js").read_text(encoding="utf-8")
    assert "ic_provenance" in app_js


def test_mcp_extra_and_agent_requirements_and_real_sdk_when_installed() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    pyproject_path = repo_root / "pyproject.toml"
    if pyproject_path.is_file():
        pyproj = tomllib.loads(pyproject_path.read_text(encoding="utf-8"))
        opt_deps = pyproj["project"]["optional-dependencies"]
        assert "mcp" in opt_deps
        assert any(dep.startswith("mcp") for dep in opt_deps["mcp"])

    req_agent = repo_root / "container" / "requirements-agent.txt"
    if req_agent.is_file():
        req_lines = [ln.strip() for ln in req_agent.read_text(encoding="utf-8").splitlines() if ln.strip() and not ln.startswith("#")]
        assert any(ln.startswith("mcp==") or ln.startswith("mcp>=") for ln in req_lines)

    third_party = repo_root / "THIRD_PARTY.md"
    if third_party.is_file():
        tp_txt = third_party.read_text(encoding="utf-8")
        assert "langchain-google-vertexai" not in tp_txt
        assert "`mcp`" in tp_txt

    if importlib.util.find_spec("mcp") is not None:
        srv = build_fastmcp_server()
        assert srv is not None


# ---------------------------------------------------------------------------
# 11. Wrapper, CLI `--no-fetch` exit code, bare `collect`, and completion parity
# ---------------------------------------------------------------------------


def test_wrapper_approve_rw_config_mount_flag_order_json_envelopes_and_cli_parity(
    playground_env, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    wrapper_tpl = (repo_root / "container" / "aimip-wrapper.in").read_text(encoding="utf-8")
    native_sh = (repo_root / "install" / "install-native.sh").read_text(encoding="utf-8")

    # 1. Both container wrapper and native launcher include `approve` in writable `DATA_MODE=rw`.
    assert "run|ic|chat|mcp|ui|approve)" in wrapper_tpl
    assert "run|ic|chat|mcp|ui|approve)" in native_sh

    # 2. Wrapper resolves relative `--config experiment.yaml` against host `$PWD` & bind-mounts external `--config` `:ro`,
    #    handles `logs -f demo`, forwards `submit --help`, and emits canonical v1 `--json` error/success envelopes.
    prefix = tmp_path / "parity_pfx"
    (prefix / "bin").mkdir(parents=True)
    (prefix / "data").mkdir(parents=True)
    parity_sif = prefix / "img.sif"
    parity_sif.write_bytes(b"PARITY_TEST_SIF_IMAGE")
    (prefix / "image.sha256").write_text(
        f"{hashlib.sha256(parity_sif.read_bytes()).hexdigest()}\n",
        encoding="utf-8",
    )
    runs_root = Path(playground_env["default_out_root"])
    cap_argv = tmp_path / "cap.argv"
    fakert = tmp_path / "parity_fakert"
    fakert.write_text(
        f'#!/bin/bash\nprintf "%s\\n" "$@" > "{cap_argv}"\nexit 0\n',
        encoding="utf-8",
    )
    fakert.chmod(0o755)
    fbin = tmp_path / "parity_bin"
    fbin.mkdir()
    sacct_log = tmp_path / "parity_sacct.log"
    (fbin / "sacct").write_text(
        f'#!/bin/bash\necho "$@" >> "{sacct_log}"\necho "5151|RUNNING|00:01:00|0:0"\n',
        encoding="utf-8",
    )
    (fbin / "sacct").chmod(0o755)
    (fbin / "sbatch").write_text("#!/bin/bash\necho 5152\n", encoding="utf-8")
    (fbin / "sbatch").chmod(0o755)

    wrapper = prefix / "bin" / "aimip"
    wrapper.write_text(
        wrapper_tpl.replace("@PREFIX@", str(prefix))
        .replace("@RUNTIME@", "apptainer")
        .replace("@RUNTIME_BIN@", str(fakert))
        .replace("@IMAGE@", str(prefix / "img.sif"))
        .replace("@SCHEDULER@", "slurm")
        .replace("@RUN_ROOT@", str(runs_root))
        .replace("@EXTRA_RO@", ""),
        encoding="utf-8",
    )
    wrapper.chmod(0o755)
    env = {**os.environ, "PATH": f"{fbin}:{os.environ.get('PATH', '')}", "AIMIP_WRAPPER_SACCT_INTERVAL": "0"}

    ext_dir = tmp_path / "ext_cfg_dir"
    ext_dir.mkdir()
    ext_cfg = ext_dir / "experiment.yaml"
    ext_cfg.write_text("model: ace2-era5\nstart: 1979-01-01\nsteps: 4\n", encoding="utf-8")

    r_cfg = subprocess.run(
        [str(wrapper), "plan", "--config", "experiment.yaml"],
        cwd=str(ext_dir),
        env=env,
        capture_output=True,
        text=True,
    )
    assert r_cfg.returncode == 0, r_cfg.stderr
    argv_lines = cap_argv.read_text(encoding="utf-8").splitlines()
    assert f"{ext_cfg}:{ext_cfg}:ro" in argv_lines
    assert str(ext_cfg) in argv_lines[argv_lines.index("plan") :]

    # Option before positional target on `logs -f demo`: wrapper still refreshes `demo/sacct.txt`!
    demo_dir = runs_root / "demo"
    demo_dir.mkdir(exist_ok=True)
    (demo_dir / "slurm.jobid").write_text("5151\n", encoding="utf-8")
    r_logs = subprocess.run([str(wrapper), "logs", "-f", "demo"], env=env, capture_output=True, text=True)
    assert r_logs.returncode == 0
    assert (demo_dir / "sacct.txt").is_file()

    # `submit --help` forwards to container instead of failing with `unsupported option for host submit: --help`.
    r_help = subprocess.run([str(wrapper), "submit", "--help"], env=env, capture_output=True, text=True)
    assert r_help.returncode == 0

    # `submit --json` error & success envelopes match v1 schema (`code`, `exit_code`, `message`, `field`, `attempt_id`, `state`).
    r_sub_err = subprocess.run([str(wrapper), "submit", "demo", "--json"], env=env, capture_output=True, text=True)
    assert r_sub_err.returncode != 0
    err_env = json.loads(r_sub_err.stdout)
    assert err_env["aimip"] == "1" and err_env["ok"] is False
    assert {"code", "exit_code", "message", "field", "allowed", "next_command"} <= set(err_env["error"].keys())

    sub_ok_dir = runs_root / "sub_json_ok"
    sub_ok_dir.mkdir()
    (sub_ok_dir / "job.sbatch").write_text(
        f'#!/bin/bash\n# aimip-generated v1\n\nset -euo pipefail\n"{prefix}/bin/aimip" exec-fme "{sub_ok_dir}"\n',
        encoding="utf-8",
    )
    r_sub_ok = subprocess.run([str(wrapper), "submit", "sub_json_ok", "--json"], env=env, capture_output=True, text=True)
    assert r_sub_ok.returncode == 0, r_sub_ok.stderr
    ok_res = parse_cli_json(r_sub_ok.stdout)
    assert ok_res["attempt_id"] == "5152"
    assert ok_res["state"] == "SUBMITTED"
    assert ok_res["run_name"] == "sub_json_ok"

    # 3. `aimip ic spin ... --no-fetch` on unfetched model exits with code `3` (`MissingAssetsError`).
    empty_data = tmp_path / "empty_spin_data"
    empty_data.mkdir()
    with monkeypatch.context() as mp:
        mp.delenv("AIMIP_REGISTRY", raising=False)
        mp.delenv("AIMIP_PLAYGROUND", raising=False)
        mp.delenv("AIMIP_HOME", raising=False)
        mp.setenv("AIMIP_SITE_CONFIG", str(playground_env["playground_dir"] / "site.json"))
        mp.setenv("AIMIP_DATA", str(empty_data))
        rc_spin = main(
            [
                "ic",
                "spin",
                "-m",
                "ace2-era5",
                "--from",
                "1979-01-01T00:00:00,1979-02-01T00:00:00",
                "--to",
                "1979-03-01T00:00:00",
                "--out",
                str(runs_root / "spin_nf"),
                "--no-fetch",
                "--dry-run",
            ]
        )
        assert rc_spin == 3
        capsys.readouterr()

    # 4. `aimip collect <bare-spin-name> --out ...` resolves bare spin directory under `run_root`.
    spin_bare = runs_root / "bare_spin"
    assert (
        main(
            [
                "ic",
                "spin",
                "-m",
                "ace2-era5",
                "--from",
                "1979-01-01T00:00:00,1979-01-02T00:00:00",
                "--to",
                "1979-01-03T00:00:00",
                "--out",
                str(spin_bare),
                "--dry-run",
            ]
        )
        == 0
    )
    capsys.readouterr()
    for k in (0, 1):
        write_fake_ic_netcdf(
            spin_bare / str(k) / "restart.nc",
            ["1979-01-03T00:00:00"],
            calendar="proleptic_gregorian",
            sample_dim="sample",
            spatial_dims=("latitude", "longitude"),
            seed=10 + k,
        )
    merged_ic = runs_root / "collected_from_bare.nc"
    assert main(["collect", "bare_spin", "--out", str(merged_ic)]) == 0
    capsys.readouterr()
    assert merged_ic.is_file()

    # 5. Completion flag parity against `cli.build_parser()`: `cite` has no `--examples`, `plan` has no run-only flags.
    parser = build_parser()
    sub_action = next(a for a in parser._actions if hasattr(a, "choices") and isinstance(a.choices, dict))
    cite_opts = set(sub_action.choices["cite"]._option_string_actions.keys())
    plan_opts = set(sub_action.choices["plan"]._option_string_actions.keys())
    assert "--examples" not in cite_opts
    for run_only in ("--no-fetch", "--dry-run", "--submit", "--approve"):
        assert run_only not in plan_opts

    for sh in ("bash", "zsh"):
        comp = render_completion(sh)
        assert "show|cite)" not in comp
        assert "plan|run)" not in comp
