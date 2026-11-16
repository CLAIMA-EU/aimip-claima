"""Phase 2a execution integrity, approval binding, and wrapper security regression tests:
- Fail-closed external seal (`$AIMIP_STATE/prepared/`) & deterministic `render_sbatch_script` check defeat `plan.json` / `config.yaml` / `aimip-manifest.json` hash deletion or rewrite
- Custom forcing directory, forcing file content changes, staged DLESyM ERA5 IC bytes, and runtime overrides (`AIMIP_EMU_PYTHON`, `AIMIP_FME_2026_10_OVERLAY`, `AIMIP_ARCHES_CODE`) bound by digest
- Rendered `container/aimip-wrapper.in` host `submit` invokes `verify-submit` before `sbatch`, rejects resource drift, and refuses blind retries on uncertain `sbatch` replies
- Sanitized child environment strips `GEMINI_API_KEY`, `GOOGLE_API_KEY`, `AIMIP_GEMINI_KEY_FILE`, `AIMIP_GEMINI_VERTEX_TOKEN_FILE`, `GOOGLE_APPLICATION_CREDENTIALS`, and `LANGCHAIN_*` across `sbatch`, `sacct`, `scancel`, and DLESyM staging
- `squeue` reconciliation requires exact per-attempt `attempt_nonce`, never adopting an unrelated same-model queued job
- Concurrent submissions against the same run directory are serialized via `.submit.lock` `flock` (at most 1 `sbatch` invocation)
- `aimip run --local` invokes the low-level executor (`exec_fme` / `exec_emu`) exactly once without `sbatch`, and `services.submit` on a `mode="local"` plan runs locally even on a Slurm site
- Bad CLI/YAML numbers return structured JSON v1 error envelopes (`rc == 2`), and DLESyM staging routes child stdout to stderr
- `AIMIP_DATA_MODE=ro` refuses download approval upfront with `aimip fetch -m <model>` hint, while `AIMIP_DATA_MODE=rw` downloads cleanly after separate consent
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import contextlib
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Any

import pytest

from aimip import services as svc
from aimip.agent.tools import AgentTools
from aimip.cli import main
from aimip.errors import ApprovalError, ExecutionError, MissingAssetsError, ValidationError
from aimip.registry import DATA_PLACEHOLDER, _parse_model_entry, build_default_registry_payload
from aimip.services import core
from aimip.services.approval import MemoryApprovalBroker, current_user
from aimip.services.types import Authorization, RunSpec
from aimip.submit import submit_run
from conftest import parse_cli_json, pty_responder


def request(name: str) -> RunSpec:
    return RunSpec(
        model="ace2-era5",
        start="1979-01-01T00:00:00",
        steps=4,
        name=name,
        resources={"time": "00:10:00"},
    )


def fake_attempt(run_dir: str, **kwargs):
    return {
        "out_dir": run_dir,
        "scheduler": "slurm",
        "slurm_job_id": "424242",
        "local_pid": None,
        "log_path": None,
        "reconciled": False,
    }


@pytest.mark.parametrize("rewrite_hashes", [False, True])
def test_approved_script_cannot_rewrite_its_own_integrity_record(
    playground_env, monkeypatch, rewrite_hashes: bool
) -> None:
    prepared = svc.prepare(request(f"artifact-seal-{int(rewrite_hashes)}"))
    broker = MemoryApprovalBroker()
    now = time.time()
    # Trusted host fixture: stand in for a token already minted after real TTY OK.
    auth = Authorization(
        action="submit",
        digest=prepared.plan.digest,
        user=current_user(),
        issued_at=now,
        expires_at=now + 600,
        nonce="review-token",
    )
    broker.store_minted(auth)
    job = Path(prepared.sbatch_path)
    original = job.read_text(encoding="utf-8")
    assert "--time=00:10:00" in original
    job.write_text(original.replace("--time=00:10:00", "--time=00:20:00"), encoding="utf-8")
    plan_file = Path(prepared.plan_path)
    doc = json.loads(plan_file.read_text(encoding="utf-8"))
    if rewrite_hashes:
        doc["artifact_hashes"]["job.sbatch"] = hashlib.sha256(job.read_bytes()).hexdigest()
    else:
        doc.pop("artifact_hashes")
    plan_file.write_text(json.dumps(doc), encoding="utf-8")
    contacted: list[str] = []

    def scheduler(run_dir: str, **kwargs):
        contacted.append(run_dir)
        return fake_attempt(run_dir, **kwargs)

    monkeypatch.setattr(core, "submit_run", scheduler)
    monkeypatch.setattr(core.shutil, "which", lambda name: "/fake/" + name)
    with pytest.raises(ApprovalError):
        svc.submit(prepared, authorization=auth, require_authorization=True, broker=broker)
    assert not contacted

    # Cross-process `svc.submit(run_dir_str, ...)` (without in-memory `PreparedRun`) MUST ALSO reject!
    auth2 = Authorization(
        action="submit",
        digest=prepared.plan.digest,
        user=current_user(),
        issued_at=now,
        expires_at=now + 600,
        nonce="review-token-2",
    )
    broker.store_minted(auth2)
    with pytest.raises(ApprovalError):
        svc.submit(prepared.run_dir, authorization=auth2, require_authorization=True, broker=broker)
    assert not contacted


def test_missing_spec_or_schema_in_plan_json_rejected_on_submit(playground_env, monkeypatch) -> None:
    prepared = svc.prepare(request("missing-spec-seal"))
    monkeypatch.setattr(core, "submit_run", fake_attempt)
    monkeypatch.setattr(core.shutil, "which", lambda name: "/fake/" + name)
    plan_file = Path(prepared.plan_path)
    doc = json.loads(plan_file.read_text(encoding="utf-8"))
    doc.pop("spec")
    plan_file.write_text(json.dumps(doc), encoding="utf-8")
    with pytest.raises(ApprovalError):
        svc.submit(prepared.run_dir)


def test_custom_forcing_directory_and_contents_and_runtime_overrides_are_bound_by_digest(
    playground_env, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    spec = request("forcing-binding")
    before = svc.plan(spec)
    registry = playground_env["registry_path"]
    doc = json.loads(registry.read_text(encoding="utf-8"))
    replacement = playground_env["playground_dir"] / "other-forcing"
    replacement.mkdir()
    f_nc = replacement / "forcing_1979.nc"
    f_nc.write_bytes(b"different-forcing")
    entry = next(m for m in doc["models"] if m["name"] == "ace2-era5")
    entry["forcing_sets"][0]["dir"] = str(replacement)
    registry.write_text(json.dumps(doc), encoding="utf-8")
    after_dir = svc.plan(spec)
    assert after_dir.digest != before.digest

    # Mutating forcing file bytes in place MUST also change the digest!
    f_nc.write_bytes(b"mutated-forcing-in-place")
    after_bytes = svc.plan(spec)
    assert after_bytes.digest != after_dir.digest

    # Changing `AIMIP_ARCHES_CODE` tree or `AIMIP_EMU_PYTHON` changes Arches plan digest!
    data_dir = tmp_path / "arches_data"
    data_dir.mkdir()
    payload = build_default_registry_payload(root_placeholder=DATA_PLACEHOLDER)
    arches_entry = next(e for e in payload["models"] if e["name"] == "archesclimate-ssp")
    reg_arches = tmp_path / "arches_reg.json"
    reg_arches.write_text(json.dumps({"models": [arches_entry]}), encoding="utf-8")
    arches_spec = RunSpec(model="archesclimate-ssp", start="2015-01-01", length="1y", name="arch-r2")
    base_arch = svc.plan(arches_spec, registry_path=str(reg_arches), data_dir=str(data_dir))

    monkeypatch.setenv("AIMIP_EMU_PYTHON", "/custom/emu/python")
    emu_arch = svc.plan(arches_spec, registry_path=str(reg_arches), data_dir=str(data_dir))
    assert emu_arch.digest != base_arch.digest

    code_override = tmp_path / "custom_arches_code"
    code_override.mkdir()
    (code_override / "eval_multistep.py").write_text("# custom\n", encoding="utf-8")
    monkeypatch.setenv("AIMIP_ARCHES_CODE", str(code_override))
    code_arch1 = svc.plan(arches_spec, registry_path=str(reg_arches), data_dir=str(data_dir))
    assert code_arch1.digest != emu_arch.digest
    (code_override / "eval_multistep.py").write_text("# modified\n", encoding="utf-8")
    code_arch2 = svc.plan(arches_spec, registry_path=str(reg_arches), data_dir=str(data_dir))
    assert code_arch2.digest != code_arch1.digest


def test_scheduler_child_does_not_inherit_chat_auth(playground_env, monkeypatch) -> None:
    prepared = svc.prepare(request("child-environment"))
    secret_keys = (
        "GEMINI_API_KEY",
        "GOOGLE_API_KEY",
        "AIMIP_GEMINI_KEY_FILE",
        "AIMIP_GEMINI_VERTEX_TOKEN_FILE",
        "GOOGLE_APPLICATION_CREDENTIALS",
        "LANGCHAIN_API_KEY",
    )
    for key in secret_keys:
        monkeypatch.setenv(key, "review-placeholder")
    observed: list[dict[str, str]] = []

    def child(argv, **kwargs):
        observed.append(dict(kwargs.get("env") or os.environ))
        exe = Path(argv[0]).name
        if exe == "sbatch":
            return subprocess.CompletedProcess(argv, 0, "424242\n", "")
        if exe == "sacct":
            return subprocess.CompletedProcess(argv, 0, "424242|RUNNING|00:00:10|0:0\n", "")
        if exe == "scancel":
            return subprocess.CompletedProcess(argv, 0, "", "")
        return subprocess.CompletedProcess(argv, 0, "", "")

    monkeypatch.setattr(core.shutil, "which", lambda name: "/fake/" + name)
    monkeypatch.setattr("aimip.status.shutil.which", lambda name: "/fake/" + name)
    monkeypatch.setattr(subprocess, "run", child)
    svc.submit(prepared)
    svc.status(prepared.run_dir)
    svc.cancel(prepared.run_dir)
    assert len(observed) >= 3
    for env_snap in observed:
        assert all(key not in env_snap for key in secret_keys)


def test_bad_yaml_and_cli_number_have_structured_json_error_and_staging_uses_stderr(
    playground_env, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    # 1. Bad YAML number (`steps: bad`) -> structured JSON error (`rc == 2`, `field == "steps"`)
    config = tmp_path / "bad-number.yaml"
    config.write_text("model: ace2-era5\nstart: '1979-01-01'\nsteps: bad\n", encoding="utf-8")
    rc = main(["plan", "--config", str(config), "--json"])
    output = json.loads(capsys.readouterr().out)
    assert rc == 2
    assert output["aimip"] == "1" and output["ok"] is False
    assert output["error"]["field"] == "steps"

    # 2. Bad CLI `--steps bad --json` -> structured JSON error (`rc == 2`, `field == "steps"`)
    rc_cli = main(["plan", "ace2-era5", "--start", "1979-01-01", "--steps", "bad", "--json"])
    out_cli = json.loads(capsys.readouterr().out)
    assert rc_cli == 2
    assert out_cli["aimip"] == "1" and out_cli["ok"] is False
    assert out_cli["error"]["field"] == "steps"

    # 3. DLESyM `_stage_dlesym_ic` routes child stdout to stderr and strips chat credentials
    from aimip import emu

    fake_emu_py = tmp_path / "fake_emu_py"
    fake_emu_py.write_text(
        "#!/bin/sh\n"
        'echo "CHILD_PROGRESS_LINE"\n'
        'if [ -n "${GEMINI_API_KEY:-}" ] || [ -n "${AIMIP_GEMINI_KEY_FILE:-}" ]; then exit 99; fi\n'
        'printf "PT" > "$5/ic_era5.pt"\n',
        encoding="utf-8",
    )
    fake_emu_py.chmod(0o755)
    monkeypatch.setenv("AIMIP_EMU_PYTHON", str(fake_emu_py))
    monkeypatch.setenv("GEMINI_API_KEY", "secret-must-not-leak")
    monkeypatch.setenv("AIMIP_GEMINI_KEY_FILE", "/tmp/secret-key-file")
    stage_dir = Path(playground_env["default_out_root"]) / "dlesym-stage-check"
    stage_dir.mkdir(parents=True, exist_ok=True)
    staged_ic = emu._stage_dlesym_ic(str(stage_dir))
    assert Path(staged_ic).is_file()
    cap = capsys.readouterr()
    assert cap.out == ""  # ZERO contamination of stdout / MCP stdio!
    assert "CHILD_PROGRESS_LINE" in cap.err


def test_rendered_wrapper_submit_verifies_plan_rejects_drift_and_never_blindly_retries(
    playground_env, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Exercise rendered `container/aimip-wrapper.in`: drift rejected before `sbatch`, uncertain reply not retried."""
    repo_root = Path(__file__).resolve().parents[1]
    wrapper_tpl = (repo_root / "container" / "aimip-wrapper.in").read_text(encoding="utf-8")

    prefix: Path = playground_env["playground_dir"]
    (prefix / "bin").mkdir(parents=True, exist_ok=True)
    (prefix / "data").mkdir(parents=True, exist_ok=True)
    (prefix / "img.sif").write_bytes(b"fake-sif")
    (prefix / "image.sha256").write_text("sha256:deadbeef1234\n", encoding="utf-8")
    (prefix / "data" / "registry.json").write_text(
        playground_env["registry_path"].read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    fakebin = tmp_path / "wrap_fakebin"
    fakebin.mkdir()
    sbatch_log = tmp_path / "wrap_sbatch.log"

    # Fake `apptainer` applies `--env K=V` flags and invokes real `sys.executable -B -m aimip` for `verify-submit`
    fakert = fakebin / "apptainer"
    fakert.write_text(
        "#!/bin/bash\n"
        "while [ $# -gt 0 ]; do\n"
        '  case "$1" in\n'
        '    --env) export "$2"; shift 2;;\n'
        '    python) shift; break;;\n'
        '    *) shift;;\n'
        '  esac\n'
        "done\n"
        "unset AIMIP_REGISTRY\n"
        f'PYTHONPATH="{repo_root / "src"}" exec "{sys.executable}" "$@"\n',
        encoding="utf-8",
    )
    fakert.chmod(0o755)

    sbatch = fakebin / "sbatch"
    sbatch.write_text(
        "#!/bin/bash\n"
        f'echo "$@" >> "{sbatch_log}"\n'
        'echo "slurm_receive_msg: Socket timed out" >&2\n'
        "exit 1\n",
        encoding="utf-8",
    )
    sbatch.chmod(0o755)
    squeue = fakebin / "squeue"
    squeue.write_text("#!/bin/bash\nexit 0\n", encoding="utf-8")
    squeue.chmod(0o755)

    runs_root = Path(playground_env["default_out_root"])
    rendered = (
        wrapper_tpl.replace("@PREFIX@", str(prefix))
        .replace("@RUNTIME@", "apptainer")
        .replace("@RUNTIME_BIN@", str(fakert))
        .replace("@IMAGE@", str(prefix / "img.sif"))
        .replace("@SCHEDULER@", "slurm")
        .replace("@RUN_ROOT@", str(runs_root))
        .replace("@EXTRA_RO@", "")
    )
    wrapper_bin = prefix / "bin" / "aimip"
    wrapper_bin.write_text(rendered, encoding="utf-8")
    wrapper_bin.chmod(0o755)

    monkeypatch.delenv("AIMIP_REGISTRY", raising=False)
    monkeypatch.setenv("AIMIP_DATA", str(prefix / "data"))
    monkeypatch.setenv("AIMIP_SITE_CONFIG", str(prefix / "site.json"))

    prepared = svc.prepare(
        RunSpec(model="ace2-era5", start="1979-01-01T00:00:00", steps=4, name="wrap-r3b", resources={"time": "00:10:00"}),
        playground_dir=str(prefix),
        data_dir=str(prefix / "data"),
        site_config_path=str(prefix / "site.json"),
    )

    env = {
        **os.environ,
        "PATH": f"{fakebin}:/usr/bin:/bin",
        "AIMIP_RUN_ROOTS": str(runs_root),
    }

    # 1. Tamper `#SBATCH --time` AND rewrite `plan.json` hash -> rendered wrapper MUST reject before calling `sbatch`!
    job_p = Path(prepared.sbatch_path)
    orig_job = job_p.read_text(encoding="utf-8")
    job_p.write_text(orig_job.replace("--time=00:10:00", "--time=00:25:00"), encoding="utf-8")
    pdoc = json.loads(Path(prepared.plan_path).read_text(encoding="utf-8"))
    pdoc["artifact_hashes"]["job.sbatch"] = hashlib.sha256(job_p.read_bytes()).hexdigest()
    Path(prepared.plan_path).write_text(json.dumps(pdoc), encoding="utf-8")

    bad_proc = subprocess.run([str(wrapper_bin), "submit", prepared.run_dir], env=env, capture_output=True, text=True)
    assert bad_proc.returncode != 0
    assert not sbatch_log.exists(), "wrapper must not invoke sbatch when verify-submit rejects drift"

    # 2. Valid prepared run with uncertain `sbatch` timeout -> wrapper records `submit-intent.json` and refuses second `sbatch`!
    valid_prep = svc.prepare(
        RunSpec(model="ace2-era5", start="1979-01-01T00:00:00", steps=4, name="wrap-uncertain", resources={"time": "00:10:00"}),
        playground_dir=str(prefix),
        data_dir=str(prefix / "data"),
        site_config_path=str(prefix / "site.json"),
    )
    u1 = subprocess.run(
        [str(wrapper_bin), "submit", f"--expect-digest={valid_prep.plan.digest}", valid_prep.run_dir],
        env=env,
        capture_output=True,
        text=True,
    )
    assert u1.returncode == 1
    assert len(sbatch_log.read_text(encoding="utf-8").splitlines()) == 1

    u2 = subprocess.run([str(wrapper_bin), "submit", valid_prep.run_dir], env=env, capture_output=True, text=True)
    assert u2.returncode == 1
    assert "refusing blind retry" in u2.stderr
    assert len(sbatch_log.read_text(encoding="utf-8").splitlines()) == 1


def test_reconciliation_requires_exact_attempt_nonce_never_adopting_same_model_job(
    playground_env, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Two same-model runs sharing default job name `aimip-ace2-era5`: Run B never adopts Run A's job ID."""
    fbin = tmp_path / "r5_bin"
    fbin.mkdir()
    prep_a = svc.prepare(RunSpec(model="ace2-era5", start="1979-01-01T00:00:00", steps=4, name="run-a"))
    prep_b = svc.prepare(RunSpec(model="ace2-era5", start="1979-01-01T00:00:00", steps=8, name="run-b"))
    assert prep_a.plan.resources["job_name"] == prep_b.plan.resources["job_name"] == "aimip-ace2-era5"

    sbatch = fbin / "sbatch"
    sbatch.write_text("#!/bin/bash\necho 111111\n", encoding="utf-8")
    sbatch.chmod(0o755)
    squeue = fbin / "squeue"
    squeue.write_text("#!/bin/bash\nexit 0\n", encoding="utf-8")
    squeue.chmod(0o755)
    monkeypatch.setenv("PATH", f"{fbin}:{os.environ.get('PATH', '')}")

    att_a = svc.submit(prep_a)
    assert att_a.attempt_id == "111111"
    intent_a = json.loads((Path(prep_a.run_dir) / "submit-intent.json").read_text(encoding="utf-8"))

    # Now `sbatch` fails uncertainly for `run-b` while `squeue` shows ONLY `run-a` (`111111` with `intent_a["attempt_nonce"]`)!
    sbatch.write_text(
        '#!/bin/bash\necho "slurm_receive_msg: Socket timed out" >&2\nexit 1\n',
        encoding="utf-8",
    )
    squeue.write_text(
        f'#!/bin/bash\necho "111111|aimip-ace2-era5|PENDING|{intent_a["attempt_nonce"]}"\n',
        encoding="utf-8",
    )
    with pytest.raises(ExecutionError, match=r"(?i)uncertain scheduler reply"):
        svc.submit(prep_b)
    assert not (Path(prep_b.run_dir) / "slurm.jobid").exists(), "run-b must never adopt run-a's job ID"


def test_concurrent_submissions_serialized_by_per_run_flock(
    playground_env, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Two concurrent submitters on the same run directory invoke `sbatch` at most ONCE."""
    fbin = tmp_path / "r6_bin"
    fbin.mkdir()
    counter = tmp_path / "r6_count.txt"
    sbatch = fbin / "sbatch"
    sbatch.write_text(
        "#!/bin/bash\n"
        "sleep 0.2\n"
        f'echo 1 >> "{counter}"\n'
        "echo 555666\n",
        encoding="utf-8",
    )
    sbatch.chmod(0o755)
    monkeypatch.setenv("PATH", f"{fbin}:{os.environ.get('PATH', '')}")

    prepared = svc.prepare(request("concurrent-flock"))

    def _worker() -> str:
        try:
            res = submit_run(prepared.run_dir)
            return f"ok:{res['slurm_job_id']}"
        except Exception as exc:  # noqa: BLE001
            return f"err:{exc}"

    with ThreadPoolExecutor(max_workers=2) as pool:
        f1 = pool.submit(_worker)
        f2 = pool.submit(_worker)
        outcomes = sorted([f1.result(), f2.result()])

    assert outcomes[0].startswith("err:") and "already submitted" in outcomes[0]
    assert outcomes[1] == "ok:555666"
    assert len(counter.read_text(encoding="utf-8").splitlines()) == 1


def test_cli_run_local_executes_once_and_local_plan_submit_ignores_slurm_site(
    playground_env, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """`aimip run --local` executes locally once without `sbatch`, and `submit` honors `mode='local'`."""
    sbatch_calls: list[str] = []
    local_exec_calls: list[str] = []

    fbin = tmp_path / "r7_bin"
    fbin.mkdir()
    sbatch_log = tmp_path / "r7_sbatch.log"
    sbatch = fbin / "sbatch"
    sbatch.write_text(f'#!/bin/bash\necho "$@" >> "{sbatch_log}"\necho 888\n', encoding="utf-8")
    sbatch.chmod(0o755)
    monkeypatch.setenv("PATH", f"{fbin}:{os.environ.get('PATH', '')}")

    def _fake_execute(self_adapter, out_dir: str):
        local_exec_calls.append(out_dir)
        manifest_p = Path(out_dir) / "aimip-manifest.json"
        mdoc = json.loads(manifest_p.read_text(encoding="utf-8"))
        mdoc["finished"] = "2026-11-16T00:00:00Z"
        mdoc["rc"] = 0
        manifest_p.write_text(json.dumps(mdoc), encoding="utf-8")
        return mdoc

    from aimip.families import FmeFamilyAdapter

    monkeypatch.setattr(FmeFamilyAdapter, "execute", _fake_execute)

    # 1. `aimip run ... --local` executes once immediately and never calls `sbatch`
    assert main(["run", "ace2-era5", "--start", "1979-01-01", "--steps", "4", "--name", "local-cli", "--local", "--json"]) == 0
    out_local = parse_cli_json(capsys.readouterr().out)
    assert out_local["rc"] == 0
    assert len(local_exec_calls) == 1
    assert not sbatch_log.exists() and not sbatch_calls

    # 2. Separately `services.prepare(mode='local')` does NOT execute during `prepare`, and `services.submit` uses `scheduler='none'`
    prep_local = svc.prepare(RunSpec(model="ace2-era5", start="1979-01-01", steps=4, name="local-sub", mode="local"))
    assert len(local_exec_calls) == 1  # `prepare()` alone did not execute!
    # Replace `job.sbatch` body check? No, `job.sbatch` runs `<prefix>/bin/aimip exec-fme` via `nohup bash` -> stub wrapper bin!
    wrapper_bin = playground_env["playground_dir"] / "bin" / "aimip"
    wrapper_bin.parent.mkdir(parents=True, exist_ok=True)
    wrapper_bin.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    wrapper_bin.chmod(0o755)
    prep_local2 = svc.prepare(RunSpec(model="ace2-era5", start="1979-01-01", steps=4, name="local-sub2", mode="local"))
    att = svc.submit(prep_local2)
    assert att.scheduler == "none"
    assert not sbatch_log.exists()


def test_download_approval_respects_launcher_data_mode_ro_vs_rw(
    tmp_path: Path, allow_tmp_run_root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`AIMIP_DATA_MODE=ro` refuses download approval upfront; `AIMIP_DATA_MODE=rw` downloads cleanly."""
    import aimip.setup_data as setup_data_mod
    from aimip.setup_data import SetupYearConfig
    from test_models_and_fetch import _populate_extra_fme_data_tree_and_lock

    fbin = tmp_path / "r9_slurm_bin"
    fbin.mkdir()
    (fbin / "sbatch").write_text("#!/bin/bash\necho 919191\n", encoding="utf-8")
    (fbin / "sbatch").chmod(0o755)
    (fbin / "squeue").write_text("#!/bin/bash\nexit 0\n", encoding="utf-8")
    (fbin / "squeue").chmod(0o755)
    monkeypatch.setenv("PATH", f"{fbin}:{os.environ.get('PATH', '')}")

    yc = SetupYearConfig(som_years=(2032, 2033), hiro_native_years=(2014, 2015))
    monkeypatch.setattr(setup_data_mod, "SetupYearConfig", lambda: yc)
    hf = tmp_path / "hf_r9"
    lock = tmp_path / "r9.lock"
    _populate_extra_fme_data_tree_and_lock(hf, yc, lock)

    data_dir = tmp_path / "r9_data"
    pg_dir = tmp_path / "r9_pg"
    (pg_dir / "bin").mkdir(parents=True)
    (pg_dir / "bin" / "aimip").write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    runs = allow_tmp_run_root / "r9_runs"
    runs.mkdir()
    (pg_dir / "site.json").write_text(
        json.dumps(
            {
                "name": "levante",
                "scheduler": "slurm",
                "runtime": "native",
                "sbatch": {
                    "partition": "gpu",
                    "account": "acct",
                    "gres": "gpu:1",
                    "cpus_per_task": 8,
                    "mem": "32G",
                    "max_time": "08:00:00",
                },
                "run_root": str(runs),
                "extra_ro_binds": [],
                "gpu_check": "nvidia-smi",
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.delenv("AIMIP_REGISTRY", raising=False)
    monkeypatch.setenv("AIMIP_PLAYGROUND", str(pg_dir))
    monkeypatch.setenv("AIMIP_SITE_CONFIG", str(pg_dir / "site.json"))
    monkeypatch.setenv("AIMIP_DATA", str(data_dir))
    monkeypatch.setenv("AIMIP_LOCK", str(lock))
    monkeypatch.setenv("AIMIP_HF_MIRROR", f"file://{hf}")
    monkeypatch.setenv("AIMIP_DEFAULT_OUT_ROOT", str(runs))

    tools = AgentTools()

    # 1. `AIMIP_DATA_MODE=ro` -> refused BEFORE soliciting unusable download approval, naming `aimip fetch -m ace2-som`!
    monkeypatch.setenv("AIMIP_DATA_MODE", "ro")
    prop_ro = tools.propose_run("ace2-som", "2032-01-01T00:00:00", steps=4, forcing_set="1xCO2", name="r9-ro")
    with pty_responder(prop_ro["approval_prefix"], "yes") as (tty_path, _):
        with pytest.raises(ApprovalError, match=r"aimip fetch -m ace2-som"):
            tools.approve_and_execute_on_tty(tty_path=tty_path)
    assert not data_dir.exists()

    # 2. `AIMIP_DATA_MODE=rw` (owner launcher mode) -> download approval succeeds end-to-end!
    monkeypatch.setenv("AIMIP_DATA_MODE", "rw")
    prop_rw = tools.propose_run("ace2-som", "2032-01-01T00:00:00", steps=4, forcing_set="1xCO2", name="r9-rw")
    with pty_responder(prop_rw["approval_prefix"], "yes") as (tty_path, _):
        attempt = tools.approve_and_execute_on_tty(tty_path=tty_path)
    assert attempt["state"] == "SUBMITTED"
    assert (data_dir / "models" / "ace2-som" / "ace2_som_ckpt.tar").is_file()


def test_missing_external_seal_with_rewritten_config_and_manifest_hashes_is_refused(
    playground_env, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Deleting `$AIMIP_STATE/prepared/<digest>.json` and rewriting `config.yaml`, `aimip-manifest.json`, and `plan.json` hashes is refused fail-closed."""
    ext_state = tmp_path / "ext_state"
    monkeypatch.setenv("AIMIP_STATE", str(ext_state))

    prepared = svc.prepare(request("missing-ext-seal"))
    seal_file = ext_state / "prepared" / f"{prepared.plan.digest}.json"
    assert seal_file.is_file()

    # Delete the external seal and consistently rewrite `config.yaml`, `aimip-manifest.json`, and `plan.json["artifact_hashes"]`.
    seal_file.unlink()
    cfg_path = Path(prepared.config_path)
    cfg_text = cfg_path.read_text(encoding="utf-8")
    assert "n_forward_steps: 4" in cfg_text
    cfg_path.write_text(cfg_text.replace("n_forward_steps: 4", "n_forward_steps: 400"), encoding="utf-8")
    new_cfg_sha = hashlib.sha256(cfg_path.read_bytes()).hexdigest()

    manifest_path = Path(prepared.manifest_path)
    mdoc = json.loads(manifest_path.read_text(encoding="utf-8"))
    mdoc["config_sha256"] = new_cfg_sha
    manifest_path.write_text(json.dumps(mdoc, indent=2), encoding="utf-8")
    new_manifest_sha = hashlib.sha256(manifest_path.read_bytes()).hexdigest()

    plan_path = Path(prepared.plan_path)
    pdoc = json.loads(plan_path.read_text(encoding="utf-8"))
    pdoc["artifact_hashes"]["config.yaml"] = new_cfg_sha
    pdoc["artifact_hashes"]["aimip-manifest.json"] = new_manifest_sha
    plan_path.write_text(json.dumps(pdoc, indent=2), encoding="utf-8")

    contacted: list[str] = []

    def _scheduler(run_dir: str, **kwargs):
        contacted.append(run_dir)
        return fake_attempt(run_dir, **kwargs)

    monkeypatch.setattr(core, "submit_run", _scheduler)
    monkeypatch.setattr(core.shutil, "which", lambda name: "/fake/" + name)

    with pytest.raises(ApprovalError, match="prepared-run seal"):
        svc.submit(prepared.run_dir)
    assert not contacted

    broker = MemoryApprovalBroker()
    now = time.time()
    auth = Authorization(
        action="submit",
        digest=prepared.plan.digest,
        user=current_user(),
        issued_at=now,
        expires_at=now + 600,
        nonce="seal-missing-token",
    )
    broker.store_minted(auth)
    with pytest.raises(ApprovalError, match="prepared-run seal"):
        svc.submit(prepared.run_dir, authorization=auth, require_authorization=True, broker=broker)
    assert not contacted

    # Preparation itself MUST fail closed if `$AIMIP_STATE` is a symlink and cannot create a private seal!
    bad_state = tmp_path / "symlink_state"
    real_target = tmp_path / "real_target"
    real_target.mkdir()
    bad_state.symlink_to(real_target)
    monkeypatch.setenv("AIMIP_STATE", str(bad_state))
    with pytest.raises(ApprovalError):
        svc.prepare(request("unsealable-run"))


def _setup_synthetic_dlesym_env(tmp_path: Path, allow_tmp_run_root: Path, monkeypatch: pytest.MonkeyPatch):
    """Build an offline `dlesym-v1.0` data/lock/registry + fake `$AIMIP_EMU_PYTHON` interpreter."""
    import shutil
    from aimip.emu_catalog import registry_entries
    from conftest import sha256_path

    emu_data = tmp_path / "dlesym_data"
    emu_pg = tmp_path / "dlesym_pg"
    (emu_pg / "bin").mkdir(parents=True, exist_ok=True)
    (emu_pg / "bin" / "aimip").write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    (emu_pg / "bin" / "aimip").chmod(0o755)

    emu_runs = allow_tmp_run_root / "dlesym_runs"
    emu_runs.mkdir(parents=True, exist_ok=True)
    site_file = emu_pg / "site.json"
    site_file.write_text(
        json.dumps(
            {
                "name": "cluster",
                "scheduler": "slurm",
                "runtime": "native",
                "sbatch": {
                    "partition": "gpu",
                    "account": "acct",
                    "gres": "gpu:1",
                    "cpus_per_task": 8,
                    "mem": "32G",
                    "max_time": "08:00:00",
                },
                "run_root": str(emu_runs),
                "extra_ro_binds": [],
                "gpu_check": "nvidia-smi",
            }
        ),
        encoding="utf-8",
    )

    run_counter = tmp_path / "emu_run_calls.log"
    stage_counter = tmp_path / "emu_stage_calls.log"
    stage_payload_file = tmp_path / "stage_payload.bin"
    stage_payload_file.write_bytes(b"default-era5-ic")
    fake_py = tmp_path / "fake_dlesym_py"
    fake_py.write_text(
        "#!/bin/bash\n"
        '[ "$1 $2 $3" = "-I -m aimip.emu_job" ] || exit 9\n'
        'mode=$4; out=$5\n'
        'case "$mode" in\n'
        f'  stage) echo 1 >> "{stage_counter}"; cp "{stage_payload_file}" "$out/ic_era5.pt";;\n'
        f'  run) echo 1 >> "{run_counter}"; printf \'{{"fake": true, "rc": 0}}\' > "$out/emu-result.json"; exit 0;;\n'
        '  *) exit 7;;\n'
        'esac\n',
        encoding="utf-8",
    )
    fake_py.chmod(0o755)

    dlesym_entry = next(e for e in registry_entries(str(emu_data)) if e["name"] == "dlesym-v1.0")
    lock_files: dict[str, dict] = {}
    for rel in dlesym_entry["required_files"]:
        fp = emu_data / rel
        fp.parent.mkdir(parents=True, exist_ok=True)
        fp.write_bytes(f"dlesym-file:{rel}".encode("utf-8"))
        lock_files[rel] = {
            "size": fp.stat().st_size,
            "sha256": sha256_path(fp),
            "source": {"kind": "hf", "repo": "x/y", "path": rel, "revision": "0" * 40},
        }
    dlesym_entry = dict(dlesym_entry, checkpoint_sha256=sha256_path(Path(dlesym_entry["checkpoint"])))

    reg_path = tmp_path / "dlesym_registry.json"
    reg_path.write_text(json.dumps({"schema_version": 3, "models": [dlesym_entry]}, indent=2), encoding="utf-8")
    lock_path = tmp_path / "dlesym.lock"
    lock_path.write_text(json.dumps({"schema_version": 1, "files": lock_files}, indent=2), encoding="utf-8")

    monkeypatch.setenv("AIMIP_EMU_PYTHON", str(fake_py))
    monkeypatch.setenv("AIMIP_PLAYGROUND", str(emu_pg))
    monkeypatch.setenv("AIMIP_SITE_CONFIG", str(site_file))
    monkeypatch.setenv("AIMIP_DATA", str(emu_data))
    monkeypatch.setenv("AIMIP_REGISTRY", str(reg_path))
    monkeypatch.setenv("AIMIP_LOCK", str(lock_path))
    monkeypatch.setenv("AIMIP_DEFAULT_OUT_ROOT", str(emu_runs))
    monkeypatch.setenv("AIMIP_STATE", str(tmp_path / "dlesym_state"))
    return {
        "runs": emu_runs,
        "run_counter": run_counter,
        "stage_counter": stage_counter,
        "stage_payload_file": stage_payload_file,
        "shutil": shutil,
    }


def test_staged_dlesym_ic_bytes_are_hashed_into_post_stage_plan_digest(
    tmp_path: Path, allow_tmp_run_root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Staged DLESyM `ic_era5.pt` bytes are hashed into `asset_locks`, `ic_sha256`, and `plan.digest` after staging (with zero time-dependence)."""
    env = _setup_synthetic_dlesym_env(tmp_path, allow_tmp_run_root, monkeypatch)
    target_out = env["runs"] / "dlesym-ic-binding"
    spec = RunSpec(model="dlesym-v1.0", start="1995-01-01T00:00:00", steps=8, out=str(target_out))

    pre_plan = svc.plan(spec)
    assert any(a.prospective and a.role == "dynamic_era5_ic" for a in pre_plan.missing_assets)

    env["stage_payload_file"].write_bytes(b"era5-ic-stream-alpha")
    prep_a = svc.prepare(spec, allow_fetch=True)
    expected_sha_a = hashlib.sha256(b"era5-ic-stream-alpha").hexdigest()
    assert prep_a.plan.missing_assets == ()
    assert prep_a.plan.digest != pre_plan.digest
    assert prep_a.plan.canonical_payload["ic_sha256"] == expected_sha_a
    assert (
        prep_a.plan.canonical_payload["asset_locks"][f"dynamic_era5_ic:{target_out}/ic_era5.pt"]
        == expected_sha_a
    )

    # Re-prepare at the exact same `--start` and `--out` with different staged ERA5 IC bytes -> digest MUST change!
    env["shutil"].rmtree(target_out)
    env["stage_payload_file"].write_bytes(b"era5-ic-stream-beta")
    prep_b = svc.prepare(spec, allow_fetch=True)
    expected_sha_b = hashlib.sha256(b"era5-ic-stream-beta").hexdigest()
    assert prep_b.plan.canonical_payload["ic_sha256"] == expected_sha_b
    assert prep_b.plan.digest != prep_a.plan.digest

    # Re-prepare a third time with the original alpha bytes -> digest MUST match `prep_a.plan.digest` bit-for-bit (nothing time-dependent)!
    env["shutil"].rmtree(target_out)
    env["stage_payload_file"].write_bytes(b"era5-ic-stream-alpha")
    prep_a_repeat = svc.prepare(spec, allow_fetch=True)
    assert prep_a_repeat.plan.digest == prep_a.plan.digest

    # Mutating `ic_era5.pt` on disk after `prepare()` invalidates submission before scheduler contact!
    (target_out / "ic_era5.pt").write_bytes(b"post-stage-tampered-ic")
    contacted: list[str] = []
    monkeypatch.setattr(core, "submit_run", lambda rd, **kw: contacted.append(rd) or fake_attempt(rd, **kw))
    monkeypatch.setattr(core.shutil, "which", lambda name: "/fake/" + name)
    with pytest.raises(ApprovalError):
        svc.submit(prep_a_repeat.run_dir)
    assert not contacted


def test_local_execution_invokes_low_level_executor_exactly_once_for_fme_and_emu(
    playground_env, tmp_path: Path, allow_tmp_run_root: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """`aimip run --local` invokes low-level `runner.exec_fme` (FME) and `emu.exec_emu` (emulator) exactly ONCE."""
    import aimip.emu as emu_mod
    import aimip.runner as runner_mod

    # 1. FME local run (`ace2-era5`): count calls to `aimip.runner.exec_fme` directly!
    fme_low_calls: list[str] = []

    def _spy_exec_fme(out_dir: str):
        fme_low_calls.append(out_dir)
        mp = Path(out_dir) / "aimip-manifest.json"
        mdoc = json.loads(mp.read_text(encoding="utf-8"))
        mdoc["finished"] = "2026-11-16T00:00:00Z"
        mdoc["rc"] = 0
        mp.write_text(json.dumps(mdoc, indent=2), encoding="utf-8")
        return mdoc

    with monkeypatch.context() as mp_fme:
        mp_fme.setattr(runner_mod, "exec_fme", _spy_exec_fme)
        assert (
            main(["run", "ace2-era5", "--start", "1979-01-01", "--steps", "4", "--name", "fme-once", "--local", "--json"])
            == 0
        )
        res_fme = parse_cli_json(capsys.readouterr().out)
        assert res_fme["rc"] == 0
        assert res_fme["mode"] == "local"
        assert len(fme_low_calls) == 1

    # 2. Emulator local run (`dlesym-v1.0`): count calls to BOTH `aimip.emu.exec_emu` AND subprocess `emu_job run`!
    env = _setup_synthetic_dlesym_env(tmp_path, allow_tmp_run_root, monkeypatch)
    emu_low_calls: list[str] = []
    real_exec_emu = emu_mod.exec_emu

    def _spy_exec_emu(validated_out: str, manifest: dict, manifest_path: str):
        emu_low_calls.append(validated_out)
        return real_exec_emu(validated_out, manifest, manifest_path)

    monkeypatch.setattr(emu_mod, "exec_emu", _spy_exec_emu)
    assert (
        main(["run", "dlesym-v1.0", "--start", "1995-01-01", "--steps", "4", "--name", "emu-once", "--local", "--json"])
        == 0
    )
    res_emu = parse_cli_json(capsys.readouterr().out)
    assert res_emu["rc"] == 0
    assert res_emu["mode"] == "local"
    assert len(emu_low_calls) == 1
    assert len(env["run_counter"].read_text(encoding="utf-8").splitlines()) == 1


@contextlib.contextmanager
def _interactive_two_step_pty(*, before_final_submit=None):
    """Drive the two-step `/dev/tty` prepare -> download yes -> final submit prompts interactively."""
    import pty
    import re
    import select
    import termios
    import threading

    master_fd, slave_fd = pty.openpty()
    stop_evt = threading.Event()
    transcript: dict[str, Any] = {
        "prepare_prefix": None,
        "submit_prefix": None,
        "output": "",
    }

    attrs = termios.tcgetattr(slave_fd)
    attrs[3] &= ~termios.ECHO
    termios.tcsetattr(slave_fd, termios.TCSANOW, attrs)
    tty_path = os.ttyname(slave_fd)

    prep_re = re.compile(r"To approve prepare, type the first 8 hex characters of digest \[([0-9a-f]{8})\][^:]*: ")
    sub_re = re.compile(r"To approve submit, type the first 8 hex characters of digest \[([0-9a-f]{8})\][^:]*: ")

    def _driver() -> None:
        buf = ""
        stage = 0
        while not stop_evt.is_set():
            rdy, _, _ = select.select([master_fd], [], [], 0.03)
            if not rdy:
                continue
            try:
                chunk = os.read(master_fd, 4096)
            except OSError:
                break
            if not chunk:
                break
            decoded = chunk.decode("utf-8", errors="replace")
            transcript["output"] += decoded
            buf += decoded

            if stage == 0:
                m = prep_re.search(buf)
                if m:
                    transcript["prepare_prefix"] = m.group(1)
                    buf = buf[m.end() :]
                    os.write(master_fd, (m.group(1) + "\n").encode("utf-8"))
                    stage = 1
            if stage == 1 and "Allow download now? [yes/no]: " in buf:
                idx = buf.index("Allow download now? [yes/no]: ") + len("Allow download now? [yes/no]: ")
                buf = buf[idx:]
                os.write(master_fd, b"yes\n")
                stage = 2
            if stage == 2:
                m2 = sub_re.search(buf)
                if m2:
                    transcript["submit_prefix"] = m2.group(1)
                    buf = buf[m2.end() :]
                    if before_final_submit is not None:
                        before_final_submit()
                    os.write(master_fd, (m2.group(1) + "\n").encode("utf-8"))
                    stage = 3

    worker = threading.Thread(target=_driver, daemon=True)
    worker.start()
    try:
        yield tty_path, transcript
    finally:
        stop_evt.set()
        worker.join(timeout=5.0)
        for fd in (slave_fd, master_fd):
            try:
                os.close(fd)
            except OSError:
                pass


def test_prepare_only_authorization_can_never_submit_and_output_path_is_stable_across_replanning(
    tmp_path: Path, allow_tmp_run_root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Prepare-only `Authorization` can only prepare (never submit), and unnamed default `output_dir` stays stable across re-planning."""
    env = _setup_synthetic_dlesym_env(tmp_path, allow_tmp_run_root, monkeypatch)
    env["stage_payload_file"].write_bytes(b"stable-out-era5-bytes")

    # Unnamed spec (neither `name` nor `out` passed): resolved `output_dir` MUST stay identical pre-stage and post-stage!
    spec = RunSpec(model="dlesym-v1.0", start="1995-01-01T00:00:00", steps=8)
    pre_plan = svc.plan(spec)
    assert pre_plan.requires_staging is True
    assert pre_plan.required_approval_action == "prepare"

    now = time.time()
    broker = MemoryApprovalBroker()

    # A `'submit'` token CANNOT be used to prepare a prospective staging plan
    wrong_act_auth = Authorization(
        action="submit",
        digest=pre_plan.digest,
        user=current_user(),
        issued_at=now,
        expires_at=now + 600,
        nonce="wrong-action-for-stage",
        allow_download=True,
    )
    broker.store_minted(wrong_act_auth)
    with pytest.raises(ApprovalError, match="expected 'prepare'"):
        svc.prepare(spec, authorization=wrong_act_auth, require_authorization=True, broker=broker)

    # Valid `'prepare'` token prepares and stages into `pre_plan.output_dir`
    prep_auth = Authorization(
        action="prepare",
        digest=pre_plan.digest,
        user=current_user(),
        issued_at=now,
        expires_at=now + 600,
        nonce="valid-prepare-only",
        allow_download=True,
    )
    broker.store_minted(prep_auth)
    prepared = svc.prepare(spec, authorization=prep_auth, require_authorization=True, broker=broker)
    assert prepared.run_dir == pre_plan.output_dir
    assert prepared.plan.output_dir == pre_plan.output_dir
    assert prepared.plan.digest != pre_plan.digest

    # Re-calling `svc.plan(spec)` with `name=None, out=None` resolves to the EXACT SAME `output_dir` and final digest!
    replanned = svc.plan(spec)
    assert replanned.output_dir == pre_plan.output_dir
    assert replanned.digest == prepared.plan.digest
    assert replanned.requires_staging is False
    assert replanned.required_approval_action == "submit"

    # A prepare-only `Authorization` can NEVER submit — even if minted with the final post-stage digest!
    contacted: list[str] = []
    monkeypatch.setattr(core, "submit_run", lambda rd, **kw: contacted.append(rd) or fake_attempt(rd, **kw))
    monkeypatch.setattr(core.shutil, "which", lambda name: "/fake/" + name)

    with pytest.raises(ApprovalError):
        svc.submit(prepared, authorization=prep_auth, require_authorization=True)

    forged_prepare_with_final_digest = Authorization(
        action="prepare",
        digest=prepared.plan.digest,
        user=current_user(),
        issued_at=now,
        expires_at=now + 600,
        nonce="prepare-token-with-final-digest",
        allow_download=False,
    )
    broker.store_minted(forged_prepare_with_final_digest)
    with pytest.raises(ApprovalError, match="expected 'submit'"):
        svc.submit(prepared, authorization=forged_prepare_with_final_digest, require_authorization=True, broker=broker)
    with pytest.raises(ApprovalError, match="expected 'submit'"):
        svc.submit(prepared.run_dir, authorization=forged_prepare_with_final_digest, require_authorization=True)
    assert not contacted


def test_two_step_staging_flow_across_chat_pty_mcp_approve_twice_cli_and_ui(
    tmp_path: Path, allow_tmp_run_root: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """End-to-end two-step staging approval across Chat (`/dev/tty`), MCP (`aimip approve` twice), CLI, and UI."""
    from aimip.mcp_server import McpToolset
    from aimip.ui.server import UiApprovalManager

    env = _setup_synthetic_dlesym_env(tmp_path, allow_tmp_run_root, monkeypatch)
    env["stage_payload_file"].write_bytes(b"two-step-era5-ic-v1")

    fbin = tmp_path / "twostep_slurm_bin"
    fbin.mkdir()
    sbatch_log = tmp_path / "twostep_sbatch.log"
    (fbin / "sbatch").write_text(f'#!/bin/bash\necho "$@" >> "{sbatch_log}"\necho 654321\n', encoding="utf-8")
    (fbin / "sbatch").chmod(0o755)
    (fbin / "squeue").write_text("#!/bin/bash\nexit 0\n", encoding="utf-8")
    (fbin / "squeue").chmod(0o755)
    monkeypatch.setenv("PATH", f"{fbin}:{os.environ.get('PATH', '')}")

    # 1. CHAT (`AgentTools` + interactive `/dev/tty` two-step prompt)
    tools = AgentTools()
    prop_chat = tools.propose_run("dlesym-v1.0", "1995-01-01T00:00:00", steps=8, name="chat-dl-ok")
    assert prop_chat["action"] == "prepare" and prop_chat["requires_staging"] is True
    with _interactive_two_step_pty() as (tty_path, tr):
        att_chat = tools.approve_and_execute_on_tty(tty_path=tty_path)
    assert tr["prepare_prefix"] == prop_chat["approval_prefix"]
    assert tr["submit_prefix"] is not None and tr["submit_prefix"] != tr["prepare_prefix"]
    assert "ic_era5.pt" in tr["output"]
    assert att_chat["state"] == "SUBMITTED" and att_chat["attempt_id"] == "654321"
    assert len(sbatch_log.read_text(encoding="utf-8").splitlines()) == 1

    # Chat tamper check: mutating `ic_era5.pt` before final submit confirmation is refused!
    prop_tamper = tools.propose_run("dlesym-v1.0", "1995-01-01T00:00:00", steps=8, name="chat-dl-tamper")
    tamper_ic = Path(prop_tamper["run_dir"]) / "ic_era5.pt"
    with _interactive_two_step_pty(before_final_submit=lambda: tamper_ic.write_bytes(b"tampered-ic")) as (tty_path, _):
        with pytest.raises(ApprovalError):
            tools.approve_and_execute_on_tty(tty_path=tty_path)
    assert len(sbatch_log.read_text(encoding="utf-8").splitlines()) == 1  # sbatch NOT called!

    # 2. MCP (`propose_run` -> `aimip approve` #1 -> `submit_approved` stages ONLY -> `aimip approve` #2 -> `submit_approved` submits)
    mcp_state = tmp_path / "mcp_twostep_state"
    ts = McpToolset(state_dir=str(mcp_state))
    p_mcp = ts.propose_run("dlesym-v1.0", "1995-01-01T00:00:00", steps=8)  # unnamed: verifies stable `run_dir`!
    assert p_mcp["action"] == "prepare" and p_mcp["requires_staging"] is True

    with pty_responder(p_mcp["approval_prefix"], "yes") as (tty_path, _):
        monkeypatch.setenv("AIMIP_TTY_PATH", tty_path)
        assert main(["approve", p_mcp["approval_prefix"], "--state-dir", str(mcp_state), "--json"]) == 0
    app1_json = parse_cli_json(capsys.readouterr().out)
    assert app1_json["action"] == "prepare"

    stage_mcp = ts.submit_approved(p_mcp["digest"])
    assert stage_mcp["state"] == "PREPARED_AWAITING_SUBMIT_APPROVAL"
    assert stage_mcp["submitted"] is False
    assert stage_mcp["run_dir"] == p_mcp["run_dir"]
    assert stage_mcp["digest"] != p_mcp["digest"]
    assert len(sbatch_log.read_text(encoding="utf-8").splitlines()) == 1  # prepare-only approval did NOT submit!

    # Replaying the consumed prepare digest refuses and instructs the caller to approve the final submit digest!
    with pytest.raises(ApprovalError, match=stage_mcp["approval_prefix"]) as replay_prep_exc:
        ts.submit_approved(p_mcp["digest"])
    assert replay_prep_exc.value.next_command == f"aimip approve {stage_mcp['approval_prefix']}"

    with pty_responder(stage_mcp["approval_prefix"]) as (tty_path, _):
        monkeypatch.setenv("AIMIP_TTY_PATH", tty_path)
        assert main(["approve", stage_mcp["approval_prefix"], "--state-dir", str(mcp_state), "--json"]) == 0
    app2_json = parse_cli_json(capsys.readouterr().out)
    assert app2_json["action"] == "submit"

    att_mcp = ts.submit_approved(stage_mcp["digest"])
    assert att_mcp["state"] == "SUBMITTED" and att_mcp["attempt_id"] == "654321"
    assert len(sbatch_log.read_text(encoding="utf-8").splitlines()) == 2

    # 3. CLI (`aimip plan` -> `aimip run` -> `aimip submit --expect-digest` AND interactive `aimip run --approve`)
    assert main(["plan", "dlesym-v1.0", "--start", "1995-01-01", "--steps", "8", "--name", "cli-dl-2step", "--json"]) == 0
    cli_pre = parse_cli_json(capsys.readouterr().out)
    assert cli_pre["requires_staging"] is True

    assert main(["run", "dlesym-v1.0", "--start", "1995-01-01", "--steps", "8", "--name", "cli-dl-2step", "--submit", "--json"]) == 0
    cli_prep = parse_cli_json(capsys.readouterr().out)
    assert cli_prep["digest"] != cli_pre["digest"]

    # Prospective digest is refused by `aimip submit --expect-digest`!
    assert main(["submit", f"--expect-digest={cli_pre['digest']}", "cli-dl-2step"]) == 4
    capsys.readouterr()
    assert len(sbatch_log.read_text(encoding="utf-8").splitlines()) == 2

    # Final post-stage digest succeeds!
    assert main(["submit", f"--expect-digest={cli_prep['digest']}", "cli-dl-2step", "--json"]) == 0
    cli_sub = parse_cli_json(capsys.readouterr().out)
    assert cli_sub["state"] == "SUBMITTED"
    assert len(sbatch_log.read_text(encoding="utf-8").splitlines()) == 3

    # Interactive `aimip run ... --approve` on `/dev/tty`
    with _interactive_two_step_pty() as (tty_path, tr_cli):
        monkeypatch.setenv("AIMIP_TTY_PATH", tty_path)
        assert main(["run", "dlesym-v1.0", "--start", "1995-01-01", "--steps", "8", "--name", "cli-dl-tty", "--approve", "--json"]) == 0
    cli_tty_res = parse_cli_json(capsys.readouterr().out)
    assert cli_tty_res["state"] == "SUBMITTED"
    assert tr_cli["prepare_prefix"] != tr_cli["submit_prefix"]
    assert len(sbatch_log.read_text(encoding="utf-8").splitlines()) == 4

    # 4. UI (`UiApprovalManager`: prepare-only nonce refused by `execute_submit`, succeeds via `execute_prepare` -> 2nd `approve` -> `execute_submit`)
    ui_mgr = UiApprovalManager()
    ui_spec = RunSpec(model="dlesym-v1.0", start="1995-01-01T00:00:00", steps=8, name="ui-dl-2step")
    ui_pre = svc.plan(ui_spec)
    ui_prop1 = ui_mgr.register_submit_proposal(
        spec=ui_spec,
        resolved=ui_pre,
        card_text=svc.render_plan_card(ui_pre, is_tty=False),
    )
    assert ui_prop1["action"] == "prepare"

    # Requesting `action='submit'` on a prospective plan is refused
    with pytest.raises(ApprovalError, match="prepare"):
        ui_mgr.approve(digest_prefix=ui_prop1["approval_prefix"], action="submit", allow_download=True)

    # Mint `'prepare'` token and verify `execute_submit` refuses it and burns it
    ui_tok_bad = ui_mgr.approve(digest_prefix=ui_prop1["approval_prefix"], action="prepare", allow_download=True)
    with pytest.raises(ApprovalError, match="Prepare-only"):
        ui_mgr.execute_submit(nonce=ui_tok_bad["authorization_nonce"])
    assert len(sbatch_log.read_text(encoding="utf-8").splitlines()) == 4

    # Re-register, run `execute_prepare`, approve final digest, and `execute_submit`
    ui_prop1b = ui_mgr.register_submit_proposal(
        spec=ui_spec,
        resolved=ui_pre,
        card_text=svc.render_plan_card(ui_pre, is_tty=False),
    )
    ui_tok_prep = ui_mgr.approve(digest_prefix=ui_prop1b["approval_prefix"], action="prepare", allow_download=True)
    ui_staged = ui_mgr.execute_prepare(nonce=ui_tok_prep["authorization_nonce"])
    assert ui_staged["proposal"]["action"] == "submit"
    assert ui_staged["digest"] != ui_pre.digest
    ui_tok_sub = ui_mgr.approve(digest_prefix=ui_staged["approval_prefix"], action="submit")
    ui_att = ui_mgr.execute_submit(nonce=ui_tok_sub["authorization_nonce"])
    assert ui_att["state"] == "SUBMITTED"
    assert len(sbatch_log.read_text(encoding="utf-8").splitlines()) == 5


def test_authorized_prepare_rejects_inline_local_execution_and_burns_token(
    playground_env, tmp_path: Path, allow_tmp_run_root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Authorization-controlled `prepare()` forbids `execute_local=True` (both staging and non-staging tokens), invokes zero adapter executions, and burns the rejected token."""
    from aimip.families import EmuFamilyAdapter, FmeFamilyAdapter

    adapter_exec_calls: list[str] = []
    monkeypatch.setattr(
        EmuFamilyAdapter,
        "execute",
        lambda self_adapter, out_dir: adapter_exec_calls.append(out_dir) or {"out_dir": out_dir},
    )
    monkeypatch.setattr(
        FmeFamilyAdapter,
        "execute",
        lambda self_adapter, out_dir: adapter_exec_calls.append(out_dir) or {"out_dir": out_dir},
    )

    # 1. Synthetic staging model (`dlesym-v1.0`) with `mode="local"` and a real broker-minted prepare token
    with monkeypatch.context() as mp:
        env = _setup_synthetic_dlesym_env(tmp_path / "dl_loc", allow_tmp_run_root, mp)
        dl_spec = RunSpec(
            model="dlesym-v1.0",
            start="1995-01-01T00:00:00",
            steps=8,
            name="dl-inline-local-refused",
            mode="local",
        )
        dl_plan = svc.plan(dl_spec)
        assert dl_plan.requires_staging is True

        dl_broker = MemoryApprovalBroker()
        dl_broker.register_proposal(
            action="prepare",
            digest=dl_plan.digest,
            card_text=svc.render_plan_card(dl_plan, is_tty=False),
            card_data=dl_plan.to_dict(),
            missing_bytes=dl_plan.missing_bytes,
            missing_count=len(dl_plan.missing_assets),
        )
        with pty_responder(dl_plan.approval_prefix, "yes") as (tty_path, _):
            dl_prep_auth, _ = dl_broker.approve_pending_on_tty(tty_path=tty_path)
        assert dl_prep_auth.action == "prepare"

        with pytest.raises(ApprovalError, match="execute_local"):
            svc.prepare(
                dl_spec,
                authorization=dl_prep_auth,
                require_authorization=True,
                broker=dl_broker,
                execute_local=True,
            )

        assert adapter_exec_calls == []
        assert not env["run_counter"].exists()
        assert not env["stage_counter"].exists()
        assert not Path(dl_plan.output_dir).exists()

        # Rejected token was burned — cannot be reused even with `execute_local=False`!
        with pytest.raises(ApprovalError, match="already used, revoked, or not minted"):
            svc.prepare(
                dl_spec,
                authorization=dl_prep_auth,
                require_authorization=True,
                broker=dl_broker,
                execute_local=False,
            )

    # 2. Non-staging model (`ace2-era5`) with `mode="local"` and a broker-minted prepare token
    fme_spec = RunSpec(
        model="ace2-era5",
        start="1979-01-01T00:00:00",
        steps=4,
        name="fme-inline-local-refused",
        mode="local",
    )
    fme_plan = svc.plan(fme_spec)
    assert fme_plan.requires_staging is False

    fme_broker = MemoryApprovalBroker()
    fme_broker.register_proposal(
        action="prepare",
        digest=fme_plan.digest,
        card_text=svc.render_plan_card(fme_plan, is_tty=False),
        card_data=fme_plan.to_dict(),
        missing_bytes=0,
        missing_count=0,
    )
    with pty_responder(fme_plan.approval_prefix) as (tty_path, _):
        fme_prep_auth, _ = fme_broker.approve_pending_on_tty(tty_path=tty_path)
    assert fme_prep_auth.action == "prepare"

    with pytest.raises(ApprovalError, match="execute_local"):
        svc.prepare(
            fme_spec,
            authorization=fme_prep_auth,
            require_authorization=True,
            broker=fme_broker,
            execute_local=True,
        )

    assert adapter_exec_calls == []
    assert not Path(fme_plan.output_dir).exists()

    with pytest.raises(ApprovalError, match="already used, revoked, or not minted"):
        svc.prepare(
            fme_spec,
            authorization=fme_prep_auth,
            require_authorization=True,
            broker=fme_broker,
            execute_local=False,
        )


def test_output_dir_binds_canonical_physical_path_across_ancestor_and_run_root_alias_retargeting(
    tmp_path: Path, allow_tmp_run_root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Prospective plans bind the canonical `realpath` output destination; retargeting an `--out` or `run_root` alias after approval is rejected before writes (or submits exclusively via the bound physical path)."""
    env = _setup_synthetic_dlesym_env(tmp_path, allow_tmp_run_root, monkeypatch)
    dir_a = env["runs"] / "phys_a"
    dir_b = env["runs"] / "phys_b"
    dir_a.mkdir()
    dir_b.mkdir()
    real_a = str(dir_a.resolve())
    real_b = str(dir_b.resolve())

    # 1. `--out <rw-root>/out_alias/new-run` retargeted between approval and preparation -> rejected before writes!
    out_alias = env["runs"] / "out_alias"
    out_alias.symlink_to(dir_a, target_is_directory=True)

    # Passing a symlink as the leaf `--out` run directory itself is strictly refused
    with pytest.raises(ValidationError, match="refusing a symlinked run directory"):
        svc.plan(
            RunSpec(
                model="dlesym-v1.0",
                start="1995-01-01T00:00:00",
                steps=8,
                out=str(out_alias),
            )
        )

    spec_out = RunSpec(
        model="dlesym-v1.0",
        start="1995-01-01T00:00:00",
        steps=8,
        out=str(out_alias / "alias-out-run"),
    )
    pre_out = svc.plan(spec_out)
    assert pre_out.output_dir == f"{real_a}/alias-out-run"

    now = time.time()
    broker = MemoryApprovalBroker()
    auth_out = Authorization(
        action="prepare",
        digest=pre_out.digest,
        user=current_user(),
        issued_at=now,
        expires_at=now + 600,
        nonce="out-alias-prep-1",
        allow_download=True,
    )
    broker.store_minted(auth_out)

    # Retarget `out_alias` from `dir_a` to `dir_b` after approval, before preparation
    out_alias.unlink()
    out_alias.symlink_to(dir_b, target_is_directory=True)

    with pytest.raises(ApprovalError, match="Plan changed"):
        svc.prepare(spec_out, authorization=auth_out, require_authorization=True, broker=broker)

    assert not (dir_a / "alias-out-run").exists()
    assert not (dir_b / "alias-out-run").exists()
    assert not env["stage_counter"].exists()

    # Token was burned even if alias is restored to `dir_a`
    out_alias.unlink()
    out_alias.symlink_to(dir_a, target_is_directory=True)
    with pytest.raises(ApprovalError):
        svc.prepare(spec_out, authorization=auth_out, require_authorization=True, broker=broker)

    # 2. Configured run-root alias (`AIMIP_DEFAULT_OUT_ROOT` -> `dir_a`, retargeted -> `dir_b`) for BOTH `--name` and unnamed default output
    root_alias = env["runs"] / "root_alias"
    root_alias.symlink_to(dir_a, target_is_directory=True)
    monkeypatch.setenv("AIMIP_DEFAULT_OUT_ROOT", str(root_alias))

    spec_named = RunSpec(model="dlesym-v1.0", start="1995-01-01T00:00:00", steps=8, name="named-under-alias")
    spec_unnamed = RunSpec(model="dlesym-v1.0", start="1995-01-01T00:00:00", steps=8)
    plan_named = svc.plan(spec_named)
    plan_unnamed = svc.plan(spec_unnamed)
    assert plan_named.output_dir == f"{real_a}/named-under-alias"
    assert plan_unnamed.output_dir.startswith(f"{real_a}/dlesym-v1.0-")

    auth_named = Authorization(
        action="prepare",
        digest=plan_named.digest,
        user=current_user(),
        issued_at=now,
        expires_at=now + 600,
        nonce="root-alias-named",
        allow_download=True,
    )
    auth_unnamed = Authorization(
        action="prepare",
        digest=plan_unnamed.digest,
        user=current_user(),
        issued_at=now,
        expires_at=now + 600,
        nonce="root-alias-unnamed",
        allow_download=True,
    )
    broker.store_minted(auth_named)
    broker.store_minted(auth_unnamed)

    root_alias.unlink()
    root_alias.symlink_to(dir_b, target_is_directory=True)

    with pytest.raises(ApprovalError, match="Plan changed"):
        svc.prepare(spec_named, authorization=auth_named, require_authorization=True, broker=broker)
    with pytest.raises(ApprovalError, match="Plan changed"):
        svc.prepare(spec_unnamed, authorization=auth_unnamed, require_authorization=True, broker=broker)

    assert list(dir_a.iterdir()) == []
    assert list(dir_b.iterdir()) == []
    assert not env["stage_counter"].exists()

    # 3. Retargeting `root_alias` AFTER Step 1 `prepare()` completes continues exclusively via the originally approved physical `dir_a` path!
    root_alias.unlink()
    root_alias.symlink_to(dir_a, target_is_directory=True)

    spec_bound = RunSpec(model="dlesym-v1.0", start="1995-01-01T00:00:00", steps=8, name="bound-phys-ok")
    plan_bound = svc.plan(spec_bound)
    auth_bound_prep = Authorization(
        action="prepare",
        digest=plan_bound.digest,
        user=current_user(),
        issued_at=now,
        expires_at=now + 600,
        nonce="bound-phys-prep",
        allow_download=True,
    )
    broker.store_minted(auth_bound_prep)
    prepared_bound = svc.prepare(
        spec_bound,
        authorization=auth_bound_prep,
        require_authorization=True,
        broker=broker,
    )
    assert prepared_bound.run_dir == f"{real_a}/bound-phys-ok"

    # Retarget `root_alias` to `dir_b` before final Step 2 submission
    root_alias.unlink()
    root_alias.symlink_to(dir_b, target_is_directory=True)

    submitted_dirs: list[str] = []
    monkeypatch.setattr(
        core,
        "submit_run",
        lambda rd, **kw: submitted_dirs.append(rd) or fake_attempt(rd, **kw),
    )
    monkeypatch.setattr(core.shutil, "which", lambda name: "/fake/" + name)

    auth_bound_sub = Authorization(
        action="submit",
        digest=prepared_bound.plan.digest,
        user=current_user(),
        issued_at=now,
        expires_at=now + 600,
        nonce="bound-phys-sub",
        allow_download=False,
    )
    broker.store_minted(auth_bound_sub)
    att = svc.submit(
        prepared_bound,
        authorization=auth_bound_sub,
        require_authorization=True,
        broker=broker,
    )
    assert att.run_dir == f"{real_a}/bound-phys-ok"
    assert submitted_dirs == [f"{real_a}/bound-phys-ok"]
    assert list(dir_b.iterdir()) == []


def test_run_approve_no_fetch_refuses_prospective_staging_and_missing_pinned_assets(
    tmp_path: Path, allow_tmp_run_root: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """`aimip run --approve --no-fetch` refuses both prospective staging and missing pinned packages with zero fetch/stage calls, even when the TTY answers 'yes'."""
    import aimip.setup_data as setup_data_mod
    from aimip.setup_data import SetupYearConfig
    from test_models_and_fetch import _populate_extra_fme_data_tree_and_lock

    # 1. Prospective staging (`dlesym-v1.0`) with `aimip run ... --approve --no-fetch`
    with monkeypatch.context() as mp:
        env = _setup_synthetic_dlesym_env(tmp_path / "nf_stage", allow_tmp_run_root, mp)
        dl_spec = RunSpec(
            model="dlesym-v1.0",
            start="1995-01-01T00:00:00",
            steps=8,
            name="nofetch-prospective",
        )
        dl_plan = svc.plan(dl_spec)

        with pty_responder(dl_plan.approval_prefix, "yes") as (tty_path, _):
            mp.setenv("AIMIP_TTY_PATH", tty_path)
            rc = main(
                [
                    "run",
                    "dlesym-v1.0",
                    "--start",
                    "1995-01-01",
                    "--steps",
                    "8",
                    "--name",
                    "nofetch-prospective",
                    "--approve",
                    "--no-fetch",
                    "--json",
                ]
            )
        assert rc == 3
        err_doc = json.loads(capsys.readouterr().out)
        assert err_doc["ok"] is False
        assert err_doc["error"]["code"] == "missing_assets"
        assert "--no-fetch" in err_doc["error"]["message"]
        assert not env["stage_counter"].exists()
        assert not Path(dl_plan.output_dir).exists()

        # Direct service check: `allow_download=True` on an `Authorization` NEVER overrides `allow_fetch=False`!
        now = time.time()
        broker = MemoryApprovalBroker()
        consent_auth = Authorization(
            action="prepare",
            digest=dl_plan.digest,
            user=current_user(),
            issued_at=now,
            expires_at=now + 600,
            nonce="consent-vs-nofetch",
            allow_download=True,
        )
        broker.store_minted(consent_auth)
        with pytest.raises(MissingAssetsError, match="--no-fetch"):
            svc.prepare(
                dl_spec,
                allow_fetch=False,
                authorization=consent_auth,
                require_authorization=True,
                broker=broker,
            )
        assert not env["stage_counter"].exists()

    # 2. Missing pinned package (`ace2-som` unfetched in `$AIMIP_DATA`) with `aimip run ... --approve --no-fetch`
    with monkeypatch.context() as mp:
        yc = SetupYearConfig(som_years=(2032, 2033), hiro_native_years=(2014, 2015))
        mp.setattr(setup_data_mod, "SetupYearConfig", lambda: yc)
        hf = tmp_path / "hf_nf"
        lock = tmp_path / "nf.lock"
        _populate_extra_fme_data_tree_and_lock(hf, yc, lock)

        fetch_calls: list[str] = []
        mp.setattr(
            setup_data_mod,
            "fetch_model",
            lambda *a, **kw: fetch_calls.append("called"),
        )

        pg_dir = tmp_path / "nf_pg"
        (pg_dir / "bin").mkdir(parents=True)
        (pg_dir / "bin" / "aimip").write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
        (pg_dir / "bin" / "aimip").chmod(0o755)
        runs = allow_tmp_run_root / "nf_runs"
        runs.mkdir()
        data_dir = tmp_path / "nf_data"  # intentionally unfetched!
        (pg_dir / "site.json").write_text(
            json.dumps(
                {
                    "name": "cluster",
                    "scheduler": "slurm",
                    "runtime": "native",
                    "sbatch": {
                        "partition": "gpu",
                        "account": "acct",
                        "gres": "gpu:1",
                        "cpus_per_task": 8,
                        "mem": "32G",
                        "max_time": "08:00:00",
                    },
                    "run_root": str(runs),
                    "extra_ro_binds": [],
                    "gpu_check": "nvidia-smi",
                }
            ),
            encoding="utf-8",
        )
        mp.delenv("AIMIP_REGISTRY", raising=False)
        mp.setenv("AIMIP_PLAYGROUND", str(pg_dir))
        mp.setenv("AIMIP_SITE_CONFIG", str(pg_dir / "site.json"))
        mp.setenv("AIMIP_DATA", str(data_dir))
        mp.setenv("AIMIP_LOCK", str(lock))
        mp.setenv("AIMIP_HF_MIRROR", f"file://{hf}")
        mp.setenv("AIMIP_DEFAULT_OUT_ROOT", str(runs))
        mp.setenv("AIMIP_DATA_MODE", "rw")

        som_spec = RunSpec(
            model="ace2-som",
            start="2032-01-01T00:00:00",
            steps=4,
            forcing_set="1xCO2",
            name="nofetch-pinned",
        )
        som_plan = svc.plan(som_spec)
        assert som_plan.missing_assets

        with pty_responder(som_plan.approval_prefix, "yes") as (tty_path, _):
            mp.setenv("AIMIP_TTY_PATH", tty_path)
            rc_pinned = main(
                [
                    "run",
                    "ace2-som",
                    "--start",
                    "2032-01-01",
                    "--steps",
                    "4",
                    "--forcing-set",
                    "1xCO2",
                    "--name",
                    "nofetch-pinned",
                    "--approve",
                    "--no-fetch",
                    "--json",
                ]
            )
        assert rc_pinned == 3
        pinned_err = json.loads(capsys.readouterr().out)
        assert pinned_err["ok"] is False
        assert pinned_err["error"]["code"] == "missing_assets"
        assert "--no-fetch" in pinned_err["error"]["message"]
        assert pinned_err["error"]["next_command"] == "aimip fetch -m ace2-som"
        assert fetch_calls == []
        assert not data_dir.exists()
        assert not Path(som_plan.output_dir).exists()
