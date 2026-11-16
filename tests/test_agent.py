"""Offline agent tests: real LangGraph + scripted model + `/dev/tty` 8-hex digest approval."""

from __future__ import annotations

import contextlib
import io
import json
import os
from pathlib import Path
import socket
import sys
from types import SimpleNamespace

import pytest

pytest.importorskip("langgraph")
pytest.importorskip("langchain_core")

from langchain_core.messages import AIMessage

from aimip import services
from aimip.agent import disable_tracing
from aimip.agent.chat import ChatAgent, Transcript, read_api_key, repl
from aimip.agent.tools import AgentTools, cli_json
from aimip.cli import main as cli_main
from aimip.errors import ApprovalError, MissingAssetsError
from aimip.services.types import RunSpec
from conftest import pty_responder, write_fake_ic_netcdf


@pytest.fixture(autouse=True)
def offline(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("Network is forbidden in agent tests")

    monkeypatch.setattr(socket.socket, "connect", forbidden)
    monkeypatch.setattr(socket, "create_connection", forbidden)
    for key in ("GEMINI_API_KEY", "GOOGLE_API_KEY", "AIMIP_GEMINI_KEY_FILE"):
        monkeypatch.delenv(key, raising=False)


@pytest.fixture
def fake_slurm(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict[str, Path]:
    """Install offline fake `sbatch`, `squeue`, `sacct`, and `scancel` binaries on PATH."""
    fbin = tmp_path / "fake_slurm_bin"
    fbin.mkdir(exist_ok=True)
    sbatch_log = tmp_path / "sbatch_calls.log"
    scancel_log = tmp_path / "scancel_calls.log"

    sbatch = fbin / "sbatch"
    sbatch.write_text(
        "#!/bin/bash\n"
        f'echo "$@" >> "{sbatch_log}"\n'
        "echo 818181\n",
        encoding="utf-8",
    )
    sbatch.chmod(0o755)

    squeue = fbin / "squeue"
    squeue.write_text("#!/bin/bash\nexit 0\n", encoding="utf-8")
    squeue.chmod(0o755)

    sacct = fbin / "sacct"
    sacct.write_text("#!/bin/bash\necho '818181|RUNNING|00:01:00|0:0'\n", encoding="utf-8")
    sacct.chmod(0o755)

    scancel = fbin / "scancel"
    scancel.write_text(
        "#!/bin/bash\n"
        f'echo "$@" >> "{scancel_log}"\n',
        encoding="utf-8",
    )
    scancel.chmod(0o755)

    monkeypatch.setenv("PATH", f"{fbin}:{os.environ.get('PATH', '')}")
    return {"bin": fbin, "sbatch_log": sbatch_log, "scancel_log": scancel_log}


def in_process_cli(args):
    output, error = io.StringIO(), io.StringIO()
    with contextlib.redirect_stdout(output), contextlib.redirect_stderr(error):
        rc = cli_main([*args, "--json"])
    if rc:
        raise ValueError(error.getvalue() or output.getvalue())
    doc = json.loads(output.getvalue())
    if isinstance(doc, dict) and doc.get("aimip") == "1" and "result" in doc:
        return doc["result"]
    return doc


class ScriptedModel:
    def __init__(self, *replies):
        self.replies = iter(replies)
        self.seen = []

    def bind_tools(self, tools):
        self.names = {tool.name for tool in tools}
        return self

    def invoke(self, messages):
        self.seen.append(messages)
        return next(self.replies)


def call(name, args=None, id="call-1"):
    return AIMessage(content="", tool_calls=[{"name": name, "args": args or {}, "id": id}])


@pytest.fixture
def backend(playground_env, fake_slurm):
    return AgentTools()


def plan(backend):
    return backend.plan_run("ace2-era5", "1979-01-01T00:00:00", 4)


def agent_for(tmp_path, backend, *replies):
    return ChatAgent(ScriptedModel(*replies), backend, Transcript(tmp_path / "chat-logs"))


def test_real_cli_plan_and_one_use_confirmation(backend):
    card = plan(backend)
    assert card["runs"][0]["end"] == "1979-01-02T00:00:00"
    assert card["gpu_hours_estimate"] > 0
    assert card["runs"][0]["forcing_set"]["name"] == "amip-hist"
    assert card["runs"][0]["sbatch_resources"]["time"]
    assert Path(card["runs"][0]["ic_source"]).is_file()
    assert not Path(card["run_dir"]).exists()

    # Submitting without human `/dev/tty` digest authorization is refused.
    prepared_unauth = services.prepare(RunSpec(model="ace2-era5", start="1979-01-01T00:00:00", steps=4, name="unauth-check"))
    with pytest.raises(ApprovalError, match="human approval"):
        services.submit(prepared_unauth, require_authorization=True, broker=backend.broker)

    # Matching 8-hex digest prefix on `/dev/tty` prepares and submits once; replay fails.
    with pty_responder(card["approval_prefix"]) as (tty_path, _):
        attempt = backend.approve_and_execute_on_tty(tty_path=tty_path)
    assert attempt["scheduler"] == "slurm"
    assert attempt["attempt_id"] == "818181"
    assert (Path(card["run_dir"]) / "plan.json").is_file()
    with pty_responder(card["approval_prefix"]) as (tty_path, _):
        with pytest.raises(ApprovalError, match="No pending"):
            backend.approve_and_execute_on_tty(tty_path=tty_path)


def test_yes_for_a_does_not_authorize_b(backend):
    a = backend.plan_run("ace2-era5", "1979-01-01T00:00:00", 4)
    with pty_responder(a["approval_prefix"]) as (tty_path, _):
        auth_a, _ = backend.broker.approve_pending_on_tty(tty_path=tty_path)

    b = backend.plan_run("ace2-era5", "1979-01-01T00:00:00", 8)
    assert a["run_dir"] != b["run_dir"]
    assert a["digest"] != b["digest"]

    # `auth_a` was revoked when `b` was proposed AND its digest does not match `b`.
    prep_b = services.prepare(RunSpec(model="ace2-era5", start="1979-01-01T00:00:00", steps=8, out=b["run_dir"]))
    with pytest.raises(ApprovalError):
        services.submit(prep_b, authorization=auth_a, require_authorization=True, broker=backend.broker)

    # Typing A's 8-hex prefix when B is pending is refused on `/dev/tty`.
    if a["approval_prefix"] != b["approval_prefix"]:
        with pty_responder(a["approval_prefix"]) as (tty_path, _):
            with pytest.raises(ApprovalError, match="does not match expected digest prefix"):
                backend.approve_and_execute_on_tty(tty_path=tty_path)


def test_wrong_directory_consumes_approval(backend):
    card = backend.plan_run("ace2-era5", "1979-01-01T00:00:00", 4, name="approved-a")
    with pty_responder(card["approval_prefix"]) as (tty_path, _):
        auth, _ = backend.broker.approve_pending_on_tty(tty_path=tty_path)

    prep_a = services.prepare(RunSpec(model="ace2-era5", start="1979-01-01T00:00:00", steps=4, name="approved-a"))
    prep_other = services.prepare(RunSpec(model="ace2-era5", start="1979-01-01T00:00:00", steps=8, name="other-b"))

    # Presenting `auth` against a different run directory fails AND burns the single-use token.
    with pytest.raises(ApprovalError, match="Plan changed|digest mismatch"):
        services.submit(prep_other, authorization=auth, require_authorization=True, broker=backend.broker)
    with pytest.raises(ApprovalError, match="already used, revoked"):
        services.submit(prep_a, authorization=auth, require_authorization=True, broker=backend.broker)


@pytest.mark.parametrize("after_yes", [False, True])
@pytest.mark.parametrize("artifact", ["config.yaml", "job.sbatch", "aimip-manifest.json"])
def test_changed_plan_is_rejected(backend, playground_env, after_yes, artifact):
    spec = RunSpec(
        model="ace2-era5",
        start="1979-01-01T00:00:00",
        steps=4,
        name=f"tamper-{artifact.replace('.', '-')}-{int(after_yes)}",
    )
    if after_yes:
        card = backend.propose_run(spec.model, spec.start, steps=spec.steps, name=spec.name)
        with pty_responder(card["approval_prefix"]) as (tty_path, _):
            auth, _ = backend.broker.approve_pending_on_tty(tty_path=tty_path)
        prepared = services.prepare(spec)
        target = Path(prepared.run_dir) / artifact
        target.write_text(target.read_text(encoding="utf-8") + "\n", encoding="utf-8")
        with pytest.raises(ApprovalError, match="Plan changed"):
            services.submit(prepared, authorization=auth, require_authorization=True, broker=backend.broker)
    else:
        prepared = services.prepare(spec)
        target = Path(prepared.run_dir) / artifact
        target.write_text(target.read_text(encoding="utf-8") + "\n", encoding="utf-8")
        with pytest.raises(ApprovalError, match="Plan changed"):
            services.submit(prepared)


def test_failed_replan_revokes_approval(backend):
    a = plan(backend)
    with pty_responder(a["approval_prefix"]) as (tty_path, _):
        auth_a, _ = backend.broker.approve_pending_on_tty(tty_path=tty_path)
    prep_a = services.prepare(RunSpec(model="ace2-era5", start="1979-01-01T00:00:00", steps=4, out=a["run_dir"]))

    with pytest.raises(ValueError):
        backend.plan_run("ace2-era5", "1500-01-01T00:00:00", 4)

    assert backend.pending is None
    with pytest.raises(ApprovalError, match="already used, revoked"):
        services.submit(prep_a, authorization=auth_a, require_authorization=True, broker=backend.broker)


def test_plan_card_cannot_be_mutated_by_caller(backend):
    card = plan(backend)
    card["run_dir"] = "/some/other/run"
    card["plan"]["output_dir"] = "/some/other/run"
    assert "/some/other/run" not in json.dumps(backend.pending)


def test_spin_card_has_all_members_and_hashes_them(backend):
    card = backend.plan_spin(
        "ace2-era5",
        ["1979-01-01T00:00:00", "1979-01-02T00:00:00"],
        "1979-01-03T00:00:00",
    )
    assert [run["steps"] for run in card["runs"]] == [8, 4]
    assert all(run["members"] == 1 for run in card["runs"])
    (Path(card["run_dir"]) / "1" / "config.yaml").write_text("changed", encoding="utf-8")
    with pty_responder(card["approval_prefix"]) as (tty_path, _):
        with pytest.raises(ApprovalError, match="Plan changed"):
            backend.approve_and_execute_on_tty(tty_path=tty_path)


def test_scripted_model_cannot_approve_or_submit(tmp_path, backend):
    card = plan(backend)
    agent = agent_for(
        tmp_path,
        backend,
        call("submit", {"run_dir": card["run_dir"]}),
        AIMessage(content="Refused"),
    )
    assert agent.turn("yes, submit whatever you like") == "Refused"
    result = json.loads(agent.history[-1][2].content)
    assert "Unknown or forbidden tool: submit" in result["error"]
    assert "submit" not in agent.tools
    assert "approve" not in agent.tools
    assert "confirm" not in agent.tools
    assert "present" not in agent.tools


def test_repl_displays_authoritative_card_and_yes_emits_command(tmp_path, backend):
    preview = backend.plan("ace2-era5", "1979-01-01T00:00:00", steps=4)
    agent = agent_for(
        tmp_path,
        backend,
        call("plan_run", {"model": "ace2-era5", "start": "1979-01-01T00:00:00", "steps": 4}),
        AIMessage(content="Here is the plan."),
    )
    lines = iter(["plan a run", "quit"])
    output = []
    with pty_responder(preview["approval_prefix"]) as (tty_path, drain_tty):
        assert repl(agent, read=lambda _: next(lines), write=output.append, tty_path=tty_path) == 0
        tty_text = drain_tty()
    assert preview["run_dir"] in tty_text
    assert f"[{preview['approval_prefix']}]" in tty_text
    assert any("Submitted Slurm job 818181" in line for line in output)
    assert backend.pending is None


@pytest.mark.parametrize("answer", ["y", "yes", "sure", "yes please", "00000000", "no"])
def test_only_explicit_yes_confirms(backend, answer):
    card = plan(backend)
    bad = "ffffffff" if answer == card["approval_prefix"] else answer
    with pty_responder(bad) as (tty_path, _):
        with pytest.raises(ApprovalError, match="does not match expected digest prefix"):
            backend.approve_and_execute_on_tty(tty_path=tty_path)


def test_tool_errors_reach_model(tmp_path, backend):
    agent = agent_for(
        tmp_path,
        backend,
        call("list_ics", {"model": "does-not-exist"}),
        AIMessage(content="Unknown model"),
    )
    assert agent.turn("inspect this model") == "Unknown model"
    assert "error" in json.loads(agent.history[-1][2].content)


def test_budget_counts_parallel_tool_calls_and_errors(tmp_path):
    calls = []
    backend = AgentTools(cli=lambda args: calls.append(args) or [])
    reply = AIMessage(
        content="",
        tool_calls=[{"name": "list_models", "args": {}, "id": str(i)} for i in range(15)],
    )
    agent = agent_for(tmp_path, backend, reply, call("list_models"), AIMessage(content="Done"))
    assert "12-tool-call limit" in agent.turn("list many times")
    assert len(calls) == 12
    assert len([m for m in agent.history[-1] if m.type == "tool"]) == 15
    # Counter resets on each human turn, including turns after a limit.
    assert agent.turn("one more") == "Done"
    assert len(calls) == 13


def test_sequential_budget_stops_graph_loop(tmp_path):
    backend = AgentTools(cli=lambda _: [])
    agent = agent_for(tmp_path, backend, *(call("list_models", id=str(i)) for i in range(12)))
    assert "12-tool-call limit" in agent.turn("keep calling")


@pytest.mark.parametrize("mode", [0o644, 0o640, 0o400, 0o700])
def test_key_file_rejects_wrong_modes(tmp_path, monkeypatch, mode):
    path = tmp_path / "fake-key"
    path.write_text("test-key")
    path.chmod(mode)
    monkeypatch.setenv("AIMIP_GEMINI_KEY_FILE", str(path))
    with pytest.raises(ValueError, match="mode 0600"):
        read_api_key()


def test_key_owner_symlink_and_environment_precedence(tmp_path, monkeypatch):
    path = tmp_path / "fake-key"
    path.write_text("file-test-key\n")
    path.chmod(0o600)
    monkeypatch.setenv("AIMIP_GEMINI_KEY_FILE", str(path))
    assert read_api_key() == "file-test-key"
    with monkeypatch.context() as patch:
        patch.setattr(os, "getuid", lambda: path.stat().st_uid + 1)
        with pytest.raises(ValueError, match="owned by the caller"):
            read_api_key()
    link = tmp_path / "link"
    link.symlink_to(path)
    monkeypatch.setenv("AIMIP_GEMINI_KEY_FILE", str(link))
    with pytest.raises(OSError):
        read_api_key()
    monkeypatch.setenv("GOOGLE_API_KEY", "google-test-key")
    assert read_api_key() == "google-test-key"
    monkeypatch.setenv("GEMINI_API_KEY", "gemini-test-key")
    assert read_api_key() == "gemini-test-key"


def test_log_redacts_keys_and_is_private(tmp_path):
    secret = 'synthetic"key'
    log = Transcript(tmp_path / "chat-logs", [secret])
    log.write("human", {"text": f"My key is {secret}", "nested": [secret]})
    path = next((tmp_path / "chat-logs").glob("*.jsonl"))
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    assert rows[-1]["value"]["nested"] == ["[REDACTED]"]
    assert path.stat().st_mode & 0o777 == 0o600


def test_trace_environment_removed(monkeypatch):
    monkeypatch.setenv("LANGCHAIN_TRACING_V2", "true")
    monkeypatch.setenv("LANGSMITH_API_KEY", "fake")
    disable_tracing()
    assert "LANGCHAIN_TRACING_V2" not in os.environ
    assert "LANGSMITH_API_KEY" not in os.environ


def test_cli_subprocess_uses_argv_and_scrubs_secrets(monkeypatch):
    def run(argv, **kwargs):
        assert isinstance(argv, list)
        assert argv[:4] == [sys.executable, "-I", "-B", "-m"]
        assert argv[4:] == ["aimip", "models", "--json"]
        assert kwargs.get("shell", False) is False
        assert "GEMINI_API_KEY" not in kwargs["env"]
        return SimpleNamespace(returncode=0, stdout='[{"name":"test"}]', stderr="")

    monkeypatch.setenv("GEMINI_API_KEY", "fake")
    monkeypatch.setattr("aimip.agent.tools.subprocess.run", run)
    assert cli_json(["models"]) == [{"name": "test"}]


def test_cli_errors_are_visible(monkeypatch):
    monkeypatch.setattr(
        "aimip.agent.tools.subprocess.run",
        lambda *a, **k: SimpleNamespace(returncode=2, stderr="bad window", stdout=""),
    )
    with pytest.raises(ValueError, match="bad window"):
        cli_json(["models"])


def _isolated_python_with_aimip(tmp_path: Path) -> str:
    """Return `sys.executable` if `-I` already imports `aimip`, else a stdlib-only venv with `.pth` to `sys.path`."""
    import subprocess as sp
    import venv

    probe = sp.run(
        [sys.executable, "-I", "-c", "import aimip, netCDF4"],
        capture_output=True,
        check=False,
    )
    if probe.returncode == 0:
        return sys.executable

    vdir = tmp_path / "isolated_venv"
    venv.EnvBuilder(with_pip=False).create(vdir)
    vpy = vdir / "bin" / "python"
    purelib = sp.check_output(
        [str(vpy), "-I", "-c", "import sysconfig; print(sysconfig.get_path('purelib'))"],
        text=True,
    ).strip()
    abs_paths = [p for p in sys.path if p and os.path.isabs(p)]
    (Path(purelib) / "aimip_test_deps.pth").write_text("\n".join(abs_paths) + "\n", encoding="utf-8")
    return str(vpy)


def test_real_cli_subprocess_plan_and_status(playground_env, fake_slurm, monkeypatch, tmp_path):
    monkeypatch.setattr(sys, "executable", _isolated_python_with_aimip(tmp_path))
    # Stage shadow `aimip` packages in both CWD and PYTHONPATH: `-I` must ignore both.
    for shadow_name in ("shadow_cwd", "shadow_pypath"):
        pkg = tmp_path / shadow_name / "aimip"
        pkg.mkdir(parents=True)
        (pkg / "__init__.py").write_text("raise RuntimeError('SHADOWED_AIMIP')\n", encoding="utf-8")
        (pkg / "__main__.py").write_text("raise RuntimeError('SHADOWED_AIMIP')\n", encoding="utf-8")
    monkeypatch.chdir(tmp_path / "shadow_cwd")
    monkeypatch.setenv("PYTHONPATH", str(tmp_path / "shadow_pypath"))
    monkeypatch.setenv("AIMIP_RUN_ROOTS", str(tmp_path / "scratch"))

    assert cli_json(["models"])[0]["name"] == "ace2-era5"
    backend = AgentTools()
    assert backend.list_models()[0]["name"] == "ace2-era5"
    assert backend.list_ics("ace2-era5")["ic_files"]
    card = plan(backend)
    with pty_responder(card["approval_prefix"]) as (tty_path, _):
        backend.approve_and_execute_on_tty(tty_path=tty_path)
    assert backend.status(card["run_dir"])["model"] == "ace2-era5"
    assert cli_json(["status", card["run_dir"]])["model"] == "ace2-era5"


def test_scripted_replan_then_yes_selects_only_last_plan(tmp_path, backend):
    args = {"model": "ace2-era5", "start": "1979-01-01T00:00:00", "steps": 4}
    agent = agent_for(
        tmp_path,
        backend,
        call("plan_run", args),
        AIMessage(content="First"),
        call("plan_run", {**args, "steps": 8}),
        AIMessage(content="Second"),
    )
    agent.turn("plan A")
    first = backend.pending
    agent.turn("change to plan B")
    second = backend.pending
    assert first["digest"] != second["digest"]

    with pty_responder(second["digest"][:8]) as (tty_path, _):
        result = agent.approve_on_tty(tty_path=tty_path)
    assert result["run_dir"] == second["card_data"]["output_dir"]
    assert result["run_dir"] != first["card_data"]["output_dir"]


def test_provider_failure_clears_pending_plan(tmp_path, backend):
    # Script ends after the tool call, causing the next model invocation to fail.
    agent = agent_for(
        tmp_path,
        backend,
        call("plan_run", {"model": "ace2-era5", "start": "1979-01-01T00:00:00", "steps": 4}),
    )
    with pytest.raises(RuntimeError, match="no approval retained"):
        agent.turn("plan")
    assert backend.pending is None


@pytest.mark.parametrize("override", ["chat-logs", "custom-audit", None, ""])
def test_cli_entrypoint_model_options_without_network(tmp_path, monkeypatch, override):
    langchain_google_genai = pytest.importorskip("langchain_google_genai", exc_type=ImportError)
    import aimip.agent.chat as chat

    seen = {}

    def model(**kwargs):
        seen.update(kwargs)
        return ScriptedModel()

    monkeypatch.setattr(langchain_google_genai, "ChatGoogleGenerativeAI", model)
    monkeypatch.setattr(chat, "repl", lambda agent: 0)
    monkeypatch.setenv("AIMIP_HOME", str(tmp_path / "install-prefix"))
    monkeypatch.setenv("AIMIP_DEFAULT_OUT_ROOT", str(tmp_path / "runs"))
    if override is None:
        monkeypatch.delenv("AIMIP_CHAT_LOG_DIR", raising=False)
    else:
        monkeypatch.setenv("AIMIP_CHAT_LOG_DIR", str(tmp_path / override) if override else "")
    expected_dir = tmp_path / override if override else tmp_path / "runs" / "chat-logs"
    monkeypatch.setenv("GEMINI_API_KEY", "fake-test-key")
    monkeypatch.setenv("AIMIP_AGENT_MODEL", "environment-model")
    assert chat.main([]) == 0
    assert seen["model"] == "environment-model"
    assert seen["temperature"] == 0
    assert seen["google_api_key"] == "fake-test-key"
    assert chat.main(["--model", "explicit-model"]) == 0
    assert seen["model"] == "explicit-model"
    assert len(list(expected_dir.glob("*.jsonl"))) == 1
    assert not (tmp_path / "install-prefix").exists()
    assert not (expected_dir / "chat-logs").exists()


def test_missing_log_location_fails_before_model_creation(tmp_path, monkeypatch, capsys):
    import aimip.agent.chat as chat

    monkeypatch.setenv("GEMINI_API_KEY", "fake-test-key")
    monkeypatch.setenv("AIMIP_HOME", str(tmp_path))
    monkeypatch.delenv("AIMIP_CHAT_LOG_DIR", raising=False)
    monkeypatch.delenv("AIMIP_DEFAULT_OUT_ROOT", raising=False)
    assert chat.main([]) == 2
    assert "AIMIP_CHAT_LOG_DIR" in capsys.readouterr().err
    assert not (tmp_path / "chat-logs").exists()


def test_card_science_and_default_multisample_ic(backend, playground_env):
    path = playground_env["runs_dir"] / "custom.nc"
    write_fake_ic_netcdf(path, ["1979-01-01T00:00:00"] * 3, sample_dim="sample")
    card = backend.plan_run("ace2-era5", "1979-01-01T00:00:00", 4, ic=str(path))
    run = card["runs"][0]
    assert run["members"] == 3
    assert run["stochastic"] is False
    assert run["calendar"] == "proleptic_gregorian"
    assert run["ensemble_strategy"] == "multi_sample_ic"
    assert "Custom IC" in run["ic_provenance"]
    assert run["outputs"] == {"monthly": True, "prediction": False}
    assert run["lora_info"] is None
    assert isinstance(run["walltime_capped"], bool)
    assert not Path(card["run_dir"]).exists()
    with pty_responder(card["approval_prefix"]) as (tty_path, _):
        backend.approve_and_execute_on_tty(tty_path=tty_path)
    manifest = json.loads((Path(card["run_dir"]) / "aimip-manifest.json").read_text(encoding="utf-8"))
    assert manifest["seed_explicit"] is False


def test_card_reports_registry_ic_provenance(backend):
    card = backend.plan_run("ace2-eamv3", "1994-01-01T00:00:00", 4)
    assert "1971" in card["runs"][0]["ic_provenance"]
    assert card["runs"][0]["calendar"] == "noleap"


def test_spin_card_uses_combined_walltime_cap(backend, tmp_path, monkeypatch):
    from aimip.site_config import load_site_config

    site = load_site_config().to_dict()
    site["sbatch"]["max_time"] = "00:20:00"
    path = tmp_path / "site.json"
    path.write_text(json.dumps(site))
    monkeypatch.setenv("AIMIP_SITE_CONFIG", str(path))
    card = backend.plan_spin(
        "ace2-era5",
        ["1979-01-01T00:00:00", "1979-01-02T00:00:00"],
        "2001-01-01T00:00:00",
    )
    assert card["walltime_capped"] is True
    assert all(r["outputs"] == {"monthly": False, "prediction": False} for r in card["runs"])


def test_replication_and_inferred_member_limits_precede_writes(backend, playground_env, monkeypatch):
    import aimip.agent.tools as module

    root = playground_env["default_out_root"]
    with pytest.raises(ValueError, match="at most"):
        backend.plan_run("ace2s-shield-plus", "1979-01-01T00:00:00", 4, members=65)
    monkeypatch.setattr(module, "MAX_REPLICATION_BYTES", 1)
    with pytest.raises(ValueError, match="replication estimate"):
        backend.plan_run("ace2s-shield-plus", "1979-01-01T00:00:00", 4, members=2)
    path = playground_env["runs_dir"] / "many.nc"
    write_fake_ic_netcdf(path, ["1979-01-01T00:00:00"] * 65, sample_dim="sample")
    with pytest.raises(ValueError, match="IC implies 65"):
        backend.plan_run("ace2-era5", "1979-01-01T00:00:00", 4, ic=str(path))
    assert not list(root.glob("chat-*"))


def test_failed_cli_plan_removes_only_fresh_directory(backend, playground_env):
    existing = playground_env["default_out_root"] / "keep"
    existing.mkdir()

    def fail(args):
        raise ValueError("CLI failed")

    backend.cli = fail
    with pytest.raises(ValueError, match="CLI failed"):
        plan(backend)
    assert existing.is_dir()
    assert not list(existing.parent.glob("chat-*"))


def test_collect_plan_is_confirmed_host_command(backend):
    spin = backend.plan_spin(
        "ace2-era5",
        ["1979-01-01T00:00:00", "1979-01-02T00:00:00"],
        "1979-01-03T00:00:00",
    )
    for i in range(2):
        write_fake_ic_netcdf(
            Path(spin["run_dir"]) / str(i) / "restart.nc",
            ["1979-01-03T00:00:00"],
            sample_dim="sample",
        )
    card = backend.plan_collect(spin["run_dir"])
    assert card["kind"] == "collect" and card["members"] == 2
    with pty_responder(card["approval_prefix"]) as (tty_path, _):
        result = backend.approve_and_execute_on_tty(tty_path=tty_path)
    assert result["host_command"] == f"aimip collect {spin['run_dir']} --out {card['output']}"
    assert result["submitted"] is False
    assert not Path(card["output"]).exists()
    # Execute only the offline CLI collection of synthetic files to validate the command contract.
    merged = in_process_cli(["collect", spin["run_dir"], "--out", card["output"]])
    assert merged["n_samples"] == 2


def test_collect_refuses_missing_restarts_and_large_spin(backend):
    with pytest.raises(ValueError, match="1–24"):
        backend.plan_spin("ace2-era5", ["1979-01-01T00:00:00"] * 25, "1979-01-03T00:00:00")
    spin = backend.plan_spin("ace2-era5", ["1979-01-01T00:00:00"], "1979-01-02T00:00:00")
    with pytest.raises(ValueError, match="not ready"):
        backend.plan_collect(spin["run_dir"])
    assert backend.pending is None


@pytest.mark.parametrize("method", ["lora_info", "plan_run", "plan_spin"])
def test_unsafe_lora_never_reaches_cli(playground_env, method):
    checkpoint = playground_env["runs_dir"] / "unsafe.tar"
    checkpoint.write_bytes(b"not a pickle")
    checkpoint.chmod(0o666)
    calls = []
    backend = AgentTools(cli=lambda args: calls.append(args))
    with pytest.raises(ValueError, match="group/world-writable"):
        if method == "lora_info":
            backend.lora_info(str(checkpoint))
        elif method == "plan_run":
            backend.plan_run("ace2-era5", "1979-01-01T00:00:00", 4, lora=str(checkpoint))
        else:
            backend.plan_spin(
                "ace2-era5",
                ["1979-01-01T00:00:00"],
                "1979-01-02T00:00:00",
                lora=str(checkpoint),
            )
    assert calls == []


def test_lora_trust_ownership_symlinks_and_resolved_ancestry(playground_env, monkeypatch):
    from aimip.agent.security import trusted_lora_path

    checkpoint = playground_env["runs_dir"] / "owned.tar"
    checkpoint.write_bytes(b"not a pickle")
    assert trusted_lora_path(str(checkpoint)) == str(checkpoint.resolve())
    link = checkpoint.with_name("link.tar")
    link.symlink_to(checkpoint)
    with pytest.raises(OSError):
        trusted_lora_path(str(link))
    actual_uid = os.getuid()
    with monkeypatch.context() as patch:
        patch.setattr(os, "getuid", lambda: actual_uid + 10000)
        with pytest.raises(ValueError, match="Untrusted|caller-owned"):
            trusted_lora_path(str(checkpoint))
        patch.setenv("AIMIP_DATA", str(playground_env["runs_dir"]))
        assert trusted_lora_path(str(checkpoint)) == str(checkpoint.resolve())
    shared = playground_env["runs_dir"] / "shared"
    shared.mkdir()
    shared.chmod(0o777)
    unsafe = shared / "data.tar"
    unsafe.write_bytes(b"not a pickle")
    alias = shared.with_name("alias")
    alias.symlink_to(shared, target_is_directory=True)
    with pytest.raises(ValueError, match="Parent is writable"):
        trusted_lora_path(str(alias / "data.tar"))


def test_bounded_payload_is_valid_json_and_reaches_model(tmp_path):
    from aimip.agent.chat import RESULT_BUDGET

    backend = AgentTools(cli=lambda _: [{"untrusted_attr": "x" * (RESULT_BUDGET * 2)}])
    agent = agent_for(tmp_path, backend, call("list_models"), AIMessage(content="Result too large"))
    agent.turn("list")
    message = agent.history[-1][2]
    assert len(message.content) <= RESULT_BUDGET
    assert json.loads(message.content)["truncated"] is True


def test_log_directory_symlink_shared_mode_and_replacement(tmp_path):
    target = tmp_path / "target"
    target.mkdir()
    link = tmp_path / "logs"
    link.symlink_to(target, target_is_directory=True)
    with pytest.raises(OSError):
        Transcript(link)
    target.chmod(0o777)
    with pytest.raises(ValueError, match="group/world-writable"):
        Transcript(target)
    target.chmod(0o700)
    log = Transcript(target)
    target.rename(tmp_path / "old")
    target.mkdir(mode=0o700)
    with pytest.raises(ValueError, match="changed"):
        log.write("human", "not redirected")


def test_turn_failure_reports_redacted_error_type(tmp_path, backend):
    class FailingModel(ScriptedModel):
        def invoke(self, messages):
            raise ValueError("provider denied https://example.invalid?api_key=UNKNOWN_SECRET&code=403")

    log = Transcript(tmp_path / "logs")
    agent = ChatAgent(FailingModel(), backend, log)
    with pytest.raises(RuntimeError, match="ValueError.*provider denied") as error:
        agent.turn("hello")
    assert "UNKNOWN_SECRET" not in str(error.value)
    rows = [json.loads(line) for p in log.directory.glob("*.jsonl") for line in p.read_text().splitlines()]
    assert rows[-1]["value"]["error_type"] == "ValueError"
    assert "UNKNOWN_SECRET" not in json.dumps(rows)


def test_optional_agent_import_error_is_actionable(monkeypatch, capsys):
    import aimip.agent
    import builtins

    original = builtins.__import__

    def fail(name, *args, **kwargs):
        if name == "chat":
            raise ModuleNotFoundError("langgraph")
        return original(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fail)
    assert aimip.agent.main([]) == 2
    assert "container/requirements-agent.txt" in capsys.readouterr().err


def test_vertex_backend_selected_by_project_env(tmp_path, monkeypatch):
    """AIMIP_GEMINI_VERTEX_PROJECT switches to Vertex AI (project + location); unset keeps the Gemini API."""
    langchain_google_genai = pytest.importorskip("langchain_google_genai", exc_type=ImportError)
    import aimip.agent.chat as chat

    seen = {}

    def model(**kwargs):
        seen.clear()
        seen.update(kwargs)
        return ScriptedModel()

    monkeypatch.setattr(langchain_google_genai, "ChatGoogleGenerativeAI", model)
    monkeypatch.setattr(chat, "repl", lambda agent: 0)
    monkeypatch.setenv("AIMIP_DEFAULT_OUT_ROOT", str(tmp_path / "runs"))
    monkeypatch.delenv("AIMIP_CHAT_LOG_DIR", raising=False)
    monkeypatch.setenv("GEMINI_API_KEY", "fake-test-key")
    monkeypatch.delenv("AIMIP_GEMINI_VERTEX_PROJECT", raising=False)
    assert chat.main([]) == 0
    assert "vertexai" not in seen and "project" not in seen
    assert seen["google_api_key"] == "fake-test-key"
    monkeypatch.setenv("AIMIP_GEMINI_VERTEX_PROJECT", "my-gcp-project")
    monkeypatch.delenv("GOOGLE_APPLICATION_CREDENTIALS", raising=False)
    monkeypatch.delenv("AIMIP_GEMINI_VERTEX_TOKEN_FILE", raising=False)
    assert chat.main([]) == 2  # Vertex refuses an AI Studio key: it needs a token file or ADC
    token = tmp_path / "vertex.token"
    token.write_text("ya29.fake-access-token\n")
    token.chmod(0o600)
    monkeypatch.setenv("AIMIP_GEMINI_VERTEX_TOKEN_FILE", str(token))
    assert chat.main([]) == 0
    assert seen["vertexai"] is True and seen["project"] == "my-gcp-project" and seen["location"] == "global"
    assert "google_api_key" not in seen and seen["credentials"].token == "ya29.fake-access-token"
    token.chmod(0o644)
    assert chat.main([]) == 2  # token file must be 0600
    monkeypatch.delenv("AIMIP_GEMINI_VERTEX_TOKEN_FILE")
    monkeypatch.setenv("GOOGLE_APPLICATION_CREDENTIALS", str(tmp_path / "sa.json"))
    monkeypatch.setenv("AIMIP_GEMINI_VERTEX_LOCATION", "europe-west4")
    assert chat.main([]) == 0
    assert seen["location"] == "europe-west4" and "credentials" not in seen and "google_api_key" not in seen


def test_system_prompt_lists_all_canonical_models():
    """`chat.SYSTEM` names all canonical models and the `aimip fetch -m <model>` host workflow."""
    from aimip.agent.chat import SYSTEM
    from aimip.setup_data import CANONICAL_MODELS

    for model_name in CANONICAL_MODELS:
        assert model_name in SYSTEM, f"Missing {model_name} in chat.SYSTEM"
    assert "aimip fetch -m <model>" in SYSTEM


def test_coupled_models_plans_and_unfetched_hint(
    tmp_path, allow_tmp_run_root, monkeypatch, capsys
):
    """Unfetched hints (empty & partial registry) and coupled `plan_run`, `plan_spin`, and `plan_collect`."""
    import aimip.setup_data as setup_data_mod
    from aimip.setup_data import SetupYearConfig
    from test_models_and_fetch import _populate_extra_fme_data_tree_and_lock

    yc = SetupYearConfig(som_years=(2032, 2033, 2034, 2035), hiro_native_years=tuple(range(2014, 2024)))
    monkeypatch.setattr(setup_data_mod, "SetupYearConfig", lambda: yc)
    hf = tmp_path / "hf"
    lock = tmp_path / "models.lock"
    _populate_extra_fme_data_tree_and_lock(hf, yc, lock)

    data_dir = tmp_path / "data"
    pg_dir = tmp_path / "pg"
    (pg_dir / "bin").mkdir(parents=True)
    (pg_dir / "bin" / "aimip").write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    (pg_dir / "site.json").write_text(
        json.dumps(
            {
                "name": "levante",
                "scheduler": "slurm",
                "runtime": "apptainer",
                "sbatch": {
                    "partition": "gpu",
                    "account": "test_gpu",
                    "gres": "gpu:a100_80:1",
                    "cpus_per_task": 32,
                    "mem": "64G",
                    "max_time": "12:00:00",
                },
                "run_root": str(allow_tmp_run_root / "runs"),
                "extra_ro_binds": [],
                "gpu_check": "nvidia-smi",
            }
        ),
        encoding="utf-8",
    )
    runs = allow_tmp_run_root / "runs"
    runs.mkdir()

    monkeypatch.setenv("AIMIP_PLAYGROUND", str(pg_dir))
    monkeypatch.setenv("AIMIP_SITE_CONFIG", str(pg_dir / "site.json"))
    monkeypatch.setenv("AIMIP_DATA", str(data_dir))
    monkeypatch.setenv("AIMIP_LOCK", str(lock))
    monkeypatch.setenv("AIMIP_HF_MIRROR", f"file://{hf}")
    monkeypatch.setenv("AIMIP_DEFAULT_OUT_ROOT", str(runs))
    monkeypatch.delenv("AIMIP_REGISTRY", raising=False)

    backend = AgentTools()

    # Before any fetch (empty $AIMIP_DATA):
    # - `propose_run` / `plan_run` is side-effect-free (no auto-fetch!) and reports missing assets/bytes;
    # - `services.prepare(..., allow_fetch=False)`, `plan_spin`, and `list_ics` refuse with `aimip fetch -m ace2s` hint.
    unfetched_prop = backend.plan_run("ace2s", "2014-01-01T00:00:00", 4)
    assert unfetched_prop["missing_count"] > 0 and unfetched_prop["missing_bytes"] > 0
    with pytest.raises(MissingAssetsError, match=r"aimip fetch -m ace2s"):
        services.prepare(RunSpec(model="ace2s", start="2014-01-01T00:00:00", steps=4), allow_fetch=False)
    with pytest.raises(ValueError, match=r"aimip fetch -m ace2s"):
        backend.plan_spin("ace2s", ["2014-01-01T00:00:00"], "2014-01-02T00:00:00")
    with pytest.raises(ValueError, match=r"aimip fetch -m ace2s"):
        backend.list_ics("ace2s")
    assert not (data_dir / "registry.json").exists()

    # Unknown / bogus model names do NOT falsely suggest `aimip fetch`.
    with pytest.raises(ValueError) as bogus_exc:
        backend.plan_run("does-not-exist", "2014-01-01T00:00:00", 4)
    assert "aimip fetch" not in str(bogus_exc.value)

    # Partial `<AIMIP_DATA>/registry.json` (only `ace2-som` fetched so far) still hints `aimip fetch -m ace2s`.
    assert cli_main(["fetch", "-m", "ace2-som", "--json"]) == 0
    capsys.readouterr()
    assert (data_dir / "registry.json").is_file()
    assert backend.plan_run("ace2s", "2014-01-01T00:00:00", 4)["missing_count"] > 0
    with pytest.raises(ValueError, match=r"aimip fetch -m ace2s"):
        backend.plan_spin("ace2s", ["2014-01-01T00:00:00"], "2014-01-02T00:00:00")
    with pytest.raises(ValueError, match=r"aimip fetch -m ace2s"):
        backend.list_ics("ace2s")

    # Fetch all 5 coupled/scenario fme models and verify coupled nested `data_writer` outputs + spin + collect.
    assert cli_main(["fetch", "-m", "ace2-som,ace2s,hiro-acev2,samudrace-cm4,samudrace-e3smv3", "--json"]) == 0
    capsys.readouterr()

    som_1x = backend.plan_run("ace2-som", "2032-01-01T00:00:00", 4)
    assert som_1x["runs"][0]["forcing_set"]["name"] == "1xCO2"
    som_3x = backend.plan_run("ace2-som", "2032-01-01T00:00:00", 4, forcing_set="3xCO2")
    assert som_3x["runs"][0]["forcing_set"]["name"] == "3xCO2"

    hiro2 = backend.plan_run("hiro-acev2", "2014-01-01T00:00:00", 4, members=2)
    assert hiro2["runs"][0]["members"] == 2
    assert hiro2["runs"][0]["outputs"] == {"monthly": True, "prediction": False}

    cm4 = backend.plan_run("samudrace-cm4", "0311-01-01T00:00:00", 6)
    assert cm4["runs"][0]["outputs"] == {
        "ocean": {"monthly": True, "prediction": False},
        "atmosphere": {"monthly": True, "prediction": False},
    }

    e3sm = backend.plan_run("samudrace-e3smv3", "0425-01-03T12:00:00", 6, members=6)
    assert e3sm["runs"][0]["members"] == 6
    assert e3sm["runs"][0]["outputs"] == {
        "ocean": {"monthly": True, "prediction": False},
        "atmosphere": {"monthly": True, "prediction": False},
    }

    cm4_spin = backend.plan_spin(
        "samudrace-cm4",
        ["0311-01-01T00:00:00", "0313-01-01T00:00:00"],
        "0314-01-01T00:00:00",
    )
    assert [r["steps"] for r in cm4_spin["runs"]] == [219, 73]
    assert all(
        r["outputs"]
        == {
            "ocean": {"monthly": False, "prediction": False},
            "atmosphere": {"monthly": False, "prediction": False},
        }
        for r in cm4_spin["runs"]
    )

    # Coupled spin collection (`plan_collect` -> `/dev/tty` 8-hex digest -> `aimip collect <spin_dir> --out <ic.nc>`).
    for i in range(2):
        write_fake_ic_netcdf(
            Path(cm4_spin["run_dir"]) / str(i) / "atmosphere" / "restart.nc",
            ["0314-01-01T00:00:00"],
            calendar="noleap",
            spatial_dims=("lat", "lon"),
            var_names=("TMP2m", "surface_temperature"),
            sample_dim="sample",
        )
        write_fake_ic_netcdf(
            Path(cm4_spin["run_dir"]) / str(i) / "ocean" / "restart.nc",
            ["0314-01-01T00:00:00"],
            calendar="noleap",
            spatial_dims=("yh", "xh"),
            var_names=("sst", "thetao_0"),
            sample_dim="sample",
        )
    collect_card = backend.plan_collect(cm4_spin["run_dir"])
    assert collect_card["kind"] == "collect" and collect_card["members"] == 2
    assert collect_card["ocean_output"].endswith("/ic_ocean.nc")
    with pty_responder(collect_card["approval_prefix"]) as (tty_path, _):
        collect_submit = backend.approve_and_execute_on_tty(tty_path=tty_path)
    assert collect_submit["host_command"] == f"aimip collect {cm4_spin['run_dir']} --out {collect_card['output']}"
    merged_coupled = in_process_cli(["collect", cm4_spin["run_dir"], "--out", collect_card["output"]])
    assert merged_coupled["n_samples"] == 2
    assert Path(collect_card["ocean_output"]).is_file()
