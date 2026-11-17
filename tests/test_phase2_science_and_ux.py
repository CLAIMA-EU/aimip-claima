"""Phase 2a science, metadata, and console UX regression tests:
- `python -I -c "import aimip.families"` first-import subprocess check (no circular import)
- `test_emu_models_validate_through_adapter` (all 4 DLESyM & ArchesClimate-SSP models)
- `services.plan` across all 14 canonical models, Arches period/step card formatting, and unfetched IC checks
- `test_card_examples_validate` (all 28 `MODEL_CARDS` examples + `--end <window_end>` window-edge hint)
- `aimip cite` verified arXiv/DOI references, parent-architecture note for derived checkpoints, upstream URL when unconfirmed, and honest `experimental` default card
- `aimip doctor` login-node missing GPU reported as INFO (`rc == 0`), host Slurm/GPU env passthrough, no `--preset local` hint, and read-only `run_root` check
- `test_logs_follow_keeps_printing_after_max_bytes` + multi-file offsets, >=15s `sacct` backoff during `logs --follow`, empty-state guards, `watch` exit codes (`1` on FAILED, `75` on timeout), text `allowed:` / `next:` stderr lines, dead local PID detection, and truthful `cancel`
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import shlex
import subprocess
import sys

import pytest

from aimip import services
from aimip.cli import main
from aimip.errors import ConfigError, ExecutionError
from aimip.families import get_family_adapter
from aimip.registry import DATA_PLACEHOLDER, _parse_model_entry, build_default_registry_payload
from aimip.services.metadata import MODEL_CARDS, card_for_model
from aimip.services.types import RunSpec
from aimip.setup_data import CANONICAL_MODELS
from aimip.status import get_run_status
from conftest import parse_cli_json


def _default_specs(data_dir: Path) -> dict:
    payload = build_default_registry_payload(root_placeholder=DATA_PLACEHOLDER)
    return {e["name"]: _parse_model_entry(dict(e), None, str(data_dir)) for e in payload["models"]}


def test_import_aimip_families_first_in_isolated_subprocess() -> None:
    """`import aimip.families` as the very first import must succeed without circular ImportError."""
    src_root = str(Path(__file__).resolve().parents[1] / "src")
    sub_timeout = float(os.environ.get("AIMIP_TEST_SUBPROCESS_TIMEOUT", "120"))
    proc = subprocess.run(
        [
            sys.executable,
            "-I",
            "-B",
            "-c",
            (
                f"import sys; sys.path.insert(0, {src_root!r}); "
                "import aimip.families as fam; "
                "assert {'fme', 'e2s-dlesym', 'arches'} <= set(fam.INSTALLED_ADAPTERS)"
            ),
        ],
        capture_output=True,
        text=True,
        timeout=sub_timeout,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr


@pytest.mark.parametrize(
    "name,start,steps",
    [
        ("dlesym-v1.0", "1995-01-01", 1460),
        ("dlesym-v1.1", "1995-01-01", 1460),
        ("dlesym-v0-isccp", "1995-01-01", 1460),
        ("dlesym-aimip", "1978-10-03", 1460),
        ("archesclimate-ssp", "2015-01-01", 12),
    ],
)
def test_emu_models_validate_through_adapter(tmp_path: Path, name: str, start: str, steps: int) -> None:
    model = _default_specs(tmp_path)[name]
    out = get_family_adapter(model.family).validate(
        model, RunSpec(model=name, start=start, length="1y"), data_dir=str(tmp_path)
    )
    assert out["steps"] == steps


def _examples() -> list[tuple[str, str]]:
    return [(name, ex) for name, card in sorted(MODEL_CARDS.items()) for ex in card["examples"]]


@pytest.mark.parametrize("name,example", _examples())
def test_card_examples_validate(tmp_path: Path, name: str, example: str) -> None:
    """Every example printed by `aimip show --examples` must pass the model's own horizon and window checks."""
    toks = shlex.split(example)
    assert toks[0] == "aimip" and toks[1] in ("plan", "run") and toks[2] == name
    opts: dict[str, str] = {}
    i = 3
    while i < len(toks):
        assert toks[i].startswith("--"), example
        opts[toks[i][2:].replace("-", "_")] = toks[i + 1]
        i += 2
    assert "/path/to/" not in example, "an example must be runnable as printed"
    model = _default_specs(tmp_path)[name]
    spec = RunSpec(
        model=name,
        start=opts["start"],
        steps=int(opts["steps"]) if "steps" in opts else None,
        length=opts.get("length"),
        end=opts.get("end"),
        members=int(opts["members"]) if "members" in opts else None,
        forcing_set=opts.get("forcing_set"),
    )
    get_family_adapter(model.family).validate(model, spec, data_dir=str(tmp_path))


def test_logs_follow_keeps_printing_after_max_bytes(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from aimip.services import core

    run = tmp_path / "r"
    run.mkdir()
    log = run / "slurm-1.out"
    log.write_text("a" * 200 + "\n", encoding="utf-8")
    err_log = run / "slurm-1.err"
    err_log.write_text("warn-0\n", encoding="utf-8")
    monkeypatch.setattr(core, "resolve_run_target", lambda r, **k: (str(run), "r"))
    monkeypatch.setattr(core, "check_trusted_run_dir", lambda d: d)
    states = iter(["running", "running", "completed"])
    monkeypatch.setattr(core, "status", lambda r, **k: {"state": next(states)})
    seen: list[str] = []

    def _sleep(_seconds: float) -> None:
        with open(log, "a", encoding="utf-8") as fh:
            fh.write(f"NEW-LINE-{len(seen)}\n")
        with open(err_log, "a", encoding="utf-8") as efh:
            efh.write(f"ERR-LINE-{len(seen)}\n")

    core.logs("r", follow=True, max_bytes=128, on_chunk=seen.append, sleep_fn=_sleep)
    assert any("NEW-LINE" in chunk for chunk in seen[1:]), seen
    assert any("ERR-LINE" in chunk for chunk in seen[1:]), seen


def test_all_canonical_models_plan_through_services_and_check_ic_and_arches_card(
    tmp_path: Path, allow_tmp_run_root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    data_dir = tmp_path / "empty_data"
    data_dir.mkdir()
    runs_dir = allow_tmp_run_root / "all14_runs"
    runs_dir.mkdir()
    site_path = tmp_path / "site.json"
    site_path.write_text(
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
                "run_root": str(runs_dir),
                "extra_ro_binds": [],
                "gpu_check": "nvidia-smi",
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.delenv("AIMIP_REGISTRY", raising=False)
    monkeypatch.setenv("AIMIP_DATA", str(data_dir))
    monkeypatch.setenv("AIMIP_SITE_CONFIG", str(site_path))
    monkeypatch.setenv("AIMIP_DEFAULT_OUT_ROOT", str(runs_dir))

    valid_requests: dict[str, dict] = {
        "ace2-era5": {"start": "1979-01-01", "steps": 4},
        "ace2-shield": {"start": "1979-01-01", "steps": 4},
        "ace2s-shield-plus": {"start": "1979-01-01", "steps": 4, "members": 2},
        "ace2-eamv3": {"start": "1994-01-01", "steps": 4},
        "hiro-ace": {"start": "1994-01-01", "steps": 4, "members": 2},
        "ace2-som": {"start": "2032-01-01", "steps": 4, "forcing_set": "1xCO2"},
        "ace2s": {"start": "2014-01-01", "steps": 4, "members": 2},
        "hiro-acev2": {"start": "2014-01-01", "steps": 4, "members": 2},
        "samudrace-cm4": {"start": "0311-01-01", "steps": 2},
        "samudrace-e3smv3": {"start": "0425-01-03T12:00:00", "steps": 2, "members": 3},
        "dlesym-v1.0": {"start": "1995-01-01", "steps": 4},
        "dlesym-v1.1": {"start": "1995-01-01", "steps": 4, "members": 2},
        "dlesym-v0-isccp": {"start": "1995-01-01", "steps": 4},
        "dlesym-aimip": {"start": "1978-10-03", "steps": 16},
        "md-1.5": {"start": "1978-10-01", "steps": 12, "members": 2},
        "archesclimate-ssp": {"start": "2015-01-01", "length": "1y", "forcing_set": "ssp585"},
        "archesweather": {"start": "1978-09-30", "steps": 4},
        "archesweathergen": {"start": "1978-09-30", "length": "30d", "members": 2},
    }
    assert set(valid_requests) == set(CANONICAL_MODELS)

    for mname, kwargs in valid_requests.items():
        resolved = services.plan(RunSpec(model=mname, name=f"p-{mname}", **kwargs))
        assert len(resolved.digest) == 64
        card_txt = services.render_plan_card(resolved, is_tty=False)
        if mname == "archesclimate-ssp":
            assert "outputs 2015-03 .. 2016-02" in card_txt
            assert "monthly means" in card_txt
            show_txt = services.render_show_card(services.show(mname))
            assert "1mo step" in show_txt and "730h step" not in show_txt
        elif resolved.family == "geoarches":
            assert "outputs 1978-10-01 .." in card_txt and "daily steps / monthly means" in card_txt
            assert "ERA5 daily-mean state of 1978-09-30" in card_txt
            assert "pressure levels in monthly_levels.nc" in card_txt and "restart.nc" in card_txt
        elif resolved.family == "fme":
            assert "initial condition: not checked until the files are fetched" in card_txt

    # ArchesWeatherGen continuation: members come from the restart, and the plan digest binds the restart's bytes.
    from test_geoarches import write_test_restart

    prev = runs_dir / "aw-seg1"
    prev.mkdir()
    write_test_restart(prev / "restart.nc", step=20, members=2)
    cont = services.plan(RunSpec(model="archesweathergen", name="p-cont", start="1978-10-20", steps=11, ic=str(prev)))
    card_txt = services.render_plan_card(cont, is_tty=False)
    assert "outputs 1978-10-21 .. 1978-10-31" in card_txt and "restart of a previous run" in card_txt
    write_test_restart(prev / "restart.nc", step=20, members=2, month="")
    again = services.plan(RunSpec(model="archesweathergen", name="p-cont", start="1978-10-20", steps=11, ic=str(prev)))
    assert again.digest != cont.digest
    with pytest.raises(ConfigError, match="use --start 1978-10-20T00:00:00"):
        services.plan(RunSpec(model="archesweathergen", start="1978-10-25", steps=5, ic=str(prev)))
    # A run writes its files only at the end: the card says when the job cannot finish (site limit 8 h here).
    fits = services.render_plan_card(services.plan(RunSpec(model="archesweathergen", start="1978-09-30", length="6y")),
                                     is_tty=False)
    assert "CAPPED from" in fits and "keeps nothing" in fits and "expected time exceeds" not in fits
    long = services.render_plan_card(services.plan(RunSpec(model="archesweathergen", start="1978-09-30", length="20y")),
                                     is_tty=False)
    assert "keeps nothing; the expected time exceeds the wall time" in long

    # Unfetched FME model rejects impossible `--start` date BEFORE downloading anything
    with pytest.raises(ConfigError, match="No shipped initial condition"):
        services.plan(RunSpec(model="ace2-era5", start="1985-07-13", length="1y"))

    # Window-exceeded error offers `--end <window_end>`
    with pytest.raises(ConfigError, match=r"--end 2014-12-31T18:00:00") as win_exc:
        services.plan(RunSpec(model="ace2-era5", start="1979-01-01", length="36y"))
    assert win_exc.value.next_command and "--end 2014-12-31T18:00:00" in win_exc.value.next_command


def test_cite_references_verified_and_unconfirmed_uses_upstream_url(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    data_dir = tmp_path / "cite_data"
    data_dir.mkdir()
    monkeypatch.delenv("AIMIP_REGISTRY", raising=False)
    monkeypatch.setenv("AIMIP_DATA", str(data_dir))

    era5_cite = services.cite("ace2-era5")
    assert "2411.11268" in era5_cite["bibtex"]
    assert era5_cite["note"] is None
    assert "Henn, Brian" in era5_cite["bibtex"] and "Kwa, Anna" in era5_cite["bibtex"]
    assert "Dresdner" not in era5_cite["bibtex"]  # ACE1 author list must not appear on ACE2!
    assert "10.1002/qj.3803" in era5_cite["bibtex"]

    for variant in ("ace2-shield", "ace2s-shield-plus", "ace2-eamv3"):
        vcite = services.cite(variant)
        assert vcite["note"] and "parent ACE2 base architecture" in vcite["note"]
        assert "% Note:" in vcite["bibtex"]

    for hiro_var in ("hiro-ace", "ace2s", "hiro-acev2"):
        hcite = services.cite(hiro_var)
        assert "2512.18224" in hcite["bibtex"]
        assert "Perkins" in hcite["bibtex"]

    som_cite = services.cite("ace2-som")
    assert "2412.04418" in som_cite["bibtex"]
    assert "Clark, Spencer K." in som_cite["bibtex"]

    sam_cite = services.cite("samudrace-cm4")
    assert "2509.12490" in sam_cite["bibtex"]
    assert "Duncan, James" in sam_cite["bibtex"] and "Zanna, Laure" in sam_cite["bibtex"]

    dlesym_cite = services.cite("dlesym-v1.0")
    assert "10.1029/2025AV001706" in dlesym_cite["bibtex"]
    assert "AGU Advances" in dlesym_cite["bibtex"]

    arches_cite = services.cite("archesclimate-ssp")
    assert arches_cite["entries"] == []
    assert arches_cite["confirmed"] is False
    assert "https://github.com/grahamclyne/ArchesClimate-SSP" in arches_cite["bibtex"]
    assert "@article" not in arches_cite["bibtex"]

    unknown_card = card_for_model("user-custom-model")
    assert unknown_card["status"] == "experimental"
    assert unknown_card["licence"] == "not recorded"
    assert unknown_card["cost_note"] == "not recorded"
    assert "AIMIP benchmark suite" not in unknown_card["status_definition"]
    assert unknown_card["bibtex"] == []


def test_doctor_login_node_gpu_info_container_host_env_and_readonly_run_root(
    tmp_path: Path, allow_tmp_run_root: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    fbin = tmp_path / "doc_bin"
    fbin.mkdir()
    for tool in ("sbatch", "squeue", "sacct", "scancel"):
        p = fbin / tool
        p.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
        p.chmod(0o755)
    monkeypatch.setenv("PATH", f"{fbin}:/usr/bin:/bin")

    data_dir = tmp_path / "doc_data"
    data_dir.mkdir()
    future_run_root = allow_tmp_run_root / "uncreated_run_root"
    site_file = tmp_path / "site.json"
    site_file.write_text(
        json.dumps(
            {
                "name": "levante",
                "scheduler": "slurm",
                "runtime": "apptainer",
                "sbatch": {
                    "partition": "gpu",
                    "account": "acct",
                    "gres": "gpu:1",
                    "cpus_per_task": 8,
                    "mem": "32G",
                    "max_time": "08:00:00",
                },
                "run_root": str(future_run_root),
                "extra_ro_binds": [],
                "gpu_check": "nvidia-smi-nonexistent",
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("AIMIP_DATA", str(data_dir))
    monkeypatch.setenv("AIMIP_SITE_CONFIG", str(site_file))
    monkeypatch.setenv("AIMIP_DEFAULT_OUT_ROOT", str(future_run_root))
    monkeypatch.setenv("AIMIP_IMAGE_SHA", "a" * 64)
    monkeypatch.setenv("AIMIP_HOST_RUNTIME_OK", "1")
    monkeypatch.setenv("AIMIP_HOST_SLURM_OK", "1")
    monkeypatch.setenv("AIMIP_HOST_GPU_STATUS", "absent")

    assert main(["doctor"]) == 0
    out = capsys.readouterr().out
    assert "INFO   gpu" in out
    assert "no GPU here — jobs run on GPU nodes via the scheduler" in out
    assert "PASS   scheduler" in out
    assert "--preset local" not in out
    assert not future_run_root.exists(), "doctor must not create run_root on disk"


def test_stderr_hints_watch_exit_codes_dead_local_pid_and_truthful_cancel(
    playground_env, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    # 1. Text error prints `allowed:` and `next:` on stderr
    assert main(["models", "--task", "bogus"]) == 2
    err = capsys.readouterr().err
    assert "allowed:" in err and "next:" in err

    # 2. Dead local PID without `"finished"` in manifest reports `"failed"`, and `cancel` reports process not found
    prepared = services.prepare(RunSpec(model="ace2-era5", start="1979-01-01", steps=4, name="dead-local", mode="local"))
    (Path(prepared.run_dir) / "local.pid").write_text("99999999\n", encoding="utf-8")
    monkeypatch.setenv("AIMIP_IMAGE_SHA", "native")
    st = get_run_status(prepared.run_dir)
    assert st["state"] == "failed"
    with pytest.raises(ConfigError, match="terminal state"):
        services.cancel("dead-local")

    # Force non-terminal (and whitespace-only) status via monkeypatch to verify no IndexError and ProcessLookupError raises ExecutionError
    monkeypatch.setattr("aimip.services.core.get_run_status", lambda _d: {"state": "   ", "local_pid": 99999999})
    with pytest.raises(ExecutionError, match="not found"):
        services.cancel("dead-local")

    # 3. `watch` returns exit 1 on FAILED, handles empty state without IndexError, and returns exit 75 with message on timeout
    monkeypatch.setattr(
        "aimip.services.core.status",
        lambda _r, **_k: {"state": "", "slurm_job_id": None, "run_name": "wempty"},
    )
    assert main(["watch", "dead-local", "--interval", "0.05", "--max-polls", "1"]) == 0
    capsys.readouterr()

    monkeypatch.setattr(
        "aimip.services.core.status",
        lambda _r, **_k: {"state": "FAILED", "slurm_job_id": "123", "run_name": "wfail"},
    )
    assert main(["watch", "dead-local", "--interval", "0.05"]) == 1
    capsys.readouterr()

    monkeypatch.setattr(
        "aimip.services.core.status",
        lambda _r, **_k: {"state": "RUNNING", "slurm_job_id": "123", "run_name": "wtime"},
    )
    assert main(["watch", "dead-local", "--interval", "0.05", "--timeout", "0.01"]) == 75
    assert "still running, watch timed out" in capsys.readouterr().out


def test_logs_follow_throttles_live_sacct_queries_to_15_seconds(
    playground_env, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`logs --follow` streams incremental bytes every poll while calling `sacct` at most once per 15 s."""
    fbin = tmp_path / "sacct_throttle_bin"
    fbin.mkdir()
    sacct_counter = tmp_path / "sacct_calls.log"
    sacct = fbin / "sacct"
    sacct.write_text(
        "#!/bin/sh\n"
        f'echo 1 >> "{sacct_counter}"\n'
        'echo "777888|RUNNING|00:00:30|0:0"\n',
        encoding="utf-8",
    )
    sacct.chmod(0o755)
    monkeypatch.setenv("PATH", f"{fbin}:/usr/bin:/bin")

    prepared = services.prepare(RunSpec(model="ace2-era5", start="1979-01-01", steps=4, name="follow-sacct"))
    rdir = Path(prepared.run_dir)
    (rdir / "slurm.jobid").write_text("777888\n", encoding="utf-8")
    out_log = rdir / "slurm-777888.out"
    out_log.write_text("step-0\n", encoding="utf-8")

    chunks: list[str] = []

    def _fast_sleep(_sec: float) -> None:
        with open(out_log, "a", encoding="utf-8") as fh:
            fh.write(f"step-{len(chunks)}\n")

    res = services.logs(
        "follow-sacct",
        follow=True,
        poll_interval=0.05,
        max_polls=5,
        on_chunk=chunks.append,
        sleep_fn=_fast_sleep,
    )
    assert "step-4" in res["content"]
    assert len(sacct_counter.read_text(encoding="utf-8").splitlines()) == 1


def test_doctor_never_crashes_on_filesystem_oserror_107_and_continues_other_checks(
    tmp_path: Path, allow_tmp_run_root: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """`aimip doctor` catches `OSError(107, 'Transport endpoint is not connected')` on filesystem probes, reports FAIL with error text + retry/mount hint, continues remaining checks, and exits with code 1."""
    from aimip.errors import EXIT_CODES
    from aimip.services import core

    fbin = tmp_path / "doc_enotconn_bin"
    fbin.mkdir()
    for tool in ("sbatch", "squeue", "sacct", "scancel"):
        p = fbin / tool
        p.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
        p.chmod(0o755)
    monkeypatch.setenv("PATH", f"{fbin}:/usr/bin:/bin")

    data_dir = tmp_path / "enotconn_data"
    data_dir.mkdir()
    run_root = allow_tmp_run_root / "enotconn_runs"
    run_root.mkdir()
    site_file = tmp_path / "site_enotconn.json"
    site_file.write_text(
        json.dumps(
            {
                "name": "levante",
                "scheduler": "slurm",
                "runtime": "apptainer",
                "sbatch": {
                    "partition": "gpu",
                    "account": "acct",
                    "gres": "gpu:1",
                    "cpus_per_task": 8,
                    "mem": "32G",
                    "max_time": "08:00:00",
                },
                "run_root": str(run_root),
                "extra_ro_binds": [],
                "gpu_check": "nvidia-smi-nonexistent",
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("AIMIP_DATA", str(data_dir))
    monkeypatch.setenv("AIMIP_SITE_CONFIG", str(site_file))
    monkeypatch.setenv("AIMIP_DEFAULT_OUT_ROOT", str(run_root))
    monkeypatch.setenv("AIMIP_IMAGE_SHA", "a" * 64)
    monkeypatch.setenv("AIMIP_HOST_RUNTIME_OK", "1")
    monkeypatch.setenv("AIMIP_HOST_SLURM_OK", "1")
    monkeypatch.setenv("AIMIP_HOST_GPU_STATUS", "absent")

    def _enotconn_disk_usage(path: str):
        raise OSError(107, "Transport endpoint is not connected", str(path))

    monkeypatch.setattr(core.shutil, "disk_usage", _enotconn_disk_usage)

    assert main(["doctor"]) == EXIT_CODES["doctor_failed"] == 1
    out_txt = capsys.readouterr().out
    assert "FAIL   data_root" in out_txt
    assert "Transport endpoint is not connected" in out_txt
    assert "fix:" in out_txt
    # Subsequent checks after `data_root` still ran and succeeded!
    assert "PASS   permissions" in out_txt
    assert "PASS   image_version" in out_txt

    assert main(["doctor", "--json"]) == EXIT_CODES["doctor_failed"] == 1
    report = parse_cli_json(capsys.readouterr().out)
    assert report["ok"] is False
    assert report["total"] == 7
    by_name = {c["name"]: c for c in report["checks"]}
    assert by_name["data_root"]["ok"] is False
    assert by_name["data_root"]["status"] == "fail"
    assert "Transport endpoint is not connected" in by_name["data_root"]["detail"]
    assert by_name["data_root"]["fix_hint"] and "mount" in by_name["data_root"]["fix_hint"].lower()
    assert by_name["permissions"]["ok"] is True
    assert by_name["image_version"]["ok"] is True
