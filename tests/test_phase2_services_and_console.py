"""Phase 2a unit and CLI golden tests:
- `aimip.services` (`models`, `show`, `cite`, `plan`, `prepare`, `submit`, `status`, `watch`, `logs`, `ls`, `cancel`, `summary`, `doctor`)
- `aimip.families` closed adapter allowlist (`INSTALLED_ADAPTERS`)
- Calendar & horizon conversions (`noleap`, `360_day`, `proleptic_gregorian`, `julian`, `6h` / `5d` (`120h`) / `arches` monthly steps, non-integral refusals)
- Every CLI verb in TTY/plain mode and `--json` v1 envelope, `--name`, `--config experiment.yaml` with `cli_overrides`, `completion bash|zsh`, and stable exit codes
- Pre-`sbatch` `submit-intent.json` recording and `squeue` uncertain-reply reconciliation
"""

from __future__ import annotations

import cftime
import json
import os
from pathlib import Path
import numpy as np
import pytest
import xarray as xr
import yaml

from aimip import services
from aimip.cli import main
from aimip.errors import (
    EXIT_CODES,
    AimipError,
    ApprovalError,
    ConfigError,
    ExecutionError,
    MissingAssetsError,
    RegistryError,
    ValidationError,
)
from aimip.families import INSTALLED_ADAPTERS, get_family_adapter
from aimip.registry import _parse_model_entry
from aimip.services.types import RunSpec
from aimip.time_utils import resolve_horizon
from conftest import parse_cli_json


@pytest.fixture
def fake_cluster_tools(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict[str, Path]:
    """Install offline fake `nvidia-smi`, `apptainer`, `sbatch`, `squeue`, `sacct`, and `scancel`."""
    fbin = tmp_path / "fake_cluster_bin"
    fbin.mkdir(exist_ok=True)
    log_dir = tmp_path / "fake_cluster_logs"
    log_dir.mkdir(exist_ok=True)

    for name, body in (
        ("nvidia-smi", '#!/bin/bash\necho "GPU 0: Fake A100-SXM4-80GB"\n'),
        ("apptainer", '#!/bin/bash\necho "apptainer version 1.3.0"\n'),
        (
            "sbatch",
            "#!/bin/bash\n"
            f'echo "$@" >> "{log_dir / "sbatch.log"}"\n'
            "echo 654321\n",
        ),
        ("squeue", "#!/bin/bash\nexit 0\n"),
        ("sacct", '#!/bin/bash\necho "654321|COMPLETED|00:02:15|0:0"\n'),
        (
            "scancel",
            "#!/bin/bash\n"
            f'echo "$@" >> "{log_dir / "scancel.log"}"\n',
        ),
    ):
        p = fbin / name
        p.write_text(body, encoding="utf-8")
        p.chmod(0o755)

    monkeypatch.setenv("PATH", f"{fbin}:{os.environ.get('PATH', '')}")
    return {"bin": fbin, "logs": log_dir}


def _write_monthly_nc(out_dir: Path, mean_val: float = 288.5) -> None:
    lats = np.array([-60.0, 0.0, 0.0, 60.0], dtype=np.float64)
    lons = np.linspace(0.0, 315.0, 8, dtype=np.float64)
    data = np.full((1, 2, 4, 8), mean_val, dtype=np.float32)
    vt = np.array(
        [
            [
                cftime.datetime(1979, 1, 15, calendar="proleptic_gregorian"),
                cftime.datetime(1979, 2, 15, calendar="proleptic_gregorian"),
            ]
        ],
        dtype=object,
    )
    ds = xr.Dataset(
        data_vars={
            "TMP2m": xr.DataArray(
                data,
                dims=("sample", "time", "latitude", "longitude"),
                attrs={"units": "K", "long_name": "2m air temperature"},
            ),
            "counts": xr.DataArray(np.array([[124, 112]], dtype=np.int32), dims=("sample", "time")),
        },
        coords={
            "time": xr.DataArray(np.array([0, 1], dtype=np.int64), dims=("time",)),
            "latitude": xr.DataArray(lats, dims=("latitude",)),
            "longitude": xr.DataArray(lons, dims=("longitude",)),
            "valid_time": xr.DataArray(vt, dims=("sample", "time")),
        },
    )
    ds.to_netcdf(
        str(out_dir / "monthly_mean_predictions.nc"),
        engine="netcdf4",
        encoding={
            "valid_time": {
                "units": "days since 1970-01-01 00:00:00",
                "calendar": "proleptic_gregorian",
                "dtype": "int32",
            }
        },
    )


def test_horizon_calendar_and_step_conversions_and_non_integral_refusals() -> None:
    # 1. Proleptic Gregorian (`6h` step): leap year 1980 has 366 days -> 1464 steps
    steps_leap, end_leap = resolve_horizon(
        start_iso="1980-01-01T00:00:00",
        steps=None,
        length="1y",
        end_iso=None,
        timestep_hours=6,
        calendar="proleptic_gregorian",
    )
    assert (steps_leap, end_leap) == (1464, "1981-01-01T00:00:00")

    # 20-year Gregorian span 1979..1999 (5 leap years: 80,84,88,92,96 -> 7305 days * 4 = 29220 steps)
    steps_20y, end_20y = resolve_horizon(
        start_iso="1979-01-01T00:00:00",
        steps=None,
        length="20y",
        end_iso=None,
        timestep_hours=6,
        calendar="proleptic_gregorian",
    )
    assert (steps_20y, end_20y) == (29220, "1999-01-01T00:00:00")

    # 2. `noleap` (`365_day`): 1y = 1460 6h-steps; 5-day (`120h`) coupled model 1y = 73 steps!
    assert resolve_horizon(
        start_iso="1994-01-01T00:00:00",
        steps=None,
        length="1y",
        end_iso=None,
        timestep_hours=6,
        calendar="noleap",
    ) == (1460, "1995-01-01T00:00:00")

    assert resolve_horizon(
        start_iso="0311-01-01T00:00:00",
        steps=None,
        length="20y",
        end_iso=None,
        timestep_hours=120,
        calendar="noleap",
    ) == (1460, "0331-01-01T00:00:00")

    # Non-integral `6m` on 5-day (`120h`) `noleap` (Jan..Jun = 181 days = 4344h, not divisible by 120h) -> refused!
    with pytest.raises(ValidationError, match=r"not an (?:exact|integral) multiple"):
        resolve_horizon(
            start_iso="0311-01-01T00:00:00",
            steps=None,
            length="6m",
            end_iso=None,
            timestep_hours=120,
            calendar="noleap",
        )

    # 3. `360_day` calendar: every month is 30 days, so `6m` IS integral even for `120h` (180d / 5d = 36 steps)!
    assert resolve_horizon(
        start_iso="2000-01-01T00:00:00",
        steps=None,
        length="6m",
        end_iso=None,
        timestep_hours=120,
        calendar="360_day",
    ) == (36, "2000-07-01T00:00:00")
    assert resolve_horizon(
        start_iso="2000-01-01T00:00:00",
        steps=None,
        length="1y",
        end_iso=None,
        timestep_hours=6,
        calendar="360_day",
    ) == (1440, "2001-01-01T00:00:00")

    # 4. `--end <iso>` conversion + non-integral refusal (`12h` on `120h` step; `3h` on `6h` step)
    assert resolve_horizon(
        start_iso="1979-01-01T00:00:00",
        steps=None,
        length=None,
        end_iso="1979-01-31T00:00:00",
        timestep_hours=6,
        calendar="proleptic_gregorian",
    ) == (120, "1979-01-31T00:00:00")

    with pytest.raises(ValidationError, match=r"not an (?:exact|integral) multiple"):
        resolve_horizon(
            start_iso="1979-01-01T00:00:00",
            steps=None,
            length=None,
            end_iso="1979-01-01T03:00:00",
            timestep_hours=6,
            calendar="proleptic_gregorian",
        )

    # 5. Monthly `arches` family: `2y` = 24 monthly steps (first output 2015-03 -> 24th output 2017-02); `30d` refused
    assert resolve_horizon(
        start_iso="2015-01-01T00:00:00",
        steps=None,
        length="2y",
        end_iso=None,
        timestep_hours=730,
        calendar="proleptic_gregorian",
        family="arches",
    ) == (24, "2017-02-01T00:00:00")

    with pytest.raises(ValidationError, match="monthly steps"):
        resolve_horizon(
            start_iso="2015-01-01T00:00:00",
            steps=None,
            length="30d",
            end_iso=None,
            timestep_hours=730,
            calendar="proleptic_gregorian",
            family="arches",
        )


def test_family_adapter_allowlist_blocks_arbitrary_imports_and_commands(tmp_path: Path) -> None:
    assert {"fme", "e2s-dlesym", "arches", "emu"} <= set(INSTALLED_ADAPTERS.keys())
    for bad in ("os.system", "subprocess:Popen", "custom_pkg.adapter", "../evil", ""):
        with pytest.raises(RegistryError, match="installed adapters"):
            get_family_adapter(bad)

    ckpt = tmp_path / "c.tar"
    ckpt.write_bytes(b"ckpt")
    raw_entry = {
        "name": "ace2-era5",
        "family": "os.system",
        "fme_module": "fme.ace.inference",
        "checkpoint": str(ckpt),
        "ic_files": [str(ckpt)],
        "forcing_sets": [{"name": "amip-hist", "dir": str(tmp_path), "window": ["1979-01-01T00:00:00", "1980-01-01T00:00:00"]}],
        "timestep_hours": 6,
        "stochastic": False,
    }
    with pytest.raises(RegistryError):
        _parse_model_entry(raw_entry, str(tmp_path), str(tmp_path))


def test_structured_error_fields_and_stable_exit_codes(
    playground_env, capsys: pytest.CaptureFixture[str]
) -> None:
    assert EXIT_CODES["ok"] == 0
    assert EXIT_CODES["doctor_failed"] == 1
    assert EXIT_CODES["execution_error"] == 1
    assert EXIT_CODES["validation_error"] == 2
    assert EXIT_CODES["config_error"] == 2
    assert EXIT_CODES["registry_error"] == 2
    assert EXIT_CODES["missing_assets"] == 3
    assert EXIT_CODES["approval_error"] == 4
    assert EXIT_CODES["deadline"] == 75

    rc = main(["show", "not-a-model", "--json"])
    assert rc == 2
    err_env = json.loads(capsys.readouterr().out)
    assert err_env["aimip"] == "1" and err_env["ok"] is False
    err_obj = err_env["error"]
    assert err_obj["field"] == "model"
    assert "ace2-era5" in err_obj["allowed"]


def test_every_cli_verb_tty_and_json_plus_config_overrides_and_run_names(
    playground_env,
    fake_cluster_tools: dict[str, Path],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    import aimip

    data_dir = playground_env["playground_dir"]
    monkeypatch.setenv("AIMIP_DATA", str(data_dir))
    (data_dir / "VERSION").write_text(f"{aimip.__version__}\n", encoding="utf-8")

    # 1. `doctor` (pass on fakes; fail with fix_hint when GPU binary missing)
    assert main(["doctor"]) == 0
    doc_txt = capsys.readouterr().out
    assert "PASS   site_config" in doc_txt and "Overall: ok" in doc_txt
    assert main(["doctor", "--json"]) == 0
    doc_json = parse_cli_json(capsys.readouterr().out)
    assert doc_json["ok"] is True and doc_json["passed"] == doc_json["total"]

    # Slurm login node without local GPU (`nvidia-smi` absent) reports INFO (`rc == 0`),
    # whereas missing `sbatch` fails `scheduler` (`rc == 1`) with fix_hint.
    nvidia_smi_path = fake_cluster_tools["bin"] / "nvidia-smi"
    saved_nvidia_smi = nvidia_smi_path.read_text(encoding="utf-8")
    nvidia_smi_path.unlink()
    monkeypatch.setenv("PATH", str(fake_cluster_tools["bin"]))  # isolate from any system nvidia-smi
    assert main(["doctor"]) == 0
    info_txt = capsys.readouterr().out
    assert "INFO   gpu" in info_txt and "no GPU here — jobs run on GPU nodes via the scheduler" in info_txt

    sbatch_path = fake_cluster_tools["bin"] / "sbatch"
    saved_sbatch = sbatch_path.read_text(encoding="utf-8")
    sbatch_path.unlink()
    assert main(["doctor"]) == 1
    fail_txt = capsys.readouterr().out
    assert "FAIL   scheduler" in fail_txt and "fix:" in fail_txt
    sbatch_path.write_text(saved_sbatch, encoding="utf-8")
    sbatch_path.chmod(0o755)
    nvidia_smi_path.write_text(saved_nvidia_smi, encoding="utf-8")
    nvidia_smi_path.chmod(0o755)

    # 2. `models --task --family --status --format table|json`
    assert main(["models", "--task", "amip", "--status", "certified"]) == 0
    m_txt = capsys.readouterr().out
    assert "ace2-era5" in m_txt and "certified" in m_txt
    assert main(["models", "--family", "fme", "--format", "json"]) == 0
    m_json = parse_cli_json(capsys.readouterr().out)
    assert len(m_json) == 3

    # 3. `show <model> [--examples]` and `cite <model>`
    assert main(["show", "ace2-era5", "--examples"]) == 0
    s_txt = capsys.readouterr().out
    assert "ACE2-ERA5" in s_txt and "examples:" in s_txt
    assert main(["show", "ace2-era5", "--json"]) == 0
    assert parse_cli_json(capsys.readouterr().out)["status"] == "certified"

    assert main(["cite", "ace2-era5"]) == 0
    assert "@article" in capsys.readouterr().out
    assert main(["cite", "ace2-era5", "--json"]) == 0
    assert len(parse_cli_json(capsys.readouterr().out)["entries"]) >= 1

    # 4. `plan` (NO side effects!), TTY vs plain card, and `--config experiment.yaml` with `cli_overrides`
    exp_yaml = tmp_path / "experiment.yaml"
    exp_yaml.write_text(
        yaml.safe_dump(
            {
                "model": "ace2-era5",
                "start": "1979-01-01T00:00:00",
                "length": "30d",
                "seed": 3,
                "name": "demo",
            }
        ),
        encoding="utf-8",
    )
    assert main(["plan", "--config", str(exp_yaml), "--seed", "11"]) == 0
    plan_plain = capsys.readouterr().out
    assert "PLAN demo" in plan_plain
    assert "120 steps" in plan_plain
    assert "overrides:   seed=11" in plan_plain
    expected_demo_dir = Path(playground_env["default_out_root"]) / "demo"
    assert not expected_demo_dir.exists()  # `plan` MUST have zero side effects!

    # Verify TTY ANSI rendering vs plain non-TTY rendering
    p_obj = services.plan(RunSpec(model="ace2-era5", start="1979-01-01T00:00:00", length="30d", name="demo"))
    tty_card = services.render_plan_card(p_obj, is_tty=True)
    plain_card = services.render_plan_card(p_obj, is_tty=False)
    assert "\x1b[1m" in tty_card and "\x1b[" not in plain_card

    assert main(["plan", "--config", str(exp_yaml), "--seed", "11", "--json"]) == 0
    plan_json = parse_cli_json(capsys.readouterr().out)
    assert plan_json["steps"] == 120
    assert plan_json["cli_overrides"] == ["seed=11"]

    # 5. `run --config ... --submit` prepares `<run_root>/demo` AND keeps `Next:     aimip submit`
    assert main(["run", "--config", str(exp_yaml), "--seed", "11", "--submit"]) == 0
    run_txt = capsys.readouterr().out
    assert f"Next:     aimip submit {expected_demo_dir}" in run_txt
    assert (expected_demo_dir / "plan.json").is_file()
    assert (expected_demo_dir / "experiment.yaml").is_file()

    # 6. `submit demo` (by bare run name!), `watch demo`, `logs demo`, `ls`, `summary demo`, `cancel`
    assert main(["submit", "demo", "--json"]) == 0
    sub_json = parse_cli_json(capsys.readouterr().out)
    assert sub_json["slurm_job_id"] == "654321"
    assert (expected_demo_dir / "submit-intent.json").is_file()

    assert main(["watch", "demo", "--interval", "0.05", "--json"]) == 0
    w_cap = capsys.readouterr()
    assert "[1] demo" in w_cap.err  # progress on stderr when `--json` is passed!
    w_json = parse_cli_json(w_cap.out)
    assert w_json["state"] == "completed" and w_json["terminal"] is True

    # Plant log containing ANSI/OSC terminal escape injection and verify `logs` strips control bytes
    (expected_demo_dir / "slurm-654321.out").write_text(
        "\x1b[31mSTEP 1\x1b[0m\r\n\x1b]52;c;ZXY=\x07STEP 2\x08 CLEAN\n",
        encoding="utf-8",
    )
    assert main(["logs", "demo", "--max-bytes", "1024"]) == 0
    log_txt = capsys.readouterr().out
    assert "\x1b" not in log_txt and "\x08" not in log_txt
    assert "STEP 1" in log_txt and "STEP 2 CLEAN" in log_txt

    # Symlinked log file inside run_dir is refused!
    bad_sym = expected_demo_dir / "slurm-999.err"
    bad_sym.symlink_to(exp_yaml)
    assert main(["logs", "demo"]) == 2
    capsys.readouterr()
    bad_sym.unlink()

    assert main(["ls"]) == 0
    assert "demo" in capsys.readouterr().out
    assert main(["ls", "--json"]) == 0
    ls_json = parse_cli_json(capsys.readouterr().out)
    assert ls_json[0]["name"] == "demo" and ls_json[0]["job_id"] == "654321"

    _write_monthly_nc(expected_demo_dir, mean_val=289.25)
    assert main(["summary", "demo", "--json"]) == 0
    sum_json = parse_cli_json(capsys.readouterr().out)
    assert sum_json["variables"]["TMP2m"]["series"][0]["ensemble_mean"] == pytest.approx(289.25)

    # `demo` is already COMPLETED -> `cancel` truthfully refuses terminal state (`rc == 2`)!
    assert main(["cancel", "demo", "--json"]) == 2
    err_can = json.loads(capsys.readouterr().out)
    assert err_can["ok"] is False and "terminal state" in err_can["error"]["message"]

    # Active RUNNING job cancels cleanly and records `cancel_requested` without writing fake `CANCELLED` sacct.txt
    (fake_cluster_tools["bin"] / "sacct").write_text(
        '#!/bin/bash\necho "654322|RUNNING|00:00:20|0:0"\n',
        encoding="utf-8",
    )
    (fake_cluster_tools["bin"] / "sbatch").write_text(
        "#!/bin/bash\n"
        f'echo "$@" >> "{fake_cluster_tools["logs"] / "sbatch.log"}"\n'
        "echo 654322\n",
        encoding="utf-8",
    )
    assert main(["run", "ace2-era5", "--start", "1979-01-01", "--steps", "4", "--name", "demo-live", "--dry-run"]) == 0
    capsys.readouterr()
    assert main(["submit", "demo-live", "--json"]) == 0
    capsys.readouterr()
    assert main(["cancel", "demo-live", "--json"]) == 0
    can_json = parse_cli_json(capsys.readouterr().out)
    assert can_json["cancelled"] is True and can_json["attempt_id"] == "654322"
    assert "654322" in (fake_cluster_tools["logs"] / "scancel.log").read_text(encoding="utf-8")

    # 7. `completion bash|zsh`
    for sh in ("bash", "zsh"):
        assert main(["completion", sh]) == 0
        comp_out = capsys.readouterr().out
        assert "propose" not in comp_out or "plan" in comp_out
        assert "ace2-era5" in comp_out and "doctor" in comp_out


def test_submit_intent_and_squeue_reconciliation_never_blindly_retries(
    playground_env,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """When `sbatch` fails/times out uncertainly, `submit-intent.json` reconciles via `squeue` and never double-submits."""
    fbin = tmp_path / "reconcile_bin"
    fbin.mkdir()
    sbatch_count = tmp_path / "sbatch_count.txt"

    # `sbatch` increments counter and exits 1 (simulating dropped Slurm RPC after controller queued job 777888)
    sbatch = fbin / "sbatch"
    sbatch.write_text(
        "#!/bin/bash\n"
        f'echo 1 >> "{sbatch_count}"\n'
        'echo "slurm_receive_msg: Socket timed out" >&2\n'
        "exit 1\n",
        encoding="utf-8",
    )
    sbatch.chmod(0o755)

    squeue = fbin / "squeue"
    squeue.write_text("#!/bin/bash\nexit 0\n", encoding="utf-8")
    squeue.chmod(0o755)
    monkeypatch.setenv("PATH", f"{fbin}:{os.environ.get('PATH', '')}")

    assert main(["run", "ace2-era5", "--start", "1979-01-01", "--steps", "4", "--name", "uncertain1", "--dry-run"]) == 0
    capsys.readouterr()
    rdir1 = Path(playground_env["default_out_root"]) / "uncertain1"

    # First `submit` sees `sbatch` fail and `squeue` empty -> records uncertain intent and returns ExecutionError (exit 1)
    assert main(["submit", "uncertain1"]) == EXIT_CODES["execution_error"]
    capsys.readouterr()
    assert len(sbatch_count.read_text(encoding="utf-8").splitlines()) == 1
    intent = json.loads((rdir1 / "submit-intent.json").read_text(encoding="utf-8"))
    assert intent["state"] == "uncertain"

    # Second `submit` when `squeue` STILL shows nothing -> refuses blind retry WITHOUT calling `sbatch` again!
    assert main(["submit", "uncertain1"]) == EXIT_CODES["execution_error"]
    assert "refusing blind retry" in capsys.readouterr().err
    assert len(sbatch_count.read_text(encoding="utf-8").splitlines()) == 1

    # Same-name job WITHOUT matching `attempt_nonce` MUST NOT be adopted!
    squeue.write_text(
        "#!/bin/bash\n"
        f'echo "999111|{intent["job_name"]}|PENDING|aimip-attempt-unrelated000"\n',
        encoding="utf-8",
    )
    assert main(["submit", "uncertain1"]) == EXIT_CODES["execution_error"]
    capsys.readouterr()
    assert not (rdir1 / "slurm.jobid").exists()
    assert len(sbatch_count.read_text(encoding="utf-8").splitlines()) == 1

    # Exact `(user, job_name, attempt_nonce)` match reconciles cleanly WITHOUT calling `sbatch` again!
    squeue.write_text(
        "#!/bin/bash\n"
        f'echo "777888|{intent["job_name"]}|PENDING|{intent["attempt_nonce"]}"\n',
        encoding="utf-8",
    )
    assert main(["submit", "uncertain1", "--json"]) == 0
    rec = parse_cli_json(capsys.readouterr().out)
    assert rec["reconciled"] is True and rec["slurm_job_id"] == "777888"
    assert len(sbatch_count.read_text(encoding="utf-8").splitlines()) == 1
