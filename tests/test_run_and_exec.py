"""Tests for `aimip run`, submit-gate `job.sbatch` contract, `restart.nc` continuation, `exec-fme`, `status`, and `summary`."""

from __future__ import annotations

import json
import math
import os
import pathlib
import re
import subprocess
import sys
from typing import Any

import cftime
import numpy as np
import pytest
import xarray as xr
import yaml

from aimip.cli import main
from aimip.runner import estimate_slurm_walltime
from aimip.validators import validate_safe_path

from conftest import write_fake_ic_netcdf


def test_models_and_run_dry_run_work_without_torch_installed(
    playground_env: dict[str, Any],
) -> None:
    """Verify `models` and `run --dry-run` never import `torch`."""
    import aimip

    src_root = pathlib.Path(aimip.__file__).resolve().parent.parent
    scratch_sim: pathlib.Path = playground_env["scratch_sim"]
    out_dir = playground_env["runs_dir"] / "no_torch_run"

    child_code = (
        "import sys\n"
        "sys.modules['torch'] = None\n"
        "import aimip.validators as v\n"
        f"v.ALLOWED_RUN_ROOTS = ({str(scratch_sim) + '/'!r},)\n"
        "from aimip.cli import main\n"
        "assert main(['models']) == 0\n"
        f"assert main(['run', '-m', 'ace2-era5', '--start', '1979-01-01T00:00:00', "
        f"'--steps', '8', '--out', {str(out_dir)!r}, '--dry-run']) == 0\n"
        "assert sys.modules.get('torch') is None\n"
    )
    env = dict(os.environ)
    env["PYTHONPATH"] = str(src_root)
    proc = subprocess.run(
        [sys.executable, "-c", child_code],
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, f"stderr:\n{proc.stderr}\nstdout:\n{proc.stdout}"
    assert (out_dir / "config.yaml").is_file()
    assert (out_dir / "job.sbatch").is_file()
    assert (out_dir / "aimip-manifest.json").is_file()


def test_run_config_yaml_sbatch_marker_body_and_submit_prompt(
    playground_env: dict[str, Any],
    capsys: pytest.CaptureFixture[str],
) -> None:
    out_dir: pathlib.Path = playground_env["runs_dir"] / "era5_exact"
    pg_dir: pathlib.Path = playground_env["playground_dir"]

    rc = main(
        [
            "run",
            "-m",
            "ace2-era5",
            "--start",
            "2001-01-01T00:00:00",
            "--steps",
            "100",
            "--seed",
            "7",
            "--out",
            str(out_dir),
            "--submit",
        ]
    )
    assert rc == 0
    submit_stdout = capsys.readouterr().out
    # `run --submit` prepares the run directory and prints `Next:     aimip submit <dir>`.
    assert f"Next:     aimip submit {out_dir}" in submit_stdout

    parsed_cfg = yaml.safe_load((out_dir / "config.yaml").read_text(encoding="utf-8"))
    expected_cfg = {
        "experiment_dir": str(out_dir),
        "n_forward_steps": 100,
        "forward_steps_in_memory": 40,
        "checkpoint_path": str(playground_env["era5_ckpt"]),
        "n_ensemble_per_ic": 1,
        "logging": {
            "log_to_screen": True,
            "log_to_wandb": False,
            "log_to_file": True,
            "project": "ace",
        },
        "initial_condition": {
            "path": str(playground_env["era5_ic_2001"]),
            "start_indices": {
                "times": ["2001-01-01T00:00:00"],
            },
        },
        "forcing_loader": {
            "dataset": {
                "data_path": str(pg_dir / "inputs" / "ace2-era5" / "amip-hist"),
            },
            "num_data_workers": 4,
        },
        "data_writer": {
            "save_prediction_files": False,
            "save_monthly_files": True,
        },
    }
    assert parsed_cfg == expected_cfg

    # Verify submit-gate contract:
    # - Exact line 2 `# aimip-generated v1` (`sed -n 2p`)
    # - Every line matches `^(#|$|set -euo pipefail$|"<PG>/bin/aimip" exec-fme "/[A-Za-z0-9._/+-]+"$)`
    sbatch_text = (out_dir / "job.sbatch").read_text(encoding="utf-8")
    sbatch_lines = sbatch_text.splitlines()
    assert sbatch_lines[0] == "#!/bin/bash"
    assert sbatch_lines[1] == "# aimip-generated v1"
    assert "#SBATCH --partition=gpu" in sbatch_lines
    assert "#SBATCH --account=test_gpu" in sbatch_lines
    assert "#SBATCH --gres=gpu:a100_80:1" in sbatch_lines
    assert "#SBATCH --cpus-per-task=32" in sbatch_lines
    assert "#SBATCH --hint=nomultithread" in sbatch_lines
    assert "#SBATCH --mem=64G" in sbatch_lines
    assert "#SBATCH --time=00:20:00" in sbatch_lines
    wrapper_guard_re = re.compile(
        rf'^(#|$|set -euo pipefail$|"{re.escape(str(pg_dir))}/bin/aimip" exec-fme "/[A-Za-z0-9._/+-]+"$)'
    )
    bad_lines = [ln for ln in sbatch_lines if not wrapper_guard_re.match(ln)]
    assert bad_lines == []
    body_lines = [ln for ln in sbatch_lines if ln.strip() and not ln.startswith("#")]
    assert body_lines == [
        "set -euo pipefail",
        f'"{pg_dir}/bin/aimip" exec-fme "{out_dir}"',
    ]

    manifest = json.loads((out_dir / "aimip-manifest.json").read_text(encoding="utf-8"))
    assert manifest["forcing_set"]["name"] == "amip-hist"
    assert manifest["ic_spatial_dims"] == ["latitude", "longitude"]
    assert manifest["seed"] == 7
    assert manifest["checkpoint_sha256"] == playground_env["era5_sha"]
    assert manifest["forcing_files"] == [
        {"name": "forcing_1979.nc", "size_bytes": 128},
        {"name": "forcing_2001.nc", "size_bytes": 256},
    ]


def test_default_out_dir_uses_default_out_root_and_passes_safe_path(
    playground_env: dict[str, Any],
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Omitted `--out` creates `$AIMIP_DEFAULT_OUT_ROOT/<model>-<start>-<short_hash>`."""
    assert (
        main(
            [
                "run",
                "-m",
                "ace2-era5",
                "--start",
                "1979-01-01T00:00:00",
                "--steps",
                "4",
                "--dry-run",
                "--json",
            ]
        )
        == 0
    )
    manifest = json.loads(capsys.readouterr().out)
    out_dir = manifest["out_dir"]
    default_root = str(playground_env["default_out_root"])
    assert out_dir.startswith(f"{default_root}/ace2-era5-1979-01-01T000000-")
    # Verify no colons or characters forbidden by the submit-gate safe-path check.
    assert validate_safe_path(out_dir) == out_dir


def test_cross_forcing_set_refusal_and_restart_nc_continuation(
    playground_env: dict[str, Any],
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Crossing `amip-hist` -> `amip-oos` refuses cleanly and suggests splitting via `restart.nc`."""
    runs_dir: pathlib.Path = playground_env["runs_dir"]

    # 1. Attempting 1979-01-01 -> 2016 in one shot on `ace2s-shield-plus` crosses amip-hist -> amip-oos.
    rc_cross = main(
        [
            "run",
            "-m",
            "ace2s-shield-plus",
            "--start",
            "1979-01-01T00:00:00",
            "--steps",
            "54000",
            "--out",
            str(runs_dir / "cross_fail"),
            "--dry-run",
        ]
    )
    assert rc_cross == 2
    err = capsys.readouterr().err
    assert "crosses forcing sets" in err
    assert "restart.nc" in err

    # 2. Segment 1 runs in `amip-hist` up to `2014-01-01T00:00:00` (51136 steps across 9 leap years)
    # with 3 stochastic ensemble members, producing `<seg1>/restart.nc` with 3 samples at 2014-01-01.
    seg1 = runs_dir / "shield_seg1"
    assert (
        main(
            [
                "run",
                "-m",
                "ace2s-shield-plus",
                "--start",
                "1979-01-01T00:00:00",
                "--steps",
                "51136",
                "--members",
                "3",
                "--seed",
                "42",
                "--out",
                str(seg1),
                "--dry-run",
            ]
        )
        == 0
    )
    capsys.readouterr()
    seg1_manifest = json.loads((seg1 / "aimip-manifest.json").read_text(encoding="utf-8"))
    assert seg1_manifest["forcing_set"]["name"] == "amip-hist"
    assert seg1_manifest["end_time"] == "2014-01-01T00:00:00"

    # Simulate fme writing `<seg1>/restart.nc` with 3 ensemble members stamped `2014-01-01T00:00:00`.
    seg1_restart = write_fake_ic_netcdf(
        seg1 / "restart.nc",
        [
            "2014-01-01T00:00:00",
            "2014-01-01T00:00:00",
            "2014-01-01T00:00:00",
        ],
        sample_dim="sample",
        spatial_dims=("lat", "lon"),
        seed=77,
    )

    # 3. Segment 2 continues from `--ic <seg1>/restart.nc` at `2014-01-01T00:00:00` for 2920 steps
    # (to 2016-01-01T00:00:00), automatically selecting `amip-oos`!
    seg2 = runs_dir / "shield_seg2"
    assert (
        main(
            [
                "run",
                "-m",
                "ace2s-shield-plus",
                "--start",
                "2014-01-01T00:00:00",
                "--steps",
                "2920",
                "--ic",
                str(seg1_restart),
                "--seed",
                "43",
                "--out",
                str(seg2),
                "--dry-run",
            ]
        )
        == 0
    )
    capsys.readouterr()
    seg2_manifest = json.loads((seg2 / "aimip-manifest.json").read_text(encoding="utf-8"))
    assert seg2_manifest["forcing_set"]["name"] == "amip-oos"
    assert seg2_manifest["forcing_dir"] == str(playground_env["ace2s_oos"])
    assert seg2_manifest["members"] == 3
    assert seg2_manifest["n_ic_samples"] == 3
    assert seg2_manifest["n_ensemble_per_ic"] == 1

    # 4. If `--start` does NOT match `restart.nc`'s timestamp, `aimip run` refuses cleanly!
    assert (
        main(
            [
                "run",
                "-m",
                "ace2s-shield-plus",
                "--start",
                "2015-01-01T00:00:00",
                "--steps",
                "100",
                "--ic",
                str(seg1_restart),
                "--out",
                str(runs_dir / "seg2_wrong_start"),
                "--dry-run",
            ]
        )
        == 2
    )
    assert "must equal --start" in capsys.readouterr().err


def test_members_logic_deterministic_vs_stochastic_and_custom_ic(
    playground_env: dict[str, Any],
    capsys: pytest.CaptureFixture[str],
) -> None:
    runs_dir: pathlib.Path = playground_env["runs_dir"]

    # Deterministic model + shipped IC + --members 3 -> refuses & suggests `ic spin`.
    assert (
        main(
            [
                "run",
                "-m",
                "ace2-era5",
                "--start",
                "1979-01-01T00:00:00",
                "--steps",
                "4",
                "--members",
                "3",
                "--out",
                str(runs_dir / "det_fail"),
                "--dry-run",
            ]
        )
        == 2
    )
    assert "ic spin" in capsys.readouterr().err

    # Subsetting --members 2 from 3-sample custom IC uses InferenceInitialConditionIndices.
    custom_ic = write_fake_ic_netcdf(
        runs_dir / "custom_3.nc",
        [
            "1980-03-01T00:00:00",
            "1980-03-01T00:00:00",
            "1980-03-01T00:00:00",
        ],
        sample_dim="sample",
        spatial_dims=("latitude", "longitude"),
        seed=50,
    )
    det_2_out = runs_dir / "det_custom_2"
    assert (
        main(
            [
                "run",
                "-m",
                "ace2-era5",
                "--start",
                "1980-03-01T00:00:00",
                "--steps",
                "8",
                "--ic",
                str(custom_ic),
                "--members",
                "2",
                "--out",
                str(det_2_out),
                "--dry-run",
            ]
        )
        == 0
    )
    capsys.readouterr()
    det_2_cfg = yaml.safe_load((det_2_out / "config.yaml").read_text(encoding="utf-8"))
    assert det_2_cfg["initial_condition"] == {
        "path": str(custom_ic),
        "start_indices": {"n_initial_conditions": 2, "first": 0, "interval": 1},
    }


def test_refuses_output_inside_playground_and_walltime_scaling(
    playground_env: dict[str, Any],
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    import aimip.validators as v_mod

    pg_dir: pathlib.Path = playground_env["playground_dir"]
    # Temporarily allow pg_dir root prefix in run-root check so we hit the explicit playground guard.
    monkeypatch.setattr(
        v_mod,
        "ALLOWED_RUN_ROOTS",
        (*v_mod.ALLOWED_RUN_ROOTS, f"{pg_dir}/"),
    )
    rc = main(
        [
            "run",
            "-m",
            "ace2-era5",
            "--start",
            "1979-01-01T00:00:00",
            "--steps",
            "4",
            "--out",
            str(pg_dir / "illegal_out"),
            "--dry-run",
        ]
    )
    assert rc == 2
    assert "AIMIP_PLAYGROUND" in capsys.readouterr().err

    assert estimate_slurm_walltime(steps=100, timestep_hours=6) == "00:20:00"
    assert estimate_slurm_walltime(steps=52595, timestep_hours=6) == "06:57:19"
    assert (
        estimate_slurm_walltime(steps=52595, timestep_hours=6, n_trajectories=5)
        == "12:00:00"
    )


def test_exec_fme_status_submitted_txt_and_summary_custom_dims(
    playground_env: dict[str, Any],
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    runs_dir: pathlib.Path = playground_env["runs_dir"]
    out_dir = runs_dir / "local_exec_test"

    assert (
        main(
            [
                "run",
                "-m",
                "ace2-era5",
                "--start",
                "1979-01-01T00:00:00",
                "--steps",
                "8",
                "--seed",
                "314",
                "--out",
                str(out_dir),
                "--dry-run",
            ]
        )
        == 0
    )
    capsys.readouterr()

    # Before execution and without `sacct`, status state is "unknown".
    monkeypatch.setenv("PATH", "")
    assert main(["status", str(out_dir), "--json"]) == 0
    pre_status = json.loads(capsys.readouterr().out)
    assert pre_status["state"] == "unknown"

    # Build a tiny fake `fme` package in tmp_path to exercise `exec-fme`.
    fake_pkg_root = tmp_path / "fake_site_packages"
    (fake_pkg_root / "fme" / "ace" / "inference").mkdir(parents=True)
    (fake_pkg_root / "fme" / "__init__.py").write_text(
        '__version__ = "2026.5.1"\n', encoding="utf-8"
    )
    (fake_pkg_root / "fme" / "core.py").write_text(
        "LAST_SEED = None\n"
        "def set_seed(seed: int) -> None:\n"
        "    global LAST_SEED\n"
        "    LAST_SEED = seed\n",
        encoding="utf-8",
    )
    (fake_pkg_root / "fme" / "ace" / "__init__.py").write_text("", encoding="utf-8")
    (fake_pkg_root / "fme" / "ace" / "inference" / "__init__.py").write_text(
        "", encoding="utf-8"
    )
    (fake_pkg_root / "fme" / "ace" / "inference" / "__main__.py").write_text(
        "import json, pathlib, sys, yaml\n"
        "from fme.core import LAST_SEED\n"
        "cfg_path = sys.argv[1]\n"
        "cfg = yaml.safe_load(pathlib.Path(cfg_path).read_text())\n"
        "exp = pathlib.Path(cfg['experiment_dir'])\n"
        "(exp / 'seed_recorded.json').write_text(json.dumps({'seed': LAST_SEED}))\n",
        encoding="utf-8",
    )
    for mod_key in [k for k in sys.modules if k == "fme" or k.startswith("fme.")]:
        monkeypatch.delitem(sys.modules, mod_key, raising=False)
    monkeypatch.syspath_prepend(str(fake_pkg_root))
    monkeypatch.setenv("SLURM_JOB_ID", "987654")

    assert main(["exec-fme", str(out_dir)]) == 0
    capsys.readouterr()

    seed_rec = json.loads((out_dir / "seed_recorded.json").read_text(encoding="utf-8"))
    assert seed_rec == {"seed": 314}

    # Test `summary` with non-`lat`/`lon` horizontal dim names (`grid_yt`/`grid_xt`) to prove
    # zero hardcoded dim names in `summarize_run`!
    lats = np.array([-60.0, 0.0, 0.0, 60.0], dtype=np.float64)
    lons = np.linspace(0.0, 315.0, 8, dtype=np.float64)
    m0 = np.broadcast_to(np.array([260.0, 300.0, 300.0, 260.0])[:, None], (4, 8))
    m1 = np.full((4, 8), 290.0, dtype=np.float64)
    tmp2m = np.stack([m0, m1], axis=0)[None, ...]

    vt = np.array(
        [
            [
                cftime.datetime(1979, 1, 15, calendar="proleptic_gregorian"),
                cftime.datetime(1979, 2, 15, calendar="proleptic_gregorian"),
            ]
        ],
        dtype=object,
    )
    monthly_ds = xr.Dataset(
        data_vars={
            "TMP2m": xr.DataArray(
                tmp2m.astype(np.float32),
                dims=("sample", "time", "grid_yt", "grid_xt"),
                attrs={"units": "K", "long_name": "2m air temperature"},
            ),
            "counts": xr.DataArray(
                np.array([[124, 112]], dtype=np.int32),
                dims=("sample", "time"),
            ),
        },
        coords={
            "time": xr.DataArray(
                np.array([0, 1], dtype=np.int64),
                dims=("time",),
                attrs={"units": "months"},
            ),
            "grid_yt": xr.DataArray(lats, dims=("grid_yt",)),
            "grid_xt": xr.DataArray(lons, dims=("grid_xt",)),
            "valid_time": xr.DataArray(vt, dims=("sample", "time")),
        },
    )
    monthly_ds.to_netcdf(
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

    assert main(["summary", str(out_dir), "--json"]) == 0
    summary = json.loads(capsys.readouterr().out)
    assert summary["spatial_dims"] == ["grid_yt", "grid_xt"]
    series = summary["variables"]["TMP2m"]["series"]
    assert math.isclose(series[0]["ensemble_mean"], 860.0 / 3.0, rel_tol=1e-6)
    assert math.isclose(series[1]["ensemble_mean"], 290.0, rel_tol=1e-6)


def test_python_dash_m_aimip_submitted_txt_sacct_and_failed_exec(
    playground_env: dict[str, Any],
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    import aimip

    src_root = pathlib.Path(aimip.__file__).resolve().parent.parent
    env = dict(os.environ)
    env["PYTHONPATH"] = str(src_root)
    proc = subprocess.run(
        [sys.executable, "-m", "aimip", "--version"],
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0
    assert "aimip 1.3.0" in proc.stdout

    # Verify `status` parses `submitted.txt` (`job=424242 submitted=... user=...`).
    runs_dir: pathlib.Path = playground_env["runs_dir"]
    pending_dir = runs_dir / "sacct_test"
    assert (
        main(
            [
                "run",
                "-m",
                "ace2-era5",
                "--start",
                "1979-01-01T00:00:00",
                "--steps",
                "4",
                "--out",
                str(pending_dir),
                "--submit",
            ]
        )
        == 0
    )
    capsys.readouterr()
    (pending_dir / "submitted.txt").write_text(
        "job=424242;cluster submitted=2025-01-01T02:00:00+01:00 user=user_x\n",
        encoding="utf-8",
    )

    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    sacct_script = fake_bin / "sacct"
    sacct_script.write_text(
        f"#!{sys.executable}\n"
        'print("424242|RUNNING|00:03:19|0:0")\n'
        'print("424242.batch|RUNNING|00:03:19|0:0")\n',
        encoding="utf-8",
    )
    sacct_script.chmod(0o755)
    monkeypatch.setenv("PATH", str(fake_bin))

    assert main(["status", str(pending_dir), "--json"]) == 0
    st = json.loads(capsys.readouterr().out)
    assert st["state"] == "running"
    assert st["slurm_job_id"] == "424242"
    assert st["submitted"]["user"] == "user_x"

    # Test `exec-fme` records non-zero rc in manifest when fme exits with code 5.
    fail_pkg = tmp_path / "fail_fme_pkg"
    (fail_pkg / "fme" / "ace" / "inference").mkdir(parents=True)
    (fail_pkg / "fme" / "__init__.py").write_text('__version__ = "2026.5.1"\n', encoding="utf-8")
    (fail_pkg / "fme" / "core.py").write_text(
        "def set_seed(seed: int) -> None:\n    pass\n", encoding="utf-8"
    )
    (fail_pkg / "fme" / "ace" / "__init__.py").write_text("", encoding="utf-8")
    (fail_pkg / "fme" / "ace" / "inference" / "__init__.py").write_text("", encoding="utf-8")
    (fail_pkg / "fme" / "ace" / "inference" / "__main__.py").write_text(
        "import sys\nsys.exit(5)\n", encoding="utf-8"
    )
    for mod_key in [k for k in sys.modules if k == "fme" or k.startswith("fme.")]:
        monkeypatch.delitem(sys.modules, mod_key, raising=False)
    monkeypatch.syspath_prepend(str(fail_pkg))

    with pytest.raises(SystemExit) as exc_info:
        main(["exec-fme", str(pending_dir)])
    assert exc_info.value.code == 5

    failed_manifest = json.loads(
        (pending_dir / "aimip-manifest.json").read_text(encoding="utf-8")
    )
    assert failed_manifest["rc"] == 5
    assert "finished" in failed_manifest


def test_stochastic_members_replicates_ic_members_nc_and_keeps_k1_untouched(
    playground_env: dict[str, Any],
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Stochastic `--members K` (`K > 1`) writes `ic_members.nc` with `n_ensemble_per_ic: 1`."""
    runs_dir: pathlib.Path = playground_env["runs_dir"]

    # 1. K=1 keeps the original shipped IC path untouched.
    out_k1 = runs_dir / "shield_stoch_k1"
    assert (
        main(
            [
                "run",
                "-m",
                "ace2s-shield-plus",
                "--start",
                "1979-01-01T00:00:00",
                "--steps",
                "4",
                "--members",
                "1",
                "--out",
                str(out_k1),
                "--dry-run",
            ]
        )
        == 0
    )
    capsys.readouterr()
    assert not (out_k1 / "ic_members.nc").exists()
    cfg_k1 = yaml.safe_load((out_k1 / "config.yaml").read_text(encoding="utf-8"))
    assert cfg_k1["n_ensemble_per_ic"] == 1
    assert cfg_k1["initial_condition"]["path"] == str(playground_env["ace2s_ic"])

    # 2. K=3 replicates the IC 3 times along `sample` into `<out>/ic_members.nc` with `n_ensemble_per_ic: 1`.
    out_k3_a = runs_dir / "shield_stoch_k3_a"
    out_k3_b = runs_dir / "shield_stoch_k3_b"
    for out_d in (out_k3_a, out_k3_b):
        assert (
            main(
                [
                    "run",
                    "-m",
                    "ace2s-shield-plus",
                    "--start",
                    "1979-01-01T00:00:00",
                    "--steps",
                    "4",
                    "--members",
                    "3",
                    "--out",
                    str(out_d),
                    "--dry-run",
                ]
            )
            == 0
        )
        capsys.readouterr()

    ic_mem_a = out_k3_a / "ic_members.nc"
    ic_mem_b = out_k3_b / "ic_members.nc"
    assert ic_mem_a.is_file()
    assert ic_mem_a.read_bytes() == ic_mem_b.read_bytes()

    cfg_k3 = yaml.safe_load((out_k3_a / "config.yaml").read_text(encoding="utf-8"))
    assert cfg_k3["n_ensemble_per_ic"] == 1
    assert cfg_k3["initial_condition"] == {"path": str(ic_mem_a)}

    manifest_k3 = json.loads((out_k3_a / "aimip-manifest.json").read_text(encoding="utf-8"))
    assert manifest_k3["members"] == 3
    assert manifest_k3["n_ic_samples"] == 3
    assert manifest_k3["n_ensemble_per_ic"] == 1
    assert manifest_k3["ensemble_strategy"] == "replicated_ic_sample_dim"
    assert manifest_k3["ic_path"] == str(ic_mem_a)
    assert manifest_k3["source_ic_path"] == str(playground_env["ace2s_ic"])
    assert "model noise" in manifest_k3["ensemble_note"]

    with xr.open_dataset(str(ic_mem_a), engine="netcdf4", decode_times=False) as ds:
        assert ds.sizes["sample"] == 3
        assert "grid_yt" in ds.dims and "grid_xt" in ds.dims
        assert np.allclose(ds["TMP2m"].isel(sample=0).values, ds["TMP2m"].isel(sample=2).values)


def test_status_sacct_txt_sacct_json_and_local_pid_without_container_sacct(
    playground_env: dict[str, Any],
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """`status` works without `sacct` binary via `<dir>/sacct.txt`, `--sacct-json`, or `local.pid`."""
    monkeypatch.setenv("PATH", "")
    runs_dir: pathlib.Path = playground_env["runs_dir"]

    # 1. Host wrapper writes `<dir>/slurm.jobid` + `<dir>/sacct.txt` (`sacct -n -X -P -o JobID,State,Elapsed,ExitCode`).
    dir_txt = runs_dir / "status_sacct_txt"
    assert (
        main(
            [
                "run",
                "-m",
                "ace2-era5",
                "--start",
                "1979-01-01T00:00:00",
                "--steps",
                "4",
                "--out",
                str(dir_txt),
                "--dry-run",
            ]
        )
        == 0
    )
    capsys.readouterr()
    (dir_txt / "slurm.jobid").write_text("777001\n", encoding="utf-8")
    (dir_txt / "sacct.txt").write_text(
        "777001|COMPLETED|00:05:11|0:0\n777001.batch|COMPLETED|00:05:11|0:0\n",
        encoding="utf-8",
    )

    assert main(["status", str(dir_txt), "--json"]) == 0
    st_txt = json.loads(capsys.readouterr().out)
    assert st_txt["state"] == "completed"
    assert st_txt["slurm_job_id"] == "777001"
    assert st_txt["sacct_available"] is True
    assert st_txt["sacct_source"] == "sacct.txt"
    assert st_txt["sacct"]["elapsed"] == "00:05:11"

    # 2. `--sacct-json FILE` with SLURM's native `sacct --json` `"jobs"` schema.
    sacct_json_file = tmp_path / "sacct_out.json"
    sacct_json_file.write_text(
        json.dumps(
            {
                "jobs": [
                    {
                        "job_id": 777001,
                        "state": {"current": ["FAILED"]},
                        "time": {"elapsed": 125},
                        "derived_exit_code": {
                            "return_code": {"number": 2},
                            "signal": {"id": {"number": 0}},
                        },
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    assert main(["status", str(dir_txt), "--sacct-json", str(sacct_json_file), "--json"]) == 0
    st_json = json.loads(capsys.readouterr().out)
    assert st_json["state"] == "failed"
    assert st_json["sacct_source"] == "sacct-json"
    assert st_json["sacct"]["elapsed"] == "00:02:05"
    assert st_json["sacct"]["exit_code"] == "2:0"

    # 3. `local.pid` written by wrapper when `scheduler == "none"`.
    dir_local = runs_dir / "status_local_pid"
    assert (
        main(
            [
                "run",
                "-m",
                "ace2-era5",
                "--start",
                "1979-01-01T00:00:00",
                "--steps",
                "4",
                "--out",
                str(dir_local),
                "--dry-run",
            ]
        )
        == 0
    )
    capsys.readouterr()
    (dir_local / "local.pid").write_text(f"{os.getpid()}\n", encoding="utf-8")
    assert main(["status", str(dir_local), "--json"]) == 0
    st_loc = json.loads(capsys.readouterr().out)
    assert st_loc["state"] == "running"
    assert st_loc["local_pid"] == os.getpid()


def test_exec_fme_silences_git_not_a_repository_stderr(
    playground_env: dict[str, Any],
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """`exec-fme` sets cwd + GIT_CEILING_DIRECTORIES and silences `git rev-parse` stderr."""
    runs_dir: pathlib.Path = playground_env["runs_dir"]
    out_dir = runs_dir / "quiet_git_run"
    assert (
        main(
            [
                "run",
                "-m",
                "ace2-era5",
                "--start",
                "1979-01-01T00:00:00",
                "--steps",
                "4",
                "--out",
                str(out_dir),
                "--dry-run",
            ]
        )
        == 0
    )
    capsys.readouterr()

    fake_pkg = tmp_path / "quiet_fme_pkg"
    (fake_pkg / "fme" / "core").mkdir(parents=True)
    (fake_pkg / "fme" / "ace" / "inference" / "data_writer").mkdir(parents=True)
    (fake_pkg / "fme" / "__init__.py").write_text('__version__ = "2026.5.1"\n', encoding="utf-8")
    (fake_pkg / "fme" / "core" / "__init__.py").write_text(
        "def set_seed(seed: int) -> None:\n    pass\n", encoding="utf-8"
    )
    # Mirror `fme.core.training_history` (`check_output` without `stderr=DEVNULL`).
    (fake_pkg / "fme" / "core" / "training_history.py").write_text(
        "import subprocess\n"
        "def git_revision_short_hash():\n"
        "    try:\n"
        "        return subprocess.check_output(['git', 'rev-parse', '--short', 'HEAD']).decode('ascii').strip()\n"
        "    except subprocess.CalledProcessError:\n"
        "        return None\n"
        "def get_job_env():\n"
        "    return {'git_sha': git_revision_short_hash()}\n",
        encoding="utf-8",
    )
    (fake_pkg / "fme" / "ace" / "__init__.py").write_text("", encoding="utf-8")
    (fake_pkg / "fme" / "ace" / "inference" / "__init__.py").write_text("", encoding="utf-8")
    (fake_pkg / "fme" / "ace" / "inference" / "data_writer" / "__init__.py").write_text("", encoding="utf-8")
    (fake_pkg / "fme" / "ace" / "inference" / "data_writer" / "dataset_metadata.py").write_text(
        "from fme.core.training_history import get_job_env\n", encoding="utf-8"
    )
    (fake_pkg / "fme" / "ace" / "inference" / "__main__.py").write_text(
        "import json, os, pathlib, sys, yaml\n"
        "from fme.ace.inference.data_writer.dataset_metadata import get_job_env\n"
        "cfg = yaml.safe_load(pathlib.Path(sys.argv[1]).read_text())\n"
        "exp = pathlib.Path(cfg['experiment_dir'])\n"
        "(exp / 'env_record.json').write_text(json.dumps({\n"
        "    'cwd': os.getcwd(),\n"
        "    'ceiling': os.environ.get('GIT_CEILING_DIRECTORIES'),\n"
        "    'job_env': get_job_env(),\n"
        "}))\n",
        encoding="utf-8",
    )

    for mod_key in [k for k in sys.modules if k == "fme" or k.startswith("fme.")]:
        monkeypatch.delitem(sys.modules, mod_key, raising=False)
    monkeypatch.syspath_prepend(str(fake_pkg))

    # Capture OS-level fd 2 as well via subprocess to verify zero `fatal: not a git repository` bytes!
    import aimip

    src_root = pathlib.Path(aimip.__file__).resolve().parent.parent
    env = dict(os.environ)
    env["PYTHONPATH"] = f"{fake_pkg}:{src_root}"
    child_cmd = (
        "import aimip.validators as v\n"
        f"v.ALLOWED_RUN_ROOTS = ({str(playground_env['scratch_sim']) + '/'!r},)\n"
        "from aimip.cli import main\n"
        f"raise SystemExit(main(['exec-fme', {str(out_dir)!r}]))\n"
    )
    proc = subprocess.run(
        [sys.executable, "-c", child_cmd],
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, f"stderr:\n{proc.stderr}\nstdout:\n{proc.stdout}"
    assert "fatal: not a git repository" not in proc.stderr

    recorded_env = json.loads((out_dir / "env_record.json").read_text(encoding="utf-8"))
    assert os.path.realpath(recorded_env["cwd"]) == os.path.realpath(str(out_dir))
    assert recorded_env["ceiling"] is not None
    assert recorded_env["job_env"] == {"git_sha": None}


def test_stochastic_ic_continuation_warns_when_seed_omitted_and_records_manifest(
    playground_env: dict[str, Any],
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Warn on stderr and record `seed_warning` when `--seed` is omitted on stochastic `--ic`."""
    runs_dir: pathlib.Path = playground_env["runs_dir"]
    restart_ic = write_fake_ic_netcdf(
        runs_dir / "seg1_restart.nc",
        ["2014-01-01T00:00:00"],
        sample_dim="sample",
        spatial_dims=("lat", "lon"),
        seed=19,
    )

    # 1. Omitted `--seed` on stochastic `--ic` preserves `seed=0`, warns on stderr, and records `seed_warning`.
    out_warn = runs_dir / "stoch_cont_no_seed"
    assert (
        main(
            [
                "run",
                "-m",
                "ace2s-shield-plus",
                "--start",
                "2014-01-01T00:00:00",
                "--steps",
                "8",
                "--ic",
                str(restart_ic),
                "--out",
                str(out_warn),
                "--dry-run",
            ]
        )
        == 0
    )
    captured = capsys.readouterr()
    assert "aimip warning:" in captured.err
    assert "seed=0" in captured.err
    manifest_warn = json.loads((out_warn / "aimip-manifest.json").read_text(encoding="utf-8"))
    assert manifest_warn["seed"] == 0
    assert manifest_warn["seed_explicit"] is False
    assert isinstance(manifest_warn["seed_warning"], str) and "--seed" in manifest_warn["seed_warning"]

    # 2. Explicit `--seed 43` suppresses warning and sets `seed_warning: None`.
    out_explicit = runs_dir / "stoch_cont_explicit_seed"
    assert (
        main(
            [
                "run",
                "-m",
                "ace2s-shield-plus",
                "--start",
                "2014-01-01T00:00:00",
                "--steps",
                "8",
                "--ic",
                str(restart_ic),
                "--seed",
                "43",
                "--out",
                str(out_explicit),
                "--dry-run",
            ]
        )
        == 0
    )
    captured_exp = capsys.readouterr()
    assert "aimip warning:" not in captured_exp.err
    manifest_exp = json.loads((out_explicit / "aimip-manifest.json").read_text(encoding="utf-8"))
    assert manifest_exp["seed"] == 43
    assert manifest_exp["seed_explicit"] is True
    assert manifest_exp["seed_warning"] is None


def test_summary_supports_eamv3_trefht_and_ts_variables(
    playground_env: dict[str, Any],
    capsys: pytest.CaptureFixture[str],
) -> None:
    """`aimip summary` supports EAMv3 `TREFHT` and `TS` temperature variables."""
    runs_dir: pathlib.Path = playground_env["runs_dir"]
    eam_out = runs_dir / "eamv3_summary_run"
    eam_out.mkdir(parents=True)

    lats = np.array([-60.0, 0.0, 0.0, 60.0], dtype=np.float64)
    lons = np.linspace(0.0, 315.0, 8, dtype=np.float64)
    trefht = np.full((2, 1, 4, 8), 287.25, dtype=np.float32)
    ts = np.full((2, 1, 4, 8), 289.75, dtype=np.float32)
    vt = np.array(
        [
            [cftime.DatetimeNoLeap(1994, 1, 15)],
            [cftime.DatetimeNoLeap(1994, 1, 15)],
        ],
        dtype=object,
    )
    ds = xr.Dataset(
        data_vars={
            "TREFHT": xr.DataArray(
                trefht,
                dims=("sample", "time", "lat", "lon"),
                attrs={"units": "K", "long_name": "Reference height temperature"},
            ),
            "TS": xr.DataArray(
                ts,
                dims=("sample", "time", "lat", "lon"),
                attrs={"units": "K", "long_name": "Surface temperature (radiative)"},
            ),
            "counts": xr.DataArray(
                np.array([[124], [124]], dtype=np.int32),
                dims=("sample", "time"),
            ),
        },
        coords={
            "time": xr.DataArray(np.array([0], dtype=np.int64), dims=("time",)),
            "lat": xr.DataArray(lats, dims=("lat",)),
            "lon": xr.DataArray(lons, dims=("lon",)),
            "valid_time": xr.DataArray(vt, dims=("sample", "time")),
        },
    )
    ds.to_netcdf(
        str(eam_out / "monthly_mean_predictions.nc"),
        engine="netcdf4",
        encoding={
            "valid_time": {
                "units": "days since 1994-01-01 00:00:00",
                "calendar": "noleap",
                "dtype": "int32",
            }
        },
    )

    assert main(["summary", str(eam_out), "--json"]) == 0
    summary = json.loads(capsys.readouterr().out)
    assert list(summary["variables"].keys()) == ["TREFHT", "TS"]
    assert math.isclose(summary["variables"]["TREFHT"]["series"][0]["ensemble_mean"], 287.25, rel_tol=1e-6)
    assert math.isclose(summary["variables"]["TS"]["series"][0]["ensemble_mean"], 289.75, rel_tol=1e-6)


def test_exec_fme_rehashes_checkpoint_and_ic_and_verifies_config_against_manifest(
    playground_env: dict[str, Any],
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """`exec-fme` verifies `config.yaml` paths and re-hashes checkpoint + IC before running `fme`."""
    runs_dir: pathlib.Path = playground_env["runs_dir"]
    custom_ic = write_fake_ic_netcdf(
        runs_dir / "tamper_test_ic.nc",
        ["1979-01-01T00:00:00"],
        spatial_dims=("latitude", "longitude"),
        seed=21,
    )
    out_dir = runs_dir / "tamper_check_run"
    assert (
        main(
            [
                "run",
                "-m",
                "ace2-era5",
                "--start",
                "1979-01-01T00:00:00",
                "--steps",
                "4",
                "--ic",
                str(custom_ic),
                "--out",
                str(out_dir),
                "--dry-run",
            ]
        )
        == 0
    )
    capsys.readouterr()

    # Fake `fme` that writes a marker if ever invoked (it MUST NOT be invoked when verification fails!)
    sentinel = tmp_path / "fme_ran_sentinel"
    fake_pkg = tmp_path / "guard_fme_pkg"
    (fake_pkg / "fme" / "ace" / "inference").mkdir(parents=True)
    (fake_pkg / "fme" / "__init__.py").write_text('__version__ = "2026.5.1"\n', encoding="utf-8")
    (fake_pkg / "fme" / "core.py").write_text("def set_seed(s): pass\n", encoding="utf-8")
    (fake_pkg / "fme" / "ace" / "__init__.py").write_text("", encoding="utf-8")
    (fake_pkg / "fme" / "ace" / "inference" / "__init__.py").write_text("", encoding="utf-8")
    (fake_pkg / "fme" / "ace" / "inference" / "__main__.py").write_text(
        f"import pathlib\npathlib.Path({str(sentinel)!r}).write_text('ran')\n",
        encoding="utf-8",
    )
    for mod_key in [k for k in sys.modules if k == "fme" or k.startswith("fme.")]:
        monkeypatch.delitem(sys.modules, mod_key, raising=False)
    monkeypatch.syspath_prepend(str(fake_pkg))

    # 1. Tamper `config.yaml` `checkpoint_path` -> refused before `fme` runs!
    orig_cfg_bytes = (out_dir / "config.yaml").read_bytes()
    tampered_cfg = yaml.safe_load(orig_cfg_bytes.decode("utf-8"))
    tampered_cfg["checkpoint_path"] = str(runs_dir / "swapped_ckpt.tar")
    (out_dir / "config.yaml").write_text(yaml.safe_dump(tampered_cfg), encoding="utf-8")
    assert main(["exec-fme", str(out_dir)]) == 2
    assert "does not match aimip-manifest.json" in capsys.readouterr().err
    assert not sentinel.exists()
    (out_dir / "config.yaml").write_bytes(orig_cfg_bytes)

    # 2. Tamper IC bytes on disk after `prepare_run` -> refused on SHA-256 mismatch!
    orig_ic_bytes = custom_ic.read_bytes()
    custom_ic.write_bytes(orig_ic_bytes + b"TAMPER")
    assert main(["exec-fme", str(out_dir)]) == 2
    assert "Initial-condition SHA-256 mismatch" in capsys.readouterr().err
    assert not sentinel.exists()
    custom_ic.write_bytes(orig_ic_bytes)

    # 3. Tamper checkpoint bytes on disk after `prepare_run` -> refused on SHA-256 mismatch!
    era5_ckpt: pathlib.Path = playground_env["era5_ckpt"]
    orig_ckpt_bytes = era5_ckpt.read_bytes()
    era5_ckpt.write_bytes(b"TAMPERED_CHECKPOINT_BYTES")
    try:
        assert main(["exec-fme", str(out_dir)]) == 2
        assert "Checkpoint SHA-256 mismatch" in capsys.readouterr().err
        assert not sentinel.exists()
    finally:
        era5_ckpt.write_bytes(orig_ckpt_bytes)

    # 4. Restored intact files -> `exec-fme` succeeds!
    assert main(["exec-fme", str(out_dir)]) == 0
    assert sentinel.exists()


def test_manifest_version_lock_sha_walltime_capped_account_none_and_help(
    playground_env: dict[str, Any],
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Verify manifest version/lock sha, `walltime_capped` warning, `--account none`, and CLI help."""
    from aimip.cli import build_parser

    runs_dir: pathlib.Path = playground_env["runs_dir"]

    # 1. CLI help text checks
    top_help = build_parser().format_help()
    assert "submit <run-dir>" in top_help
    assert "chat" in top_help

    # 2. Long multi-member run hits `max_time` (`12:00:00`) -> `walltime_capped: True` + stderr warning;
    # `--account none` and `--gres none` omit `#SBATCH --account` and `#SBATCH --gres`.
    capped_dir = runs_dir / "capped_walltime_run"
    assert (
        main(
            [
                "run",
                "-m",
                "ace2s-shield-plus",
                "--start",
                "1979-01-01T00:00:00",
                "--steps",
                "51136",
                "--members",
                "5",
                "--seed",
                "1",
                "--account",
                "none",
                "--gres",
                "none",
                "--out",
                str(capped_dir),
                "--dry-run",
            ]
        )
        == 0
    )
    cap = capsys.readouterr()
    assert "exceeds site max_time (12:00:00) and was clamped to 12:00:00" in cap.err

    manifest = json.loads((capped_dir / "aimip-manifest.json").read_text(encoding="utf-8"))
    assert manifest["aimip_version"] == "1.3.0"
    assert isinstance(manifest["aimip_lock_sha256"], str) and len(manifest["aimip_lock_sha256"]) == 64
    assert isinstance(manifest["ic_provenance"], str) and len(manifest["ic_provenance"]) > 0
    assert manifest["slurm"]["walltime_capped"] is True
    assert manifest["slurm"]["time"] == "12:00:00"

    sbatch_lines = (capped_dir / "job.sbatch").read_text(encoding="utf-8").splitlines()
    assert not any(ln.startswith("#SBATCH --account") for ln in sbatch_lines)
    assert not any(ln.startswith("#SBATCH --gres") for ln in sbatch_lines)


def test_exec_fme_private_aimip_tmp_dir_cleanup_restore_symlink_refusal_and_leftover_tolerance(
    playground_env: dict[str, Any],
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """`exec_fme` pins `multiprocessing.util.get_temp_dir()` before redirecting `TMPDIR`/`tempfile.gettempdir()` to `<run>/.aimip-tmp` (0700), cleans up and restores state even on error, refuses symlinked `.aimip-tmp`, and `status`/`collect` ignore leftover `.aimip-tmp`."""
    import multiprocessing
    import multiprocessing.util
    import stat
    import tempfile

    from aimip.errors import ConfigError
    from aimip.runner import exec_fme
    from aimip.status import get_run_status

    runs_dir: pathlib.Path = playground_env["runs_dir"]
    mp_cfg = multiprocessing.current_process()._config
    saved_mp_tempdir = mp_cfg.get("tempdir")

    try:
        # Helper to install an offline fake `fme` package in `sys.path`.
        fake_pkg = tmp_path / "fake_fme_tmpdir"
        (fake_pkg / "fme" / "core").mkdir(parents=True)
        (fake_pkg / "fme" / "ace" / "inference").mkdir(parents=True)
        (fake_pkg / "fme" / "__init__.py").write_text('__version__ = "2026.5.1"\n', encoding="utf-8")
        (fake_pkg / "fme" / "core" / "__init__.py").write_text(
            "def set_seed(s):\n    pass\n", encoding="utf-8"
        )
        (fake_pkg / "fme" / "ace" / "__init__.py").write_text("", encoding="utf-8")

        record_file = tmp_path / "tmpdir_record.json"
        main_py = fake_pkg / "fme" / "ace" / "inference" / "__main__.py"
        main_py.write_text(
            "import json, multiprocessing.util, os, pathlib, stat, tempfile\n"
            "td = tempfile.gettempdir()\n"
            "env_td = os.environ.get('TMPDIR')\n"
            "mp_td = multiprocessing.util.get_temp_dir()\n"
            "mode = stat.S_IMODE(os.lstat(td).st_mode)\n"
            "with tempfile.TemporaryDirectory() as sub:\n"
            "    sample_nc = pathlib.Path(sub) / 'ocean_ic.nc'\n"
            "    sample_nc.write_bytes(b'CDF\\x02')\n"
            "    inside = pathlib.Path(sub).parent == pathlib.Path(td)\n"
            "(pathlib.Path(td) / 'leftover_during_run.tmp').write_bytes(b'scratch')\n"
            f"pathlib.Path({str(record_file)!r}).write_text(json.dumps({{\n"
            "    'gettempdir': td,\n"
            "    'env_tmpdir': env_td,\n"
            "    'mp_tempdir': mp_td,\n"
            "    'mode': mode,\n"
            "    'inside': inside,\n"
            "}))\n"
            "if os.environ.get('FAKE_FME_RAISE') == '1':\n"
            "    raise RuntimeError('simulated fme failure mid-write')\n",
            encoding="utf-8",
        )

        for mod_key in [k for k in list(sys.modules) if k == "fme" or k.startswith("fme.")]:
            monkeypatch.delitem(sys.modules, mod_key, raising=False)
        monkeypatch.syspath_prepend(str(fake_pkg))

        # 1. Normal execution with pre-existing `TMPDIR`, pre-warmed `tempfile.tempdir` cache,
        # and freshly cleared `multiprocessing` tempdir cache.
        orig_tmp = tmp_path / "caller_orig_tmp"
        orig_tmp.mkdir()
        monkeypatch.setenv("TMPDIR", str(orig_tmp))
        monkeypatch.setattr(tempfile, "tempdir", None)
        assert tempfile.gettempdir() == str(orig_tmp)  # prime `tempfile.tempdir` cache before `exec_fme`
        mp_cfg.pop("tempdir", None)

        run_ok = runs_dir / ("tmpdir_ok_run_" + "x" * 60)
        assert (
            main(
                [
                    "run",
                    "-m",
                    "ace2-era5",
                    "--start",
                    "1979-01-01T00:00:00",
                    "--steps",
                    "4",
                    "--out",
                    str(run_ok),
                    "--dry-run",
                ]
            )
            == 0
        )
        capsys.readouterr()

        res_ok = exec_fme(str(run_ok))
        assert res_ok["rc"] == 0
        expected_run_tmp = str(run_ok / ".aimip-tmp")
        recorded = json.loads(record_file.read_text(encoding="utf-8"))
        assert recorded["gettempdir"] == expected_run_tmp
        assert recorded["env_tmpdir"] == expected_run_tmp
        assert not recorded["mp_tempdir"].startswith(expected_run_tmp)
        assert not recorded["mp_tempdir"].startswith(str(run_ok))
        assert recorded["mp_tempdir"].startswith(str(orig_tmp))
        assert recorded["mode"] == 0o700
        assert recorded["inside"] is True
        assert not (run_ok / ".aimip-tmp").exists()
        assert not (run_ok / ".aimip-tmp").is_symlink()
        assert os.environ.get("TMPDIR") == str(orig_tmp)
        assert tempfile.gettempdir() == str(orig_tmp)

        # 2. Exception inside `fme` (with `TMPDIR` originally unset, `tempfile.tempdir = None`,
        # and `multiprocessing` tempdir cache reset): `mp_tempdir` stays in short system `/tmp`
        # (well within AF_UNIX 104/107-char socket path limits even on a >110-char `<run>` path),
        # `<run>/.aimip-tmp` is removed, and both `TMPDIR` and `tempfile.tempdir` are restored.
        monkeypatch.delenv("TMPDIR", raising=False)
        monkeypatch.setattr(tempfile, "tempdir", None)
        mp_cfg.pop("tempdir", None)
        monkeypatch.setenv("FAKE_FME_RAISE", "1")
        record_file.unlink()

        run_fail = runs_dir / ("tmpdir_fail_long_run_" + "y" * 70)
        assert len(str(run_fail)) > 110
        assert (
            main(
                [
                    "run",
                    "-m",
                    "ace2-era5",
                    "--start",
                    "1979-01-01T00:00:00",
                    "--steps",
                    "4",
                    "--out",
                    str(run_fail),
                    "--dry-run",
                ]
            )
            == 0
        )
        capsys.readouterr()

        with pytest.raises(RuntimeError, match="simulated fme failure mid-write"):
            exec_fme(str(run_fail))
        fail_manifest = json.loads((run_fail / "aimip-manifest.json").read_text(encoding="utf-8"))
        assert fail_manifest["rc"] == 1
        assert "RuntimeError: simulated fme failure mid-write" in fail_manifest["error"]
        fail_recorded = json.loads(record_file.read_text(encoding="utf-8"))
        assert fail_recorded["gettempdir"] == str(run_fail / ".aimip-tmp")
        assert fail_recorded["env_tmpdir"] == str(run_fail / ".aimip-tmp")
        assert not fail_recorded["mp_tempdir"].startswith(str(run_fail / ".aimip-tmp"))
        assert not fail_recorded["mp_tempdir"].startswith(str(run_fail))
        assert len(fail_recorded["mp_tempdir"]) + len("/listener-12345678") <= 104
        assert not (run_fail / ".aimip-tmp").exists()
        assert "TMPDIR" not in os.environ
        assert tempfile.tempdir != str(run_fail / ".aimip-tmp")
        assert not tempfile.gettempdir().startswith(str(run_fail))
        monkeypatch.delenv("FAKE_FME_RAISE", raising=False)

        # 3. Pre-planted symlink `.aimip-tmp` (both live target and dangling) is refused before `fme` runs,
        # and the symlink target directory is NOT deleted.
        record_file.unlink()
        victim_dir = tmp_path / "victim_keep"
        victim_dir.mkdir()
        victim_file = victim_dir / "precious.nc"
        victim_file.write_bytes(b"DO_NOT_DELETE")

        run_sym = runs_dir / "tmpdir_symlink_run"
        assert (
            main(
                [
                    "run",
                    "-m",
                    "ace2-era5",
                    "--start",
                    "1979-01-01T00:00:00",
                    "--steps",
                    "4",
                    "--out",
                    str(run_sym),
                    "--dry-run",
                ]
            )
            == 0
        )
        capsys.readouterr()

        (run_sym / ".aimip-tmp").symlink_to(victim_dir, target_is_directory=True)
        with pytest.raises(ConfigError, match="Refusing a symlinked temporary directory"):
            exec_fme(str(run_sym))
        assert not record_file.exists()
        assert victim_file.read_bytes() == b"DO_NOT_DELETE"

        (run_sym / ".aimip-tmp").unlink()
        (run_sym / ".aimip-tmp").symlink_to(tmp_path / "dangling_target")
        assert main(["exec-fme", str(run_sym)]) == 2
        assert "Refusing a symlinked temporary directory" in capsys.readouterr().err
        assert not record_file.exists()

        # 4. Leftover `.aimip-tmp` from a SIGKILLed run is ignored by `status` and `collect`.
        leftover_run_tmp = run_ok / ".aimip-tmp"
        leftover_run_tmp.mkdir(mode=0o700)
        (leftover_run_tmp / "partial.nc").write_bytes(b"half-written")
        status_ok = get_run_status(str(run_ok))
        assert status_ok["state"] == "completed"
        assert ".aimip-tmp" not in status_ok["artifacts"]

        spin_dir = runs_dir / "tmpdir_spin"
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
                    str(spin_dir),
                    "--dry-run",
                ]
            )
            == 0
        )
        capsys.readouterr()
        (spin_dir / ".aimip-tmp").mkdir(mode=0o700)
        (spin_dir / ".aimip-tmp" / "stale.tmp").write_bytes(b"x")
        for idx in (0, 1):
            write_fake_ic_netcdf(
                spin_dir / str(idx) / "restart.nc",
                ["1979-01-03T00:00:00"],
                sample_dim="sample",
            )
            (spin_dir / str(idx) / ".aimip-tmp").mkdir(mode=0o700)
            (spin_dir / str(idx) / ".aimip-tmp" / "stale_member.tmp").write_bytes(b"y")

        spin_status = get_run_status(str(spin_dir))
        assert spin_status["state"] == "completed"
        assert spin_status["spin_members_completed"] == 2

        collected_ic = runs_dir / "tmpdir_collected_ic.nc"
        assert main(["collect", str(spin_dir), "--out", str(collected_ic)]) == 0
        capsys.readouterr()
        assert collected_ic.is_file()
    finally:
        if saved_mp_tempdir is None:
            mp_cfg.pop("tempdir", None)
        else:
            mp_cfg["tempdir"] = saved_mp_tempdir


def test_pin_cudnn_benchmark_off_holds_against_fme_and_restores(monkeypatch):
    """fme's coupled inference sets cudnn.benchmark=True; aimip keeps it off while fme runs, then restores."""
    torch = pytest.importorskip("torch")
    from aimip.runner import pin_cudnn_benchmark_off

    monkeypatch.delenv("AIMIP_CUDNN_BENCHMARK", raising=False)
    cls = type(torch.backends.cudnn)
    before = cls.__dict__.get("benchmark")
    restore = pin_cudnn_benchmark_off()
    try:
        torch.backends.cudnn.benchmark = True  # what fme.coupled.inference does
        assert torch.backends.cudnn.benchmark is False
    finally:
        restore()
    assert cls.__dict__.get("benchmark") is before
    torch.backends.cudnn.benchmark = True
    assert torch.backends.cudnn.benchmark is True
    torch.backends.cudnn.benchmark = False

    monkeypatch.setenv("AIMIP_CUDNN_BENCHMARK", "1")
    restore = pin_cudnn_benchmark_off()
    try:
        torch.backends.cudnn.benchmark = True
        assert torch.backends.cudnn.benchmark is True
    finally:
        restore()
        torch.backends.cudnn.benchmark = False


def test_pin_cudnn_benchmark_restores_a_prior_true_value_even_after_an_exception(monkeypatch):
    torch = pytest.importorskip("torch")
    from aimip.runner import pin_cudnn_benchmark_off

    monkeypatch.delenv("AIMIP_CUDNN_BENCHMARK", raising=False)
    torch.backends.cudnn.benchmark = True
    try:
        restore = pin_cudnn_benchmark_off()
        with pytest.raises(RuntimeError):
            try:
                torch.backends.cudnn.benchmark = True
                assert torch.backends.cudnn.benchmark is False
                raise RuntimeError("fme failed")
            finally:
                restore()
        assert torch.backends.cudnn.benchmark is True
    finally:
        torch.backends.cudnn.benchmark = False
