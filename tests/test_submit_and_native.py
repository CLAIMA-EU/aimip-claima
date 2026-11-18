"""Tests for Python `aimip submit`, shared shell/Python gate parity, and native installer/launcher."""

from __future__ import annotations

import hashlib
import json
import os
import pathlib
import shutil
import stat
import subprocess
import sys
import tempfile
import time
from typing import Any

import pytest

from aimip.cli import main
from aimip.errors import ConfigError, ValidationError
from aimip.site_config import SiteConfig, init_site_config, load_site_config

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
GATE_CASES_PATH = REPO_ROOT / "tests" / "wrapper" / "gate_cases.json"
WRAPPER_TEMPLATE_PATH = REPO_ROOT / "container" / "aimip-wrapper.in"
INSTALL_NATIVE_PATH = REPO_ROOT / "install" / "install-native.sh"
INSTALL_CONTAINER_PATH = REPO_ROOT / "install" / "install-container.sh"


def _load_gate_cases() -> list[dict[str, Any]]:
    return json.loads(GATE_CASES_PATH.read_text(encoding="utf-8"))


GATE_CASES = _load_gate_cases()


@pytest.fixture
def safe_gate_root() -> Any:
    """Create a 0700 workspace under a sticky/safe directory so ancestry checks depend only on test setup."""
    base = "/tmp" if os.path.isdir("/tmp") else None
    root = pathlib.Path(tempfile.mkdtemp(prefix="aimip-gate.", dir=base))
    os.chmod(root, 0o700)
    try:
        yield root
    finally:
        # Restore permissions on any 0000/0777 subdirs before cleanup.
        for dirpath, dirnames, _ in os.walk(root):
            for dname in dirnames:
                p = os.path.join(dirpath, dname)
                if not os.path.islink(p):
                    try:
                        os.chmod(p, 0o700)
                    except OSError:
                        pass
        shutil.rmtree(root, ignore_errors=True)


def _render_shell_wrapper(
    prefix: pathlib.Path,
    fakebin: pathlib.Path,
    runs_root: pathlib.Path,
    *,
    scheduler: str = "slurm",
) -> pathlib.Path:
    wrapper_out = prefix / "bin" / "aimip"
    wrapper_out.parent.mkdir(parents=True, exist_ok=True)
    sif_path = prefix / "img.sif"
    if not sif_path.exists():
        sif_path.write_bytes(b"FAKE_SIF_IMAGE_FOR_GATE_TESTS\n")
    (prefix / "image.sha256").write_text(
        f"{hashlib.sha256(sif_path.read_bytes()).hexdigest()}\n",
        encoding="utf-8",
    )
    tpl = WRAPPER_TEMPLATE_PATH.read_text(encoding="utf-8")
    rendered = (
        tpl.replace("@PREFIX@", str(prefix))
        .replace("@RUNTIME@", "apptainer")
        .replace("@RUNTIME_BIN@", str(fakebin / "fakert"))
        .replace("@IMAGE@", str(prefix / "img.sif"))
        .replace("@SCHEDULER@", scheduler)
        .replace("@RUN_ROOT@", str(runs_root))
        .replace("@EXTRA_RO@", "")
    )
    wrapper_out.write_text(rendered, encoding="utf-8")
    wrapper_out.chmod(0o755)
    return wrapper_out


def _prepare_gate_case_dir(
    case: dict[str, Any],
    base_dir: pathlib.Path,
    prefix: pathlib.Path,
    wrong_prefix: pathlib.Path,
) -> tuple[pathlib.Path, pathlib.Path]:
    """Materialize one gate case directory and return `(submit_target_dir, other_dir)`."""
    setup = case.get("setup")
    other_dir = base_dir / "other_run"
    other_dir.mkdir(parents=True, exist_ok=True)

    if setup == "parent_0777_no_sticky":
        open_parent = base_dir / "open_no_sticky"
        run_dir = open_parent / "sub" / "run"
        run_dir.mkdir(parents=True)
        os.chmod(open_parent, 0o777)
    elif setup == "parent_1777_sticky":
        sticky_parent = base_dir / "open_sticky"
        run_dir = sticky_parent / "sub" / "run"
        run_dir.mkdir(parents=True)
        os.chmod(sticky_parent, 0o1777)
    elif setup == "alias_into_0777_parent":
        shared = base_dir / "shared_0777"
        (shared / "sub" / "run").mkdir(parents=True)
        os.chmod(shared, 0o777)
        safe_runs = base_dir / "runs"
        safe_runs.mkdir(parents=True, exist_ok=True)
        alias = safe_runs / "alias"
        os.symlink(shared / "sub", alias)
        run_dir = alias / "run"
    else:
        run_dir = base_dir / "runs" / "case_run"
        run_dir.mkdir(parents=True, exist_ok=True)

    submit_target = run_dir

    if setup != "missing_job_sbatch":
        if "raw_text" in case:
            text = (
                str(case["raw_text"])
                .replace("{PREFIX}", str(prefix))
                .replace("{WRONG_PREFIX}", str(wrong_prefix))
                .replace("{RUN_DIR}", str(run_dir))
                .replace("{OTHER_DIR}", str(other_dir))
            )
        else:
            rendered_lines = [
                line.replace("{PREFIX}", str(prefix))
                .replace("{WRONG_PREFIX}", str(wrong_prefix))
                .replace("{RUN_DIR}", str(run_dir))
                .replace("{OTHER_DIR}", str(other_dir))
                for line in case["lines"]
            ]
            text = "\n".join(rendered_lines) + "\n"
        (run_dir / "job.sbatch").write_bytes(text.encode("utf-8"))
        os.chmod(run_dir / "job.sbatch", 0o644)

    if setup == "chmod_dir_gw":
        os.chmod(run_dir, 0o775)
    elif setup == "chmod_config_0666":
        cfg = run_dir / "config.yaml"
        cfg.write_text("x: 1\n", encoding="utf-8")
        os.chmod(cfg, 0o666)
    elif setup == "symlink_run_dir":
        link_dir = base_dir / "runs" / "case_run_link"
        os.symlink(run_dir, link_dir)
        submit_target = link_dir
    elif setup == "symlink_slurm_jobid":
        os.symlink(base_dir / "victim", run_dir / "slurm.jobid")
    elif setup == "symlink_log_subdir":
        os.symlink(other_dir, run_dir / "logs")
    elif setup == "symlink_log_file":
        os.symlink(other_dir / "victim.err", run_dir / "slurm-%j.err")
    elif setup == "symlink_percent_x_log":
        os.symlink(other_dir / "victim_x.out", run_dir / "escape.out")
    elif setup == "symlink_expanded_job_log":
        os.symlink(other_dir / "victim_123.out", run_dir / "slurm-123.out")
    elif setup == "nested_log_subdir":
        (run_dir / "sub").mkdir(parents=True, exist_ok=True)
    elif setup == "preexisting_slurm_jobid":
        (run_dir / "slurm.jobid").write_text("12345\n", encoding="utf-8")
    elif setup == "preexisting_local_pid":
        (run_dir / "local.pid").write_text("54321\n", encoding="utf-8")

    return submit_target, other_dir


@pytest.mark.parametrize("case", GATE_CASES, ids=[c["name"] for c in GATE_CASES])
def test_shared_submit_gate_parity_between_shell_wrapper_and_python_cli(
    case: dict[str, Any],
    safe_gate_root: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Every case in `tests/wrapper/gate_cases.json` must be identically accepted or refused by both gates."""
    fakebin = safe_gate_root / "fakebin"
    fakebin.mkdir()
    fake_sbatch = fakebin / "sbatch"
    fake_sbatch.write_text(
        "#!/bin/bash\n"
        'echo "$@" >> "$FAKE_SBATCH_LOG"\n'
        'cat "${@: -1}" > "$FAKE_SBATCH_SCRIPT"\n'
        "echo '98765;cluster'\n",
        encoding="utf-8",
    )
    fake_rt = fakebin / "fakert"
    fake_rt.write_text("#!/bin/bash\necho FAKERT\n", encoding="utf-8")
    fake_sbatch.chmod(0o755)
    fake_rt.chmod(0o755)

    # Prefix intentionally includes '.' to catch unescaped regex metacharacters.
    prefix = safe_gate_root / "pre.v1"
    wrong_prefix = safe_gate_root / "preXv1"

    shell_sub = safe_gate_root / "shell_case"
    shell_sub.mkdir()
    shell_wrapper = _render_shell_wrapper(prefix, fakebin, shell_sub / "runs")
    shell_target, _ = _prepare_gate_case_dir(case, shell_sub, prefix, wrong_prefix)

    env_shell = dict(os.environ)
    env_shell["PATH"] = f"{fakebin}:{env_shell.get('PATH', '')}"
    env_shell["FAKE_SBATCH_LOG"] = str(shell_sub / "sbatch.log")
    env_shell["FAKE_SBATCH_SCRIPT"] = str(shell_sub / "sbatch.script")
    env_shell["TMPDIR"] = str(safe_gate_root)

    shell_res = subprocess.run(
        ["/bin/bash", str(shell_wrapper), "submit", str(shell_target)],
        env=env_shell,
        capture_output=True,
        text=True,
        check=False,
    )
    shell_accepted = shell_res.returncode == 0

    py_sub = safe_gate_root / "py_case"
    py_sub.mkdir()
    py_target, _ = _prepare_gate_case_dir(case, py_sub, prefix, wrong_prefix)

    site_file = prefix / "site.json"
    init_site_config(
        preset="generic-slurm",
        out_path=str(site_file),
        partition="gpu",
        account="proj_gpu",
        overwrite=True,
    )
    monkeypatch.setenv("PATH", f"{fakebin}:{os.environ.get('PATH', '')}")
    monkeypatch.setenv("FAKE_SBATCH_LOG", str(py_sub / "sbatch.log"))
    monkeypatch.setenv("FAKE_SBATCH_SCRIPT", str(py_sub / "sbatch.script"))
    monkeypatch.setenv("AIMIP_HOME", str(prefix))
    monkeypatch.setenv("AIMIP_PLAYGROUND", str(prefix))
    monkeypatch.setenv("AIMIP_SITE_CONFIG", str(site_file))

    py_rc = main(["submit", str(py_target)])
    py_cap = capsys.readouterr()
    py_accepted = py_rc == 0

    want = bool(case["accept"])
    assert shell_accepted is want, (
        f"Shell wrapper mismatch on {case['name']}: expected accept={want}, "
        f"got rc={shell_res.returncode}, stderr={shell_res.stderr!r}"
    )
    assert py_accepted is want, (
        f"Python submit mismatch on {case['name']}: expected accept={want}, "
        f"got rc={py_rc}, out={py_cap.out!r}, err={py_cap.err!r}"
    )
    assert shell_accepted == py_accepted

    if want:
        assert (py_target / "slurm.jobid").read_text(encoding="utf-8").strip() == "98765"
        sbatch_arg = (py_sub / "sbatch.log").read_text(encoding="utf-8").strip()
        assert "aimip-submit." in sbatch_arg and sbatch_arg.endswith("/job.sbatch")
        assert str(py_target) not in sbatch_arg
        assert (py_sub / "sbatch.script").read_bytes() == (py_target / "job.sbatch").read_bytes()
    elif case.get("setup") == "symlink_slurm_jobid":
        assert not (py_sub / "victim").exists()
        assert not (shell_sub / "victim").exists()


def test_python_submit_local_scheduler_detached_background_and_exec_fme_gate(
    safe_gate_root: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """`aimip submit` under `scheduler='none'` launches detached bash, writes `local.pid` + `local-run.log`, and `exec-fme` enforces `trusted_dir`."""
    prefix = safe_gate_root / "native_prefix"
    (prefix / "bin").mkdir(parents=True)
    marker_file = safe_gate_root / "ran_exec_target.txt"
    fake_aimip = prefix / "bin" / "aimip"
    fake_aimip.write_text(
        "#!/bin/bash\n"
        f'printf "%s:%s\\n" "$1" "$2" > "{marker_file}"\n'
        'echo "LOCAL_EXEC_OK $2"\n',
        encoding="utf-8",
    )
    fake_aimip.chmod(0o755)

    site_file = prefix / "site.json"
    init_site_config(
        preset="local",
        out_path=str(site_file),
        runtime="native",
        overwrite=True,
    )
    loaded = load_site_config(str(site_file))
    assert loaded.runtime == "native" and loaded.scheduler == "none"

    run_dir = safe_gate_root / "runs" / "local_job"
    run_dir.mkdir(parents=True)
    (run_dir / "job.sbatch").write_text(
        "#!/bin/bash\n"
        "# aimip-generated v1\n"
        "\n"
        "set -euo pipefail\n"
        f'"{prefix}/bin/aimip" exec-fme "{run_dir}"\n',
        encoding="utf-8",
    )

    monkeypatch.setenv("AIMIP_HOME", str(prefix))
    monkeypatch.setenv("AIMIP_PLAYGROUND", str(prefix))
    monkeypatch.setenv("AIMIP_SITE_CONFIG", str(site_file))
    monkeypatch.setenv("AIMIP_SCHEDULER", "none")

    assert main(["submit", str(run_dir), "--json"]) == 0
    from conftest import parse_cli_json

    payload = parse_cli_json(capsys.readouterr().out)
    assert payload["scheduler"] == "none"
    assert isinstance(payload["local_pid"], int) and payload["local_pid"] > 0
    assert (run_dir / "local.pid").read_text(encoding="utf-8").strip() == str(payload["local_pid"])

    # Wait briefly for detached background script to finish writing local-run.log and marker_file.
    deadline = time.monotonic() + 5.0
    while time.monotonic() < deadline:
        if marker_file.is_file() and "LOCAL_EXEC_OK" in (run_dir / "local-run.log").read_text(encoding="utf-8"):
            break
        time.sleep(0.05)

    assert marker_file.read_text(encoding="utf-8").strip() == f"exec-fme:{run_dir}"
    assert f"LOCAL_EXEC_OK {run_dir}" in (run_dir / "local-run.log").read_text(encoding="utf-8")

    # Second submit is refused because local.pid exists.
    assert main(["submit", str(run_dir)]) == 2
    assert "already submitted" in capsys.readouterr().err

    # CLI `aimip exec-fme` refuses a group-writable run directory at job time, matching container wrapper.
    gw_dir = safe_gate_root / "runs" / "gw_exec"
    gw_dir.mkdir(parents=True)
    os.chmod(gw_dir, 0o775)
    assert main(["exec-fme", str(gw_dir)]) == 2
    assert "group- or world-writable" in capsys.readouterr().err


def test_native_status_refreshes_sacct_txt_and_preserves_last_good_record(
    safe_gate_root: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """In native mode, `aimip status` queries live `sacct`, refreshes `sacct.txt`, and falls back to `sacct.txt` if `sacct` later returns empty."""
    run_dir = safe_gate_root / "runs" / "slurm_native_status"
    run_dir.mkdir(parents=True)
    (run_dir / "aimip-manifest.json").write_text(
        json.dumps(
            {
                "schema_version": 2,
                "mode": "submit",
                "out_dir": str(run_dir),
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
    (run_dir / "slurm.jobid").write_text("555111\n", encoding="utf-8")

    fakebin = safe_gate_root / "fakebin"
    fakebin.mkdir()
    sacct_bin = fakebin / "sacct"
    sacct_bin.write_text(
        "#!/bin/bash\n"
        'echo "555111|RUNNING|00:04:12|0:0"\n',
        encoding="utf-8",
    )
    sacct_bin.chmod(0o755)
    monkeypatch.setenv("PATH", f"{fakebin}:{os.environ.get('PATH', '')}")
    monkeypatch.setattr(
        "aimip.validators.ALLOWED_RUN_ROOTS",
        (f"{safe_gate_root}/",),
    )

    from conftest import parse_cli_json

    assert main(["status", str(run_dir), "--json"]) == 0
    st1 = parse_cli_json(capsys.readouterr().out)
    assert st1["state"] == "running"
    assert st1["sacct_source"] == "sacct"
    assert (run_dir / "sacct.txt").is_file()
    assert "555111|RUNNING|00:04:12|0:0" in (run_dir / "sacct.txt").read_text(encoding="utf-8")

    # Live sacct transitions to COMPLETED -> status sees COMPLETED (not stale sacct.txt) and updates sacct.txt.
    sacct_bin.write_text(
        "#!/bin/bash\n"
        'echo "555111|COMPLETED|00:10:00|0:0"\n',
        encoding="utf-8",
    )
    assert main(["status", str(run_dir), "--json"]) == 0
    st2 = parse_cli_json(capsys.readouterr().out)
    assert st2["state"] == "completed"
    assert st2["sacct_source"] == "sacct"
    assert "555111|COMPLETED|00:10:00|0:0" in (run_dir / "sacct.txt").read_text(encoding="utf-8")

    # When sacct returns empty output later, status preserves and falls back to the last good sacct.txt.
    sacct_bin.write_text("#!/bin/bash\nexit 0\n", encoding="utf-8")
    assert main(["status", str(run_dir), "--json"]) == 0
    st3 = parse_cli_json(capsys.readouterr().out)
    assert st3["state"] == "completed"
    assert st3["sacct_source"] == "sacct.txt"


def _make_py312_shim_if_needed(work_dir: pathlib.Path) -> str:
    """Return `sys.executable` when >= 3.12, or a stdlib-only shim reporting 3.12 for `--dry-run` installer tests."""
    if sys.version_info >= (3, 12):
        return sys.executable
    shim = work_dir / "py312_shim"
    shim.write_text(
        f"#!{sys.executable}\n"
        "import sys\n"
        "class _V(tuple):\n"
        "    major, minor, micro, releaselevel, serial = 3, 12, 0, 'final', 0\n"
        "sys.version_info = _V((3, 12, 0, 'final', 0))\n"
        "args = sys.argv[1:]\n"
        "if args and args[0] == '-c':\n"
        "    code = args[1]; sys.argv = ['-c', *args[2:]]; exec(compile(code, '<string>', 'exec'), {'__name__': '__main__'})\n"
        "elif args and args[0] == '-':\n"
        "    code = sys.stdin.read(); sys.argv = ['-', *args[1:]]; exec(compile(code, '<stdin>', 'exec'), {'__name__': '__main__'})\n"
        "elif args and args[0] == '-V':\n"
        "    print('Python 3.12.0')\n",
        encoding="utf-8",
    )
    shim.chmod(0o755)
    return str(shim)


def test_install_native_dry_run_launcher_env_and_emu_discovery(
    safe_gate_root: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`install/install-native.sh --dry-run` validates presets/Python, writes `site.json` (`runtime='native'`) and `bin/aimip`."""
    py312 = _make_py312_shim_if_needed(safe_gate_root)

    # 1. Missing --account on levante fails immediately before creating venvs.
    bad_prefix = safe_gate_root / "levante_missing_acc"
    bad_res = subprocess.run(
        [
            "/bin/bash",
            str(INSTALL_NATIVE_PATH),
            "--prefix",
            str(bad_prefix),
            "--site",
            "levante",
            "--python",
            py312,
            "--dry-run",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert bad_res.returncode != 0
    assert "account" in (bad_res.stderr + bad_res.stdout)

    # 2. Python < 3.12 is refused up front.
    old_py = safe_gate_root / "py310_fake"
    old_py.write_text("#!/bin/sh\nif [ \"$1\" = \"-V\" ]; then echo 'Python 3.10.9'; exit 0; fi\nexit 1\n", encoding="utf-8")
    old_py.chmod(0o755)
    old_res = subprocess.run(
        [
            "/bin/bash",
            str(INSTALL_NATIVE_PATH),
            "--prefix",
            str(safe_gate_root / "old_py_prefix"),
            "--site",
            "local",
            "--python",
            str(old_py),
            "--dry-run",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert old_res.returncode != 0
    assert "3.12" in old_res.stderr

    # 3. Valid --dry-run writes site.json ("runtime": "native"), VERSION, and bin/aimip launcher.
    prefix = safe_gate_root / "native_install"
    runs_root = safe_gate_root / "runs_out"
    ok_res = subprocess.run(
        [
            "/bin/bash",
            str(INSTALL_NATIVE_PATH),
            "--prefix",
            str(prefix),
            "--site",
            "levante",
            "--account",
            "proj_gpu",
            "--run-root",
            str(runs_root),
            "--python",
            py312,
            "--dry-run",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert ok_res.returncode == 0, ok_res.stderr
    site_doc = json.loads((prefix / "site.json").read_text(encoding="utf-8"))
    assert site_doc["runtime"] == "native"
    assert site_doc["scheduler"] == "slurm"
    assert site_doc["sbatch"]["account"] == "proj_gpu"
    assert SiteConfig.from_dict(site_doc).validate().runtime == "native"

    # Populate a fake venv python in prefix/venv/bin/python to verify every env var exported by bin/aimip.
    venv_bin = prefix / "venv" / "bin"
    venv_bin.mkdir(parents=True)
    env_dump = safe_gate_root / "launcher_env.json"
    fake_venv_py = venv_bin / "python"
    fake_venv_py.write_text(
        f"#!{sys.executable}\n"
        "import json, os, sys\n"
        "keys = [\n"
        "    'AIMIP_HOME', 'AIMIP_PLAYGROUND', 'AIMIP_DATA', 'AIMIP_SITE_CONFIG',\n"
        "    'AIMIP_DEFAULT_OUT_ROOT', 'AIMIP_SCHEDULER', 'AIMIP_FME_2026_10_OVERLAY',\n"
        "    'AIMIP_EMU_PYTHON', 'AIMIP_CAMULATOR_PYTHON', 'AIMIP_SDYFF_PYTHON',\n"
        "    'AIMIP_IMAGE_SHA', 'AIMIP_DATA_MODE', 'PYTHONNOUSERSITE',\n"
        "    'GEMINI_API_KEY',\n"
        "]\n"
        "doc = {k: os.environ.get(k) for k in keys}\n"
        "doc['__argv__'] = sys.argv[1:]\n"
        f"open({str(env_dump)!r}, 'w').write(json.dumps(doc))\n",
        encoding="utf-8",
    )
    fake_venv_py.chmod(0o755)

    launch_env = dict(os.environ)
    launch_env["GEMINI_API_KEY"] = "SECRET_SHOULD_NOT_LEAK_TO_MODELS"
    launch_env.pop("AIMIP_EMU_PYTHON", None)
    launch_env.pop("AIMIP_CAMULATOR_PYTHON", None)
    launch_env.pop("AIMIP_SDYFF_PYTHON", None)
    r1 = subprocess.run(
        [str(prefix / "bin" / "aimip"), "models"],
        env=launch_env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert r1.returncode == 0, r1.stderr
    exported1 = json.loads(env_dump.read_text(encoding="utf-8"))
    # isolated start: neither the current directory nor PYTHONPATH can shadow the installed package
    assert exported1["__argv__"][:4] == ["-I", "-B", "-m", "aimip"], exported1["__argv__"]
    assert exported1["AIMIP_HOME"] == str(prefix)
    assert exported1["AIMIP_PLAYGROUND"] == str(prefix)
    assert exported1["AIMIP_DATA"] == str(prefix / "data")
    assert exported1["AIMIP_SITE_CONFIG"] == str(prefix / "site.json")
    assert exported1["AIMIP_DEFAULT_OUT_ROOT"] == str(runs_root)
    assert exported1["AIMIP_SCHEDULER"] == "slurm"
    assert exported1["AIMIP_FME_2026_10_OVERLAY"] == str(prefix / "overlay-fme-2026.10.0")
    assert exported1["AIMIP_EMU_PYTHON"] is None
    assert exported1["AIMIP_CAMULATOR_PYTHON"] is None
    assert exported1["AIMIP_SDYFF_PYTHON"] is None
    assert exported1["AIMIP_IMAGE_SHA"] == "native"
    assert exported1["AIMIP_DATA_MODE"] == "ro"
    assert exported1["PYTHONNOUSERSITE"] == "1"
    assert exported1["GEMINI_API_KEY"] is None

    # Create venv-emu, venv-camulator, venv-sdyff -> launcher exports all three and `chat` forwards GEMINI_API_KEY.
    emu_py = prefix / "venv-emu" / "bin" / "python"
    cam_py = prefix / "venv-camulator" / "bin" / "python"
    sdyff_py = prefix / "venv-sdyff" / "bin" / "python"
    for py_p in (emu_py, cam_py, sdyff_py):
        py_p.parent.mkdir(parents=True)
        py_p.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
        py_p.chmod(0o755)

    r2 = subprocess.run(
        [str(prefix / "bin" / "aimip"), "chat"],
        env=launch_env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert r2.returncode == 0, r2.stderr
    exported2 = json.loads(env_dump.read_text(encoding="utf-8"))
    assert exported2["AIMIP_EMU_PYTHON"] == str(emu_py)
    assert exported2["AIMIP_CAMULATOR_PYTHON"] == str(cam_py)
    assert exported2["AIMIP_SDYFF_PYTHON"] == str(sdyff_py)
    assert exported2["GEMINI_API_KEY"] == "SECRET_SHOULD_NOT_LEAK_TO_MODELS"

    # Also check `emu_python()` discovers `$AIMIP_HOME/venv-emu/bin/python` automatically when AIMIP_EMU_PYTHON is unset.
    from aimip.emu import emu_python

    monkeypatch.delenv("AIMIP_EMU_PYTHON", raising=False)
    monkeypatch.setenv("AIMIP_HOME", str(prefix))
    assert emu_python() == str(emu_py)

    emu_py.unlink()
    # Inside the image the default `/opt/aimip/venv-emu` exists; point the default elsewhere for this check.
    monkeypatch.setattr("aimip.emu.DEFAULT_EMU_PYTHON", str(prefix / "no-default-venv-emu" / "bin" / "python"))
    with pytest.raises(ConfigError, match="install-native.sh --with-emu"):
        emu_python()


@pytest.mark.parametrize(
    "script,flag",
    [
        (INSTALL_NATIVE_PATH, "--prefix"),
        (INSTALL_NATIVE_PATH, "--site"),
        (INSTALL_NATIVE_PATH, "--partition"),
        (INSTALL_NATIVE_PATH, "--account"),
        (INSTALL_NATIVE_PATH, "--gres"),
        (INSTALL_NATIVE_PATH, "--run-root"),
        (INSTALL_NATIVE_PATH, "--python"),
        (INSTALL_CONTAINER_PATH, "--prefix"),
        (INSTALL_CONTAINER_PATH, "--site"),
        (INSTALL_CONTAINER_PATH, "--runtime"),
        (INSTALL_CONTAINER_PATH, "--partition"),
        (INSTALL_CONTAINER_PATH, "--account"),
        (INSTALL_CONTAINER_PATH, "--gres"),
        (INSTALL_CONTAINER_PATH, "--run-root"),
        (INSTALL_CONTAINER_PATH, "--image"),
    ],
)
def test_installers_reject_missing_option_values_without_hanging(
    script: pathlib.Path,
    flag: str,
) -> None:
    """Every value-taking flag on both installers must exit 2 immediately when given no value."""
    res = subprocess.run(
        ["/bin/bash", str(script), flag],
        capture_output=True,
        text=True,
        timeout=2.0,
        check=False,
    )
    assert res.returncode == 2
    assert "requires a value" in res.stderr


def test_installers_honour_gres_none_and_account_none_through_sbatch_rendering(
    safe_gate_root: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`--gres none` and `--account none` write explicit `\"\"` in `site.json`, and `render_sbatch_script` omits `--gres` and `--account`."""
    from aimip.runner import SlurmOptions, render_sbatch_script

    py312 = _make_py312_shim_if_needed(safe_gate_root)
    runs_root = safe_gate_root / "runs"
    runs_root.mkdir()

    # 1. install-native.sh --gres none --account none
    nat_prefix = safe_gate_root / "nat_gres_none"
    nat_res = subprocess.run(
        [
            "/bin/bash",
            str(INSTALL_NATIVE_PATH),
            "--prefix",
            str(nat_prefix),
            "--site",
            "generic-slurm",
            "--partition",
            "gpu_shared",
            "--account",
            "none",
            "--gres",
            "none",
            "--run-root",
            str(runs_root),
            "--python",
            py312,
            "--dry-run",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert nat_res.returncode == 0, nat_res.stderr
    nat_raw = json.loads((nat_prefix / "site.json").read_text(encoding="utf-8"))
    assert nat_raw["sbatch"]["gres"] == ""
    assert nat_raw["sbatch"]["account"] == ""
    monkeypatch.setenv("AIMIP_SITE_CONFIG", str(nat_prefix / "site.json"))
    nat_site = load_site_config(str(nat_prefix / "site.json"))
    assert nat_site.gres == ""
    assert nat_site.account == ""

    run_dir = runs_root / "gres_none_job"
    run_dir.mkdir(parents=True)
    sbatch = render_sbatch_script(
        model_name="ace2-era5",
        out_dir=str(run_dir),
        playground_dir=str(nat_prefix),
        slurm_opts=SlurmOptions.from_site(nat_site),
        default_walltime="00:20:00",
    )
    assert "#SBATCH --partition=gpu_shared" in sbatch
    assert "--gres" not in sbatch
    assert "--account" not in sbatch

    # 2. install-container.sh --gres none --account none (using a fake docker runtime so it runs offline)
    fakebin = safe_gate_root / "fakebin_cnt"
    fakebin.mkdir()
    version = (REPO_ROOT / "container" / "VERSION").read_text(encoding="utf-8").strip()
    fake_docker = fakebin / "docker"
    fake_docker.write_text(
        "#!/bin/bash\n"
        'case "$1" in\n'
        "  pull) exit 0;;\n"
        "  run) exit 0;;\n"
        '  image) echo "sha256:deadbeefcafebabe"; exit 0;;\n'
        "  *) exit 0;;\n"
        "esac\n",
        encoding="utf-8",
    )
    fake_docker.chmod(0o755)
    fake_uname = fakebin / "uname"
    fake_uname.write_text("#!/bin/sh\necho x86_64\n", encoding="utf-8")
    fake_uname.chmod(0o755)

    cnt_prefix = safe_gate_root / "cnt_gres_none"
    env_cnt = dict(os.environ)
    env_cnt["PATH"] = f"{fakebin}:{env_cnt.get('PATH', '')}"
    env_cnt["TMPDIR"] = str(safe_gate_root)
    cnt_res = subprocess.run(
        [
            "/bin/bash",
            str(INSTALL_CONTAINER_PATH),
            "--prefix",
            str(cnt_prefix),
            "--site",
            "generic-slurm",
            "--runtime",
            "docker",
            "--partition",
            "gpu_shared",
            "--account",
            "none",
            "--gres",
            "none",
            "--run-root",
            str(runs_root),
        ],
        env=env_cnt,
        capture_output=True,
        text=True,
        check=False,
    )
    assert cnt_res.returncode == 0, cnt_res.stderr + cnt_res.stdout
    monkeypatch.setenv("AIMIP_SITE_CONFIG", str(cnt_prefix / "site.json"))
    cnt_site = load_site_config(str(cnt_prefix / "site.json"))
    assert cnt_site.gres == ""
    assert cnt_site.account == ""
    sbatch_cnt = render_sbatch_script(
        model_name="ace2-era5",
        out_dir=str(run_dir),
        playground_dir=str(cnt_prefix),
        slurm_opts=SlurmOptions.from_site(cnt_site),
        default_walltime="00:20:00",
    )
    assert "--gres" not in sbatch_cnt
    assert "--account" not in sbatch_cnt


def _build_fake_miles_credit_archive(
    out_tar: pathlib.Path,
    *,
    include_deprecated_transforms: bool = True,
    postblock_available_override: bool | None = None,
) -> dict[str, Any]:
    """Create an offline `miles-credit` commit archive exercising PEP 420 namespace subpackages & relative symlinks."""
    import hashlib
    import io
    import tarfile
    from aimip.safe_pickle import compute_sha256sum_tree_digest

    top = "miles-credit-d0eb9add439f09bd25883f4393296f3266b5876d"
    files: dict[str, str] = {
        "applications/train.py": "def main_cli():\n    return 0\n",
        "credit/__init__.py": "",
        "credit/parser.py": "def credit_main_parser(*a, **k):\n    return {}\n",
        "credit/models/__init__.py": (
            "from credit.models.crossformer import CrossFormer\n"
            "from credit.models.camulator import Camulator\n"
            "from credit.models.wxformer.crossformer import CrossFormer as WXFormer\n"
            "model_types = {\n"
            "    'crossformer': (CrossFormer, 'CrossFormer'),\n"
            "    'camulator': (Camulator, 'CAMulator'),\n"
            "    'wxformer': (WXFormer, 'WXFormer'),\n"
            "}\n"
            "def load_model(*a, **k):\n    return Camulator()\n"
            "def load_model_name(*a, **k):\n    return 'checkpoint.pt'\n"
        ),
        "credit/models/crossformer.py": (
            "from credit.postblock import PostBlock\n"
            "class CrossFormer:\n    pass\n"
        ),
        "credit/models/camulator.py": (
            "from credit.postblock import PostBlock\n"
            "class Camulator:\n    pass\n"
        ),
        "credit/models/wxformer/__init__.py": "",
        "credit/models/wxformer/crossformer.py": (
            "from credit.postblock import PostBlock\n"
            "class CrossFormer:\n    pass\n"
        ),
        "credit/models/checkpoint.py": "def load_model_state(*a, **k):\n    return None\n",
        "credit/postblock/__init__.py": (
            "from credit.transforms import load_transforms\n"
            "class PostBlock:\n    pass\n"
            "class TracerFixer:\n    pass\n"
            "class GlobalMassFixer:\n    pass\n"
            "class GlobalWaterFixer:\n    pass\n"
            "class GlobalEnergyFixer:\n    pass\n"
            "class GlobalEnergyFixerUpDown:\n    pass\n"
        ),
        "credit/transforms/__init__.py": (
            "from credit.transforms.deprecated._transforms import ToTensor\n"
            "def load_transforms(*a, **k):\n    return ToTensor()\n"
            "class Normalize_ERA5_and_Forcing:\n    pass\n"
        ),
        # Deliberately NO `credit/preblock/__init__.py` or `credit/transforms/deprecated/__init__.py`
        # to mirror upstream's PEP 420 namespace directories omitted by `pyproject.toml`.
        "credit/preblock/scaler.py": "class Scaler:\n    pass\n",
        "credit/distributed.py": "def distributed_model_wrapper(m, *a, **k):\n    return m\n",
        "credit/output.py": (
            "def load_metadata(*a, **k):\n    return {}\n"
            "def make_xarray(*a, **k):\n    return None\n"
            "def save_netcdf_increment(*a, **k):\n    return None\n"
        ),
        "climate/Quick_Climate.py": "# Quick_Climate entrypoint\n",
        "climate/WindPP.py": "def post_process_wind_artifacts(*a, **k):\n    return None\n",
        "climate/Model_State.py": (
            "try:\n"
            "    from credit.models import load_model, load_model_name\n"
            "    from credit.transforms import load_transforms, Normalize_ERA5_and_Forcing\n"
            "    from credit.distributed import distributed_model_wrapper\n"
            "    from credit.models.checkpoint import load_model_state\n"
            "    from credit.parser import credit_main_parser\n"
            "    from credit.output import load_metadata\n"
            "    CREDIT_AVAILABLE = True\n"
            "except ImportError:\n"
            "    CREDIT_AVAILABLE = False\n"
            "try:\n"
            "    from credit.postblock import GlobalMassFixer, GlobalWaterFixer, GlobalEnergyFixer, GlobalEnergyFixerUpDown\n"
            "    POSTBLOCK_AVAILABLE = True\n"
            "except ImportError:\n"
            "    POSTBLOCK_AVAILABLE = False\n"
            + (
                f"POSTBLOCK_AVAILABLE = {postblock_available_override!r}\n"
                if postblock_available_override is not None
                else ""
            )
            + "try:\n"
            "    from WindPP import post_process_wind_artifacts\n"
            "    WINDPP_AVAILABLE = True\n"
            "except ImportError:\n"
            "    WINDPP_AVAILABLE = False\n"
            "def initialize_camulator(*a, **k):\n    return None\n"
            "class CAMulatorStepper:\n    pass\n"
        ),
    }
    if include_deprecated_transforms:
        files["credit/transforms/deprecated/_transforms.py"] = "class ToTensor:\n    pass\n"

    stage_tree = out_tar.parent / f"_stage_{out_tar.stem}"
    if stage_tree.exists():
        import shutil as _sh
        _sh.rmtree(stage_tree)
    for rel, txt in sorted(files.items()):
        fp = stage_tree / rel
        fp.parent.mkdir(parents=True, exist_ok=True)
        fp.write_text(txt, encoding="utf-8")
    tree_sha, tree_files = compute_sha256sum_tree_digest(str(stage_tree))
    import shutil as _sh
    _sh.rmtree(stage_tree)

    with tarfile.open(out_tar, "w:gz") as tf:
        for rel, txt in sorted(files.items()):
            raw = txt.encode("utf-8")
            ti = tarfile.TarInfo(f"{top}/{rel}")
            ti.size = len(raw)
            ti.mode = 0o644
            tf.addfile(ti, io.BytesIO(raw))
        sym = tarfile.TarInfo(f"{top}/credit/applications/train.py")
        sym.type = tarfile.SYMTYPE
        sym.linkname = "../../applications/train.py"
        tf.addfile(sym)

    ar_bytes = out_tar.read_bytes()
    return {
        "url": f"file://{out_tar}",
        "archive_size": str(len(ar_bytes)),
        "archive_sha256": hashlib.sha256(ar_bytes).hexdigest(),
        "tree_sha256": tree_sha,
        "tree_files": str(tree_files),
    }


def _make_stub_native_python(work_dir: pathlib.Path, real_version: str) -> pathlib.Path:
    """Build an offline stub Python >= 3.12 whose `-m venv` creates stub `python`/`pip` controlled by env vars."""
    import textwrap

    py_driver = work_dir / "venv_py_driver.py"
    py_driver.write_text(
        textwrap.dedent(
            """\
            import os, pathlib, site, stat, sys
            sp = sys.argv[1]
            a = sys.argv[2:]
            isolated = False
            while a and a[0] in ("-I", "-B"):
                if a[0] == "-I":
                    isolated = True
                elif a[0] == "-B":
                    sys.dont_write_bytecode = True
                a = a[1:]
            if isolated:
                base_paths = [p for p in sys.path if p and "site-packages" not in p and p != os.getcwd()]
                sys.path = []
                site.addsitedir(sp)
                pth_added = [p for p in sys.path if p != sp]
                sys.path = [sp, *pth_added, *base_paths]
            else:
                env_pp = [p for p in os.environ.get("PYTHONPATH", "").split(":") if p]
                base_paths = list(sys.path)
                sys.path = []
                site.addsitedir(sp)
                pth_added = [p for p in sys.path if p != sp]
                sys.path = [*env_pp, sp, *pth_added, *base_paths]
            if a and a[0] == "-c":
                code = a[1]
                if os.environ.get("FAKE_NATIVE_OMIT_APEX_SHIM") == "1" and "install_sdyff_apex_shim" in code:
                    sys.exit(0)
                if "install_camulator_source_tree(sys.argv[1])" in code and os.environ.get("FAKE_CAMULATOR_ARCHIVE_URL"):
                    code = (
                        "import os; "
                        + code.replace(
                            "install_camulator_source_tree(sys.argv[1])",
                            "install_camulator_source_tree("
                            "sys.argv[1], "
                            "archive_url=os.environ['FAKE_CAMULATOR_ARCHIVE_URL'], "
                            "expected_archive_sha256=os.environ['FAKE_CAMULATOR_ARCHIVE_SHA256'], "
                            "expected_archive_size=int(os.environ['FAKE_CAMULATOR_ARCHIVE_SIZE']), "
                            "expected_tree_sha256=os.environ['FAKE_CAMULATOR_TREE_SHA256'], "
                            "expected_tree_files=int(os.environ['FAKE_CAMULATOR_TREE_FILES']),"
                            ")",
                        )
                    )
                sys.argv = ["-c", *a[2:]]
                exec(compile(code, "<string>", "exec"), {"__name__": "__main__"})
            elif a[:2] == ["-m", "pytest"]:
                bt = next((x.split("=", 1)[1] for x in a if x.startswith("--basetemp=")), "")
                for check_p in [bt, os.environ.get("TMPDIR", "")]:
                    if not check_p:
                        continue
                    cur = pathlib.Path(check_p).resolve()
                    while str(cur) != "/":
                        if cur.exists():
                            m = cur.stat().st_mode
                            if (m & (stat.S_IWGRP | stat.S_IWOTH)) and not (m & stat.S_ISVTX):
                                sys.exit(f"unsafe writable ancestor {cur} ({oct(m)})")
                        cur = cur.parent
                log_p = os.environ.get("FAKE_PYTEST_LOG")
                if log_p:
                    pathlib.Path(log_p).write_text(f"basetemp={bt}\\nTMPDIR={os.environ.get('TMPDIR', '')}\\n")
                sys.exit(int(os.environ.get("FAKE_PYTEST_RC", "0")))
            elif a[:2] == ["-m", "aimip"]:
                import aimip
                if "--version" in a[2:]:
                    print(aimip.__version__)
                sys.exit(0)
            """
        ),
        encoding="utf-8",
    )

    pip_driver = work_dir / "venv_pip_driver.py"
    pip_driver.write_text(
        textwrap.dedent(
            f"""\
            import os, pathlib, sys
            SP = pathlib.Path(sys.argv[1])
            REAL_VER = {real_version!r}
            REAL_AIMIP_DIR = {str(REPO_ROOT / "src" / "aimip")!r}
            args = sys.argv[2:]
            cmdline = " ".join(args)
            fail_sub = os.environ.get("FAKE_NATIVE_FAIL_MATCH", "")
            if fail_sub and fail_sub in cmdline:
                sys.stderr.write(f"simulated pip failure on {{fail_sub}}\\n")
                sys.exit(17)
            if args[:1] == ["freeze"]:
                print("aimip==" + os.environ.get("FAKE_NATIVE_AIMIP_VER", REAL_VER))
                sys.exit(0)
            target = SP
            if "--target" in args:
                target = pathlib.Path(args[args.index("--target") + 1])
                target.mkdir(parents=True, exist_ok=True)
            if "torch==2.14.0" in cmdline:
                d = target / "torch"
                d.mkdir(parents=True, exist_ok=True)
                (d / "__init__.py").write_text(
                    f"__version__ = {{os.environ.get('FAKE_NATIVE_TORCH_VER', '2.14.0+cu130')!r}}\\n"
                )
            if "torch==2.4.1" in cmdline:
                d = target / "torch"
                d.mkdir(parents=True, exist_ok=True)
                (d / "nn.py").write_text("class LayerNorm:\\n    pass\\n")
                (d / "__init__.py").write_text(
                    "from . import nn\\n"
                    f"__version__ = {{os.environ.get('FAKE_NATIVE_TORCH24_VER', '2.4.1+cu121')!r}}\\n"
                )
            if "numpy==1.26.4" in cmdline:
                np_d = target / "numpy"
                np_d.mkdir(parents=True, exist_ok=True)
                (np_d / "__init__.py").write_text("__version__ = '1.26.4'\\n")
            if "torch_harmonics==0.7.2" in cmdline:
                th_d = target / "torch_harmonics"
                th_d.mkdir(parents=True, exist_ok=True)
                (th_d / "__init__.py").write_text("__version__ = '0.7.2'\\n")
            if "pytorch-lightning==2.4.0" in cmdline:
                for mname, mver in (
                    ("pytorch_lightning", "2.4.0"),
                    ("hydra", "1.3.2"),
                    ("omegaconf", "2.3.0"),
                    ("dacite", "1.8.1"),
                    ("tensorly", "0.9.0"),
                    ("tltorch", "0.5.0"),
                    ("tensordict", "0.5.0"),
                    ("toolz", "1.0.0"),
                    ("boto3", "1.37.1"),
                ):
                    md = target / mname
                    md.mkdir(parents=True, exist_ok=True)
                    (md / "__init__.py").write_text(f"__version__ = {{mver!r}}\\n")
            if "fme==2026.5.1" in cmdline:
                d = target / "fme"
                (d / "ace").mkdir(parents=True, exist_ok=True)
                (d / "__init__.py").write_text(
                    f"__version__ = {{os.environ.get('FAKE_NATIVE_FME_VER', '2026.5.1')!r}}\\n"
                )
                (d / "core.py").write_text("def set_seed(*a, **k): pass\\n")
                (d / "ace" / "__init__.py").write_text("")
                (d / "ace" / "inference.py").write_text("")
                # Tight 0600/0700 entries to exercise `ensure_world_readable()` selective `find ... ! -perm` repair.
                tf = target / "_tight_file.txt"
                tf.write_text("x\\n")
                tf.chmod(0o600)
                tx = target / "_tight_exec.sh"
                tx.write_text("#!/bin/sh\\n")
                tx.chmod(0o700)
                td = target / "_tight_dir"
                td.mkdir(exist_ok=True)
                td.chmod(0o700)
            if "fme==2026.10.0" in cmdline:
                d = target / "fme"
                (d / "coupled").mkdir(parents=True, exist_ok=True)
                (d / "__init__.py").write_text(
                    f"__version__ = {{os.environ.get('FAKE_NATIVE_OVERLAY_VER', '2026.10.0')!r}}\\n"
                )
                (d / "coupled" / "__init__.py").write_text("")
                (d / "coupled" / "inference.py").write_text("")
            if "--no-deps" in args and "--target" not in args and "torch_harmonics==0.7.2" not in cmdline:
                d = target / "aimip"
                d.mkdir(parents=True, exist_ok=True)
                (d / "__init__.py").write_text(
                    "import os as _os\\n"
                    f"__version__ = {{os.environ.get('FAKE_NATIVE_AIMIP_VER', REAL_VER)!r}}\\n"
                    f"__path__ = [_os.path.dirname(__file__), {{REAL_AIMIP_DIR!r}}]\\n"
                )
                (d / "emu_job.py").write_text("")
                if "cbottle @" in cmdline:
                    cb = target / "cbottle"
                    cb.mkdir(parents=True, exist_ok=True)
                    (cb / "__init__.py").write_text("")
                if "nvidia-modulus @" in cmdline:
                    mod = target / "modulus" / "models" / "sfno"
                    mod.mkdir(parents=True, exist_ok=True)
                    (target / "modulus" / "__init__.py").write_text("")
                    (target / "modulus" / "models" / "__init__.py").write_text("")
                    (mod / "__init__.py").write_text("")
                    (mod / "initialization.py").write_text("def trunc_normal_(*a, **k): pass\\n")
                    (mod / "sfnonet.py").write_text(
                        "from apex.normalization import FusedLayerNorm\\n"
                        "class SphericalFourierNeuralOperatorNet: pass\\n"
                    )
            if "earth2studio" in cmdline:
                e2s = target / "earth2studio"
                (e2s / "models" / "px").mkdir(parents=True, exist_ok=True)
                (e2s / "__init__.py").write_text("__version__ = '0.13.0'\\n")
                (e2s / "models" / "__init__.py").write_text("")
                (e2s / "models" / "px" / "__init__.py").write_text("class DLESyMLatLon: pass\\n")
                (e2s / "models" / "px" / "dlesym_v0_isccp_era5.py").write_text("class DLESyMv0_ISCCP_ERA5LatLon: pass\\n")
                (e2s / "data.py").write_text("class ARCO_ERA5: pass\\nclass CBottle3D: pass\\n")
                pn = target / "physicsnemo"
                pn.mkdir(parents=True, exist_ok=True)
                (pn / "__init__.py").write_text("__version__ = '1.0.0'\\n")
            if "nvidia-cuda-nvcc" in cmdline:
                (target / "nvidia" / "cu13" / "lib").mkdir(parents=True, exist_ok=True)
            if "earth2grid" in cmdline:
                eg = target / "earth2grid"
                eg.mkdir(parents=True, exist_ok=True)
                (eg / "__init__.py").write_text("")
                if os.environ.get("FAKE_NATIVE_OMIT_HEALPIXPAD") != "1":
                    (target / "healpixpad_cuda.py").write_text("")
            sys.exit(0)
            """
        ),
        encoding="utf-8",
    )

    stub_py = work_dir / "stub_py312"
    stub_py.write_text(
        f"#!{sys.executable}\n"
        "import pathlib, sys\n"
        "class _V(tuple):\n"
        "    major, minor, micro, releaselevel, serial = 3, 12, 0, 'final', 0\n"
        "sys.version_info = _V((3, 12, 0, 'final', 0))\n"
        "args = sys.argv[1:]\n"
        "while args and args[0] in ('-I', '-B'):\n"
        "    args = args[1:]\n"
        "if args[:2] == ['-m', 'venv']:\n"
        "    vdir = pathlib.Path(args[2])\n"
        "    (vdir / 'bin').mkdir(parents=True, exist_ok=True)\n"
        "    sp = vdir / 'lib' / 'python3.12' / 'site-packages'\n"
        "    sp.mkdir(parents=True, exist_ok=True)\n"
        "    vpy = vdir / 'bin' / 'python'\n"
        f"    vpy.write_text('#!/bin/sh\\nexec \"{sys.executable}\" \"{py_driver}\" \"' + str(sp) + '\" \"$@\"\\n')\n"
        "    vpy.chmod(0o755)\n"
        "    vpip = vdir / 'bin' / 'pip'\n"
        f"    vpip.write_text('#!/bin/sh\\nexec \"{sys.executable}\" \"{pip_driver}\" \"' + str(sp) + '\" \"$@\"\\n')\n"
        "    vpip.chmod(0o755)\n"
        "    sys.exit(0)\n"
        "elif args and args[0] == '-c':\n"
        "    code = args[1]; sys.argv = ['-c', *args[2:]]; exec(compile(code, '<string>', 'exec'), {'__name__': '__main__'})\n"
        "elif args and args[0] == '-':\n"
        "    code = sys.stdin.read(); sys.argv = ['-', *args[1:]]; exec(compile(code, '<stdin>', 'exec'), {'__name__': '__main__'})\n"
        "elif args and args[0] == '-V':\n"
        "    print('Python 3.12.0')\n",
        encoding="utf-8",
    )
    stub_py.chmod(0o755)
    return stub_py


def test_install_native_fails_on_step_or_version_errors_without_launcher_and_uses_safe_pytest_tmp(
    safe_gate_root: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`install-native.sh` chains every build command, stages verified `miles-credit` via `.pth`, asserts exact imports/versions, leaves no launcher on failure, and picks a `selftest_dir_ok` pytest tmpdir."""
    from aimip.safe_pickle import compute_sha256sum_tree_digest

    real_version = (REPO_ROOT / "container" / "VERSION").read_text(encoding="utf-8").strip()
    stub_py = _make_stub_native_python(safe_gate_root, real_version)
    runs_root = safe_gate_root / "runs_native"
    runs_root.mkdir()

    good_cam = _build_fake_miles_credit_archive(
        safe_gate_root / "fake_miles_credit_good.tar.gz",
        include_deprecated_transforms=True,
    )
    missing_ns_cam = _build_fake_miles_credit_archive(
        safe_gate_root / "fake_miles_credit_missing_deprecated.tar.gz",
        include_deprecated_transforms=False,
    )
    degraded_ms_cam = _build_fake_miles_credit_archive(
        safe_gate_root / "fake_miles_credit_degraded_ms.tar.gz",
        include_deprecated_transforms=True,
        postblock_available_override=False,
    )

    def _cam_env(spec: dict[str, str]) -> dict[str, str]:
        return {
            "FAKE_CAMULATOR_ARCHIVE_URL": spec["url"],
            "FAKE_CAMULATOR_ARCHIVE_SIZE": spec["archive_size"],
            "FAKE_CAMULATOR_ARCHIVE_SHA256": spec["archive_sha256"],
            "FAKE_CAMULATOR_TREE_SHA256": spec["tree_sha256"],
            "FAKE_CAMULATOR_TREE_FILES": spec["tree_files"],
        }

    def _run_native(prefix: pathlib.Path, extra_args: list[str], env_overrides: dict[str, str]) -> subprocess.CompletedProcess[str]:
        env = dict(os.environ)
        for k in (
            "FAKE_NATIVE_FAIL_MATCH",
            "FAKE_NATIVE_FME_VER",
            "FAKE_NATIVE_OVERLAY_VER",
            "FAKE_NATIVE_TORCH_VER",
            "FAKE_NATIVE_TORCH24_VER",
            "FAKE_NATIVE_AIMIP_VER",
            "FAKE_NATIVE_OMIT_HEALPIXPAD",
            "FAKE_NATIVE_OMIT_APEX_SHIM",
            "FAKE_PYTEST_LOG",
            "FAKE_PYTEST_RC",
            "FAKE_CAMULATOR_ARCHIVE_URL",
            "FAKE_CAMULATOR_ARCHIVE_SIZE",
            "FAKE_CAMULATOR_ARCHIVE_SHA256",
            "FAKE_CAMULATOR_TREE_SHA256",
            "FAKE_CAMULATOR_TREE_FILES",
        ):
            env.pop(k, None)
        env.update(_cam_env(good_cam))
        env.update(env_overrides)
        return subprocess.run(
            [
                "/bin/bash",
                str(INSTALL_NATIVE_PATH),
                "--prefix",
                str(prefix),
                "--site",
                "local",
                "--run-root",
                str(runs_root),
                "--python",
                str(stub_py),
                *extra_args,
            ],
            env=env,
            capture_output=True,
            text=True,
            check=False,
        )

    # 1. Failure cases: intermediate pip failure (followed by succeeding `pip freeze`), wrong aimip/fme/overlay/torch version,
    #    missing `healpixpad_cuda` under --with-emu, tampered `miles-credit` archive SHA, missing namespace subpackage
    #    `credit.transforms.deprecated`, or degraded `Model_State.POSTBLOCK_AVAILABLE=False` -> ALL must exit non-zero at
    #    the exact guard and leave NO bin/aimip and NO VERSION!
    failure_cases: list[tuple[str, list[str], dict[str, str], str]] = [
        ("mid_block_pip_fail", [], {"FAKE_NATIVE_FAIL_MATCH": "requirements-agent.txt"}, "venv build failed"),
        ("wrong_aimip_ver", [], {"FAKE_NATIVE_AIMIP_VER": "0.0.1"}, "import/version check failed"),
        ("wrong_fme_ver", [], {"FAKE_NATIVE_FME_VER": "2026.4.0"}, "import/version check failed"),
        ("wrong_overlay_ver", [], {"FAKE_NATIVE_OVERLAY_VER": "2026.5.1"}, "overlay check failed"),
        ("wrong_torch_ver", [], {"FAKE_NATIVE_TORCH_VER": "2.13.0+cu130"}, "import/version check failed"),
        ("emu_mid_pip_fail", ["--with-emu"], {"FAKE_NATIVE_FAIL_MATCH": "nvidia-cuda-nvcc"}, "rerun without --with-emu"),
        ("emu_missing_healpixpad", ["--with-emu"], {"FAKE_NATIVE_OMIT_HEALPIXPAD": "1"}, "venv-emu import/version check failed"),
        ("camulator_mid_pip_fail", ["--with-camulator"], {"FAKE_NATIVE_FAIL_MATCH": "constraints-camulator.txt"}, "rerun without --with-camulator"),
        ("camulator_bad_archive_sha", ["--with-camulator"], {"FAKE_CAMULATOR_ARCHIVE_SHA256": "0" * 64}, "SHA-256 mismatch"),
        ("camulator_missing_deprecated_ns", ["--with-camulator"], _cam_env(missing_ns_cam), "venv-camulator import/version check failed"),
        ("camulator_degraded_postblock", ["--with-camulator"], _cam_env(degraded_ms_cam), "venv-camulator import/version check failed"),
        ("camulator_wrong_torch", ["--with-camulator"], {"FAKE_NATIVE_TORCH24_VER": "2.14.0+cu130"}, "venv-camulator import/version check failed"),
        ("sdyff_mid_pip_fail", ["--with-sdyff"], {"FAKE_NATIVE_FAIL_MATCH": "constraints-sdyff.txt"}, "rerun without --with-sdyff"),
        ("sdyff_missing_apex_shim", ["--with-sdyff"], {"FAKE_NATIVE_OMIT_APEX_SHIM": "1"}, "venv-sdyff import/version check failed"),
        ("sdyff_wrong_torch", ["--with-sdyff"], {"FAKE_NATIVE_TORCH24_VER": "2.14.0+cu130"}, "venv-sdyff import/version check failed"),
    ]
    for case_name, extra, overrides, expected_err in failure_cases:
        pfx = safe_gate_root / f"fail_{case_name}"
        res = _run_native(pfx, extra, overrides)
        assert res.returncode != 0, f"{case_name} unexpectedly succeeded:\n{res.stdout}\n{res.stderr}"
        assert expected_err in res.stderr, f"{case_name} failed for wrong reason: {res.stderr!r}"
        assert not (pfx / "bin" / "aimip").exists(), f"{case_name} left {pfx / 'bin' / 'aimip'}"
        assert not (pfx / "VERSION").exists(), f"{case_name} left {pfx / 'VERSION'}"

    # 2. Group-writable prefix parent + world-writable non-sticky TMPDIR -> installer skips both via `selftest_dir_ok`
    #    (selecting safe `runs_root` or `/tmp`), stages `venv-camulator/src/miles-credit` + `miles_credit_camulator.pth`
    #    with zero `.pyc` pollution and zero symlinks, runs pytest cleanly, and writes VERSION + bin/aimip only at the end!
    shared_parent = safe_gate_root / "shared_group_parent"
    shared_parent.mkdir(mode=0o775)
    os.chmod(shared_parent, 0o775)
    bad_tmp = safe_gate_root / "world_writable_tmp"
    bad_tmp.mkdir(mode=0o777)
    os.chmod(bad_tmp, 0o777)
    pytest_log = safe_gate_root / "pytest_invoked.log"
    ok_pfx = shared_parent / "aimip_native_ok"
    ok_res = _run_native(
        ok_pfx,
        ["--with-emu", "--with-camulator", "--with-sdyff"],
        {"TMPDIR": str(bad_tmp), "FAKE_PYTEST_LOG": str(pytest_log)},
    )
    assert ok_res.returncode == 0, ok_res.stderr + ok_res.stdout
    assert (ok_pfx / "bin" / "aimip").is_file()
    assert (ok_pfx / "VERSION").read_text(encoding="utf-8").strip() == real_version
    pth_file = ok_pfx / "venv-camulator" / "lib" / "python3.12" / "site-packages" / "miles_credit_camulator.pth"
    cam_src_dir = ok_pfx / "venv-camulator" / "src" / "miles-credit"
    assert pth_file.is_file()
    assert pth_file.read_text(encoding="utf-8").strip() == os.path.realpath(cam_src_dir)
    assert not (cam_src_dir / "credit" / "applications" / "train.py").exists()
    assert compute_sha256sum_tree_digest(str(cam_src_dir)) == (
        good_cam["tree_sha256"],
        int(good_cam["tree_files"]),
    )
    # Direct production `install_camulator_source_tree` ignores ambient `AIMIP_CAMULATOR_{ARCHIVE,TREE}_*` env vars!
    from aimip.camulator import install_camulator_source_tree

    monkeypatch.setenv("AIMIP_CAMULATOR_ARCHIVE_URL", good_cam["url"])
    monkeypatch.setenv("AIMIP_CAMULATOR_ARCHIVE_SIZE", good_cam["archive_size"])
    monkeypatch.setenv("AIMIP_CAMULATOR_ARCHIVE_SHA256", "0" * 64)
    monkeypatch.setenv("AIMIP_CAMULATOR_TREE_SHA256", "0" * 64)
    monkeypatch.setenv("AIMIP_CAMULATOR_TREE_FILES", "1")
    try:
        direct_venv = safe_gate_root / "direct_cam_venv"
        direct_sp = direct_venv / "lib" / "python3.12" / "site-packages"
        direct_sp.mkdir(parents=True)
        direct_installed = install_camulator_source_tree(
            str(direct_venv),
            site_packages_dir=str(direct_sp),
            archive_url=good_cam["url"],
            expected_archive_sha256=good_cam["archive_sha256"],
            expected_archive_size=int(good_cam["archive_size"]),
            expected_tree_sha256=good_cam["tree_sha256"],
            expected_tree_files=int(good_cam["tree_files"]),
        )
        assert direct_installed["tree_sha256"] == good_cam["tree_sha256"]
        with pytest.raises(ValidationError):
            # Without explicit pin overrides, `install_camulator_source_tree` enforces pinned production constants
            # and ignores ambient `AIMIP_CAMULATOR_ARCHIVE_*` / `AIMIP_CAMULATOR_TREE_*`!
            monkeypatch.setenv("AIMIP_CAMULATOR_ARCHIVE_SHA256", good_cam["archive_sha256"])
            monkeypatch.setenv("AIMIP_CAMULATOR_TREE_SHA256", good_cam["tree_sha256"])
            monkeypatch.setenv("AIMIP_CAMULATOR_TREE_FILES", good_cam["tree_files"])
            install_camulator_source_tree(
                str(direct_venv),
                site_packages_dir=str(direct_sp),
                archive_url=good_cam["url"],
            )
    finally:
        for env_k in (
            "AIMIP_CAMULATOR_ARCHIVE_URL",
            "AIMIP_CAMULATOR_ARCHIVE_SIZE",
            "AIMIP_CAMULATOR_ARCHIVE_SHA256",
            "AIMIP_CAMULATOR_TREE_SHA256",
            "AIMIP_CAMULATOR_TREE_FILES",
        ):
            monkeypatch.delenv(env_k, raising=False)
    sdyff_sp = ok_pfx / "venv-sdyff" / "lib" / "python3.12" / "site-packages"
    main_sp = ok_pfx / "venv" / "lib" / "python3.12" / "site-packages"
    assert (sdyff_sp / "apex" / "normalization.py").is_file()
    for other_v in ("venv", "venv-emu", "venv-camulator"):
        assert not (ok_pfx / other_v / "lib" / "python3.12" / "site-packages" / "apex").exists()
    assert stat.S_IMODE((main_sp / "_tight_file.txt").stat().st_mode) & 0o444 == 0o444
    assert stat.S_IMODE((main_sp / "_tight_file.txt").stat().st_mode) & 0o111 == 0
    assert stat.S_IMODE((main_sp / "_tight_exec.sh").stat().st_mode) & 0o555 == 0o555
    assert stat.S_IMODE((main_sp / "_tight_dir").stat().st_mode) & 0o555 == 0o555
    logged = pytest_log.read_text(encoding="utf-8")
    assert str(bad_tmp) not in logged
    assert str(shared_parent) not in logged

    # 3. CI workflow `tests.yml` sets `TMPDIR: /tmp` for the pytest step as well as the wrapper step
    #    (only in a source checkout; the image ships no `.github/`).
    tests_yml_path = REPO_ROOT / ".github" / "workflows" / "tests.yml"
    if tests_yml_path.is_file():
        assert tests_yml_path.read_text(encoding="utf-8").count("TMPDIR: /tmp") >= 2


def test_install_container_version_checks_aarch64_autobuild_image_ref_and_error_hints(
    safe_gate_root: pathlib.Path,
) -> None:
    """`install-container.sh` rejects `-` in `--image`, derives registry login hints, checks `$SIF.incoming`/existing `$SIF`/docker versions with `--cleanenv` & `python -I`, auto-builds on `aarch64`, and writes `image.ref`."""
    import textwrap

    real_version = (REPO_ROOT / "container" / "VERSION").read_text(encoding="utf-8").strip()
    runs_root = safe_gate_root / "runs_cnt"
    runs_root.mkdir()

    # 1. --image starting with '-' is rejected up front.
    for bad_img in ("-bad", "--help", "-v"):
        r_bad = subprocess.run(
            [
                "/bin/bash",
                str(INSTALL_CONTAINER_PATH),
                "--prefix",
                str(safe_gate_root / "bad_img_pfx"),
                "--site",
                "local",
                "--image",
                bad_img,
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        assert r_bad.returncode == 2
        assert "--image" in r_bad.stderr

    # Set up fake apptainer, docker, and uname controlled by env vars.
    fakebin = safe_gate_root / "fakebin_cnt_full"
    fakebin.mkdir()
    fake_uname = fakebin / "uname"
    fake_uname.write_text('#!/bin/sh\necho "${FAKE_UNAME_ARCH:-x86_64}"\n', encoding="utf-8")
    fake_uname.chmod(0o755)

    rt_log = safe_gate_root / "runtime_calls.log"
    fake_apptainer = fakebin / "apptainer"
    fake_apptainer.write_text(
        textwrap.dedent(
            f"""\
            #!{sys.executable}
            import os, pathlib, sys
            log_path = pathlib.Path({str(rt_log)!r})
            with log_path.open("a", encoding="utf-8") as fh:
                fh.write("APPTAINER " + " ".join(sys.argv[1:]) + "\\n")
            args = sys.argv[1:]
            while args and args[0] == "-s":
                args = args[1:]
            if args and args[0] == "build":
                if os.environ.get("FAKE_PULL_FAIL") == "1":
                    sys.stderr.write("simulated registry failure\\n")
                    sys.exit(1)
                # target SIF is second-to-last arg (`build [--mksquashfs-args ...] <dst> <src>`)
                dst = pathlib.Path(args[-2])
                dst.parent.mkdir(parents=True, exist_ok=True)
                dst.write_bytes(b"FAKE_SIF_IMAGE")
                sys.exit(0)
            if args and args[0] == "exec":
                # Handle both wrapper `--version` self-test and installer `python -I ...` check.
                if "--version" in args:
                    print({real_version!r})
                    sys.exit(0)
                if "python" in args:
                    py_idx = args.index("python")
                    py_args = args[py_idx + 1:]
                    code_idx = py_args.index("-c")
                    code = py_args[code_idx + 1]
                    rest = py_args[code_idx + 2:]
                    ver = os.environ.get("FAKE_IMAGE_VER", {real_version!r})
                    import types
                    sys.modules["aimip"] = types.SimpleNamespace(__version__=ver)
                    sys.argv = ["-c", *rest]
                    exec(compile(code, "<string>", "exec"), {{"__name__": "__main__"}})
                sys.exit(0)
            sys.exit(0)
            """
        ),
        encoding="utf-8",
    )
    fake_apptainer.chmod(0o755)

    fake_docker = fakebin / "docker"
    fake_docker.write_text(
        textwrap.dedent(
            f"""\
            #!{sys.executable}
            import os, pathlib, sys
            log_path = pathlib.Path({str(rt_log)!r})
            with log_path.open("a", encoding="utf-8") as fh:
                fh.write("DOCKER " + " ".join(sys.argv[1:]) + "\\n")
            args = sys.argv[1:]
            if args[:1] == ["pull"]:
                if os.environ.get("FAKE_PULL_FAIL") == "1":
                    sys.stderr.write("simulated docker pull failure\\n")
                    sys.exit(1)
                sys.exit(0)
            if args[:1] == ["build"]:
                sys.exit(0)
            if args[:1] == ["image"] and len(args) >= 2 and args[1] == "inspect":
                fmt = args[args.index("--format") + 1] if "--format" in args else ""
                if "RepoDigests" in fmt:
                    print(os.environ.get("FAKE_REPO_DIGEST", "registry.example.org:5000/org/aimip@sha256:abcdef0123456789"))
                else:
                    print("sha256:deadbeefcafebabe")
                sys.exit(0)
            if args[:1] == ["run"]:
                if "--version" in args:
                    print({real_version!r})
                    sys.exit(0)
                if "--entrypoint" in args and "python" in args:
                    # find `-c`
                    c_idx = args.index("-c")
                    code = args[c_idx + 1]
                    rest = args[c_idx + 2:]
                    ver = os.environ.get("FAKE_IMAGE_VER", {real_version!r})
                    import types
                    sys.modules["aimip"] = types.SimpleNamespace(__version__=ver)
                    sys.argv = ["-c", *rest]
                    exec(compile(code, "<string>", "exec"), {{"__name__": "__main__"}})
                sys.exit(0)
            sys.exit(0)
            """
        ),
        encoding="utf-8",
    )
    fake_docker.chmod(0o755)

    def _run_cnt(prefix: pathlib.Path, extra: list[str], env_ov: dict[str, str]) -> subprocess.CompletedProcess[str]:
        env = dict(os.environ)
        env["PATH"] = f"{fakebin}:{env.get('PATH', '')}"
        env["TMPDIR"] = str(safe_gate_root)
        for k in ("FAKE_PULL_FAIL", "FAKE_IMAGE_VER", "FAKE_UNAME_ARCH", "FAKE_REPO_DIGEST"):
            env.pop(k, None)
        env.update(env_ov)
        return subprocess.run(
            [
                "/bin/bash",
                str(INSTALL_CONTAINER_PATH),
                "--prefix",
                str(prefix),
                "--site",
                "local",
                "--run-root",
                str(runs_root),
                *extra,
            ],
            env=env,
            capture_output=True,
            text=True,
            check=False,
        )

    # 2. Error hints derive registry host dynamically instead of hard-coding ghcr.io.
    r_hint_app = _run_cnt(
        safe_gate_root / "hint_app",
        ["--runtime", "apptainer", "--image", "registry.example.org/team/aimip:1.3.0"],
        {"FAKE_PULL_FAIL": "1"},
    )
    assert r_hint_app.returncode != 0
    assert "docker://registry.example.org" in r_hint_app.stderr
    assert "ghcr.io" not in r_hint_app.stderr

    r_hint_doc = _run_cnt(
        safe_gate_root / "hint_doc",
        ["--runtime", "docker", "--image", "registry.example.org:5000/org/aimip:1.3.0"],
        {"FAKE_PULL_FAIL": "1"},
    )
    assert r_hint_doc.returncode != 0
    assert "login registry.example.org:5000" in r_hint_doc.stderr
    assert "ghcr.io" not in r_hint_doc.stderr

    # 3. Apptainer wrong-version pull fails BEFORE `mv`: neither `$SIF` nor `$SIF.incoming` nor `bin/aimip` remains!
    pfx_bad_sif = safe_gate_root / "bad_sif_pull"
    r_bad_sif = _run_cnt(
        pfx_bad_sif,
        ["--runtime", "apptainer", "--image", "registry.example.org/team/aimip:0.9.0"],
        {"FAKE_IMAGE_VER": "0.9.0"},
    )
    assert r_bad_sif.returncode != 0
    assert not (pfx_bad_sif / "images" / f"aimip-{real_version}.sif").exists()
    assert not (pfx_bad_sif / "images" / f"aimip-{real_version}.sif.incoming").exists()
    assert not (pfx_bad_sif / "bin" / "aimip").exists()

    # 4. Pre-existing offline hand-copied `$SIF` with wrong version is ALSO checked and refused before writing `bin/aimip`!
    pfx_existing_bad = safe_gate_root / "existing_bad_sif"
    existing_sif = pfx_existing_bad / "images" / f"aimip-{real_version}.sif"
    existing_sif.parent.mkdir(parents=True)
    existing_sif.write_bytes(b"HAND_COPIED_WRONG_VERSION_SIF")
    r_existing_bad = _run_cnt(
        pfx_existing_bad,
        ["--runtime", "apptainer"],
        {"FAKE_IMAGE_VER": "0.9.0"},
    )
    assert r_existing_bad.returncode != 0
    assert not (pfx_existing_bad / "bin" / "aimip").exists()

    # 5. Valid Apptainer pull passes `--cleanenv` and `python -I`, writes `image.ref`, `image.sha256`, and `bin/aimip`.
    rt_log.write_text("", encoding="utf-8")
    pfx_ok_app = safe_gate_root / "ok_app"
    r_ok_app = _run_cnt(
        pfx_ok_app,
        ["--runtime", "apptainer", "--image", f"registry.example.org/team/aimip:{real_version}"],
        {"FAKE_IMAGE_VER": real_version},
    )
    assert r_ok_app.returncode == 0, r_ok_app.stderr + r_ok_app.stdout
    app_calls = rt_log.read_text(encoding="utf-8")
    assert "--cleanenv" in app_calls
    assert "python -I" in app_calls
    assert f"{pfx_ok_app}/images/aimip-{real_version}.sif.incoming" in app_calls
    assert (pfx_ok_app / "image.ref").read_text(encoding="utf-8").strip() == f"docker://registry.example.org/team/aimip:{real_version}"
    assert (pfx_ok_app / "bin" / "aimip").is_file()

    # 6. Docker wrong-version fails on both pull and `--build` before `bin/aimip`; valid pull uses `python -I` and records RepoDigest in `image.ref`.
    for mode_args, sub_name in (([], "bad_doc_pull"), (["--build"], "bad_doc_build")):
        pfx_bd = safe_gate_root / sub_name
        r_bd = _run_cnt(pfx_bd, ["--runtime", "docker", *mode_args], {"FAKE_IMAGE_VER": "0.9.0"})
        assert r_bd.returncode != 0
        assert not (pfx_bd / "bin" / "aimip").exists()

    rt_log.write_text("", encoding="utf-8")
    pfx_ok_doc = safe_gate_root / "ok_doc"
    r_ok_doc = _run_cnt(
        pfx_ok_doc,
        ["--runtime", "docker", "--image", f"registry.example.org:5000/org/aimip:{real_version}"],
        {"FAKE_IMAGE_VER": real_version, "FAKE_REPO_DIGEST": "registry.example.org:5000/org/aimip@sha256:112233445566"},
    )
    assert r_ok_doc.returncode == 0, r_ok_doc.stderr + r_ok_doc.stdout
    doc_calls = rt_log.read_text(encoding="utf-8")
    assert "--entrypoint python" in doc_calls and " -I " in doc_calls
    img_ref_txt = (pfx_ok_doc / "image.ref").read_text(encoding="utf-8")
    assert f"registry.example.org:5000/org/aimip:{real_version}" in img_ref_txt
    assert "registry.example.org:5000/org/aimip@sha256:112233445566" in img_ref_txt

    # 7. aarch64 host without `--image` switches automatically to `--build` and explains published image is x86_64 only.
    rt_log.write_text("", encoding="utf-8")
    pfx_arm = safe_gate_root / "arm64_auto_build"
    r_arm = _run_cnt(
        pfx_arm,
        ["--runtime", "docker"],
        {"FAKE_UNAME_ARCH": "aarch64", "FAKE_IMAGE_VER": real_version},
    )
    assert r_arm.returncode == 0, r_arm.stderr + r_arm.stdout
    assert "x86_64" in r_arm.stdout
    arm_calls = rt_log.read_text(encoding="utf-8")
    assert "DOCKER build" in arm_calls
    assert "DOCKER pull" not in arm_calls
    assert "BASE_IMAGE=python:3.12.11-bookworm" in arm_calls


def test_apptainer_def_umask_and_dockerfile_chmod_fix_umask_000_exec_fme_failures(
    safe_gate_root: pathlib.Path,
) -> None:
    """`gen_apptainer_def.py` sets `umask 022` in `%post`, `install-container.sh` sets `umask 022`, and `Dockerfile` `chmod go-w` fixes the 4 `exec-fme` test failures under `umask 000` `/opt/aimip`."""
    def_out = safe_gate_root / "apptainer.def"
    gen_res = subprocess.run(
        [sys.executable, str(REPO_ROOT / "container" / "gen_apptainer_def.py"), str(REPO_ROOT / "container" / "Dockerfile"), str(def_out)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert gen_res.returncode == 0, gen_res.stderr
    def_txt = def_out.read_text(encoding="utf-8")
    post_section = def_txt.split("%post\n", 1)[1].split("\n%", 1)[0]
    post_lines = [ln.strip() for ln in post_section.splitlines() if ln.strip()]
    assert "umask 022" in post_lines[:3], post_lines[:5]
    assert "chmod go-w /opt" in post_section and "chmod -R go-w /opt/aimip" in post_section
    assert post_section.index("chmod -R go-w /opt/aimip") < post_section.index("python -m pytest")

    cnt_sh_head = "\n".join(INSTALL_CONTAINER_PATH.read_text(encoding="utf-8").splitlines()[:20])
    assert "umask 022" in cnt_sh_head

    # Simulate Apptainer `%files` creating `/opt` and `/opt/aimip` under umask 000 (mode 0777).
    sim_opt = safe_gate_root / "opt"
    sim_aimip = sim_opt / "aimip"
    sim_aimip.mkdir(parents=True)
    os.chmod(sim_opt, 0o777)
    os.chmod(sim_aimip, 0o777)

    four_tests_expr = (
        "test_exec_fme_status_submitted_txt_and_summary_custom_dims or "
        "test_python_dash_m_aimip_submitted_txt_sacct_and_failed_exec or "
        "test_exec_fme_silences_git_not_a_repository_stderr or "
        "test_exec_fme_rehashes_checkpoint_and_ic_and_verifies_config_against_manifest"
    )
    env = dict(os.environ)
    env["PYTHONPATH"] = str(REPO_ROOT / "src")
    env["HF_HUB_OFFLINE"] = "1"
    env["TMPDIR"] = "/tmp"

    # Without `chmod go-w`, all 4 tests fail under `--basetemp=<sim_aimip>/.pytest-tmp` due to 0777 ancestry!
    before = subprocess.run(
        [
            sys.executable,
            "-B",
            "-m",
            "pytest",
            "-q",
            "-p",
            "no:cacheprovider",
            f"--basetemp={sim_aimip / '.pytest-tmp-before'}",
            str(REPO_ROOT / "tests" / "test_run_and_exec.py"),
            "-k",
            four_tests_expr,
        ],
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert before.returncode != 0
    assert "4 failed" in before.stdout

    # Apply the exact Dockerfile remediation (`chmod go-w /opt && chmod -R go-w /opt/aimip`) -> all 4 tests pass!
    os.chmod(sim_opt, sim_opt.stat().st_mode & ~0o022)
    for root_d, dirs, files in os.walk(sim_aimip):
        os.chmod(root_d, os.stat(root_d).st_mode & ~0o022)
        for fn in files:
            fp = os.path.join(root_d, fn)
            if not os.path.islink(fp):
                os.chmod(fp, os.stat(fp).st_mode & ~0o022)

    after = subprocess.run(
        [
            sys.executable,
            "-B",
            "-m",
            "pytest",
            "-q",
            "-p",
            "no:cacheprovider",
            f"--basetemp={sim_aimip / '.pytest-tmp-after'}",
            str(REPO_ROOT / "tests" / "test_run_and_exec.py"),
            "-k",
            four_tests_expr,
        ],
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert after.returncode == 0, after.stdout + after.stderr
    assert "4 passed" in after.stdout


def test_image_workflow_push_gate_version_validation_env_passing_and_digest_summary(
    safe_gate_root: pathlib.Path,
) -> None:
    """`image.yml` validates `container/VERSION`, passes outputs via `env:`, pushes only `:<version>-dev-<shortsha>` on `workflow_dispatch`, pushes `:<version>` + `:latest` only on matching `v<version>` tag, and records the digest in `$GITHUB_STEP_SUMMARY`."""
    import textwrap
    import yaml

    wf_path = REPO_ROOT / ".github" / "workflows" / "image.yml"
    if not wf_path.is_file():
        pytest.skip("workflow files are not shipped in the image; this test runs in a source checkout")
    wf = yaml.safe_load(wf_path.read_text(encoding="utf-8"))
    steps = wf["jobs"]["image"]["steps"]

    # Every `run:` block must pass `steps.meta.outputs.*` via `env:`, never direct `${{ steps.` shell interpolation.
    for st in steps:
        if "run" in st:
            assert "${{ steps." not in st["run"], f"unsafe interpolation in step {st.get('name')!r}"

    meta_step = next(st for st in steps if st.get("id") == "meta")
    push_step = next(st for st in steps if st.get("name") == "Push")

    real_version = (REPO_ROOT / "container" / "VERSION").read_text(encoding="utf-8").strip()
    fakebin = safe_gate_root / "fakebin_wf"
    fakebin.mkdir()
    docker_log = safe_gate_root / "wf_docker.log"
    fake_docker = fakebin / "docker"
    fake_docker.write_text(
        textwrap.dedent(
            f"""\
            #!{sys.executable}
            import os, pathlib, sys
            args = sys.argv[1:]
            log_p = pathlib.Path({str(docker_log)!r})
            with log_p.open("a", encoding="utf-8") as fh:
                fh.write(" ".join(args) + "\\n")
            if args[:2] == ["manifest", "inspect"]:
                sys.exit(0 if os.environ.get("FAKE_TAG_EXISTS") == "1" else 1)
            if args[:2] == ["image", "inspect"]:
                print("ghcr.io/clidyn/aimip-claima@sha256:99887766554433221100ffeeddccbbaa")
                sys.exit(0)
            sys.exit(0)
            """
        ),
        encoding="utf-8",
    )
    fake_docker.chmod(0o755)

    def _run_wf(
        *,
        version_txt: str,
        ref_type: str,
        ref_name: str,
        event_name: str = "push",
        sha: str = "deadbeef1234567890abcdef1234567890abcdef",
        tag_exists: str = "0",
    ) -> tuple[int, dict[str, str], str, str]:
        wdir = safe_gate_root / f"wf_{event_name}_{ref_type}_{ref_name}_{tag_exists}_{ abs(hash(version_txt)) % 10000 }"
        (wdir / "container").mkdir(parents=True, exist_ok=True)
        (wdir / "container" / "VERSION").write_text(version_txt, encoding="utf-8")
        gh_out = wdir / "gh_output.txt"
        gh_out.write_text("", encoding="utf-8")
        gh_sum = wdir / "gh_summary.md"
        gh_sum.write_text("", encoding="utf-8")

        base_env = dict(os.environ)
        base_env["PATH"] = f"{fakebin}:{base_env.get('PATH', '')}"
        base_env["GITHUB_OUTPUT"] = str(gh_out)
        base_env["GITHUB_STEP_SUMMARY"] = str(gh_sum)
        base_env["GITHUB_REPOSITORY_OWNER"] = "CliDyn"
        base_env["GITHUB_EVENT_NAME"] = event_name
        base_env["GITHUB_REF_TYPE"] = ref_type
        base_env["GITHUB_REF_NAME"] = ref_name
        base_env["GITHUB_SHA"] = sha
        base_env["FAKE_TAG_EXISTS"] = tag_exists

        docker_log.write_text("", encoding="utf-8")
        r_meta = subprocess.run(
            ["/bin/bash", "-euo", "pipefail", "-c", meta_step["run"]],
            cwd=str(wdir),
            env=base_env,
            capture_output=True,
            text=True,
            check=False,
        )
        if r_meta.returncode != 0:
            return r_meta.returncode, {}, "", r_meta.stderr
        outputs: dict[str, str] = {}
        for line in gh_out.read_text(encoding="utf-8").splitlines():
            if "=" in line:
                k, v = line.split("=", 1)
                outputs[k] = v

        push_env = dict(base_env)
        for env_k, expr in (push_step.get("env") or {}).items():
            # Resolve `${{ steps.meta.outputs.<field> }}`
            field = expr.split("steps.meta.outputs.", 1)[1].split("}}", 1)[0].strip()
            push_env[env_k] = outputs[field]

        r_push = subprocess.run(
            ["/bin/bash", "-euo", "pipefail", "-c", push_step["run"]],
            cwd=str(wdir),
            env=push_env,
            capture_output=True,
            text=True,
            check=False,
        )
        return (
            r_push.returncode,
            outputs,
            docker_log.read_text(encoding="utf-8"),
            gh_sum.read_text(encoding="utf-8") + r_push.stderr,
        )

    # 1. Manual dispatch on a branch pushes ONLY `:<version>-dev-<shortsha>`, never `:<version>` or `:latest`, and records digest!
    rc1, out1, dlog1, sum1 = _run_wf(
        version_txt=f"{real_version}\n",
        event_name="workflow_dispatch",
        ref_type="branch",
        ref_name="main",
    )
    assert rc1 == 0, sum1
    assert out1["release"] == "false" and out1["primary_tag"] == f"{real_version}-dev-deadbeef"
    assert f"push ghcr.io/clidyn/aimip-claima:{real_version}-dev-deadbeef" in dlog1
    assert f"push ghcr.io/clidyn/aimip-claima:{real_version}\n" not in dlog1
    assert ":latest" not in dlog1
    assert "sha256:99887766554433221100ffeeddccbbaa" in sum1

    # 1b. Manual dispatch AT A MATCHING TAG still emits dev tag only (`release=false`), never `:<version>` or `:latest`!
    rc1b, out1b, dlog1b, sum1b = _run_wf(
        version_txt=f"{real_version}\n",
        event_name="workflow_dispatch",
        ref_type="tag",
        ref_name=f"v{real_version}",
    )
    assert rc1b == 0, sum1b
    assert out1b["release"] == "false" and out1b["primary_tag"] == f"{real_version}-dev-deadbeef"
    assert f"push ghcr.io/clidyn/aimip-claima:{real_version}-dev-deadbeef" in dlog1b
    assert f"push ghcr.io/clidyn/aimip-claima:{real_version}\n" not in dlog1b
    assert ":latest" not in dlog1b

    # 2. Push of matching release tag `v<release_version>` pushes `:<release_version>` and `:latest` and records digest in summary;
    #    a `.devN` version refuses any release tag push so dev branches never clash with release tags!
    if ".dev" in real_version:
        rc_dev_tag, _, dlog_dev_tag, _ = _run_wf(
            version_txt=f"{real_version}\n",
            event_name="push",
            ref_type="tag",
            ref_name=f"v{real_version}",
        )
        assert rc_dev_tag != 0 and dlog_dev_tag == ""

    release_ver = real_version.split(".dev")[0]
    rc2, out2, dlog2, sum2 = _run_wf(
        version_txt=f"{release_ver}\n",
        event_name="push",
        ref_type="tag",
        ref_name=f"v{release_ver}",
    )
    assert rc2 == 0, sum2
    assert out2["release"] == "true" and out2["primary_tag"] == release_ver
    assert f"push ghcr.io/clidyn/aimip-claima:{release_ver}\n" in dlog2
    assert "push ghcr.io/clidyn/aimip-claima:latest" in dlog2
    assert "sha256:99887766554433221100ffeeddccbbaa" in sum2

    # 3. Existing release tag on registry is not overwritten.
    rc3, _, dlog3, _ = _run_wf(
        version_txt=f"{release_ver}\n",
        event_name="push",
        ref_type="tag",
        ref_name=f"v{release_ver}",
        tag_exists="1",
    )
    assert rc3 != 0
    assert "push " not in dlog3

    # 4. Push of mismatched tag `v9.9.9` or malformed `container/VERSION` fails in `meta` before build/push.
    rc4, _, dlog4, _ = _run_wf(
        version_txt=f"{real_version}\n", event_name="push", ref_type="tag", ref_name="v9.9.9"
    )
    assert rc4 != 0 and dlog4 == ""
    rc5, _, dlog5, _ = _run_wf(
        version_txt="1.3.0;id\n", event_name="workflow_dispatch", ref_type="branch", ref_name="main"
    )
    assert rc5 != 0 and dlog5 == ""
