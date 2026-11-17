"""Tests for input validators (shell/SBATCH/path injection + /scratch|/work) and registry v2."""

from __future__ import annotations

import json
import pathlib

import pytest

from aimip.cli import main
from aimip.errors import RegistryError, ValidationError
from aimip.registry import load_registry
from aimip.validators import (
    MAX_SEED,
    validate_int_range,
    validate_iso_timestamp,
    validate_model_name,
    validate_python_module,
    validate_run_path,
    validate_safe_path,
    validate_seed,
    validate_slurm_job_id,
    validate_slurm_mem,
    validate_slurm_time,
    validate_slurm_token,
)


@pytest.mark.parametrize(
    "bad_path",
    [
        "",
        "/",
        "relative/path/out",
        "./out",
        "../escape",
        "/scratch/work/../escape",
        "/scratch/work/..",
        "/scratch/work/$(whoami)/run",
        "/scratch/work/`id`/run",
        "/scratch/work/$HOME/run",
        "/scratch/work/${USER}/run",
        "/scratch/work/run;rm -rf /",
        "/scratch/work/run && echo pwned",
        "/scratch/work/run | cat",
        "/scratch/work/run > /tmp/x",
        "/scratch/work/run < /tmp/x",
        '/scratch/work/"quoted"/run',
        "/scratch/work/'single'/run",
        "/scratch/work/with space/run",
        "/scratch/work/colon:in:path",
        "/scratch/work/line\n#SBATCH --uid=0",
        "/scratch/work/cr\r#SBATCH --gid=0",
        "/scratch/work/tab\tpath",
        "/scratch/work/nul\x00byte",
        "/scratch/work/glob*/run",
        "/scratch/work/qmark?/run",
        "/scratch/work/hash#comment",
        "/scratch/work/tilde~user",
        "/scratch/work/back\\slash",
    ],
)
def test_validate_safe_path_rejects_injection_and_traversal(bad_path: str) -> None:
    with pytest.raises(ValidationError):
        validate_safe_path(bad_path)


def test_validate_safe_path_and_run_path_enforce_scratch_or_work() -> None:
    assert (
        validate_run_path("/scratch/u/user_01/aimip-runs/ace2s-shield-plus+r1")
        == "/scratch/u/user_01/aimip-runs/ace2s-shield-plus+r1"
    )
    assert (
        validate_run_path("/work/xx0000/user_01/runs/r2")
        == "/work/xx0000/user_01/runs/r2"
    )

    # Production `ALLOWED_RUN_ROOTS` strictly rejects paths outside `/scratch/*` and `/work/*`.
    for disallowed in (
        "/tmp/aimip-run",
        "/var/tmp/aimip-run",
        "/opt/aimip/run",
        "/home/user/run",
        "/scratch",
        "/work",
    ):
        with pytest.raises(ValidationError, match="must live under /scratch or /work"):
            validate_run_path(disallowed)


@pytest.mark.parametrize(
    "bad_ts",
    [
        "",
        "1979-13-01T00:00:00",
        "1979-00-01T00:00:00",
        "1979-04-31T00:00:00",
        "1979-02-31T00:00:00",
        "1979-01-01T24:00:00",
        "1979-01-01T00:60:00",
        "1979-01-01T00:00:60",
        "1979-01-01T00:00:00;id",
        "1979-01-01T00:00:00\n#SBATCH",
        "$(date)",
    ],
)
def test_validate_iso_timestamp_rejects_malformed_or_injected(bad_ts: str) -> None:
    with pytest.raises(ValidationError):
        validate_iso_timestamp(bad_ts)


def test_validate_iso_timestamp_normalizes_date_only() -> None:
    assert validate_iso_timestamp("1979-01-01") == "1979-01-01T00:00:00"
    assert validate_iso_timestamp("2014-12-31T18:00:00") == "2014-12-31T18:00:00"


@pytest.mark.parametrize(
    "bad_token",
    [
        "gpu;id",
        "gpu\n#SBATCH --exclusive",
        "xx0000 gpu",
        "$(id)",
        "`uname`",
        "gpu'quote",
    ],
)
def test_slurm_validators_reject_injection(bad_token: str) -> None:
    with pytest.raises(ValidationError):
        validate_slurm_token(bad_token, "partition")
    with pytest.raises(ValidationError):
        validate_slurm_mem(bad_token)
    with pytest.raises(ValidationError):
        validate_slurm_time(bad_token)
    with pytest.raises(ValidationError):
        validate_slurm_job_id(bad_token)


def test_int_seed_and_module_validators() -> None:
    with pytest.raises(ValidationError):
        validate_int_range(True, "steps")
    with pytest.raises(ValidationError):
        validate_int_range(0, "steps", min_value=1)
    with pytest.raises(ValidationError):
        validate_seed(-1)
    with pytest.raises(ValidationError):
        validate_seed(MAX_SEED + 1)
    assert validate_seed(0) == 0
    assert validate_python_module("fme.ace.inference") == "fme.ace.inference"
    with pytest.raises(ValidationError):
        validate_python_module("fme.ace.inference;bad_cmd")
    with pytest.raises(ValidationError):
        validate_model_name("ACE2_UPPERCASE")
    with pytest.raises(ValidationError):
        validate_model_name("unknown-model", allowed={"ace2-era5"})


def test_example_registry_loads_all_models_and_forcing_sets(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import aimip

    example_reg = pathlib.Path(aimip.__file__).resolve().parent / "registry.example.json"
    monkeypatch.setenv("AIMIP_REGISTRY", str(example_reg))
    monkeypatch.setenv("AIMIP_PLAYGROUND", "/opt/aimip/playground")

    catalog = load_registry()
    assert len(catalog) == 18
    assert list(catalog.keys())[:5] == [
        "ace2-era5",
        "ace2-shield",
        "ace2s-shield-plus",
        "ace2-eamv3",
        "hiro-ace",
    ]
    assert catalog["ace2-era5"].stochastic is False
    assert len(catalog["ace2-era5"].ic_files) == 2
    assert [fs.name for fs in catalog["ace2-era5"].forcing_sets] == ["amip-hist"]

    assert catalog["ace2-shield"].stochastic is False
    assert [fs.name for fs in catalog["ace2-shield"].forcing_sets] == [
        "amip-hist",
        "amip-oos",
    ]

    assert catalog["ace2s-shield-plus"].stochastic is True
    assert [fs.name for fs in catalog["ace2s-shield-plus"].forcing_sets] == [
        "amip-hist",
        "amip-oos",
    ]

    assert catalog["ace2-eamv3"].stochastic is False
    assert [fs.name for fs in catalog["ace2-eamv3"].forcing_sets] == ["amip-hist"]
    assert catalog["ace2-eamv3"].forcing_sets[0].window == (
        "1994-01-01T00:00:00",
        "2014-12-31T18:00:00",
    )

    assert catalog["hiro-ace"].stochastic is True
    assert [fs.name for fs in catalog["hiro-ace"].forcing_sets] == [
        "amip-hist",
        "amip-oos",
    ]


def test_example_registry_requires_playground_env_for_placeholders(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import aimip

    example_reg = pathlib.Path(aimip.__file__).resolve().parent / "registry.example.json"
    monkeypatch.setenv("AIMIP_REGISTRY", str(example_reg))
    monkeypatch.delenv("AIMIP_PLAYGROUND", raising=False)

    with pytest.raises(RegistryError, match="AIMIP_PLAYGROUND"):
        load_registry()


def test_cli_models_table_and_json(
    playground_env: dict[str, object],
    capsys: pytest.CaptureFixture[str],
) -> None:
    rc = main(["models"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "ace2-era5" in out
    assert "ace2s-shield-plus" in out
    assert "amip-hist" in out
    assert "amip-oos" in out

    rc_json = main(["models", "--json"])
    assert rc_json == 0
    from conftest import parse_cli_json

    parsed = parse_cli_json(capsys.readouterr().out)
    assert len(parsed) == 3
    era5_entry = next(item for item in parsed if item["name"] == "ace2-era5")
    assert len(era5_entry["available_ic_timestamps"]) == 6


def test_container_visible_input_roots_vs_writable_output_roots(
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Visible input roots include read-only mounts, while writable output roots are restricted to run roots."""
    import os

    from aimip.validators import (
        get_allowed_run_roots,
        get_visible_input_roots,
        validate_visible_input_path,
    )

    aimip_home = tmp_path / "prefix"
    aimip_data = aimip_home / "data"
    caller_runs = tmp_path / "scratch_caller" / "runs"
    ro_archive = tmp_path / "work_shared_ro"
    unmounted_dir = tmp_path / "other_host_dir"
    for d in (aimip_home, aimip_data, caller_runs, ro_archive, unmounted_dir):
        d.mkdir(parents=True, exist_ok=True)

    monkeypatch.setenv("AIMIP_HOME", str(aimip_home))
    monkeypatch.setenv("AIMIP_PLAYGROUND", str(aimip_home))
    monkeypatch.setenv("AIMIP_DATA", str(aimip_data))
    monkeypatch.setenv("AIMIP_DEFAULT_OUT_ROOT", str(caller_runs))
    monkeypatch.setenv("AIMIP_EXTRA_RO_BINDS", f"{ro_archive} /nonexistent_optional_ro")
    monkeypatch.delenv("AIMIP_RUN_ROOTS", raising=False)

    # 1. Writable output roots include ONLY caller's run root (not `/scratch/*`, `/work/*`, `AIMIP_DATA`, or `EXTRA_RO`).
    assert get_allowed_run_roots() == (f"{caller_runs}/",)
    assert validate_run_path(str(caller_runs / "exp1")) == str(caller_runs / "exp1")

    with pytest.raises(ValidationError, match="read-only"):
        validate_run_path(str(ro_archive / "cannot_write_here"))
    with pytest.raises(ValidationError, match="read-only"):
        validate_run_path(str(aimip_data / "cannot_write_here"))
    with pytest.raises(ValidationError, match="must live under"):
        validate_run_path("/scratch/other_user/exp1")

    # 2. Visible input roots include `AIMIP_DEFAULT_OUT_ROOT`, `AIMIP_HOME`, `AIMIP_DATA`, and `AIMIP_EXTRA_RO_BINDS`.
    vis = get_visible_input_roots()
    assert f"{caller_runs}/" in vis
    assert f"{aimip_home}/" in vis
    assert f"{aimip_data}/" in vis
    assert f"{ro_archive}/" in vis

    ro_ic = ro_archive / "spun_ic.nc"
    ro_ic.write_bytes(b"ic")
    assert validate_visible_input_path(str(ro_ic), label="--ic") == str(ro_ic)

    # Unmounted host path is refused naming the allowed visible roots.
    hidden_ic = unmounted_dir / "hidden_ic.nc"
    hidden_ic.write_bytes(b"ic")
    with pytest.raises(ValidationError, match="AIMIP_EXTRA_RO_BINDS"):
        validate_visible_input_path(str(hidden_ic), label="--ic")

    # Symlink inside `caller_runs` pointing to unmounted host file is ALSO refused!
    escape_link = caller_runs / "escape_ic.nc"
    os.symlink(hidden_ic, escape_link)
    with pytest.raises(ValidationError, match="symlink resolves to"):
        validate_visible_input_path(str(escape_link), label="--ic")


def test_default_registry_and_ic_list_expose_relabelled_ic_provenance(
    playground_env: dict[str, object],
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Relabelled IC origins and first-year spin-up guidance appear in `registry.default.json` and `ic list`."""
    import aimip
    from aimip.registry import build_default_registry_payload

    default_reg_file = pathlib.Path(aimip.__file__).resolve().parent / "registry.default.json"
    shipped_doc = json.loads(default_reg_file.read_text(encoding="utf-8"))
    built_doc = build_default_registry_payload()
    shipped_by_name = {m["name"]: m for m in shipped_doc["models"]}
    built_by_name = {m["name"]: m for m in built_doc["models"]}
    assert shipped_by_name.keys() == built_by_name.keys()
    for name in shipped_by_name:
        assert shipped_by_name[name]["notes"] == built_by_name[name]["notes"]

    assert "1971-01-01" in shipped_by_name["ace2-eamv3"]["notes"]
    assert "discard the first simulated year" in shipped_by_name["ace2-eamv3"]["notes"]
    assert "2014-01-01" in shipped_by_name["hiro-ace"]["notes"]
    assert "discard the first simulated year" in shipped_by_name["hiro-ace"]["notes"]
    assert "ACE2S-SHiELD-plus" in shipped_by_name["ace2-shield"]["notes"]

    # `aimip ic list -m ace2-eamv3` prints provenance in both table and `--json` output.
    assert main(["ic", "list", "-m", "ace2-eamv3"]) == 0
    txt_out = capsys.readouterr().out
    assert "Provenance:" in txt_out
    assert "1971-01-01" in txt_out
    assert "spin-up" in txt_out

    assert main(["ic", "list", "-m", "ace2-eamv3", "--json"]) == 0
    from conftest import parse_cli_json

    json_out = parse_cli_json(capsys.readouterr().out)
    assert "1971-01-01" in json_out["ic_provenance"]
    assert "discard the first simulated year" in json_out["ic_provenance"]


def test_permissions_strip_group_other_write_under_umask_002_and_preserve_077(
    tmp_path: pathlib.Path,
) -> None:
    """Created dirs and files strip `g+w`/`o+w` under umask 002 and keep 0700/0600 under umask 077."""
    import os
    import stat

    from aimip.validators import makedirs_no_group_other_write, strip_group_other_write

    old_umask = os.umask(0o002)
    try:
        nested = tmp_path / "u002_parent" / "u002_mid" / "run_dir"
        makedirs_no_group_other_write(str(nested))
        for check_dir in (tmp_path / "u002_parent", tmp_path / "u002_parent" / "u002_mid", nested):
            mode = stat.S_IMODE(check_dir.stat().st_mode)
            assert (mode & 0o022) == 0, f"{check_dir} still has g+w/o+w: {oct(mode)}"
            assert mode == 0o755

        f_002 = nested / "config.yaml"
        f_002.write_text("x: 1\n", encoding="utf-8")
        strip_group_other_write(str(f_002))
        assert stat.S_IMODE(f_002.stat().st_mode) == 0o644

        os.umask(0o077)
        priv_dir = tmp_path / "u077_parent" / "run_dir"
        makedirs_no_group_other_write(str(priv_dir))
        assert stat.S_IMODE(priv_dir.stat().st_mode) == 0o700
        f_077 = priv_dir / "config.yaml"
        f_077.write_text("x: 1\n", encoding="utf-8")
        strip_group_other_write(str(f_077))
        assert stat.S_IMODE(f_077.stat().st_mode) == 0o600
    finally:
        os.umask(old_umask)


def test_intermediate_symlinks_and_root_aliases_in_input_and_output_policies(
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Intermediate symlinks cannot escape visible/writable roots; configured root aliases work."""
    import os

    from aimip.validators import validate_visible_input_path

    aimip_home = tmp_path / "prefix"
    aimip_data = aimip_home / "data"
    run_root = tmp_path / "runs"
    outside = tmp_path / "outside"
    for d in (aimip_home, aimip_data, run_root, outside):
        d.mkdir(parents=True, exist_ok=True)

    monkeypatch.setenv("AIMIP_HOME", str(aimip_home))
    monkeypatch.setenv("AIMIP_PLAYGROUND", str(aimip_home))
    monkeypatch.setenv("AIMIP_DATA", str(aimip_data))
    monkeypatch.setenv("AIMIP_DEFAULT_OUT_ROOT", str(run_root))
    monkeypatch.delenv("AIMIP_EXTRA_RO_BINDS", raising=False)
    monkeypatch.delenv("AIMIP_RUN_ROOTS", raising=False)

    # Symlink `R/alias -> outside` with regular file `outside/ic.nc` and non-existent `new-output.nc`
    alias = run_root / "alias"
    os.symlink(outside, alias)
    outside_ic = outside / "ic.nc"
    outside_ic.write_bytes(b"ic")

    with pytest.raises(ValidationError, match="symlink resolves to"):
        validate_visible_input_path(f"{run_root}/alias/ic.nc", label="--ic")

    with pytest.raises(ValidationError, match="symlink resolves to"):
        validate_run_path(f"{run_root}/alias/new-output.nc", label="--out")

    with pytest.raises(ValidationError, match="symlink resolves to"):
        validate_run_path(f"{run_root}/alias/new_sub/new-output.nc", label="--out")

    # Dangling leaf symlink on output escaping `run_root` is also refused
    out_dangle = run_root / "dangle_out.nc"
    os.symlink(outside / "missing.nc", out_dangle)
    with pytest.raises(ValidationError, match="symlink resolves to"):
        validate_run_path(str(out_dangle), label="--out")

    # Intermediate alias pointing from `run_root` into read-only `aimip_data` is refused as read-only
    ro_alias = run_root / "ro_alias"
    os.symlink(aimip_data, ro_alias)
    with pytest.raises(ValidationError, match="read-only"):
        validate_run_path(f"{run_root}/ro_alias/new-output.nc", label="--out")

    # Nested writable root with more-specific read-only subroot: intermediate alias into RO subroot still loses
    nested_w = tmp_path / "nested_w"
    nested_ro = nested_w / "ro_sub"
    nested_ro.mkdir(parents=True)
    os.symlink(nested_ro, nested_w / "sneak")
    monkeypatch.setenv("AIMIP_DEFAULT_OUT_ROOT", str(nested_w))
    monkeypatch.setenv("AIMIP_EXTRA_RO_BINDS", str(nested_ro))
    with pytest.raises(ValidationError, match="read-only"):
        validate_run_path(f"{nested_w}/sneak/new-output.nc", label="--out")

    # Symlink loop raises clean `ValidationError`
    loop_link = nested_w / "loop"
    os.symlink(loop_link, loop_link)
    with pytest.raises(ValidationError, match="symlink loop"):
        validate_run_path(f"{nested_w}/loop/new-output.nc", label="--out")
    with pytest.raises(ValidationError, match="symlink loop"):
        validate_visible_input_path(f"{nested_w}/loop/ic.nc", label="--ic")

    # Legitimate configured root aliases (e.g. `alias_runs -> real_runs` and macOS `/var` <-> `/private/var`) work
    real_runs = tmp_path / "real_runs"
    real_runs.mkdir()
    (real_runs / "ok_ic.nc").write_bytes(b"ok")
    alias_runs = tmp_path / "alias_runs"
    os.symlink(real_runs, alias_runs)
    os.symlink(outside, real_runs / "escape")
    monkeypatch.setenv("AIMIP_DEFAULT_OUT_ROOT", str(alias_runs))
    monkeypatch.delenv("AIMIP_EXTRA_RO_BINDS", raising=False)

    assert validate_run_path(f"{alias_runs}/exp_new") == f"{alias_runs}/exp_new"
    assert validate_run_path(f"{real_runs}/exp_new") == f"{real_runs}/exp_new"
    assert validate_visible_input_path(f"{alias_runs}/ok_ic.nc") == f"{alias_runs}/ok_ic.nc"
    assert validate_visible_input_path(f"{real_runs}/ok_ic.nc") == f"{real_runs}/ok_ic.nc"
    with pytest.raises(ValidationError, match="symlink resolves to"):
        validate_run_path(f"{alias_runs}/escape/new-output.nc")
    with pytest.raises(ValidationError, match="symlink resolves to"):
        validate_visible_input_path(f"{alias_runs}/escape/ic.nc")

    real_runs_str = str(real_runs.resolve())
    if real_runs_str.startswith("/private/var/") and os.path.islink("/var"):
        var_alias_path = "/var/" + real_runs_str[len("/private/var/") :]
        monkeypatch.setenv("AIMIP_DEFAULT_OUT_ROOT", real_runs_str)
        assert validate_run_path(f"{var_alias_path}/macos_exp") == f"{var_alias_path}/macos_exp"
        assert validate_visible_input_path(f"{var_alias_path}/ok_ic.nc") == f"{var_alias_path}/ok_ic.nc"

    # Empty-registry CLI behavior: `aimip models --json` -> `[]`, unknown model -> `(none)`
    empty_reg = aimip_data / "registry.json"
    empty_reg.write_text('{"schema_version": 3, "generated_by": "aimip setup", "models": []}\n', encoding="utf-8")
    monkeypatch.delenv("AIMIP_REGISTRY", raising=False)
    assert main(["models", "--json"]) == 0
    from conftest import parse_cli_json

    assert parse_cli_json(capsys.readouterr().out) == []
    assert main(["ic", "list", "-m", "ace2-eamv3"]) == 2
    assert "(none)" in capsys.readouterr().err


def test_user_agent_version_slurm_token_error_emu_floor_example_registry_and_cli_help(
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify runtime User-Agent version header, Slurm token error message, walltime floor, example registry v3, and mode-neutral CLI/IC strings."""
    import io
    import urllib.request

    import aimip
    import aimip.emu as emu_mod
    import aimip.ic as ic_mod
    import aimip.setup_data as setup_data_mod
    from aimip.cli import build_parser

    # 1. HTTP User-Agent in Zenodo download uses `aimip-setup/<__version__>`, not `0.2`.
    seen_uas: list[str] = []

    class _FakeResp(io.BytesIO):
        status = 200

        def __enter__(self) -> _FakeResp:
            return self

        def __exit__(self, *exc: object) -> None:
            self.close()

    def _fake_urlopen(req: urllib.request.Request, timeout: float = 60.0) -> _FakeResp:
        seen_uas.append(req.get_header("User-agent") or "")
        return _FakeResp(b"abc")

    monkeypatch.delenv("AIMIP_ZENODO_MIRROR", raising=False)
    monkeypatch.setattr(urllib.request, "urlopen", _fake_urlopen)
    dst = tmp_path / "zenodo.bin"
    setup_data_mod.download_zenodo_artifact(
        record="17065758",
        filename="test.bin",
        url="https://zenodo.org/records/17065758/files/test.bin",
        dst_path=str(dst),
        expected_size=3,
    )
    assert seen_uas == [f"aimip-setup/{aimip.__version__}"]

    # 2. `validate_slurm_token` error message includes `+` matching `_SLURM_TOKEN_RE`.
    with pytest.raises(ValidationError) as exc_info:
        validate_slurm_token("bad token", "account")
    assert "[A-Za-z0-9_.:+-]" in str(exc_info.value)

    # 3. `EMU_MIN_WALLTIME` aligns with the 45-minute (2700s) floor (`3.0 * 900`).
    assert emu_mod.EMU_MIN_WALLTIME == 2700

    # 4. `registry.example.json` is schema_version 3 with all 14 canonical models.
    pkg_dir = pathlib.Path(aimip.__file__).resolve().parent
    example_doc = json.loads((pkg_dir / "registry.example.json").read_text(encoding="utf-8"))
    default_doc = json.loads((pkg_dir / "registry.default.json").read_text(encoding="utf-8"))
    assert example_doc["schema_version"] == 3
    assert [m["name"] for m in example_doc["models"]] == [m["name"] for m in default_doc["models"]]

    # 5. CLI help and playground collision error are container/native-neutral and cover all families.
    help_text = build_parser().format_help()
    assert "ACE-family models" not in help_text
    assert "Internal container entrypoint" not in help_text
    with pytest.raises(Exception) as pg_exc:
        ic_mod._ensure_outside_playground("/opt/aimip/out.nc", "/opt/aimip", "IC output")
    assert "/scratch or /work" not in str(pg_exc.value)
    assert "configured run root" in str(pg_exc.value)


def test_per_test_allowed_roots_isolate_outside_paths_even_under_scratch_basetemp(
    tmp_path: pathlib.Path,
    allow_tmp_run_root: pathlib.Path,
) -> None:
    """`allow_tmp_run_root` scopes allowed roots exclusively to `tmp_path/scratch` regardless of basetemp parent."""
    from aimip.validators import (
        get_allowed_run_roots,
        get_visible_input_roots,
        validate_visible_input_path,
    )

    assert get_allowed_run_roots() == (f"{allow_tmp_run_root}/",)
    assert get_visible_input_roots() == (f"{allow_tmp_run_root}/",)

    inside_file = allow_tmp_run_root / "ok.nc"
    inside_file.write_bytes(b"ok")
    assert validate_run_path(str(allow_tmp_run_root / "exp1")) == str(allow_tmp_run_root / "exp1")
    assert validate_visible_input_path(str(inside_file), label="--ic") == str(inside_file)

    outside_dir = tmp_path / "outside"
    outside_dir.mkdir(exist_ok=True)
    outside_file = outside_dir / "unmounted.nc"
    outside_file.write_bytes(b"no")

    with pytest.raises(ValidationError, match="must live under"):
        validate_run_path(str(outside_dir / "exp_bad"))
    with pytest.raises(ValidationError, match="will not be visible"):
        validate_visible_input_path(str(outside_file), label="--ic")
