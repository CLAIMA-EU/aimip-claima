"""DLESyM / ArchesClimate-SSP families: request checks, prepare -> job.sbatch/manifest, job-time integrity,
outcome recording, and the monthly writer layout read by `aimip summary`. The second environment is replaced by a fake
interpreter (`$AIMIP_EMU_PYTHON`), so these tests need neither torch nor network."""
from __future__ import annotations

import json
import pathlib
import stat

import numpy as np
import pytest

from aimip.emu_catalog import registry_entries
from aimip.errors import ConfigError, ValidationError
from aimip.registry import _parse_model_entry

from conftest import sha256_path

FAKE_PY = """#!/bin/bash
[ "$1 $2 $3" = "-I -m aimip.emu_job" ] || { echo "bad argv $*" >&2; exit 9; }
mode=$4; out=$5
[ -n "$AIMIP_EMU_TMP" ] || { echo "no AIMIP_EMU_TMP" >&2; exit 8; }
case "$mode" in
  stage) printf 'fake-era5-state' > "$out/ic_era5.pt";;
  run) printf '{"fake": true, "rc": %s}' "${AIMIP_FAKE_RC:-0}" > "$out/emu-result.json"; exit "${AIMIP_FAKE_RC:-0}";;
  *) exit 7;;
esac
"""


@pytest.fixture
def emu_env(tmp_path, allow_tmp_run_root, monkeypatch):
    data = tmp_path / "data"
    pg = tmp_path / "pg"
    (pg / "bin").mkdir(parents=True)
    (pg / "bin" / "aimip").write_text("#!/bin/sh\nexit 0\n")
    fake = tmp_path / "fakepy"
    fake.write_text(FAKE_PY)
    fake.chmod(fake.stat().st_mode | stat.S_IXUSR)
    monkeypatch.setenv("AIMIP_EMU_PYTHON", str(fake))
    monkeypatch.setenv("AIMIP_PLAYGROUND", str(pg))
    monkeypatch.delenv("AIMIP_FAKE_RC", raising=False)
    specs = {}
    lock_files = {}
    (data / "code" / "ArchesClimate-SSP").mkdir(parents=True, exist_ok=True)
    for entry in registry_entries(str(data)):
        for rel in entry["required_files"]:
            p = data / rel
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_bytes(f"fake {rel}".encode())
            lock_files[rel] = {"size": p.stat().st_size, "sha256": sha256_path(p),
                               "source": {"kind": "hf", "repo": "x/y", "path": rel, "revision": "0" * 40}}
        ckpt = pathlib.Path(entry["checkpoint"])
        entry = dict(entry, checkpoint_sha256=sha256_path(ckpt))
        specs[entry["name"]] = _parse_model_entry(entry, str(pg), str(data))
    lock = tmp_path / "test.lock"
    lock.write_text(json.dumps({"schema_version": 1, "files": lock_files}))
    monkeypatch.setenv("AIMIP_LOCK", str(lock))   # prepare verifies every model file against aimip.lock
    runs = allow_tmp_run_root / "runs"
    runs.mkdir()
    return {"data": data, "pg": pg, "specs": specs, "runs": runs}


def _prepare(env, name, out, **kw):
    from aimip.runner import SlurmOptions, prepare_run

    args = dict(model=env["specs"][name], start_iso="1995-01-01T00:00:00", steps=8, out_dir=str(out),
                mode="dry-run", slurm_opts=SlurmOptions(), data_dir=str(env["data"]), playground_dir=str(env["pg"]))
    args.update(kw)
    return prepare_run(**args)


def test_request_rules_without_disk(emu_env):
    from aimip.emu import precheck_emu_request

    s = emu_env["specs"]
    ok = precheck_emu_request(s["dlesym-v1.1"], start_iso="1995-01-01T00:00:00", steps=4 * 365, members=3,
                              forcing_set_name=None, ic_path=None)
    assert ok["end"] == "1996-01-01T00:00:00" and ok["members"] == 3
    for kw, msg in (
        (dict(start_iso="1930-01-01T00:00:00"), "outside the ERA5 initial states"),
        (dict(start_iso="1995-01-01T03:00:00"), "00, 06, 12 or 18 UTC"),
        (dict(members=2), "deterministic"),
        (dict(ic_path="/scratch/x/restart.nc"), "cannot be continued yet"),
        (dict(forcing_set_name="ssp585"), "no forcing sets"),
    ):
        args = dict(start_iso="1995-01-01T00:00:00", steps=8, members=None, forcing_set_name=None, ic_path=None)
        args.update(kw)
        with pytest.raises(ConfigError, match=msg):
            precheck_emu_request(s["dlesym-v1.0"], **args)
    a = precheck_emu_request(s["archesclimate-ssp"], start_iso="2015-01-01T00:00:00", steps=12, members=None,
                             forcing_set_name=None, ic_path=None)
    assert (a["scenario"], a["end"]) == ("ssp434", "2016-02-01T00:00:00")   # first output 2015-03, 12th 2016-02
    for kw, msg in (
        (dict(start_iso="2020-01-01T00:00:00"), "starts from the scenario's 2015 state"),
        (dict(steps=1030), "at most 1029"),
        (dict(forcing_set_name="ssp245"), "choose one of ssp434, ssp585"),
        (dict(members=2), "one member"),
    ):
        args = dict(start_iso="2015-01-01T00:00:00", steps=12, members=None, forcing_set_name=None, ic_path=None)
        args.update(kw)
        with pytest.raises(ConfigError, match=msg):
            precheck_emu_request(s["archesclimate-ssp"], **args)


def test_dlesym_prepare_exec_and_tamper_refusal(emu_env):
    from aimip.runner import exec_fme

    out = emu_env["runs"] / "d11"
    m = _prepare(emu_env, "dlesym-v1.1", out, members=2, seed=3)
    assert m["family"] == "e2s-dlesym" and m["member_seeds"] == [3, 4] and m["steps"] == 8
    assert m["end_time"] == "1995-01-03T00:00:00" and m["ic_path"] == str(out / "ic_era5.pt")
    assert m["ic_sha256"] == sha256_path(out / "ic_era5.pt")
    assert m["checkpoint_path"].endswith("checkpoints/v1.1/atmos/atmos_model_0.mdlus")
    assert len(m["package_files_sha256"]) == len(emu_env["specs"]["dlesym-v1.1"].required_files)
    sb = (out / "job.sbatch").read_text()
    assert sb.splitlines()[1] == "# aimip-generated v1" and f'exec-fme "{out}"' in sb
    job = json.loads((out / "config.yaml").read_text())
    assert job["version"] == "v1.1" and job["package_dir"].endswith("models/dlesym-v1")

    done = exec_fme(str(out))
    assert done["rc"] == 0 and done["result"] == {"fake": True, "rc": 0} and "finished" in done
    assert not (out / ".aimip-tmp").exists()

    # a changed package file is refused at job time, before anything runs
    out2 = emu_env["runs"] / "d11b"
    _prepare(emu_env, "dlesym-v1.1", out2)
    hpx = emu_env["data"] / "models/dlesym-v1/hpx_lat.npy"
    hpx.write_bytes(b"tampered")
    with pytest.raises(ValidationError, match="SHA-256 mismatch at job time"):
        exec_fme(str(out2))
    # a completed run directory is not overwritten
    with pytest.raises(ConfigError, match="Refusing to overwrite completed run"):
        _prepare(emu_env, "dlesym-v1.1", out)


def test_arches_prepare_records_failure_rc(emu_env, monkeypatch):
    from aimip.runner import exec_fme

    out = emu_env["runs"] / "arches"
    m = _prepare(emu_env, "archesclimate-ssp", out, start_iso="2015-01-01T00:00:00", steps=24,
                 forcing_set_name="ssp585")
    assert m["forcing_set"]["name"] == "ssp585" and m["end_time"] == "2017-02-01T00:00:00"
    assert m["forcing_files"] and all("ssp585" in r["path"] and r["size"] > 0 for r in m["forcing_files"])
    job = json.loads((out / "config.yaml").read_text())
    assert job["domain"] == "test" and job["first_output_month"] == "2015-03"
    monkeypatch.setenv("AIMIP_FAKE_RC", "3")
    with pytest.raises(SystemExit) as exc:
        exec_fme(str(out))
    assert exc.value.code == 3
    rec = json.loads((out / "aimip-manifest.json").read_text())
    assert rec["rc"] == 3 and "rc=3" in rec["error"]

    # Unstable DLESyM rollout uses dedicated exit code 5 (distinct from approval error 4).
    out_unstable = emu_env["runs"] / "d11_unstable"
    _prepare(emu_env, "dlesym-v1.1", out_unstable, members=1)
    monkeypatch.setenv("AIMIP_FAKE_RC", "5")
    with pytest.raises(SystemExit) as exc_u:
        exec_fme(str(out_unstable))
    assert exc_u.value.code == 5
    rec_u = json.loads((out_unstable / "aimip-manifest.json").read_text())
    assert rec_u["rc"] == 5 and "unstable" in rec_u["error"]


def test_cli_refuses_ic_commands_and_missing_runtime(emu_env, monkeypatch, capsys):
    from aimip.cli import main
    from aimip.emu import emu_python

    assert main(["ic", "list", "--model", "dlesym-v1.1"]) != 0
    assert "works with the fme models only" in capsys.readouterr().err
    monkeypatch.setenv("AIMIP_EMU_PYTHON", "/nonexistent/python")
    with pytest.raises(ConfigError, match="environment is missing"):
        emu_python()


def test_monthly_writer_layout_is_read_by_summary(tmp_path, allow_tmp_run_root):
    from aimip.emu_job import MonthlyWriter
    from aimip.status import summarize_run

    run = allow_tmp_run_root / "w"
    lat = np.array([-45.0, 45.0])
    lon = np.array([0.0, 180.0])
    w = MonthlyWriter(str(run / "atmosphere" / "monthly_mean_predictions.nc"), ["t2m", "z500"], lat, lon, 2)
    for member in (0, 1):
        for k, month in enumerate((np.datetime64("1995-01", "M"), np.datetime64("1995-02", "M"))):
            w.write(member, month, {"t2m": np.full((2, 2), 280.0 + k + 10 * member, "f4"),
                                    "z500": np.zeros((2, 2), "f4")}, 124)
    w.close()
    s = summarize_run(str(run))
    rows = s["variables"]["t2m"]["series"]
    assert [r["month"] for r in rows] == ["1995-01", "1995-02"]
    assert rows[1]["members"] == [281.0, 291.0] and rows[0]["counts"] == 124


def test_config_binding_lock_check_and_no_fetch(emu_env):
    """config.yaml edits after prepare are refused; a changed secondary model file is refused
    before staging; --no-fetch never downloads the ERA5 state."""
    from aimip.runner import exec_fme

    for field, value in (("package_dir", "/scratch/elsewhere/models/dlesym-v1"), ("members", 64),
                         ("start", "2000-01-01T00:00:00")):
        out = emu_env["runs"] / f"tamper-{field}"
        _prepare(emu_env, "dlesym-v1.1", out)
        cfg = json.loads((out / "config.yaml").read_text())
        cfg[field] = value
        (out / "config.yaml").write_text(json.dumps(cfg, indent=2) + "\n")
        with pytest.raises(ValidationError, match="changed after"):
            exec_fme(str(out))
        assert not (out / "emu-result.json").exists()

    out = emu_env["runs"] / "tamper-arches"
    _prepare(emu_env, "archesclimate-ssp", out, start_iso="2015-01-01T00:00:00", steps=3)
    m = json.loads((out / "aimip-manifest.json").read_text())
    from aimip.emu import _check_job_matches_manifest

    job = json.loads((out / "config.yaml").read_text())
    _check_job_matches_manifest(job, m)                     # consistent as prepared
    for field, value in (("domain", "test"), ("model_dir", str(emu_env["data"])), ("first_output_month", "2015-02")):
        with pytest.raises(ValidationError, match=field.split("_")[0]):
            _check_job_matches_manifest(dict(job, **{field: value}), m)

    ocean = emu_env["data"] / "models/dlesym-v1/checkpoints/v1.1/ocean/ocean_model_0.mdlus"
    ocean.write_bytes(b"swapped ocean weights")
    with pytest.raises(ValidationError, match="do not match aimip.lock"):
        _prepare(emu_env, "dlesym-v1.1", emu_env["runs"] / "lock-mismatch")
    assert not (emu_env["runs"] / "lock-mismatch" / "ic_era5.pt").exists()

    with pytest.raises(ConfigError, match="--no-fetch forbids downloads"):
        _prepare(emu_env, "dlesym-v1.0", emu_env["runs"] / "no-fetch-run", auto_fetch=False)
    assert not (emu_env["runs"] / "no-fetch-run" / "ic_era5.pt").exists()


def test_arches_loads_the_exact_checkpoint_file_and_resolves_fetched_code_dir(tmp_path, monkeypatch):
    """The job never calls load_module(<dir>) (newest *.ckpt wins); it passes the verified file and resolves $AIMIP_DATA/code/ArchesClimate-SSP."""
    import inspect

    import aimip.emu_job as ej

    src = inspect.getsource(ej.run_arches)
    assert "load_module(" not in src.replace("Not load_module()", "")
    assert 'init_from_ckpt(job["checkpoint"]' in src

    repo_root = pathlib.Path(__file__).resolve().parent.parent
    dockerfile = (repo_root / "container" / "Dockerfile").read_text(encoding="utf-8")
    native_sh = (repo_root / "install" / "install-native.sh").read_text(encoding="utf-8")
    assert "ArchesClimate-SSP.git" not in dockerfile
    assert "archesclimate.pth" not in dockerfile
    assert "ArchesClimate-SSP.git" not in native_sh
    assert "archesclimate.pth" not in native_sh
    assert (repo_root / "THIRD_PARTY.md").is_file()

    # Old baked-in location ($AIMIP_HOME/code/ArchesClimate-SSP) is not used; missing dir raises cleanly.
    old_baked = tmp_path / "prefix" / "code" / "ArchesClimate-SSP"
    old_baked.mkdir(parents=True)
    monkeypatch.delenv("AIMIP_ARCHES_CODE", raising=False)
    monkeypatch.delenv("AIMIP_DATA", raising=False)
    monkeypatch.setenv("AIMIP_HOME", str(tmp_path / "prefix"))
    with pytest.raises(FileNotFoundError, match="aimip fetch -m archesclimate-ssp"):
        ej._arches_code_dir()

    fetched_code = tmp_path / "prefix" / "data" / "code" / "ArchesClimate-SSP"
    fetched_code.mkdir(parents=True)
    assert ej._arches_code_dir() == str(fetched_code)


def test_arches_archive_extraction_and_tree_digest_verification(tmp_path, monkeypatch):
    """Fake ArchesClimate-SSP archive tests exact sha256, regenerated archive tree-digest fallback, tamper refusal, and tar traversal guard."""
    import gzip
    import hashlib
    import io
    import os
    import tarfile

    from aimip.setup_data import (
        compute_arches_tree_digest,
        extract_and_verify_arches_archive,
        verify_arches_code_dir,
    )

    def _make_archive(
        out_tar_gz: pathlib.Path,
        entries: list[tuple[str, bytes | str]],
        *,
        mtime: int = 1700000000,
        compresslevel: int = 6,
    ) -> tuple[str, int]:
        raw_tar = io.BytesIO()
        with tarfile.open(fileobj=raw_tar, mode="w") as tf:
            for name, payload in entries:
                info = tarfile.TarInfo(name=name)
                info.mtime = mtime
                if isinstance(payload, str):
                    info.type = tarfile.SYMTYPE
                    info.linkname = payload
                    info.mode = 0o777
                    tf.addfile(info)
                else:
                    info.type = tarfile.REGTYPE
                    info.size = len(payload)
                    info.mode = 0o666  # extractor must strip group/other write bits
                    tf.addfile(info, io.BytesIO(payload))
        with gzip.GzipFile(filename="", mode="wb", fileobj=out_tar_gz.open("wb"), compresslevel=compresslevel, mtime=mtime) as gz:
            gz.write(raw_tar.getvalue())
        return sha256_path(out_tar_gz), out_tar_gz.stat().st_size

    prefix = "ArchesClimate-SSP-06ce82dca2414db4f1b3d6940b61c5cd723f4620"
    good_entries: list[tuple[str, bytes | str]] = [
        (f"{prefix}/ArchesClimate/__init__.py", b"__all__ = []\n"),
        (f"{prefix}/ArchesClimate/model/base_module.py", b"class Base: pass\n"),
        (f"{prefix}/stats/cmip_stats_new_ozone.pt", "../nonexistent/upstream_stats.pt"),
    ]
    # Compute expected tree digest directly from the specification:
    # sha256 over sorted relpaths of `relpath + NUL + (sha256(file) | "LINK:" + target) + "\n"`.
    manual = hashlib.sha256()
    for rel, val in sorted(
        [
            ("ArchesClimate/__init__.py", hashlib.sha256(b"__all__ = []\n").hexdigest()),
            ("ArchesClimate/model/base_module.py", hashlib.sha256(b"class Base: pass\n").hexdigest()),
            ("stats/cmip_stats_new_ozone.pt", "LINK:../nonexistent/upstream_stats.pt"),
        ]
    ):
        manual.update(f"{rel}\0{val}\n".encode("utf-8"))
    want_tree_sha = manual.hexdigest()

    tar1 = tmp_path / "arches_v1.tar.gz"
    sha1, size1 = _make_archive(tar1, good_entries, mtime=1700000000, compresslevel=9)

    # 1. Exact archive sha256 + tree digest accepted; group/other write stripped; broken symlink preserved.
    dest1 = tmp_path / "data1"
    res1 = extract_and_verify_arches_archive(
        str(tar1),
        str(dest1 / "code" / "ArchesClimate-SSP"),
        expected_archive_sha256=sha1,
        expected_archive_size=size1,
        expected_tree_sha256=want_tree_sha,
        expected_tree_files=3,
    )
    assert res1["tree_sha256"] == want_tree_sha and res1["files"] == 3
    init_py = dest1 / "code" / "ArchesClimate-SSP" / "ArchesClimate" / "__init__.py"
    assert init_py.is_file()
    assert (init_py.stat().st_mode & 0o022) == 0
    sym = dest1 / "code" / "ArchesClimate-SSP" / "stats" / "cmip_stats_new_ozone.pt"
    assert sym.is_symlink() and sym.readlink() == pathlib.Path("../nonexistent/upstream_stats.pt")
    assert compute_arches_tree_digest(str(dest1 / "code" / "ArchesClimate-SSP")) == (want_tree_sha, 3)
    verify_arches_code_dir(
        str(dest1),
        expected_tree_sha256=want_tree_sha,
        expected_tree_files=3,
    )

    # 2. Regenerated GitHub archive (different gzip stream/mtime -> different archive sha256 & size) with identical tree is accepted!
    tar2 = tmp_path / "arches_regenerated.tar.gz"
    sha2, size2 = _make_archive(tar2, list(reversed(good_entries)), mtime=1800000000, compresslevel=1)
    assert sha2 != sha1
    dest2 = tmp_path / "data2"
    res2 = extract_and_verify_arches_archive(
        str(tar2),
        str(dest2 / "code" / "ArchesClimate-SSP"),
        expected_archive_sha256=sha1,
        expected_archive_size=size1,
        expected_tree_sha256=want_tree_sha,
        expected_tree_files=3,
    )
    assert res2["tree_sha256"] == want_tree_sha

    # 3. Tampered file content, extra file, or modified symlink target are all refused and leave no target directory!
    for idx, bad_entries in enumerate(
        [
            [
                (f"{prefix}/ArchesClimate/__init__.py", b"__all__ = ['tampered']\n"),
                (f"{prefix}/ArchesClimate/model/base_module.py", b"class Base: pass\n"),
                (f"{prefix}/stats/cmip_stats_new_ozone.pt", "../nonexistent/upstream_stats.pt"),
            ],
            [*good_entries, (f"{prefix}/extra.py", b"x = 1\n")],
            [
                (f"{prefix}/ArchesClimate/__init__.py", b"__all__ = []\n"),
                (f"{prefix}/ArchesClimate/model/base_module.py", b"class Base: pass\n"),
                (f"{prefix}/stats/cmip_stats_new_ozone.pt", "/etc/passwd"),
            ],
            [(f"{prefix}/../escape.py", b"boom\n")],
        ]
    ):
        bad_tar = tmp_path / f"bad_{idx}.tar.gz"
        _make_archive(bad_tar, bad_entries)
        bad_dst = tmp_path / f"bad_dest_{idx}" / "code" / "ArchesClimate-SSP"
        with pytest.raises(ValidationError):
            extract_and_verify_arches_archive(
                str(bad_tar),
                str(bad_dst),
                expected_archive_sha256=sha1,
                expected_archive_size=size1,
                expected_tree_sha256=want_tree_sha,
                expected_tree_files=3,
            )
        assert not bad_dst.exists()

    # 4. model_has_missing_files reports code/ArchesClimate-SSP until the fetched directory is present.
    from aimip.runner import model_has_missing_files

    data_check = tmp_path / "data_missing_check"
    arches_entry = next(e for e in registry_entries(str(data_check)) if e["name"] == "archesclimate-ssp")
    for rel in arches_entry["required_files"]:
        p = data_check / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(b"stub")
    spec = _parse_model_entry(
        dict(arches_entry, checkpoint_sha256=sha256_path(pathlib.Path(arches_entry["checkpoint"]))),
        str(tmp_path / "pg_check"),
        str(data_check),
    )
    assert model_has_missing_files(spec, data_dir=str(data_check)) == ["code/ArchesClimate-SSP"]
    extract_and_verify_arches_archive(
        str(tar1),
        str(data_check / "code" / "ArchesClimate-SSP"),
        expected_archive_sha256=sha1,
        expected_archive_size=size1,
        expected_tree_sha256=want_tree_sha,
        expected_tree_files=3,
    )
    assert model_has_missing_files(spec, data_dir=str(data_check)) == []

    # 5. Partial POSIX `os.write` drains every chunk completely to exact output bytes, while a zero-progress write or
    #    post-extraction tree-digest mismatch (even with a valid archive hash) refuses publication and preserves any prior tree!
    real_os_write = os.write

    def _half_write(fd: int, buf: bytes | memoryview) -> int:
        mv = memoryview(buf)
        n = max(1, len(mv) // 2)
        return real_os_write(fd, mv[:n])

    dest_half = tmp_path / "data_half" / "code" / "ArchesClimate-SSP"
    with monkeypatch.context() as mp:
        mp.setattr(os, "write", _half_write)
        res_half = extract_and_verify_arches_archive(
            str(tar1),
            str(dest_half),
            expected_archive_sha256=sha1,
            expected_archive_size=size1,
            expected_tree_sha256=want_tree_sha,
            expected_tree_files=3,
        )
    assert res_half["tree_sha256"] == want_tree_sha
    assert (dest_half / "ArchesClimate" / "__init__.py").read_bytes() == b"__all__ = []\n"
    assert (dest_half / "ArchesClimate" / "model" / "base_module.py").read_bytes() == b"class Base: pass\n"

    # Zero-progress `os.write` on a pre-existing valid tree raises OSError, removes staging, and preserves the existing tree.
    with monkeypatch.context() as mp:
        mp.setattr(os, "write", lambda _fd, _buf: 0)
        with pytest.raises(OSError, match="[Ss]hort write"):
            extract_and_verify_arches_archive(
                str(tar1),
                str(dest_half),
                expected_archive_sha256=sha1,
                expected_archive_size=size1,
                expected_tree_sha256=want_tree_sha,
                expected_tree_files=3,
            )
    assert compute_arches_tree_digest(str(dest_half)) == (want_tree_sha, 3)

    # Even when `archive_sha256` + `archive_size` match `tar1`, a mismatched extracted tree digest MUST be refused!
    fresh_unused = tmp_path / "data_tree_required" / "code" / "ArchesClimate-SSP"
    with pytest.raises(ValidationError):
        extract_and_verify_arches_archive(
            str(tar1),
            str(fresh_unused),
            expected_archive_sha256=sha1,
            expected_archive_size=size1,
            expected_tree_sha256="0" * 64,
            expected_tree_files=3,
        )
    assert not fresh_unused.exists()


def test_unstable_month_is_detected():
    """A month with NaN or an out-of-band global-mean t2m stops the DLESyM run."""
    from aimip.emu_job import Unstable, _check_month

    lat = np.array([-45.0, 45.0])
    good = np.full((2, 2, 2), 285.0, "f4")
    _check_month(["t2m", "z500"], good, lat, "1995-01", 0)
    nan = good.copy()
    nan[1, 0, 0] = np.nan
    with pytest.raises(Unstable, match="non-finite values in z500"):
        _check_month(["t2m", "z500"], nan, lat, "2001-07", 0)
    hot = good.copy()
    hot[0] = 400.0
    with pytest.raises(Unstable, match="outside 200-330 K") as exc:
        _check_month(["t2m", "z500"], hot, lat, "2001-07", 1)
    assert exc.value.month == "2001-07" and exc.value.member == 1
    _check_month(["sst"], np.full((1, 2, 2), 400.0, "f4"), lat, "1995-01", 0)   # band only for t2m
