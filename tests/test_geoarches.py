"""ArchesWeather / ArchesWeatherGen (`geoarches` family): request rules, prepare -> job config/manifest, job-time binding,
the pinned geoarches source (download check, safe unpacking, tree digest), checkpoint extraction from the Zenodo zip,
the pressure-level writer, and the recipes that install everything. The emulator environment is the fake interpreter
of test_emu.py, so no torch, geoarches or network is needed."""
from __future__ import annotations

import hashlib
import io
import json
import pathlib
import tarfile
import zipfile

import numpy as np
import pytest

from aimip.errors import ConfigError, ValidationError

from test_emu import _prepare, emu_env  # noqa: F401  (fixture)

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
AW_START = "1978-09-30T00:00:00"


def test_geoarches_request_rules_without_disk(emu_env):  # noqa: F811
    from aimip.emu import precheck_emu_request

    s = emu_env["specs"]
    ok = precheck_emu_request(s["archesweather"], start_iso=AW_START, steps=365, members=None, forcing_set_name=None,
                              ic_path=None)
    assert (ok["scenario"], ok["end"], ok["members"]) == ("aimip", "1979-09-30T00:00:00", 1)
    full = precheck_emu_request(s["archesweathergen"], start_iso=AW_START, steps=16894, members=3,
                                forcing_set_name="aimip", ic_path=None)
    assert full["end"] == "2024-12-31T00:00:00" and full["members"] == 3
    for name, kw, msg in (
        ("archesweather", dict(start_iso="1979-01-01T00:00:00"), "1978-09-30 only"),
        ("archesweather", dict(steps=16895), "at most 16894 daily steps"),
        ("archesweather", dict(members=2), "deterministic"),
        ("archesweathergen", dict(forcing_set_name="aimip-p4k"), "choose one of aimip"),
        ("archesweathergen", dict(ic_path="/scratch/x/restart.nc"), "cannot be continued yet"),
    ):
        args = dict(start_iso=AW_START, steps=10, members=None, forcing_set_name=None, ic_path=None)
        args.update(kw)
        with pytest.raises(ConfigError, match=msg):
            precheck_emu_request(s[name], **args)


def test_geoarches_prepare_exec_and_job_binding(emu_env, monkeypatch, tmp_path):  # noqa: F811
    from aimip import emu
    from aimip.emu_catalog import AW_CKPT, AW_IC, AW_SEEDS
    from aimip.runner import exec_fme

    # The geoarches source is part of the install (<env parent>/geoarches); without it, preparing a run stops with a
    # reinstall hint, and a tree that differs from the pinned commit is refused.
    import os
    import shutil

    monkeypatch.delenv("AIMIP_GEOARCHES_CODE", raising=False)
    env_py = tmp_path / "pfx" / "venv-emu" / "bin" / "python"
    env_py.parent.mkdir(parents=True)
    shutil.copy2(os.environ["AIMIP_EMU_PYTHON"], env_py)
    monkeypatch.setenv("AIMIP_EMU_PYTHON", str(env_py))
    with pytest.raises(ConfigError, match="geoarches source is not installed"):
        _prepare(emu_env, "archesweather", emu_env["runs"] / "nocode", start_iso=AW_START, steps=4)
    (tmp_path / "pfx" / "geoarches" / "geoarches").mkdir(parents=True)
    with pytest.raises(ValidationError, match="does not match the pinned commit"):
        _prepare(emu_env, "archesweather", emu_env["runs"] / "badcode", start_iso=AW_START, steps=4)
    code = tmp_path / "geoarches-src"
    (code / "geoarches").mkdir(parents=True)
    monkeypatch.setenv("AIMIP_GEOARCHES_CODE", str(code))

    data = emu_env["data"]
    out = emu_env["runs"] / "aw"
    m = _prepare(emu_env, "archesweather", out, start_iso=AW_START, steps=62)
    assert (m["family"], m["step_unit"], m["end_time"]) == ("geoarches", "day", "1978-12-01T00:00:00")
    assert m["forcing_set"]["name"] == "aimip" and m["forcing_files"] == []
    assert m["ic_path"] == str(data / AW_IC) and "1978-09-30" in m["ic_provenance"]
    assert len(m["package_files_sha256"]) == len(emu_env["specs"]["archesweather"].required_files) == 12
    job = json.loads((out / "config.yaml").read_text())
    assert job["kind"] == "mx4" and job["gen_dir"] is None and job["residual_stats_file"] is None
    assert [pathlib.Path(d).name for d in job["seed_dirs"]] == list(AW_SEEDS)
    assert job["checkpoint"] == str(pathlib.Path(job["seed_dirs"][0]) / AW_CKPT)
    done = exec_fme(str(out))
    assert done["rc"] == 0 and done["result"] == {"fake": True, "rc": 0}

    out_g = emu_env["runs"] / "awgen"
    mg = _prepare(emu_env, "archesweathergen", out_g, start_iso=AW_START, steps=31, members=2, seed=7)
    assert mg["member_seeds"] == [7, 8] and "sampler noise seed" in mg["ensemble_note"]
    assert len(mg["package_files_sha256"]) == 15
    job_g = json.loads((out_g / "config.yaml").read_text())
    assert job_g["kind"] == "gen" and job_g["checkpoint"] == str(pathlib.Path(job_g["gen_dir"]) / AW_CKPT)

    # Every job field must agree with the manifest and the pinned files (checked before anything runs).
    emu._check_job_matches_manifest(job_g, mg)
    for field, value in (("kind", "mx4"), ("seed_dirs", job_g["seed_dirs"][:3]), ("gen_dir", str(tmp_path)),
                         ("residual_stats_file", None), ("ic_file", str(tmp_path / "ic.nc")),
                         ("forcing_file", str(tmp_path / "f.nc"))):
        bad = dict(job_g, **{field: value})
        with pytest.raises(ValidationError, match="does not match aimip-manifest.json"):
            emu._check_job_matches_manifest(bad, mg)
    # ... and a changed model file is refused at job time.
    (pathlib.Path(job_g["gen_dir"]) / "config.yaml").write_bytes(b"tampered")
    with pytest.raises(ValidationError, match="SHA-256 mismatch at job time"):
        exec_fme(str(out_g))


def _synthetic_geoarches_archive(path: pathlib.Path, commit: str, extra: list[tarfile.TarInfo] | None = None) -> bytes:
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tf:
        def add(name: str, data: bytes) -> None:
            ti = tarfile.TarInfo(name)
            ti.size = len(data)
            tf.addfile(ti, io.BytesIO(data))
        d = tarfile.TarInfo(f"geoarches-{commit}")
        d.type = tarfile.DIRTYPE
        tf.addfile(d)
        add(f"geoarches-{commit}/LICENSE", b"BSD-3-Clause")
        add(f"geoarches-{commit}/geoarches/__init__.py", b"")
        add(f"geoarches-{commit}/geoarches/stats/aimip_norm_stats.nc", b"stats")
        add(f"geoarches-{commit}/docs/index.md", b"not unpacked")
        for ti in extra or []:
            tf.addfile(ti, io.BytesIO(b"x" * ti.size) if ti.isfile() else None)
    path.write_bytes(buf.getvalue())
    return buf.getvalue()


def test_geoarches_source_install_verify_and_refusals(tmp_path, monkeypatch):
    from aimip import geoarches_src as g

    arc = tmp_path / "geoarches.tar.gz"
    blob = _synthetic_geoarches_archive(arc, g.GEOARCHES_COMMIT)
    # the expected tree: geoarches/ and LICENSE only (docs/ is not unpacked)
    ref = tmp_path / "ref"
    (ref / "geoarches" / "stats").mkdir(parents=True)
    (ref / "LICENSE").write_bytes(b"BSD-3-Clause")
    (ref / "geoarches" / "__init__.py").write_bytes(b"")
    (ref / "geoarches" / "stats" / "aimip_norm_stats.nc").write_bytes(b"stats")
    digest, n = g.tree_digest(str(ref))
    assert n == 3
    monkeypatch.setattr(g, "GEOARCHES_ARCHIVE_SIZE", len(blob))
    monkeypatch.setattr(g, "GEOARCHES_ARCHIVE_SHA256", hashlib.sha256(blob).hexdigest())
    monkeypatch.setattr(g, "GEOARCHES_TREE_SHA256", digest)
    monkeypatch.setattr(g, "GEOARCHES_TREE_FILES", 3)

    target = tmp_path / "opt" / "geoarches"
    g.install(str(target), archive=str(arc))
    assert sorted(p.relative_to(target).as_posix() for p in target.rglob("*") if p.is_file()) == [
        "LICENSE", "geoarches/__init__.py", "geoarches/stats/aimip_norm_stats.nc"]
    g.verify(str(target))
    assert not [p for p in target.parent.iterdir() if p.name.startswith(".geoarches-")]   # no staging left behind
    with pytest.raises(SystemExit, match="refusing to overwrite"):
        g.install(str(target), archive=str(arc))
    (target / "geoarches" / "__pycache__").mkdir()
    (target / "geoarches" / "__pycache__" / "__init__.cpython-312.pyc").write_bytes(b"\0")
    with pytest.raises(SystemExit, match="it contains __pycache__"):
        g.verify(str(target))
    (target / "geoarches" / "__pycache__" / "__init__.cpython-312.pyc").rename(tmp_path / "moved.pyc")
    (target / "geoarches" / "__pycache__").rmdir()
    g.verify(str(target))
    (target / "geoarches" / "__init__.py").write_text("import os  # changed\n")
    with pytest.raises(SystemExit, match="does not match the pinned commit"):
        g.verify(str(target))

    # a different archive is refused before anything is unpacked
    other = tmp_path / "other.tar.gz"
    other.write_bytes(blob + b"\0")
    with pytest.raises(SystemExit, match="archive mismatch"):
        g.install(str(tmp_path / "t2"), archive=str(other))
    assert not (tmp_path / "t2").exists()
    # unsafe entries inside the kept subtree are refused
    for bad in ("geoarches/../../evil.py", "geoarches/link"):
        ti = tarfile.TarInfo(f"geoarches-{g.GEOARCHES_COMMIT}/{bad}")
        if bad.endswith("link"):
            ti.type, ti.linkname = tarfile.SYMTYPE, "/etc/passwd"
        else:
            ti.size = 1
        arc_bad = tmp_path / "bad.tar.gz"
        blob_bad = _synthetic_geoarches_archive(arc_bad, g.GEOARCHES_COMMIT, extra=[ti])
        monkeypatch.setattr(g, "GEOARCHES_ARCHIVE_SIZE", len(blob_bad))
        monkeypatch.setattr(g, "GEOARCHES_ARCHIVE_SHA256", hashlib.sha256(blob_bad).hexdigest())
        with pytest.raises(SystemExit, match="refusing archive entry"):
            g.install(str(tmp_path / "t3"), archive=str(arc_bad))
        assert not (tmp_path / "t3").exists()

    # the job finds the tree next to the emulator environment (or at $AIMIP_GEOARCHES_CODE)
    monkeypatch.delenv("AIMIP_GEOARCHES_CODE", raising=False)
    assert g.code_dir("/opt/aimip/venv-emu/bin/python") == "/opt/aimip/geoarches"
    monkeypatch.setenv("AIMIP_GEOARCHES_CODE", "/dev/geo")
    assert g.code_dir("/opt/aimip/venv-emu/bin/python") == "/dev/geo"


def test_extract_zip_member_copies_one_file_unchanged(tmp_path):
    from aimip.builders import extract_zip_member

    z = tmp_path / "model_checkpoints.zip"
    with zipfile.ZipFile(z, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("Downloads/model_checkpoints/m/checkpoints/checkpoint_global_step=300000.ckpt", b"W" * 1000)
        zf.writestr("Downloads/model_checkpoints/m/", b"")
    dst = tmp_path / "models" / "m" / "checkpoint.ckpt"
    extract_zip_member(str(z), "Downloads/model_checkpoints/m/checkpoints/checkpoint_global_step=300000.ckpt", str(dst))
    assert dst.read_bytes() == b"W" * 1000
    with pytest.raises(ValidationError, match="is not in"):
        extract_zip_member(str(z), "Downloads/model_checkpoints/other.ckpt", str(tmp_path / "x"))
    with pytest.raises(ValidationError, match="not a regular file"):
        extract_zip_member(str(z), "Downloads/model_checkpoints/m/", str(tmp_path / "y"))
    assert sorted(p.name for p in dst.parent.iterdir()) == ["checkpoint.ckpt"]


def test_monthly_writer_with_levels(tmp_path, allow_tmp_run_root):
    import netCDF4

    from aimip.emu_job import MonthlyWriter
    from aimip.status import summarize_run

    run = allow_tmp_run_root / "awlev"
    lat, lon, levels = np.array([90.0, 0.0, -90.0]), np.array([0.0, 120.0, 240.0]), [500.0, 850.0]
    ws = MonthlyWriter(str(run / "atmosphere" / "monthly_mean_predictions.nc"), ["tas"], lat, lon, 2)
    wl = MonthlyWriter(str(run / "atmosphere" / "monthly_levels.nc"), ["zg", "ta"], lat, lon, 2, levels=levels,
                       extra={"zg": {"units": "m"}})
    for member in (0, 1):
        for k, month in enumerate((np.datetime64("1978-10", "M"), np.datetime64("1978-11", "M"))):
            ws.write(member, month, {"tas": np.full((3, 3), 287.0 + k + member, "f4")}, 31 - k)
            wl.write(member, month, {"zg": np.full((2, 3, 3), 5500.0 + k, "f4"), "ta": np.zeros((2, 3, 3), "f4")}, 31 - k)
    ws.close()
    wl.close()
    with netCDF4.Dataset(run / "atmosphere" / "monthly_levels.nc") as ds:
        assert ds["zg"].dimensions == ("sample", "time", "level", "lat", "lon") and ds["zg"].units == "m"
        assert list(ds["level"][:]) == levels and float(ds["zg"][1, 1, 0, 0, 0]) == 5501.0
    rows = summarize_run(str(run))["variables"]["tas"]["series"]
    assert [r["month"] for r in rows] == ["1978-10", "1978-11"] and rows[1]["members"] == [288.0, 289.0]


def test_geoarches_pins_agree_across_catalog_lock_and_recipes():
    from aimip import geoarches_src as g
    from aimip.emu_catalog import AW_MODELS, AW_ZENODO_FILES, AW_ZIP_MEMBERS, AW_ZIP_REL, required_files

    lock = json.loads((REPO_ROOT / "src" / "aimip" / "aimip.lock").read_text())["files"]
    for rel, (fname, size, md5, sha, _models) in AW_ZENODO_FILES.items():
        assert lock[rel]["size"] == size and lock[rel]["sha256"] == sha and lock[rel]["source"]["md5"] == md5
        assert lock[rel]["source"]["url"] == f"https://zenodo.org/records/22795233/files/{fname}?download=1"
    for rel, (member, size, sha, _models) in AW_ZIP_MEMBERS.items():
        assert "=" not in rel and member.startswith("Downloads/model_checkpoints/aimip-archesweather")
        assert lock[rel]["sha256"] == sha and lock[rel]["size"] == size
        assert lock[rel]["source"]["inputs"] == {AW_ZIP_REL: AW_ZENODO_FILES[AW_ZIP_REL][3]}
    gen, mx4 = (set(required_files(m)) for m in ("archesweathergen", "archesweather"))
    assert mx4 < gen and len(gen - mx4) == 3 and AW_ZIP_REL not in gen and AW_MODELS == ("archesweather", "archesweathergen")

    dockerfile = (REPO_ROOT / "container" / "Dockerfile").read_text()
    native = (REPO_ROOT / "install" / "install-native.sh").read_text()
    constraints = (REPO_ROOT / "container" / "constraints-emu.txt").read_text().splitlines()
    assert "COPY src/aimip/geoarches_src.py /opt/aimip/geoarches_src.py" in dockerfile
    assert "geoarches_src.py install /opt/aimip/geoarches" in dockerfile and "AIMIP_GEOARCHES_OK" in dockerfile
    assert 'geoarches_src.py" install "$PREFIX/geoarches"' in native and "AIMIP_GEOARCHES_OK" in native
    for text in (dockerfile, native):
        assert "axial-attention fasteners" in text
    assert "axial-attention==0.6.1" in constraints and "fasteners==0.19" in constraints
    assert g.GEOARCHES_ARCHIVE_URL.endswith(f"/{g.GEOARCHES_COMMIT}.tar.gz") and len(g.GEOARCHES_COMMIT) == 40
