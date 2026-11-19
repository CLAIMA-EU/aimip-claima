"""`aimip bundle`: the files of chosen models, the image and the source, laid out as an installation prefix."""
from __future__ import annotations

import json
import os
import pathlib

import pytest

from aimip import __version__
from aimip.bundle import bundle_contents, make_bundle
from aimip.errors import ConfigError


def _fake_install(tmp_path: pathlib.Path, models: list[str]) -> tuple[pathlib.Path, pathlib.Path, pathlib.Path]:
    """A prefix with an image and every file the models need, and a source tree of this version."""
    from aimip.registry import build_default_registry_payload
    from aimip.setup_data import build_artifact_catalog

    prefix = tmp_path / "prefix"
    data = prefix / "data"
    (prefix / "images").mkdir(parents=True)
    (prefix / "images" / f"aimip-{__version__}.sif").write_bytes(b"image")
    reg = {m["name"]: m for m in build_default_registry_payload()["models"]}
    rels = {rel for rel, spec in build_artifact_catalog().items() if set(models) & set(spec.models_needed)}
    rels |= {r for m in models for r in reg[m]["required_files"]}
    for rel in rels:
        p = data / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(b"x" * 10)
    src = tmp_path / "checkout"
    (src / "install").mkdir(parents=True)
    (src / "install" / "install-container.sh").write_text("#!/bin/bash\n")
    (src / "container").mkdir()
    (src / "container" / "VERSION").write_text(__version__ + "\n")
    (src / ".git").mkdir()
    (src / ".git" / "HEAD").write_text("ref\n")
    return prefix, data, src


def test_contents_and_refusals(tmp_path):
    prefix, data, _ = _fake_install(tmp_path, ["ace2-era5"])
    c = bundle_contents(["ace2-era5"], str(data))
    assert c["missing"] == [] and "models/ace2-era5/ace2_era5_ckpt.tar" in c["files"] and c["code_dirs"] == []
    os.remove(data / "models/ace2-era5/ace2_era5_ckpt.tar")
    assert bundle_contents(["ace2-era5"], str(data))["missing"] == ["models/ace2-era5/ace2_era5_ckpt.tar"]
    with pytest.raises(ConfigError, match=r"dlesym-aimip \(downloads its ERA5 starting state"):
        bundle_contents(["ace2-era5", "dlesym-aimip"], str(data))
    with pytest.raises(ConfigError, match="Unknown model"):
        bundle_contents(["no-such-model"], str(data))


def test_make_bundle(tmp_path, monkeypatch):
    prefix, data, src = _fake_install(tmp_path, ["ace2-era5", "archesweather"])
    monkeypatch.setenv("AIMIP_PLAYGROUND", str(prefix))
    monkeypatch.setenv("AIMIP_DATA", str(data))
    out = tmp_path / "bundle"
    msgs: list[str] = []
    m = make_bundle(["ace2-era5", "archesweather"], str(out), source=str(src), runtime="apptainer",
                    progress=msgs.append)
    assert (out / "images" / f"aimip-{__version__}.sif").read_bytes() == b"image"
    assert (out / "data/models/ace2-era5/ace2_era5_ckpt.tar").is_file()
    assert (out / "aimip-claima/install/install-container.sh").is_file() and not (out / "aimip-claima/.git").exists()
    saved = json.loads((out / "bundle.json").read_text())
    assert saved["models"] == ["ace2-era5", "archesweather"] and saved == m
    assert all((out / "data" / rel).is_file() for rel in saved["files"])
    steps = (out / "BUNDLE.md").read_text()
    assert "install/install-container.sh --prefix /path/to/aimip" in steps and "aimip fetch -m archesweather" in steps
    assert msgs[-1].startswith("aimip bundle: done")

    with pytest.raises(ConfigError, match="not empty"):
        make_bundle(["ace2-era5"], str(out), source=str(src), runtime="apptainer", progress=msgs.append)
    with pytest.raises(ConfigError, match="container installation"):
        make_bundle(["ace2-era5"], str(tmp_path / "b2"), source=str(src), runtime="native", progress=msgs.append)
    (src / "container" / "VERSION").write_text("0.0.1\n")
    with pytest.raises(ConfigError, match="holds aimip-claima 0.0.1"):
        make_bundle(["ace2-era5"], str(tmp_path / "b3"), source=str(src), runtime="apptainer", progress=msgs.append)
