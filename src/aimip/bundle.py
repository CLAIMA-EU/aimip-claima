"""`aimip bundle`: everything an installation needs for chosen models, in one directory, for a cluster without
internet (docs/installation.md, "Clusters without internet").

The directory is laid out like an installation prefix, plus the source tree and the instructions:

    <out>/images/aimip-<version>.sif   the container image of this installation
    <out>/data/...                     every file aimip.lock lists for the models (and the code they download)
    <out>/aimip-claima/                the source tree of the same version (it holds the installer)
    <out>/bundle.json, BUNDLE.md       what is inside, and the steps on the cluster

On the cluster, copy the directory to where the installation should live and run the container installer with it as
`--prefix`: it reuses the image, and `aimip fetch -m <model>` checks the copied files without downloading.
"""
from __future__ import annotations

import datetime
import json
import os
import shutil
from collections.abc import Sequence
from typing import Any

from . import __version__
from .errors import ConfigError

# models that download data each time a run is prepared, so they cannot run without internet
ONLINE_FAMILIES = {"e2s-dlesym": "its ERA5 starting state (Google Cloud)", "samudra": "its OM4 ocean states (OSN)"}


def _code_dirs() -> dict[str, str]:
    from .camulator import CAMULATOR_CODE_REL_DIR
    from .md15 import MD15_CODE_REL_DIR
    from .sdyff import SDYFF_CODE_REL_DIR
    from .setup_data import ARCHES_CODE_REL_DIR

    return {"arches": ARCHES_CODE_REL_DIR, "md": MD15_CODE_REL_DIR, "camulator": CAMULATOR_CODE_REL_DIR,
            "sdyff": SDYFF_CODE_REL_DIR}


def bundle_contents(models: Sequence[str], data_dir: str) -> dict[str, Any]:
    """Files (relative to the data directory) and code directories the *models* need; missing ones listed."""
    from .emu_catalog import on_demand_artifacts
    from .registry import build_default_registry_payload
    from .setup_data import build_artifact_catalog

    registry = {m["name"]: m for m in build_default_registry_payload()["models"]}
    unknown = [m for m in models if m not in registry]
    if unknown:
        raise ConfigError(f"Unknown model(s) {', '.join(unknown)}; `aimip models` lists them.", field="models")
    online = [f"{m} (downloads {ONLINE_FAMILIES[registry[m]['family']]} when a run is prepared)"
              for m in models if registry[m]["family"] in ONLINE_FAMILIES]
    if online:
        raise ConfigError("These models cannot run without internet and are not bundled: " + "; ".join(online) + ".",
                          field="models")
    wanted = set(models)
    files: dict[str, int] = {}
    for rel, spec in build_artifact_catalog().items():
        if wanted & set(spec.models_needed):
            files[rel] = 0
    for rel in (r for m in models for r in registry[m]["required_files"]):
        files[rel] = 0
    for a in on_demand_artifacts():                  # extra files fetched earlier for other start dates
        if wanted & set(a["models"]) and os.path.isfile(os.path.join(data_dir, a["rel_path"])):
            files[a["rel_path"]] = 0
    missing = []
    for rel in sorted(files):
        p = os.path.join(data_dir, rel)
        if os.path.isfile(p):
            files[rel] = os.path.getsize(p)
        else:
            missing.append(rel)
    code = sorted({_code_dirs()[registry[m]["family"]] for m in models if registry[m]["family"] in _code_dirs()})
    missing += [c for c in code if not os.path.isdir(os.path.join(data_dir, c))]
    return {"files": {r: s for r, s in files.items() if r not in missing}, "code_dirs": code, "missing": missing}


def _copy_file(src: str, dst: str) -> None:
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    try:
        os.link(src, dst)                            # no extra space where the file system allows it
    except OSError:
        shutil.copy2(src, dst)


def _source_tree(source: str | None) -> str:
    """A source tree of this version that holds the installer: --source, or the copy inside the image."""
    for cand in ([source] if source else []) + ["/opt/aimip/src"]:
        if cand and os.path.isfile(os.path.join(cand, "install", "install-container.sh")):
            ver_file = os.path.join(cand, "container", "VERSION")
            ver = open(ver_file, encoding="utf-8").read().strip() if os.path.isfile(ver_file) else None
            if ver != __version__:
                raise ConfigError(f"{cand} holds aimip-claima {ver}, but this installation is {__version__}: check "
                                  f"out the matching version (git checkout v{__version__}).", field="source")
            return cand
    raise ConfigError("Pass the aimip-claima source tree of this version with --source <checkout> (this image does "
                      "not carry the installer).", field="source")


def make_bundle(models: Sequence[str], out: str, *, source: str | None = None, prefix: str | None = None,
                data_dir: str | None = None, runtime: str | None = None, progress=print) -> dict[str, Any]:
    from .registry import resolve_data_dir, resolve_playground_dir
    from .validators import validate_safe_path

    models = [m.strip() for m in models if m.strip()]
    if not models:
        raise ConfigError("Name the models to bundle: --models ace2-era5,archesweather.", field="models")
    if runtime not in ("apptainer", "singularity"):
        raise ConfigError(
            f"aimip bundle needs a container installation with Apptainer or Singularity (this one: {runtime or 'native'}). "
            "A native installation installs Python packages from the internet and cannot be moved.", field="runtime")
    prefix = resolve_playground_dir(prefix)
    data_dir = resolve_data_dir(data_dir, playground_dir=prefix)
    if not prefix or not data_dir:
        raise ConfigError("Cannot find the installation (AIMIP_HOME / AIMIP_DATA are not set).", field="prefix")
    out = validate_safe_path(out, label="--out")
    if os.path.exists(out) and os.listdir(out):
        raise ConfigError(f"{out} exists and is not empty; give a new directory.", field="out")
    image = os.path.join(prefix, "images", f"aimip-{__version__}.sif")
    if not os.path.isfile(image):
        raise ConfigError(f"The image {image} is missing.", field="image")
    src_tree = _source_tree(source)
    contents = bundle_contents(models, data_dir)
    if contents["missing"]:
        raise ConfigError(
            "Some files of these models are not here yet: " + ", ".join(contents["missing"][:5])
            + (" …" if len(contents["missing"]) > 5 else "") + f". Download them first: "
            + " ".join(f"aimip fetch -m {m};" for m in models), field="models")

    os.makedirs(out, exist_ok=True)
    total = os.path.getsize(image) + sum(contents["files"].values())
    progress(f"aimip bundle: {len(contents['files'])} files of {', '.join(models)} and the image "
             f"({total / 2**30:.1f} GiB) -> {out}")
    _copy_file(image, os.path.join(out, "images", os.path.basename(image)))
    for i, (rel, _) in enumerate(sorted(contents["files"].items()), 1):
        _copy_file(os.path.join(data_dir, rel), os.path.join(out, "data", rel))
        if i % 50 == 0:
            progress(f"  {i}/{len(contents['files'])} files")
    for rel in contents["code_dirs"]:
        shutil.copytree(os.path.join(data_dir, rel), os.path.join(out, "data", rel), symlinks=False)
    shutil.copytree(src_tree, os.path.join(out, "aimip-claima"), symlinks=False,
                    ignore=shutil.ignore_patterns(".git", "__pycache__", "*.pyc", ".pytest_cache", "build", "*.egg-info"))
    manifest = {
        "aimip_version": __version__,
        "created": datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "models": list(models),
        "image": f"images/{os.path.basename(image)}",
        "files": contents["files"],
        "code_dirs": contents["code_dirs"],
        "bytes": total,
    }
    with open(os.path.join(out, "bundle.json"), "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=1)
        fh.write("\n")
    with open(os.path.join(out, "BUNDLE.md"), "w", encoding="utf-8") as fh:
        fh.write(_instructions(models))
    progress(f"aimip bundle: done. Next: copy {out} to the cluster and follow {out}/BUNDLE.md")
    return manifest


def _instructions(models: Sequence[str]) -> str:
    fetch = "\n".join(f"aimip fetch -m {m}" for m in models)
    return f"""# aimip-claima {__version__} bundle: {', '.join(models)}

This directory holds the container image, the files of these models and the aimip-claima source of the same version.
It becomes the installation directory (the prefix) on a cluster without internet.

1. Copy it to the cluster, to where the installation should live:

   ```bash
   rsync -a <this directory>/ mycluster:/path/to/aimip/
   ```

2. On the cluster, load Apptainer or Singularity if it is a module, then run the installer with that directory as
   the prefix and without --build. It finds the image in images/ and downloads nothing:

   ```bash
   cd /path/to/aimip/aimip-claima
   install/install-container.sh --prefix /path/to/aimip --run-root /path/to/aimip-runs \\
       --site generic-slurm --partition <partition> --account <project>
   ```

3. Check the models' files against the lock file (nothing is downloaded when they are all there):

   ```bash
   export PATH=/path/to/aimip/bin:$PATH
{chr(10).join('   ' + line for line in fetch.splitlines())}
   aimip doctor
   ```

Prepare runs with --no-fetch, so that a missing file stops at once instead of waiting for a network.
"""
