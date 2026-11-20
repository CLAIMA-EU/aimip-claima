"""Collect the output variables of every aimip-claima model into src/aimip/model_variables.json.

Run it in the main environment of an installation (it needs `fme` for its registry of derived quantities), with
internet access, after a change to the models:

    python tools/model_variables/collect.py [--data <AIMIP_DATA>] [--out src/aimip/model_variables.json]

Sources, per family:
- fme (ACE, SamudrACE): `in_names` / `out_names` of each stepper in the checkpoint's `data.pkl` (read from the local
  file under --data if present, otherwise with HTTP range requests from the pinned Hugging Face revision in
  aimip.lock: only the zip directory and the small pickle are downloaded); `derived` = the names in fme's
  registries of derived quantities, which fme writes when their inputs exist;
- e2s-dlesym: the package config.yaml (atmosphere, ocean; v1.1 adds precipitation);
- arches (ArchesClimate-SSP): the model config.yaml (surface and level variables) plus the ocean `thetao`;
- geoarches, md, camulator, cbottle, sdyff, samudra: the names their writers use (aimip modules).
"""
from __future__ import annotations

import argparse
import io
import json
import os
import pickle
import sys
import urllib.request
import zipfile

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(REPO, "src"))

from aimip.registry import build_default_registry_payload  # noqa: E402
from aimip.setup_data import CANONICAL_MODELS, load_lock_file  # noqa: E402

UA = {"User-Agent": "python-urllib"}


class _Stub:
    def __init__(self, *a, **k):
        pass

    def __setstate__(self, state):
        self.__dict__["_state"] = state

    def __call__(self, *a, **k):
        return _Stub()


class _Unpickler(pickle.Unpickler):
    """Reads the structure of a torch checkpoint without torch: tensors become None, classes stubs."""

    def find_class(self, module, name):
        if module == "collections" and name == "OrderedDict":
            import collections

            return collections.OrderedDict
        return _Stub

    def persistent_load(self, pid):
        return None


class _HTTPRange(io.RawIOBase):
    def __init__(self, url: str):
        self.url, self.pos = url, 0
        req = urllib.request.Request(url, method="HEAD", headers=UA)
        with urllib.request.urlopen(req, timeout=60) as r:
            self.url = r.geturl()
            self.size = int(r.headers["Content-Length"])

    def seekable(self):
        return True

    def readable(self):
        return True

    def tell(self):
        return self.pos

    def seek(self, off, whence=0):
        self.pos = {0: off, 1: self.pos + off, 2: self.size + off}[whence]
        return self.pos

    def readinto(self, b):
        n = min(len(b), self.size - self.pos)
        if n <= 0:
            return 0
        req = urllib.request.Request(self.url, headers={**UA, "Range": f"bytes={self.pos}-{self.pos + n - 1}"})
        with urllib.request.urlopen(req, timeout=120) as r:
            data = r.read()
        b[:len(data)] = data
        self.pos += len(data)
        return len(data)


def _hf_url(src: dict) -> str:
    return f"https://huggingface.co/{src['repo']}/resolve/{src['revision']}/{src['path']}"


def _open(rel: str, lock: dict, data_dir: str | None):
    local = os.path.join(data_dir, rel) if data_dir else None
    if local and os.path.isfile(local):
        return open(local, "rb"), f"local {rel}"
    src = lock[rel]["source"]
    if src.get("kind") != "hf":
        raise SystemExit(f"{rel}: no local copy and source {src.get('kind')} is not Hugging Face")
    return io.BufferedReader(_HTTPRange(_hf_url(src)), buffer_size=1 << 20), f"hf {src['repo']}@{src['revision'][:10]}"


def _steppers(obj, path=""):
    """(path, in_names, out_names) of every stepper config in a checkpoint."""
    if isinstance(obj, dict):
        if "out_names" in obj and "in_names" in obj:
            yield path, list(obj["in_names"]), list(obj["out_names"])
        for k, v in obj.items():
            yield from _steppers(v, f"{path}.{k}")
    elif isinstance(obj, (list, tuple)):
        for i, v in enumerate(obj):
            yield from _steppers(v, f"{path}[{i}]")


def fme_entry(model: dict, lock: dict, data_dir: str | None, derived: dict) -> dict:
    prefix = "${DATA}/"
    rel = model["checkpoint"][len(prefix):] if model["checkpoint"].startswith(prefix) else model["checkpoint"]
    fh, where = _open(rel, lock, data_dir)
    with fh:
        z = zipfile.ZipFile(fh)
        pkl = next(n for n in z.namelist() if n.endswith("data.pkl"))
        obj = _Unpickler(z.open(pkl)).load()
    found = list(_steppers(obj))
    if not found:
        raise SystemExit(f"{model['name']}: no in_names/out_names in {rel}")
    names: list = []
    for _, _, out in found:
        names += [n for n in out if n not in names]
    coupled = model.get("fme_module") == "fme.coupled.inference"
    der = sorted(set(derived["atmosphere"]) | (set(derived["ocean"]) if coupled else set()))
    return {"source": f"checkpoint {rel} ({where}): stepper out_names; derived: fme registries",
            "variables": names, "derived": [d for d in der if d not in names]}


def fme_derived() -> dict:
    from fme.core import derived_variables as dv

    out = {"atmosphere": sorted(dv._DERIVED_VARIABLE_REGISTRY.keys()), "ocean": []}
    try:
        from fme.core import ocean_derived_variables as odv

        out["ocean"] = sorted(odv._OCEAN_DERIVED_VARIABLE_REGISTRY.keys())
    except ImportError:
        pass
    return out


def _yaml(rel: str, lock: dict, data_dir: str | None) -> dict:
    import yaml

    fh, _ = _open(rel, lock, data_dir)
    with fh:
        return yaml.safe_load(fh.read())


def dlesym_entry(model: dict, lock: dict, data_dir: str | None) -> dict:
    from aimip.emu import DLESYM_MODELS

    info = DLESYM_MODELS[model["name"]]
    rel = ("models/dlesym-v0-isccp/config.yaml" if info["flavor"] == "v0"
           else f"models/dlesym-v1/checkpoints/{info['version']}/config.yaml")
    if rel not in lock:
        rel = "models/dlesym-v1/config.yaml"
    io_cfg = _yaml(rel, lock, data_dir).get("io", {})
    atmos = list(io_cfg.get("atmos_variables") or []) + list(io_cfg.get("atmos_diagnostic_variables") or [])
    ocean = list(io_cfg.get("ocean_variables") or []) + list(io_cfg.get("ocean_diagnostic_variables") or [])
    # The job moves a predicted tp6 out of the atmosphere into atmosphere/monthly_precip.nc (with tp6_exp and tp06);
    # the separate precipitation model of the v0 package is not run.
    precip = ["tp6", "tp6_exp", "tp06"] if "tp6" in atmos else []
    atmos = [v for v in atmos if v != "tp6"]
    return {"source": f"{rel}: io.atmos_variables / ocean_variables",
            "variables": atmos + ocean + precip,
            "groups": {"atmosphere": atmos, "ocean": ocean, "precipitation": precip}}


def aw_entry() -> dict:
    from aimip.emu_job import AW_LEVEL_OUT, AW_SURFACE_OUT

    surf = [v[0] for v in AW_SURFACE_OUT.values()]
    lev = [v[0] for v in AW_LEVEL_OUT.values()]
    return {"source": "aimip.emu_job AW_SURFACE_OUT / AW_LEVEL_OUT", "variables": surf + lev,
            "units": {v[0]: v[1] for v in list(AW_SURFACE_OUT.values()) + list(AW_LEVEL_OUT.values())},
            "levels": {n: [50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 850, 925, 1000] for n in lev}}


def arches_entry(model: dict, lock: dict, data_dir: str | None) -> dict:
    rel = "models/archesclimate-ssp/archesclimate-ssp-deterministic/config.yaml"
    cfg = _yaml(rel, lock, data_dir)

    def find(d, key):
        if isinstance(d, dict):
            if key in d:
                return d[key]
            for v in d.values():
                r = find(v, key)
                if r is not None:
                    return r
        return None

    surf = [str(v) for v in (find(cfg, "surface_variables") or [])]
    lev = [str(v) for v in (find(cfg, "level_variables") or [])]
    return {"source": f"{rel}: surface_variables / level_variables, plus the ocean thetao", "variables": surf + lev,
            "groups": {"atmosphere": surf + lev, "ocean": ["thetao_0", "thetao"]}}


def code_entry(family: str) -> dict:
    if family == "md":
        from aimip.md15 import MD15_OUTPUT_CHANNELS

        return {"source": "aimip.md15 MD15_OUTPUT_CHANNELS", "variables": [c[1] for c in MD15_OUTPUT_CHANNELS],
                "units": {c[1]: c[2] for c in MD15_OUTPUT_CHANNELS}}
    if family == "camulator":
        from aimip.camulator import CAMULATOR_SURFACE_DIAG_VARS, CAMULATOR_UPPER_AIR_VARS

        return {"source": "aimip.camulator CAMULATOR_UPPER_AIR_VARS / CAMULATOR_SURFACE_DIAG_VARS",
                "variables": list(CAMULATOR_UPPER_AIR_VARS) + list(CAMULATOR_SURFACE_DIAG_VARS)}
    if family == "cbottle":
        from aimip.cbottle import CBOTTLE_OUTPUT_VARS

        return {"source": "aimip.cbottle CBOTTLE_OUTPUT_VARS", "variables": list(CBOTTLE_OUTPUT_VARS)}
    if family == "sdyff":
        from aimip.sdyff import SDYFF_DERIVED_VARS, SDYFF_PREDICTED_VARS, SDYFF_PRESCRIBED_SURFACE_VAR

        names = [v for v in SDYFF_PREDICTED_VARS if v != "surface_temperature"] + [SDYFF_PRESCRIBED_SURFACE_VAR]
        return {"source": "aimip.sdyff SDYFF_PREDICTED_VARS / SDYFF_DERIVED_VARS",
                "variables": names + list(SDYFF_DERIVED_VARS)}
    if family == "samudra":
        from aimip.samudra import SAMUDRA_STATE_VARS

        return {"source": "aimip.samudra SAMUDRA_STATE_VARS", "variables": list(SAMUDRA_STATE_VARS)}
    raise SystemExit(f"no collector for family {family}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--data", default=os.environ.get("AIMIP_DATA"))
    ap.add_argument("--out", default=os.path.join(REPO, "src", "aimip", "model_variables.json"))
    args = ap.parse_args()
    lock = load_lock_file(os.path.join(REPO, "src", "aimip", "aimip.lock"))["files"]
    models = {m["name"]: m for m in build_default_registry_payload()["models"]}
    derived = fme_derived()
    out: dict = {}
    for name in CANONICAL_MODELS:
        m = models[name]
        fam = m["family"]
        print(f"{name} ({fam}) ...", flush=True)
        if fam == "fme":
            entry = fme_entry(m, lock, args.data, derived)
        elif fam == "e2s-dlesym":
            entry = dlesym_entry(m, lock, args.data)
        elif fam == "geoarches":
            entry = aw_entry()
        elif fam == "arches":
            entry = arches_entry(m, lock, args.data)
        else:
            entry = code_entry(fam)
        out[name] = entry
        print(f"  {len(entry['variables'])} variables" + (f" + {len(entry['derived'])} derived" if entry.get("derived") else ""))
    payload = {"schema_version": 1, "generated_by": "tools/model_variables/collect.py", "models": out}
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=1)
        fh.write("\n")
    print(f"wrote {os.path.normpath(args.out)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
