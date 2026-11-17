"""dlesym-aimip: DLESyM v0-ISCCP with the SST prescribed from the AIMIP forcing. Request window, pinned files, the
job's SST binding, and the time interpolation (no torch, no network: the emulator interpreter is faked)."""
from __future__ import annotations

import json
import pathlib

import numpy as np
import pytest

from aimip.emu_catalog import AIMIP_SST_REL, AIMIP_SST_SHA256, registry_entries, required_files
from aimip.errors import ConfigError, ValidationError

from conftest import sha256_path
from test_emu import _prepare, emu_env  # noqa: F401  (fixture)


def test_catalog_pins_the_v0_package_and_the_aimip_forcing():
    from aimip import setup_data

    rels = required_files("dlesym-aimip")
    assert AIMIP_SST_REL in rels and AIMIP_SST_REL == setup_data.ZENODO_AIMIP_REL_PATH
    assert sorted(r for r in rels if r != AIMIP_SST_REL) == sorted(required_files("dlesym-v0-isccp"))
    lock = json.loads((pathlib.Path(setup_data.__file__).parent / "aimip.lock").read_text())
    assert lock["files"][AIMIP_SST_REL]["sha256"] == AIMIP_SST_SHA256
    entry = next(e for e in registry_entries("${DATA}") if e["name"] == "dlesym-aimip")
    assert [f["name"] for f in entry["forcing_sets"]] == ["aimip"] and not entry["stochastic"]
    assert entry["checkpoint"].endswith("models/dlesym-v0-isccp/atmos_model_0.mdlus")
    spec = setup_data.build_artifact_catalog()[AIMIP_SST_REL]
    assert "dlesym-aimip" in spec.models_needed and "dlesym-aimip" in spec.models_output


def test_request_window_and_forcing_end(emu_env):
    from aimip.emu import precheck_emu_request

    m = emu_env["specs"]["dlesym-aimip"]
    ok = precheck_emu_request(m, start_iso="1978-10-03T00:00:00", steps=67552, members=None,
                              forcing_set_name=None, ic_path=None)
    assert ok["start"] == "1978-10-03T00:00:00" and ok["end"] == "2024-12-28T00:00:00" and ok["scenario"] == "aimip"
    ok = precheck_emu_request(m, start_iso="1978-10-03T00:00:00", steps=67568, members=None,
                              forcing_set_name=None, ic_path=None)
    assert ok["end"] == "2025-01-01T00:00:00"      # 4223 whole 96-h steps end on the last forcing stamp
    ok = precheck_emu_request(m, start_iso="1978-10-07T00:00:00", steps=4, members=1,
                              forcing_set_name="aimip", ic_path=None)
    assert ok["members"] == 1
    for kw, msg in (
        (dict(start_iso="1978-10-02T18:00:00", steps=4), "outside the AIMIP SST forcing"),
        (dict(start_iso="2024-12-28T06:00:00", steps=4), "outside the AIMIP SST forcing"),
        (dict(start_iso="1978-10-03T00:00:00", steps=67569), "at most 67568 6-h steps"),
        (dict(start_iso="2024-12-20T00:00:00", steps=49), "at most 48 6-h steps"),
        (dict(start_iso="1978-10-03T03:00:00", steps=4), "at 00, 06, 12 or 18 UTC"),
        (dict(start_iso="1978-10-03T00:00:00", steps=4, members=2), "deterministic"),
        (dict(start_iso="1978-10-03T00:00:00", steps=4, forcing_set_name="ssp585"), "one forcing set only"),
    ):
        args = dict(members=None, forcing_set_name=None, ic_path=None)
        args.update(kw)
        with pytest.raises(ConfigError, match=msg):
            precheck_emu_request(m, **args)


def test_prepare_binds_the_prescribed_sst(emu_env):
    from aimip.runner import exec_fme

    out = emu_env["runs"] / "da"
    m = _prepare(emu_env, "dlesym-aimip", out, start_iso="1978-10-03T00:00:00", steps=16)
    sst = str(emu_env["data"] / AIMIP_SST_REL)
    job = json.loads((out / "config.yaml").read_text())
    assert job["sst_forcing"] == sst and job["flavor"] == "v0" and job["package_dir"].endswith("models/dlesym-v0-isccp")
    assert m["package_files_sha256"][sst] == sha256_path(emu_env["data"] / AIMIP_SST_REL)
    assert "SST prescribed from the AIMIP forcing" in m["ic_provenance"]
    assert exec_fme(str(out))["rc"] == 0

    # the plain v0 model carries no SST file, and a job file that adds or moves one is refused
    out2 = emu_env["runs"] / "dv0"
    _prepare(emu_env, "dlesym-v0-isccp", out2)
    assert json.loads((out2 / "config.yaml").read_text())["sst_forcing"] is None
    for name, change in (("dc1", lambda j: j.update(sst_forcing=None)),
                         ("dc2", lambda j: j.update(sst_forcing="/elsewhere/" + AIMIP_SST_REL))):
        o = emu_env["runs"] / name
        _prepare(emu_env, "dlesym-aimip", o, start_iso="1978-10-03T00:00:00", steps=16)
        j = json.loads((o / "config.yaml").read_text())
        change(j)
        (o / "config.yaml").write_text(json.dumps(j))
        with pytest.raises(ValidationError):
            exec_fme(str(o))
    # a changed forcing file is refused at job time
    o = emu_env["runs"] / "dc3"
    _prepare(emu_env, "dlesym-aimip", o, start_iso="1978-10-03T00:00:00", steps=16)
    (emu_env["data"] / AIMIP_SST_REL).write_bytes(b"other forcing")
    with pytest.raises(ValidationError, match="SHA-256 mismatch at job time"):
        exec_fme(str(o))


def test_month_weights_are_linear_between_month_starts():
    from aimip.emu_job import month_weights

    starts = np.array(["1978-10-01", "1978-11-01", "1978-12-01"], dtype="datetime64[ns]")
    assert month_weights("1978-10-01T00:00", starts) == (0, 0.0)
    k, w = month_weights("1978-10-16T12:00", starts)
    assert k == 0 and w == pytest.approx(15.5 / 31)
    assert month_weights("1978-11-01T00:00", starts) == (1, 0.0)
    assert month_weights("1978-12-01T00:00", starts) == (1, 1.0)     # the last stamp itself
    for t in ("1978-09-30T18:00", "1978-12-01T06:00", "1978-12-15T00:00"):
        with pytest.raises(SystemExit, match="does not cover"):
            month_weights(t, starts)
