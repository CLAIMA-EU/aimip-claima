"""`--output FREQ[:VARS][@START..END]`: parsing, the rules checked at plan time, fme's `data_writer.files`, the
plan digest, the experiment file, and the check of the written files after the job."""
from __future__ import annotations

import json
import pathlib
from typing import Any

import netCDF4
import numpy as np
import pytest
import yaml

from aimip.cli import main
from aimip.errors import ConfigError
from aimip.outputs import (
    check_output_files,
    describe_outputs,
    fme_file_writers,
    parse_output,
    resolve_outputs,
    steps_in_memory,
)

ACE = dict(model_name="ace2-era5", family="fme", is_coupled=False, timestep_hours=6,
           start_iso="2001-01-01T00:00:00", calendar="proleptic_gregorian")


def _resolve(texts, steps=1460, end="2002-01-01T00:00:00", **kw):
    args = dict(ACE, end_iso=end, steps=steps)
    args.update(kw)
    return resolve_outputs(texts, **args)


def test_parse_forms():
    r = parse_output("daily")
    assert (r.freq, r.hours, r.names, r.window) == ("1d", 24, (), None)
    r = parse_output("12h:TMP2m,PRATEsfc,TMP2m")
    assert (r.freq, r.hours, r.names) == ("12h", 12, ("TMP2m", "PRATEsfc"))
    r = parse_output("step:TMP2m@2001-01-01..2001-02-01T00:00:00")
    assert (r.freq, r.window) == ("step", ("2001-01-01", "2001-02-01T00:00:00"))
    assert parse_output("5d").hours == 120 and parse_output("monthly").freq == "monthly"
    for bad, msg in (("weekly", "Unknown output frequency"), ("daily:", "Cannot read"), ("daily:1abc", "Invalid variable"),
                     ("monthly@2001-01-01..2001-02-01", "not available for monthly"), ("daily@2001-01-01", "Cannot read")):
        with pytest.raises(ConfigError, match=msg):
            parse_output(bad)


def test_resolve_files_blocks_periods_and_labels():
    out = _resolve(["daily:TMP2m,PRATEsfc", "step:TMP2m@2001-01-01T06:00:00..2001-01-02", "5d", "daily",
                    "monthly:TMP2m"])
    assert [o["file"] for o in out] == ["output_daily.nc", "output_6h.nc", "output_5d.nc", "output_daily_2.nc",
                                        "output_monthly.nc"]
    assert [o["block_steps"] for o in out] == [4, 1, 20, 4, None]
    assert out[0]["values_per_member"] == 365 and out[2]["values_per_member"] == 73
    assert out[1]["index_slice"] == [1, 4] and out[1]["window"] == ["2001-01-01T06:00:00", "2001-01-02T00:00:00"]
    assert out[1]["values_per_member"] == 3 and out[4]["values_per_member"] is None
    assert json.loads(json.dumps(out)) == out           # stored in the plan and the manifest as JSON
    noleap = resolve_outputs(["daily@2001-02-28..2001-03-01"], **dict(ACE, calendar="noleap", end_iso="2002-01-01T00:00:00",
                                                                         steps=1460))
    assert noleap[0]["index_slice"] == [58, 59]


@pytest.mark.parametrize("texts, kw, msg", [
    (["daily"], dict(steps=1461), "use --steps 1460 or --steps 1464"),
    (["7h"], {}, "multiple of 6 h"),
    (["90d"], {}, "more than 240 model steps"),
    (["step@2001-01-01T03:00:00..2001-01-02"], {}, "boundaries of the model steps"),
    (["daily@2001-01-01T06:00:00..2001-01-03"], {}, "boundaries of the days"),
    (["daily@2001-06-01..2001-03-01"], {}, "END must be after START"),
    (["daily@2001-06-01..2003-01-01"], {}, "inside the run"),
    (["daily", "daily"], {}, "given twice"),
    ([f"{h}h" for h in range(6, 60, 6)], {}, "At most 8"),
    (["daily"], dict(family="geoarches", model_name="archesweather"), "not available for archesweather"),
    (["daily"], dict(is_coupled=True, model_name="samudrace-cm4"), "coupled atmosphere-ocean"),
])
def test_resolve_refuses(texts, kw, msg):
    with pytest.raises(ConfigError, match=msg):
        _resolve(texts, **kw)


def test_fme_entries_and_steps_in_memory():
    out = _resolve(["daily:TMP2m", "step:TMP2m@2001-01-01T06:00:00..2001-01-02", "monthly"])
    assert fme_file_writers(out) == [
        {"label": "output_daily", "save_reference": False, "names": ["TMP2m"], "time_coarsen": {"coarsen_factor": 4}},
        {"label": "output_6h", "save_reference": False, "names": ["TMP2m"], "time_selection": {"start": 1, "stop": 4}},
        {"label": "output_monthly", "save_reference": False, "time_coarsen": {"method": "monthly_mean"}},
    ]
    assert steps_in_memory(1460, []) == 40 and steps_in_memory(8, []) == 8
    assert steps_in_memory(1460, _resolve(["daily"])) == 40
    assert steps_in_memory(1464, _resolve(["36h"], steps=1464, end="2002-01-02T00:00:00")) == 36   # 40 is not a multiple of 6
    assert steps_in_memory(1460, _resolve(["daily", "5d"])) == 40
    assert steps_in_memory(1440, _resolve(["30d"], steps=1440, end="2000-01-01T00:00:00",
                                          start_iso="1999-01-06T00:00:00")) == 120
    assert steps_in_memory(8, _resolve(["daily"], steps=8, end="2001-01-03T00:00:00")) == 8
    with pytest.raises(ConfigError, match="combine to blocks"):
        steps_in_memory(1460, [{"block_steps": 236}, {"block_steps": 240}])


def test_card_lines():
    out = _resolve(["daily:TMP2m,PRATEsfc", "5d", "monthly:TMP2m"])
    lines = describe_outputs(out, members=2, grid_cells=180 * 360)
    assert lines[0] == "output_daily.nc: daily means of TMP2m, PRATEsfc, whole run (365 per member, ~361 MiB)"
    assert lines[1] == "output_5d.nc: 5-day means of all variables, whole run (73 per member)"
    assert lines[2] == "output_monthly.nc: monthly means of TMP2m, whole run (one value per month)"


def test_run_writes_the_files_section_and_binds_it(playground_env: dict[str, Any]) -> None:
    from aimip import services
    from aimip.plan_types import RunSpec

    out_dir: pathlib.Path = playground_env["runs_dir"] / "era5_outputs"
    rc = main(["run", "-m", "ace2-era5", "--start", "2001-01-01T00:00:00", "--steps", "8", "--out", str(out_dir),
               "--output", "daily:TMP2m", "--output", "step:TMP2m@2001-01-01T06:00:00..2001-01-02", "--dry-run"])
    assert rc == 0
    cfg = yaml.safe_load((out_dir / "config.yaml").read_text(encoding="utf-8"))
    assert cfg["forward_steps_in_memory"] == 8
    assert cfg["data_writer"]["save_monthly_files"] is True
    assert [f["label"] for f in cfg["data_writer"]["files"]] == ["output_daily", "output_6h"]
    manifest = json.loads((out_dir / "aimip-manifest.json").read_text(encoding="utf-8"))
    assert [o["file"] for o in manifest["outputs"]] == ["output_daily.nc", "output_6h.nc"]

    base = dict(model="ace2-era5", start="2001-01-01T00:00:00", steps=8)
    d0 = services.plan(RunSpec(**base)).digest
    d1 = services.plan(RunSpec(**base, outputs=("daily:TMP2m",))).digest
    d2 = services.plan(RunSpec(**base, outputs=("daily:PRATEsfc",))).digest
    assert len({d0, d1, d2}) == 3                       # the outputs are part of the approved plan
    assert RunSpec.from_dict(RunSpec(**base, outputs=("daily",)).to_dict()).outputs == ("daily",)


def test_experiment_file_outputs(tmp_path: pathlib.Path) -> None:
    from aimip.services.cards import merge_config_and_cli

    cfg = tmp_path / "exp.yaml"
    cfg.write_text("model: ace2-era5\nstart: 2001-01-01\nsteps: 8\noutputs:\n  - daily:TMP2m\n  - 5d\n", encoding="utf-8")
    assert merge_config_and_cli(str(cfg), {}).outputs == ("daily:TMP2m", "5d")
    over = merge_config_and_cli(str(cfg), {"outputs": ("step",)})
    assert over.outputs == ("step",) and any(o.startswith("outputs=") for o in over.cli_overrides)
    cfg.write_text("model: ace2-era5\nstart: 2001-01-01\nsteps: 8\noutputs: {daily: true}\n", encoding="utf-8")
    with pytest.raises(Exception, match="outputs"):
        merge_config_and_cli(str(cfg), {})


def test_check_output_files(tmp_path: pathlib.Path, capsys: pytest.CaptureFixture[str]) -> None:
    with netCDF4.Dataset(tmp_path / "output_daily.nc", "w") as ds:
        ds.createDimension("time", 2)
        ds.createVariable("TMP2m", "f4", ("time",))[:] = np.array([1.0, 2.0])
    resolved = _resolve(["daily:TMP2m,PRATEsfc", "5d"])
    report = check_output_files(str(tmp_path), resolved)
    assert report["ok"] is False
    assert report["files"][0] == {"file": "output_daily.nc", "exists": True, "times": 2,
                                  "missing_variables": ["PRATEsfc"]}
    assert report["files"][1] == {"file": "output_5d.nc", "exists": False}
    err = capsys.readouterr().out
    assert "output_daily.nc without PRATEsfc" in err and "output_5d.nc missing" in err
    ok = check_output_files(str(tmp_path), _resolve(["daily:TMP2m"]))
    assert ok["ok"] is True
