"""model_variables.json: one entry per model, used for `--output` names at plan time and by `aimip show`."""
from __future__ import annotations

import pytest

from aimip.errors import ConfigError
from aimip.model_variables import full_listing, output_names, summary, variables_of
from aimip.outputs import resolve_outputs
from aimip.setup_data import CANONICAL_MODELS


def test_every_model_has_its_variables():
    for name in CANONICAL_MODELS:
        entry = variables_of(name)
        assert entry is not None, name
        names = entry["variables"]
        assert names and all(isinstance(n, str) and n for n in names), name
        assert len(set(names)) == len(names), name
        assert entry.get("source"), name
    era5 = variables_of("ace2-era5")
    assert {"TMP2m", "PRATEsfc", "PRESsfc"} <= set(era5["variables"])
    assert {"total_water_path", "windspeed_at_10m"} <= set(era5["derived"])
    assert variables_of("not-a-model") is None and output_names("not-a-model", "fme") is None


def test_output_names_per_family():
    assert "sst" not in output_names("dlesym-aimip", "e2s-dlesym") and "t2m" in output_names("dlesym-aimip", "e2s-dlesym")
    assert "msl" in output_names("dlesym-v1.1", "e2s-dlesym")
    assert {"tas", "zg"} <= set(output_names("archesweather", "geoarches"))
    assert "total_water_path" in output_names("ace2-era5", "fme")


def test_plan_time_name_check_for_ace():
    kw = dict(model_name="ace2-era5", family="fme", is_coupled=False, timestep_hours=6, start_iso="1979-01-01T00:00:00",
              end_iso="1979-01-03T00:00:00", steps=8, calendar="proleptic_gregorian")
    with pytest.raises(ConfigError, match=r"no variable PRATEsfz \(did you mean PRATEsfc"):
        resolve_outputs(["daily:TMP2m,PRATEsfz"], **kw)
    assert resolve_outputs(["daily:total_water_path"], **kw)[0]["names"] == ["total_water_path"]


def test_show_lines():
    line = summary("ace2-era5")
    assert line.startswith("50 + 11 derived: PRESsfc, ") and "aimip show ace2-era5 --variables" in line
    listing = full_listing("dlesym-v1.1")
    assert listing[0].startswith("atmosphere: z500") and any(x.startswith("precipitation: tp6") for x in listing)
    assert any("13 levels" in x for x in full_listing("archesweather"))
