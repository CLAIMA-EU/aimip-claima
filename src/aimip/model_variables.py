"""The output variables of each model, from model_variables.json (written by tools/model_variables/collect.py).

Per model: `variables` (what the model writes; for the fme models the checkpoint's `out_names`), and for the fme
models `derived` (fme's derived quantities, written when their inputs exist), `groups` (DLESyM and ArchesClimate:
atmosphere / ocean / precipitation), `units` and `levels` where known, and `source` (where the names come from).
"""
from __future__ import annotations

import functools
import json
import os
from typing import Any

_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "model_variables.json")


@functools.lru_cache(maxsize=1)
def _models() -> dict[str, Any]:
    with open(_PATH, encoding="utf-8") as fh:
        return json.load(fh)["models"]


def variables_of(model: str) -> dict[str, Any] | None:
    """The entry of *model*, or None if the model is not listed."""
    entry = _models().get(model)
    return dict(entry) if entry is not None else None


def output_names(model: str, family: str) -> list[str] | None:
    """The names `--output` accepts for *model*: what its files can hold. None if unknown (no check before the run)."""
    entry = _models().get(model)
    if entry is None:
        return None
    if family == "e2s-dlesym":                  # aimip-claima writes DLESyM's atmosphere states only
        return list((entry.get("groups") or {}).get("atmosphere") or entry["variables"])
    return list(entry["variables"]) + list(entry.get("derived") or [])


def summary(model: str, *, limit: int = 8) -> str | None:
    """One line for `aimip show`: the count and the first names."""
    entry = _models().get(model)
    if entry is None:
        return None
    names = list(entry["variables"])
    derived = list(entry.get("derived") or [])
    count = f"{len(names)}" + (f" + {len(derived)} derived" if derived else "")
    head = ", ".join(names[:limit]) + (", …" if len(names) > limit else "")
    return f"{count}: {head} (aimip show {model} --variables lists all)"


def full_listing(model: str) -> list[str]:
    """All names of *model*, grouped, for `aimip show --variables`."""
    entry = _models().get(model)
    if entry is None:
        return ["(not recorded)"]
    units = entry.get("units") or {}
    levels = entry.get("levels") or {}

    def fmt(n: str) -> str:
        extra = [units[n]] if units.get(n) else []
        if levels.get(n):
            extra.append(f"{len(levels[n])} levels")
        return f"{n} ({', '.join(extra)})" if extra else n

    lines = []
    groups = entry.get("groups")
    if groups:
        for g, names in groups.items():
            if names:
                lines.append(f"{g}: " + ", ".join(fmt(n) for n in names))
    else:
        lines.append(", ".join(fmt(n) for n in entry["variables"]))
    if entry.get("derived"):
        lines.append("derived (written when their inputs exist): " + ", ".join(entry["derived"]))
    lines.append(f"source: {entry.get('source', '')}")
    return lines
