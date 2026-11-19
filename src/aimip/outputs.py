"""Extra output files of a run (`--output FREQ[:VARS][@START..END]`, repeatable).

Every run writes its monthly means (`monthly_mean_predictions.nc`) as before. Each `--output` request adds one file
with its own frequency, variables and period:

- `FREQ`: `step` (every model step, instantaneous states), `<N>h`, `daily` (= `1d`), `<N>d` (means over blocks of
  model steps, a multiple of the model step), or `monthly` (monthly means);
- `VARS`: comma-separated variable names as the model writes them (default: all variables);
- `@START..END`: only output times from START (inclusive) to END (exclusive); both aligned to the output period
  and inside the run. Not with `monthly`.

The ACE models implement the files with fme's per-file writers (`data_writer.files`); a block mean of N steps is
fme's `block_mean` over steps 1..N after the initial condition, then N+1..2N, and so on. Other families refuse the
option until their writers support it.
"""
from __future__ import annotations

import dataclasses
import math
import re
from collections.abc import Sequence
from typing import Any

from .errors import ConfigError

MAX_OUTPUTS = 8
MAX_BLOCK_STEPS = 240          # longest block mean (60 days of 6-h steps); use `monthly` beyond
DEFAULT_STEPS_IN_MEMORY = 40   # fme forward_steps_in_memory used by aimip without block means

_SPEC_RE = re.compile(
    r"^(?P<freq>[A-Za-z0-9]+)"
    r"(?::(?P<names>[A-Za-z0-9_.,-]+))?"
    r"(?:@(?P<start>[0-9T:-]+)\.\.(?P<end>[0-9T:-]+))?$"
)
_FREQ_RE = re.compile(r"^(?:(?P<n>[1-9][0-9]*)(?P<unit>[hd])|(?P<word>step|daily|monthly))$")
_NAME_RE = re.compile(r"^[A-Za-z][A-Za-z0-9_.-]*$")


@dataclasses.dataclass(frozen=True)
class OutputRequest:
    """One parsed `--output` value."""

    text: str
    freq: str                          # "step", "<N>h", "<N>d" or "monthly" (normalized; "daily" -> "1d")
    hours: int | None                  # length of one output period in hours (None for "step" and "monthly")
    names: tuple[str, ...]
    window: tuple[str, str] | None     # (start, end) as given; normalized when resolved


def parse_output(text: str) -> OutputRequest:
    """Parse one `--output` value; raises ConfigError with the accepted forms."""
    raw = str(text or "").strip()
    form = "FREQ[:VAR,VAR...][@START..END], e.g. daily, daily:TMP2m,PRATEsfc or step:TMP2m@1979-01-01..1979-02-01"
    m = _SPEC_RE.match(raw)
    if not m:
        raise ConfigError(f"Cannot read --output {raw!r}; expected {form}.", field="output")
    fm = _FREQ_RE.match(m.group("freq").lower())
    if not fm:
        raise ConfigError(
            f"Unknown output frequency {m.group('freq')!r} in --output {raw!r}: use step, <N>h, daily, <N>d or "
            "monthly.",
            field="output",
        )
    if fm.group("word") == "step":
        freq, hours = "step", None
    elif fm.group("word") == "monthly":
        freq, hours = "monthly", None
    elif fm.group("word") == "daily":
        freq, hours = "1d", 24
    else:
        n = int(fm.group("n"))
        freq = f"{n}{fm.group('unit')}"
        hours = n * (24 if fm.group("unit") == "d" else 1)
    names: list[str] = []
    for part in (m.group("names") or "").split(","):
        if not part:
            continue
        if not _NAME_RE.match(part):
            raise ConfigError(f"Invalid variable name {part!r} in --output {raw!r}.", field="output")
        if part not in names:
            names.append(part)
    window = (m.group("start"), m.group("end")) if m.group("start") else None
    if window is not None and freq == "monthly":
        raise ConfigError(
            f"--output {raw!r}: a period (@START..END) is not available for monthly output; use daily or step output "
            "for part of a run, or the monthly means of the whole run.",
            field="output",
        )
    return OutputRequest(text=raw, freq=freq, hours=hours, names=tuple(names), window=window)


def _hours_between(a: Any, b: Any) -> float:
    return (b - a).total_seconds() / 3600.0


def _period_noun(freq: str) -> str:
    if freq == "1d":
        return "days"
    return f"{freq[:-1]}-day periods" if freq.endswith("d") else f"{freq[:-1]}-hour periods"


def _period_words(freq: str, timestep_hours: int) -> tuple[str, str]:
    """(file tag, description) of an output frequency."""
    if freq == "step":
        tag = "daily" if timestep_hours == 24 else (f"{timestep_hours}h" if timestep_hours < 24 or timestep_hours % 24
                                                    else f"{timestep_hours // 24}d")
        return tag, f"states at every model step ({timestep_hours} h)"
    if freq == "monthly":
        return "monthly", "monthly means"
    if freq == "1d":
        return "daily", "daily means"
    if freq.endswith("d"):
        return freq, f"{freq[:-1]}-day means"
    return freq, f"{freq[:-1]}-hourly means"


def resolve_outputs(
    texts: Sequence[str],
    *,
    model_name: str,
    family: str,
    is_coupled: bool,
    timestep_hours: int,
    start_iso: str,
    end_iso: str,
    steps: int,
    calendar: str,
) -> list[dict[str, Any]]:
    """Check the `--output` requests against the model and the run; return one JSON-safe entry per file."""
    if not texts:
        return []
    if family != "fme" or is_coupled:
        what = "coupled atmosphere-ocean models" if is_coupled else f"the {family} models"
        raise ConfigError(
            f"--output is not available for {model_name} yet: {what} write monthly means only "
            "(the ACE models support it).",
            field="output",
        )
    if len(texts) > MAX_OUTPUTS:
        raise ConfigError(f"At most {MAX_OUTPUTS} --output files per run.", field="output")
    from .time_utils import format_cftime, parse_cftime

    t0 = parse_cftime(start_iso, calendar)
    t1 = parse_cftime(end_iso, calendar)
    run_hours = _hours_between(t0, t1)
    seen: set[str] = set()
    tags: dict[str, int] = {}
    out: list[dict[str, Any]] = []
    for text in texts:
        req = parse_output(text)
        key = f"{req.freq}|{','.join(req.names)}|{req.window}"
        if key in seen:
            raise ConfigError(f"--output {req.text!r} is given twice.", field="output")
        seen.add(key)
        if req.freq == "monthly":
            block = None
            period_hours = None
        elif req.freq == "step":
            block = 1
            period_hours = timestep_hours
        else:
            assert req.hours is not None
            if req.hours % timestep_hours:
                raise ConfigError(
                    f"--output {req.text!r}: {model_name} steps {timestep_hours} h, so the output period must be a "
                    f"multiple of {timestep_hours} h.",
                    field="output",
                )
            block = req.hours // timestep_hours
            period_hours = req.hours
            if block > MAX_BLOCK_STEPS:
                raise ConfigError(
                    f"--output {req.text!r}: means over more than {MAX_BLOCK_STEPS} model steps are not available; "
                    "use monthly.",
                    field="output",
                )
            if steps % block:
                lo, hi = steps - steps % block, steps - steps % block + block
                raise ConfigError(
                    f"--output {req.text!r} needs a whole number of {_period_noun(req.freq)} ({block} model steps "
                    f"each), but the run has {steps} steps: use --steps {lo} or --steps {hi}.",
                    field="output",
                )
        index_slice: tuple[int, int] | None = None
        window_norm: tuple[str, str] | None = None
        n_values: int | None = steps // block if block else None
        if req.window is not None:
            assert period_hours is not None and block is not None
            w0 = parse_cftime(req.window[0], calendar)
            w1 = parse_cftime(req.window[1], calendar)
            h0, h1 = _hours_between(t0, w0), _hours_between(t0, w1)
            if not (0 <= h0 < h1 <= run_hours):
                raise ConfigError(
                    f"--output {req.text!r}: the period must lie inside the run ({start_iso} .. {end_iso}) and END "
                    "must be after START.",
                    field="output",
                )
            if h0 % period_hours or h1 % period_hours:
                raise ConfigError(
                    f"--output {req.text!r}: START and END must fall on the boundaries of the "
                    f"{'model steps' if req.freq == 'step' else _period_noun(req.freq)}, counted from the start of the "
                    f"run ({start_iso}, every {period_hours} h).",
                    field="output",
                )
            index_slice = (int(h0 // period_hours), int(h1 // period_hours))
            window_norm = (format_cftime(w0), format_cftime(w1))
            n_values = index_slice[1] - index_slice[0]
        tag, description = _period_words(req.freq, timestep_hours)
        tags[tag] = tags.get(tag, 0) + 1
        label = f"output_{tag}" if tags[tag] == 1 else f"output_{tag}_{tags[tag]}"
        out.append({
            "request": req.text,
            "label": label,
            "file": f"{label}.nc",
            "freq": req.freq,
            "description": description,
            "block_steps": block,
            "names": list(req.names),
            "window": list(window_norm) if window_norm else None,
            "index_slice": list(index_slice) if index_slice else None,
            "values_per_member": n_values,
        })
    return out


def fme_file_writers(resolved: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    """`data_writer.files` entries (fme `FileWriterConfig`) for resolved outputs."""
    files: list[dict[str, Any]] = []
    for item in resolved:
        entry: dict[str, Any] = {"label": item["label"], "save_reference": False}
        if item["names"]:
            entry["names"] = list(item["names"])
        if item["freq"] == "monthly":
            entry["time_coarsen"] = {"method": "monthly_mean"}
        elif int(item["block_steps"]) > 1:
            entry["time_coarsen"] = {"coarsen_factor": int(item["block_steps"])}
        if item["index_slice"]:
            entry["time_selection"] = {"start": int(item["index_slice"][0]), "stop": int(item["index_slice"][1])}
        files.append(entry)
    return files


def steps_in_memory(steps: int, resolved: Sequence[dict[str, Any]]) -> int:
    """fme forward_steps_in_memory: divisible by every block length (fme's requirement), near the default."""
    blocks = [int(r["block_steps"]) for r in resolved if r.get("block_steps") and int(r["block_steps"]) > 1]
    if not blocks:
        return min(DEFAULT_STEPS_IN_MEMORY, steps)
    lcm = 1
    for b in blocks:
        lcm = lcm * b // math.gcd(lcm, b)
    if lcm > MAX_BLOCK_STEPS:
        raise ConfigError(
            f"The --output periods combine to blocks of {lcm} model steps, more than {MAX_BLOCK_STEPS}; use fewer "
            "or compatible periods (for example daily and 5d).",
            field="output",
        )
    per = max(lcm, (DEFAULT_STEPS_IN_MEMORY // lcm) * lcm)
    return min(per, steps)


def describe_outputs(resolved: Sequence[dict[str, Any]], *, members: int, grid_cells: int | None) -> list[str]:
    """One plan-card line per file, with the number of values and, when known, its size."""
    lines = []
    for item in resolved:
        what = ", ".join(item["names"]) if item["names"] else "all variables"
        span = f"{item['window'][0]} .. {item['window'][1]}" if item["window"] else "whole run"
        n = item["values_per_member"]
        count = "one value per month" if n is None else f"{n} per member"
        size = ""
        if n is not None and item["names"] and grid_cells:
            nbytes = n * len(item["names"]) * grid_cells * 4 * max(1, members)
            size = f", ~{nbytes / 2**30:.2f} GiB" if nbytes >= 2**30 else f", ~{nbytes / 2**20:.0f} MiB"
        lines.append(f"{item['file']}: {item['description']} of {what}, {span} ({count}{size})")
    return lines


def check_output_files(run_dir: str, resolved: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """After the job: each requested file exists and holds the requested variables.

    fme skips names a model does not write without an error, so a misspelt name would otherwise go unnoticed."""
    import os

    report: dict[str, Any] = {"ok": True, "files": []}
    for item in resolved:
        path = os.path.join(run_dir, item["file"])
        entry: dict[str, Any] = {"file": item["file"], "exists": os.path.isfile(path)}
        if entry["exists"]:
            try:
                import netCDF4

                with netCDF4.Dataset(path) as ds:
                    present = set(ds.variables)
                    entry["times"] = int(ds.dimensions["time"].size) if "time" in ds.dimensions else None
                missing = [n for n in item["names"] if n not in present]
                entry["missing_variables"] = missing
            except Exception as exc:  # noqa: BLE001 - recorded, never fails a finished run
                entry["error"] = f"{type(exc).__name__}: {exc}"
        ok = entry["exists"] and not entry.get("missing_variables") and "error" not in entry
        report["ok"] = report["ok"] and ok
        report["files"].append(entry)
        if not ok:
            detail = ("missing" if not entry["exists"] else
                      f"without {', '.join(entry['missing_variables'])}" if entry.get("missing_variables") else
                      entry.get("error", ""))
            print(f"aimip warning: output file {item['file']} {detail} (requested --output {item['request']})",
                  flush=True)
    return report
