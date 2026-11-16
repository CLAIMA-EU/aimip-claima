"""Calendar-aware timestamp, forcing-set selection, and spin-step utilities backed by cftime."""

from __future__ import annotations

import datetime
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

from .errors import ConfigError, ValidationError
from .validators import (
    validate_forcing_set_name,
    validate_int_range,
    validate_iso_timestamp,
)

if TYPE_CHECKING:
    from .registry import ForcingSet


def parse_cftime(iso_value: str, calendar: str = "proleptic_gregorian") -> Any:
    """Parse an ISO timestamp string into a `cftime.datetime` in *calendar*.

    Validates calendar-specific day bounds (e.g., rejects Feb 29 on `noleap` / `365_day`
    calendars and non-leap years on `proleptic_gregorian` / `standard`).
    """
    import cftime

    norm = validate_iso_timestamp(iso_value)
    year = int(norm[0:4])
    month = int(norm[5:7])
    day = int(norm[8:10])
    hour = int(norm[11:13])
    minute = int(norm[14:16])
    second = int(norm[17:19])

    try:
        return cftime.datetime(
            year, month, day, hour, minute, second, calendar=calendar
        )
    except ValueError as exc:
        raise ValidationError(
            f"Timestamp {norm!r} is invalid for calendar {calendar!r}: {exc}"
        ) from exc


def format_cftime(dt: Any) -> str:
    """Format a `cftime.datetime` (or `datetime.datetime` / `np.datetime64`) as 'YYYY-MM-DDTHH:MM:SS'."""
    if not hasattr(dt, "year"):
        return validate_iso_timestamp(str(dt)[:19], label="decoded timestamp")
    return (
        f"{int(dt.year):04d}-{int(dt.month):02d}-{int(dt.day):02d}"
        f"T{int(dt.hour):02d}:{int(dt.minute):02d}:{int(dt.second):02d}"
    )


def check_within_window(
    start_iso: str,
    steps: int,
    timestep_hours: int,
    window: tuple[str, str],
    calendar: str = "proleptic_gregorian",
) -> str:
    """Verify `start` and `start + steps * timestep_hours` stay within a single `window`."""
    validate_int_range(steps, "steps", min_value=1, max_value=10_000_000)
    validate_int_range(timestep_hours, "timestep_hours", min_value=1, max_value=24 * 366)

    win_first = parse_cftime(window[0], calendar=calendar)
    win_last = parse_cftime(window[1], calendar=calendar)
    start_dt = parse_cftime(start_iso, calendar=calendar)

    if start_dt < win_first:
        raise ConfigError(
            f"Refusing run: start time {format_cftime(start_dt)} is earlier than "
            f"valid window start {format_cftime(win_first)} "
            f"(window: [{window[0]}, {window[1]}])."
        )
    if start_dt > win_last:
        raise ConfigError(
            f"Refusing run: start time {format_cftime(start_dt)} is after "
            f"valid window end {format_cftime(win_last)} "
            f"(window: [{window[0]}, {window[1]}])."
        )

    step_td = datetime.timedelta(hours=timestep_hours)
    end_dt = start_dt + (step_td * steps)

    if end_dt > win_last:
        max_steps = (win_last - start_dt) // step_td
        raise ConfigError(
            f"Refusing run: end time {format_cftime(end_dt)} "
            f"(start {format_cftime(start_dt)} + {steps} steps * {timestep_hours}h, "
            f"calendar={calendar}) exceeds valid window end {format_cftime(win_last)}; "
            f"maximum steps from {format_cftime(start_dt)} is {max_steps}."
        )

    return format_cftime(end_dt)


def select_forcing_set(
    start_iso: str,
    steps: int,
    timestep_hours: int,
    forcing_sets: Sequence[ForcingSet],
    *,
    calendar: str = "proleptic_gregorian",
    model_name: str = "",
    preferred_set_name: str | None = None,
) -> tuple[ForcingSet, str]:
    """Pick the single `ForcingSet` whose `window` covers `[start, start + steps * timestep_hours]`.

    If no single forcing set covers the interval (for instance when a run crosses from
    `amip-hist` into `amip-oos`, or exceeds the latest set), raises `ConfigError`
    explaining why and instructing the user how to split the run and continue from
    `<prev_dir>/restart.nc` using `aimip run ... --ic <prev_dir>/restart.nc`.
    """
    validate_int_range(steps, "steps", min_value=1, max_value=10_000_000)
    validate_int_range(timestep_hours, "timestep_hours", min_value=1, max_value=24 * 366)
    if not forcing_sets:
        raise ConfigError(f"Model {model_name!r} has no forcing_sets configured.")

    start_dt = parse_cftime(start_iso, calendar=calendar)
    step_td = datetime.timedelta(hours=timestep_hours)
    end_dt = start_dt + (step_td * steps)
    start_fmt = format_cftime(start_dt)
    end_fmt = format_cftime(end_dt)

    if preferred_set_name is not None:
        validate_forcing_set_name(preferred_set_name)
        matched_by_name = [fs for fs in forcing_sets if fs.name == preferred_set_name]
        if not matched_by_name:
            matched_by_name = [
                fs for fs in forcing_sets if fs.name.lower() == preferred_set_name.lower()
            ]
        if not matched_by_name:
            known = ", ".join(fs.name for fs in forcing_sets)
            raise ConfigError(
                f"Unknown forcing set {preferred_set_name!r} for model {model_name!r} "
                f"(available: {known})."
            )
        candidates = matched_by_name
    else:
        candidates = list(forcing_sets)

    parsed_bounds: list[tuple[ForcingSet, Any, Any]] = []
    for fs in candidates:
        w_first = parse_cftime(fs.window[0], calendar=calendar)
        w_last = parse_cftime(fs.window[1], calendar=calendar)
        parsed_bounds.append((fs, w_first, w_last))
        if w_first <= start_dt and end_dt <= w_last:
            return fs, end_fmt

    sets_desc = "; ".join(
        f"{fs.name} [{fs.window[0]} .. {fs.window[1]}]" for fs in forcing_sets
    )

    # Check if `start_dt` lives in an earlier forcing set while `end_dt` crosses past it.
    covering_start = [
        (fs, w_first, w_last)
        for fs, w_first, w_last in parsed_bounds
        if w_first <= start_dt <= w_last
    ]
    covering_end = [
        (fs, w_first, w_last)
        for fs, w_first, w_last in parsed_bounds
        if w_first <= end_dt <= w_last
    ]

    if covering_start and covering_end:
        first_fs, _, first_last = covering_start[0]
        second_fs, second_first, _ = covering_end[0]
        # Suggest splitting at second_first if start_dt < second_first <= first_last, else first_last.
        split_dt = (
            second_first
            if (start_dt < second_first <= first_last)
            else first_last
        )
        seg1_steps = int((split_dt - start_dt) // step_td)
        seg2_steps = steps - seg1_steps
        split_fmt = format_cftime(split_dt)
        model_flag = f"-m {model_name} " if model_name else ""
        raise ConfigError(
            f"Refusing run: requested span [{start_fmt} .. {end_fmt}] ({steps} steps, "
            f"calendar={calendar}) crosses forcing sets ({first_fs.name} -> {second_fs.name}; "
            f"available sets: {sets_desc}). No single forcing directory covers the whole run. "
            f"Split the run into two segments and continue from restart.nc:\n"
            f"  1) aimip run {model_flag}--start {start_fmt} --steps {seg1_steps} --out <seg1-dir>\n"
            f"  2) aimip run {model_flag}--start {split_fmt} --steps {seg2_steps} "
            f"--ic <seg1-dir>/restart.nc --out <seg2-dir>"
        )

    if covering_start:
        first_fs, _, first_last = covering_start[0]
        max_steps = int((first_last - start_dt) // step_td)
        raise ConfigError(
            f"Refusing run: end time {end_fmt} (start {start_fmt} + {steps} steps * "
            f"{timestep_hours}h, calendar={calendar}) exceeds forcing set {first_fs.name!r} "
            f"end {format_cftime(first_last)} (maximum steps in {first_fs.name!r} from "
            f"{start_fmt} is {max_steps}; available sets: {sets_desc}). "
            "If continuing into another forcing set, split the run at the boundary and "
            "continue from `<prev-dir>/restart.nc` via `--ic <prev-dir>/restart.nc`."
        )

    raise ConfigError(
        f"Refusing run: start time {start_fmt} (end {end_fmt}, calendar={calendar}) is not "
        f"covered by any forcing set for model {model_name!r} (available sets: {sets_desc})."
    )


def compute_spin_steps(
    from_iso: str,
    to_iso: str,
    timestep_hours: int,
    calendar: str = "proleptic_gregorian",
) -> int:
    """Compute exact `n_forward_steps = (to - from) / timestep_hours` in *calendar*."""
    validate_int_range(timestep_hours, "timestep_hours", min_value=1, max_value=24 * 366)
    from_dt = parse_cftime(from_iso, calendar=calendar)
    to_dt = parse_cftime(to_iso, calendar=calendar)

    if from_dt >= to_dt:
        raise ConfigError(
            f"Spin start time {format_cftime(from_dt)} must be strictly earlier than "
            f"target time {format_cftime(to_dt)}."
        )

    delta: datetime.timedelta = to_dt - from_dt
    step_td = datetime.timedelta(hours=timestep_hours)
    if delta % step_td != datetime.timedelta(0):
        raise ConfigError(
            f"Time span from {format_cftime(from_dt)} to {format_cftime(to_dt)} ({delta}) "
            f"is not an exact multiple of model timestep ({timestep_hours}h)."
        )

    steps = int(delta // step_td)
    if steps < 1:
        raise ConfigError("Spin run requires at least 1 forward step.")
    return steps
