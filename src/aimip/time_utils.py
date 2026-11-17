"""Calendar-aware timestamp, forcing-set selection, and spin-step utilities backed by cftime."""

from __future__ import annotations

import datetime
import re
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

from .errors import ConfigError, ValidationError

_LENGTH_SPEC_RE = re.compile(r"\A(?P<count>[1-9]\d{0,6})(?P<unit>[ymdh])\Z")
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
        last_fmt = format_cftime(first_last)
        next_cmd = (
            f"aimip plan {model_name} --start {start_fmt} --end {last_fmt}"
            if model_name
            else None
        )
        raise ConfigError(
            f"Refusing run: end time {end_fmt} (start {start_fmt} + {steps} steps * "
            f"{timestep_hours}h, calendar={calendar}) exceeds forcing set {first_fs.name!r} "
            f"end {last_fmt} (maximum steps in {first_fs.name!r} from "
            f"{start_fmt} is {max_steps}; use `--end {last_fmt}` or `--steps {max_steps}`; "
            f"available sets: {sets_desc}). "
            "If continuing into another forcing set, split the run at the boundary and "
            "continue from `<prev-dir>/restart.nc` via `--ic <prev-dir>/restart.nc`.",
            field="end",
            next_command=next_cmd,
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


def parse_length_spec(raw: str) -> tuple[int, str]:
    """Parse a human duration string such as `'20y'`, `'6m'`, `'30d'`, or `'12h'`."""
    if not isinstance(raw, str) or not raw.strip():
        raise ValidationError(
            "Invalid --length: expected '<N>y', '<N>m', '<N>d', or '<N>h' (e.g. '20y', '6m', '30d', '12h').",
            field="length",
            allowed=["<N>y", "<N>m", "<N>d", "<N>h"],
        )
    text = raw.strip()
    m = _LENGTH_SPEC_RE.match(text)
    if not m:
        raise ValidationError(
            f"Invalid --length {raw!r}: expected positive integer followed by y, m, d, or h (e.g. '20y', '6m', '30d', '12h').",
            field="length",
            allowed=["<N>y", "<N>m", "<N>d", "<N>h"],
        )
    return int(m.group("count")), m.group("unit")


def add_calendar_months(dt: Any, months: int, calendar: str) -> Any:
    """Advance `cftime.datetime` *dt* by *months* calendar months in *calendar*, refusing invalid target dates."""
    import cftime

    total_m = (int(dt.year) * 12 + (int(dt.month) - 1)) + int(months)
    if total_m < 0:
        raise ValidationError(
            f"Target year/month out of range when advancing {format_cftime(dt)} by {months} months.",
            field="length",
        )
    target_year, target_month_zero = divmod(total_m, 12)
    target_month = target_month_zero + 1
    if target_year < 1:
        raise ValidationError(
            f"Target year {target_year} out of range when advancing {format_cftime(dt)} by {months} months.",
            field="length",
        )
    try:
        return cftime.datetime(
            target_year,
            target_month,
            int(dt.day),
            int(dt.hour),
            int(dt.minute),
            int(dt.second),
            calendar=calendar,
        )
    except ValueError as exc:
        raise ValidationError(
            f"Duration cannot advance {format_cftime(dt)} to {target_year:04d}-{target_month:02d}-{int(dt.day):02d} "
            f"in calendar {calendar!r} ({exc}); pass an explicit --end or --steps instead.",
            field="length",
        ) from exc


def resolve_horizon(
    *,
    start_iso: str,
    steps: int | None = None,
    length: str | None = None,
    end_iso: str | None = None,
    timestep_hours: int = 6,
    calendar: str = "proleptic_gregorian",
    family: str = "fme",
    model_name: str = "",
) -> tuple[int, str]:
    """Resolve exactly one of `(steps, length, end_iso)` into `(resolved_steps, resolved_end_iso)`.

    Rules:
    - Exactly one of `steps`, `length`, or `end_iso` must be provided.
    - Non-integral spans relative to the model's `timestep_hours` and `calendar` are refused.
    - Monthly/yearly lengths require exact calendar date existence in `calendar` AND exact divisibility
      by `timestep_hours` (or monthly stepping for `family == 'arches'`).
    """
    provided = [
        name
        for name, val in (("steps", steps), ("length", length), ("end", end_iso))
        if val is not None
    ]
    if len(provided) == 0:
        raise ValidationError(
            "Specify run horizon using one of --steps <N>, --length <Ny|Nm|Nd|Nh>, or --end <ISO>.",
            field="steps",
            allowed=["--steps", "--length", "--end"],
            next_command=(
                f"aimip plan {model_name} --start {start_iso} --length 1y"
                if model_name
                else None
            ),
        )
    if len(provided) > 1:
        raise ValidationError(
            f"Options {', '.join('--' + p for p in provided)} are mutually exclusive; pass only one.",
            field=provided[1],
            allowed=["--steps", "--length", "--end"],
        )

    start_dt = parse_cftime(start_iso, calendar=calendar)
    validate_int_range(timestep_hours, "timestep_hours", min_value=1, max_value=24 * 366)
    step_td = datetime.timedelta(hours=timestep_hours)

    # Monthly-stepping models (`archesclimate-ssp`, `md-1.5`) integrate whole calendar months.
    # For `arches`, starting at 2015-01-01 uses 2015-01 and 2015-02 as lagged states so step 1 outputs 2015-03-01;
    # for `md-1.5`, step 0 outputs `start_dt` and step N-1 outputs `start_dt + (N - 1) months`.
    if family in ("arches", "md"):
        first_output_dt = (
            add_calendar_months(start_dt, 2, calendar) if family == "arches" else start_dt
        )
        if steps is not None:
            resolved_steps = validate_int_range(steps, "steps", min_value=1, max_value=10_000_000)
            end_dt = add_calendar_months(first_output_dt, resolved_steps - 1, calendar)
            return resolved_steps, format_cftime(end_dt)
        if length is not None:
            count, unit = parse_length_spec(length)
            if unit not in ("m", "y"):
                raise ValidationError(
                    f"Model {model_name or family!r} integrates in whole calendar monthly steps; "
                    f"--length {length!r} with unit {unit!r} is not integral (use '<N>m' or '<N>y').",
                    field="length",
                    allowed=["<N>m", "<N>y"],
                )
            resolved_steps = count if unit == "m" else count * 12
            end_dt = add_calendar_months(first_output_dt, resolved_steps - 1, calendar)
            return resolved_steps, format_cftime(end_dt)
        assert end_iso is not None
        end_dt = parse_cftime(end_iso, calendar=calendar)
        if end_dt < first_output_dt:
            raise ValidationError(
                f"--end {format_cftime(end_dt)} must be at or after the first output month "
                f"{format_cftime(first_output_dt)} for {model_name or family!r}.",
                field="end",
            )
        if (
            int(end_dt.day) != int(start_dt.day)
            or int(end_dt.hour) != int(start_dt.hour)
            or int(end_dt.minute) != int(start_dt.minute)
            or int(end_dt.second) != int(start_dt.second)
        ):
            raise ValidationError(
                f"Span from {format_cftime(start_dt)} to {format_cftime(end_dt)} is not an exact whole number "
                f"of calendar monthly steps for {model_name or family!r}.",
                field="end",
            )
        resolved_steps = (
            (int(end_dt.year) - int(first_output_dt.year)) * 12
            + (int(end_dt.month) - int(first_output_dt.month))
            + 1
        )
        return resolved_steps, format_cftime(end_dt)

    # Fixed-hour / multi-day stepping (`6h`, `120h` = 5d, etc.) in `calendar`.
    if steps is not None:
        resolved_steps = validate_int_range(steps, "steps", min_value=1, max_value=10_000_000)
        end_dt = start_dt + (step_td * resolved_steps)
        return resolved_steps, format_cftime(end_dt)

    if end_iso is not None:
        end_dt = parse_cftime(end_iso, calendar=calendar)
        if end_dt <= start_dt:
            raise ValidationError(
                f"--end {format_cftime(end_dt)} must be strictly after --start {format_cftime(start_dt)}.",
                field="end",
            )
        delta = end_dt - start_dt
        if delta % step_td != datetime.timedelta(0):
            total_hours = delta.total_seconds() / 3600.0
            raise ValidationError(
                f"Span from {format_cftime(start_dt)} to {format_cftime(end_dt)} ({total_hours:g}h in calendar "
                f"{calendar!r}) is not an exact multiple of model step ({timestep_hours}h).",
                field="end",
            )
        resolved_steps = int(delta // step_td)
        return resolved_steps, format_cftime(end_dt)

    assert length is not None
    count, unit = parse_length_spec(length)
    if unit in ("h", "d"):
        total_hours_int = count if unit == "h" else count * 24
        if total_hours_int % timestep_hours != 0:
            raise ValidationError(
                f"--length {length!r} ({total_hours_int}h) is not an exact multiple of model step ({timestep_hours}h).",
                field="length",
            )
        resolved_steps = total_hours_int // timestep_hours
        end_dt = start_dt + (step_td * resolved_steps)
        return resolved_steps, format_cftime(end_dt)

    months = count if unit == "m" else count * 12
    end_dt = add_calendar_months(start_dt, months, calendar)
    delta = end_dt - start_dt
    if delta % step_td != datetime.timedelta(0):
        total_hours = delta.total_seconds() / 3600.0
        raise ValidationError(
            f"--length {length!r} from {format_cftime(start_dt)} to {format_cftime(end_dt)} "
            f"({total_hours:g}h in calendar {calendar!r}) is not an exact multiple of model step ({timestep_hours}h); "
            "use a divisible --length (e.g. days/years) or explicit --steps.",
            field="length",
        )
    resolved_steps = int(delta // step_td)
    return resolved_steps, format_cftime(end_dt)
