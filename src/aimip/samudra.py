# Model architecture classes (CappedGELU, CoreBlock, ConvNeXtBlock, AvgPool,
# BilinearUpsample, TransposedConvUpsample, UNet) adapted from m2lines/Samudra
# (https://github.com/m2lines/Samudra) at licensed commit
# 0c2a1bd8be6260fe02507fbb697ef50be8835fe6 (2025-02-02).
#
# Copyright 2025 Surya Dheeshjith and Samudra Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Modifications by the AIMIP contributors: consolidated CappedGELU, CoreBlock,
# ConvNeXtBlock, AvgPool, BilinearUpsample, TransposedConvUpsample, and UNet
# from src/models/{unet.py,modules/blocks.py,modules/activations.py} at commit
# 0c2a1bd8be6260fe02507fbb697ef50be8835fe6 into a self-contained inference module
# with lazy PyTorch imports, O_NOFOLLOW SHA-256 pre-verified checkpoint loading,
# OSN Zarr window staging, and Julian-calendar overlap-day monthly NetCDF output.
"""Samudra global ocean emulator of GFDL OM4 (`samudra-om4`, `family="samudra"`)."""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Callable, Mapping, Sequence
import contextlib
import json
import os
import sys
import tempfile
import time
from typing import Any

from .safe_pickle import extract_verified_zip_store, torch_load_verified
from .validators import makedirs_no_group_other_write, strip_group_other_write

SAMUDRA_SOURCE_REPO = "https://github.com/m2lines/Samudra"
SAMUDRA_LICENSED_COMMIT = "0c2a1bd8be6260fe02507fbb697ef50be8835fe6"
SAMUDRA_CODE_LICENCE = "Apache-2.0"

SAMUDRA_OSN_SOURCE = "s3://m2lines-pubs/Samudra"
SAMUDRA_OSN_ENDPOINT = "https://nyu1.osn.mghpcc.org"
SAMUDRA_WINDOW: tuple[str, str] = ("1975-01-08T12:00:00", "2022-12-29T12:00:00")
SAMUDRA_TRAINING_END = "2014-09-20T12:00:00"
SAMUDRA_INDEPENDENT_TEST_START = "2014-09-30T12:00:00"
SAMUDRA_BYTES_PER_STEP = 20_995_200
SAMUDRA_SECONDS_PER_YEAR = 18.0

SAMUDRA_LEVELS: tuple[str, ...] = (
    "2_5",
    "10_0",
    "22_5",
    "40_0",
    "65_0",
    "105_0",
    "165_0",
    "250_0",
    "375_0",
    "550_0",
    "775_0",
    "1050_0",
    "1400_0",
    "1850_0",
    "2400_0",
    "3100_0",
    "4000_0",
    "5000_0",
    "6000_0",
)
SAMUDRA_DEPTH_METERS: tuple[float, ...] = tuple(
    float(lv.replace("_", ".")) for lv in SAMUDRA_LEVELS
)
SAMUDRA_STATE_VARS: tuple[str, ...] = tuple(
    f"{k}_lev_{j}" for k in ("uo", "vo", "thetao", "so") for j in SAMUDRA_LEVELS
) + ("zos",)
SAMUDRA_BOUNDARY_VARS: tuple[str, ...] = (
    "tauuo",
    "tauvo",
    "hfds",
    "hfds_anomalies",
)
SAMUDRA_HIST = 1  # two 5-day input states -> two 5-day output states per model call
SAMUDRA_INPUT_STEPS = SAMUDRA_HIST + 1
SAMUDRA_CH_WIDTH: tuple[int, ...] = (
    len(SAMUDRA_STATE_VARS) * SAMUDRA_INPUT_STEPS + len(SAMUDRA_BOUNDARY_VARS),
    200,
    250,
    300,
    400,
)  # (158, 200, 250, 300, 400)
SAMUDRA_DILATION: tuple[int, ...] = (1, 2, 4, 8)
SAMUDRA_N_OUT = len(SAMUDRA_STATE_VARS) * SAMUDRA_INPUT_STEPS  # 154
SAMUDRA_REMOTE_TRIES = 5


def level_of(var: str) -> int:
    """Return the depth-level index (`0..18`) of a flat Samudra state variable (`0` for surface `zos`)."""
    return SAMUDRA_LEVELS.index(var.split("lev_")[-1]) if "_lev_" in var else 0


def julian_is_leap(year: int) -> bool:
    """Return True when *year* is a leap year on OM4's Julian calendar (`year % 4 == 0`)."""
    return int(year) % 4 == 0


def julian_days_in_month(year: int, month: int) -> int:
    """Return the number of calendar days in `(year, month)` on the Julian calendar."""
    m = int(month)
    if not 1 <= m <= 12:
        raise ValueError(f"Invalid month {month!r}")
    if m == 2:
        return 29 if julian_is_leap(year) else 28
    if m in (4, 6, 9, 11):
        return 30
    return 31


def julian_step_days(year: int, month: int, day: int, offset_days: int) -> tuple[int, int, int]:
    """Step `(year, month, day)` by *offset_days* directly on the Julian calendar without day-number conversion."""
    y, m, d = int(year), int(month), int(day)
    rem = int(offset_days)
    while rem > 0:
        dim = julian_days_in_month(y, m)
        if d + rem <= dim:
            d += rem
            rem = 0
        else:
            rem -= dim - d + 1
            d = 1
            m += 1
            if m > 12:
                m = 1
                y += 1
    while rem < 0:
        if d + rem >= 1:
            d += rem
            rem = 0
        else:
            rem += d
            m -= 1
            if m < 1:
                m = 12
                y -= 1
            d = julian_days_in_month(y, m)
    return y, m, d


def julian_date_parts(ts: Any) -> tuple[int, int, int]:
    """Extract `(year, month, day)` directly from a `cftime.DatetimeJulian` or ISO string.

    Never converts through epoch day counts, which would shift 1901-2099 dates by 13 days.
    """
    if hasattr(ts, "year") and hasattr(ts, "month") and hasattr(ts, "day"):
        return int(ts.year), int(ts.month), int(ts.day)
    text = str(ts).strip()
    return int(text[0:4]), int(text[5:7]), int(text[8:10])


def julian_state_overlap_months(
    year: int,
    month: int,
    day: int,
    *,
    span_days: int = 5,
) -> list[tuple[str, int]]:
    """Return `[('YYYY-MM', overlap_days), ...]` for one 5-day state (or 6-day `Dec 28 -> Jan 3` leap-year end step) at `(year, month, day)`."""
    if span_days not in (5, 6):
        raise ValueError(f"Unsupported Samudra state span_days={span_days!r} (expected 5 or 6).")
    start_offset = -2 if span_days == 5 else -3
    counts: OrderedDict[str, int] = OrderedDict()
    for off in range(start_offset, 3):
        yy, mm, _ = julian_step_days(year, month, day, off)
        ym = f"{yy:04d}-{mm:02d}"
        counts[ym] = counts.get(ym, 0) + 1
    return list(counts.items())


def _time_delta_days(t_curr: Any, t_prev: Any) -> float:
    diff = t_curr - t_prev
    if hasattr(diff, "days") and hasattr(diff, "seconds"):
        return float(diff.days) + float(diff.seconds) / 86400.0
    import numpy as np

    return float((np.datetime64(str(t_curr)[:19]) - np.datetime64(str(t_prev)[:19])) / np.timedelta64(1, "s")) / 86400.0


def aggregate_samudra_5day_to_monthly(
    times: Sequence[Any],
    fields_by_name: Mapping[str, Any],
    *,
    prev_time: Any | None = None,
    complete_months_only: bool = True,
) -> list[dict[str, Any]]:
    """Aggregate 5-day mean states into calendar months using exact overlap-day weights (`w_d / days_in_month`).

    Copies `(year, month, day)` labels directly from OM4's Julian timestamps without converting
    through day numbers. When `complete_months_only=True` (default), returns only months whose
    covered days equal `julian_days_in_month(year, month)`.
    """
    import numpy as np

    n_steps = len(times)
    if n_steps == 0:
        return []
    for name, arr in fields_by_name.items():
        if np.asarray(arr).shape[0] != n_steps:
            raise ValueError(
                f"Field {name!r} leading time length {np.asarray(arr).shape[0]} != {n_steps}"
            )

    covered_days: OrderedDict[str, int] = OrderedDict()
    weighted_sums: dict[str, dict[str, Any]] = {}

    for idx, ts in enumerate(times):
        ref_prev = prev_time if idx == 0 else times[idx - 1]
        span_days = 5
        if ref_prev is not None:
            dt_d = _time_delta_days(ts, ref_prev)
            if abs(dt_d - 6.0) < 0.25:
                span_days = 6
        y, m, d = julian_date_parts(ts)
        for ym, w_days in julian_state_overlap_months(y, m, d, span_days=span_days):
            covered_days[ym] = covered_days.get(ym, 0) + w_days
            bucket = weighted_sums.setdefault(ym, {})
            for name, arr in fields_by_name.items():
                slab = np.asarray(arr[idx], dtype=np.float64)
                if name not in bucket:
                    bucket[name] = float(w_days) * slab
                else:
                    bucket[name] = bucket[name] + float(w_days) * slab

    rows: list[dict[str, Any]] = []
    for ym, cov in covered_days.items():
        yy, mm = int(ym[:4]), int(ym[5:7])
        dim_days = julian_days_in_month(yy, mm)
        if complete_months_only and cov != dim_days:
            continue
        denom = float(dim_days if cov == dim_days else cov)
        mean_fields = {
            name: (acc / denom).astype(np.float32)
            for name, acc in weighted_sums[ym].items()
        }
        rows.append(
            {
                "month": ym,
                "fields": mean_fields,
                "covered_days": int(cov),
                "days_in_month": int(dim_days),
            }
        )
    return rows


def rollout(
    step_fn: Callable[[Any, Any], Any],
    sanitize_fn: Callable[[Any], tuple[Any, int, int]],
    states0: Any,
    bounds: Sequence[Any],
) -> tuple[list[Any], list[dict[str, int]]]:
    """Autoregressive Samudra loop (`BaseModel.inference` / `generate_model_rollout`).

    Feeds the raw normalized prediction directly into the next step and sanitizes only
    the stored copy (`NaN -> 0`, clip `+-1e5` in normalized space before denormalization).
    """
    states = states0
    stored: list[Any] = []
    counts: list[dict[str, int]] = []
    for bound in bounds:
        raw = step_fn(states, bound)
        clean, n_nonfinite, n_clipped = sanitize_fn(raw)
        stored.append(clean)
        counts.append({"nonfinite": int(n_nonfinite), "clipped": int(n_clipped)})
        states = raw
    return stored, counts


def check_samudra_window(
    facts: Mapping[str, Any],
    time_keys: Sequence[str],
    day_offsets: Sequence[float],
    start: str,
    steps: int,
) -> None:
    """Validate a staged Samudra window (`start`, length, Julian calendar, and 5-day OM4 cadence plus the 6-day `Dec 28 -> Jan 3` step at the end of leap years)."""
    norm_key = lambda s: str(s)[:16].replace(" ", "T")
    if norm_key(facts.get("start")) != norm_key(start):
        raise SystemExit(
            f"Staged window was prepared for start {facts.get('start')!r}, but this run requested {start!r}."
        )
    if len(time_keys) < SAMUDRA_INPUT_STEPS + steps:
        raise SystemExit(
            f"Staged window holds {len(time_keys)} states; {SAMUDRA_INPUT_STEPS + steps} are required "
            f"({SAMUDRA_INPUT_STEPS} input states + {steps} target states)."
        )
    if norm_key(time_keys[SAMUDRA_HIST]) != norm_key(start):
        raise SystemExit(
            f"Staged window's last input state is {time_keys[SAMUDRA_HIST]!r}, expected requested start {start!r}."
        )
    if str(facts.get("calendar", "")).lower() != "julian":
        raise SystemExit(
            f"Staged window declares calendar {facts.get('calendar')!r}; Samudra OM4 data use 'julian'."
        )
    for idx in range(1, len(day_offsets)):
        dt_days = float(day_offsets[idx]) - float(day_offsets[idx - 1])
        if not 4.0 <= dt_days <= 6.5:
            raise SystemExit(
                f"Staged window cadence out of 5-day OM4 range between states {idx - 1} and {idx} ({dt_days:.4f} d)."
            )


def device_check(requested: str, cuda_available: bool) -> str:
    """Refuse silent CPU fallback when a CUDA device was explicitly requested."""
    if requested.startswith("cuda"):
        if not cuda_available:
            raise SystemExit(
                f"{requested} requested for samudra-om4, but CUDA is unavailable in this job; refusing silent CPU fallback."
            )
        return requested
    if requested == "cpu":
        return "cpu"
    raise SystemExit(f"Unknown torch device {requested!r}")


def build_samudra_unet(
    wet: Any,
    *,
    ch_width: Sequence[int] = SAMUDRA_CH_WIDTH,
    n_out: int = SAMUDRA_N_OUT,
    dilation: Sequence[int] = SAMUDRA_DILATION,
    last_kernel_size: int = 3,
    pad: str = "circular",
) -> Any:
    """Construct the licensed `m2lines/Samudra@0c2a1bd8be6260fe02507fbb697ef50be8835fe6` ConvNeXt U-Net.

    Preserves the exact `layers.0 .. layers.17` hierarchy and 178 state_dict tensors
    expected by `samudra_thermo_dynamic_seed1.pt` under `load_state_dict(..., strict=True)`.
    """
    import numpy as np
    import torch
    import torch.nn as nn

    class CappedGELU(nn.Module):
        def __init__(self, cap_value: float = 10.0, **kwargs: Any) -> None:
            super().__init__()
            self.add_module("gelu", nn.GELU(**kwargs))
            self.register_buffer("cap", torch.tensor(cap_value, dtype=torch.float32))

        def forward(self, inputs: Any) -> Any:
            x = self.gelu(inputs)
            return torch.clamp(x, max=self.cap)

    class TransposedConvUpsample(nn.Module):
        def __init__(
            self,
            in_channels: int = 3,
            out_channels: int = 1,
            upsampling: int = 2,
            activation: Any = CappedGELU,
        ) -> None:
            super().__init__()
            upsampler: list[nn.Module] = [
                nn.ConvTranspose2d(
                    in_channels,
                    out_channels,
                    kernel_size=upsampling,
                    stride=upsampling,
                    padding=0,
                )
            ]
            if activation is not None:
                upsampler.append(activation())
            self.upsampler = nn.Sequential(*upsampler)

        def forward(self, x: Any) -> Any:
            return self.upsampler(x)

    class BilinearUpsample(nn.Module):
        def __init__(self, upsampling: int = 2, **_kwargs: Any) -> None:
            super().__init__()
            self.upsampler = nn.Upsample(scale_factor=upsampling, mode="bilinear")

        def forward(self, x: Any) -> Any:
            return self.upsampler(x)

    class AvgPool(nn.Module):
        def __init__(self, pooling: int = 2) -> None:
            super().__init__()
            self.avgpool = nn.AvgPool2d(pooling)

        def forward(self, x: Any) -> Any:
            return self.avgpool(x)

    class CoreBlock(nn.Module):
        def __init__(
            self,
            in_channels: int,
            out_channels: int,
            kernel_size: int,
            block_dilation: int,
            block_pad: str,
        ) -> None:
            super().__init__()
            if kernel_size % 2 == 0:
                raise ValueError("Cannot use even kernel sizes in Samudra CoreBlock.")
            self.N_in = in_channels
            self.N_pad = int((kernel_size + (kernel_size - 1) * (block_dilation - 1) - 1) / 2)
            self.pad = block_pad

    class ConvNeXtBlock(CoreBlock):
        def __init__(
            self,
            in_channels: int = 300,
            out_channels: int = 1,
            kernel_size: int = 3,
            block_dilation: int = 1,
            n_layers: int = 1,
            activation: Any = CappedGELU,
            block_pad: str = "circular",
            upscale_factor: int = 4,
            norm: str = "batch",
        ) -> None:
            super().__init__(in_channels, out_channels, kernel_size, block_dilation, block_pad)
            if n_layers != 1:
                raise ValueError("Samudra ConvNeXtBlock supports n_layers == 1 only.")
            if in_channels == out_channels:
                self.skip_module: Any = lambda x: x
            else:
                self.skip_module = nn.Conv2d(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=1,
                    padding="same",
                )

            latent = int(in_channels * upscale_factor)
            convblock: list[nn.Module] = [
                nn.Conv2d(
                    in_channels=in_channels,
                    out_channels=latent,
                    kernel_size=kernel_size,
                    dilation=block_dilation,
                )
            ]
            if norm == "batch":
                convblock.append(nn.BatchNorm2d(latent))
            elif norm == "instance":
                convblock.append(nn.InstanceNorm2d(latent))
            elif norm != "nonorm":
                raise NotImplementedError(f"Unsupported norm {norm!r}")
            if activation is not None:
                convblock.append(activation())

            convblock.append(
                nn.Conv2d(
                    in_channels=latent,
                    out_channels=latent,
                    kernel_size=kernel_size,
                    dilation=block_dilation,
                )
            )
            if norm == "batch":
                convblock.append(nn.BatchNorm2d(latent))
            elif norm == "instance":
                convblock.append(nn.InstanceNorm2d(latent))
            elif norm != "nonorm":
                raise NotImplementedError(f"Unsupported norm {norm!r}")
            if activation is not None:
                convblock.append(activation())

            convblock.append(
                nn.Conv2d(
                    in_channels=latent,
                    out_channels=out_channels,
                    kernel_size=1,
                    padding="same",
                )
            )
            self.convblock = nn.Sequential(*convblock)

        def forward(self, x: Any) -> Any:
            skip = self.skip_module(x)
            for layer in self.convblock:
                if isinstance(layer, nn.Conv2d) and layer.kernel_size[0] != 1:
                    x = nn.functional.pad(
                        x, (self.N_pad, self.N_pad, 0, 0), mode=self.pad
                    )
                    x = nn.functional.pad(
                        x, (0, 0, self.N_pad, self.N_pad), mode="constant"
                    )
                x = layer(x)
            return skip + x

    class UNet(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            ch = list(ch_width)
            dil = list(dilation)
            layers: list[nn.Module] = []
            i = 0
            b = ch[-1]
            for i, (a, b) in enumerate(zip(ch[:-1], ch[1:])):
                layers.append(
                    ConvNeXtBlock(
                        in_channels=a,
                        out_channels=b,
                        block_dilation=dil[i],
                        block_pad=pad,
                    )
                )
                layers.append(AvgPool())
            layers.append(
                ConvNeXtBlock(
                    in_channels=b,
                    out_channels=b,
                    block_dilation=dil[i],
                    block_pad=pad,
                )
            )
            layers.append(BilinearUpsample())

            ch.reverse()
            dil.reverse()
            for i, (a, b) in enumerate(zip(ch[:-2], ch[1:-1])):
                layers.append(
                    ConvNeXtBlock(
                        in_channels=a,
                        out_channels=b,
                        block_dilation=dil[i],
                        block_pad=pad,
                    )
                )
                layers.append(BilinearUpsample())
            layers.append(
                ConvNeXtBlock(
                    in_channels=b,
                    out_channels=b,
                    block_dilation=dil[i],
                    block_pad=pad,
                )
            )
            layers.append(nn.Conv2d(b, n_out, last_kernel_size))
            self.layers = nn.ModuleList(layers)
            self.num_steps = int(len(ch_width) - 1)
            self.N_pad = int((last_kernel_size - 1) // 2)
            self.pad = pad
            self.wet = wet

        def forward_once(self, fts: Any) -> Any:
            temp: list[Any] = [torch.zeros_like(fts) for _ in range(self.num_steps)]
            count = 0
            for layer in self.layers:
                if isinstance(layer, nn.Conv2d):
                    fts = nn.functional.pad(
                        fts, (self.N_pad, self.N_pad, 0, 0), mode=self.pad
                    )
                    fts = nn.functional.pad(
                        fts, (0, 0, self.N_pad, self.N_pad), mode="constant"
                    )
                fts = layer(fts)
                if count < self.num_steps:
                    if isinstance(layer, CoreBlock):
                        temp[count] = fts
                        count += 1
                elif count >= self.num_steps:
                    if isinstance(layer, (BilinearUpsample, TransposedConvUpsample)):
                        crop = np.array(fts.shape[2:])
                        shape = np.array(
                            temp[int(2 * self.num_steps - count - 1)].shape[2:]
                        )
                        pads = shape - crop
                        pad_spec = [
                            int(pads[1] // 2),
                            int(pads[1] - pads[1] // 2),
                            int(pads[0] // 2),
                            int(pads[0] - pads[0] // 2),
                        ]
                        fts = nn.functional.pad(fts, pad_spec)
                        fts = fts + temp[int(2 * self.num_steps - count - 1)]
                        count += 1
            return torch.mul(fts, self.wet)

        def forward(self, fts: Any) -> Any:
            return self.forward_once(fts)

    return UNet()


def _open_zarr_any(path: str, *, storage_options: dict[str, Any] | None = None, **kwargs: Any) -> Any:
    import xarray as xr

    if storage_options:
        return xr.open_zarr(path, consolidated=True, storage_options=storage_options, **kwargs)
    has_consolidated = os.path.isfile(os.path.join(path, ".zmetadata"))
    return xr.open_zarr(path, consolidated=has_consolidated, **kwargs)


def _open_samudra_source(source_root: str, store_name: str, **kwargs: Any) -> Any:
    root = source_root.rstrip("/")
    if root.startswith(("s3://", "http://", "https://")):
        storage = {"anon": True, "client_kwargs": {"endpoint_url": SAMUDRA_OSN_ENDPOINT}}
        remote_path = f"{root}/{store_name}"
        last_exc: Exception | None = None
        for attempt in range(SAMUDRA_REMOTE_TRIES):
            try:
                return _open_zarr_any(remote_path, storage_options=storage, **kwargs)
            except Exception as exc:  # noqa: BLE001
                last_exc = exc
                time.sleep(5 * (attempt + 1))
        raise SystemExit(
            f"Could not open remote OSN store {remote_path} after {SAMUDRA_REMOTE_TRIES} attempts: {last_exc}"
        )

    for cand in (os.path.join(root, store_name), os.path.join(root, f"{store_name}.zarr")):
        if os.path.isdir(cand):
            return _open_zarr_any(cand, **kwargs)
    raise SystemExit(f"Local Samudra source store not found under {root!r} ({store_name}).")


def stage_samudra(out_dir: str, job: Mapping[str, Any]) -> dict[str, Any]:
    """Stage the exact `[start - 5d, start, target_1 .. target_steps]` OM4 window into `<out_dir>/window`."""
    import numpy as np

    dest_dir = os.path.join(out_dir, "window")
    if os.path.exists(dest_dir):
        raise SystemExit(f"Staged window directory already exists: {dest_dir}")
    steps = int(job["steps"])
    if steps < 2 or steps % 2 != 0:
        raise SystemExit("Samudra --steps must be an even integer >= 2 (two 5-day states per call).")

    source_root = (
        os.environ.get("AIMIP_SAMUDRA_SOURCE")
        or str(job.get("osn_source") or SAMUDRA_OSN_SOURCE)
    ).strip()
    is_remote = source_root.startswith(("s3://", "http://", "https://"))
    t0 = time.time()
    ds = _open_samudra_source(source_root, "OM4", chunks={"time": 1} if is_remote else None)
    times = ds["time"].values
    keys = np.array([str(t)[:16].replace(" ", "T") for t in times])
    start_key = str(job["start"])[:16].replace(" ", "T")
    hits = np.where(keys == start_key)[0]
    if hits.size != 1:
        sample_near = [str(k) + ":00" for k in keys[:4]]
        raise SystemExit(
            f"--start {job['start']} is not on Samudra's 5-day OM4 time axis "
            f"(first={keys[0]}:00, last={keys[-1]}:00, e.g. {', '.join(sample_near)} or 2014-10-10T12:00:00)."
        )
    i0 = int(hits[0])
    if i0 < SAMUDRA_INPUT_STEPS - 1 or i0 + steps >= int(times.size):
        raise SystemExit(
            f"Requested Samudra window [{i0 - SAMUDRA_INPUT_STEPS + 1}, {i0 + steps}] "
            f"falls outside OM4 store bounds [0 .. {int(times.size) - 1}]."
        )

    sl = slice(i0 - SAMUDRA_INPUT_STEPS + 1, i0 + steps + 1)
    window = ds.isel(time=sl).reset_coords()
    makedirs_no_group_other_write(dest_dir)
    zarr_dest = os.path.join(dest_dir, "OM4.zarr")
    var_names = sorted(str(v) for v in window.data_vars)
    tries = SAMUDRA_REMOTE_TRIES if is_remote else 1
    for k, name in enumerate(var_names):
        last_exc: Exception | None = None
        for attempt in range(tries):
            try:
                piece = window[[name]].load()
                piece.to_zarr(
                    zarr_dest,
                    mode="w" if k == 0 else "a",
                    zarr_format=2,
                    consolidated=True,
                )
                break
            except Exception as exc:  # noqa: BLE001
                last_exc = exc
                if is_remote:
                    time.sleep(5 * (attempt + 1))
        else:
            raise SystemExit(
                f"Could not stage OM4 variable {name!r} after {tries} attempts: {last_exc}"
            )
        if is_remote and ((k + 1) % 10 == 0 or k + 1 == len(var_names)):
            print(
                f"[aimip.samudra] staged {k + 1}/{len(var_names)} OM4 variables ({name})",
                file=sys.stderr,
                flush=True,
            )

    total_bytes = sum(
        os.path.getsize(os.path.join(walk_root, fn))
        for walk_root, _, files in os.walk(dest_dir)
        for fn in files
    )
    facts: dict[str, Any] = {
        "source": source_root,
        "start": str(job["start"]),
        "steps": steps,
        "input_steps": SAMUDRA_INPUT_STEPS,
        "time_indices": [int(sl.start), int(sl.stop - 1)],
        "times": [str(t) for t in window["time"].values],
        "variables": var_names,
        "bytes": int(total_bytes),
        "seconds": round(time.time() - t0, 2),
        "calendar": "julian",
    }
    facts_path = os.path.join(dest_dir, "window.json")
    with open(facts_path, "w", encoding="utf-8") as fh:
        json.dump(facts, fh, indent=2)
        fh.write("\n")

    for walk_root, dirs, files in os.walk(dest_dir):
        strip_group_other_write(walk_root)
        for dn in dirs:
            strip_group_other_write(os.path.join(walk_root, dn))
        for fn in files:
            strip_group_other_write(os.path.join(walk_root, fn))

    print(
        f"[aimip.emu_job] STAGED samudra-om4 steps={steps} bytes={total_bytes} -> {dest_dir}",
        file=sys.stderr,
        flush=True,
    )
    return facts


def _extract_1d_lat_lon(window: Any, y_len: int, x_len: int) -> tuple[Any, Any]:
    import numpy as np

    lat_1d = None
    if "lat" in window:
        raw_lat = np.asarray(window["lat"].values, dtype=np.float64)
        if raw_lat.ndim == 1 and raw_lat.size == y_len and np.isfinite(raw_lat).all():
            lat_1d = raw_lat
        elif raw_lat.ndim == 2 and raw_lat.shape == (y_len, x_len):
            row_mean = np.nanmean(raw_lat, axis=1)
            if np.isfinite(row_mean).all():
                lat_1d = row_mean
    if lat_1d is None and "y" in window:
        cand_y = np.asarray(window["y"].values, dtype=np.float64)
        if cand_y.ndim == 1 and cand_y.size == y_len and np.isfinite(cand_y).all():
            lat_1d = cand_y
    if lat_1d is None:
        lat_1d = np.linspace(-89.5, 89.5, y_len, dtype=np.float64)

    lon_1d = None
    if "lon" in window:
        raw_lon = np.asarray(window["lon"].values, dtype=np.float64)
        if raw_lon.ndim == 1 and raw_lon.size == x_len and np.isfinite(raw_lon).all():
            lon_1d = raw_lon
        elif raw_lon.ndim == 2 and raw_lon.shape == (y_len, x_len):
            col_mean = np.nanmean(raw_lon, axis=0)
            if np.isfinite(col_mean).all():
                lon_1d = col_mean
    if lon_1d is None and "x" in window:
        cand_x = np.asarray(window["x"].values, dtype=np.float64)
        if cand_x.ndim == 1 and cand_x.size == x_len and np.isfinite(cand_x).all():
            lon_1d = cand_x
    if lon_1d is None:
        lon_1d = np.linspace(0.5, 359.5, x_len, dtype=np.float64)

    return lat_1d, lon_1d


def _load_samudra_norm_stores(job: Mapping[str, Any], window_dir: str, tmp_dir: str) -> tuple[Any, Any]:
    means_dir_pre = os.path.join(window_dir, "OM4_means.zarr")
    stds_dir_pre = os.path.join(window_dir, "OM4_stds.zarr")
    if job.get("means_zip") and job.get("stds_zip"):
        means_root = extract_verified_zip_store(
            str(job["means_zip"]),
            str(job["means_zip_sha256"]),
            os.path.join(tmp_dir, "OM4_means"),
            label="Samudra OM4_means.zip",
        )
        stds_root = extract_verified_zip_store(
            str(job["stds_zip"]),
            str(job["stds_zip_sha256"]),
            os.path.join(tmp_dir, "OM4_stds"),
            label="Samudra OM4_stds.zip",
        )
        return _open_zarr_any(means_root).load(), _open_zarr_any(stds_root).load()
    if os.path.isdir(means_dir_pre) and os.path.isdir(stds_dir_pre):
        return _open_zarr_any(means_dir_pre).load(), _open_zarr_any(stds_dir_pre).load()
    raise SystemExit("Samudra normalization stores (OM4_means / OM4_stds) are missing from job configuration.")


def run_samudra(out_dir: str, job: Mapping[str, Any]) -> dict[str, Any]:
    """Execute a verified Samudra rollout and write `<out_dir>/ocean/monthly_mean_predictions.nc` in `degC`."""
    import numpy as np
    import torch

    from .emu_job import MonthlyWriter, _write_levels

    steps = int(job["steps"])
    if steps < 2 or steps % 2 != 0:
        raise SystemExit("Samudra --steps must be an even integer >= 2.")

    window_dir = str(job.get("window_dir") or os.path.join(out_dir, "window"))
    with open(os.path.join(window_dir, "window.json"), encoding="utf-8") as fh:
        facts = json.load(fh)
    if int(facts.get("input_steps", 0)) != SAMUDRA_INPUT_STEPS or int(facts.get("steps", 0)) < steps:
        raise SystemExit(
            f"Staged window holds {facts.get('input_steps')} input states and {facts.get('steps')} target states; "
            f"run requires {SAMUDRA_INPUT_STEPS} and {steps}."
        )

    window = _open_zarr_any(os.path.join(window_dir, "OM4.zarr"))
    times = window["time"].values
    if not type(times[0]).__name__.endswith("Julian"):
        raise SystemExit(
            f"Staged window time coordinate decoded as {type(times[0]).__name__!r}, expected cftime DatetimeJulian."
        )
    day_offsets = [_time_delta_days(t, times[0]) for t in times]
    check_samudra_window(facts, [str(t) for t in times], day_offsets, str(job["start"]), steps)

    with contextlib.ExitStack() as norm_stack:
        tmp_dir = (
            os.environ.get("AIMIP_EMU_TMP")
            or norm_stack.enter_context(tempfile.TemporaryDirectory(prefix="aimip-samudra-"))
        )
        means, stds = _load_samudra_norm_stores(job, window_dir, tmp_dir)

    requested_dev = str(job.get("device") or ("cuda:0" if torch.cuda.is_available() else "cpu"))
    dev = torch.device(device_check(requested_dev, torch.cuda.is_available()))

    wm = window["wetmask"].load().values.astype(np.float32)  # (lev, y, x)
    y_len, x_len = int(wm.shape[1]), int(wm.shape[2])
    wet_np = np.concatenate(
        [np.stack([wm[level_of(v)] for v in SAMUDRA_STATE_VARS])] * SAMUDRA_INPUT_STEPS
    )
    wet = torch.from_numpy(wet_np).to(dev)
    model = build_samudra_unet(wet).to(dev)

    ck = torch_load_verified(
        str(job["checkpoint"]),
        str(job["checkpoint_sha256"]),
        map_location="cpu",
        weights_only=True,
        label="Samudra checkpoint",
    )
    raw_sd = ck["model"] if isinstance(ck, dict) and "model" in ck else ck
    state_dict = {str(k).removeprefix("module."): v for k, v in raw_sd.items()}
    model.load_state_dict(state_dict, strict=True)
    model.eval()

    state_list = list(SAMUDRA_STATE_VARS)
    bound_list = list(SAMUDRA_BOUNDARY_VARS)

    def _normalized_block(ds_slice: Any, var_names: list[str]) -> Any:
        normed = ((ds_slice[var_names] - means[var_names]) / stds[var_names]).fillna(0.0)
        arr = normed.to_array().transpose("time", "variable", "y", "x").values.astype(np.float32)
        return arr.reshape(-1, *arr.shape[2:])

    state_mean = np.stack(
        [np.broadcast_to(means[v].values, (y_len, x_len)) for v in state_list]
    ).astype(np.float32)
    state_std = np.stack(
        [np.broadcast_to(stds[v].values, (y_len, x_len)) for v in state_list]
    ).astype(np.float32)

    states0 = _normalized_block(window.isel(time=slice(0, SAMUDRA_INPUT_STEPS)), state_list)
    n_calls = steps // SAMUDRA_INPUT_STEPS
    bounds = [
        _normalized_block(window.isel(time=[k * SAMUDRA_INPUT_STEPS + SAMUDRA_HIST]), bound_list)
        for k in range(n_calls)
    ]

    def _step_fn(states_in: Any, bound_in: Any) -> Any:
        with torch.inference_mode():
            x_in = torch.from_numpy(np.concatenate([states_in, bound_in]))[None].to(dev)
            return model(x_in)[0].detach().cpu().numpy()

    def _sanitize_fn(raw_out: Any) -> tuple[Any, int, int]:
        finite = np.isfinite(raw_out)
        n_nonfinite = int((~finite).sum())
        n_clipped = int((np.abs(raw_out[finite]) > 1e5).sum())
        clean = np.clip(np.nan_to_num(raw_out), -1e5, 1e5).reshape(
            SAMUDRA_INPUT_STEPS, len(state_list), y_len, x_len
        )
        return clean, n_nonfinite, n_clipped

    t_roll0 = time.time()
    stored, call_counts = rollout(_step_fn, _sanitize_fn, states0, bounds)
    if dev.type == "cuda":
        torch.cuda.synchronize(dev)
    roll_seconds = time.time() - t_roll0

    out_times = times[SAMUDRA_INPUT_STEPS : SAMUDRA_INPUT_STEPS + steps]
    bad_calls = [idx for idx, c in enumerate(call_counts) if c["nonfinite"] > 0]
    if bad_calls:
        first_bad_ts = out_times[min(bad_calls[0] * SAMUDRA_INPUT_STEPS, len(out_times) - 1)]
        by, bm, _ = julian_date_parts(first_bad_ts)
        return {
            "steps": steps,
            "months_written": {"ocean": 0},
            "rollout_seconds": round(roll_seconds, 2),
            "nonfinite_per_call": [c["nonfinite"] for c in call_counts],
            "clipped_per_call": [c["clipped"] for c in call_counts],
            "unstable_from": f"{by:04d}-{bm:02d}",
            "unstable_member": 0,
            "unstable_reason": f"non-finite values in Samudra rollout ({[c['nonfinite'] for c in call_counts]})",
        }

    pred = np.concatenate(stored, axis=0)  # (steps, 77, y, x)
    pred = pred * state_std[None] + state_mean[None]
    for j, vname in enumerate(state_list):
        pred[:, j, wm[level_of(vname)] == 0] = np.nan

    idx_thetao = [state_list.index(f"thetao_lev_{lv}") for lv in SAMUDRA_LEVELS]
    idx_so = [state_list.index(f"so_lev_{lv}") for lv in SAMUDRA_LEVELS]
    idx_uo0 = state_list.index("uo_lev_2_5")
    idx_vo0 = state_list.index("vo_lev_2_5")
    idx_zos = state_list.index("zos")

    fields_5d = {
        "thetao_0": pred[:, idx_thetao[0]],
        "so_0": pred[:, idx_so[0]],
        "zos": pred[:, idx_zos],
        "uo_0": pred[:, idx_uo0],
        "vo_0": pred[:, idx_vo0],
        "thetao_3d": pred[:, idx_thetao],
        "so_3d": pred[:, idx_so],
    }
    monthly_rows = aggregate_samudra_5day_to_monthly(
        out_times,
        fields_5d,
        prev_time=times[SAMUDRA_HIST],
        complete_months_only=True,
    )

    lat_1d, lon_1d = _extract_1d_lat_lon(window, y_len, x_len)
    in_training = str(job["start"])[:19] < SAMUDRA_INDEPENDENT_TEST_START
    global_attrs = {
        "source": (
            f"samudra-om4 (Samudra thermo-dynamic seed1, m2lines/Samudra@{SAMUDRA_LICENSED_COMMIT[:8]})"
        ),
        "aimip_family": "samudra",
        "start": str(job["start"]),
        "target_simulation": (
            "GFDL OM4 JRA55-do OMIP-2 replay (1975-2022); compare with OM4, not directly with observations"
        ),
        "monthly_aggregation": (
            "overlap-day weighted mean of 5-day OM4 states across complete Julian calendar months "
            "(counts = covered days in month)"
        ),
        "calendar_note": (
            "OM4 julian calendar; (year, month, day) labels copied directly without day-number conversion"
        ),
        "leap_year_note": (
            "OSN OM4.zarr stores 73 timestamps per year (Jan 3 .. Dec 28) with no time bounds or averaging attribute "
            "('5-day mean' is the Samudra paper's description); the 12 leap years in 1976..2020 each end with a 6-day "
            "Dec 28 -> Jan 3 step, so Dec 31 of a leap year is covered by no 5-day state and receives weight 1/31 from "
            "the Jan 3 state"
        ),
        "period_note": (
            "start precedes 2014-09-30 independent test start (inside training/validation period)"
            if in_training
            else "independent post-training test period (>= 2014-09-30)"
        ),
    }

    surf_vars = ["thetao_0", "so_0", "zos", "uo_0", "vo_0"]
    cell_note = "time: mean (5-day state overlap-day weighted over complete Julian calendar months)"
    var_meta = {
        "thetao_0": {
            "units": "degC",
            "long_name": "Ocean potential temperature at 2.5 m (not sea surface skin temperature)",
            "depth_m": 2.5,
            "cell_methods": cell_note,
        },
        "so_0": {
            "units": "psu",
            "long_name": "Ocean practical salinity at 2.5 m",
            "depth_m": 2.5,
            "cell_methods": cell_note,
        },
        "zos": {
            "units": "m",
            "long_name": "Sea surface height above geoid",
            "cell_methods": cell_note,
        },
        "uo_0": {
            "units": "m s-1",
            "long_name": "Zonal ocean velocity at 2.5 m",
            "depth_m": 2.5,
            "cell_methods": cell_note,
        },
        "vo_0": {
            "units": "m s-1",
            "long_name": "Meridional ocean velocity at 2.5 m",
            "depth_m": 2.5,
            "cell_methods": cell_note,
        },
    }

    global_mean_thetao0: float | None = None
    if monthly_rows:
        ocean_nc = os.path.join(out_dir, "ocean", "monthly_mean_predictions.nc")
        writer = MonthlyWriter(
            ocean_nc,
            surf_vars,
            lat_1d,
            lon_1d,
            1,
            attrs=global_attrs,
            extra=var_meta,
        )
        for row in monthly_rows:
            writer.write(
                0,
                row["month"],
                {k: row["fields"][k] for k in surf_vars},
                int(row["covered_days"]),
            )
        writer.close()
        strip_group_other_write(ocean_nc)

        months_dt = [np.datetime64(r["month"], "M") for r in monthly_rows]
        thetao_stack = np.stack([r["fields"]["thetao_3d"] for r in monthly_rows], axis=0)
        so_stack = np.stack([r["fields"]["so_3d"] for r in monthly_rows], axis=0)
        levels_arr = np.stack([thetao_stack, so_stack], axis=1)  # (time, 2, 19, lat, lon)
        thetao_nc = os.path.join(out_dir, "ocean", "monthly_thetao.nc")
        _write_levels(
            thetao_nc,
            levels_arr,
            ["thetao", "so"],
            np.asarray(SAMUDRA_DEPTH_METERS, dtype=np.float64),
            lat_1d,
            lon_1d,
            months_dt,
            "depth",
            "m",
            global_attrs,
        )
        strip_group_other_write(thetao_nc)

        first_t0 = monthly_rows[0]["fields"]["thetao_0"]
        valid_mask = np.isfinite(first_t0)
        if np.any(valid_mask):
            cos_w = np.broadcast_to(np.maximum(0.0, np.cos(np.deg2rad(lat_1d)))[:, None], (y_len, x_len))
            w_sum = float(np.sum(cos_w[valid_mask]))
            if w_sum > 0.0:
                global_mean_thetao0 = round(
                    float(np.sum(first_t0[valid_mask] * cos_w[valid_mask]) / w_sum), 4
                )

    return {
        "steps": steps,
        "months_written": {"ocean": len(monthly_rows)},
        "first_month": monthly_rows[0]["month"] if monthly_rows else None,
        "last_month": monthly_rows[-1]["month"] if monthly_rows else None,
        "rollout_seconds": round(roll_seconds, 2),
        "device": str(dev),
        "clipped_per_call": [c["clipped"] for c in call_counts],
        "nonfinite_per_call": [c["nonfinite"] for c in call_counts],
        "global_mean_thetao_0_degC": global_mean_thetao0,
        "in_training_period": in_training,
    }
