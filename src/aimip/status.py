"""Run status inspection (`sacct.txt`, `--sacct-json`, `sacct`, `submitted.txt`, manifest) and monthly summaries.

Dimension names are NEVER hardcoded: `summarize_run` detects horizontal dimensions
dynamically from `dims[-2:]` (`(y_dim, x_dim)`), matching
`fme.core.dataset.utils.get_horizontal_coordinates` (`fme/core/dataset/utils.py:219-229`).

fme 2026.5.1 references:
- Monthly output writer & filename (`monthly_mean_predictions.nc`):
  `fme/ace/inference/data_writer/monthly.py:61-69, 121`
- Monthly dataset layout (`dims=("sample", "time", "lat", "lon")`, `time` in months,
  `valid_time` in `days since 1970-01-01 00:00:00` on the 15th of each month,
  `counts` of shape `("sample", "time")`, values stored as running monthly means):
  `fme/ace/inference/data_writer/monthly.py:197-227, 364-385`,
  `fme/ace/inference/data_writer/test_monthly.py:74-78`
- Temperature variable names (`TMP2m`, `surface_temperature`):
  `fme/ace/inference/default_metadata.py:24, 146`
- Spherical area weights (`cos(deg2rad(lat))`):
  `fme/core/metrics.py:14-32`.
"""

from __future__ import annotations

import collections
from collections.abc import Sequence
import json
import os
import shutil
import subprocess
import threading
import time
from typing import Any

from .errors import ConfigError, ValidationError
from .validators import validate_run_path, validate_safe_path, validate_slurm_job_id

MONTHLY_PREDICTIONS_FILENAME = "monthly_mean_predictions.nc"
# Includes EAMv3 (`TREFHT`, `TS`) and SamudrACE-E3SMv3 (`Tat2m`) alongside ERA5/SHiELD/HiRO (`TMP2m`, `surface_temperature`).
SUMMARY_TEMP_VARS = ("TMP2m", "Tat2m", "surface_temperature", "TREFHT", "TS", "t2m", "tas")   # + DLESyM, ArchesClimate
SUMMARY_OCEAN_TEMP_VARS = ("sst", "surface_temperature", "thetao_0", "temperatureCoarsened_0")

MAX_SUMMARY_MONTHS = 2400
MAX_SUMMARY_MEMBERS = 128
MAX_SUMMARY_VAR_BYTES = 512 * 1024 * 1024
SUMMARY_TIME_CHUNK_MONTHS = 24
MAX_SUMMARY_CACHE_ENTRIES = 32
_SUMMARY_CACHE_LOCK = threading.Lock()
_SUMMARY_CACHE: collections.OrderedDict[tuple[Any, ...], dict[str, Any]] = collections.OrderedDict()


def _parse_submitted_txt(out_dir: str) -> dict[str, str] | None:
    """Parse `<out>/submitted.txt` (`job=<id> submitted=<ts> user=<u>`)."""
    sub_file = os.path.join(out_dir, "submitted.txt")
    if not os.path.isfile(sub_file):
        return None
    try:
        with open(sub_file, encoding="utf-8") as fh:
            line = fh.readline().strip()
    except OSError:
        return None
    if not line:
        return None

    parsed: dict[str, str] = {}
    for token in line.split():
        if "=" in token:
            k, v = token.split("=", 1)
            parsed[k] = v
    return parsed or None


def _parse_local_pid(out_dir: str) -> int | None:
    """Parse `<out>/local.pid` written when `scheduler == 'none'`."""
    pid_file = os.path.join(out_dir, "local.pid")
    if not os.path.isfile(pid_file):
        return None
    try:
        with open(pid_file, encoding="utf-8") as fh:
            raw = fh.read().strip()
        if raw.isdigit():
            return int(raw)
    except OSError:
        return None
    return None


def _discover_job_id(
    out_dir: str,
    manifest: dict[str, Any],
    submitted_info: dict[str, str] | None,
) -> str | None:
    """Resolve SLURM job ID from manifest, `submitted.txt`, or `slurm.jobid`."""
    raw = manifest.get("slurm_job_id")
    if isinstance(raw, str) and raw.strip():
        try:
            return validate_slurm_job_id(raw.strip().split(";")[0])
        except ValidationError:
            pass
    elif isinstance(raw, int):
        return validate_slurm_job_id(str(raw))

    if submitted_info and "job" in submitted_info:
        cand = submitted_info["job"].split(";")[0].strip()
        try:
            return validate_slurm_job_id(cand)
        except ValidationError:
            pass

    jobid_file = os.path.join(out_dir, "slurm.jobid")
    if os.path.isfile(jobid_file):
        try:
            with open(jobid_file, encoding="utf-8") as fh:
                token = fh.read().strip().split(";")[0]
            if token:
                return validate_slurm_job_id(token)
        except (OSError, ValidationError):
            return None
    return None


def _parse_sacct_pipe_lines(
    lines: Sequence[str],
    target_job_id: str | None = None,
) -> dict[str, str] | None:
    """Parse pipe-delimited `JobID|State|Elapsed|ExitCode` lines (`sacct -P` / `sacct.txt`)."""
    fallback: dict[str, str] | None = None
    non_step_fallback: dict[str, str] | None = None
    for raw_line in lines:
        line = raw_line.strip()
        if not line:
            continue
        parts = line.split("|")
        if len(parts) < 4:
            continue
        jid = parts[0].strip()
        if jid.lower() in ("jobid", "jobidraw"):
            continue
        rec = {
            "job_id": jid,
            "state": parts[1].strip(),
            "elapsed": parts[2].strip(),
            "exit_code": parts[3].strip(),
        }
        if target_job_id and jid == target_job_id:
            return rec
        if "." not in jid and non_step_fallback is None:
            non_step_fallback = rec
        if fallback is None:
            fallback = rec
    return non_step_fallback or fallback


def _format_elapsed_value(raw_elapsed: Any) -> str:
    """Format `sacct --json` integer seconds or string elapsed field as `HH:MM:SS`."""
    if isinstance(raw_elapsed, (int, float)) and not isinstance(raw_elapsed, bool):
        total = max(0, int(raw_elapsed))
        hh, rem = divmod(total, 3600)
        mm, ss = divmod(rem, 60)
        return f"{hh:02d}:{mm:02d}:{ss:02d}"
    return str(raw_elapsed or "00:00:00")


def _normalize_sacct_json_job(item: dict[str, Any]) -> dict[str, str] | None:
    """Normalize one job entry from SLURM `sacct --json` (`jobs[]`) or a flat dict."""
    raw_jid = item.get("job_id") if "job_id" in item else item.get("JobID")
    if raw_jid is None:
        return None
    job_id_str = str(raw_jid).strip()
    if not job_id_str:
        return None

    # State: either string `"RUNNING"` or SLURM `--json` `{"current": ["RUNNING"] | "RUNNING"}`.
    raw_state = item.get("state") if "state" in item else item.get("State")
    state_str = ""
    if isinstance(raw_state, dict):
        cur = raw_state.get("current")
        if isinstance(cur, list) and cur:
            state_str = str(cur[0]).strip()
        elif cur is not None:
            state_str = str(cur).strip()
    elif raw_state is not None:
        state_str = str(raw_state).strip()
    if not state_str:
        return None

    # Elapsed: either `"elapsed"` string or SLURM `--json` `time.elapsed` seconds.
    if "elapsed" in item or "Elapsed" in item:
        elapsed_str = _format_elapsed_value(
            item.get("elapsed") if "elapsed" in item else item.get("Elapsed")
        )
    elif isinstance(item.get("time"), dict):
        elapsed_str = _format_elapsed_value(item["time"].get("elapsed", 0))
    else:
        elapsed_str = "00:00:00"

    # ExitCode: either `"0:0"` string or SLURM `--json` `exit_code` / `derived_exit_code`.
    raw_exit = (
        item.get("exit_code")
        if "exit_code" in item
        else (
            item.get("derived_exit_code")
            if "derived_exit_code" in item
            else item.get("ExitCode")
        )
    )
    if isinstance(raw_exit, dict):
        rc_field = raw_exit.get("return_code", 0)
        if isinstance(rc_field, dict):
            rc_num = rc_field.get("number", 0)
        else:
            rc_num = rc_field
        sig_field = raw_exit.get("signal", {})
        sig_num = 0
        if isinstance(sig_field, dict):
            sig_id = sig_field.get("id", 0)
            sig_num = sig_id.get("number", 0) if isinstance(sig_id, dict) else sig_id
        exit_str = f"{int(rc_num or 0)}:{int(sig_num or 0)}"
    elif raw_exit is not None:
        exit_str = str(raw_exit).strip()
    else:
        exit_str = "0:0"

    return {
        "job_id": job_id_str,
        "state": state_str,
        "elapsed": elapsed_str,
        "exit_code": exit_str,
    }


def _parse_sacct_json_file(
    sacct_json_path: str,
    target_job_id: str | None = None,
) -> dict[str, str] | None:
    """Read and parse `--sacct-json FILE` (SLURM `sacct --json` schema or flat JSON)."""
    validated = validate_safe_path(sacct_json_path, label="--sacct-json file")
    if not os.path.isfile(validated):
        raise ConfigError(f"sacct JSON file not found: {validated}")
    try:
        with open(validated, encoding="utf-8") as fh:
            payload = json.load(fh)
    except (OSError, json.JSONDecodeError) as exc:
        raise ConfigError(f"Failed to parse sacct JSON file {validated}: {exc}") from exc

    candidates: list[dict[str, Any]]
    if isinstance(payload, dict) and isinstance(payload.get("jobs"), list):
        candidates = [x for x in payload["jobs"] if isinstance(x, dict)]
    elif isinstance(payload, list):
        candidates = [x for x in payload if isinstance(x, dict)]
    elif isinstance(payload, dict):
        candidates = [payload]
    else:
        raise ConfigError(
            f"Unsupported JSON structure in {validated}: expected object with 'jobs' or job record."
        )

    fallback: dict[str, str] | None = None
    for entry in candidates:
        norm = _normalize_sacct_json_job(entry)
        if norm is None:
            continue
        if target_job_id and norm["job_id"] == target_job_id:
            return norm
        if fallback is None:
            fallback = norm
    return fallback


def _read_sacct_txt(out_dir: str, target_job_id: str | None) -> dict[str, str] | None:
    """Read `<out>/sacct.txt` written by host wrapper (`sacct -n -X -P -j <id> -o JobID,State,Elapsed,ExitCode`)."""
    txt_path = os.path.join(out_dir, "sacct.txt")
    if not os.path.isfile(txt_path):
        return None
    try:
        with open(txt_path, encoding="utf-8") as fh:
            lines = fh.read().splitlines()
    except OSError:
        return None
    return _parse_sacct_pipe_lines(lines, target_job_id=target_job_id)


def _query_sacct_with_raw(job_id: str) -> tuple[dict[str, str] | None, str | None]:
    """Query `sacct` without shell invocation if `sacct` is available on PATH, returning `(parsed, raw_stdout)`."""
    validated_id = validate_slurm_job_id(job_id)
    sacct_bin = shutil.which("sacct")
    if not sacct_bin:
        return None, None

    from .submit import sanitized_subprocess_env

    try:
        proc = subprocess.run(
            [
                sacct_bin,
                "-j",
                validated_id,
                "--format=JobIDRaw,State,Elapsed,ExitCode",
                "--noheader",
                "--parsable2",
            ],
            capture_output=True,
            text=True,
            env=sanitized_subprocess_env(),
            check=False,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError):
        return None, None

    raw_out = proc.stdout.strip()
    if proc.returncode != 0 or not raw_out:
        return None, None

    parsed = _parse_sacct_pipe_lines(raw_out.splitlines(), target_job_id=validated_id)
    return parsed, raw_out


def _query_sacct(job_id: str) -> dict[str, str] | None:
    """Query `sacct` without shell invocation if `sacct` is available on PATH."""
    parsed, _ = _query_sacct_with_raw(job_id)
    return parsed


_SACCT_CACHE_LOCK = threading.Lock()
_LAST_SACCT_QUERY: dict[tuple[str, str], tuple[float, dict[str, str] | None]] = {}


def refresh_sacct_batch(
    run_jobs: Sequence[tuple[str, str]],
    *,
    min_interval_seconds: float = 15.0,
) -> None:
    """Refresh `_LAST_SACCT_QUERY` and `<run>/sacct.txt` for multiple active Slurm jobs in one `sacct` call."""
    if not run_jobs:
        return
    sacct_bin = shutil.which("sacct")
    if not sacct_bin:
        return

    now_mono = time.monotonic()
    pending: list[tuple[str, str]] = []
    with _SACCT_CACHE_LOCK:
        for out_dir, raw_jid in run_jobs:
            try:
                vid = validate_slurm_job_id(str(raw_jid))
            except ValidationError:
                continue
            cache_key = (out_dir, vid)
            if min_interval_seconds > 0.0:
                cached_entry = _LAST_SACCT_QUERY.get(cache_key)
                if cached_entry is not None and (now_mono - cached_entry[0]) < min_interval_seconds:
                    continue
            pending.append((out_dir, vid))

    if not pending:
        return

    from .submit import is_trusted_run_dir, sanitized_subprocess_env, write_in_trusted_dir

    unique_ids = list(dict.fromkeys(vid for _, vid in pending))
    for chunk_start in range(0, len(unique_ids), 100):
        chunk_ids = unique_ids[chunk_start : chunk_start + 100]
        chunk_set = set(chunk_ids)
        try:
            proc = subprocess.run(
                [
                    sacct_bin,
                    "-j",
                    ",".join(chunk_ids),
                    "--format=JobIDRaw,State,Elapsed,ExitCode",
                    "--noheader",
                    "--parsable2",
                ],
                capture_output=True,
                text=True,
                env=sanitized_subprocess_env(),
                check=False,
                timeout=10,
            )
        except (OSError, subprocess.SubprocessError):
            continue

        raw_out = proc.stdout.strip()
        all_lines = raw_out.splitlines() if (proc.returncode == 0 and raw_out) else []
        by_base_id: dict[str, list[str]] = {}
        for line in all_lines:
            stripped = line.strip()
            if not stripped or "|" not in stripped:
                continue
            base_jid = stripped.split("|", 1)[0].strip().split(".", 1)[0]
            if base_jid:
                by_base_id.setdefault(base_jid, []).append(stripped)

        stamp = time.monotonic()
        for out_dir, vid in pending:
            if vid not in chunk_set:
                continue
            job_lines = by_base_id.get(vid)
            if not job_lines and len(chunk_ids) == 1 and all_lines:
                job_lines = all_lines
            parsed = (
                _parse_sacct_pipe_lines(job_lines, target_job_id=vid)
                if job_lines
                else None
            )
            with _SACCT_CACHE_LOCK:
                _LAST_SACCT_QUERY[(out_dir, vid)] = (
                    stamp,
                    dict(parsed) if parsed is not None else None,
                )
            if parsed is not None and job_lines and is_trusted_run_dir(out_dir):
                try:
                    write_in_trusted_dir(out_dir, "sacct.txt", "\n".join(job_lines))
                except Exception:
                    pass


def _resolve_sacct_info(
    out_dir: str,
    job_id: str | None,
    sacct_json_path: str | None = None,
    *,
    min_sacct_interval_seconds: float = 0.0,
) -> tuple[dict[str, str] | None, bool, str | None]:
    """Resolve `(sacct_info, sacct_available, sacct_source)` from `--sacct-json`, live `sacct`, or `sacct.txt`."""
    if sacct_json_path is not None:
        info = _parse_sacct_json_file(sacct_json_path, target_job_id=job_id)
        return info, True, "sacct-json"

    sacct_bin_available = shutil.which("sacct") is not None
    if job_id and sacct_bin_available:
        cache_key = (out_dir, str(job_id))
        now_mono = time.monotonic()
        if min_sacct_interval_seconds > 0.0:
            with _SACCT_CACHE_LOCK:
                cached_entry = _LAST_SACCT_QUERY.get(cache_key)
            if cached_entry is not None and (now_mono - cached_entry[0]) < min_sacct_interval_seconds:
                if cached_entry[1] is not None:
                    return dict(cached_entry[1]), True, "sacct"
                txt_cached = _read_sacct_txt(out_dir, target_job_id=job_id)
                if txt_cached is not None:
                    return txt_cached, True, "sacct.txt"
                return None, sacct_bin_available, None

        live_info, raw_out = _query_sacct_with_raw(job_id)
        with _SACCT_CACHE_LOCK:
            _LAST_SACCT_QUERY[cache_key] = (
                time.monotonic(),
                dict(live_info) if live_info is not None else None,
            )
        if live_info is not None:
            from .submit import is_trusted_run_dir, write_in_trusted_dir

            if raw_out and is_trusted_run_dir(out_dir):
                try:
                    write_in_trusted_dir(out_dir, "sacct.txt", raw_out)
                except Exception:
                    pass
            return live_info, True, "sacct"

    txt_info = _read_sacct_txt(out_dir, target_job_id=job_id)
    if txt_info is not None:
        return txt_info, True, "sacct.txt"

    return None, sacct_bin_available, None


def _get_spin_status(
    validated_out: str,
    spin_manifest_path: str,
    *,
    sacct_json_path: str | None = None,
    min_sacct_interval_seconds: float = 0.0,
) -> dict[str, Any]:
    """Summarize status of a `<spin-dir>` created by `aimip ic spin`."""
    try:
        with open(spin_manifest_path, encoding="utf-8") as fh:
            manifest = json.load(fh)
    except (OSError, json.JSONDecodeError) as exc:
        raise ConfigError(f"Failed to parse {spin_manifest_path}: {exc}") from exc

    submitted_info = _parse_submitted_txt(validated_out)
    local_pid = _parse_local_pid(validated_out)
    job_id = _discover_job_id(validated_out, manifest, submitted_info)
    sacct_info, sacct_available, sacct_source = _resolve_sacct_info(
        validated_out,
        job_id,
        sacct_json_path=sacct_json_path,
        min_sacct_interval_seconds=min_sacct_interval_seconds,
    )
    if job_id is None and sacct_info is not None and sacct_info.get("job_id"):
        try:
            job_id = validate_slurm_job_id(sacct_info["job_id"].split(".")[0])
        except ValidationError:
            pass

    members_list = manifest.get("members") or []
    completed_count = 0
    failed_rc: int | None = None
    last_finished: str | None = None
    total_wall = 0.0
    forcing_sets_used: list[str] = []

    for item in members_list:
        fs_name = item.get("forcing_set")
        if isinstance(fs_name, str) and fs_name not in forcing_sets_used:
            forcing_sets_used.append(fs_name)
        m_dir = item.get("experiment_dir")
        if not isinstance(m_dir, str):
            continue
        m_manifest_path = os.path.join(m_dir, "aimip-manifest.json")
        if not os.path.isfile(m_manifest_path):
            continue
        try:
            with open(m_manifest_path, encoding="utf-8") as mfh:
                m_manifest = json.load(mfh)
        except (OSError, json.JSONDecodeError):
            continue
        if "finished" in m_manifest and "rc" in m_manifest:
            rc_i = int(m_manifest["rc"])
            if rc_i == 0:
                completed_count += 1
            elif failed_rc is None:
                failed_rc = rc_i
            last_finished = str(m_manifest.get("finished"))
            total_wall += float(m_manifest.get("wall_seconds") or 0.0)
        elif os.path.isfile(os.path.join(m_dir, "restart.nc")) or (
            os.path.isfile(os.path.join(m_dir, "atmosphere", "restart.nc"))
            and os.path.isfile(os.path.join(m_dir, "ocean", "restart.nc"))
        ):
            completed_count += 1

    total_members = len(members_list)
    if failed_rc is not None:
        state = "failed"
        rc_out: int | None = failed_rc
    elif total_members > 0 and completed_count == total_members:
        state = "completed"
        rc_out = 0
    elif sacct_info is not None and str(sacct_info.get("state") or "").strip():
        state = (sacct_info["state"].lower().split() or ["unknown"])[0]
        rc_out = None
    elif completed_count > 0 or local_pid is not None:
        state = "running"
        rc_out = None
    else:
        state = "unknown"
        rc_out = None

    artifacts = {
        name: os.path.isfile(os.path.join(validated_out, name))
        for name in (
            "spin-manifest.json",
            "job.sbatch",
            "slurm.jobid",
            "sacct.txt",
            "local.pid",
            "submitted.txt",
        )
    }

    return {
        "out_dir": validated_out,
        "model": manifest.get("model", "unknown"),
        "forcing_set": ",".join(forcing_sets_used) if forcing_sets_used else None,
        "mode": manifest.get("mode", "unknown"),
        "state": state,
        "rc": rc_out,
        "created_utc": manifest.get("created_utc"),
        "started_utc": None,
        "finished": last_finished if state in ("completed", "failed") else None,
        "wall_seconds": round(total_wall, 3) if total_wall > 0 else None,
        "slurm_job_id": job_id,
        "local_pid": local_pid,
        "submitted": submitted_info,
        "sacct_available": sacct_available,
        "sacct_source": sacct_source,
        "sacct": sacct_info,
        "start_time": members_list[0].get("from_time") if members_list else None,
        "end_time": manifest.get("target_time"),
        "steps": sum(int(m.get("steps", 0)) for m in members_list),
        "members": total_members,
        "spin_members_completed": completed_count,
        "artifacts": artifacts,
    }


def get_run_status(
    out_dir: str,
    *,
    sacct_json_path: str | None = None,
    min_sacct_interval_seconds: float = 0.0,
) -> dict[str, Any]:
    """Read `<out>/aimip-manifest.json` (or `spin-manifest.json`), `sacct.txt`/`--sacct-json`/`sacct`, and artifacts."""
    validated_out = validate_run_path(out_dir, label="run directory")
    if not os.path.isdir(validated_out):
        raise ValidationError(f"Run directory not found: {validated_out}")

    manifest_path = validate_run_path(
        f"{validated_out}/aimip-manifest.json", label="aimip-manifest.json"
    )
    if not os.path.isfile(manifest_path):
        spin_manifest_path = validate_run_path(
            f"{validated_out}/spin-manifest.json", label="spin-manifest.json"
        )
        if os.path.isfile(spin_manifest_path):
            return _get_spin_status(
                validated_out,
                spin_manifest_path,
                sacct_json_path=sacct_json_path,
                min_sacct_interval_seconds=min_sacct_interval_seconds,
            )
        raise ConfigError(f"Run manifest not found: {manifest_path}")

    try:
        with open(manifest_path, encoding="utf-8") as fh:
            manifest = json.load(fh)
    except (OSError, json.JSONDecodeError) as exc:
        raise ConfigError(f"Failed to parse {manifest_path}: {exc}") from exc

    submitted_info = _parse_submitted_txt(validated_out)
    local_pid = _parse_local_pid(validated_out)
    job_id = _discover_job_id(validated_out, manifest, submitted_info)
    has_finished = "finished" in manifest and "rc" in manifest
    if has_finished and sacct_json_path is None:
        sacct_info = _read_sacct_txt(validated_out, target_job_id=job_id)
        sacct_available = True if sacct_info is not None else (shutil.which("sacct") is not None)
        sacct_source = "sacct.txt" if sacct_info is not None else None
    else:
        sacct_info, sacct_available, sacct_source = _resolve_sacct_info(
            validated_out,
            job_id,
            sacct_json_path=sacct_json_path,
            min_sacct_interval_seconds=min_sacct_interval_seconds,
        )
    if job_id is None and sacct_info is not None and sacct_info.get("job_id"):
        try:
            job_id = validate_slurm_job_id(sacct_info["job_id"].split(".")[0])
        except ValidationError:
            pass

    sacct_age_seconds: float | None = None
    if sacct_source == "sacct.txt":
        try:
            sacct_mtime = os.path.getmtime(os.path.join(validated_out, "sacct.txt"))
            sacct_age_seconds = round(max(0.0, time.time() - sacct_mtime), 1)
        except OSError:
            sacct_age_seconds = None

    if has_finished:
        rc_val = int(manifest["rc"])
        state = "completed" if rc_val == 0 else "failed"
    elif sacct_info is not None and str(sacct_info.get("state") or "").strip():
        state = (sacct_info["state"].lower().split() or ["unknown"])[0]
    elif local_pid is not None:
        in_container = os.environ.get("AIMIP_IMAGE_SHA", "native") not in ("", "native")
        if not in_container:
            try:
                os.kill(local_pid, 0)
                state = "running"
            except ProcessLookupError:
                state = "failed"
            except PermissionError:
                state = "running"
        else:
            state = "running"
    else:
        state = "unknown"

    reg = manifest.get("registry_entry") or {}
    fs_entry = manifest.get("forcing_set") or {}
    artifacts = {
        name: os.path.isfile(os.path.join(validated_out, name))
        for name in (
            "config.yaml",
            "job.sbatch",
            "ic_members.nc",
            "ic_members_atmosphere.nc",
            "ic_members_ocean.nc",
            "slurm.jobid",
            "sacct.txt",
            "local.pid",
            "submitted.txt",
            "cancel-requested.txt",
            "restart.nc",
            "atmosphere/restart.nc",
            "ocean/restart.nc",
            MONTHLY_PREDICTIONS_FILENAME,
            f"atmosphere/{MONTHLY_PREDICTIONS_FILENAME}",
            f"ocean/{MONTHLY_PREDICTIONS_FILENAME}",
            "autoregressive_predictions.nc",
            "inference_out.log",
        )
    }

    return {
        "out_dir": validated_out,
        "model": reg.get("name", "unknown"),
        "forcing_set": fs_entry.get("name"),
        "mode": manifest.get("mode", "unknown"),
        "state": state,
        "rc": manifest.get("rc"),
        "created_utc": manifest.get("created_utc"),
        "started_utc": manifest.get("started_utc"),
        "finished": manifest.get("finished"),
        "wall_seconds": manifest.get("wall_seconds"),
        "slurm_job_id": job_id,
        "local_pid": local_pid,
        "submitted": submitted_info,
        "sacct_available": sacct_available,
        "sacct_source": sacct_source,
        "sacct_age_seconds": sacct_age_seconds,
        "sacct": sacct_info,
        "cancel_requested": bool(artifacts.get("cancel-requested.txt") and not has_finished),
        "start_time": manifest.get("start_time"),
        "end_time": manifest.get("end_time"),
        "steps": manifest.get("steps"),
        "members": manifest.get("members"),
        "artifacts": artifacts,
    }


def _summarize_monthly_dataset(
    monthly_path: str,
    candidate_vars: Sequence[str],
    *,
    prefix: str = "",
    max_var_bytes: int | None = None,
    max_months: int | None = None,
    max_members: int | None = None,
) -> dict[str, Any]:
    """Compute area-weighted (`cos(lat)`) global-mean monthly series for *candidate_vars* in *monthly_path*.

    Opens *monthly_path* lazily without loading unrelated variables, enforces dimension and
    uncompressed-byte limits up front, and streams selected variables in bounded time chunks.
    """
    import numpy as np
    import xarray as xr
    from xarray.coding.times import CFDatetimeCoder

    eff_max_months = MAX_SUMMARY_MONTHS if max_months is None else int(max_months)
    eff_max_members = MAX_SUMMARY_MEMBERS if max_members is None else int(max_members)
    eff_max_bytes = MAX_SUMMARY_VAR_BYTES if max_var_bytes is None else int(max_var_bytes)

    try:
        with xr.open_dataset(
            monthly_path,
            engine="netcdf4",
            decode_timedelta=False,
            decode_times=CFDatetimeCoder(use_cftime=True),
        ) as ds:
            present_vars = [v for v in candidate_vars if v in ds.data_vars]
            if not present_vars:
                available = sorted(
                    str(k)
                    for k in ds.data_vars
                    if k not in ("counts", "valid_time", "init_time")
                )
                expected_names = ", ".join(repr(v) for v in candidate_vars)
                raise ConfigError(
                    f"None of the supported temperature variables ({expected_names}) is present in {monthly_path} "
                    f"(available variables: {', '.join(available) or 'none'})."
                )

            first_var = ds[present_vars[0]]
            if first_var.ndim != 4:
                raise ValidationError(
                    f"Expected 4D ('sample', 'time', y_dim, x_dim) variable {present_vars[0]!r}, "
                    f"got dims {first_var.dims} with shape {first_var.shape}."
                )

            # Detect horizontal dims dynamically (`dims[-2:]`), matching
            # fme.core.dataset.utils.get_horizontal_coordinates (utils.py:219-229).
            sample_dim, time_dim, y_dim, x_dim = (str(d) for d in first_var.dims)
            n_samples = int(first_var.sizes[sample_dim])
            n_months = int(first_var.sizes[time_dim])
            n_lat = int(first_var.sizes[y_dim])
            n_lon = int(first_var.sizes[x_dim])

            if n_months > eff_max_months:
                raise ValidationError(
                    f"Monthly dataset {monthly_path} has {n_months} months, exceeding diagnostic limit ({eff_max_months})."
                )
            if n_samples > eff_max_members:
                raise ValidationError(
                    f"Monthly dataset {monthly_path} has {n_samples} ensemble members, exceeding diagnostic limit ({eff_max_members})."
                )
            for vname in present_vars:
                v_da = ds[vname]
                if v_da.ndim != 4 or set(str(d) for d in v_da.dims) != {
                    sample_dim,
                    time_dim,
                    y_dim,
                    x_dim,
                }:
                    raise ValidationError(
                        f"Expected 4D ({sample_dim!r}, {time_dim!r}, {y_dim!r}, {x_dim!r}) variable {vname!r}, "
                        f"got dims {v_da.dims} with shape {v_da.shape}."
                    )
                itemsize = max(int(getattr(v_da.dtype, "itemsize", 4) or 4), 1)
                est_bytes = int(v_da.size) * itemsize
                if est_bytes > eff_max_bytes:
                    raise ValidationError(
                        f"Variable {vname!r} in {monthly_path} requires {est_bytes} uncompressed bytes, "
                        f"exceeding diagnostic budget ({eff_max_bytes} bytes)."
                    )

            lat_var_name: str | None = None
            if y_dim in ds:
                lat_var_name = y_dim
            else:
                for alt in ("geolat", "lat", "latitude"):
                    if alt in ds and ds[alt].dims in ((y_dim,), (y_dim, x_dim)):
                        lat_var_name = alt
                        break
            if lat_var_name is None:
                raise ValidationError(
                    f"Monthly dataset {monthly_path} is missing latitude coordinate {y_dim!r}."
                )
            if int(ds[lat_var_name].size) > max(n_lat * max(n_lon, 1), 1):
                raise ValidationError(
                    f"Monthly dataset {monthly_path}: {lat_var_name!r} coordinate exceeds spatial grid size."
                )

            lat_vals = np.asarray(ds[lat_var_name].values, dtype=np.float64)
            if lat_vals.ndim == 1 and lat_vals.size == n_lat and n_lat > 0:
                cos_lat = np.maximum(0.0, np.cos(np.deg2rad(lat_vals)))
                if float(np.sum(cos_lat)) <= 0.0:
                    raise ValidationError(
                        f"Monthly dataset {monthly_path}: degenerate {lat_var_name!r} area weights."
                    )
                weights_2d = np.broadcast_to(cos_lat[:, None], (n_lat, n_lon))
            elif lat_vals.ndim == 2 and lat_vals.shape == (n_lat, n_lon):
                weights_2d = np.maximum(0.0, np.cos(np.deg2rad(lat_vals)))
                if float(np.sum(weights_2d)) <= 0.0:
                    raise ValidationError(
                        f"Monthly dataset {monthly_path}: degenerate {lat_var_name!r} area weights."
                    )
            else:
                raise ValidationError(
                    f"Monthly dataset {monthly_path}: {lat_var_name!r} must be a non-empty 1D or 2D array."
                )

            counts_arr: Any = None
            if "counts" in ds:
                counts_arr = ds["counts"].transpose(sample_dim, time_dim).values

            valid_labels: list[str] = []
            if "valid_time" in ds:
                vt = ds["valid_time"].transpose(sample_dim, time_dim).values
                for m_idx in range(n_months):
                    dt0 = vt[0, m_idx]
                    if hasattr(dt0, "year") and hasattr(dt0, "month"):
                        valid_labels.append(f"{int(dt0.year):04d}-{int(dt0.month):02d}")
                    else:
                        valid_labels.append(f"month_{m_idx:04d}")
            else:
                valid_labels = [f"month_{m_idx:04d}" for m_idx in range(n_months)]

            chunk_step = max(1, int(SUMMARY_TIME_CHUNK_MONTHS))
            series_by_var: dict[str, Any] = {}
            for vname in present_vars:
                da = ds[vname]
                raw_units = da.attrs.get("units")
                units = str(raw_units) if raw_units is not None else ""
                long_name = str(da.attrs.get("long_name", vname))

                rows: list[dict[str, Any]] = []
                for t0 in range(0, n_months, chunk_step):
                    t1 = min(n_months, t0 + chunk_step)
                    chunk_da = da.isel({time_dim: slice(t0, t1)}).transpose(
                        sample_dim, time_dim, y_dim, x_dim
                    )
                    chunk_arr = np.asarray(chunk_da.values, dtype=np.float64)
                    for rel_m, m_idx in enumerate(range(t0, t1)):
                        member_means: list[float | None] = []
                        for s_idx in range(n_samples):
                            if counts_arr is not None and int(counts_arr[s_idx, m_idx]) <= 0:
                                member_means.append(None)
                                continue
                            field = chunk_arr[s_idx, rel_m]
                            valid_mask = np.isfinite(field)
                            if not np.any(valid_mask):
                                member_means.append(None)
                                continue
                            w_sum = float(np.sum(weights_2d[valid_mask]))
                            if w_sum <= 0.0:
                                member_means.append(None)
                                continue
                            val = float(
                                np.sum(field[valid_mask] * weights_2d[valid_mask]) / w_sum
                            )
                            member_means.append(val)

                        finite_members = [v for v in member_means if v is not None]
                        ens_mean = (
                            float(sum(finite_members) / len(finite_members))
                            if finite_members
                            else None
                        )
                        step_count = (
                            int(counts_arr[0, m_idx]) if counts_arr is not None else None
                        )
                        rows.append(
                            {
                                "month_index": m_idx,
                                "month": valid_labels[m_idx],
                                "counts": step_count,
                                "ensemble_mean": ens_mean,
                                "members": member_means,
                            }
                        )

                out_key = f"{prefix}{vname}" if prefix else vname
                series_by_var[out_key] = {
                    "units": units,
                    "long_name": long_name,
                    "series": rows,
                }
    except (ValidationError, ConfigError):
        raise
    except Exception as exc:
        raise ValidationError(
            f"Failed to summarize monthly predictions {monthly_path}: {exc}"
        ) from exc

    return {
        "monthly_file": monthly_path,
        "spatial_dims": [y_dim, x_dim],
        "n_samples": n_samples,
        "n_months": n_months,
        "variables": series_by_var,
    }


def _file_cache_sig(path: str) -> tuple[str, int, int]:
    st = os.stat(path)
    return (os.path.realpath(path), int(st.st_size), int(st.st_mtime_ns))


def summarize_run(out_dir: str) -> dict[str, Any]:
    """Compute area-weighted (`cos(lat)`) global-mean monthly series for single-component or coupled runs.

    Caches validated summaries in memory keyed by canonical monthly NetCDF `(path, st_size, st_mtime_ns)`
    so repeated summary/plot queries do not re-read unchanged NetCDF files.
    """
    validated_out = validate_run_path(out_dir, label="run directory")
    if not os.path.isdir(validated_out):
        raise ValidationError(f"Run directory not found: {validated_out}")

    manifest_model: str | None = None
    manifest_file = os.path.join(validated_out, "aimip-manifest.json")
    if os.path.isfile(manifest_file) and not os.path.islink(manifest_file):
        try:
            with open(manifest_file, encoding="utf-8") as mfh:
                mdoc = json.load(mfh)
            reg_name = (mdoc.get("registry_entry") or {}).get("name") or mdoc.get("model")
            if isinstance(reg_name, str) and reg_name.strip():
                manifest_model = reg_name.strip()
        except (OSError, json.JSONDecodeError):
            manifest_model = None

    top_monthly = validate_run_path(
        f"{validated_out}/{MONTHLY_PREDICTIONS_FILENAME}",
        label=MONTHLY_PREDICTIONS_FILENAME,
    )
    atmos_monthly = validate_run_path(
        f"{validated_out}/atmosphere/{MONTHLY_PREDICTIONS_FILENAME}",
        label=f"atmosphere/{MONTHLY_PREDICTIONS_FILENAME}",
    )
    ocean_monthly = validate_run_path(
        f"{validated_out}/ocean/{MONTHLY_PREDICTIONS_FILENAME}",
        label=f"ocean/{MONTHLY_PREDICTIONS_FILENAME}",
    )

    if os.path.isfile(top_monthly):
        cache_key = (
            "single",
            validated_out,
            manifest_model,
            _file_cache_sig(top_monthly),
            MAX_SUMMARY_MONTHS,
            MAX_SUMMARY_MEMBERS,
            MAX_SUMMARY_VAR_BYTES,
        )
        with _SUMMARY_CACHE_LOCK:
            cached = _SUMMARY_CACHE.get(cache_key)
            if cached is not None:
                _SUMMARY_CACHE.move_to_end(cache_key)
                return json.loads(json.dumps(cached))

        primary = _summarize_monthly_dataset(top_monthly, SUMMARY_TEMP_VARS)
        res_single: dict[str, Any] = {
            "out_dir": validated_out,
            **primary,
        }
        if manifest_model:
            res_single["model"] = manifest_model
        with _SUMMARY_CACHE_LOCK:
            _SUMMARY_CACHE[cache_key] = json.loads(json.dumps(res_single))
            while len(_SUMMARY_CACHE) > MAX_SUMMARY_CACHE_ENTRIES:
                _SUMMARY_CACHE.popitem(last=False)
        return res_single

    if os.path.isfile(atmos_monthly):
        has_ocean = os.path.isfile(ocean_monthly)
        cache_key_coupled = (
            "coupled",
            validated_out,
            manifest_model,
            _file_cache_sig(atmos_monthly),
            _file_cache_sig(ocean_monthly) if has_ocean else None,
            MAX_SUMMARY_MONTHS,
            MAX_SUMMARY_MEMBERS,
            MAX_SUMMARY_VAR_BYTES,
        )
        with _SUMMARY_CACHE_LOCK:
            cached_c = _SUMMARY_CACHE.get(cache_key_coupled)
            if cached_c is not None:
                _SUMMARY_CACHE.move_to_end(cache_key_coupled)
                return json.loads(json.dumps(cached_c))

        primary = _summarize_monthly_dataset(atmos_monthly, SUMMARY_TEMP_VARS)
        variables = dict(primary["variables"])
        ocean_file_out: str | None = None
        if has_ocean:
            try:
                ocean_summary = _summarize_monthly_dataset(
                    ocean_monthly, SUMMARY_OCEAN_TEMP_VARS, prefix="ocean."
                )
                variables.update(ocean_summary["variables"])
                ocean_file_out = ocean_monthly
            except ConfigError:
                ocean_file_out = None
        res: dict[str, Any] = {
            "out_dir": validated_out,
            "monthly_file": atmos_monthly,
            "spatial_dims": primary["spatial_dims"],
            "n_samples": primary["n_samples"],
            "n_months": primary["n_months"],
            "variables": variables,
        }
        if manifest_model:
            res["model"] = manifest_model
        if ocean_file_out is not None:
            res["ocean_monthly_file"] = ocean_file_out
        with _SUMMARY_CACHE_LOCK:
            _SUMMARY_CACHE[cache_key_coupled] = json.loads(json.dumps(res))
            while len(_SUMMARY_CACHE) > MAX_SUMMARY_CACHE_ENTRIES:
                _SUMMARY_CACHE.popitem(last=False)
        return res

    raise ConfigError(
        f"Monthly output file not found: {top_monthly} (or coupled {atmos_monthly}; "
        "ensure the run completed with `save_monthly_files: true`)."
    )
