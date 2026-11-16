"""Command-line interface for AIMIP (`python -m aimip`)."""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections.abc import Sequence
from typing import Any

from . import __version__
from .errors import AimipError, ConfigError
from .registry import (
    ModelSpec,
    get_model,
    get_model_ic_provenance,
    load_registry,
    resolve_playground_dir,
)


def _add_registry_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--registry",
        default=None,
        help="Path to registry.json (defaults to $AIMIP_REGISTRY, $AIMIP_DATA/registry.json, or packaged registry.default.json).",
    )
    parser.add_argument(
        "--playground",
        default=None,
        help="Override $AIMIP_PLAYGROUND root directory for ${PLAYGROUND} expansion.",
    )
    parser.add_argument(
        "--data",
        dest="data_dir",
        default=None,
        help="Override $AIMIP_DATA root directory for ${DATA} expansion.",
    )


def _add_slurm_args(parser: argparse.ArgumentParser) -> None:
    slurm_grp = parser.add_argument_group("SLURM / site batch options")
    slurm_grp.add_argument(
        "--site-config",
        dest="site_config",
        default=None,
        help="Path to site.json (defaults to $AIMIP_SITE_CONFIG or $AIMIP_HOME/site.json).",
    )
    slurm_grp.add_argument(
        "--partition",
        default=None,
        help="SLURM partition override (defaults to site.json sbatch.partition).",
    )
    slurm_grp.add_argument(
        "--account",
        default=None,
        help="SLURM account override (defaults to site.json sbatch.account; '' or 'none' omits --account).",
    )
    slurm_grp.add_argument(
        "--gres",
        default=None,
        help="SLURM GPU generic resource override (defaults to site.json sbatch.gres; '' or 'none' omits --gres).",
    )
    slurm_grp.add_argument(
        "--cpus-per-task",
        type=int,
        default=None,
        help="SLURM CPUs per task override (defaults to site.json sbatch.cpus_per_task).",
    )
    slurm_grp.add_argument(
        "--mem",
        default=None,
        help="SLURM memory allocation override (defaults to site.json sbatch.mem).",
    )
    slurm_grp.add_argument(
        "--time",
        default=None,
        help="Override SLURM walltime HH:MM:SS (default: auto-estimated, capped by site.json max_time).",
    )
    slurm_grp.add_argument(
        "--job-name",
        default=None,
        help="Optional SLURM job name token.",
    )


def _format_table(headers: Sequence[str], rows: Sequence[Sequence[str]]) -> str:
    """Render a plain-text aligned table without external formatting dependencies."""
    widths = [len(h) for h in headers]
    for row in rows:
        for idx, cell in enumerate(row):
            if len(cell) > widths[idx]:
                widths[idx] = len(cell)

    def _fmt_row(cells: Sequence[str]) -> str:
        return "  ".join(
            cells[i].ljust(widths[i]) if i < len(cells) - 1 else cells[i]
            for i in range(len(cells))
        )

    lines = [_fmt_row(headers), _fmt_row(["-" * w for w in widths])]
    for row in rows:
        lines.append(_fmt_row(row))
    return "\n".join(lines)


def _probe_model_ic_summary(spec: ModelSpec) -> tuple[list[str] | None, str]:
    """Inspect staged IC timestamps when files exist locally; stay graceful when unstaged."""
    if spec.family == "e2s-dlesym":
        return None, "ERA5 at --start (ARCO-ERA5)"
    if spec.family == "arches":
        return None, "2015 scenario state"
    existing = [p for p in spec.ic_files if os.path.isfile(p)]
    ocean_ok = all(os.path.isfile(p) for p in spec.ocean_ic_files)
    if len(existing) != len(spec.ic_files) or not ocean_ok:
        return None, f"unstaged ({len(spec.ic_files)} file)"

    from .ic import inspect_ic_file

    all_ts: list[str] = []
    try:
        for path in existing:
            info = inspect_ic_file(path)
            all_ts.extend(info.timestamps)
    except Exception:
        return None, f"unreadable ({len(spec.ic_files)} file)"

    if not all_ts:
        return [], "0"
    if len(all_ts) <= 2:
        return all_ts, ", ".join(all_ts)
    return all_ts, f"{len(all_ts)} ({all_ts[0]} .. {all_ts[-1]})"


def _cmd_models(args: argparse.Namespace) -> int:
    catalog = load_registry(
        registry_path=args.registry,
        playground_dir=args.playground,
        data_dir=args.data_dir,
    )
    specs = list(catalog.values())

    if args.json:
        out_list: list[dict[str, Any]] = []
        for spec in specs:
            entry = spec.to_dict()
            ts_list, _ = _probe_model_ic_summary(spec)
            entry["available_ic_timestamps"] = ts_list
            out_list.append(entry)
        print(json.dumps(out_list, indent=2))
        return 0

    rows: list[list[str]] = []
    for spec in specs:
        _, ic_summary = _probe_model_ic_summary(spec)
        base_kind = "stochastic" if spec.stochastic else "deterministic"
        kind = f"coupled/{base_kind}" if spec.is_coupled else base_kind
        sets_str = ", ".join(
            f"{fs.name}[{fs.window[0][:10]}..{fs.window[1][:10]}]"
            for fs in spec.forcing_sets
        )
        runtime = {"e2s-dlesym": "e2s 0.19", "arches": "arches"}.get(spec.family, spec.fme_version)
        dt = "1mo" if spec.family == "arches" else f"{spec.timestep_hours}h"
        if spec.family == "e2s-dlesym":
            kind = "coupled/" + kind
        rows.append(
            [
                spec.name,
                runtime,
                kind,
                dt,
                sets_str,
                ic_summary,
                spec.notes,
            ]
        )

    print(
        _format_table(
            ["MODEL", "RUNTIME", "TYPE", "DT", "FORCING_SETS", "IC_TIMES", "NOTES"],
            rows,
        )
    )
    return 0


def _cmd_ic_list(args: argparse.Namespace) -> int:
    from .ic import list_model_ics, list_model_ocean_ics

    spec = get_model(
        args.model,
        registry_path=args.registry,
        playground_dir=args.playground,
        data_dir=args.data_dir,
    )
    infos = list_model_ics(spec)
    ocean_infos = list_model_ocean_ics(spec) if spec.is_coupled else []
    provenance = get_model_ic_provenance(spec.name, spec.notes)

    if args.json:
        payload: dict[str, Any] = {
            "model": spec.name,
            "ic_provenance": provenance,
            "ic_files": [info.to_dict() for info in infos],
        }
        if spec.is_coupled:
            payload["ocean_ic_files"] = [info.to_dict() for info in ocean_infos]
        print(json.dumps(payload, indent=2))
        return 0

    rows: list[list[str]] = []
    if spec.is_coupled and len(ocean_infos) == len(infos):
        for info, o_info in zip(infos, ocean_infos):
            dims_str = "/".join(info.spatial_dims) if info.spatial_dims else "-"
            for idx, ts in enumerate(info.timestamps):
                rows.append(
                    [ts, info.calendar, dims_str, str(idx), info.path, o_info.path]
                )
        print(
            _format_table(
                ["TIMESTAMP", "CALENDAR", "DIMS", "INDEX", "ATMOS_FILE", "OCEAN_FILE"],
                rows,
            )
        )
    else:
        for info in infos:
            dims_str = "/".join(info.spatial_dims) if info.spatial_dims else "-"
            for idx, ts in enumerate(info.timestamps):
                rows.append([ts, info.calendar, dims_str, str(idx), info.path])
        print(_format_table(["TIMESTAMP", "CALENDAR", "DIMS", "INDEX", "FILE"], rows))

    if provenance:
        print(f"\nProvenance: {provenance}")
    return 0


def _resolve_mode(args: argparse.Namespace) -> str:
    if getattr(args, "local", False):
        return "local"
    if getattr(args, "submit", False):
        return "submit"
    return "dry-run"


def _build_slurm_and_site(args: argparse.Namespace) -> tuple[Any, Any]:
    from .runner import SlurmOptions
    from .site_config import load_site_config

    site_cfg = load_site_config(getattr(args, "site_config", None))
    slurm_opts = SlurmOptions.from_site(
        site_cfg,
        partition=args.partition,
        account=args.account,
        gres=args.gres,
        cpus_per_task=args.cpus_per_task,
        mem=args.mem,
        time=args.time,
        job_name=args.job_name,
    )
    return slurm_opts, site_cfg


def _cmd_ic_spin(args: argparse.Namespace, raw_argv: Sequence[str]) -> int:
    from .ic import collect_spin, prepare_spin
    from .runner import resolve_model_for_prepare
    from .time_utils import compute_spin_steps, select_forcing_set
    from .validators import validate_iso_timestamp

    playground_dir = resolve_playground_dir(args.playground)

    if getattr(args, "collect", False):
        if not args.out:
            raise ConfigError("`aimip ic spin --collect` requires `--out <ic.nc>`.")
        ref_ic: str | None = None
        ref_ocean_ic: str | None = None
        if args.model:
            spec = get_model(
                args.model,
                registry_path=args.registry,
                playground_dir=args.playground,
                data_dir=args.data_dir,
            )
            if spec.ic_files and os.path.isfile(spec.ic_files[0]):
                ref_ic = spec.ic_files[0]
            if spec.ocean_ic_files and os.path.isfile(spec.ocean_ic_files[0]):
                ref_ocean_ic = spec.ocean_ic_files[0]
        resolved_spin = args.spin_dir or f"{args.out}.spin"
        merged_info = collect_spin(
            resolved_spin,
            args.out,
            out_ocean_ic_path=args.ocean_out,
            target_time=args.to_time,
            shipped_ic_reference=ref_ic,
            shipped_ocean_ic_reference=ref_ocean_ic,
            playground_dir=playground_dir,
        )
        if args.json:
            print(json.dumps(merged_info.to_dict(), indent=2))
        else:
            dims_str = "/".join(merged_info.spatial_dims) if merged_info.spatial_dims else "-"
            print(
                f"Merged {merged_info.n_samples} spin restart(s) from {resolved_spin} into {merged_info.path} "
                f"(time={merged_info.timestamps[0]}, calendar={merged_info.calendar}, "
                f"dims=sample/{dims_str}, vars={len(merged_info.variable_names)})."
            )
        return 0

    if not args.model or not args.from_times or not args.to_time:
        raise ConfigError(
            "`aimip ic spin` requires `-m/--model`, `--from T1,T2,...`, and `--to T` "
            "(unless `--collect` is specified)."
        )

    from_list = [tok.strip() for tok in args.from_times.split(",") if tok.strip()]
    mode = _resolve_mode(args)
    slurm_opts, site_cfg = _build_slurm_and_site(args)
    auto_fetch = not bool(getattr(args, "no_fetch", False))

    def _spin_precheck(candidate_spec: ModelSpec) -> None:
        norm_to = validate_iso_timestamp(args.to_time, label="--to")
        for idx, ft in enumerate(from_list):
            norm_from = validate_iso_timestamp(ft, label=f"--from[{idx}]")
            spin_steps = compute_spin_steps(
                norm_from,
                norm_to,
                candidate_spec.timestep_hours,
                calendar=candidate_spec.calendar,
            )
            select_forcing_set(
                norm_from,
                spin_steps,
                candidate_spec.timestep_hours,
                candidate_spec.forcing_sets,
                calendar=candidate_spec.calendar,
                model_name=candidate_spec.name,
                preferred_set_name=args.forcing_set,
            )

    spec = resolve_model_for_prepare(
        args.model,
        registry_path=args.registry,
        playground_dir=args.playground,
        data_dir=args.data_dir,
        auto_fetch=auto_fetch,
        precheck_fn=_spin_precheck,
    )

    # Support both `--out <spin-dir>` and `--out <ic.nc>` (which spins into `<ic.nc>.spin`).
    target_ic_hint: str | None = None
    effective_spin_dir = args.out
    if args.out and args.out.endswith(".nc"):
        target_ic_hint = args.out
        effective_spin_dir = f"{args.out}.spin"
    elif args.spin_dir:
        effective_spin_dir = args.spin_dir

    manifest = prepare_spin(
        model=spec,
        from_times=from_list,
        to_time=args.to_time,
        spin_dir=effective_spin_dir,
        mode=mode,
        seed=0 if args.seed is None else args.seed,
        lora_path=args.lora,
        forcing_set_name=args.forcing_set,
        slurm_opts=slurm_opts,
        site_config=site_cfg,
        playground_dir=playground_dir,
        data_dir=args.data_dir,
        auto_fetch=auto_fetch,
        argv=raw_argv,
    )

    if args.print_sbatch_path:
        print(manifest["sbatch_path"])
        return 0

    if args.json:
        print(json.dumps(manifest, indent=2))
        return 0

    spin_dir = manifest["spin_dir"]
    ic_out_hint = target_ic_hint or f"{spin_dir}/ic.nc"
    print(
        f"Prepared {manifest['member_count']} spin run(s) in {spin_dir} "
        f"(mode={mode}, target={manifest['target_time']})."
    )
    print(f"Sbatch:   {manifest['sbatch_path']}")
    if mode != "local":
        print(f"Next:     aimip submit {spin_dir}")
    print(f"Collect:  aimip collect {spin_dir} --out {ic_out_hint}")
    if target_ic_hint:
        print(f"          (or: aimip ic spin --collect --out {target_ic_hint})")
    else:
        print(f"          (or: aimip ic spin --collect --spin-dir {spin_dir} --out {ic_out_hint})")
    return 0


def _cmd_collect(args: argparse.Namespace) -> int:
    from .ic import collect_spin

    playground_dir = resolve_playground_dir(args.playground)
    ref_ic: str | None = None
    ref_ocean_ic: str | None = None
    if args.model:
        spec = get_model(
            args.model,
            registry_path=args.registry,
            playground_dir=args.playground,
            data_dir=args.data_dir,
        )
        if spec.ic_files and os.path.isfile(spec.ic_files[0]):
            ref_ic = spec.ic_files[0]
        if spec.ocean_ic_files and os.path.isfile(spec.ocean_ic_files[0]):
            ref_ocean_ic = spec.ocean_ic_files[0]

    merged_info = collect_spin(
        args.spin_dir,
        args.out,
        out_ocean_ic_path=args.ocean_out,
        target_time=args.to_time,
        shipped_ic_reference=ref_ic,
        shipped_ocean_ic_reference=ref_ocean_ic,
        playground_dir=playground_dir,
    )
    if args.json:
        print(json.dumps(merged_info.to_dict(), indent=2))
    else:
        dims_str = "/".join(merged_info.spatial_dims) if merged_info.spatial_dims else "-"
        print(
            f"Merged {merged_info.n_samples} spin restart(s) from {args.spin_dir} into {merged_info.path} "
            f"(time={merged_info.timestamps[0]}, calendar={merged_info.calendar}, "
            f"dims=sample/{dims_str}, vars={len(merged_info.variable_names)})."
        )
    return 0


def _cmd_run(args: argparse.Namespace, raw_argv: Sequence[str]) -> int:
    from .runner import prepare_run, resolve_model_for_prepare
    from .time_utils import select_forcing_set
    from .validators import validate_int_range, validate_iso_timestamp

    mode = _resolve_mode(args)
    slurm_opts, site_cfg = _build_slurm_and_site(args)
    playground_dir = resolve_playground_dir(args.playground)
    auto_fetch = not bool(getattr(args, "no_fetch", False))

    def _run_precheck(candidate_spec: ModelSpec) -> None:
        if candidate_spec.family != "fme":
            from .emu import precheck_emu_request

            precheck_emu_request(
                candidate_spec, start_iso=args.start, steps=args.steps, members=args.members,
                forcing_set_name=args.forcing_set, ic_path=args.ic, ocean_ic_path=args.ocean_ic, lora_path=args.lora,
            )
            return
        norm_start = validate_iso_timestamp(args.start, label="--start")
        validated_steps = validate_int_range(
            args.steps, "--steps", min_value=1, max_value=10_000_000
        )
        if args.ocean_ic is not None and not candidate_spec.is_coupled:
            raise ConfigError(
                f"--ocean-ic is only valid for coupled models (fme.coupled.inference), not {candidate_spec.name!r}."
            )
        select_forcing_set(
            norm_start,
            validated_steps,
            candidate_spec.timestep_hours,
            candidate_spec.forcing_sets,
            calendar=candidate_spec.calendar,
            model_name=candidate_spec.name,
            preferred_set_name=args.forcing_set,
        )

    spec = resolve_model_for_prepare(
        args.model,
        registry_path=args.registry,
        playground_dir=args.playground,
        data_dir=args.data_dir,
        auto_fetch=auto_fetch,
        precheck_fn=_run_precheck,
    )

    manifest = prepare_run(
        model=spec,
        start_iso=args.start,
        steps=args.steps,
        out_dir=args.out,
        ic_path=args.ic,
        ocean_ic_path=args.ocean_ic,
        members=args.members,
        seed=args.seed,
        lora_path=args.lora,
        forcing_set_name=args.forcing_set,
        save_monthly=args.monthly,
        save_prediction=args.prediction,
        mode=mode,
        slurm_opts=slurm_opts,
        site_config=site_cfg,
        playground_dir=playground_dir,
        data_dir=args.data_dir,
        auto_fetch=auto_fetch,
        argv=raw_argv,
    )

    if args.print_sbatch_path:
        print(manifest["sbatch_path"])
        return 0

    if args.json:
        print(json.dumps(manifest, indent=2))
        return 0

    out_dir = manifest["out_dir"]
    fs_name = manifest["forcing_set"]["name"]
    print(
        f"Prepared {spec.name} run in {out_dir} "
        f"(mode={mode}, forcing_set={fs_name}, start={manifest['start_time']}, "
        f"end={manifest['end_time']}, steps={manifest['steps']}, "
        f"members={manifest['members']}, seed={manifest['seed']})."
    )
    print(f"Config:   {manifest['config_path']}")
    print(f"Manifest: {out_dir}/aimip-manifest.json")
    print(f"Sbatch:   {manifest['sbatch_path']}")
    if mode != "local":
        print(f"Next:     aimip submit {out_dir}")
    return 0


def _cmd_submit(args: argparse.Namespace) -> int:
    from .submit import submit_run

    res = submit_run(
        args.out,
        playground_dir=args.playground,
        site_config_path=args.site_config,
    )
    if args.json:
        print(json.dumps(res, indent=2))
        return 0
    if res["scheduler"] == "slurm":
        print(
            f"submitted job {res['slurm_job_id']}   (then: aimip status {res['out_dir']})"
        )
    else:
        print(
            f"started locally, pid {res['local_pid']}   (log: {res['log_path']})"
        )
    return 0


def _cmd_exec_fme(args: argparse.Namespace) -> int:
    from .runner import exec_fme
    from .submit import check_trusted_run_dir

    check_trusted_run_dir(args.out)
    exec_fme(args.out)
    return 0


def _cmd_lora_info(args: argparse.Namespace) -> int:
    from .lora import inspect_lora

    info = inspect_lora(args.ckpt)
    if args.json:
        print(json.dumps(info.to_dict(), indent=2))
        return 0

    print(f"Checkpoint:          {info.checkpoint_path}")
    print(f"Loader note:         {info.loader_note}")
    print(f"lora_rank:           {info.lora_rank}")
    print(f"lora_alpha:          {info.lora_alpha} (effective: {info.effective_lora_alpha})")
    print(f"spectral_lora_rank:  {info.spectral_lora_rank}")
    print(
        f"spectral_lora_alpha: {info.spectral_lora_alpha} "
        f"(effective: {info.effective_spectral_lora_alpha})"
    )

    if info.tensors:
        rows = [
            [
                t.module_path,
                t.name,
                t.role,
                "x".join(str(d) for d in t.shape),
                t.dtype,
                f"{t.frobenius_norm:.6g}",
            ]
            for t in info.tensors
        ]
        print()
        print(
            _format_table(
                ["MODULE", "TENSOR", "ROLE", "SHAPE", "DTYPE", "FROBENIUS_NORM"],
                rows,
            )
        )
    else:
        print("\nNo LoRAConv2d (lora_down/lora_up) tensors present in checkpoint.")

    if info.spectral_tensors:
        s_rows = [
            [
                t.module_path,
                t.name,
                t.role,
                "x".join(str(d) for d in t.shape),
                t.dtype,
                f"{t.frobenius_norm:.6g}",
            ]
            for t in info.spectral_tensors
        ]
        print()
        print(
            _format_table(
                ["MODULE", "SPECTRAL_TENSOR", "ROLE", "SHAPE", "DTYPE", "FROBENIUS_NORM"],
                s_rows,
            )
        )
    return 0


def _cmd_lora_scale(args: argparse.Namespace) -> int:
    from .lora import scale_lora

    result = scale_lora(args.ckpt, args.alpha, args.out)
    if args.json:
        print(json.dumps(result, indent=2))
        return 0

    sf = result.get("scale_factor")
    sf_str = f"{sf:.8g}" if sf is not None else "n/a"
    print(f"Input checkpoint:  {result['input_checkpoint']} (sha256={result['input_sha256']})")
    print(f"Output checkpoint: {result['output_checkpoint']} (sha256={result['output_sha256']})")
    print(
        f"Alpha scaling:     mode={result['scaling_mode']}, "
        f"alpha_old={result['alpha_old']} -> alpha_new={result['alpha_new']} "
        f"(scale_factor={sf_str}, weights_modified={result['weights_modified']})"
    )
    if result.get("spectral_lora_rank", 0) > 0:
        print(
            f"Spectral scaling:  spectral_alpha_old={result['spectral_alpha_old']} -> "
            f"spectral_alpha_new={result['spectral_alpha_new']} "
            f"(rank={result['spectral_lora_rank']})"
        )
    print(f"Sidecar metadata:  {result['sidecar_path']}")
    return 0


def _cmd_status(args: argparse.Namespace) -> int:
    from .status import get_run_status

    st = get_run_status(args.out, sacct_json_path=args.sacct_json)
    if args.json:
        print(json.dumps(st, indent=2))
        return 0

    print(f"Run directory: {st['out_dir']}")
    print(f"Model:         {st['model']} (forcing_set={st['forcing_set'] or 'unknown'})")
    print(f"State:         {st['state']}")
    if "spin_members_completed" in st:
        print(f"Spin members:  {st['spin_members_completed']}/{st['members']} completed")
    print(f"Mode:          {st['mode']}")
    print(
        f"SLURM job ID:  {st['slurm_job_id'] or 'none'} "
        f"(sacct_available={st['sacct_available']}, source={st.get('sacct_source') or 'none'})"
    )
    if st.get("local_pid") is not None:
        print(f"Local PID:     {st['local_pid']}")
    if st["sacct"] is not None:
        s = st["sacct"]
        print(f"SLURM sacct:   state={s['state']} elapsed={s['elapsed']} exit={s['exit_code']}")
    if st["rc"] is not None:
        print(f"Exit code:     {st['rc']} (wall_seconds={st['wall_seconds']}, finished={st['finished']})")
    present = [k for k, v in st["artifacts"].items() if v]
    print(f"Artifacts:     {', '.join(present) if present else 'none'}")
    return 0


def _cmd_summary(args: argparse.Namespace) -> int:
    from .status import summarize_run

    summary = summarize_run(args.out)
    if args.json:
        print(json.dumps(summary, indent=2))
        return 0

    print(
        f"Summary for {summary['out_dir']} "
        f"({summary['n_samples']} member(s), {summary['n_months']} month(s), cos(lat) global mean):"
    )
    var_names = list(summary["variables"].keys())
    headers = ["MONTH", "COUNTS"] + [
        f"{v} ({summary['variables'][v]['units']})" for v in var_names
    ]
    rows: list[list[str]] = []
    for m_idx in range(summary["n_months"]):
        first_row = summary["variables"][var_names[0]]["series"][m_idx]
        cnt_str = "-" if first_row["counts"] is None else str(first_row["counts"])
        cells = [first_row["month"], cnt_str]
        for v in var_names:
            val = summary["variables"][v]["series"][m_idx]["ensemble_mean"]
            cells.append("nan" if val is None else f"{val:.4f}")
        rows.append(cells)

    print(_format_table(headers, rows))
    return 0


def _cmd_setup(args: argparse.Namespace) -> int:
    from .setup_data import run_setup

    result = run_setup(
        models=args.models,
        dest_dir=args.dest,
        playground_dir=args.playground,
        lock_path=args.lock,
        only_download=args.only_download,
        verify_only=args.verify_only,
        write_lock=args.write_lock,
        search_dirs=args.search_dirs or (),
        force_rebuild=args.force_rebuild,
        accept_derived=args.accept_derived,
    )
    if args.json:
        print(json.dumps(result, indent=2))
        return 0

    mode = result["mode"]
    if mode == "write-lock":
        print(
            f"Wrote lock {result['lock_path']} from {result['dest_dir']} "
            f"(hashed={result['hashed_files']}, baseline={result['preserved_baseline_files']}, "
            f"total={result['total_entries']})."
        )
        return 0
    if mode == "verify-only":
        print(
            f"Verified {result['verified_files']} file(s) in {result['dest_dir']} "
            f"against {result['lock_path']} for models={','.join(result['models'])}."
        )
        return 0
    if mode == "only-download":
        print(
            f"Downloaded {result['downloaded_files']} file(s) (skipped {result['skipped_existing_files']} existing) "
            f"into {result['dest_dir']} and verified {result['verified_files']} file(s) against {result['lock_path']}."
        )
        return 0

    print(
        f"Setup complete in {result['dest_dir']} for models={','.join(result['models'])} "
        f"(downloaded={result['downloaded_files']}, skipped={result['skipped_existing_files']}, "
        f"built={result['built_files']}, verified={result['verified_files']})."
    )
    print(f"Registry:  {result['registry_path']}")
    print(f"Lock:      {result['lock_path']}")
    if result.get("derived_overrides_path"):
        print(
            f"Overrides: {result['derived_overrides_path']} "
            f"({len(result.get('derived_overrides', ()))} derived override(s))"
        )
    return 0


def _cmd_fetch(args: argparse.Namespace) -> int:
    from .setup_data import fetch_model

    result = fetch_model(
        args.model,
        dest_dir=args.dest,
        playground_dir=args.playground,
        lock_path=args.lock,
        accept_derived=args.accept_derived,
        progress_stream=sys.stderr,
    )
    if args.json:
        print(json.dumps(result, indent=2))
        return 0

    print(
        f"Fetched model(s) {','.join(result['models'])} into {result['dest_dir']} "
        f"(downloaded={result['downloaded_files']}, bytes={result['downloaded_bytes']}, "
        f"skipped={result['skipped_existing_files']}, built={result['built_files']}, "
        f"verified={result['verified_files']})."
    )
    print(f"Registry:  {result['registry_path']}")
    print(f"Lock:      {result['lock_path']}")
    pin_computed = result.get("pin_after_first_fetch") or {}
    if pin_computed:
        for rel_p, info in sorted(pin_computed.items()):
            print(
                f"Pin-after-first-fetch: {rel_p} size={info['size']} sha256={info['sha256']}"
            )
    return 0


def _cmd_site_init(args: argparse.Namespace) -> int:
    from .site_config import init_site_config

    target_path, cfg = init_site_config(
        preset=args.preset,
        out_path=args.out,
        scheduler=args.scheduler,
        runtime=args.runtime,
        partition=args.partition,
        account=args.account,
        gres=args.gres,
        cpus_per_task=args.cpus_per_task,
        mem=args.mem,
        max_time=args.max_time,
        run_root=args.run_root,
        overwrite=args.overwrite,
    )
    payload = {"path": target_path, "site": cfg.to_dict()}
    if args.json:
        print(json.dumps(payload, indent=2))
        return 0

    print(
        f"Wrote site profile {target_path} "
        f"(preset={args.preset}, scheduler={cfg.scheduler}, runtime={cfg.runtime})."
    )
    if cfg.scheduler == "slurm":
        print(
            f"SLURM defaults: partition={cfg.partition} account={cfg.account or '<none>'} "
            f"gres={cfg.gres or '<none>'} cpus={cfg.cpus_per_task} mem={cfg.mem} max_time={cfg.max_time}"
        )
    print(f"Run root:       {cfg.run_root}")
    return 0


def _cmd_site_show(args: argparse.Namespace) -> int:
    from .site_config import load_site_config, resolve_site_config_path

    path, _ = resolve_site_config_path(args.site_config, require_target=False)
    cfg = load_site_config(args.site_config)
    payload = {
        "path": path if (path and os.path.isfile(path)) else None,
        "site": cfg.to_dict(),
    }
    if args.json:
        print(json.dumps(payload, indent=2))
        return 0

    print(f"Site config:    {payload['path'] or 'built-in default'}")
    print(f"Name:           {cfg.name}")
    print(f"Scheduler:      {cfg.scheduler}")
    print(f"Runtime:        {cfg.runtime}")
    if cfg.scheduler == "slurm":
        print(
            f"SLURM defaults: partition={cfg.partition} account={cfg.account or '<none>'} "
            f"gres={cfg.gres or '<none>'} cpus={cfg.cpus_per_task} mem={cfg.mem} max_time={cfg.max_time}"
        )
    print(f"Run root:       {cfg.run_root}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    """Build top-level `aimip` argument parser."""
    parser = argparse.ArgumentParser(
        prog="aimip",
        description="AIMIP CLI for ACE/SamudrACE (`fme`), DLESyM, and ArchesClimate-SSP climate emulators.",
        epilog=(
            "Submission and interactive commands:\n"
            "  submit <run-dir>    Validate and submit <run-dir>/job.sbatch (Slurm or local background).\n"
            "  chat [args...]      Launch the interactive Gemini planning agent (API key or Vertex AI)."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"aimip {__version__}",
    )

    subparsers = parser.add_subparsers(dest="command", required=True)

    # 1. `aimip models`
    p_models = subparsers.add_parser(
        "models",
        help="List registered climate emulator models, forcing sets, and IC availability.",
    )
    p_models.add_argument("--json", action="store_true", help="Emit JSON output.")
    _add_registry_args(p_models)

    # 2. `aimip ic {list,spin}`
    p_ic = subparsers.add_parser(
        "ic",
        help="Inspect shipped initial conditions or prepare multi-start spin runs.",
    )
    ic_sub = p_ic.add_subparsers(dest="ic_command", required=True)

    p_ic_list = ic_sub.add_parser(
        "list",
        help="Decode and print timestamps in shipped IC netCDF files for a model.",
    )
    p_ic_list.add_argument("-m", "--model", required=True, help="Model name in registry.")
    p_ic_list.add_argument("--json", action="store_true", help="Emit JSON output.")
    _add_registry_args(p_ic_list)

    p_ic_spin = ic_sub.add_parser(
        "spin",
        help="Prepare spin runs from shipped ICs T1,T2,... up to target T (or --collect restarts).",
    )
    p_ic_spin.add_argument("-m", "--model", default=None, help="Model name in registry.")
    p_ic_spin.add_argument(
        "--from",
        dest="from_times",
        default=None,
        help="Comma-separated shipped IC timestamps T1,T2,... (each Ti < T).",
    )
    p_ic_spin.add_argument(
        "--to",
        dest="to_time",
        default=None,
        help="Target ISO timestamp T to spin every member up to.",
    )
    p_ic_spin.add_argument(
        "--out",
        default=None,
        help="Spin directory (or target .nc file) under an allowed run root.",
    )
    p_ic_spin.add_argument(
        "--ocean-out",
        dest="ocean_out",
        default=None,
        help="Optional companion ocean IC output path when collecting a coupled spin (default: <out>_ocean.nc).",
    )
    p_ic_spin.add_argument(
        "--spin-dir",
        dest="spin_dir",
        default=None,
        help="Explicit spin directory when running `ic spin --collect --out <ic.nc>`.",
    )
    p_ic_spin.add_argument("--seed", type=int, default=None, help="Base RNG seed (default: 0).")
    p_ic_spin.add_argument("--lora", default=None, help="Optional LoRA checkpoint path.")
    p_ic_spin.add_argument(
        "--forcing-set",
        default=None,
        help="Optional explicit forcing set name when windows overlap.",
    )
    p_ic_spin.add_argument(
        "--no-fetch",
        action="store_true",
        help="Forbid automatic fetch-on-first-use when model files are missing.",
    )
    spin_mode = p_ic_spin.add_mutually_exclusive_group()
    spin_mode.add_argument("--dry-run", action="store_true", help="Prepare spin configs/sbatch only.")
    spin_mode.add_argument(
        "--submit",
        action="store_true",
        help="Prepare spin dirs and print `Next:     aimip submit <spin-dir>`.",
    )
    spin_mode.add_argument("--local", action="store_true", help="Run spin members locally in-process.")
    spin_mode.add_argument(
        "--collect",
        action="store_true",
        help="Merge completed <spin-dir>/0..K-1/restart.nc into `--out <ic.nc>`.",
    )
    p_ic_spin.add_argument(
        "--print-sbatch-path",
        action="store_true",
        help="Print only the generated job.sbatch path(s) on stdout.",
    )
    p_ic_spin.add_argument("--json", action="store_true", help="Emit JSON output.")
    _add_registry_args(p_ic_spin)
    _add_slurm_args(p_ic_spin)

    # 3. `aimip collect <spin-dir> --out <ic.nc>`
    p_collect = subparsers.add_parser(
        "collect",
        help="Merge <spin-dir>/0..K-1/restart.nc into one multi-sample IC netCDF (--out ic.nc).",
    )
    p_collect.add_argument("spin_dir", help="Completed spin directory (<spin-dir>).")
    p_collect.add_argument(
        "--out",
        required=True,
        help="Output merged IC netCDF path under an allowed run root.",
    )
    p_collect.add_argument(
        "--ocean-out",
        dest="ocean_out",
        default=None,
        help="Optional companion ocean IC output path for coupled spins (default: <out>_ocean.nc).",
    )
    p_collect.add_argument("-m", "--model", default=None, help="Optional model name for IC metadata reference.")
    p_collect.add_argument(
        "--to",
        dest="to_time",
        default=None,
        help="Optional expected target ISO timestamp T verification override.",
    )
    p_collect.add_argument("--json", action="store_true", help="Emit JSON output.")
    _add_registry_args(p_collect)

    # 4. `aimip run`
    p_run = subparsers.add_parser(
        "run",
        help="Prepare, submit, or locally execute an AIMIP model run.",
    )
    p_run.add_argument("-m", "--model", required=True, help="Model name in registry.")
    p_run.add_argument("--start", required=True, help="Start ISO timestamp (YYYY-MM-DDTHH:MM:SS).")
    p_run.add_argument("--steps", type=int, required=True, help="Number of forward steps.")
    p_run.add_argument(
        "--ic",
        default=None,
        help="Custom/spun IC netCDF, <prev>/restart.nc, or <prev-run-dir> for coupled runs.",
    )
    p_run.add_argument(
        "--ocean-ic",
        dest="ocean_ic",
        default=None,
        help="Companion ocean IC netCDF for coupled models (`fme.coupled.inference`).",
    )
    p_run.add_argument("--members", type=int, default=None, help="Number of ensemble members K.")
    p_run.add_argument("--seed", type=int, default=None, help="RNG seed for fme.core.set_seed / rollout (default: 0).")
    p_run.add_argument("--lora", default=None, help="Replace model checkpoint with LoRA checkpoint.")
    p_run.add_argument(
        "--forcing-set",
        default=None,
        help="Optional explicit forcing set name (default: auto-selected by [start, end] window).",
    )
    p_run.add_argument(
        "--no-fetch",
        action="store_true",
        help="Forbid automatic fetch-on-first-use when model files are missing.",
    )
    p_run.add_argument(
        "--out",
        default=None,
        help="Output run dir under an allowed run root (default: $AIMIP_DEFAULT_OUT_ROOT/<model>-<start>-<hash>).",
    )
    p_run.add_argument(
        "--monthly",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable/disable monthly_mean_predictions.nc output (default: --monthly).",
    )
    p_run.add_argument(
        "--prediction",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Enable/disable step-by-step autoregressive_predictions.nc output "
            "(~6 GB/yr per member; default: --no-prediction)."
        ),
    )
    run_mode = p_run.add_mutually_exclusive_group()
    run_mode.add_argument("--dry-run", action="store_true", help="Write config, sbatch, and manifest only.")
    run_mode.add_argument(
        "--submit",
        action="store_true",
        help="Prepare files and print `aimip submit <out>` as the next host command.",
    )
    run_mode.add_argument("--local", action="store_true", help="Execute fme immediately in-process.")
    p_run.add_argument(
        "--print-sbatch-path",
        action="store_true",
        help="Print only <out>/job.sbatch on stdout.",
    )
    p_run.add_argument("--json", action="store_true", help="Emit JSON manifest on stdout.")
    _add_registry_args(p_run)
    _add_slurm_args(p_run)

    # 4b. `aimip submit <out>`
    p_submit = subparsers.add_parser(
        "submit",
        help="Validate <out>/job.sbatch in a private copy and submit via sbatch or local background bash.",
    )
    p_submit.add_argument("out", help="Prepared run or spin directory containing job.sbatch.")
    p_submit.add_argument(
        "--playground",
        default=None,
        help="Override $AIMIP_HOME / $AIMIP_PLAYGROUND install prefix.",
    )
    p_submit.add_argument(
        "--site-config",
        dest="site_config",
        default=None,
        help="Explicit site.json path (defaults to $AIMIP_SITE_CONFIG or $AIMIP_HOME/site.json).",
    )
    p_submit.add_argument("--json", action="store_true", help="Emit JSON output.")

    # 5. `aimip exec-fme <out>`
    p_exec = subparsers.add_parser(
        "exec-fme",
        help="Batch/job entrypoint: verify run directory, execute fme or emulator job, and update manifest.",
    )
    p_exec.add_argument("out", help="Prepared run output directory containing config.yaml.")

    # 6. `aimip lora {info,scale}`
    p_lora = subparsers.add_parser(
        "lora",
        help="Inspect or dose-scale LoRA weights in an fme checkpoint.",
    )
    lora_sub = p_lora.add_subparsers(dest="lora_command", required=True)

    p_lora_info = lora_sub.add_parser(
        "info",
        help="Inspect LoRA config and lora_down/lora_up tensor Frobenius norms.",
    )
    p_lora_info.add_argument("ckpt", help="Absolute path to fme checkpoint.")
    p_lora_info.add_argument("--json", action="store_true", help="Emit JSON output.")

    p_lora_scale = lora_sub.add_parser(
        "scale",
        help="Write a copy of CKPT with stepper.config LoRA alpha set to --alpha (weights untouched).",
    )
    p_lora_scale.add_argument("ckpt", help="Absolute path to input fme LoRA checkpoint.")
    p_lora_scale.add_argument(
        "--alpha",
        type=float,
        required=True,
        help="Target effective LoRA alpha (dose knob; 0 zeros out LoRA).",
    )
    p_lora_scale.add_argument(
        "--out",
        required=True,
        help="Absolute path under an allowed run root for new scaled checkpoint (refuses if exists).",
    )
    p_lora_scale.add_argument("--json", action="store_true", help="Emit JSON output.")

    # 7. `aimip status <out>`
    p_status = subparsers.add_parser(
        "status",
        help="Report run state from aimip-manifest.json, sacct.txt / --sacct-json, or sacct.",
    )
    p_status.add_argument("out", help="Run output directory.")
    p_status.add_argument(
        "--sacct-json",
        dest="sacct_json",
        default=None,
        help="Read SLURM sacct JSON from FILE instead of invoking sacct.",
    )
    p_status.add_argument("--json", action="store_true", help="Emit JSON output.")

    # 8. `aimip summary <out>`
    p_summary = subparsers.add_parser(
        "summary",
        help="Print cos(lat)-weighted global-mean monthly temperature / ocean diagnostic series (TMP2m, Tat2m, t2m, tas, TREFHT, TS, ocean.*).",
    )
    p_summary.add_argument("out", help="Run output directory.")
    p_summary.add_argument("--json", action="store_true", help="Emit JSON output.")

    # 9. `aimip setup`
    p_setup = subparsers.add_parser(
        "setup",
        help="Download models & forcings from HF/Zenodo, run deterministic builders, and verify aimip.lock.",
    )
    p_setup.add_argument(
        "--models",
        default="all",
        help="Models to install: 'all' or comma-separated subset (e.g. ace2-era5,hiro-ace).",
    )
    p_setup.add_argument(
        "--dest",
        default=None,
        help="Destination data directory (defaults to $AIMIP_DATA).",
    )
    p_setup.add_argument(
        "--lock",
        default=None,
        help="Path to aimip.lock (defaults to shipped aimip.lock).",
    )
    p_setup.add_argument(
        "--playground",
        default=None,
        help="Fallback $AIMIP_PLAYGROUND root if $AIMIP_DATA is unset.",
    )
    setup_mode = p_setup.add_mutually_exclusive_group()
    setup_mode.add_argument(
        "--only-download",
        action="store_true",
        help="Download and verify raw HF/Zenodo artifacts only (skip derived builders).",
    )
    setup_mode.add_argument(
        "--verify-only",
        action="store_true",
        help="Verify existing files under --dest against aimip.lock without downloading or building.",
    )
    setup_mode.add_argument(
        "--write-lock",
        action="store_true",
        help="Generate or update aimip.lock by hashing an existing staged tree under --dest.",
    )
    p_setup.add_argument(
        "--search-dir",
        dest="search_dirs",
        action="append",
        default=None,
        help="Extra root directory to search when running --write-lock (repeatable).",
    )
    p_setup.add_argument(
        "--force-rebuild",
        action="store_true",
        help="Rebuild derived netCDF files even if already present and matching aimip.lock.",
    )
    p_setup.add_argument(
        "--accept-derived",
        action="store_true",
        help=(
            "Opt-in: accept non-locked derived netCDF hashes (recording <dest>/derived-overrides.json) "
            "while still strictly verifying HF/Zenodo downloads (e.g. experimental/untested aarch64/arm64 hosts)."
        ),
    )
    p_setup.add_argument("--json", action="store_true", help="Emit JSON output.")

    # 9b. `aimip fetch -m <model>`
    p_fetch = subparsers.add_parser(
        "fetch",
        help="Download, build (if derived), and lock-verify only the requested model(s) into $AIMIP_DATA.",
    )
    p_fetch.add_argument(
        "-m",
        "--model",
        required=True,
        help="Model name (or comma-separated model names) to fetch and verify against aimip.lock.",
    )
    p_fetch.add_argument(
        "--dest",
        default=None,
        help="Destination data directory (defaults to $AIMIP_DATA).",
    )
    p_fetch.add_argument(
        "--lock",
        default=None,
        help="Path to aimip.lock (defaults to shipped aimip.lock).",
    )
    p_fetch.add_argument(
        "--playground",
        default=None,
        help="Fallback $AIMIP_PLAYGROUND root if $AIMIP_DATA is unset.",
    )
    p_fetch.add_argument(
        "--accept-derived",
        action="store_true",
        help="Opt-in: accept non-locked derived netCDF hashes while strictly verifying HF/Zenodo downloads.",
    )
    p_fetch.add_argument("--json", action="store_true", help="Emit JSON output.")

    # 10. `aimip site {init,show}`
    p_site = subparsers.add_parser(
        "site",
        help="Initialize or inspect portable HPC site configuration ($AIMIP_HOME/site.json).",
    )
    site_sub = p_site.add_subparsers(dest="site_command", required=True)

    p_site_init = site_sub.add_parser(
        "init",
        help="Write $AIMIP_HOME/site.json from a preset (levante, generic-slurm, local) and overrides.",
    )
    p_site_init.add_argument(
        "--preset",
        choices=("levante", "generic-slurm", "local"),
        default="levante",
        help="Site preset (default: levante).",
    )
    p_site_init.add_argument(
        "--out",
        default=None,
        help="Output site.json path (defaults to $AIMIP_HOME/site.json).",
    )
    p_site_init.add_argument(
        "--scheduler",
        choices=("slurm", "none"),
        default=None,
        help="Override batch scheduler (slurm or none).",
    )
    p_site_init.add_argument(
        "--runtime",
        choices=("apptainer", "singularity", "docker", "podman", "native"),
        default=None,
        help="Override runtime (container runtime or native).",
    )
    p_site_init.add_argument("--partition", default=None, help="Override SLURM partition.")
    p_site_init.add_argument(
        "--account", default=None, help="Override SLURM account ('' or 'none' omits --account)."
    )
    p_site_init.add_argument(
        "--gres", default=None, help="Override SLURM --gres ('' or 'none' omits --gres)."
    )
    p_site_init.add_argument(
        "--cpus-per-task", type=int, default=None, help="Override SLURM --cpus-per-task."
    )
    p_site_init.add_argument("--mem", default=None, help="Override SLURM --mem.")
    p_site_init.add_argument(
        "--max-time", default=None, help="Override default SLURM walltime cap HH:MM:SS."
    )
    p_site_init.add_argument(
        "--run-root", default=None, help="Override default run output root path or template."
    )
    p_site_init.add_argument(
        "--overwrite", action="store_true", help="Overwrite an existing site.json file."
    )
    p_site_init.add_argument("--json", action="store_true", help="Emit JSON output.")

    p_site_show = site_sub.add_parser(
        "show",
        help="Display effective site.json configuration.",
    )
    p_site_show.add_argument(
        "--site-config",
        default=None,
        help="Explicit site.json path (defaults to $AIMIP_SITE_CONFIG or $AIMIP_HOME/site.json).",
    )
    p_site_show.add_argument("--json", action="store_true", help="Emit JSON output.")

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entrypoint returning POSIX exit code."""
    effective_argv = list(sys.argv[1:] if argv is None else argv)
    if effective_argv[:1] == ["chat"]:  # the Gemini agent has its own argument parser
        from .agent import main as agent_main

        return int(agent_main(effective_argv[1:]) or 0)
    parser = build_parser()
    args = parser.parse_args(effective_argv)
    recorded_argv = ["aimip", *effective_argv]

    try:
        if args.command == "models":
            return _cmd_models(args)
        if args.command == "ic":
            from .emu_catalog import EMU_MODELS

            if getattr(args, "model", None) in EMU_MODELS:   # by name: the model may not be fetched yet
                raise ConfigError(
                    f"`aimip ic` works with the fme models only; {args.model} starts from "
                    + ("the scenario's 2015 state." if args.model == "archesclimate-ssp"
                       else "ERA5 at --start (fetched by `aimip run`).")
                )
            if args.ic_command == "list":
                return _cmd_ic_list(args)
            if args.ic_command == "spin":
                return _cmd_ic_spin(args, recorded_argv)
        if args.command == "collect":
            return _cmd_collect(args)
        if args.command == "run":
            return _cmd_run(args, recorded_argv)
        if args.command == "submit":
            return _cmd_submit(args)
        if args.command == "exec-fme":
            return _cmd_exec_fme(args)
        if args.command == "lora":
            if args.lora_command == "info":
                return _cmd_lora_info(args)
            if args.lora_command == "scale":
                return _cmd_lora_scale(args)
        if args.command == "status":
            return _cmd_status(args)
        if args.command == "summary":
            return _cmd_summary(args)
        if args.command == "setup":
            return _cmd_setup(args)
        if args.command == "fetch":
            return _cmd_fetch(args)
        if args.command == "site":
            if args.site_command == "init":
                return _cmd_site_init(args)
            if args.site_command == "show":
                return _cmd_site_show(args)
    except AimipError as exc:
        print(f"aimip error: {exc}", file=sys.stderr)
        return 2

    parser.error(f"Unhandled command {args.command!r}")
    return 2
