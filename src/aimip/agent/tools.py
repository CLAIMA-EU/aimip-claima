"""CLI adapters and session-local approval. No shell or scheduler execution."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
import tempfile

from .security import trusted_lora_path

MAX_MEMBERS = 64
MAX_SPIN_MEMBERS = 24
MAX_REPLICATION_BYTES = 2 * 1024**3


def cli_json(args: list[str]):
    # Preparation/inspection subprocesses do not need the model's credentials and must never auto-fetch into $AIMIP_DATA.
    env = {k: v for k, v in os.environ.items() if k not in
           {"GEMINI_API_KEY", "GOOGLE_API_KEY", "AIMIP_GEMINI_KEY_FILE",
            "AIMIP_GEMINI_VERTEX_TOKEN_FILE", "GOOGLE_APPLICATION_CREDENTIALS"}
           and not k.startswith(("LANGCHAIN_", "LANGSMITH_", "APPTAINERENV_GEMINI",
                                  "APPTAINERENV_GOOGLE", "SINGULARITYENV_GEMINI", "SINGULARITYENV_GOOGLE"))}
    env["AIMIP_DATA_MODE"] = "ro"
    result = subprocess.run(
        [sys.executable, "-I", "-B", "-m", "aimip", *args, "--json"],
        capture_output=True, text=True, env=env, timeout=300, check=False,
    )
    if result.returncode:
        raise ValueError(result.stderr.strip() or f"CLI exited {result.returncode}")
    return json.loads(result.stdout)


def _with_fetch_hint(model: str, text: str, *, custom_ic: bool = False) -> str:
    """Append an `aimip fetch -m <model>` hint when a canonical model is not installed yet (empty or partial registry)."""
    from aimip.setup_data import CANONICAL_MODELS

    low = text.lower()
    if "aimip fetch" in low or model not in CANONICAL_MODELS:
        return text
    missing_model_or_data = (
        "unknown model" in low
        or "missing required data" in low
        or (not custom_ic and "file not found" in low)
    )
    if missing_model_or_data:
        return f"{text} If model {model} is not downloaded yet, run `aimip fetch -m {model}` on the host first."
    return text


def fingerprint(files: list[Path]) -> str:
    digest = hashlib.sha256()
    for path in files:
        if path.is_symlink() or not path.is_file():
            raise ValueError(f"Plan artifact missing or symlinked: {path}")
        digest.update(str(path.resolve()).encode())
        with path.open("rb") as stream:
            digest.update(hashlib.file_digest(stream, "sha256").digest())
    return digest.hexdigest()


class AgentTools:
    """Only the REPL may call present/confirm; these are never model tools."""

    def __init__(self, cli=cli_json):
        self.cli = cli
        self.pending = None
        self.shown = None
        self.approved = None

    def revoke(self):
        self.pending = self.shown = self.approved = None

    def list_models(self) -> list:
        """List the installed model catalogue, forcing windows, and IC availability."""
        return self.cli(["models"])

    def list_ics(self, model: str) -> dict:
        """List actual initial-condition files and timestamps for a model."""
        try:
            return self.cli(["ic", "list", f"--model={model}"])
        except ValueError as exc:
            raise ValueError(_with_fetch_hint(model, str(exc))) from exc

    def plan_run(self, model: str, start: str, steps: int, members: int | None = None,
                 ic: str | None = None, seed: int | None = None,
                 forcing_set: str | None = None, lora: str | None = None) -> dict:
        """Prepare a fresh dry-run plan; never execute it. Use tool-provided ICs/windows."""
        self.revoke()
        if lora is not None:
            lora = trusted_lora_path(lora)
        self._check_run_size(model, start, members, ic)
        args = ["run", "--no-fetch", f"--model={model}", f"--start={start}",
                f"--steps={steps}"]
        for name, value in (("members", members), ("seed", seed), ("ic", ic),
                            ("forcing-set", forcing_set), ("lora", lora)):
            if value is not None:
                args.append(f"--{name}={value}")
        try:
            return self._plan(args, spin=False)
        except ValueError as exc:
            raise ValueError(_with_fetch_hint(model, str(exc), custom_ic=ic is not None)) from exc

    def plan_spin(self, model: str, from_times: list[str], to_time: str,
                  seed: int = 0, forcing_set: str | None = None,
                  lora: str | None = None) -> dict:
        """Prepare fresh spin members from shipped IC timestamps to the target time."""
        self.revoke()
        if not 1 <= len(from_times) <= MAX_SPIN_MEMBERS:
            raise ValueError(f"Chat spin plans require 1–{MAX_SPIN_MEMBERS} starts")
        if lora is not None:
            lora = trusted_lora_path(lora)
        args = ["ic", "spin", "--no-fetch", f"--model={model}", f"--from={','.join(from_times)}",
                f"--to={to_time}", f"--seed={seed}"]
        for name, value in (("forcing-set", forcing_set), ("lora", lora)):
            if value is not None:
                args.append(f"--{name}={value}")
        try:
            return self._plan(args, spin=True)
        except ValueError as exc:
            raise ValueError(_with_fetch_hint(model, str(exc))) from exc

    @staticmethod
    def _check_run_size(model, start, members, ic):
        from aimip.errors import AimipError
        from aimip.registry import get_model
        from aimip.runner import resolve_ic_and_members
        from aimip.validators import validate_iso_timestamp
        import netCDF4

        if members is not None and not 1 <= members <= MAX_MEMBERS:
            raise ValueError(f"Chat plans support at most {MAX_MEMBERS} members")
        from aimip.emu_catalog import EMU_MODELS

        if model in EMU_MODELS:   # DLESyM / ArchesClimate: no IC file to inspect; aimip run validates the rest
            return
        try:
            resolved = resolve_ic_and_members(model=get_model(model),
                        norm_start=validate_iso_timestamp(start), ic_path=ic, members=members)
        except AimipError as exc:
            raise ValueError(_with_fetch_hint(model, str(exc), custom_ic=ic is not None)) from exc
        if resolved.total_members > MAX_MEMBERS:
            raise ValueError(f"IC implies {resolved.total_members} members; chat limit is {MAX_MEMBERS}")
        if resolved.replicate_repeats > 1:
            # Metadata only: use uncompressed variable sizes (atmosphere + ocean if coupled), not compressed file size.
            ic_paths = [resolved.source_ic.path]
            if resolved.source_ocean_ic is not None:
                ic_paths.append(resolved.source_ocean_ic.path)
            size = 0
            for ic_file in ic_paths:
                with netCDF4.Dataset(ic_file) as dataset:
                    size += sum(v.size * getattr(v.dtype, "itemsize", 8) for v in dataset.variables.values())
            estimate = size * resolved.replicate_repeats
            if estimate > MAX_REPLICATION_BYTES:
                raise ValueError(f"IC replication estimate {estimate} bytes exceeds chat limit {MAX_REPLICATION_BYTES}; use the CLI deliberately")

    def plan_collect(self, spin_dir: str) -> dict:
        """Plan collection of completed spin restarts; yes prints a host collect command."""
        from aimip.validators import validate_visible_input_path, validate_run_path
        self.revoke()
        source = Path(validate_visible_input_path(spin_dir, label="spin directory"))
        manifest_path = source / "spin-manifest.json"
        manifest = json.loads(manifest_path.read_text())
        members = manifest["members"]
        if not 1 <= len(members) <= MAX_SPIN_MEMBERS:
            raise ValueError(f"Chat collection supports 1–{MAX_SPIN_MEMBERS} spin members")
        coupled = bool(manifest.get("is_coupled"))
        files = [manifest_path]
        for member in members:
            directory = Path(member["experiment_dir"])
            if directory.parent.resolve() != source.resolve():
                raise ValueError("Spin member escaped its directory")
            restarts = ([directory / "atmosphere" / "restart.nc", directory / "ocean" / "restart.nc"]
                        if coupled else [directory / "restart.nc"])
            for restart in restarts:
                if not restart.is_file():
                    raise ValueError(f"Spin restart not ready: {restart}")
                files.append(restart)
        if sum(p.stat().st_size for p in files) > MAX_REPLICATION_BYTES:
            raise ValueError("Collection exceeds chat size limit; use the CLI deliberately")
        output_dir = self._fresh_dir()
        output = validate_run_path(str(output_dir / "ic.nc"), label="collected IC output")
        card = {"kind": "collect", "run_dir": str(output_dir), "spin_dir": str(source),
                "model": manifest["model"], "members": len(members),
                "target_time": manifest["target_time"], "calendar": manifest["calendar"],
                "output": output, "gpu_hours_estimate": 0,
                "submission": "After yes, run the printed collect command on the host"}
        if coupled:
            card["ocean_output"] = validate_run_path(str(output_dir / "ic_ocean.nc"), label="collected ocean IC output")
        try:
            plan_file = output_dir / "collect-plan.json"
            plan_file.write_text(json.dumps(card))
            files.append(plan_file)
            return self._remember(card, files)
        except Exception:
            shutil.rmtree(output_dir)
            raise

    @staticmethod
    def _fresh_dir():
        from aimip.runner import _resolve_default_out_root
        root = Path(_resolve_default_out_root())
        root.mkdir(parents=True, exist_ok=True)
        return Path(tempfile.mkdtemp(prefix="chat-", dir=root))

    def _remember(self, card, files):
        digest = fingerprint(files)
        card["plan_id"] = digest
        self.pending = {"card": card, "files": files, "digest": digest}
        return json.loads(json.dumps(card))

    @staticmethod
    def _outputs(manifest):
        import yaml

        if manifest.get("family") in ("e2s-dlesym", "arches"):
            return {"monthly": True, "prediction": False,
                    "files": ["atmosphere/monthly_mean_predictions.nc", "ocean/monthly_mean_predictions.nc"]}
        writer = yaml.safe_load(Path(manifest["config_path"]).read_text())["data_writer"]
        if "save_monthly_files" not in writer:  # coupled: data_writer = {ocean: {...}, atmosphere: {...}}
            return {part: {"monthly": w.get("save_monthly_files"), "prediction": w.get("save_prediction_files")}
                    for part, w in writer.items() if isinstance(w, dict)}
        return {"monthly": writer["save_monthly_files"], "prediction": writer["save_prediction_files"]}

    def _plan(self, args, *, spin):
        # Even a failed replacement invalidates the previous approval.
        self.revoke()
        out = self._fresh_dir()
        try:
            return self._prepare(args, spin, out)
        except BaseException:
            self.revoke()
            shutil.rmtree(out)  # only this call's newly allocated directory
            raise

    def _prepare(self, args, spin, out):
        from aimip.runner import (ACE2_ERA5_GPU_HOURS_PER_MODEL_YEAR,
                                  HOURS_PER_JULIAN_YEAR, compute_slurm_walltime_details)
        from aimip.site_config import load_site_config

        manifest = self.cli([*args, f"--out={out}", "--dry-run"])
        files = [out / "job.sbatch"]
        if spin:
            files.append(out / "spin-manifest.json")
            runs = []
            for member in manifest["members"]:
                directory = Path(member["experiment_dir"])
                if directory.parent != out:
                    raise ValueError("Spin member escaped its fresh plan directory")
                runs.append(json.loads((directory / "aimip-manifest.json").read_text()))
                files.extend(directory / name for name in
                             ("aimip-manifest.json", "config.yaml", "job.sbatch"))
        else:
            runs = [manifest]
            files.extend(out / name for name in ("aimip-manifest.json", "config.yaml"))
        details = [{"model": m["registry_entry"]["name"], "start": m["start_time"],
                    "end": m["end_time"], "steps": m["steps"], "members": m["members"],
                    "ic_source": m["source_ic_path"], "ic_path": m["ic_path"],
                    "forcing_set": m["forcing_set"], "seed": m["seed"],
                    "checkpoint": m["checkpoint_path"], "sbatch_resources": m["slurm"],
                    "stochastic": m["registry_entry"]["stochastic"],
                    "calendar": m["ic_calendar"], "ic_provenance": m["ic_provenance"],
                    "model_notes": m["registry_entry"].get("notes", ""),
                    "ensemble_strategy": m["ensemble_strategy"], "ensemble_note": m["ensemble_note"],
                    "seed_warning": m.get("seed_warning"),
                    "lora_checkpoint": m["lora_checkpoint"], "lora_info": m["lora_info"],
                    "outputs": self._outputs(m),
                    "walltime_capped": m["slurm"].get("walltime_capped", False)}
                   for m in runs]
        gpu_hours = sum(m["gpu_hours_estimate"] if "gpu_hours_estimate" in m else
                        m["steps"] * m["members"] * m["registry_entry"]["timestep_hours"]
                        / HOURS_PER_JULIAN_YEAR * ACE2_ERA5_GPU_HOURS_PER_MODEL_YEAR for m in runs)
        capped = any(run["walltime_capped"] for run in details)
        if spin:
            _, _, capped = compute_slurm_walltime_details(
                steps=sum(m["steps"] for m in runs),
                timestep_hours=runs[0]["registry_entry"]["timestep_hours"], n_trajectories=1,
                max_walltime=load_site_config().max_time)
        card = {"run_dir": str(out), "kind": "spin" if spin else "run",
                "runs": details, "gpu_hours_estimate": round(gpu_hours, 6),
                "walltime_capped": capped,
                "estimate_basis": "ACE2-ERA5 baseline; approximate for other models, not a quote",
                "sbatch_script": (out / "job.sbatch").read_text(),
                "submission": "Requires a standalone yes after the REPL displays this plan"}
        return self._remember(card, files)

    def present(self) -> str | None:
        if self.pending is None:
            return None
        self.shown = self.pending["digest"]
        return json.dumps(self.pending["card"], indent=2)

    def confirm(self, answer: str):
        self.approved = None
        if answer.strip().lower() != "yes" or self.pending is None or self.shown is None:
            raise ValueError("No displayed plan approved; type exactly yes after a plan")
        current = fingerprint(self.pending["files"])
        if current != self.shown or current != self.pending["digest"]:
            self.revoke()
            raise ValueError("Plan changed; prepare and review a new plan")
        self.approved = current

    def submit(self, run_dir: str) -> dict:
        """Return the host submit command only for the exact human-approved plan."""
        approval, self.approved = self.approved, None  # one attempt consumes approval
        if self.pending is None or approval is None:
            raise ValueError("Submission refused: the human has not approved this plan")
        if run_dir != self.pending["card"]["run_dir"]:
            raise ValueError("Submission refused: approval is for a different plan")
        if fingerprint(self.pending["files"]) != approval:
            self.revoke()
            raise ValueError("Plan changed; prepare and review a new plan")
        card = self.pending["card"]
        collecting = card["kind"] == "collect"
        if collecting:
            targets = [Path(card["output"])] + ([Path(card["ocean_output"])] if card.get("ocean_output") else [])
            if any(t.exists() or t.is_symlink() for t in targets):
                self.revoke()
                raise ValueError("Collection output already exists; prepare a new plan")
            command = shlex.join(["aimip", "collect", card["spin_dir"], "--out", card["output"]])
        else:
            command = shlex.join(["aimip", "submit", run_dir])
        self.revoke()
        return {"submitted": False, "host_command": command,
                "message": "Run this command on the host; " +
                ("no IC has been collected by chat" if collecting else "no job has been submitted by chat")}

    def status(self, run_dir: str) -> dict:
        """Inspect run status; container accounting may lag host sacct."""
        from aimip.validators import validate_safe_path
        return self.cli(["status", validate_safe_path(run_dir)])

    def summary(self, run_dir: str) -> dict:
        """Summarize monthly mean output for a completed run."""
        from aimip.validators import validate_safe_path
        return self.cli(["summary", validate_safe_path(run_dir)])

    def lora_info(self, ckpt: str) -> dict:
        """Inspect a trusted checkpoint (CLI uses pickle; this is not a safe unpickler)."""
        return self.cli(["lora", "info", trusted_lora_path(ckpt)])
