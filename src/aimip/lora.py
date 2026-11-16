"""Inspection and alpha-dose scaling for LoRA weights in fme stepper checkpoints.

All `torch` imports are intentionally lazy inside functions so that CLI commands
such as `aimip models`, `aimip ic list`, and `aimip run --dry-run` execute without
importing PyTorch.

fme 2026.5.1 references:
- Checkpoint loading (`weights_only=False` required because fme serializes Python
  tuples/metadata objects inside `checkpoint["stepper"]`):
  `fme/ace/stepper/single_module.py:1813, 1854`
- Checkpoint saving layout (`{"stepper": stepper.get_state(), ...}`):
  `fme/core/generics/trainer.py:667-680`,
  `fme/ace/stepper/single_module.py:1288-1298`
- Stepper config & module state paths:
  `fme/core/step/single_module.py:58, 224-228` (`step.config.builder`, `step.module`),
  `fme/core/step/multi_call.py:80, 179-181` (`step.config.wrapped_step...`),
  `fme/core/step/secondary_module.py:66-70` (`builder`, `secondary_builder`),
  `fme/core/step/radiation.py:56-57` (`builder`, `radiation_builder`),
  `fme/ace/stepper/single_module.py:114, 1347-1380` (legacy `config.builder`, `module`).
- `LoRAConv2d` implementation & scaling (`lora_down`, `lora_up`,
  `lora_alpha = float(lora_alpha) if lora_alpha is not None else float(lora_rank)`,
  `lora_scaling = lora_alpha / lora_rank`):
  `fme/core/models/conditional_sfno/lora.py:61-107, 139-141`
- Builder LoRA config fields (`lora_rank`, `lora_alpha`, `spectral_lora_rank`,
  `spectral_lora_alpha`):
  `fme/core/models/conditional_sfno/sfnonet.py:115-118`,
  `fme/ace/registry/stochastic_sfno.py:257-260`,
  `fme/ace/registry/local_net.py:176-177`,
  `fme/core/models/conditional_sfno/s2convolutions.py:148-169`.
"""

from __future__ import annotations

import copy
import dataclasses
import datetime
import hashlib
import json
import os
from typing import Any

from .errors import ConfigError, ValidationError
from .registry import resolve_playground_dir
from .validators import (
    makedirs_no_group_other_write,
    strip_group_other_write,
    validate_finite_float,
    validate_run_path,
    validate_safe_path,
    validate_visible_input_path,
)

WEIGHTS_ONLY_NOTE = (
    "Loaded with torch.load(..., map_location='cpu', weights_only=False): "
    "required for fme checkpoints because checkpoint['stepper'] serializes "
    "Python metadata/config objects alongside module state_dicts "
    "(see fme/ace/stepper/single_module.py:1813,1854)."
)

_MODULE_STATE_KEYS = ("module", "secondary_module", "radiation_module")
_BUILDER_CONFIG_KEYS = ("builder", "secondary_builder", "radiation_builder")


@dataclasses.dataclass(frozen=True)
class LoRATensorStat:
    """Metadata and Frobenius norm for a single LoRA parameter tensor."""

    module_path: str
    name: str
    role: str  # "lora_down" | "lora_up" | "lora_A" | "lora_B"
    shape: tuple[int, ...]
    dtype: str
    frobenius_norm: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "module_path": self.module_path,
            "name": self.name,
            "role": self.role,
            "shape": list(self.shape),
            "dtype": self.dtype,
            "frobenius_norm": self.frobenius_norm,
        }


@dataclasses.dataclass(frozen=True)
class BuilderLoRAConfig:
    """LoRA configuration extracted from a module builder selector in stepper config."""

    builder_path: str
    builder_type: str
    lora_rank: int
    lora_alpha: float | None
    effective_lora_alpha: float
    spectral_lora_rank: int
    spectral_lora_alpha: float | None
    effective_spectral_lora_alpha: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "builder_path": self.builder_path,
            "builder_type": self.builder_type,
            "lora_rank": self.lora_rank,
            "lora_alpha": self.lora_alpha,
            "effective_lora_alpha": self.effective_lora_alpha,
            "spectral_lora_rank": self.spectral_lora_rank,
            "spectral_lora_alpha": self.spectral_lora_alpha,
            "effective_spectral_lora_alpha": self.effective_spectral_lora_alpha,
        }


@dataclasses.dataclass(frozen=True)
class LoRAInfo:
    """Structured result of `aimip lora info CKPT`."""

    checkpoint_path: str
    lora_rank: int
    lora_alpha: float | None
    effective_lora_alpha: float
    spectral_lora_rank: int
    spectral_lora_alpha: float | None
    builders: tuple[BuilderLoRAConfig, ...]
    tensors: tuple[LoRATensorStat, ...]
    spectral_tensors: tuple[LoRATensorStat, ...]
    effective_spectral_lora_alpha: float = 0.0
    in_names: tuple[str, ...] = ()
    out_names: tuple[str, ...] = ()
    loader_note: str = WEIGHTS_ONLY_NOTE

    @property
    def has_lora_tensors(self) -> bool:
        return any(t.role == "lora_up" for t in self.tensors) or any(
            t.role == "lora_B" for t in self.spectral_tensors
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "checkpoint_path": self.checkpoint_path,
            "loader_note": self.loader_note,
            "lora_rank": self.lora_rank,
            "lora_alpha": self.lora_alpha,
            "effective_lora_alpha": self.effective_lora_alpha,
            "spectral_lora_rank": self.spectral_lora_rank,
            "spectral_lora_alpha": self.spectral_lora_alpha,
            "effective_spectral_lora_alpha": self.effective_spectral_lora_alpha,
            "in_names": list(self.in_names),
            "out_names": list(self.out_names),
            "builders": [b.to_dict() for b in self.builders],
            "tensors": [t.to_dict() for t in self.tensors],
            "spectral_tensors": [t.to_dict() for t in self.spectral_tensors],
        }


def _import_torch() -> Any:
    """Lazy import of PyTorch."""
    try:
        import torch
    except ImportError as exc:
        raise ConfigError(
            "PyTorch (`torch`) is required for LoRA checkpoint operations "
            "(`aimip lora ...` or `aimip run --lora`), but is not installed."
        ) from exc
    return torch


def sha256_file(path: str) -> str:
    """Compute hex SHA-256 digest of *path* in streaming 1 MiB chunks."""
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        while True:
            chunk = fh.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _extract_builders_from_step_config(
    step_cfg: dict[str, Any],
    prefix: str,
    out: list[BuilderLoRAConfig],
) -> None:
    """Recursively inspect fme StepSelector / legacy stepper config dicts for builders."""
    if not isinstance(step_cfg, dict):
        return

    for key in _BUILDER_CONFIG_KEYS:
        selector = step_cfg.get(key)
        if isinstance(selector, dict) and "type" in selector:
            b_type = str(selector.get("type", ""))
            b_cfg = selector.get("config")
            if not isinstance(b_cfg, dict):
                b_cfg = {}
            raw_rank = b_cfg.get("lora_rank", 0)
            lora_rank = int(raw_rank) if raw_rank is not None else 0
            raw_alpha = b_cfg.get("lora_alpha", None)
            lora_alpha = float(raw_alpha) if raw_alpha is not None else None
            # Mirrors LoRAConv2d.__init__ (fme/core/models/conditional_sfno/lora.py:84-86):
            # self.lora_alpha = float(lora_alpha) if lora_alpha is not None else float(lora_rank)
            effective_alpha = lora_alpha if lora_alpha is not None else float(lora_rank)

            raw_spec_rank = b_cfg.get("spectral_lora_rank", 0)
            spec_rank = int(raw_spec_rank) if raw_spec_rank is not None else 0
            raw_spec_alpha = b_cfg.get("spectral_lora_alpha", None)
            spec_alpha = float(raw_spec_alpha) if raw_spec_alpha is not None else None
            # Mirrors SpectralConvS2.__init__ (fme/core/models/conditional_sfno/s2convolutions.py:167-168):
            # self.lora_alpha = lora_alpha if lora_alpha is not None else lora_rank
            effective_spec_alpha = (
                (spec_alpha if spec_alpha is not None else float(spec_rank))
                if spec_rank > 0
                else 0.0
            )

            out.append(
                BuilderLoRAConfig(
                    builder_path=f"{prefix}.{key}",
                    builder_type=b_type,
                    lora_rank=lora_rank,
                    lora_alpha=lora_alpha,
                    effective_lora_alpha=effective_alpha,
                    spectral_lora_rank=spec_rank,
                    spectral_lora_alpha=spec_alpha,
                    effective_spectral_lora_alpha=effective_spec_alpha,
                )
            )

    # MultiCallStepConfig nests another StepSelector under `wrapped_step`:
    # fme/core/step/multi_call.py:80
    wrapped = step_cfg.get("wrapped_step")
    if isinstance(wrapped, dict):
        wrapped_cfg = wrapped.get("config")
        if isinstance(wrapped_cfg, dict):
            _extract_builders_from_step_config(
                wrapped_cfg, f"{prefix}.wrapped_step.config", out
            )


def _extract_builders(stepper_state: dict[str, Any]) -> list[BuilderLoRAConfig]:
    """Extract all module builder LoRA configs from `checkpoint['stepper']`."""
    config = stepper_state.get("config")
    if not isinstance(config, dict):
        raise ValidationError(
            "Invalid fme checkpoint: checkpoint['stepper']['config'] must be a dict."
        )

    builders: list[BuilderLoRAConfig] = []
    step_selector = config.get("step")
    if isinstance(step_selector, dict):
        step_cfg = step_selector.get("config")
        if isinstance(step_cfg, dict):
            _extract_builders_from_step_config(
                step_cfg, "stepper.config.step.config", builders
            )

    # Legacy SingleModuleStepperConfig stores `builder` directly under `stepper["config"]`
    # (fme/ace/stepper/single_module.py:114, 1347-1380).
    if "builder" in config:
        _extract_builders_from_step_config(config, "stepper.config", builders)

    return builders


def _collect_io_names_from_cfg(
    cfg: dict[str, Any],
    in_acc: list[str],
    out_acc: list[str],
) -> None:
    """Collect string variable names from `in_names` and `out_names` in an fme step/stepper config."""
    if not isinstance(cfg, dict):
        return
    raw_in = cfg.get("in_names")
    if isinstance(raw_in, (list, tuple)):
        for item in raw_in:
            if isinstance(item, str) and item and item not in in_acc:
                in_acc.append(item)
    raw_out = cfg.get("out_names")
    if isinstance(raw_out, (list, tuple)):
        for item in raw_out:
            if isinstance(item, str) and item and item not in out_acc:
                out_acc.append(item)

    wrapped = cfg.get("wrapped_step")
    if isinstance(wrapped, dict) and isinstance(wrapped.get("config"), dict):
        _collect_io_names_from_cfg(wrapped["config"], in_acc, out_acc)


def _extract_io_names(stepper_state: dict[str, Any]) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Extract `(in_names, out_names)` from `checkpoint['stepper']['config']` when present."""
    config = stepper_state.get("config")
    if not isinstance(config, dict):
        return (), ()
    in_names: list[str] = []
    out_names: list[str] = []
    step_selector = config.get("step")
    if isinstance(step_selector, dict) and isinstance(step_selector.get("config"), dict):
        _collect_io_names_from_cfg(step_selector["config"], in_names, out_names)
    _collect_io_names_from_cfg(config, in_names, out_names)
    return tuple(in_names), tuple(out_names)


def _collect_module_state_dicts(
    container: dict[str, Any],
    prefix: str,
    out: list[tuple[str, dict[str, Any]]],
) -> None:
    """Recursively locate module state_dicts inside `checkpoint['stepper']`."""
    for key in _MODULE_STATE_KEYS:
        candidate = container.get(key)
        if isinstance(candidate, dict):
            out.append((f"{prefix}.{key}", candidate))

    wrapped = container.get("wrapped_step")
    if isinstance(wrapped, dict):
        _collect_module_state_dicts(wrapped, f"{prefix}.wrapped_step", out)


def _find_all_module_state_dicts(
    stepper_state: dict[str, Any],
) -> list[tuple[str, dict[str, Any]]]:
    """Find all module parameter state_dicts in current or legacy fme stepper states."""
    found: list[tuple[str, dict[str, Any]]] = []
    step_state = stepper_state.get("step")
    if isinstance(step_state, dict):
        _collect_module_state_dicts(step_state, "stepper.step", found)
    if "module" in stepper_state and isinstance(stepper_state["module"], dict):
        found.append(("stepper.module", stepper_state["module"]))
    if not found:
        raise ValidationError(
            "Invalid fme checkpoint: no module state_dict found under "
            "checkpoint['stepper']['step'] or checkpoint['stepper']['module']."
        )
    return found


def _classify_lora_key(param_name: str) -> str | None:
    """Classify a state_dict parameter key into LoRA role if applicable."""
    parts = param_name.split(".")
    if "lora_down" in parts:
        return "lora_down"
    if "lora_up" in parts:
        return "lora_up"
    if "lora_A" in parts:
        return "lora_A"
    if "lora_B" in parts:
        return "lora_B"
    return None


def _load_fme_checkpoint(ckpt_path: str) -> tuple[Any, dict[str, Any]]:
    """Validate path, load checkpoint on CPU with `weights_only=False`, and validate fme layout."""
    validated_path = validate_safe_path(ckpt_path, label="checkpoint path")
    if not os.path.isfile(validated_path):
        raise ValidationError(f"Checkpoint file not found: {validated_path}")

    torch = _import_torch()
    try:
        # weights_only=False is required for fme checkpoints (fme/ace/stepper/single_module.py:1813,1854).
        checkpoint = torch.load(validated_path, map_location="cpu", weights_only=False)
    except Exception as exc:
        raise ValidationError(
            f"Failed to load checkpoint {validated_path} with torch.load: {exc}"
        ) from exc

    if not isinstance(checkpoint, dict) or "stepper" not in checkpoint:
        raise ValidationError(
            f"Invalid fme checkpoint {validated_path}: expected top-level dict containing 'stepper'."
        )
    stepper_state = checkpoint["stepper"]
    if not isinstance(stepper_state, dict):
        raise ValidationError(
            f"Invalid fme checkpoint {validated_path}: checkpoint['stepper'] must be a dict."
        )
    if "config" not in stepper_state or not isinstance(stepper_state["config"], dict):
        raise ValidationError(
            f"Invalid fme checkpoint {validated_path}: checkpoint['stepper']['config'] is missing."
        )

    # Also verify at least one module state_dict exists.
    _find_all_module_state_dicts(stepper_state)
    return torch, checkpoint


def _inspect_loaded_checkpoint(
    validated_path: str,
    torch: Any,
    checkpoint: dict[str, Any],
) -> LoRAInfo:
    """Build `LoRAInfo` from an already-loaded fme checkpoint dict."""
    stepper_state = checkpoint["stepper"]
    builders = _extract_builders(stepper_state)
    module_dicts = _find_all_module_state_dicts(stepper_state)

    conv_tensors: list[LoRATensorStat] = []
    spectral_tensors: list[LoRATensorStat] = []

    for mod_path, state_dict in module_dicts:
        for key, value in state_dict.items():
            if not isinstance(key, str) or not torch.is_tensor(value):
                continue
            role = _classify_lora_key(key)
            if role is None:
                continue
            frob = float(torch.linalg.vector_norm(value.detach().to(torch.float64)).item())
            stat = LoRATensorStat(
                module_path=mod_path,
                name=key,
                role=role,
                shape=tuple(int(d) for d in value.shape),
                dtype=str(value.dtype).replace("torch.", ""),
                frobenius_norm=frob,
            )
            if role in ("lora_down", "lora_up"):
                conv_tensors.append(stat)
            else:
                spectral_tensors.append(stat)

    # Choose primary LoRA builder config (preferring a builder with active LoRA if multiple exist).
    primary: BuilderLoRAConfig | None = None
    for b in builders:
        if b.lora_rank > 0 or b.spectral_lora_rank > 0:
            primary = b
            break
    if primary is None and builders:
        primary = builders[0]

    lora_rank = primary.lora_rank if primary is not None else 0
    lora_alpha = primary.lora_alpha if primary is not None else None
    effective_alpha = (
        primary.effective_lora_alpha
        if primary is not None
        else (lora_alpha if lora_alpha is not None else float(lora_rank))
    )
    spectral_lora_rank = primary.spectral_lora_rank if primary is not None else 0
    spectral_lora_alpha = primary.spectral_lora_alpha if primary is not None else None
    effective_spectral_alpha = (
        primary.effective_spectral_lora_alpha
        if primary is not None
        else (
            (spectral_lora_alpha if spectral_lora_alpha is not None else float(spectral_lora_rank))
            if spectral_lora_rank > 0
            else 0.0
        )
    )
    in_names, out_names = _extract_io_names(stepper_state)

    return LoRAInfo(
        checkpoint_path=validated_path,
        lora_rank=lora_rank,
        lora_alpha=lora_alpha,
        effective_lora_alpha=effective_alpha,
        spectral_lora_rank=spectral_lora_rank,
        spectral_lora_alpha=spectral_lora_alpha,
        effective_spectral_lora_alpha=effective_spectral_alpha,
        in_names=in_names,
        out_names=out_names,
        builders=tuple(builders),
        tensors=tuple(conv_tensors),
        spectral_tensors=tuple(spectral_tensors),
    )


def inspect_lora(ckpt_path: str) -> LoRAInfo:
    """Load an fme checkpoint on CPU and return its LoRA configuration and tensor norms."""
    validated_path = validate_safe_path(ckpt_path, label="checkpoint path")
    torch, checkpoint = _load_fme_checkpoint(validated_path)
    return _inspect_loaded_checkpoint(validated_path, torch, checkpoint)


def _update_step_config_lora_alphas(
    step_cfg: dict[str, Any],
    prefix: str,
    target_alpha: float,
    retained_ratios: dict[str, float],
    new_ratios: dict[str, float],
    updated_builders: list[str],
) -> None:
    """Update `lora_alpha` / `spectral_lora_alpha` in-place on a deep-copied step config.

    Never inserts `spectral_lora_alpha` on builders with `spectral_lora_rank <= 0`
    (such as `LocalNetBuilder`, which only defines `lora_rank` / `lora_alpha` under
    `dacite.Config(strict=True)`), and never inserts `lora_alpha` when `lora_rank <= 0`.
    """
    if not isinstance(step_cfg, dict):
        return

    for key in _BUILDER_CONFIG_KEYS:
        selector = step_cfg.get(key)
        if not (isinstance(selector, dict) and "type" in selector):
            continue
        b_cfg = selector.get("config")
        if not isinstance(b_cfg, dict):
            continue

        b_path = f"{prefix}.{key}"
        raw_rank = b_cfg.get("lora_rank", 0)
        lora_rank = int(raw_rank) if raw_rank is not None else 0
        raw_alpha = b_cfg.get("lora_alpha", None)
        lora_alpha = float(raw_alpha) if raw_alpha is not None else None
        conv_old = lora_alpha if lora_alpha is not None else float(lora_rank)

        raw_spec_rank = b_cfg.get("spectral_lora_rank", 0)
        spec_rank = int(raw_spec_rank) if raw_spec_rank is not None else 0
        raw_spec_alpha = b_cfg.get("spectral_lora_alpha", None)
        spec_alpha = float(raw_spec_alpha) if raw_spec_alpha is not None else None
        spec_old = spec_alpha if spec_alpha is not None else float(spec_rank)

        if lora_rank > 0 and spec_rank <= 0:
            b_cfg["lora_alpha"] = float(target_alpha)
            updated_builders.append(b_path)
        elif lora_rank <= 0 and spec_rank > 0:
            b_cfg["spectral_lora_alpha"] = float(target_alpha)
            updated_builders.append(b_path)
        elif lora_rank > 0 and spec_rank > 0:
            if conv_old != 0.0:
                ratio = spec_old / conv_old
            elif b_path in retained_ratios:
                ratio = float(retained_ratios[b_path])
            elif target_alpha == 0.0:
                ratio = None
            else:
                raise ConfigError(
                    f"Builder {b_path} has both lora_rank={lora_rank} and "
                    f"spectral_lora_rank={spec_rank} with effective lora_alpha=0.0 and "
                    "no retained spectral-to-conv alpha ratio; cannot infer spectral_lora_alpha."
                )

            b_cfg["lora_alpha"] = float(target_alpha)
            if target_alpha == 0.0:
                b_cfg["spectral_lora_alpha"] = 0.0
            else:
                assert ratio is not None
                b_cfg["spectral_lora_alpha"] = float(target_alpha) * ratio
            if ratio is not None:
                new_ratios[b_path] = ratio
            updated_builders.append(b_path)

    wrapped = step_cfg.get("wrapped_step")
    if isinstance(wrapped, dict) and isinstance(wrapped.get("config"), dict):
        _update_step_config_lora_alphas(
            wrapped["config"],
            f"{prefix}.wrapped_step.config",
            target_alpha,
            retained_ratios,
            new_ratios,
            updated_builders,
        )


def scale_lora(
    ckpt_path: str,
    alpha: float,
    out_path: str,
) -> dict[str, Any]:
    """Write a copy of *ckpt_path* to *out_path* with LoRA dose set in `stepper['config']`.

    Why config-based scaling (`lora_alpha` / `spectral_lora_alpha`) is used:
    Both `LoRAConv2d` (`fme/core/models/conditional_sfno/lora.py:84-107`) and
    `SpectralConvS2` (`fme/core/models/conditional_sfno/s2convolutions.py:167-168`)
    reconstruct `lora_scaling = alpha / rank` from `stepper['config']` at load time,
    treating `0.0` (`is not None`) as exact zero contribution (`alpha=0` recovers the
    exact base model across both conv and spectral LoRA branches). Keeping all parameter
    tensors untouched means `aimip lora info` and `aimip-manifest.json` report the true
    effective alpha and repeated `lora scale` operations are lossless and idempotent.

    Safety guarantees:
    - Enforces container-visible input roots (`validate_visible_input_path`) and writable
      output roots (`validate_run_path`).
    - Never adds `spectral_lora_alpha` to builders with `spectral_lora_rank <= 0` (`LocalNet` safe).
    - Never modifies *ckpt_path* (verified via pre/post SHA-256 check).
    - Refuses to overwrite *out_path* or `<out_path>.json` (atomic `'xb'` creation).
    """
    safe_out_pre = validate_safe_path(out_path, label="output checkpoint path")
    pg_dir = resolve_playground_dir()
    if pg_dir:
        pg = pg_dir.rstrip("/")
        if safe_out_pre == pg or safe_out_pre.startswith(pg + "/"):
            raise ConfigError(
                f"Refusing to write scaled checkpoint ({safe_out_pre}) inside "
                f"$AIMIP_PLAYGROUND ({pg}); choose a writable run root outside the playground."
            )

    validated_in = validate_visible_input_path(ckpt_path, label="input checkpoint path")
    validated_out = validate_run_path(out_path, label="output checkpoint path")
    target_alpha = validate_finite_float(alpha, label="alpha")

    if validated_in == validated_out:
        raise ConfigError(
            f"Refusing to overwrite input checkpoint: --out ({validated_out}) equals input path."
        )

    sidecar_path = validate_run_path(f"{validated_out}.json", label="sidecar json path")
    if os.path.exists(validated_out):
        raise ConfigError(
            f"Refusing to overwrite existing output checkpoint: {validated_out}"
        )
    if os.path.exists(sidecar_path):
        raise ConfigError(
            f"Refusing to overwrite existing sidecar metadata file: {sidecar_path}"
        )
    if not os.path.isfile(validated_in):
        raise ValidationError(f"Input checkpoint file not found: {validated_in}")

    input_sha_before = sha256_file(validated_in)
    torch, checkpoint = _load_fme_checkpoint(validated_in)
    info = _inspect_loaded_checkpoint(validated_in, torch, checkpoint)

    has_conv = any(b.lora_rank > 0 for b in info.builders)
    has_spec = any(b.spectral_lora_rank > 0 for b in info.builders)
    if not has_conv and not has_spec:
        raise ConfigError(
            f"Checkpoint {validated_in} has lora_rank={info.lora_rank} and "
            f"spectral_lora_rank={info.spectral_lora_rank}; cannot scale LoRA on a checkpoint without LoRA enabled."
        )

    lora_up_count = sum(1 for t in info.tensors if t.role == "lora_up")
    if has_conv and lora_up_count == 0:
        raise ConfigError(
            f"Checkpoint {validated_in} declares lora_rank > 0 but contains no 'lora_up' tensors."
        )
    lora_b_count = sum(1 for t in info.spectral_tensors if t.role == "lora_B")
    if has_spec and lora_b_count == 0:
        raise ConfigError(
            f"Checkpoint {validated_in} declares spectral_lora_rank > 0 but contains no 'lora_B' spectral LoRA tensors."
        )

    alpha_old = (
        info.effective_lora_alpha
        if info.lora_rank > 0
        else info.effective_spectral_lora_alpha
    )
    spectral_alpha_old = (
        info.effective_spectral_lora_alpha if info.spectral_lora_rank > 0 else None
    )
    if alpha_old != 0.0:
        scale_factor: float | None = target_alpha / alpha_old
    elif target_alpha == 0.0:
        scale_factor = 0.0
    else:
        scale_factor = None

    # Copy outer containers and deep-copy ONLY `stepper["config"]`; every parameter
    # tensor in `state_dict` (`lora_down`, `lora_up`, `lora_A`, `lora_B`, base weights)
    # remains bit-for-bit identical to the input checkpoint.
    out_ckpt = copy.copy(checkpoint)
    out_stepper = copy.copy(checkpoint["stepper"])
    new_stepper_cfg = copy.deepcopy(checkpoint["stepper"]["config"])
    out_stepper["config"] = new_stepper_cfg
    out_ckpt["stepper"] = out_stepper

    prior_meta = checkpoint.get("aimip_lora_scaling")
    retained_ratios: dict[str, float] = {}
    if isinstance(prior_meta, dict) and isinstance(prior_meta.get("builder_ratios"), dict):
        for k, v in prior_meta["builder_ratios"].items():
            if isinstance(k, str) and isinstance(v, (int, float)):
                retained_ratios[k] = float(v)

    new_ratios: dict[str, float] = dict(retained_ratios)
    updated_builders: list[str] = []

    step_selector = new_stepper_cfg.get("step")
    if isinstance(step_selector, dict) and isinstance(step_selector.get("config"), dict):
        _update_step_config_lora_alphas(
            step_selector["config"],
            "stepper.config.step.config",
            target_alpha,
            retained_ratios,
            new_ratios,
            updated_builders,
        )
    if "builder" in new_stepper_cfg:
        _update_step_config_lora_alphas(
            new_stepper_cfg,
            "stepper.config",
            target_alpha,
            retained_ratios,
            new_ratios,
            updated_builders,
        )

    if new_ratios:
        out_ckpt["aimip_lora_scaling"] = {
            "builder_ratios": new_ratios,
        }

    updated_info = _inspect_loaded_checkpoint(validated_out, torch, out_ckpt)
    spectral_alpha_new = (
        updated_info.effective_spectral_lora_alpha
        if updated_info.spectral_lora_rank > 0
        else None
    )

    parent_dir = os.path.dirname(validated_out)
    if parent_dir:
        makedirs_no_group_other_write(parent_dir)

    # Atomic exclusive open ('xb') guarantees we never clobber an existing file.
    try:
        with open(validated_out, "xb") as fh:
            torch.save(out_ckpt, fh)
    except FileExistsError as exc:
        raise ConfigError(
            f"Refusing to overwrite existing output checkpoint: {validated_out}"
        ) from exc
    strip_group_other_write(validated_out)

    output_sha = sha256_file(validated_out)
    input_sha_after = sha256_file(validated_in)
    if input_sha_after != input_sha_before:
        raise ConfigError(
            f"Integrity check failed: input checkpoint {validated_in} SHA-256 changed!"
        )

    sidecar_payload: dict[str, Any] = {
        "input_checkpoint": validated_in,
        "input_sha256": input_sha_before,
        "output_checkpoint": validated_out,
        "output_sha256": output_sha,
        "scaling_mode": "stepper_config",
        "weights_modified": False,
        "lora_rank": info.lora_rank,
        "config_lora_alpha": info.lora_alpha,
        "alpha_old": alpha_old,
        "alpha_new": target_alpha,
        "scale_factor": scale_factor,
        "spectral_lora_rank": info.spectral_lora_rank,
        "config_spectral_lora_alpha": info.spectral_lora_alpha,
        "spectral_alpha_old": spectral_alpha_old,
        "spectral_alpha_new": spectral_alpha_new,
        "updated_builders": updated_builders,
        "scaled_lora_up_count": 0,
        "scaled_tensors": [],
        "loader_note": WEIGHTS_ONLY_NOTE,
        "created_utc": datetime.datetime.now(datetime.timezone.utc).strftime(
            "%Y-%m-%dT%H:%M:%SZ"
        ),
    }

    try:
        with open(sidecar_path, "xb") as raw_fh:
            raw_fh.write(
                (json.dumps(sidecar_payload, indent=2, sort_keys=False) + "\n").encode(
                    "utf-8"
                )
            )
    except FileExistsError as exc:
        raise ConfigError(
            f"Refusing to overwrite existing sidecar metadata file: {sidecar_path}"
        ) from exc
    strip_group_other_write(sidecar_path)

    sidecar_payload["sidecar_path"] = sidecar_path
    return sidecar_payload
