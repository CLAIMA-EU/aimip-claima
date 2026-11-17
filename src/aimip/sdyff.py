"""Spherical DYffusion (`spherical-dyffusion`, 6-hourly probabilistic 1° FV3GFS emulator) adapter & runner.

Upstream provenance & licences:
- Code (Apache-2.0): `Rose-STL-Lab/spherical-dyffusion` commit `a10b3e7f1ae529193537a128c86b531adb45f23b`
  (`284,547` B archive, SHA-256 `ec3b96de...`, 190 files, tree SHA-256 `21c0c611...`;
  Rühling Cachay et al., *NeurIPS 2024*, `arXiv:2406.14798`) + pinned SFNO dependency
  `ai2cm/modulus` (`94f62e1ce2083640829ec12d80b00619c40a47f8`, pinned in upstream `setup.py`,
  `529,142` B archive, SHA-256 `8785a70f...`, 219 files, tree SHA-256 `988a84ae...`, Apache-2.0).
- Weights (Apache-2.0): `hf://salv47/spherical-dyffusion@56fcb5d995717e91ba770f5ee41e8c3a8cfb00f6`
  (`forecaster-sfno-best-inference_avg_crps.ckpt`, `interpolator-sfno-best-val_avg_crps.ckpt`,
  `interpolator_sfno_paper_v0_hydra_config.yaml`).
- Reference initial-state & forcing data (CC-BY-4.0): Zenodo record `10791087`
  (`2021010100.nc .. 2021120100.nc`, 12 monthly files of 1° `180 x 360` FV3GFS repeating-SST
  reference simulation, `20.38 GB`; `ace_ckpt.tar` is skipped).

Scientific & security invariants:
- Cryptographic SHA-256 verification on an `O_NOFOLLOW` descriptor is the primary integrity gate
  (both in parent pre-flight and inside the child via `install_sdyff_verified_loaders` wrapping
  `torch.load` and `OmegaConf.load` so checkpoints and Hydra configs are read directly from their
  verified descriptors before `hydra.utils.instantiate` runs). As defense-in-depth at the pickle
  layer, both `.ckpt` files are loaded with `torch.load(..., weights_only=False,
  pickle_module=make_sdyff_restricted_pickle_module())` whose `Unpickler.find_class` allows ONLY
  the exact globals present in the published checkpoints (`builtins` `dict`/`list`/`int`(`long`),
  `collections` `OrderedDict`/`defaultdict`, `omegaconf.base` `ContainerMetadata`/`Metadata`,
  `omegaconf.dictconfig.DictConfig`, `omegaconf.listconfig.ListConfig`, `omegaconf.nodes.AnyNode`,
  `typing.Any`, `torch._utils._rebuild_tensor_v2` (+ `_rebuild_parameter` if present), and `torch`
  `FloatStorage`/`IntStorage`/`DoubleStorage`) and refuses everything else (PyTorch 2.14's
  `WeightsUnpickler` cannot `SETITEM` into `collections.defaultdict`, even under `safe_globals`).
- Runs with `WANDB_MODE=disabled`, `HF_HUB_OFFLINE=1`, `logging.log_to_wandb=False`, explicit
  `seed` (`overrides.seed` + Python/NumPy/PyTorch/Lightning RNG seeding), and
  `data_writer.time_coarsen.coarsen_factor = 4` (exact daily means from `4 x 6h` steps, never
  approximate 30-day blocks).
- Only complete 365-day calendar months are written (`--steps 240` = `60` days from Jan 1 ->
  `2` full cycle months: Month 1 = 31 days / 124 steps, Month 2 = 28 days / 112 steps).
- Output timestamps use nominal cycle indices (`0001-01`, `0001-02`, ... in `noleap`
  `"days since 0001-01-01 00:00:00"`), NEVER relabelled to `1995..` or `2021..`, with
  `calendar_note="repeating single-year SST cycle; year numbers are nominal cycle indices, not historical calendar years"`.
- `surface_temperature` is overwritten over ocean (`ocean_fraction == 1`) by the prescriber, so
  it is written as `surface_temperature_prescribed_ocean` in `monthly_mean_predictions.nc` while
  predicted lowest-model-layer air temperature `air_temperature_7` (`K`) is exposed for
  `aimip summary` and `aimip ui`.
"""

from __future__ import annotations

import builtins
import collections
from collections.abc import Callable, Iterator, Mapping, MutableMapping
import contextlib
import datetime
import hashlib
import importlib
import io
import json
import os
import pickle
import shutil
import stat
import subprocess
import sys
import tarfile
import time
import types
import typing
import uuid
from typing import Any
import zipfile

from .errors import ConfigError, ValidationError
from .safe_pickle import compute_sha256sum_tree_digest, normalize_sha256, open_verified_hash_stream
from .validators import makedirs_no_group_other_write, strip_group_other_write, validate_safe_path

SDYFF_CODE_REPO = "Rose-STL-Lab/spherical-dyffusion"
SDYFF_CODE_COMMIT = "a10b3e7f1ae529193537a128c86b531adb45f23b"
SDYFF_CODE_LICENCE = "Apache-2.0"
SDYFF_GITHUB_ARCHIVE_URL = (
    f"https://github.com/{SDYFF_CODE_REPO}/archive/{SDYFF_CODE_COMMIT}.tar.gz"
)
SDYFF_GITHUB_ARCHIVE_SIZE = 284_547
SDYFF_GITHUB_ARCHIVE_SHA256 = (
    "ec3b96dec5c3648441199903651883e6485dd0a3e385ee4e7f17119a094fa6d7"
)
SDYFF_TREE_FILES = 190
SDYFF_TREE_SHA256 = (
    "21c0c611999d10c5d8904b4d3f828320677ee9628a062f9fc1e31a4e5944106a"
)
SDYFF_CODE_REL_DIR = "code/spherical-dyffusion"

SDYFF_MODULUS_REPO = "ai2cm/modulus"
SDYFF_MODULUS_COMMIT = "94f62e1ce2083640829ec12d80b00619c40a47f8"
SDYFF_MODULUS_LICENCE = "Apache-2.0"
SDYFF_MODULUS_ARCHIVE_URL = (
    f"https://github.com/{SDYFF_MODULUS_REPO}/archive/{SDYFF_MODULUS_COMMIT}.tar.gz"
)
SDYFF_MODULUS_ARCHIVE_SIZE = 529_142
SDYFF_MODULUS_ARCHIVE_SHA256 = (
    "8785a70f2609746beaa5a3ff166b1a99393a9a969afef44c77fe1270ac5af246"
)
SDYFF_MODULUS_TREE_FILES = 219
SDYFF_MODULUS_TREE_SHA256 = (
    "988a84aefeaefcab618a87bcd48ff010e986dad7e5e378453bc5d62d75b6b52c"
)

SDYFF_HF_REPO = "salv47/spherical-dyffusion"
SDYFF_HF_REVISION = "56fcb5d995717e91ba770f5ee41e8c3a8cfb00f6"
SDYFF_WEIGHTS_LICENCE = "Apache-2.0"
SDYFF_DATA_LICENCE = "CC-BY-4.0"

SDYFF_FORECASTER_CKPT_NAME = "forecaster-sfno-best-inference_avg_crps.ckpt"
SDYFF_FORECASTER_CKPT_REL_PATH = f"models/spherical-dyffusion/{SDYFF_FORECASTER_CKPT_NAME}"
SDYFF_FORECASTER_CKPT_SIZE = 3_370_273_818
SDYFF_FORECASTER_CKPT_SHA256 = (
    "cd2f836b5c2238201f65cb620cf885b283209d5d778bca13bea890df76694618"
)

SDYFF_INTERPOLATOR_CKPT_NAME = "interpolator-sfno-best-val_avg_crps.ckpt"
SDYFF_INTERPOLATOR_CKPT_REL_PATH = f"models/spherical-dyffusion/{SDYFF_INTERPOLATOR_CKPT_NAME}"
SDYFF_INTERPOLATOR_CKPT_SIZE = 3_370_550_315
SDYFF_INTERPOLATOR_CKPT_SHA256 = (
    "aecd9a59b893d750241054b8e7b162cd26c5304ded3eb180692b19bd0d7cf1db"
)

SDYFF_INTERPOLATOR_CFG_NAME = "interpolator_sfno_paper_v0_hydra_config.yaml"
SDYFF_INTERPOLATOR_CFG_REL_PATH = f"models/spherical-dyffusion/{SDYFF_INTERPOLATOR_CFG_NAME}"
SDYFF_INTERPOLATOR_CFG_SIZE = 5_265
SDYFF_INTERPOLATOR_CFG_SHA256 = (
    "f7535343996a7dbba3826775906d0026153e40b4f53111d1f1adc503bcf4f2df"
)

SDYFF_ZENODO_RECORD = "10791087"
SDYFF_DATA_REL_DIR = "inputs/spherical-dyffusion/fv3gfs-2021"
SDYFF_FORCING_NAME = "fv3gfs-repeating-climsst-1deg"
SDYFF_WINDOW: tuple[str, str] = ("2021-01-01T00:00:00", "2120-12-31T18:00:00")
SDYFF_START_WINDOW: tuple[str, str] = ("2021-01-01T00:00:00", "2021-12-31T00:00:00")

SDYFF_ZENODO_FILES: dict[str, dict[str, Any]] = {
    "2021010100.nc": {
        "size": 1_735_886_648,
        "md5": "e889455aa6023ace37a71fe23731dc1f",
        "sha256": "a7275a79ca038be393796c4cecd5ae1f1b2078b9410f5c340aa7d538d9ceb042",
    },
    "2021020100.nc": {
        "size": 1_567_925_000,
        "md5": "8827dd2ca796fccf719e10a6bdcbd291",
        "sha256": "a859ef1ed4966867772229a03211eae6dccd1a37a9fb6509e39e4688295f8d28",
    },
    "2021030100.nc": {
        "size": 1_735_886_648,
        "md5": "49ca3b82a7c9ca6a62959eed51bd8608",
        "sha256": "858c2c02c3a088f49323ea70f2f528bac81e2549416c315508b7b012aa592ae6",
    },
    "2021040100.nc": {
        "size": 1_679_899_432,
        "md5": "f086f500af85256fa264a13baa2e2401",
        "sha256": "1b379842913fd8570dee4dc05fe46ce6828ad4b5fae36c989faa941b3e0d8129",
    },
    "2021050100.nc": {
        "size": 1_735_886_648,
        "md5": "7bdb1e428842cdce1a8f93e20bdda3f3",
        "sha256": "0003b1ab78bc8300ca1dbc03491153aaf235fb2327b4b196c4195dad6e6a7350",
    },
    "2021060100.nc": {
        "size": 1_679_899_432,
        "md5": "09f578ea272fe3a2bad7d14dcb424348",
        "sha256": "496d638a0ba5e2499c306a25d28e7b13afdb8fdf717c805566829d5c1c6bd121",
    },
    "2021070100.nc": {
        "size": 1_735_886_648,
        "md5": "017fe77ca320db54029d902592069343",
        "sha256": "8ec44868bd2df28cdc6c8401b8066a8a984d22c1053d3bd9e8f1601d45f86df4",
    },
    "2021080100.nc": {
        "size": 1_735_886_648,
        "md5": "09ad3ab34600ac9f75da6a2a77374b5b",
        "sha256": "ec4d6cb5f39e3e1c7d15ea2a8dc035cd2c521c2c72ab36a451af2b96acff77bd",
    },
    "2021090100.nc": {
        "size": 1_679_899_432,
        "md5": "a47e120e408445b0738780295ed2fcb0",
        "sha256": "bfb12eac6e81515fd3c8dd592fe2258804dc5672fe34b211b1bcd0c1717649a3",
    },
    "2021100100.nc": {
        "size": 1_735_886_648,
        "md5": "64dfa8ce8d96221e5de13cf8c8dd04d1",
        "sha256": "06e0bbc032f19a3c5a348986ef360cb31fa277beb4aee7fd954c3680d3dd582b",
    },
    "2021110100.nc": {
        "size": 1_679_899_432,
        "md5": "4f14bafd59b6aa5c9d78450819434bf8",
        "sha256": "01e789c74c8159b8ddba2c7ed3419a9653d30068c2e237b921dbd2071384c188",
    },
    "2021120100.nc": {
        "size": 1_735_886_648,
        "md5": "11d2ac46dad4c6d859b03f9419a07dfb",
        "sha256": "736380bf11c8a80a8d20802c57f799e8542f15f74b0450d064786cb3338b92c6",
    },
}

# Exact 34 output channels from `forecaster-sfno-best-inference_avg_crps.ckpt` `datamodule_config.out_names`.
SDYFF_PREDICTED_VARS: tuple[str, ...] = (
    "PRESsfc",
    "surface_temperature",
    *(f"air_temperature_{k}" for k in range(8)),
    *(f"specific_total_water_{k}" for k in range(8)),
    *(f"eastward_wind_{k}" for k in range(8)),
    *(f"northward_wind_{k}" for k in range(8)),
)
SDYFF_DERIVED_VARS: tuple[str, ...] = ("total_water_path",)
SDYFF_PRESCRIBED_SURFACE_VAR = "surface_temperature_prescribed_ocean"
SDYFF_SUMMARY_TEMP_VAR = "air_temperature_7"

SDYFF_GRID_SHAPE: tuple[int, int] = (180, 360)
SDYFF_COARSEN_FACTOR = 4   # 4 x 6h steps = exact 1-day means -> exact 365-day calendar monthly means
SDYFF_MAX_CYCLE_YEARS = 100
SDYFF_STEPS_PER_YEAR = 1460
SDYFF_SECONDS_PER_YEAR = 180.0
SDYFF_TA7_BAND_K: tuple[float, float] = (180.0, 340.0)
EXIT_UNSTABLE = 5

DEFAULT_SDYFF_PYTHON = "/opt/aimip/venv-sdyff/bin/python"
SDYFF_NOMINAL_TIME_UNITS = "days since 0001-01-01 00:00:00"
SDYFF_OFFLINE_WANDB_ENTITY = "salv47"
SDYFF_CALENDAR_NOTE = (
    "repeating single-year SST cycle; year numbers are nominal cycle indices, not historical calendar years"
)

_NOLEAP_MONTH_DAYS: tuple[int, ...] = (31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31)

_SDYFF_BOOTSTRAP = (
    "import json, os, random, runpy, sys, types\n"
    "sys.dont_write_bytecode = True\n"
    "os.environ['PYTHONDONTWRITEBYTECODE'] = '1'\n"
    "os.environ['WANDB_MODE'] = 'disabled'\n"
    "os.environ['WANDB_DISABLED'] = 'true'\n"
    "os.environ['HF_HUB_OFFLINE'] = '1'\n"
    "os.environ['MPLBACKEND'] = 'Agg'\n"
    "code_dir, run_inference_py, cfg_path, seed_s, ipol_ckpt, ipol_cfg = (\n"
    "    sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4]), sys.argv[5], sys.argv[6]\n"
    ")\n"
    "os.environ['PL_GLOBAL_SEED'] = str(seed_s)\n"
    "import numpy as np, torch\n"
    "try:\n"
    "    import apex.normalization\n"
    "except ImportError:\n"
    "    _apex_pkg = types.ModuleType('apex')\n"
    "    _apex_pkg.__path__ = []\n"
    "    _apex_norm = types.ModuleType('apex.normalization')\n"
    "    _apex_norm.FusedLayerNorm = torch.nn.LayerNorm\n"
    "    _apex_pkg.normalization = _apex_norm\n"
    "    sys.modules['apex'] = _apex_pkg\n"
    "    sys.modules['apex.normalization'] = _apex_norm\n"
    "_pkg_root = os.environ.get('AIMIP_BOOTSTRAP_PKG_ROOT', '')\n"
    "if _pkg_root and _pkg_root not in sys.path:\n"
    "    sys.path.insert(0, _pkg_root)\n"
    "from aimip.sdyff import install_sdyff_offline_runtime_hooks, install_sdyff_verified_loaders\n"
    "install_sdyff_verified_loaders(\n"
    "    json.loads(os.environ['AIMIP_SDYFF_VERIFIED_DIGESTS']),\n"
    "    audit_log_path=os.environ.get('AIMIP_SDYFF_VERIFIED_LOG'),\n"
    ")\n"
    "random.seed(seed_s)\n"
    "np.random.seed(seed_s % (2**32))\n"
    "torch.manual_seed(seed_s)\n"
    "if torch.cuda.is_available():\n"
    "    torch.cuda.manual_seed_all(seed_s)\n"
    "try:\n"
    "    from pytorch_lightning import seed_everything\n"
    "    seed_everything(seed_s, workers=True)\n"
    "except ImportError:\n"
    "    pass\n"
    "if code_dir not in sys.path:\n"
    "    sys.path.insert(0, code_dir)\n"
    "os.chdir(code_dir)\n"
    "install_sdyff_offline_runtime_hooks(\n"
    "    code_dir, interpolator_ckpt=ipol_ckpt, interpolator_cfg=ipol_cfg\n"
    ")\n"
    "sys.argv = ['run_inference.py', cfg_path]\n"
    "runpy.run_path(run_inference_py, run_name='__main__')\n"
)


def _unlock_omegaconf_struct(node: Any) -> None:
    """Recursively clear `struct=True` on OmegaConf `DictConfig`/`ListConfig` nodes so overrides can add keys."""
    try:
        from omegaconf import DictConfig, ListConfig, OmegaConf
    except ImportError:
        return
    if isinstance(node, (DictConfig, ListConfig)):
        OmegaConf.set_struct(node, False)
        if isinstance(node, DictConfig):
            for k in list(node.keys()):
                try:
                    child = node.get(k)
                except Exception:
                    continue
                _unlock_omegaconf_struct(child)
        else:
            for item in node:
                _unlock_omegaconf_struct(item)
    elif isinstance(node, Mapping):
        for v in node.values():
            _unlock_omegaconf_struct(v)


def _validate_local_sdyff_asset(path: str, *, label: str) -> str:
    raw = str(path).strip()
    if raw.startswith(("hf:", "http:", "https:", "s3:", "gs:")):
        raise ValidationError(
            f"Refusing remote URI for {label} in offline Spherical DYffusion job: {raw!r}"
        )
    resolved = validate_safe_path(os.path.abspath(raw), label=label)
    if os.path.islink(resolved) or not os.path.isfile(resolved):
        raise ValidationError(f"Missing or non-regular local {label}: {resolved}")
    return resolved


def inject_sdyff_local_interpolator_overrides(
    state: MutableMapping[str, Any],
    overrides: Mapping[str, Any] | None,
    *,
    interpolator_ckpt: str,
    interpolator_cfg: str,
) -> dict[str, Any]:
    """Ensure `MultiStepStepper.from_state` receives local `$AIMIP_DATA` interpolator paths and never calls W&B.

    The published `forecaster-sfno-best-inference_avg_crps.ckpt` stores an `OmegaConf` `DictConfig`
    under `hyper_parameters["diffusion_config"]` carrying `interpolator_run_id` and lacking
    `interpolator_local_checkpoint_path` / `interpolator_local_config_path`. Upstream
    `reload_checkpoint_from_wandb` also evaluates `wandb_api.get_entity(entity)` (which opens a
    GraphQL connection via `wandb.api.default_entity` whenever `entity is None`) BEFORE checking
    `config_path` or `local_checkpoint_path`. Unlocking `DictConfig` struct mode and injecting
    both verified local paths plus a non-null offline `entity` eliminates every W&B/HF lookup.
    """
    ckpt_abs = _validate_local_sdyff_asset(
        interpolator_ckpt, label="Spherical DYffusion interpolator checkpoint"
    )
    cfg_abs = _validate_local_sdyff_asset(
        interpolator_cfg, label="Spherical DYffusion interpolator config"
    )

    hparams = state.get("hyper_parameters")
    _unlock_omegaconf_struct(hparams)

    merged: dict[str, Any] = dict(overrides) if overrides else {}
    raw_diff_ov = merged.get("diffusion_config")
    diff_ov: dict[str, Any] = dict(raw_diff_ov) if isinstance(raw_diff_ov, Mapping) else {}
    raw_wkw = diff_ov.get("interpolator_wandb_kwargs")
    wkw: dict[str, Any] = dict(raw_wkw) if isinstance(raw_wkw, Mapping) else {}

    offline_wkw = {
        **wkw,
        "entity": str(wkw.get("entity") or SDYFF_OFFLINE_WANDB_ENTITY),
        "project": str(wkw.get("project") or "Spherical-DYffusion"),
    }
    diff_ov["hack_for_imprecise_interpolation"] = bool(
        diff_ov.get("hack_for_imprecise_interpolation", True)
    )
    diff_ov["interpolator_local_checkpoint_path"] = ckpt_abs
    diff_ov["interpolator_local_config_path"] = cfg_abs
    diff_ov["interpolator_wandb_kwargs"] = offline_wkw
    merged["diffusion_config"] = diff_ov

    if hparams is not None and "diffusion_config" in hparams and hparams["diffusion_config"] is not None:
        dcfg = hparams["diffusion_config"]
        _unlock_omegaconf_struct(dcfg)
        dcfg["hack_for_imprecise_interpolation"] = diff_ov["hack_for_imprecise_interpolation"]
        dcfg["interpolator_local_checkpoint_path"] = ckpt_abs
        dcfg["interpolator_local_config_path"] = cfg_abs
        dcfg["interpolator_wandb_kwargs"] = dict(offline_wkw)

    return merged


def install_sdyff_offline_runtime_hooks(
    code_dir: str,
    *,
    interpolator_ckpt: str,
    interpolator_cfg: str,
) -> dict[str, bool]:
    """Install offline W&B guards, `MultiStepStepper.from_state` override injection, and ensemble writer fix."""
    ckpt_abs = _validate_local_sdyff_asset(
        interpolator_ckpt, label="Spherical DYffusion interpolator checkpoint"
    )
    cfg_abs = _validate_local_sdyff_asset(
        interpolator_cfg, label="Spherical DYffusion interpolator config"
    )
    code_abs = validate_safe_path(code_dir, label="Spherical DYffusion code_dir")

    os.environ["WANDB_MODE"] = "disabled"
    os.environ["WANDB_DISABLED"] = "true"
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["MPLBACKEND"] = "Agg"

    if code_abs not in sys.path:
        sys.path.insert(0, code_abs)

    hooks = {
        "wandb_client": False,
        "wandb_api": False,
        "from_state": False,
        "data_writer": False,
    }

    def _refuse_wandb_network(*_args: Any, **_kwargs: Any) -> Any:
        raise RuntimeError(
            "Weights & Biases network/API access is disabled in offline AIMIP jobs."
        )

    wandb_mod = sys.modules.get("wandb")
    if wandb_mod is None:
        try:
            wandb_mod = importlib.import_module("wandb")
        except ImportError:
            wandb_mod = None
    if wandb_mod is not None:
        wandb_mod.login = _refuse_wandb_network  # type: ignore[attr-defined]
        wandb_mod.Api = _refuse_wandb_network  # type: ignore[attr-defined]
        hooks["wandb_client"] = True

    wandb_api_py = os.path.join(code_abs, "src", "utilities", "wandb_api.py")
    if os.path.isfile(wandb_api_py):
        sdyff_wandb_api = importlib.import_module("src.utilities.wandb_api")
        sdyff_wandb_api._ENTITY = SDYFF_OFFLINE_WANDB_ENTITY  # type: ignore[attr-defined]
        sdyff_wandb_api.get_entity = (  # type: ignore[attr-defined]
            lambda entity=None: str(entity) if entity else SDYFF_OFFLINE_WANDB_ENTITY
        )
        sdyff_wandb_api.get_api = _refuse_wandb_network  # type: ignore[attr-defined]
        hooks["wandb_api"] = True

    stepper_ms_py = os.path.join(code_abs, "src", "ace_inference", "core", "stepper_multistep.py")
    if os.path.isfile(stepper_ms_py):
        stepper_ms = importlib.import_module("src.ace_inference.core.stepper_multistep")
        ms_cls = stepper_ms.MultiStepStepper
        orig_from_state_fn = getattr(
            ms_cls,
            "_aimip_orig_from_state_fn",
            ms_cls.__dict__["from_state"].__func__,
        )
        setattr(ms_cls, "_aimip_orig_from_state_fn", orig_from_state_fn)

        def _offline_from_state(
            cls: Any,
            state: MutableMapping[str, Any],
            load_optimizer: bool = True,
            overrides: Mapping[str, Any] | None = None,
        ) -> Any:
            merged_overrides = inject_sdyff_local_interpolator_overrides(
                state,
                overrides,
                interpolator_ckpt=ckpt_abs,
                interpolator_cfg=cfg_abs,
            )
            return orig_from_state_fn(
                cls, state, load_optimizer=load_optimizer, overrides=merged_overrides
            )

        ms_cls.from_state = classmethod(_offline_from_state)  # type: ignore[assignment]
        hooks["from_state"] = True

    dw_main_py = os.path.join(
        code_abs, "src", "ace_inference", "inference", "data_writer", "main.py"
    )
    if os.path.isfile(dw_main_py):
        dw_main = importlib.import_module("src.ace_inference.inference.data_writer.main")
        dw_cls = dw_main.DataWriter
        orig_dw_init = getattr(dw_cls, "_aimip_orig_init", dw_cls.__init__)
        orig_dw_append = getattr(dw_cls, "_aimip_orig_append_batch", dw_cls.append_batch)
        setattr(dw_cls, "_aimip_orig_init", orig_dw_init)
        setattr(dw_cls, "_aimip_orig_append_batch", orig_dw_append)

        def _ens_dw_init(
            self: Any,
            path: str,
            n_samples: int,
            n_timesteps: int,
            metadata: Any,
            coords: Any,
            enable_prediction_netcdfs: bool,
            enable_video_netcdfs: bool,
            time_coarsen: Any = None,
            n_ensemble_members: int = 1,
        ) -> None:
            eff_samples = int(n_samples) * max(1, int(n_ensemble_members))
            self._aimip_n_ensemble_members = int(n_ensemble_members)
            orig_dw_init(
                self,
                path=path,
                n_samples=eff_samples,
                n_timesteps=n_timesteps,
                metadata=metadata,
                coords=coords,
                enable_prediction_netcdfs=enable_prediction_netcdfs,
                enable_video_netcdfs=enable_video_netcdfs,
                time_coarsen=time_coarsen,
                n_ensemble_members=n_ensemble_members,
            )

        def _ens_dw_append_batch(
            self: Any,
            target: Mapping[str, Any],
            prediction: Mapping[str, Any],
            start_timestep: int,
            start_sample: int,
            batch_times: Any = None,
        ) -> None:
            n_ens = int(getattr(self, "_aimip_n_ensemble_members", 1))
            if n_ens > 1 and prediction:
                first_pred = next(iter(prediction.values()))
                if getattr(first_pred, "ndim", 0) == 5 and first_pred.shape[0] == n_ens:
                    prediction = {
                        k: v.reshape(v.shape[0] * v.shape[1], *v.shape[2:])
                        for k, v in prediction.items()
                    }
                    target = {
                        k: v.unsqueeze(0)
                        .expand(n_ens, *v.shape)
                        .reshape(n_ens * v.shape[0], *v.shape[1:])
                        for k, v in target.items()
                    }
                    start_sample = int(start_sample) * n_ens
            orig_dw_append(
                self,
                target=target,
                prediction=prediction,
                start_timestep=start_timestep,
                start_sample=start_sample,
                batch_times=batch_times,
            )

        dw_cls.__init__ = _ens_dw_init  # type: ignore[assignment]
        dw_cls.append_batch = _ens_dw_append_batch  # type: ignore[assignment]
        hooks["data_writer"] = True

    return hooks


def sdyff_python() -> str:
    """Return the validated executable path of the isolated Spherical DYffusion Python environment."""
    raw = os.environ.get("AIMIP_SDYFF_PYTHON")
    if not raw:
        home = os.environ.get("AIMIP_HOME") or os.environ.get("AIMIP_PLAYGROUND")
        cand = os.path.join(home.rstrip("/"), "venv-sdyff", "bin", "python") if home else ""
        if cand and (os.path.exists(cand) or not os.path.exists(DEFAULT_SDYFF_PYTHON)):
            raw = cand
        else:
            raw = DEFAULT_SDYFF_PYTHON
    path = validate_safe_path(raw, label="Spherical DYffusion python")
    if not os.path.isabs(path) or not os.access(path, os.X_OK):
        raise ConfigError(
            f"The Spherical DYffusion environment is missing ({path}). "
            "For native installs re-run `install/install-native.sh --with-sdyff` "
            "(or set $AIMIP_SDYFF_PYTHON)."
        )
    return path


class _SdyffRestrictedUnpickler(pickle.Unpickler):
    """Closed-allowlist unpickler allowing ONLY verified Spherical DYffusion `.ckpt` globals."""

    def find_class(self, module: str, name: str) -> Any:
        if module in ("builtins", "__builtin__"):
            if name == "dict":
                return builtins.dict
            if name == "list":
                return builtins.list
            if name in ("int", "long"):
                return builtins.int
        elif module == "collections":
            if name == "OrderedDict":
                return collections.OrderedDict
            if name == "defaultdict":
                return collections.defaultdict
        elif module == "typing" and name == "Any":
            return typing.Any
        elif module == "omegaconf.base" and name in ("ContainerMetadata", "Metadata"):
            from omegaconf import base as oc_base

            return getattr(oc_base, name)
        elif module == "omegaconf.dictconfig" and name == "DictConfig":
            from omegaconf.dictconfig import DictConfig

            return DictConfig
        elif module == "omegaconf.listconfig" and name == "ListConfig":
            from omegaconf.listconfig import ListConfig

            return ListConfig
        elif module == "omegaconf.nodes" and name == "AnyNode":
            from omegaconf.nodes import AnyNode

            return AnyNode
        elif module == "torch._utils":
            import torch._utils as torch_utils

            if name == "_rebuild_tensor_v2":
                return torch_utils._rebuild_tensor_v2
            if name == "_rebuild_parameter" and hasattr(torch_utils, "_rebuild_parameter"):
                return torch_utils._rebuild_parameter
        elif module in ("torch", "torch.storage") and name in (
            "FloatStorage",
            "IntStorage",
            "DoubleStorage",
        ):
            import torch

            if hasattr(torch, name):
                return getattr(torch, name)
        raise pickle.UnpicklingError(
            f"Refusing non-whitelisted Spherical DYffusion pickle global {module}.{name}"
        )


def make_sdyff_restricted_pickle_module() -> types.ModuleType:
    """Return a `pickle_module` for `torch.load` whose `Unpickler.find_class` allows ONLY Spherical DYffusion globals."""
    mod = types.ModuleType("_aimip_sdyff_restricted_pickle")
    mod.Unpickler = _SdyffRestrictedUnpickler  # type: ignore[attr-defined]
    mod.UnpicklingError = pickle.UnpicklingError  # type: ignore[attr-defined]
    mod.load = lambda fh, **kw: _SdyffRestrictedUnpickler(fh, **kw).load()  # type: ignore[attr-defined]
    mod.loads = lambda data, **kw: _SdyffRestrictedUnpickler(io.BytesIO(data), **kw).load()  # type: ignore[attr-defined]
    return mod


def _extract_sdyff_load_path(target: Any) -> str | None:
    if isinstance(target, (str, bytes, os.PathLike)):
        return os.fsdecode(target)
    if isinstance(target, zipfile.ZipExtFile):
        return None
    fs_path = getattr(target, "path", None)
    if isinstance(fs_path, (str, bytes, os.PathLike)):
        return os.fsdecode(fs_path)
    fh_name = getattr(target, "name", None)
    if isinstance(fh_name, (str, bytes, os.PathLike)):
        return os.fsdecode(fh_name)
    return None


def _append_sdyff_audit_record(audit_log_path: str, payload: Mapping[str, Any]) -> None:
    safe_log = validate_safe_path(audit_log_path, label="Spherical DYffusion verified-load audit log")
    if os.path.islink(safe_log):
        raise ValidationError(
            f"Refusing symlinked Spherical DYffusion verified-load audit log: {safe_log}"
        )
    flags = os.O_WRONLY | os.O_APPEND | getattr(os, "O_NOFOLLOW", 0)
    try:
        fd = os.open(safe_log, flags)
    except OSError as exc:
        raise ValidationError(
            f"Cannot open Spherical DYffusion verified-load audit log {safe_log}: {exc}"
        ) from exc
    try:
        st = os.fstat(fd)
        if not stat.S_ISREG(st.st_mode):
            raise ValidationError(
                f"Refusing non-regular Spherical DYffusion verified-load audit log: {safe_log}"
            )
        os.write(fd, (json.dumps(dict(payload)) + "\n").encode("utf-8"))
    finally:
        os.close(fd)


def _read_sdyff_audit_torch_realpaths(audit_log_path: str) -> set[str]:
    safe_log = validate_safe_path(audit_log_path, label="Spherical DYffusion verified-load audit log")
    if os.path.islink(safe_log):
        raise ValidationError(
            f"Refusing symlinked Spherical DYffusion verified-load audit log: {safe_log}"
        )
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    try:
        fd = os.open(safe_log, flags)
    except OSError as exc:
        raise ValidationError(
            f"Cannot read Spherical DYffusion verified-load audit log {safe_log}: {exc}"
        ) from exc
    fh = None
    try:
        st = os.fstat(fd)
        if not stat.S_ISREG(st.st_mode) or st.st_uid != os.geteuid():
            raise ValidationError(
                f"Refusing untrusted Spherical DYffusion verified-load audit log: {safe_log}"
            )
        fh = os.fdopen(fd, "rb")
        with fh:
            raw_lines = fh.read().decode("utf-8").splitlines()
    except BaseException:
        if fh is None:
            try:
                os.close(fd)
            except OSError:
                pass
        raise

    realpaths: set[str] = set()
    for line in raw_lines:
        if not line.strip():
            continue
        rec = json.loads(line)
        if (
            isinstance(rec, Mapping)
            and rec.get("kind") == "torch"
            and isinstance(rec.get("realpath"), str)
        ):
            realpaths.add(rec["realpath"])
    return realpaths


def install_sdyff_verified_loaders(
    expected_sha256_by_path: Mapping[str, str],
    *,
    audit_log_path: str | None = None,
    recorded_torch_realpaths: set[str] | None = None,
) -> Callable[[], None]:
    """Wrap `torch.load` and `OmegaConf.load` so checkpoints/configs load from `O_NOFOLLOW` SHA-256-verified descriptors."""
    import torch

    if not expected_sha256_by_path:
        raise ValidationError("Spherical DYffusion verified loader requires a non-empty SHA-256 map.")
    digest_map: dict[str, str] = {}
    for raw_p, raw_sha in expected_sha256_by_path.items():
        safe_p = validate_safe_path(str(raw_p), label="Spherical DYffusion verified artifact")
        if os.path.islink(safe_p):
            raise ValidationError(f"Refusing symlinked Spherical DYffusion artifact: {safe_p}")
        norm_sha = normalize_sha256(
            str(raw_sha),
            label=f"Spherical DYffusion {os.path.basename(safe_p)} SHA-256",
        )
        digest_map[os.path.abspath(safe_p)] = norm_sha
        digest_map[os.path.realpath(safe_p)] = norm_sha

    orig_load = getattr(torch, "_aimip_orig_torch_load", torch.load)
    orig_ser_load = getattr(torch.serialization, "_aimip_orig_torch_load", torch.serialization.load)
    setattr(torch, "_aimip_orig_torch_load", orig_load)
    setattr(torch.serialization, "_aimip_orig_torch_load", orig_ser_load)

    def _verified_sdyff_torch_load(f: Any, *args: Any, **kwargs: Any) -> Any:
        kwargs["weights_only"] = False
        kwargs["pickle_module"] = make_sdyff_restricted_pickle_module()
        cand_path = _extract_sdyff_load_path(f)
        if cand_path is not None:
            abs_p = os.path.abspath(cand_path)
            want_sha = digest_map.get(abs_p)
            if want_sha is None:
                raise ValidationError(
                    f"Refusing unverified Spherical DYffusion torch.load path: {abs_p}"
                )
            with open_verified_hash_stream(
                abs_p,
                want_sha,
                label=f"Spherical DYffusion {os.path.basename(abs_p)}",
            ) as stream:
                loaded_obj = orig_load(stream, *args, **kwargs)
            real_p = os.path.realpath(abs_p)
            if recorded_torch_realpaths is not None:
                recorded_torch_realpaths.add(real_p)
            if audit_log_path:
                _append_sdyff_audit_record(
                    audit_log_path,
                    {"kind": "torch", "abspath": abs_p, "realpath": real_p},
                )
            return loaded_obj
        return orig_load(f, *args, **kwargs)

    torch.load = _verified_sdyff_torch_load  # type: ignore[assignment]
    torch.serialization.load = _verified_sdyff_torch_load  # type: ignore[assignment]

    oc_cls: Any = None
    orig_oc_load: Any = None
    try:
        from omegaconf import OmegaConf

        oc_cls = OmegaConf
        orig_oc_load = getattr(OmegaConf, "_aimip_orig_load", OmegaConf.load)
        setattr(OmegaConf, "_aimip_orig_load", orig_oc_load)

        def _verified_omegaconf_load(file_: Any) -> Any:
            cand_path = _extract_sdyff_load_path(file_)
            if cand_path is not None:
                abs_p = os.path.abspath(cand_path)
                want_sha = digest_map.get(abs_p)
                if want_sha is None:
                    raise ValidationError(
                        f"Refusing unverified Spherical DYffusion OmegaConf.load path: {abs_p}"
                    )
                with open_verified_hash_stream(
                    abs_p,
                    want_sha,
                    label=f"Spherical DYffusion {os.path.basename(abs_p)}",
                ) as stream:
                    raw_text = stream.read().decode("utf-8")
                loaded_cfg = orig_oc_load(io.StringIO(raw_text))
                if audit_log_path:
                    _append_sdyff_audit_record(
                        audit_log_path,
                        {
                            "kind": "omegaconf",
                            "abspath": abs_p,
                            "realpath": os.path.realpath(abs_p),
                        },
                    )
                return loaded_cfg
            return orig_oc_load(file_)

        OmegaConf.load = staticmethod(_verified_omegaconf_load)  # type: ignore[method-assign]
    except ImportError:
        pass

    def _restore() -> None:
        torch.load = orig_load  # type: ignore[assignment]
        torch.serialization.load = orig_ser_load  # type: ignore[assignment]
        if oc_cls is not None and orig_oc_load is not None:
            oc_cls.load = orig_oc_load

    return _restore


@contextlib.contextmanager
def verified_sdyff_loaders(
    expected_sha256_by_path: Mapping[str, str],
) -> Iterator[set[str]]:
    """Context-managed `install_sdyff_verified_loaders` for tests and in-process checks."""
    loaded_set: set[str] = set()
    restore = install_sdyff_verified_loaders(
        expected_sha256_by_path,
        recorded_torch_realpaths=loaded_set,
    )
    try:
        yield loaded_set
    finally:
        restore()


def verify_sdyff_checkpoint(
    path: str,
    expected_sha256: str,
    *,
    label: str = "Spherical DYffusion checkpoint",
) -> dict[str, Any]:
    """Verify *path* SHA-256 on an `O_NOFOLLOW` fd and load via `make_sdyff_restricted_pickle_module()`."""
    import torch

    restricted_pickle = make_sdyff_restricted_pickle_module()
    with open_verified_hash_stream(path, expected_sha256, label=label) as stream:
        try:
            ckpt = torch.load(
                stream,
                map_location="cpu",
                weights_only=False,
                pickle_module=restricted_pickle,
            )
        except pickle.UnpicklingError as exc:
            raise ValidationError(
                f"Blocked unsafe or corrupted {label} ({path}): {exc}"
            ) from exc

    if not isinstance(ckpt, Mapping) or not isinstance(ckpt.get("state_dict"), Mapping):
        raise ValidationError(
            f"Expected Lightning checkpoint dict with 'state_dict' in {label} ({path})."
        )
    n_tensors = len(ckpt["state_dict"])
    if n_tensors == 0:
        raise ValidationError(f"Empty state_dict in {label} ({path}).")
    epoch = ckpt.get("epoch")
    del ckpt
    return {"n_tensors": n_tensors, "epoch": epoch}


def resolve_sdyff_code_dir(dest_dir: str) -> tuple[str, bool]:
    """Return `(code_dir, is_override)` from `$AIMIP_SDYFF_CODE` or `<dest_dir>/code/spherical-dyffusion`."""
    override = (os.environ.get("AIMIP_SDYFF_CODE") or "").strip()
    if override:
        return validate_safe_path(override, label="AIMIP_SDYFF_CODE"), True
    base = validate_safe_path(dest_dir, label="AIMIP_DATA")
    return os.path.join(base, SDYFF_CODE_REL_DIR), False


def verify_sdyff_code_dir(
    dest_dir: str,
    *,
    expected_tree_sha256: str | None = None,
    expected_tree_files: int | None = None,
) -> dict[str, Any]:
    """Verify `<dest_dir>/code/spherical-dyffusion` (or `$AIMIP_SDYFF_CODE`) against the pinned tree digest."""
    eff_tree_sha = SDYFF_TREE_SHA256 if expected_tree_sha256 is None else expected_tree_sha256
    eff_tree_files = (
        SDYFF_TREE_FILES if expected_tree_files is None else int(expected_tree_files)
    )
    code_dir, is_override = resolve_sdyff_code_dir(dest_dir)
    if not os.path.isdir(code_dir) or os.path.islink(code_dir):
        raise ValidationError(
            f"Lock verification failed for {SDYFF_CODE_REL_DIR}: directory is missing ({code_dir})."
        )
    tree_sha, n_files = compute_sha256sum_tree_digest(
        code_dir, label="Spherical DYffusion code directory"
    )
    if not is_override and (
        n_files != eff_tree_files or tree_sha.lower() != eff_tree_sha.lower()
    ):
        raise ValidationError(
            f"Lock verification failed for {SDYFF_CODE_REL_DIR}: tree digest/count mismatch "
            f"(expected {eff_tree_files} files / sha256={eff_tree_sha.lower()}, "
            f"got {n_files} files / sha256={tree_sha})."
        )
    return {
        "code_dir": code_dir,
        "tree_sha256": tree_sha,
        "files": n_files,
        "override": is_override,
    }


def extract_and_verify_sdyff_archive(
    archive_path: str,
    target_dir: str,
    *,
    expected_archive_sha256: str | None = None,
    expected_archive_size: int | None = None,
    expected_tree_sha256: str | None = None,
    expected_tree_files: int | None = None,
) -> dict[str, Any]:
    """Verify `Rose-STL-Lab/spherical-dyffusion` archive SHA-256 + `LC_ALL=C` tree SHA-256 before publishing."""
    eff_archive_sha = (
        SDYFF_GITHUB_ARCHIVE_SHA256
        if expected_archive_sha256 is None
        else expected_archive_sha256
    )
    eff_archive_size = (
        SDYFF_GITHUB_ARCHIVE_SIZE
        if expected_archive_size is None
        else int(expected_archive_size)
    )
    eff_tree_sha = SDYFF_TREE_SHA256 if expected_tree_sha256 is None else expected_tree_sha256
    eff_tree_files = (
        SDYFF_TREE_FILES if expected_tree_files is None else int(expected_tree_files)
    )

    validated_archive = validate_safe_path(
        archive_path, label="Spherical DYffusion source archive"
    )
    validated_target = validate_safe_path(
        target_dir, label="Spherical DYffusion target directory"
    )
    if not os.path.isfile(validated_archive) or os.path.islink(validated_archive):
        raise ValidationError(
            f"Spherical DYffusion archive not found: {validated_archive}"
        )

    actual_size = os.path.getsize(validated_archive)
    if eff_archive_size > 0 and actual_size != eff_archive_size:
        raise ValidationError(
            f"Spherical DYffusion archive size mismatch for {validated_archive}: "
            f"expected {eff_archive_size} B, got {actual_size} B."
        )

    digest = hashlib.sha256()
    with open(validated_archive, "rb") as fh:
        while True:
            chunk = fh.read(4 * 1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    actual_sha = digest.hexdigest()
    if eff_archive_sha and actual_sha.lower() != eff_archive_sha.lower():
        raise ValidationError(
            f"Spherical DYffusion archive SHA-256 mismatch for {validated_archive}: "
            f"expected {eff_archive_sha.lower()}, got {actual_sha}."
        )

    parent_dir = os.path.dirname(validated_target) or "/"
    os.makedirs(parent_dir, exist_ok=True)
    strip_group_other_write(parent_dir)
    stage_dir = os.path.join(parent_dir, f".sdyff_stage_{uuid.uuid4().hex}")
    old_backup: str | None = None

    try:
        os.makedirs(stage_dir, mode=0o755, exist_ok=False)
        strip_group_other_write(stage_dir)

        try:
            tf_ctx = tarfile.open(validated_archive, mode="r:*")
        except (tarfile.TarError, OSError) as exc:
            raise ValidationError(
                f"Invalid Spherical DYffusion archive {validated_archive}: {exc}"
            ) from exc

        with tf_ctx as tf:
            members = tf.getmembers()
            if not members:
                raise ValidationError(
                    f"Spherical DYffusion archive {validated_archive} is empty."
                )

            parsed: list[tuple[tarfile.TarInfo, list[str]]] = []
            for m in members:
                if not (m.isdir() or m.isfile()):
                    raise ValidationError(
                        f"Refusing unsupported tar entry type (symlinks/devices disallowed) in Spherical DYffusion archive: {m.name!r}"
                    )
                if (
                    not m.name
                    or m.name.startswith("/")
                    or "\x00" in m.name
                    or "\\" in m.name
                ):
                    raise ValidationError(
                        f"Refusing unsafe path in Spherical DYffusion archive: {m.name!r}"
                    )
                stripped = m.name.rstrip("/")
                if not stripped:
                    raise ValidationError("Refusing empty root tar entry.")
                parts = stripped.split("/")
                if any(seg in ("", ".", "..") for seg in parts):
                    raise ValidationError(
                        f"Refusing traversal segment in Spherical DYffusion archive: {m.name!r}"
                    )
                parsed.append((m, parts))

            first_top = parsed[0][1][0]
            strip_top = all(p[0] == first_top for _, p in parsed) and any(
                len(p) > 1 for _, p in parsed
            )
            for m, parts in parsed:
                rel_parts = parts[1:] if strip_top else parts
                if not rel_parts:
                    continue
                rel_p = "/".join(rel_parts)
                dst_p = os.path.join(stage_dir, *rel_parts)
                if m.isdir():
                    os.makedirs(dst_p, mode=0o755, exist_ok=True)
                    strip_group_other_write(dst_p)
                else:
                    os.makedirs(os.path.dirname(dst_p), mode=0o755, exist_ok=True)
                    strip_group_other_write(os.path.dirname(dst_p))
                    src_f = tf.extractfile(m)
                    if src_f is None:
                        raise ValidationError(
                            f"Failed to read {rel_p!r} from Spherical DYffusion archive."
                        )
                    with src_f, open(dst_p, "wb") as out_f:
                        shutil.copyfileobj(src_f, out_f, length=1024 * 1024)
                    os.chmod(dst_p, 0o755 if (m.mode & 0o111) else 0o644)

        tree_sha, n_files = compute_sha256sum_tree_digest(
            stage_dir, label="staged Spherical DYffusion tree"
        )
        if n_files != eff_tree_files or tree_sha.lower() != eff_tree_sha.lower():
            raise ValidationError(
                f"Spherical DYffusion extracted tree mismatch: expected {eff_tree_files} files / "
                f"sha256={eff_tree_sha.lower()}, got {n_files} files / sha256={tree_sha}."
            )

        if os.path.lexists(validated_target):
            if os.path.islink(validated_target) or not os.path.isdir(validated_target):
                raise ValidationError(
                    f"Refusing to overwrite non-directory at {validated_target}."
                )
            old_backup = f"{validated_target}.old_{uuid.uuid4().hex}"
            os.replace(validated_target, old_backup)

        os.replace(stage_dir, validated_target)
        if old_backup and os.path.exists(old_backup):
            shutil.rmtree(old_backup, ignore_errors=True)

        return {
            "code_dir": validated_target,
            "archive_size": actual_size,
            "archive_sha256": actual_sha,
            "tree_sha256": tree_sha,
            "files": n_files,
        }
    except Exception:
        if old_backup and os.path.exists(old_backup) and not os.path.exists(validated_target):
            try:
                os.replace(old_backup, validated_target)
            except OSError:
                pass
        if os.path.exists(stage_dir):
            shutil.rmtree(stage_dir, ignore_errors=True)
        raise


_SDYFF_APEX_INIT_SRC = (
    '"""Recorded AIMIP compatibility shim (`venv-sdyff` only).\n\n'
    "apex FusedLayerNorm replaced by torch.nn.LayerNorm; numerically equivalent up to float rounding.\n"
    '"""\n'
    "from . import normalization\n\n"
    '__all__ = ["normalization"]\n'
)

_SDYFF_APEX_NORM_SRC = (
    '"""Recorded AIMIP compatibility shim (`venv-sdyff` only).\n\n'
    "apex FusedLayerNorm replaced by torch.nn.LayerNorm; numerically equivalent up to float rounding.\n"
    "Satisfies `modulus.models.sfno.sfnonet`'s unguarded top-level `from apex.normalization import FusedLayerNorm`.\n"
    "Both `modulus.models.sfno.sfnonet` and `spherical-dyffusion`'s `src.models.sfno.sfnonet` already\n"
    "instantiate `torch.nn.LayerNorm` for `normalization_layer='layer_norm'`, and the published\n"
    "`salv47/spherical-dyffusion` checkpoints configure `normalization_layer='instance_norm'`\n"
    "(`torch.nn.InstanceNorm2d`).\n"
    '"""\n'
    "from torch.nn import LayerNorm as FusedLayerNorm\n\n"
    '__all__ = ["FusedLayerNorm"]\n'
)


def install_sdyff_apex_shim(
    venv_dir: str,
    *,
    site_packages_dir: str | None = None,
) -> dict[str, str]:
    """Install a recorded `apex.normalization.FusedLayerNorm = torch.nn.LayerNorm` shim inside `venv-sdyff` only."""
    import aimip as _aimip_pkg

    safe_venv = validate_safe_path(venv_dir, label="venv-sdyff directory")
    real_venv = os.path.realpath(safe_venv)
    if site_packages_dir:
        safe_site = validate_safe_path(site_packages_dir, label="venv-sdyff site-packages")
    else:
        inferred_site = os.path.dirname(os.path.dirname(os.path.realpath(_aimip_pkg.__file__)))
        safe_site = validate_safe_path(inferred_site, label="venv-sdyff site-packages")
    real_site = os.path.realpath(safe_site)
    if not real_site.startswith(real_venv + os.sep):
        raise ValidationError(
            f"venv-sdyff site-packages ({real_site}) does not reside inside {real_venv}"
        )
    if not os.path.isdir(safe_site) or os.path.islink(safe_site):
        raise ValidationError(
            f"venv-sdyff site-packages directory is missing or symlinked: {safe_site}"
        )

    apex_dir = os.path.join(safe_site, "apex")
    if os.path.islink(apex_dir):
        raise ValidationError(f"Refusing symlinked apex directory: {apex_dir}")
    makedirs_no_group_other_write(apex_dir)

    written: dict[str, str] = {"apex_dir": apex_dir}
    for fname, content in (
        ("__init__.py", _SDYFF_APEX_INIT_SRC),
        ("normalization.py", _SDYFF_APEX_NORM_SRC),
    ):
        dst = os.path.join(apex_dir, fname)
        tmp = f"{dst}.tmp-{os.getpid()}"
        with open(tmp, "w", encoding="utf-8") as fh:
            fh.write(content)
        strip_group_other_write(tmp)
        os.replace(tmp, dst)
        written[fname] = dst
    return written


def sdyff_plan_schedule(start_iso: str, steps: int) -> dict[str, Any]:
    """Validate `spherical-dyffusion` `--start` and `--steps`, returning exact cycle schedule & complete months.

    - `--start` must be `2021-MM-DDT00:00:00` inside the 365-day FV3GFS reference cycle (`2021`).
    - `--steps` must be a multiple of `4` (`coarsen_factor = 4` -> exact daily means) and <= `100`
      cycle years (`146,000` steps), covering at least `1` complete 365-day calendar month.
    - Output month labels use nominal cycle years (`0001-MM`, `0002-MM`, ...), NEVER `1995..` or `2021..`.
    """
    try:
        dt = datetime.datetime.strptime(start_iso.strip(), "%Y-%m-%dT%H:%M:%S")
    except ValueError as exc:
        raise ConfigError(
            f"Invalid --start {start_iso!r} for spherical-dyffusion (expected YYYY-MM-DDTHH:MM:SS).",
            field="start",
        ) from exc

    if dt.year != 2021 or dt.hour != 0 or dt.minute != 0 or dt.second != 0:
        raise ConfigError(
            f"spherical-dyffusion emulates the repeating 365-day FV3GFS reference year (labelled 2021 in the "
            f"Zenodo 10791087 files) and requires --start at 2021-MM-DDT00:00:00 so daily coarsening "
            f"(coarsen_factor=4) aligns with calendar days (got {start_iso}; e.g. --start 2021-01-01T00:00:00).",
            field="start",
        )

    if not isinstance(steps, int) or isinstance(steps, bool) or steps < 4:
        raise ConfigError(
            "spherical-dyffusion --steps must be an integer >= 4.",
            field="steps",
        )
    if steps % SDYFF_COARSEN_FACTOR != 0:
        raise ConfigError(
            f"spherical-dyffusion uses exact daily coarsening (coarsen_factor={SDYFF_COARSEN_FACTOR} x 6h steps); "
            f"--steps must be a multiple of {SDYFF_COARSEN_FACTOR} (got {steps}).",
            field="steps",
        )
    max_steps = SDYFF_MAX_CYCLE_YEARS * SDYFF_STEPS_PER_YEAR
    if steps > max_steps:
        raise ConfigError(
            f"spherical-dyffusion supports at most {SDYFF_MAX_CYCLE_YEARS} cycle years ({max_steps} steps; got {steps}).",
            field="steps",
        )

    start_doy = sum(_NOLEAP_MONTH_DAYS[: dt.month - 1]) + (dt.day - 1)
    start_index = start_doy * 4
    n_days = steps // SDYFF_COARSEN_FACTOR
    n_repeats = max(1, (start_index + steps + 1 + SDYFF_STEPS_PER_YEAR - 1) // SDYFF_STEPS_PER_YEAR)
    forward_steps_in_memory = max(
        d for d in (40, 20, 16, 12, 8, 4) if steps % d == 0
    )

    complete_months: list[dict[str, Any]] = []
    cur_year = 1
    cur_month = dt.month
    day_cursor = 0

    if dt.day > 1:
        rem_in_first = _NOLEAP_MONTH_DAYS[cur_month - 1] - (dt.day - 1)
        day_cursor += rem_in_first
        cur_month += 1
        if cur_month > 12:
            cur_month = 1
            cur_year += 1

    while True:
        m_days = _NOLEAP_MONTH_DAYS[cur_month - 1]
        if day_cursor + m_days > n_days:
            break
        complete_months.append(
            {
                "cycle_year": cur_year,
                "month": cur_month,
                "label": f"{cur_year:04d}-{cur_month:02d}",
                "day_start": day_cursor,
                "day_end": day_cursor + m_days,
                "days": m_days,
                "steps_6h": m_days * SDYFF_COARSEN_FACTOR,
            }
        )
        day_cursor += m_days
        cur_month += 1
        if cur_month > 12:
            cur_month = 1
            cur_year += 1

    if not complete_months:
        min_needed_days = (
            _NOLEAP_MONTH_DAYS[dt.month - 1]
            if dt.day == 1
            else (_NOLEAP_MONTH_DAYS[dt.month - 1] - (dt.day - 1) + _NOLEAP_MONTH_DAYS[dt.month % 12])
        )
        raise ConfigError(
            f"spherical-dyffusion writes exact means over complete 365-day calendar months only: "
            f"--steps {steps} ({n_days} days from {start_iso}) covers 0 complete calendar months "
            f"(pass at least --steps {min_needed_days * 4}, e.g. --steps 240 for 2 complete cycle months).",
            field="steps",
        )

    return {
        "start_index": start_index,
        "n_days": n_days,
        "n_repeats": n_repeats,
        "forward_steps_in_memory": forward_steps_in_memory,
        "coarsen_factor": SDYFF_COARSEN_FACTOR,
        "complete_months": complete_months,
    }


def render_sdyff_runtime_config(
    *,
    forecaster_ckpt: str,
    interpolator_ckpt: str,
    interpolator_cfg: str,
    stats_dir: str,
    data_dir: str,
    experiment_dir: str,
    start_iso: str,
    steps: int,
    members: int,
    seed: int,
    dst_config_path: str,
) -> dict[str, Any]:
    """Render offline `sdyff_runtime_config.yaml` (`coarsen_factor=4`, `log_to_wandb=False`, explicit `seed`)."""
    import yaml

    sched = sdyff_plan_schedule(start_iso, steps)
    for label, p_val in (
        ("forecaster checkpoint", forecaster_ckpt),
        ("interpolator checkpoint", interpolator_ckpt),
        ("interpolator config", interpolator_cfg),
    ):
        if str(p_val).strip().startswith(("hf:", "http:", "https:", "s3:", "gs:")):
            raise ValidationError(
                f"Refusing remote URI for Spherical DYffusion {label}: {p_val!r}"
            )
    forecaster_abs = os.path.abspath(forecaster_ckpt)
    interpolator_ckpt_abs = os.path.abspath(interpolator_ckpt)
    interpolator_cfg_abs = os.path.abspath(interpolator_cfg)
    stats_norm = stats_dir.rstrip("/") + "/"
    cfg: dict[str, Any] = {
        "checkpoint_path": forecaster_abs,
        "data_writer": {
            "save_prediction_files": True,
            "time_coarsen": {
                "coarsen_factor": SDYFF_COARSEN_FACTOR,
            },
        },
        "experiment_dir": experiment_dir,
        "forward_steps_in_memory": int(sched["forward_steps_in_memory"]),
        "log_video": False,
        "log_zonal_mean_images": False,
        "logging": {
            "entity": None,
            "log_to_file": True,
            "log_to_screen": True,
            "log_to_wandb": False,
            "project": "Spherical-DYffusion-inference",
        },
        "n_forward_steps": int(steps),
        "n_ensemble_members": int(members),
        "overrides": {
            "seed": int(seed),
            "datamodule_config": {
                "data_dir_stats": stats_norm,
            },
            "diffusion_config": {
                "hack_for_imprecise_interpolation": True,
                "interpolator_local_checkpoint_path": interpolator_ckpt_abs,
                "interpolator_local_config_path": interpolator_cfg_abs,
                "interpolator_wandb_kwargs": {
                    "entity": SDYFF_OFFLINE_WANDB_ENTITY,
                    "project": "Spherical-DYffusion",
                },
            },
        },
        "validation_loader": {
            "dataset": {
                "data_path": data_dir,
                "n_repeats": int(sched["n_repeats"]),
            },
            "num_data_workers": 4,
            "start_indices": {
                "first": int(sched["start_index"]),
                "n_initial_conditions": 1,
            },
        },
    }

    os.makedirs(os.path.dirname(dst_config_path) or ".", exist_ok=True)
    tmp_cfg = f"{dst_config_path}.tmp-{os.getpid()}"
    with open(tmp_cfg, "w", encoding="utf-8") as fh:
        yaml.safe_dump(cfg, fh, sort_keys=False)
    strip_group_other_write(tmp_cfg)
    os.replace(tmp_cfg, dst_config_path)
    return cfg


def _sdyff_var_meta(var_name: str) -> dict[str, str]:
    if var_name == SDYFF_PRESCRIBED_SURFACE_VAR:
        return {
            "units": "K",
            "long_name": "Surface temperature (prescribed over ocean where ocean_fraction == 1, predicted over land)",
        }
    if var_name == SDYFF_SUMMARY_TEMP_VAR:
        return {
            "units": "K",
            "long_name": "Lowest-model-layer air temperature (predicted)",
        }
    if var_name.startswith("air_temperature_"):
        lvl = var_name.rsplit("_", 1)[-1]
        return {"units": "K", "long_name": f"Model-layer {lvl} air temperature (predicted)"}
    if var_name.startswith("specific_total_water_"):
        lvl = var_name.rsplit("_", 1)[-1]
        return {"units": "kg/kg", "long_name": f"Model-layer {lvl} specific total water (predicted)"}
    if var_name.startswith("eastward_wind_"):
        lvl = var_name.rsplit("_", 1)[-1]
        return {"units": "m/s", "long_name": f"Model-layer {lvl} eastward wind (predicted)"}
    if var_name.startswith("northward_wind_"):
        lvl = var_name.rsplit("_", 1)[-1]
        return {"units": "m/s", "long_name": f"Model-layer {lvl} northward wind (predicted)"}
    if var_name == "PRESsfc":
        return {"units": "Pa", "long_name": "Surface pressure (predicted)"}
    if var_name == "total_water_path":
        return {"units": "kg/m**2", "long_name": "Column-integrated total water path (derived from predicted water layers)"}
    return {"units": ""}


def consolidate_sdyff_daily_to_monthly(
    autoregressive_nc_path: str,
    *,
    start_iso: str,
    steps: int,
    members: int,
    seed: int,
    out_dir: str,
) -> dict[str, Any]:
    """Convert daily-coarsened `autoregressive_predictions.nc` into exact nominal-cycle `monthly_mean_predictions.nc`."""
    import netCDF4
    import numpy as np
    import xarray as xr

    validated_ar = validate_safe_path(
        autoregressive_nc_path, label="Spherical DYffusion autoregressive_predictions.nc"
    )
    validated_out = validate_safe_path(out_dir, label="Spherical DYffusion out_dir")
    if not os.path.isfile(validated_ar) or os.path.islink(validated_ar):
        raise ValidationError(
            f"Spherical DYffusion output file missing: {validated_ar}"
        )

    sched = sdyff_plan_schedule(start_iso, steps)
    expected_days = int(sched["n_days"])
    complete_months = list(sched["complete_months"])
    n_months = len(complete_months)

    with xr.open_dataset(validated_ar, engine="netcdf4", decode_times=False) as ds_in:
        if "source" in ds_in.dims or "source" in ds_in.coords:
            sources = [
                (s.decode("utf-8") if isinstance(s, (bytes, bytearray)) else str(s))
                for s in ds_in["source"].values
            ]
            if "prediction" not in sources:
                raise ValidationError(
                    f"autoregressive_predictions.nc missing 'prediction' source (got {sources!r})."
                )
            pred_ds = ds_in.isel(source=sources.index("prediction"))
        else:
            pred_ds = ds_in

        missing_pred = [v for v in SDYFF_PREDICTED_VARS if v not in pred_ds.data_vars]
        if missing_pred:
            raise ValidationError(
                f"autoregressive_predictions.nc is missing {len(missing_pred)} of 34 predicted variables "
                f"(first missing: {', '.join(missing_pred[:6])})."
            )

        lat_coord = next((c for c in ("lat", "grid_yt", "latitude") if c in pred_ds.coords or c in pred_ds.dims), None)
        lon_coord = next((c for c in ("lon", "grid_xt", "longitude") if c in pred_ds.coords or c in pred_ds.dims), None)
        if lat_coord is None or lon_coord is None:
            raise ValidationError(
                f"autoregressive_predictions.nc missing lat/lon coordinates (coords={list(pred_ds.coords)})."
            )
        lat_vals = np.asarray(pred_ds[lat_coord].values, dtype=np.float64)
        lon_vals = np.asarray(pred_ds[lon_coord].values, dtype=np.float64)
        if (int(lat_vals.size), int(lon_vals.size)) != SDYFF_GRID_SHAPE:
            raise ValidationError(
                f"Expected 1 deg {SDYFF_GRID_SHAPE} grid in autoregressive_predictions.nc, "
                f"got {(int(lat_vals.size), int(lon_vals.size))}."
            )

        t_dim = "timestep" if "timestep" in pred_ds.dims else "time"
        raw_t_len = int(pred_ds.sizes[t_dim])
        if raw_t_len == expected_days + 1:
            t_slice = slice(1, None)   # timestep 0 in fme/sdyff writer is the initial state
        elif raw_t_len == expected_days:
            t_slice = slice(0, None)
        else:
            raise ValidationError(
                f"Unexpected {t_dim!r} length {raw_t_len} in {validated_ar} "
                f"(expected {expected_days} daily means or {expected_days + 1} with initial condition)."
            )

        src_vars = list(SDYFF_PREDICTED_VARS) + [
            v for v in SDYFF_DERIVED_VARS if v in pred_ds.data_vars
        ]

        monthly_arrays: dict[str, Any] = {}
        for raw_name in src_vars:
            da = pred_ds[raw_name].isel({t_dim: t_slice})
            if "sample" in da.dims:
                da = da.transpose("sample", t_dim, lat_coord, lon_coord)
                arr_daily = np.asarray(da.values, dtype=np.float64)
            else:
                da = da.transpose(t_dim, lat_coord, lon_coord)
                arr_daily = np.asarray(da.values, dtype=np.float64)[None, ...]

            if arr_daily.shape[0] != members:
                raise ValidationError(
                    f"Expected {members} sample member(s) for {raw_name!r}, got {arr_daily.shape[0]}."
                )
            if not np.isfinite(arr_daily).all():
                print(
                    f"[aimip.sdyff] UNSTABLE: non-finite daily values in {raw_name}",
                    flush=True,
                )
                raise SystemExit(EXIT_UNSTABLE)

            m_out = np.empty(
                (members, n_months, int(lat_vals.size), int(lon_vals.size)),
                dtype=np.float32,
            )
            for m_i, m_spec in enumerate(complete_months):
                d0, d1 = int(m_spec["day_start"]), int(m_spec["day_end"])
                m_out[:, m_i, :, :] = np.mean(arr_daily[:, d0:d1, :, :], axis=1).astype(np.float32)

            out_name = (
                SDYFF_PRESCRIBED_SURFACE_VAR
                if raw_name == "surface_temperature"
                else raw_name
            )
            monthly_arrays[out_name] = m_out

    # Stability check on predicted lowest-layer air temperature `air_temperature_7`.
    ta7 = monthly_arrays[SDYFF_SUMMARY_TEMP_VAR]
    w_lat = np.maximum(0.0, np.cos(np.deg2rad(lat_vals)))[:, None]
    w_2d = np.broadcast_to(w_lat, SDYFF_GRID_SHAPE)
    for s_i in range(members):
        for m_i, m_spec in enumerate(complete_months):
            g_ta7 = float(np.sum(ta7[s_i, m_i] * w_2d) / np.sum(w_2d))
            if not (SDYFF_TA7_BAND_K[0] <= g_ta7 <= SDYFF_TA7_BAND_K[1]):
                print(
                    f"[aimip.sdyff] UNSTABLE member={s_i} month={m_spec['label']}: "
                    f"global-mean {SDYFF_SUMMARY_TEMP_VAR}={g_ta7:.2f} K outside "
                    f"[{SDYFF_TA7_BAND_K[0]:.0f}, {SDYFF_TA7_BAND_K[1]:.0f}] K",
                    flush=True,
                )
                raise SystemExit(EXIT_UNSTABLE)

    out_nc = os.path.join(validated_out, "monthly_mean_predictions.nc")
    tmp_nc = f"{out_nc}.part-{os.getpid()}"
    try:
        with netCDF4.Dataset(tmp_nc, "w", format="NETCDF4") as ds_out:
            ds_out.createDimension("sample", int(members))
            ds_out.createDimension("time", n_months)
            ds_out.createDimension("lat", int(lat_vals.size))
            ds_out.createDimension("lon", int(lon_vals.size))

            v_lat = ds_out.createVariable("lat", "f8", ("lat",))
            v_lat[:] = lat_vals
            v_lat.units = "degrees_north"
            v_lat.standard_name = "latitude"

            v_lon = ds_out.createVariable("lon", "f8", ("lon",))
            v_lon[:] = lon_vals
            v_lon.units = "degrees_east"
            v_lon.standard_name = "longitude"

            v_time = ds_out.createVariable("time", "f8", ("time",))
            v_time.units = SDYFF_NOMINAL_TIME_UNITS
            v_time.calendar = "noleap"
            v_time.long_name = "nominal cycle month start (not a historical calendar year)"

            v_vt = ds_out.createVariable("valid_time", "f8", ("sample", "time"))
            v_vt.units = SDYFF_NOMINAL_TIME_UNITS
            v_vt.calendar = "noleap"

            v_cnt = ds_out.createVariable("counts", "i4", ("sample", "time"))
            v_cnt.long_name = "number of 6-hourly model steps in complete calendar month"

            for m_i, m_spec in enumerate(complete_months):
                cyc_y = int(m_spec["cycle_year"])
                cyc_m = int(m_spec["month"])
                days_since_0001 = float(
                    (cyc_y - 1) * 365 + sum(_NOLEAP_MONTH_DAYS[: cyc_m - 1])
                )
                v_time[m_i] = days_since_0001
                for s_i in range(members):
                    v_vt[s_i, m_i] = days_since_0001
                    v_cnt[s_i, m_i] = int(m_spec["steps_6h"])

            for out_name, arr in monthly_arrays.items():
                v = ds_out.createVariable(
                    out_name,
                    "f4",
                    ("sample", "time", "lat", "lon"),
                    zlib=True,
                    complevel=1,
                    shuffle=True,
                    chunksizes=(1, 1, int(lat_vals.size), int(lon_vals.size)),
                )
                for mk, mv in _sdyff_var_meta(out_name).items():
                    if mv:
                        setattr(v, mk, mv)
                v[:] = arr

            ds_out.Conventions = "CF-1.8"
            ds_out.source = (
                f"Spherical DYffusion ({SDYFF_CODE_REPO}@{SDYFF_CODE_COMMIT[:8]}, "
                f"{SDYFF_MODULUS_REPO}@{SDYFF_MODULUS_COMMIT[:8]})"
            )
            ds_out.weights_source = f"hf://{SDYFF_HF_REPO}@{SDYFF_HF_REVISION}"
            ds_out.reference_data = f"zenodo:{SDYFF_ZENODO_RECORD} (1 deg FV3GFS repeating annual SST cycle)"
            ds_out.reference_file_calendar = "julian"
            ds_out.calendar_note = SDYFF_CALENDAR_NOTE
            ds_out.surface_temperature_note = (
                "surface_temperature is overwritten over ocean (ocean_fraction == 1) by the prescriber "
                "and stored as surface_temperature_prescribed_ocean; air_temperature_7 is the predicted "
                "lowest-model-layer atmospheric temperature."
            )
            ds_out.coarsen_factor = int(SDYFF_COARSEN_FACTOR)
            ds_out.seed = int(seed)
        os.replace(tmp_nc, out_nc)
    finally:
        if os.path.exists(tmp_nc):
            try:
                os.unlink(tmp_nc)
            except OSError:
                pass

    return {
        "out_nc": out_nc,
        "months": [m["label"] for m in complete_months],
        "months_written": n_months,
        "predicted_variables": len(SDYFF_PREDICTED_VARS),
        "written_variables": list(monthly_arrays.keys()),
    }


def run_sdyff(out_dir: str, job: Mapping[str, Any]) -> dict[str, Any]:
    """Verify Spherical DYffusion code, checkpoints (restricted unpickler), and Zenodo files, then run rollout."""
    import torch

    requested_device = str(job.get("device", "cuda"))
    if requested_device.startswith("cuda"):
        if not torch.cuda.is_available():
            raise SystemExit(
                f"{requested_device} requested for spherical-dyffusion, but CUDA is unavailable in this job; "
                "refusing silent CPU fallback."
            )
    elif requested_device != "cpu":
        raise SystemExit(f"Unsupported device {requested_device!r} for spherical-dyffusion.")

    validated_out = validate_safe_path(out_dir, label="Spherical DYffusion out_dir")
    code_dir_raw = job.get("code_dir")
    if code_dir_raw:
        code_dir = validate_safe_path(str(code_dir_raw), label="Spherical DYffusion code_dir")
        if bool(job.get("verify_code_tree", True)):
            tree_sha, n_files = compute_sha256sum_tree_digest(
                code_dir, label="Spherical DYffusion code_dir"
            )
            exp_tree = str(job.get("code_tree_sha256") or SDYFF_TREE_SHA256)
            exp_files = int(job.get("code_tree_files") or SDYFF_TREE_FILES)
            if n_files != exp_files or tree_sha.lower() != exp_tree.lower():
                raise ValidationError(
                    f"Spherical DYffusion code tree mismatch in {code_dir}: expected {exp_files} files / "
                    f"{exp_tree}, got {n_files} files / {tree_sha}."
                )
    else:
        data_root = validate_safe_path(str(job["data_dir"]), label="AIMIP_DATA")
        info = verify_sdyff_code_dir(
            data_root,
            expected_tree_sha256=(
                str(job["code_tree_sha256"]) if job.get("code_tree_sha256") else None
            ),
            expected_tree_files=(
                int(job["code_tree_files"]) if job.get("code_tree_files") else None
            ),
        )
        code_dir = str(info["code_dir"])

    forecaster_ckpt = validate_safe_path(
        str(job["forecaster_ckpt"]), label="forecaster checkpoint"
    )
    interpolator_ckpt = validate_safe_path(
        str(job["interpolator_ckpt"]), label="interpolator checkpoint"
    )
    interpolator_cfg = validate_safe_path(
        str(job["interpolator_cfg"]), label="interpolator hydra config"
    )

    forecaster_sha = normalize_sha256(
        str(job["forecaster_sha256"]), label="Spherical DYffusion forecaster SHA-256"
    )
    interpolator_sha = normalize_sha256(
        str(job["interpolator_sha256"]), label="Spherical DYffusion interpolator SHA-256"
    )
    ipol_cfg_sha = normalize_sha256(
        str(job["interpolator_cfg_sha256"]),
        label="Spherical DYffusion interpolator config SHA-256",
    )

    f_meta = verify_sdyff_checkpoint(
        forecaster_ckpt,
        forecaster_sha,
        label="Spherical DYffusion forecaster checkpoint",
    )
    i_meta = verify_sdyff_checkpoint(
        interpolator_ckpt,
        interpolator_sha,
        label="Spherical DYffusion interpolator checkpoint",
    )
    with open_verified_hash_stream(
        interpolator_cfg,
        ipol_cfg_sha,
        label="Spherical DYffusion interpolator config",
    ) as cfg_fh:
        ipol_cfg_bytes = cfg_fh.read()

    import yaml

    parsed_ipol_cfg = yaml.safe_load(ipol_cfg_bytes.decode("utf-8"))
    if not isinstance(parsed_ipol_cfg, Mapping):
        raise ValidationError(
            f"Spherical DYffusion interpolator config {interpolator_cfg} did not parse to a mapping."
        )

    makedirs_no_group_other_write(validated_out)
    verified_ipol_cfg = os.path.join(validated_out, "sdyff_interpolator_config_verified.yaml")
    tmp_ipol_cfg = f"{verified_ipol_cfg}.tmp-{os.getpid()}"
    with open(tmp_ipol_cfg, "wb") as ipol_out:
        ipol_out.write(ipol_cfg_bytes)
    strip_group_other_write(tmp_ipol_cfg)
    os.replace(tmp_ipol_cfg, verified_ipol_cfg)

    fv3gfs_dir = validate_safe_path(
        str(job["fv3gfs_data_dir"]), label="FV3GFS 2021 reference data directory"
    )
    ref_shas = job.get("fv3gfs_file_sha256s") or {}
    for fname, spec in sorted(SDYFF_ZENODO_FILES.items()):
        fpath = os.path.join(fv3gfs_dir, fname)
        exp_sha = str(ref_shas.get(fname) or spec["sha256"])
        with open_verified_hash_stream(fpath, exp_sha, label=f"FV3GFS {fname}") as _ref_fh:
            pass

    stats_dir = os.path.join(code_dir, "data_statistics")
    for stat_name in ("centering.nc", "scaling.nc"):
        stat_p = os.path.join(stats_dir, stat_name)
        if not os.path.isfile(stat_p) or os.path.islink(stat_p):
            raise ValidationError(
                f"Spherical DYffusion statistics file missing: {stat_p}"
            )

    run_inference_py = os.path.join(code_dir, "run_inference.py")
    if not os.path.isfile(run_inference_py) or os.path.islink(run_inference_py):
        raise ValidationError(
            f"Spherical DYffusion run_inference.py not found at {run_inference_py}"
        )

    # Do not pre-create `raw_exp_dir` before `run_inference.py`: upstream `inference.py:217-219`
    # appends `-{timestamp}` whenever `os.path.exists(config.experiment_dir)` is True.
    raw_exp_dir = os.path.join(validated_out, "sdyff_raw")
    if os.path.islink(raw_exp_dir):
        raise ValidationError(
            f"Refusing symlinked Spherical DYffusion experiment_dir: {raw_exp_dir}"
        )
    if os.path.exists(raw_exp_dir):
        shutil.rmtree(raw_exp_dir)
    runtime_cfg_path = os.path.join(validated_out, "sdyff_runtime_config.yaml")

    start_iso = str(job["start"])
    steps = int(job["steps"])
    members = int(job.get("members", 1))
    seed = int(job.get("seed", 0))

    render_sdyff_runtime_config(
        forecaster_ckpt=forecaster_ckpt,
        interpolator_ckpt=interpolator_ckpt,
        interpolator_cfg=verified_ipol_cfg,
        stats_dir=stats_dir,
        data_dir=fv3gfs_dir,
        experiment_dir=raw_exp_dir,
        start_iso=start_iso,
        steps=steps,
        members=members,
        seed=seed,
        dst_config_path=runtime_cfg_path,
    )
    with open(runtime_cfg_path, "rb") as rcfg_fh:
        runtime_cfg_sha = hashlib.sha256(rcfg_fh.read()).hexdigest()

    audit_path = os.path.join(validated_out, ".sdyff_verified_loads.jsonl")
    if os.path.lexists(audit_path):
        if os.path.isdir(audit_path) and not os.path.islink(audit_path):
            shutil.rmtree(audit_path)
        else:
            os.unlink(audit_path)
    os.close(
        os.open(
            audit_path,
            os.O_CREAT | os.O_EXCL | os.O_WRONLY | getattr(os, "O_NOFOLLOW", 0),
            0o600,
        )
    )

    child_env = dict(os.environ)
    child_env["PYTHONDONTWRITEBYTECODE"] = "1"
    child_env["AIMIP_BOOTSTRAP_PKG_ROOT"] = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    child_env["WANDB_MODE"] = "disabled"
    child_env["WANDB_DISABLED"] = "true"
    child_env["HF_HUB_OFFLINE"] = "1"
    child_env["MPLBACKEND"] = "Agg"
    child_env["PL_GLOBAL_SEED"] = str(seed)
    child_env["AIMIP_SDYFF_VERIFIED_LOG"] = audit_path
    child_env["AIMIP_SDYFF_VERIFIED_DIGESTS"] = json.dumps(
        {
            os.path.abspath(forecaster_ckpt): forecaster_sha,
            os.path.abspath(interpolator_ckpt): interpolator_sha,
            os.path.abspath(interpolator_cfg): ipol_cfg_sha,
            os.path.abspath(verified_ipol_cfg): ipol_cfg_sha,
            os.path.abspath(runtime_cfg_path): runtime_cfg_sha,
        }
    )

    t0 = time.time()
    try:
        child_res = subprocess.run(
            [
                sys.executable,
                "-B",
                "-c",
                _SDYFF_BOOTSTRAP,
                code_dir,
                run_inference_py,
                runtime_cfg_path,
                str(seed),
                interpolator_ckpt,
                verified_ipol_cfg,
            ],
            cwd=code_dir,
            env=child_env,
            check=False,
        )
        if child_res.returncode != 0:
            raise SystemExit(int(child_res.returncode))
        infer_s = round(time.time() - t0, 3)

        recorded_torch_realpaths = _read_sdyff_audit_torch_realpaths(audit_path)
        missing_ckpts = [
            req_p for req_p in (forecaster_ckpt, interpolator_ckpt)
            if os.path.realpath(req_p) not in recorded_torch_realpaths
        ]
        if missing_ckpts:
            raise ValidationError(
                "Spherical DYffusion child process completed without loading required checkpoint(s) "
                f"through verified torch.load wrapper: {', '.join(missing_ckpts)}"
            )
    finally:
        if os.path.lexists(audit_path):
            try:
                os.unlink(audit_path)
            except OSError:
                pass
    if os.path.isdir(raw_exp_dir):
        strip_group_other_write(raw_exp_dir)

    ar_nc = os.path.join(raw_exp_dir, "autoregressive_predictions.nc")
    consolidated = consolidate_sdyff_daily_to_monthly(
        ar_nc,
        start_iso=start_iso,
        steps=steps,
        members=members,
        seed=seed,
        out_dir=validated_out,
    )

    result = {
        "family": "sdyff",
        "model": "spherical-dyffusion",
        "months_written": {"atmosphere": int(consolidated["months_written"])},
        "months": consolidated["months"],
        "members": members,
        "seed": seed,
        "coarsen_factor": SDYFF_COARSEN_FACTOR,
        "predicted_variables": consolidated["predicted_variables"],
        "summary_variable": SDYFF_SUMMARY_TEMP_VAR,
        "prescribed_surface_variable": SDYFF_PRESCRIBED_SURFACE_VAR,
        "forecaster_tensors": f_meta["n_tensors"],
        "interpolator_tensors": i_meta["n_tensors"],
        "inference_seconds": infer_s,
    }
    res_path = os.path.join(validated_out, "emu-result.json")
    tmp_res = f"{res_path}.tmp-{os.getpid()}"
    with open(tmp_res, "w", encoding="utf-8") as fh:
        json.dump(result, fh, indent=2)
        fh.write("\n")
    os.replace(tmp_res, res_path)
    return result
