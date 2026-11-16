"""Tests for `aimip lora info` and `aimip lora scale` on fme-layout checkpoints."""

from __future__ import annotations

import json
import math
import pathlib
from typing import Any

import pytest

torch = pytest.importorskip("torch")

from aimip.cli import main
from aimip.errors import ConfigError, ValidationError
from aimip.lora import inspect_lora, scale_lora

from conftest import sha256_path, write_fake_fme_checkpoint


def test_lora_info_reports_config_weights_only_note_and_norms(
    tmp_path: pathlib.Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    ckpt_path = write_fake_fme_checkpoint(
        tmp_path / "lora_r4_a8.tar",
        lora_rank=4,
        lora_alpha=8.0,
        spectral_lora_rank=2,
        spectral_lora_alpha=4.0,
    )

    info = inspect_lora(str(ckpt_path))
    assert "weights_only=False" in info.loader_note
    assert info.lora_rank == 4
    assert info.lora_alpha == 8.0
    assert info.effective_lora_alpha == 8.0
    assert info.spectral_lora_rank == 2
    assert info.spectral_lora_alpha == 4.0
    assert len(info.tensors) == 4
    assert {t.role for t in info.tensors} == {"lora_down", "lora_up"}
    assert len(info.spectral_tensors) == 2

    assert main(["lora", "info", str(ckpt_path)]) == 0
    out = capsys.readouterr().out
    assert "weights_only=False" in out
    assert "lora_down" in out
    assert "lora_up" in out


def test_lora_scale_updates_stepper_config_keeps_weights_untouched_and_writes_sidecar(
    tmp_path: pathlib.Path,
    allow_tmp_run_root: pathlib.Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    in_ckpt = write_fake_fme_checkpoint(
        allow_tmp_run_root / "input_lora.tar",
        lora_rank=4,
        lora_alpha=8.0,
    )
    sha_before = sha256_path(in_ckpt)
    orig_data = torch.load(str(in_ckpt), map_location="cpu", weights_only=False)

    out_ckpt = allow_tmp_run_root / "scaled_alpha_2.tar"
    rc = main(["lora", "scale", str(in_ckpt), "--alpha", "2.0", "--out", str(out_ckpt)])
    assert rc == 0
    stdout = capsys.readouterr().out
    assert "mode=stepper_config" in stdout

    # 1. Input checkpoint SHA-256 MUST remain bit-for-bit identical.
    assert sha256_path(in_ckpt) == sha_before

    # 2. Sidecar JSON records both SHA-256 digests, config mode, and scale factor 2.0 / 8.0 = 0.25.
    sidecar = pathlib.Path(f"{out_ckpt}.json")
    assert sidecar.is_file()
    meta = json.loads(sidecar.read_text(encoding="utf-8"))
    assert meta["scaling_mode"] == "stepper_config"
    assert meta["weights_modified"] is False
    assert meta["input_sha256"] == sha_before
    assert meta["output_sha256"] == sha256_path(out_ckpt)
    assert meta["alpha_old"] == 8.0
    assert meta["alpha_new"] == 2.0
    assert math.isclose(meta["scale_factor"], 0.25, rel_tol=1e-12)

    # 3. `lora info` on the scaled checkpoint reports the true `lora_alpha=2.0`,
    # and ALL `state_dict` tensors (including `lora_up`) are bit-for-bit untouched!
    scaled_info = inspect_lora(str(out_ckpt))
    assert scaled_info.lora_alpha == 2.0
    assert scaled_info.effective_lora_alpha == 2.0

    new_data = torch.load(str(out_ckpt), map_location="cpu", weights_only=False)
    assert new_data["stepper"]["dataset_info_tuple"] == orig_data["stepper"]["dataset_info_tuple"]
    assert new_data["stepper"]["config"]["step"]["config"]["builder"]["config"]["lora_alpha"] == 2.0
    assert orig_data["stepper"]["config"]["step"]["config"]["builder"]["config"]["lora_alpha"] == 8.0

    orig_sd = orig_data["stepper"]["step"]["module"]
    new_sd = new_data["stepper"]["step"]["module"]
    assert set(orig_sd.keys()) == set(new_sd.keys())
    for key in orig_sd:
        assert torch.equal(new_sd[key], orig_sd[key])

    # 4. Refuses to overwrite existing `--out`, input file, or path outside allowed run roots.
    with pytest.raises(ConfigError, match="Refusing to overwrite"):
        scale_lora(str(in_ckpt), 4.0, str(out_ckpt))
    with pytest.raises(ConfigError, match="equals input path"):
        scale_lora(str(in_ckpt), 4.0, str(in_ckpt))
    outside_dir = tmp_path / "outside"
    outside_dir.mkdir(exist_ok=True)
    with pytest.raises(ValidationError, match="must live under"):
        scale_lora(str(in_ckpt), 4.0, str(outside_dir / "outside_scratch.tar"))


def test_lora_default_alpha_equals_rank_and_multicall_wrapper(
    allow_tmp_run_root: pathlib.Path,
) -> None:
    ckpt = write_fake_fme_checkpoint(
        allow_tmp_run_root / "multicall_lora.tar",
        lora_rank=4,
        lora_alpha=None,
        wrapped_multicall=True,
    )
    info = inspect_lora(str(ckpt))
    assert info.lora_rank == 4
    assert info.lora_alpha is None
    assert info.effective_lora_alpha == 4.0

    zero_out = allow_tmp_run_root / "zero_lora.tar"
    res = scale_lora(str(ckpt), 0.0, str(zero_out))
    assert res["scale_factor"] == 0.0
    assert res["weights_modified"] is False
    zero_info = inspect_lora(str(zero_out))
    assert zero_info.lora_alpha == 0.0
    assert zero_info.effective_lora_alpha == 0.0
    # Weights remain intact so re-scaling away from 0.0 is lossless!
    for t in zero_info.tensors:
        assert t.frobenius_norm > 0.0


def test_lora_scale_forward_equivalence_spectral_mixed_and_lossless_rescaling(
    allow_tmp_run_root: pathlib.Path,
) -> None:
    """`--alpha 0` equals exact base model, covers conv/spectral/mixed LoRA, and re-scales losslessly."""
    import torch.nn.functional as F

    def _eval_ckpt_forward(ckpt_file: pathlib.Path, x_conv: torch.Tensor, x_spec: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        data = torch.load(str(ckpt_file), map_location="cpu", weights_only=False)
        bcfg = data["stepper"]["config"]["step"]["config"]["builder"]["config"]
        sd = data["stepper"]["step"]["module"]

        # Exact `LoRAConv2d.forward` (`fme/core/models/conditional_sfno/lora.py:29-42`)
        y_conv = F.conv2d(
            x_conv,
            sd["module.encoder.conv.base_conv.weight"],
            sd["module.encoder.conv.base_conv.bias"],
        )
        c_rank = int(bcfg.get("lora_rank", 0) or 0)
        if c_rank > 0:
            c_alpha = bcfg.get("lora_alpha")
            eff_c_alpha = float(c_rank) if c_alpha is None else float(c_alpha)
            delta_c = F.conv2d(
                F.conv2d(x_conv, sd["module.encoder.conv.lora_down.weight"]),
                sd["module.encoder.conv.lora_up.weight"],
            )
            y_conv = y_conv + delta_c * (eff_c_alpha / c_rank)

        # Exact `SpectralConvS2` LoRA scaling (`fme/core/models/conditional_sfno/s2convolutions.py:167-168`)
        y_spec = x_spec.clone()
        s_rank = int(bcfg.get("spectral_lora_rank", 0) or 0)
        if s_rank > 0:
            s_alpha = bcfg.get("spectral_lora_alpha")
            eff_s_alpha = float(s_rank) if s_alpha is None else float(s_alpha)
            delta_s = (x_spec @ sd["module.blocks.0.filter.lora_A"]) @ sd["module.blocks.0.filter.lora_B"]
            y_spec = y_spec + delta_s * (eff_s_alpha / s_rank)

        return y_conv, y_spec

    g = torch.Generator().manual_seed(99)
    x_conv = torch.randn((2, 3, 4, 8), generator=g)
    x_spec = torch.randn((2, 6), generator=g)

    # 1. Mixed conv + spectral LoRA (`lora_rank=4, lora_alpha=8.0, spectral_lora_rank=2, spectral_lora_alpha=4.0`)
    mixed_ckpt = write_fake_fme_checkpoint(
        allow_tmp_run_root / "mixed_r4_sr2.tar",
        lora_rank=4,
        lora_alpha=8.0,
        spectral_lora_rank=2,
        spectral_lora_alpha=4.0,
    )
    base_only_ckpt = write_fake_fme_checkpoint(
        allow_tmp_run_root / "base_no_lora.tar",
        lora_rank=0,
        lora_alpha=None,
        spectral_lora_rank=0,
        spectral_lora_alpha=None,
    )
    y_base_conv, y_base_spec = _eval_ckpt_forward(base_only_ckpt, x_conv, x_spec)
    y_full_conv, y_full_spec = _eval_ckpt_forward(mixed_ckpt, x_conv, x_spec)
    assert not torch.allclose(y_full_conv, y_base_conv)
    assert not torch.allclose(y_full_spec, y_base_spec)

    # Scale `--alpha 0` -> BOTH conv and spectral outputs equal the base model bit-for-bit!
    mixed_zero = allow_tmp_run_root / "mixed_alpha0.tar"
    res0 = scale_lora(str(mixed_ckpt), 0.0, str(mixed_zero))
    assert res0["alpha_new"] == 0.0
    assert res0["spectral_alpha_new"] == 0.0
    y_z_conv, y_z_spec = _eval_ckpt_forward(mixed_zero, x_conv, x_spec)
    assert torch.equal(y_z_conv, y_base_conv)
    assert torch.equal(y_z_spec, y_base_spec)

    # Scale `--alpha 2` (0.25x dose) -> spectral alpha scales proportionally (`4.0 * 0.25 = 1.0`)!
    mixed_qtr = allow_tmp_run_root / "mixed_alpha2.tar"
    res2 = scale_lora(str(mixed_ckpt), 2.0, str(mixed_qtr))
    assert math.isclose(res2["spectral_alpha_new"], 1.0, rel_tol=1e-12)
    y_q_conv, y_q_spec = _eval_ckpt_forward(mixed_qtr, x_conv, x_spec)
    assert torch.allclose(y_q_conv, y_base_conv + 0.25 * (y_full_conv - y_base_conv), atol=1e-6)
    assert torch.allclose(y_q_spec, y_base_spec + 0.25 * (y_full_spec - y_base_spec), atol=1e-6)

    # Re-scale `0 -> 8` from `mixed_zero` -> recovers `lora_alpha=8.0, spectral_lora_alpha=4.0` losslessly!
    mixed_restored = allow_tmp_run_root / "mixed_restored_8.tar"
    res_back = scale_lora(str(mixed_zero), 8.0, str(mixed_restored))
    assert math.isclose(res_back["spectral_alpha_new"], 4.0, rel_tol=1e-12)
    y_r_conv, y_r_spec = _eval_ckpt_forward(mixed_restored, x_conv, x_spec)
    assert torch.equal(y_r_conv, y_full_conv)
    assert torch.equal(y_r_spec, y_full_spec)

    # 2. Spectral-only LoRA (`lora_rank=0, spectral_lora_rank=2, spectral_lora_alpha=None`)
    spec_only = write_fake_fme_checkpoint(
        allow_tmp_run_root / "spectral_only.tar",
        lora_rank=0,
        lora_alpha=None,
        spectral_lora_rank=2,
        spectral_lora_alpha=None,
    )
    spec_zero = allow_tmp_run_root / "spectral_only_zero.tar"
    res_so = scale_lora(str(spec_only), 0.0, str(spec_zero))
    assert res_so["spectral_alpha_old"] == 2.0
    assert res_so["spectral_alpha_new"] == 0.0
    _, y_so_spec = _eval_ckpt_forward(spec_zero, x_conv, x_spec)
    assert torch.equal(y_so_spec, y_base_spec)


def test_localnet_builder_never_gets_spectral_lora_alpha_field(
    allow_tmp_run_root: pathlib.Path,
) -> None:
    """`LocalNetBuilder` (`dacite strict=True`) has no `spectral_lora_*` fields; never inject them."""
    localnet_ckpt = allow_tmp_run_root / "localnet_lora.tar"
    gen = torch.Generator().manual_seed(7)
    ckpt_obj = {
        "stepper": {
            "config": {
                "step": {
                    "type": "single_module",
                    "config": {
                        "builder": {
                            "type": "LocalNet",
                            "config": {
                                "embed_dim": 6,
                                "lora_rank": 4,
                                "lora_alpha": 8.0,
                            },
                        },
                        "in_names": ["TMP2m"],
                        "out_names": ["TMP2m"],
                    },
                }
            },
            "step": {
                "module": {
                    "module.conv.lora_down.weight": torch.randn((4, 3, 1, 1), generator=gen),
                    "module.conv.lora_up.weight": torch.randn((6, 4, 1, 1), generator=gen),
                }
            },
        }
    }
    torch.save(ckpt_obj, str(localnet_ckpt))

    scaled_localnet = allow_tmp_run_root / "localnet_scaled.tar"
    scale_lora(str(localnet_ckpt), 2.0, str(scaled_localnet))
    saved = torch.load(str(scaled_localnet), map_location="cpu", weights_only=False)
    bcfg = saved["stepper"]["config"]["step"]["config"]["builder"]["config"]
    assert bcfg["lora_alpha"] == 2.0
    assert "spectral_lora_alpha" not in bcfg
    assert "spectral_lora_rank" not in bcfg


def test_run_with_lora_validates_visibility_fme_schema_and_channel_compatibility(
    playground_env: dict[str, Any],
    tmp_path: pathlib.Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    runs_dir: pathlib.Path = playground_env["runs_dir"]
    valid_lora = write_fake_fme_checkpoint(runs_dir / "adapter.tar", lora_rank=4, lora_alpha=4.0)

    out_ok = runs_dir / "with_lora_ok"
    assert (
        main(
            [
                "run",
                "-m",
                "ace2-era5",
                "--start",
                "1979-01-01T00:00:00",
                "--steps",
                "4",
                "--lora",
                str(valid_lora),
                "--out",
                str(out_ok),
                "--dry-run",
            ]
        )
        == 0
    )
    capsys.readouterr()
    manifest = json.loads((out_ok / "aimip-manifest.json").read_text(encoding="utf-8"))
    assert manifest["checkpoint_path"] == str(valid_lora)
    assert manifest["checkpoint_sha256"] == sha256_path(valid_lora)
    assert manifest["lora_info"]["lora_rank"] == 4

    # `--lora` outside visible roots (`tmp_path / "outside" / "unmounted.tar"`) is refused naming allowed roots.
    unmounted_lora = write_fake_fme_checkpoint(
        tmp_path / "outside" / "unmounted.tar", lora_rank=4, lora_alpha=4.0
    )
    assert (
        main(
            [
                "run",
                "-m",
                "ace2-era5",
                "--start",
                "1979-01-01T00:00:00",
                "--steps",
                "4",
                "--lora",
                str(unmounted_lora),
                "--out",
                str(runs_dir / "with_unmounted_lora"),
                "--dry-run",
            ]
        )
        == 2
    )
    unmounted_err = capsys.readouterr().err
    assert "will not be visible inside the container at job time" in unmounted_err
    assert "AIMIP_DEFAULT_OUT_ROOT" in unmounted_err

    # `--lora` with incompatible prognostic `in_names ∩ out_names` vs IC variables is refused
    # without loading the base model's multi-GB checkpoint (`playground_env['era5_ckpt']` is non-pickle bytes).
    wrong_model_lora = runs_dir / "eamv3_channels_lora.tar"
    data_wrong = torch.load(str(valid_lora), map_location="cpu", weights_only=False)
    data_wrong["stepper"]["config"]["step"]["config"]["in_names"] = ["TREFHT", "TS"]
    data_wrong["stepper"]["config"]["step"]["config"]["out_names"] = ["TREFHT", "TS"]
    torch.save(data_wrong, str(wrong_model_lora))
    assert (
        main(
            [
                "run",
                "-m",
                "ace2-era5",
                "--start",
                "1979-01-01T00:00:00",
                "--steps",
                "4",
                "--lora",
                str(wrong_model_lora),
                "--out",
                str(runs_dir / "with_wrong_channels"),
                "--dry-run",
            ]
        )
        == 2
    )
    chan_err = capsys.readouterr().err
    assert "incompatible" in chan_err
    assert "TREFHT" in chan_err

    bogus = runs_dir / "not_fme.pt"
    torch.save({"random_weights": torch.ones(3)}, str(bogus))
    assert (
        main(
            [
                "run",
                "-m",
                "ace2-era5",
                "--start",
                "1979-01-01T00:00:00",
                "--steps",
                "4",
                "--lora",
                str(bogus),
                "--out",
                str(runs_dir / "with_lora_bad"),
                "--dry-run",
            ]
        )
        == 2
    )
    assert "stepper" in capsys.readouterr().err

    with pytest.raises(ValidationError):
        inspect_lora(str(bogus))
