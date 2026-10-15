#!/usr/bin/env python3
"""
quantize_checkpoint.py - Irodori-TTS モデル量子化ユーティリティ
（本家 quantize_checkpoint.py のフォーク版移植）

推論用チェックポイント（.safetensors）を torchao 量子化チェックポイント
（int8 / int4 / float8）へ変換します。出力はそのまま推論タブ・推論CLIで
読み込めます（読込側の実装は irodori_tts/quantization.py 側を参照）。

【使い方（CLI）】
    uv run python -m irodori_tts.quantize_checkpoint <input.safetensors> \
        [--output <out.safetensors>] [--quantization int8-weight-only] \
        [--profile core] [--int4-group-size 128] [--device auto] [--force]

【使い方（GUI）】
    gradio_app.py のモデル量子化タブから quantize_checkpoint() を直接呼び出す。

【前提】
    torchao が必要です: uv sync --extra quantization
    量子化済みチェックポイントは推論専用で、学習再開や再量子化には使用できません。
"""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file

from .config import ModelConfig, merge_dataclass_overrides
from .inference_runtime import _load_checkpoint_for_inference
from .model import TextToLatentRFDiT
from .quantization import (
    DEFAULT_INT4_GROUP_SIZE,
    INT4_GROUP_SIZES,
    QUANTIZATION_CLI_CHOICES,
    QUANTIZATION_PROFILES,
    flatten_quantized_state_dict,
    normalize_quantization_type,
    parse_quantization_metadata,
    quantization_cli_name,
    quantize_model,
)

__all__ = [
    "default_output_path",
    "quantize_checkpoint",
    "parse_args",
    "main",
]


def default_output_path(input_path: Path, quantization: str) -> Path:
    """入力ファイルと同じフォルダに、量子化方式名を付与した既定の出力パスを作る。"""
    suffix = quantization_cli_name(quantization)
    return input_path.with_name(f"{input_path.stem}.{suffix}.safetensors")


def _resolve_device(raw: str) -> torch.device:
    normalized = str(raw).strip().lower()
    if normalized == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        try:
            if torch.xpu.is_available():
                return torch.device("xpu")
        except AttributeError:
            pass
        return torch.device("cpu")
    device = torch.device(normalized)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA was requested but torch.cuda.is_available() is False.")
    if device.type not in {"cpu", "cuda", "xpu"}:
        raise ValueError("Quantization device must be one of: auto, cpu, cuda, xpu.")
    return device


def _copy_bundled_tokenizer(input_path: Path, output_path: Path) -> Path | None:
    source = input_path.parent / "tokenizer"
    if not (source / "tokenizer_config.json").is_file():
        return None
    destination = output_path.parent / "tokenizer"
    if source.resolve() != destination.resolve():
        shutil.copytree(source, destination, dirs_exist_ok=True)
    return destination


def quantize_checkpoint(
    input_path: str | Path,
    *,
    output_path: str | Path | None = None,
    quantization: str = "int8-weight-only",
    profile: str = "core",
    int4_group_size: int = DEFAULT_INT4_GROUP_SIZE,
    device: str = "auto",
    force: bool = False,
) -> dict:
    """推論用 .safetensors を torchao 量子化チェックポイントへ変換する。

    戻り値は実行報告の dict（CLI・GUI 双方の表示に使用）。
    失敗時は FileNotFoundError / ValueError / FileExistsError 等を送出する。
    """
    src = Path(input_path).expanduser().resolve()
    if not src.is_file():
        raise FileNotFoundError(f"Input checkpoint not found: {src}")
    if src.suffix.lower() != ".safetensors":
        raise ValueError(
            "Quantization requires an inference .safetensors checkpoint. Convert training or "
            "LoRA checkpoints with convert_checkpoint_to_safetensors.py first."
        )

    out = (
        Path(output_path).expanduser().resolve()
        if output_path is not None
        else default_output_path(src, quantization)
    )
    if out.suffix.lower() != ".safetensors":
        raise ValueError(f"Output must use .safetensors suffix: {out}")
    if src == out:
        raise ValueError("Input and output paths must be different.")
    if out.exists() and not bool(force):
        raise FileExistsError(f"Output already exists: {out} (use --force to overwrite)")

    with safe_open(str(src), framework="pt", device="cpu") as handle:
        source_metadata = handle.metadata() or {}
    if parse_quantization_metadata(source_metadata) is not None:
        raise ValueError(f"Input checkpoint is already quantized: {src}")

    model_state, model_cfg_dict, _, text_encoder_config = _load_checkpoint_for_inference(src)
    model_cfg = merge_dataclass_overrides(
        ModelConfig(),
        model_cfg_dict,
        section="checkpoint model_config",
    )
    model = TextToLatentRFDiT(
        model_cfg,
        pretrained_backbone_config=text_encoder_config,
        load_pretrained_backbone_weights=not model_cfg.use_pretrained_text_encoder,
    )
    model.load_state_dict(model_state, assign=model_cfg.use_pretrained_text_encoder)
    del model_state

    compute_device = _resolve_device(device)
    compute_dtype = torch.bfloat16
    model = model.eval().to(device=compute_device, dtype=compute_dtype)
    quantization_type = normalize_quantization_type(quantization)
    selected_modules = quantize_model(
        model,
        quantization_type=quantization_type,
        profile=profile,
        int4_group_size=int4_group_size,
    )
    model = model.to(device="cpu")

    flattened_state, metadata = flatten_quantized_state_dict(
        model.state_dict(),
        base_metadata=source_metadata,
        quantization_type=quantization_type,
        profile=profile,
        compute_dtype=compute_dtype,
        quantized_modules=len(selected_modules),
        int4_group_size=int4_group_size,
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    save_file(flattened_state, str(out), metadata=metadata)
    tokenizer_dir = _copy_bundled_tokenizer(src, out)

    tensor_bytes = sum(tensor.numel() * tensor.element_size() for tensor in flattened_state.values())
    return {
        "input_path": str(src),
        "output_path": str(out),
        "quantization": quantization_cli_name(quantization_type),
        "profile": str(profile),
        "int4_group_size": int(int4_group_size),
        "quantized_modules": len(selected_modules),
        "tensor_bytes": int(tensor_bytes),
        "tokenizer_dir": str(tokenizer_dir) if tokenizer_dir is not None else None,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert an Irodori inference safetensors checkpoint with torchao."
    )
    parser.add_argument("input_checkpoint", help="Source Irodori .safetensors checkpoint.")
    parser.add_argument(
        "--output",
        default=None,
        help="Output .safetensors path (default: <input>.<quantization>.safetensors).",
    )
    parser.add_argument(
        "--quantization",
        choices=QUANTIZATION_CLI_CHOICES,
        default="int8-weight-only",
        help="Weight/activation quantization scheme (default: int8-weight-only).",
    )
    parser.add_argument(
        "--profile",
        choices=QUANTIZATION_PROFILES,
        default="core",
        help=(
            "'core' quantizes transformer attention/MLP weights while preserving projectors, "
            "AdaLN, and duration prediction in BF16; 'all-linear' quantizes every compatible "
            "Linear."
        ),
    )
    parser.add_argument(
        "--int4-group-size",
        type=int,
        choices=INT4_GROUP_SIZES,
        default=DEFAULT_INT4_GROUP_SIZE,
        help="Group size for int4-weight-only (default: 128).",
    )
    parser.add_argument(
        "--device",
        default="auto",
        help="Temporary conversion device: auto, cpu, cuda, or xpu (default: auto).",
    )
    parser.add_argument("--force", action="store_true", help="Overwrite an existing output file.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = quantize_checkpoint(
        args.input_checkpoint,
        output_path=args.output,
        quantization=args.quantization,
        profile=args.profile,
        int4_group_size=args.int4_group_size,
        device=args.device,
        force=args.force,
    )
    print(f"Input: {report['input_path']}")
    print(f"Saved: {report['output_path']}")
    print(f"Quantization: {report['quantization']}")
    if report["quantization"] == "int4-weight-only":
        print(f"INT4 group size: {report['int4_group_size']}")
    print(f"Profile: {report['profile']}")
    print(f"Quantized Linear modules: {report['quantized_modules']:,}")
    print(f"Stored tensor bytes: {report['tensor_bytes'] / (1024**3):.2f} GiB")
    if report["tokenizer_dir"] is not None:
        print(f"Tokenizer: {report['tokenizer_dir']}")


if __name__ == "__main__":
    main()
