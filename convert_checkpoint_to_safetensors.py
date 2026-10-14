#!/usr/bin/env python3
from __future__ import annotations

import argparse
import inspect
import json
from pathlib import Path
from typing import Any

import torch
from safetensors.torch import save_file

CONFIG_META_KEY = "config_json"
TEXT_ENCODER_CONFIG_META_KEY = "text_encoder_config_json"
INFERENCE_CONFIG_KEYS = ("max_text_len", "fixed_target_latent_steps")


def _default_output_path(input_path: Path) -> Path:
    return input_path.with_suffix(".safetensors")


def _load_checkpoint(path: Path) -> dict[str, Any]:
    load_kwargs: dict[str, Any] = {"map_location": "cpu"}
    load_params = inspect.signature(torch.load).parameters
    if "weights_only" in load_params:
        load_kwargs["weights_only"] = True
    if "mmap" in load_params:
        load_kwargs["mmap"] = True

    payload = torch.load(path, **load_kwargs)
    if not isinstance(payload, dict):
        raise ValueError(f"Checkpoint payload must be a dictionary, got {type(payload)!r}.")
    return payload


def _extract_model_state(payload: dict[str, Any]) -> dict[str, torch.Tensor]:
    raw_model = payload.get("model")
    if raw_model is None and all(isinstance(v, torch.Tensor) for v in payload.values()):
        raw_model = payload

    if not isinstance(raw_model, dict):
        raise ValueError("Checkpoint does not contain a model state dictionary under 'model'.")

    model_state: dict[str, torch.Tensor] = {}
    for key, value in raw_model.items():
        if not isinstance(key, str):
            raise ValueError(f"Model state key must be str, got {type(key)!r}.")
        if not isinstance(value, torch.Tensor):
            raise ValueError(f"Model state '{key}' is not a tensor (got {type(value)!r}).")
        tensor = value.detach().cpu()
        if not tensor.is_contiguous():
            tensor = tensor.contiguous()
        model_state[key] = tensor

    if not model_state:
        raise ValueError("Model state is empty.")
    return model_state


def _extract_model_config(payload: dict[str, Any]) -> dict[str, Any]:
    model_cfg = payload.get("model_config")
    if not isinstance(model_cfg, dict):
        raise ValueError(
            "Checkpoint is missing 'model_config' dictionary required for inference compatibility."
        )
    return model_cfg


def _extract_inference_config(payload: dict[str, Any]) -> dict[str, int]:
    raw = payload.get("train_config")
    if raw is None:
        return {}
    if not isinstance(raw, dict):
        raise ValueError("Checkpoint 'train_config' must be a dictionary when present.")

    inference_cfg: dict[str, int] = {}
    for key in INFERENCE_CONFIG_KEYS:
        value = raw.get(key)
        if isinstance(value, int):
            inference_cfg[key] = int(value)
    return inference_cfg


def _build_flat_config(payload: dict[str, Any]) -> dict[str, Any]:
    flat_cfg = dict(_extract_model_config(payload))
    flat_cfg.update(_extract_inference_config(payload))
    return flat_cfg


def _extract_text_encoder_config(payload: dict[str, Any]) -> dict[str, Any] | None:
    text_encoder_config = payload.get("text_encoder_config")
    if text_encoder_config is None:
        return None
    if not isinstance(text_encoder_config, dict):
        raise ValueError("Checkpoint 'text_encoder_config' must be a dictionary when present.")
    return text_encoder_config


def _build_safetensors_metadata(
    *,
    flat_config: dict[str, Any],
    text_encoder_config: dict[str, Any] | None = None,
) -> dict[str, str]:
    metadata = {
        CONFIG_META_KEY: json.dumps(flat_config, ensure_ascii=False, separators=(",", ":")),
    }
    if text_encoder_config:
        # v4系 pretrained text encoder の構成。推論時にHF設定の再取得を不要にする。
        metadata[TEXT_ENCODER_CONFIG_META_KEY] = json.dumps(
            text_encoder_config, ensure_ascii=False, separators=(",", ":"),
        )
    return metadata


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=("Convert checkpoints (.pt) to safetensors for inference. ")
    )
    parser.add_argument("input_checkpoint", help="Path to source .pt checkpoint.")
    parser.add_argument(
        "--output",
        default=None,
        help="Output .safetensors path (default: input path with .safetensors suffix).",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing output files.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    input_path = Path(args.input_checkpoint).expanduser()
    if not input_path.is_file():
        raise FileNotFoundError(f"Input checkpoint not found: {input_path}")

    output_path = (
        Path(args.output).expanduser() if args.output else _default_output_path(input_path)
    )
    if output_path.suffix.lower() != ".safetensors":
        raise ValueError(f"Output must use .safetensors suffix: {output_path}")

    if output_path.exists() and not bool(args.force):
        raise FileExistsError(f"Output already exists: {output_path} (use --force to overwrite)")

    payload = _load_checkpoint(input_path)
    model_state = _extract_model_state(payload)
    flat_config = _build_flat_config(payload)
    text_encoder_config = _extract_text_encoder_config(payload)

    metadata = _build_safetensors_metadata(
        flat_config=flat_config,
        text_encoder_config=text_encoder_config,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    save_file(model_state, str(output_path), metadata=metadata)

    total_params = sum(int(t.numel()) for t in model_state.values())
    total_bytes = sum(int(t.numel()) * int(t.element_size()) for t in model_state.values())
    print(f"Input: {input_path}")
    print(f"Saved: {output_path}")
    print(f"Tensors: {len(model_state)}")
    if text_encoder_config:
        print(f"Embedded text_encoder_config_json: model_type={text_encoder_config.get('model_type')}")
    print(f"Total params: {total_params:,}")
    print(f"Approx tensor bytes: {total_bytes / (1024**3):.2f} GiB")


if __name__ == "__main__":
    main()
