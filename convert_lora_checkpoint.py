#!/usr/bin/env python3
"""LoRA Full版チェックポイントから EMA平滑化済み EMA版を生成するスクリプト。

入力: lora/run_name/lora_checkpoint_XXXXXXX_full/
  ├── adapter_config.json
  ├── adapter_model.safetensors  ← EMA適用前の生重み
  ├── ema_shadow.pt              ← EMA shadow重み
  └── train_state.json

出力: lora/run_name/lora_checkpoint_XXXXXXX_ema/（新規）
  ├── adapter_config.json        ← コピー
  └── adapter_model.safetensors  ← EMA平滑化済み重み
"""
from __future__ import annotations

import argparse
import inspect
import shutil
from pathlib import Path

import torch
from safetensors.torch import save_file


def _default_output_path(input_full_dir: Path) -> Path:
    """_full を _ema に置換した同階層フォルダパスを返す。"""
    name = input_full_dir.name
    if name.endswith("_full"):
        ema_name = name[:-5] + "_ema"
    else:
        ema_name = name + "_ema"
    return input_full_dir.parent / ema_name


def _load_checkpoint(path: Path) -> dict:
    load_kwargs: dict = {"map_location": "cpu"}
    load_params = inspect.signature(torch.load).parameters
    if "weights_only" in load_params:
        load_kwargs["weights_only"] = True
    if "mmap" in load_params:
        load_kwargs["mmap"] = True
    payload = torch.load(path, **load_kwargs)
    if not isinstance(payload, dict):
        raise ValueError(f"Checkpoint must be a dict, got {type(payload)!r}")
    return payload


def convert_lora_checkpoint(
    input_full_dir: Path,
    output_ema_dir: Path,
    *,
    force: bool = False,
) -> None:
    """LoRA Full版フォルダから EMA版フォルダを生成する。"""
    # 入力バリデーション
    if not input_full_dir.is_dir():
        raise FileNotFoundError(f"入力フォルダが見つかりません: {input_full_dir}")

    adapter_config = input_full_dir / "adapter_config.json"
    adapter_model = input_full_dir / "adapter_model.safetensors"
    ema_shadow_pt = input_full_dir / "ema_shadow.pt"

    if not adapter_config.exists():
        raise FileNotFoundError(f"adapter_config.json が見つかりません: {adapter_config}")
    if not adapter_model.exists():
        raise FileNotFoundError(f"adapter_model.safetensors が見つかりません: {adapter_model}")
    if not ema_shadow_pt.exists():
        raise FileNotFoundError(
            f"ema_shadow.pt が見つかりません: {ema_shadow_pt}\n"
            "LoRA学習時に --save-full と --ema-decay を指定して保存したチェックポイントが必要です。"
        )

    # 出力先チェック
    out_safe = output_ema_dir / "adapter_model.safetensors"
    if output_ema_dir.exists() and out_safe.exists() and not force:
        raise FileExistsError(
            f"出力先が既に存在します: {output_ema_dir} (--force で上書き可)"
        )

    output_ema_dir.mkdir(parents=True, exist_ok=True)

    # EMA shadow重みをロード
    print(f"EMA shadow重みをロード中: {ema_shadow_pt}")
    ema_shadow: dict[str, torch.Tensor] = _load_checkpoint(ema_shadow_pt)

    # adapter_model.safetensors からキー構造を取得してEMA重みと対照
    from safetensors.torch import load_file as load_safetensors
    raw_weights = load_safetensors(str(adapter_model), device="cpu")

    # EMA shadowは named_parameters の名前をキーとしている
    # adapter_model.safetensors のキーとマッピング
    # peftのキー: "base_model.model.XXX.lora_A.default.weight" 等
    # EMA shadowのキー: PeftModelの named_parameters が返すキー（同じはず）
    ema_tensors: dict[str, torch.Tensor] = {}

    # まずEMA shadowのキーで直接マッチを試みる
    matched = 0
    for raw_key in raw_weights.keys():
        if raw_key in ema_shadow:
            tensor = ema_shadow[raw_key].to(raw_weights[raw_key].dtype).contiguous()
            ema_tensors[raw_key] = tensor
            matched += 1

    if matched == 0:
        # フォールバック: キーのプレフィックスを調整してマッチを試みる
        raw_keys = list(raw_weights.keys())
        shadow_keys = list(ema_shadow.keys())
        print(f"  直接マッチ失敗。キーの例:")
        print(f"  adapter_model: {raw_keys[:3]}")
        print(f"  ema_shadow:    {shadow_keys[:3]}")
        # prefix除去してマッチ
        for raw_key in raw_keys:
            for shadow_key in shadow_keys:
                if raw_key.endswith(shadow_key) or shadow_key.endswith(raw_key):
                    tensor = ema_shadow[shadow_key].to(raw_weights[raw_key].dtype).contiguous()
                    ema_tensors[raw_key] = tensor
                    matched += 1
                    break

    if matched == 0:
        raise ValueError(
            "EMA shadowとadapter_model.safetensorsのキーが一致しませんでした。\n"
            "lora_train.py で生成されたチェックポイントであることを確認してください。"
        )

    # マッチしなかったキーは生重みをそのままコピー
    for raw_key in raw_weights.keys():
        if raw_key not in ema_tensors:
            print(f"  warning: EMA shadowにキーなし（生重みを使用）: {raw_key}")
            ema_tensors[raw_key] = raw_weights[raw_key].contiguous()

    # adapter_config.json をコピー
    shutil.copy2(str(adapter_config), str(output_ema_dir / "adapter_config.json"))

    # EMA平滑化済みアダプタを保存
    save_file(ema_tensors, str(out_safe))

    total_params = sum(int(t.numel()) for t in ema_tensors.values())
    total_bytes = sum(int(t.numel()) * int(t.element_size()) for t in ema_tensors.values())

    print(f"入力 (Full): {input_full_dir}")
    print(f"出力 (EMA):  {output_ema_dir}")
    print(f"キー数:       {len(ema_tensors)} ({matched}/{len(raw_weights)} EMAマッチ)")
    print(f"総パラメータ: {total_params:,}")
    print(f"約サイズ:     {total_bytes / (1024**2):.2f} MiB")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="LoRA Full版チェックポイントから EMA版を生成する。"
    )
    parser.add_argument(
        "input_full_dir",
        help="入力: _full チェックポイントフォルダのパス。",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="出力フォルダパス（デフォルト: input の _full を _ema に置換した同階層）。",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="既存の出力を上書きする。",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    input_full_dir = Path(args.input_full_dir).expanduser().resolve()
    if args.output:
        output_ema_dir = Path(args.output).expanduser().resolve()
    else:
        output_ema_dir = _default_output_path(input_full_dir)

    convert_lora_checkpoint(
        input_full_dir=input_full_dir,
        output_ema_dir=output_ema_dir,
        force=bool(args.force),
    )


if __name__ == "__main__":
    main()
