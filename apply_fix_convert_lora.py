#!/usr/bin/env python3
"""
convert_lora_checkpoint.py パッチ:

  ❶ docstring: train_state.json → trainer_state.pt

  ❷ キーマッチを正規化ベース方式に刷新。
     以下の差異を吸収する:
       - peft バージョン差による ".default." セグメントの有無
         例: lora_A.weight  ↔  lora_A.default.weight
       - モデルアーキテクチャ差による中間セグメント挿入
         例: blocks.0.attention.wk  ↔  text_encoder.blocks.0.attention.wk
     手順:
       1. ".default." を除去して正規化
       2. 共通プレフィックス (base_model.model) を除去
       3. 残りセグメントで完全一致 → なければ末尾セグメント一致
       4. 候補が一意の場合のみ採用（複数は警告してスキップ）
"""
import sys
from pathlib import Path


def _adapt(s: str) -> str:
    return s.replace("\r\n", "\n")


def apply(target: Path) -> None:
    original = target.read_bytes()
    text = _adapt(original.decode("utf-8"))

    # ── パッチ1: docstring誤記修正 ──────────────────────────────
    OLD_DOC = _adapt('  └── train_state.json\n')
    NEW_DOC = _adapt('  └── trainer_state.pt\n')

    if OLD_DOC not in text:
        print("ERROR: パッチ1 対象文字列が見つかりません。")
        sys.exit(1)
    text = text.replace(OLD_DOC, NEW_DOC, 1)
    print("パッチ1 適用: docstring train_state.json → trainer_state.pt")

    # ── パッチ2: フォールバックのキーマッチを正規化ベースに刷新 ──
    OLD_FALLBACK = _adapt(
        '    if matched == 0:\n'
        '        # フォールバック: キーのプレフィックスを調整してマッチを試みる\n'
        '        raw_keys = list(raw_weights.keys())\n'
        '        shadow_keys = list(ema_shadow.keys())\n'
        '        print(f"  直接マッチ失敗。キーの例:")\n'
        '        print(f"  adapter_model: {raw_keys[:3]}")\n'
        '        print(f"  ema_shadow:    {shadow_keys[:3]}")\n'
        '        # prefix除去してマッチ\n'
        '        for raw_key in raw_keys:\n'
        '            for shadow_key in shadow_keys:\n'
        '                if raw_key.endswith(shadow_key) or shadow_key.endswith(raw_key):\n'
        '                    tensor = ema_shadow[shadow_key].to(raw_weights[raw_key].dtype).contiguous()\n'
        '                    ema_tensors[raw_key] = tensor\n'
        '                    matched += 1\n'
        '                    break\n'
    )

    NEW_FALLBACK = _adapt(
        '    if matched == 0:\n'
        '        # フォールバック: 正規化ベースのキーマッチ\n'
        '        #\n'
        '        # 吸収する差異:\n'
        '        #   1. peft バージョン差: ".default." セグメントの有無\n'
        '        #      例: lora_A.weight  <->  lora_A.default.weight\n'
        '        #   2. モデル構造差: 中間セグメント挿入（text_encoder 等）\n'
        '        #      例: blocks.0.attention.wk  <->  text_encoder.blocks.0.attention.wk\n'
        '        #\n'
        '        # 手順:\n'
        '        #   1. ".default." を除去して正規化\n'
        '        #   2. 共通プレフィックス (base_model.model) を除去\n'
        '        #   3. 残りセグメントで完全一致 → なければ末尾セグメント一致\n'
        '        #   4. 候補が一意の場合のみ採用（複数は警告してスキップ）\n'
        '\n'
        '        from collections import defaultdict as _defaultdict\n'
        '\n'
        '        def _norm(k: str) -> list:\n'
        '            segs = k.replace(".default.", ".").split(".")\n'
        '            # base_model.model. プレフィックスを除去\n'
        '            if len(segs) >= 2 and segs[0] == "base_model" and segs[1] == "model":\n'
        '                segs = segs[2:]\n'
        '            return segs\n'
        '\n'
        '        raw_keys = list(raw_weights.keys())\n'
        '        shadow_keys = list(ema_shadow.keys())\n'
        '        print(f"  直接マッチ失敗。キーの例:")\n'
        '        print(f"  adapter_model: {raw_keys[:3]}")\n'
        '        print(f"  ema_shadow:    {shadow_keys[:3]}")\n'
        '\n'
        '        # 正規化済みセグメントタプル → 元の shadow_key のマップ\n'
        '        _norm_map: dict = _defaultdict(list)\n'
        '        for _sk in shadow_keys:\n'
        '            _norm_map[tuple(_norm(_sk))].append(_sk)\n'
        '\n'
        '        for raw_key in raw_keys:\n'
        '            _raw_segs = _norm(raw_key)\n'
        '            # 完全一致\n'
        '            _exact = _norm_map.get(tuple(_raw_segs), [])\n'
        '            if len(_exact) == 1:\n'
        '                tensor = ema_shadow[_exact[0]].to(raw_weights[raw_key].dtype).contiguous()\n'
        '                ema_tensors[raw_key] = tensor\n'
        '                matched += 1\n'
        '                continue\n'
        '            if len(_exact) > 1:\n'
        '                print(f"  warning: キー \'{raw_key}\' に完全一致候補が複数: {_exact}")\n'
        '                continue\n'
        '            # 末尾セグメント一致（raw_segs が shadow_segs の末尾に含まれる）\n'
        '            _suffix_cands = [\n'
        '                _sk\n'
        '                for _segs_t, _sks in _norm_map.items()\n'
        '                for _sk in _sks\n'
        '                if list(_segs_t)[-len(_raw_segs):] == _raw_segs\n'
        '            ]\n'
        '            if len(_suffix_cands) == 1:\n'
        '                tensor = ema_shadow[_suffix_cands[0]].to(raw_weights[raw_key].dtype).contiguous()\n'
        '                ema_tensors[raw_key] = tensor\n'
        '                matched += 1\n'
        '            elif len(_suffix_cands) > 1:\n'
        '                print(\n'
        '                    f"  warning: キー \'{raw_key}\' にサフィックス候補が複数存在するためスキップ: "\n'
        '                    f"{_suffix_cands}"\n'
        '                )\n'
    )

    if OLD_FALLBACK not in text:
        print("ERROR: パッチ2 対象文字列が見つかりません。")
        sys.exit(1)
    text = text.replace(OLD_FALLBACK, NEW_FALLBACK, 1)
    print("パッチ2 適用: フォールバックを正規化ベースのキーマッチに刷新")

    # ── 書き戻し ──────────────────────────────────────────────────
    out_bytes = text.encode("utf-8")
    if b"\r\n" in original:
        out_bytes = out_bytes.replace(b"\n", b"\r\n")
    target.write_bytes(out_bytes)
    print(f"書き込み完了: {target}")


if __name__ == "__main__":
    target = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("convert_lora_checkpoint.py")
    if not target.exists():
        print(f"ERROR: ファイルが見つかりません: {target}")
        sys.exit(1)
    apply(target)
