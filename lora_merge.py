#!/usr/bin/env python3
"""
lora_merge.py - Irodori-TTS LoRAマージユーティリティ

【機能1】通常LoRAマージ
  LoRAアダプタ同士をマージして新しいLoRAアダプタを生成する。
  ベースモデルへの焼き込みは行わない。

  対応手法:
    - Weighted Average  : lora_A/lora_B テンソルを alpha で線形補間
    - SLERP             : ノルム保持補間
    - 部分マージ        : グループごとに手法を独立選択
    - Task Arithmetic   : タスクベクトル加算（LoRA差分での近似）

【機能2】本体モデルマージ（焼き込み）
  LoRAアダプタをベースモデルに焼き込みマージして新しいモデルファイルを生成する。
  単体・複数合成・焼き込み後マージ・部分焼き込みに対応。

  v1 (latent_dim=128) / v2 (latent_dim=32) 両対応。
  バージョン不一致時はエラーを返して処理しない。
"""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

import torch

# merge.py の共通ユーティリティを流用
from merge import (
    CHECKPOINTS_DIR,
    LAYER_GROUPS,
    _load_weights,
    _load_model_config,
    _key_group,
    _slerp_tensor,
    weighted_average,
    slerp,
    task_arithmetic,
    partial_merge,
    check_config_compatibility,
    save_merged,
    _make_output_filename,
    _format_compat_error,
)

# ─────────────────────────────────────────────────────────────────────────────
# 定数
# ─────────────────────────────────────────────────────────────────────────────

_DIM_TO_VERSION: dict[int, str] = {
    32:  "v2 (dim32)",
    128: "v1 (dim128)",
}


# v3 は latent_dim=32 で v2 と同値のため、アーキテクチャ特徴で区別する。
_V3_DURATION_ARCHITECTURE = "token_sum_adarn_zero_no_aux"
_ADAPTER_STATES = ("adapter_model.safetensors", "adapter_model.bin")

# LoRAアダプタのキー検索に使う優先順位付きパターン
# in_proj (io グループ) → wq/wk/wv/wo (attention) の順に試みる
_VERSION_KEY_PATTERNS = [
    # (キーに含まれる文字列, lora_A/lora_B のどちらか, shape軸インデックス)
    # in_features = shape[1] of lora_A
    ("in_proj",  "lora_A", 1),   # in_proj.lora_A: (rank, latent_dim*patch_size)
    ("wq",       "lora_A", 1),   # wq.lora_A:      (rank, model_dim)
    ("wk",       "lora_A", 1),
    ("wv",       "lora_A", 1),
    ("wo",       "lora_B", 0),   # wo.lora_B:       (model_dim, rank) → shape[0]=model_dim
    ("w1",       "lora_A", 1),
    ("w2",       "lora_B", 0),
    ("w3",       "lora_A", 1),
]

# ─────────────────────────────────────────────────────────────────────────────
# バージョン判定
# ─────────────────────────────────────────────────────────────────────────────

def _is_v3_config(cfg: dict[str, Any]) -> bool:
    """
    model_config dict から v3 かどうかを判定する。
    判定順:
      1. use_duration_predictor == True
      2. duration_architecture == _V3_DURATION_ARCHITECTURE
    いずれかが True なら v3 と見なす。
    """
    if cfg.get("use_duration_predictor") is True:
        return True
    if cfg.get("duration_architecture") == _V3_DURATION_ARCHITECTURE:
        return True
    return False


def _is_v3_safetensors(safe_path: "Path") -> bool:
    """
    adapter_model.safetensors のメタデータ・テンソルキーから v3 を判定する。
    判定順（指定優先度に従う）:
      1. safetensors metadata の config_json.use_duration_predictor == "true"
      2. safetensors metadata の config_json.duration_architecture == _V3_DURATION_ARCHITECTURE
      3. テンソルキーに "duration_predictor." prefix が存在する
    Returns True if v3, False otherwise.
    """
    try:
        from safetensors import safe_open
        with safe_open(str(safe_path), framework="pt", device="cpu") as h:
            meta = h.metadata() or {}
            all_keys = list(h.keys())

        config_json_str = meta.get("config_json") or meta.get("model_config") or ""
        if config_json_str:
            try:
                import json as _json
                cfg_meta = _json.loads(config_json_str)
                udp = cfg_meta.get("use_duration_predictor")
                if udp is True or str(udp).lower() == "true":
                    return True
                if cfg_meta.get("duration_architecture") == _V3_DURATION_ARCHITECTURE:
                    return True
            except Exception:
                pass

        if str(meta.get("use_duration_predictor", "")).lower() == "true":
            return True
        if meta.get("duration_architecture", "") == _V3_DURATION_ARCHITECTURE:
            return True

        if any(k.startswith("duration_predictor.") or ".duration_predictor." in k
               for k in all_keys):
            return True

    except Exception:
        pass

    return False


def _version_label(cfg: dict[str, Any]) -> str:
    """model_config から v1/v2/v3 ラベルを返す。"""
    dim = cfg.get("latent_dim")
    if dim is None:
        return "unknown"
    if int(dim) == 32 and _is_v3_config(cfg):
        return "v3 (dim32)"
    return _DIM_TO_VERSION.get(int(dim), f"unknown(dim={dim})")

def _infer_adapter_version(adapter_dir: Path) -> str:
    """
    アダプタフォルダからバージョンを推定する。

    優先順位:
    1. _full フォルダの train_state.json の base_model_config（latent_dim + v3判定）
    2. adapter_model.safetensors の metadata / テンソルキーによる v3 判定
    3. adapter_model.safetensors の in_proj.lora_A shape（dim判定）
    4. adapter_config.json（判定不可のため現状スキップ）

    Returns: "v3 (dim32)" / "v2 (dim32)" / "v1 (dim128)" / "unknown"
    """
    p = adapter_dir

    full_candidates: list[Path] = []
    name = p.name
    if name.endswith("_ema"):
        full_name = name[:-4] + "_full"
        full_candidates.append(p.parent / full_name)
    if name.endswith("_full"):
        full_candidates.append(p)
    for sib in sorted(p.parent.glob("*_full")):
        if sib not in full_candidates:
            full_candidates.append(sib)

    for full_dir in full_candidates:
        ts_path = full_dir / "train_state.json"
        if ts_path.is_file():
            try:
                ts = json.loads(ts_path.read_text(encoding="utf-8"))
                cfg = ts.get("base_model_config", {})
                dim = cfg.get("latent_dim")
                if dim is not None:
                    if int(dim) == 32 and _is_v3_config(cfg):
                        return "v3 (dim32)"
                    return _DIM_TO_VERSION.get(int(dim), f"unknown(dim={dim})")
            except Exception:
                pass

    safe_path = p / "adapter_model.safetensors"
    if safe_path.is_file():
        if _is_v3_safetensors(safe_path):
            return "v3 (dim32)"

    if safe_path.is_file():
        try:
            from safetensors import safe_open
            with safe_open(str(safe_path), framework="pt", device="cpu") as h:
                all_keys = list(h.keys())

            for mod_pat, ab, axis in _VERSION_KEY_PATTERNS:
                key = next(
                    (k for k in all_keys if mod_pat in k and f"{ab}.weight" in k),
                    None,
                )
                if key is None:
                    continue
                with safe_open(str(safe_path), framework="pt", device="cpu") as h:
                    t = h.get_tensor(key)
                feat = int(t.shape[axis])

                if mod_pat == "in_proj":
                    for known_dim in (32, 128):
                        if feat % known_dim == 0:
                            return _DIM_TO_VERSION.get(known_dim, f"unknown(dim={known_dim})")
                continue

        except Exception:
            pass

    return "不明（train_state.json なし・in_proj 未学習）"

def _get_adapter_latent_dim_from_train_state(adapter_dir: Path) -> int | None:
    """後方互換ラッパー。latent_dim のみを返す。"""
    info = _get_adapter_version_info_from_train_state(adapter_dir)
    return info[0] if info is not None else None


def _get_adapter_version_info_from_train_state(
    adapter_dir: Path,
) -> tuple[int, bool] | None:
    """
    _full フォルダの train_state.json から (latent_dim, is_v3) を返す。
    _ema フォルダが渡された場合も対応する _full を探す。
    Returns None if train_state.json が見つからない / latent_dim が読めない。
    """
    p = adapter_dir
    candidates: list[Path] = []
    name = p.name
    if name.endswith("_ema"):
        candidates.append(p.parent / (name[:-4] + "_full"))
    if name.endswith("_full"):
        candidates.append(p)
    for sib in sorted(p.parent.glob("*_full")):
        if sib not in candidates:
            candidates.append(sib)

    for full_dir in candidates:
        ts_path = full_dir / "train_state.json"
        if ts_path.is_file():
            try:
                ts = json.loads(ts_path.read_text(encoding="utf-8"))
                cfg = ts.get("base_model_config", {})
                dim = cfg.get("latent_dim")
                if dim is not None:
                    return int(dim), _is_v3_config(cfg)
            except Exception:
                pass
    return None

def _validate_adapter_vs_base(
    adapter_dir: Path,
    cfg_base: dict[str, Any],
    label: str = "LoRAアダプタ",
) -> tuple[bool, str]:
    """
    アダプタとベースモデルのバージョン互換性を検証する。
    v2/v3 は latent_dim が同じ 32 のため、is_v3 フラグで区別する。
    Returns (ok, error_message).
    """
    base_dim   = int(cfg_base.get("latent_dim", 0))
    base_is_v3 = _is_v3_config(cfg_base)
    base_ver   = _version_label(cfg_base)

    # --- train_state.json ベースの検証（最も確実） ---
    adapter_info = _get_adapter_version_info_from_train_state(adapter_dir)
    if adapter_info is not None:
        adapter_dim, adapter_is_v3 = adapter_info
        if adapter_dim != base_dim:
            adapter_ver = _DIM_TO_VERSION.get(adapter_dim, f"unknown(dim={adapter_dim})")
            return False, (
                f"❌ バージョン不一致: {label} はベースモデルと互換性がありません。\n"
                f"   ベースモデル: {base_ver} (latent_dim={base_dim})\n"
                f"   {label}: {adapter_ver} (latent_dim={adapter_dim})\n"
                f"   同じバージョン同士の組み合わせを使用してください。"
            )
        if adapter_dim == 32 and adapter_is_v3 != base_is_v3:
            adapter_ver = "v3 (dim32)" if adapter_is_v3 else "v2 (dim32)"
            return False, (
                f"⚠️ バージョン不一致（ガードレール）: {label} とベースモデルの世代が異なります。\n"
                f"   ベースモデル: {base_ver}\n"
                f"   {label}: {adapter_ver}\n"
                f"   v2 アダプタは v2 ベース、v3 アダプタは v3 ベースと組み合わせてください。"
            )
        return True, ""

    # --- train_state.json がない場合: safetensors で v3 判定 + in_proj shape で dim 検証 ---
    safe_path = adapter_dir / "adapter_model.safetensors"
    if safe_path.is_file():
        adapter_is_v3_safe = _is_v3_safetensors(safe_path)
        if base_dim == 32 and adapter_is_v3_safe != base_is_v3:
            adapter_ver = "v3 (dim32)" if adapter_is_v3_safe else "v2 (dim32)"
            return False, (
                f"⚠️ バージョン不一致（ガードレール）: {label} とベースモデルの世代が異なります。\n"
                f"   ベースモデル: {base_ver}\n"
                f"   {label}: {adapter_ver} （safetensors メタデータ/テンソルキーから推定）\n"
                f"   v2 アダプタは v2 ベース、v3 アダプタは v3 ベースと組み合わせてください。"
            )

        try:
            from safetensors import safe_open
            with safe_open(str(safe_path), framework="pt", device="cpu") as h:
                keys = list(h.keys())
            in_proj_key = next((k for k in keys if "in_proj" in k and "lora_A.weight" in k), None)
            if in_proj_key is not None:
                with safe_open(str(safe_path), framework="pt", device="cpu") as h:
                    t = h.get_tensor(in_proj_key)
                base_patch  = int(cfg_base.get("latent_patch_size", 1))
                expected_in = base_dim * base_patch
                actual_in   = int(t.shape[1])
                if actual_in != expected_in:
                    inferred_dim = next(
                        (d for d in (32, 128) if actual_in % d == 0), None
                    )
                    adapter_ver = (
                        _DIM_TO_VERSION.get(inferred_dim, f"unknown(dim={inferred_dim})")
                        if inferred_dim else "不明"
                    )
                    return False, (
                        f"❌ バージョン不一致: {label} はベースモデルと互換性がありません。\n"
                        f"   ベースモデル: {base_ver} (expected in_features={expected_in})\n"
                        f"   {label}: in_features={actual_in} → 推定バージョン={adapter_ver}\n"
                        f"   同じバージョン同士の組み合わせを使用してください。"
                    )
        except Exception as e:
            return False, f"❌ {label} の shape 検証中にエラーが発生しました: {e}"

    return True, ""
def _validate_adapters_mutual(
    adapter_dir_a: Path,
    adapter_dir_b: Path,
    label_a: str = "アダプタA",
    label_b: str = "アダプタB",
) -> tuple[bool, str]:
    """
    LoRAアダプタ同士のバージョン互換性を検証する（ベースモデルなし）。
    v2/v3 は latent_dim が同じ 32 のため is_v3 フラグで区別する。
    """
    info_a = _get_adapter_version_info_from_train_state(adapter_dir_a)
    info_b = _get_adapter_version_info_from_train_state(adapter_dir_b)

    if info_a is not None and info_b is not None:
        dim_a, is_v3_a = info_a
        dim_b, is_v3_b = info_b
        if dim_a != dim_b:
            ver_a = _DIM_TO_VERSION.get(dim_a, f"unknown(dim={dim_a})")
            ver_b = _DIM_TO_VERSION.get(dim_b, f"unknown(dim={dim_b})")
            return False, (
                f"❌ バージョン不一致: {label_a} と {label_b} は互換性がありません。\n"
                f"   {label_a}: {ver_a} (latent_dim={dim_a})\n"
                f"   {label_b}: {ver_b} (latent_dim={dim_b})\n"
                f"   同じバージョン同士の組み合わせを使用してください。"
            )
        if dim_a == 32 and is_v3_a != is_v3_b:
            ver_a = "v3 (dim32)" if is_v3_a else "v2 (dim32)"
            ver_b = "v3 (dim32)" if is_v3_b else "v2 (dim32)"
            return False, (
                f"⚠️ バージョン不一致（ガードレール）: {label_a} と {label_b} の世代が異なります。\n"
                f"   {label_a}: {ver_a}\n"
                f"   {label_b}: {ver_b}\n"
                f"   v2 同士、または v3 同士の組み合わせを使用してください。"
            )
        return True, ""

    safe_a = adapter_dir_a / "adapter_model.safetensors"
    safe_b = adapter_dir_b / "adapter_model.safetensors"
    if safe_a.is_file() and safe_b.is_file():
        is_v3_a_safe = _is_v3_safetensors(safe_a)
        is_v3_b_safe = _is_v3_safetensors(safe_b)
        if is_v3_a_safe != is_v3_b_safe:
            ver_a = "v3 (dim32)" if is_v3_a_safe else "v2/v1"
            ver_b = "v3 (dim32)" if is_v3_b_safe else "v2/v1"
            return False, (
                f"⚠️ バージョン不一致（ガードレール）: {label_a} と {label_b} の世代が異なります。\n"
                f"   {label_a}: {ver_a} （safetensors から推定）\n"
                f"   {label_b}: {ver_b} （safetensors から推定）\n"
                f"   v2 同士、または v3 同士の組み合わせを使用してください。"
            )

    try:
        cfg_a_path = adapter_dir_a / "adapter_config.json"
        cfg_b_path = adapter_dir_b / "adapter_config.json"
        if cfg_a_path.is_file() and cfg_b_path.is_file():
            cfg_a = json.loads(cfg_a_path.read_text(encoding="utf-8"))
            cfg_b = json.loads(cfg_b_path.read_text(encoding="utf-8"))
            tm_a = set(cfg_a.get("target_modules", []) or [])
            tm_b = set(cfg_b.get("target_modules", []) or [])
            mismatches = []
            if tm_a != tm_b:
                mismatches.append(f"   target_modules: {label_a}={sorted(tm_a)} / {label_b}={sorted(tm_b)}")
            if mismatches:
                return False, (
                    f"⚠️ アダプタ構成が異なります（マージ結果が不安定になる可能性があります）:\n"
                    + "\n".join(mismatches)
                    + "\n   続行するには同じ target_modules のアダプタを使用してください。"
                )
    except Exception:
        pass

    return True, ""
# ─────────────────────────────────────────────────────────────────────────────
# アダプタロード
# ─────────────────────────────────────────────────────────────────────────────

def _load_adapter_weights(adapter_dir: Path) -> dict[str, torch.Tensor]:
    """adapter_model.safetensors / adapter_model.bin からアダプタ重みを返す。"""
    safe_path = adapter_dir / "adapter_model.safetensors"
    bin_path  = adapter_dir / "adapter_model.bin"
    if safe_path.is_file():
        from safetensors.torch import load_file
        return load_file(str(safe_path), device="cpu")
    elif bin_path.is_file():
        return torch.load(str(bin_path), map_location="cpu", weights_only=True)
    else:
        raise FileNotFoundError(
            f"adapter_model.safetensors / adapter_model.bin が見つかりません: {adapter_dir}"
        )


def _load_adapter_config(adapter_dir: Path) -> dict[str, Any]:
    """adapter_config.json を読み込む。"""
    cfg_path = adapter_dir / "adapter_config.json"
    if not cfg_path.is_file():
        raise FileNotFoundError(f"adapter_config.json が見つかりません: {adapter_dir}")
    return json.loads(cfg_path.read_text(encoding="utf-8"))


# ─────────────────────────────────────────────────────────────────────────────
# LoRA同士のマージ（通常LoRAマージ）
# ─────────────────────────────────────────────────────────────────────────────

def _lora_weighted_average(
    wts_a: dict[str, torch.Tensor],
    wts_b: dict[str, torch.Tensor],
    alpha: float,
) -> dict[str, torch.Tensor]:
    """
    LoRAアダプタ同士を Weighted Average でマージする。
    merged = alpha * A + (1 - alpha) * B（テンソルごと）
    """
    result: dict[str, torch.Tensor] = {}
    all_keys = set(wts_a) | set(wts_b)
    for key in all_keys:
        if key in wts_a and key in wts_b:
            orig_dtype = wts_a[key].dtype
            merged = alpha * wts_a[key].float() + (1.0 - alpha) * wts_b[key].float()
            result[key] = merged.to(orig_dtype)
        elif key in wts_a:
            result[key] = wts_a[key].clone()
        else:
            result[key] = wts_b[key].clone()
    return result


def _lora_slerp(
    wts_a: dict[str, torch.Tensor],
    wts_b: dict[str, torch.Tensor],
    alpha: float,
) -> tuple[dict[str, torch.Tensor], list[str]]:
    """
    LoRAアダプタ同士を SLERP でマージする。
    Returns (merged_weights, fallback_keys)
    """
    result: dict[str, torch.Tensor] = {}
    fallback_keys: list[str] = []
    all_keys = set(wts_a) | set(wts_b)
    for key in all_keys:
        if key not in wts_a:
            result[key] = wts_b[key].clone()
            continue
        if key not in wts_b:
            result[key] = wts_a[key].clone()
            continue
        t_a = wts_a[key]
        t_b = wts_b[key]
        if t_a.ndim == 0:
            orig_dtype = t_a.dtype
            result[key] = (alpha * t_a.float() + (1.0 - alpha) * t_b.float()).to(orig_dtype)
            continue
        merged, fell_back = _slerp_tensor(t_a, t_b, alpha)
        result[key] = merged
        if fell_back:
            fallback_keys.append(key)
    return result, fallback_keys


def _lora_task_arithmetic(
    wts_base: dict[str, torch.Tensor],
    wts_a: dict[str, torch.Tensor],
    wts_b: dict[str, torch.Tensor],
    lambda_a: float,
    lambda_b: float,
) -> dict[str, torch.Tensor]:
    """
    LoRAアダプタ同士を Task Arithmetic でマージする。
    merged = base + lambda_a * (A - base) + lambda_b * (B - base)
    """
    result: dict[str, torch.Tensor] = {}
    all_keys = set(wts_base) | set(wts_a) | set(wts_b)
    for key in all_keys:
        base = wts_base.get(key, torch.zeros(1)).float()
        a    = wts_a.get(key, base).float()
        b    = wts_b.get(key, base).float()
        merged = base + lambda_a * (a - base) + lambda_b * (b - base)
        dtype  = wts_a.get(key, wts_b.get(key, wts_base.get(key))).dtype
        result[key] = merged.to(dtype)
    return result


def _lora_partial_merge(
    wts_a: dict[str, torch.Tensor],
    wts_b: dict[str, torch.Tensor],
    group_methods: dict[str, dict],
    wts_base: dict[str, torch.Tensor] | None = None,
) -> tuple[dict[str, torch.Tensor], list[str]]:
    """
    LoRAアダプタ同士をグループ別手法でマージする。
    キーを LoRA形式 "base_model.model.{path}.lora_A.weight" からグループ判定する。
    """
    result: dict[str, torch.Tensor] = {}
    fallback_keys: list[str] = []
    all_keys = set(wts_a) | set(wts_b)

    for key in all_keys:
        # LoRA キーからベースモデルのキーを逆算してグループ判定
        # "base_model.model.blocks.0.attention.wq.lora_A.weight"
        #   → "blocks.0.attention.wq.weight" → group = "diffusion_core"
        if key.startswith("base_model.model."):
            raw = key[len("base_model.model."):]
        else:
            raw = key
        # lora_A.weight / lora_B.weight のサフィックスを除去
        for sfx in (".lora_A.weight", ".lora_B.weight",
                    ".lora_A.bias",   ".lora_B.bias"):
            if raw.endswith(sfx):
                raw = raw[: -len(sfx)] + ".weight"
                break
        group = _key_group(raw)
        cfg   = group_methods.get(group, {"method": "weighted_average", "alpha": 0.5})
        method = cfg.get("method", "weighted_average")

        t_a = wts_a.get(key)
        t_b = wts_b.get(key)

        if t_a is None:
            result[key] = t_b.clone()
            continue
        if t_b is None:
            result[key] = t_a.clone()
            continue

        if method == "weighted_average":
            alpha = float(cfg.get("alpha", 0.5))
            result[key] = (alpha * t_a.float() + (1.0 - alpha) * t_b.float()).to(t_a.dtype)

        elif method == "slerp":
            alpha = float(cfg.get("alpha", 0.5))
            if t_a.ndim == 0:
                result[key] = (alpha * t_a.float() + (1.0 - alpha) * t_b.float()).to(t_a.dtype)
            else:
                merged, fell_back = _slerp_tensor(t_a, t_b, alpha)
                result[key] = merged
                if fell_back:
                    fallback_keys.append(key)

        elif method == "task_arithmetic":
            if wts_base is None:
                raise ValueError("task_arithmetic にはベースアダプタが必要です。")
            base = wts_base.get(key, torch.zeros_like(t_a)).float()
            lam_a = float(cfg.get("lambda_a", 0.5))
            lam_b = float(cfg.get("lambda_b", 0.5))
            merged = base + lam_a * (t_a.float() - base) + lam_b * (t_b.float() - base)
            result[key] = merged.to(t_a.dtype)

        else:
            result[key] = (0.5 * t_a.float() + 0.5 * t_b.float()).to(t_a.dtype)

    return result, fallback_keys


def _save_lora_adapter(
    merged_weights: dict[str, torch.Tensor],
    adapter_config: dict[str, Any],
    output_dir: Path,
) -> None:
    """マージ済みLoRAアダプタを出力フォルダに保存する。"""
    from safetensors.torch import save_file
    output_dir.mkdir(parents=True, exist_ok=True)
    weights = {k: v.contiguous() for k, v in merged_weights.items()}
    save_file(weights, str(output_dir / "adapter_model.safetensors"))
    (output_dir / "adapter_config.json").write_text(
        json.dumps(adapter_config, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def _make_lora_output_dirname(method_name: str) -> str:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"lora_merged_{method_name}_{stamp}_ema"


# ─────────────────────────────────────────────────────────────────────────────
# 焼き込み用関数（本体モデルマージ）
# ─────────────────────────────────────────────────────────────────────────────

def _bake_single_adapter(
    base_weights: dict[str, torch.Tensor],
    adapter_weights: dict[str, torch.Tensor],
    adapter_config: dict[str, Any],
    scale: float = 1.0,
) -> tuple[dict[str, torch.Tensor], int, int]:
    """W_merged = W_base + scale * (alpha/r) * (lora_B @ lora_A)"""
    r          = int(adapter_config.get("r", 16))
    lora_alpha = float(adapter_config.get("lora_alpha", r))
    eff_scale  = scale * (lora_alpha / r)
    result     = {k: v.clone() for k, v in base_weights.items()}
    applied = skipped = 0
    for key_a in (k for k in adapter_weights if k.endswith(".lora_A.weight")):
        key_b    = key_a.replace(".lora_A.weight", ".lora_B.weight")
        if key_b not in adapter_weights:
            skipped += 1; continue
        raw      = key_a[len("base_model.model."):]
        base_key = raw.replace(".lora_A.weight", ".weight")
        if base_key not in result:
            skipped += 1; continue
        lora_A = adapter_weights[key_a].float()
        lora_B = adapter_weights[key_b].float()
        delta  = eff_scale * (lora_B @ lora_A)
        result[base_key] = (result[base_key].float() + delta).to(result[base_key].dtype)
        applied += 1
    return result, applied, skipped


def _bake_multi_adapters(
    base_weights: dict[str, torch.Tensor],
    adapter_weights_list: list[dict[str, torch.Tensor]],
    adapter_configs: list[dict[str, Any]],
    adapter_scales: list[float],
    group_bake: dict[str, bool] | None = None,
) -> tuple[dict[str, torch.Tensor], int, int]:
    """複数アダプタを加重合算して焼き込む。group_bake で部分焼き込みも兼用。"""
    result  = {k: v.clone() for k, v in base_weights.items()}
    applied = skipped = 0
    all_paths: set[str] = set()
    for aw in adapter_weights_list:
        for k in aw:
            if k.endswith(".lora_A.weight"):
                all_paths.add(k[len("base_model.model."):].replace(".lora_A.weight", ""))
    for mod_path in all_paths:
        key_a    = f"base_model.model.{mod_path}.lora_A.weight"
        key_b    = f"base_model.model.{mod_path}.lora_B.weight"
        base_key = f"{mod_path}.weight"
        if base_key not in result:
            skipped += 1; continue
        if group_bake is not None and not group_bake.get(_key_group(base_key), True):
            skipped += 1; continue
        delta: torch.Tensor | None = None
        for aw, ac, sc in zip(adapter_weights_list, adapter_configs, adapter_scales):
            if key_a not in aw or key_b not in aw:
                continue
            r_i    = int(ac.get("r", 16))
            alp_i  = float(ac.get("lora_alpha", r_i))
            d = sc * (alp_i / r_i) * (aw[key_b].float() @ aw[key_a].float())
            delta  = d if delta is None else delta + d
        if delta is None:
            continue
        result[base_key] = (result[base_key].float() + delta).to(result[base_key].dtype)
        applied += 1
    return result, applied, skipped


def _do_bake(
    base_weights: dict[str, torch.Tensor],
    cfgs: list[dict], wts_list: list[dict],
    scales: list[float], group_bake: dict[str, bool] | None,
) -> tuple[dict[str, torch.Tensor], int, int]:
    if len(wts_list) == 1 and group_bake is None:
        return _bake_single_adapter(base_weights, wts_list[0], cfgs[0], scales[0])
    return _bake_multi_adapters(base_weights, wts_list, cfgs, scales, group_bake)


# ─────────────────────────────────────────────────────────────────────────────
# メインエントリ1: 通常LoRAマージ
# ─────────────────────────────────────────────────────────────────────────────

def run_lora_lora_merge(
    adapter_dir_a: str,
    adapter_dir_b: str,
    method: str = "weighted_average",   # "weighted_average"|"slerp"|"task_arithmetic"
    alpha: float = 0.5,
    lambda_a: float = 0.5,
    lambda_b: float = 0.5,
    base_adapter_dir: str | None = None,  # task_arithmetic 用ベースアダプタ
    use_partial: bool = False,
    group_methods: dict[str, dict] | None = None,
    output_dir: str | None = None,
) -> tuple[bool, str]:
    """
    LoRAアダプタ同士をマージして新しいLoRAアダプタを生成する。
    ベースモデルへの焼き込みは行わない。

    Returns: (success, message)
    """
    logs: list[str] = []
    try:
        p_a = Path(adapter_dir_a)
        p_b = Path(adapter_dir_b)

        for p, label in [(p_a, "アダプタA"), (p_b, "アダプタB")]:
            if not p.is_dir():
                return False, f"❌ {label} のフォルダが見つかりません: {p}"
            if not any((p / s).is_file() for s in _ADAPTER_STATES):
                return False, f"❌ {label}: adapter_model.safetensors / adapter_model.bin が見つかりません: {p}"

        logs.append(f"📂 アダプタA: {p_a.name}")
        logs.append(f"   バージョン: {_infer_adapter_version(p_a)}")
        logs.append(f"📂 アダプタB: {p_b.name}")
        logs.append(f"   バージョン: {_infer_adapter_version(p_b)}")

        # 互換性チェック
        ok, err = _validate_adapters_mutual(p_a, p_b, "アダプタA", "アダプタB")
        if not ok:
            return False, err
        logs.append("✅ バージョン互換性チェック: OK")

        # 重み・設定読み込み
        logs.append("⏳ アダプタ重みを読み込み中...")
        wts_a = _load_adapter_weights(p_a)
        wts_b = _load_adapter_weights(p_b)
        cfg_a = _load_adapter_config(p_a)

        # task_arithmetic 用ベースアダプタ
        wts_base: dict[str, torch.Tensor] | None = None
        needs_base = method == "task_arithmetic" or (
            use_partial and group_methods and
            any(v.get("method") == "task_arithmetic" for v in group_methods.values())
        )
        if needs_base:
            if not base_adapter_dir:
                return False, "❌ Task Arithmetic にはベースアダプタの指定が必要です。"
            p_base = Path(base_adapter_dir)
            if not p_base.is_dir():
                return False, f"❌ ベースアダプタが見つかりません: {p_base}"
            logs.append(f"📂 ベースアダプタ: {p_base.name}")
            wts_base = _load_adapter_weights(p_base)

        # マージ実行
        fallback_keys: list[str] = []
        method_label = method

        if use_partial and group_methods:
            logs.append("⏳ 部分マージ（グループ別手法）実行中...")
            merged_wts, fallback_keys = _lora_partial_merge(wts_a, wts_b, group_methods, wts_base)
            logs.append("✅ 部分マージ完了")
            method_label = "partial"

        elif method == "weighted_average":
            logs.append(f"⏳ Weighted Average でマージ中 (alpha={alpha:.3f})...")
            merged_wts = _lora_weighted_average(wts_a, wts_b, alpha)
            logs.append(f"✅ Weighted Average 完了")

        elif method == "slerp":
            logs.append(f"⏳ SLERP でマージ中 (alpha={alpha:.3f})...")
            merged_wts, fallback_keys = _lora_slerp(wts_a, wts_b, alpha)
            logs.append("✅ SLERP 完了")

        elif method == "task_arithmetic":
            total = lambda_a + lambda_b
            if total <= 0:
                return False, "❌ Task Arithmetic: lambda_a + lambda_b が 0 以下です。"
            lam_a = lambda_a / total
            lam_b = lambda_b / total
            logs.append(f"⏳ Task Arithmetic でマージ中 (λA={lam_a:.3f}, λB={lam_b:.3f})...")
            merged_wts = _lora_task_arithmetic(wts_base, wts_a, wts_b, lam_a, lam_b)
            logs.append("✅ Task Arithmetic 完了")
            method_label = "ta"

        else:
            return False, f"❌ 未知のマージ手法: {method}"

        if fallback_keys:
            logs.append(
                f"⚠️ SLERPフォールバック: {len(fallback_keys)} テンソルを"
                " Weighted Average で代替しました。"
            )

        # 保存
        lora_dir = Path(__file__).resolve().parent / "lora"
        out_dir = (
            Path(output_dir) if output_dir
            else lora_dir / _make_lora_output_dirname(method_label)
        )
        logs.append(f"⏳ 保存中: {out_dir} ...")
        _save_lora_adapter(merged_wts, cfg_a, out_dir)
        logs.append(f"💾 保存先: {out_dir}")
        logs.append(f"📊 テンソル数: {len(merged_wts)}")
        return True, "\n".join(logs)

    except Exception as e:
        import traceback
        return False, f"❌ エラー: {e}\n\n{traceback.format_exc()}"


# ─────────────────────────────────────────────────────────────────────────────
# メインエントリ2: 本体モデルマージ（焼き込み）
# ─────────────────────────────────────────────────────────────────────────────

def run_lora_merge(
    base_path: str,
    adapter_dirs_a: list[str],
    adapter_scales_a: list[float] | None = None,
    adapter_dirs_b: list[str] | None = None,
    adapter_scales_b: list[float] | None = None,
    post_merge_method: str = "none",
    post_alpha: float = 0.5,
    post_lambda_a: float = 0.5,
    post_lambda_b: float = 0.5,
    post_base_path: str | None = None,
    use_partial: bool = False,
    group_methods: dict[str, dict] | None = None,
    use_partial_bake_a: bool = False,
    group_bake_a: dict[str, bool] | None = None,
    use_partial_bake_b: bool = False,
    group_bake_b: dict[str, bool] | None = None,
    output_format: str = "safetensors",
    output_dir: str | None = None,
) -> tuple[bool, str]:
    """
    LoRAアダプタをベースモデルに焼き込みマージする。
    Returns: (success, message)
    """
    logs: list[str] = []
    try:
        p_base = Path(base_path)
        if not p_base.is_file():
            return False, f"❌ ベースモデルが見つかりません: {p_base}"
        if not adapter_dirs_a:
            return False, "❌ アダプタAを1件以上指定してください。"

        if adapter_scales_a is None:
            adapter_scales_a = [1.0] * len(adapter_dirs_a)
        if len(adapter_scales_a) != len(adapter_dirs_a):
            return False, "❌ adapter_scales_a の件数が adapter_dirs_a と一致しません。"

        use_post_merge = post_merge_method != "none"
        if use_post_merge and not adapter_dirs_b:
            return False, "❌ 焼き込み後マージにはアダプタBが必要です。"
        if adapter_dirs_b:
            if adapter_scales_b is None:
                adapter_scales_b = [1.0] * len(adapter_dirs_b)
            if len(adapter_scales_b) != len(adapter_dirs_b):
                return False, "❌ adapter_scales_b の件数が adapter_dirs_b と一致しません。"

        logs.append(f"📂 ベースモデル: {p_base.name}")
        cfg_base = _load_model_config(p_base)
        base_ver = _version_label(cfg_base)
        logs.append(f"   バージョン: {base_ver}")

        def _load_and_validate(dirs, scales, grp):
            cfgs_out, wts_out = [], []
            for i, (ad, sc) in enumerate(zip(dirs, scales), 1):
                p_ad  = Path(ad)
                label = f"{grp}{i}"
                if not p_ad.is_dir():
                    return [], [], f"❌ {label} のフォルダが見つかりません: {p_ad}"
                if not any((p_ad / s).is_file() for s in _ADAPTER_STATES):
                    return [], [], f"❌ {label}: アダプタファイルが見つかりません: {p_ad}"
                logs.append(f"   {label}: {p_ad.name}  (scale={sc:.3f})")
                ok, err = _validate_adapter_vs_base(p_ad, cfg_base, label)
                if not ok:
                    return [], [], err
                cfgs_out.append(_load_adapter_config(p_ad))
                wts_out.append(_load_adapter_weights(p_ad))
            return cfgs_out, wts_out, ""

        logs.append("── アダプタA ──")
        cfgs_a, wts_a, err = _load_and_validate(adapter_dirs_a, adapter_scales_a, "アダプタA-")
        if err: return False, err

        cfgs_b, wts_b = [], []
        if adapter_dirs_b:
            logs.append("── アダプタB ──")
            cfgs_b, wts_b, err = _load_and_validate(adapter_dirs_b, adapter_scales_b, "アダプタB-")
            if err: return False, err

        logs.append("✅ バージョン互換性チェック: OK")
        logs.append("⏳ ベースモデル重みを読み込み中...")
        base_weights = _load_weights(p_base)
        logs.append(f"   パラメータ数: {sum(t.numel() for t in base_weights.values()):,}")

        gb_a = group_bake_a if use_partial_bake_a else None
        logs.append("⏳ アダプタAを焼き込み中...")
        baked_a, app_a, skp_a = _do_bake(base_weights, cfgs_a, wts_a, adapter_scales_a, gb_a)
        logs.append(f"   焼き込み済み: {app_a}  スキップ: {skp_a}")

        method_label = "lora_bake"

        if use_post_merge and wts_b:
            gb_b = group_bake_b if use_partial_bake_b else None
            logs.append("⏳ アダプタBを焼き込み中...")
            baked_b, app_b, skp_b = _do_bake(base_weights, cfgs_b, wts_b, adapter_scales_b, gb_b)
            logs.append(f"   焼き込み済み: {app_b}  スキップ: {skp_b}")

            w_base_ta: dict | None = None
            needs_ta = post_merge_method == "task_arithmetic" or (
                use_partial and group_methods and
                any(v.get("method") == "task_arithmetic" for v in (group_methods or {}).values())
            )
            if needs_ta:
                ta_path = Path(post_base_path) if post_base_path else p_base
                if not ta_path.exists():
                    return False, f"❌ Task Arithmetic用ベースが見つかりません: {ta_path}"
                cfg_ta = _load_model_config(ta_path)
                ok_ta, mm_ta = check_config_compatibility(cfg_base, cfg_ta, "ベース", "TAベース")
                if not ok_ta:
                    return False, _format_compat_error(mm_ta, cfg_base, cfg_ta, "ベース", "TAベース")
                w_base_ta = _load_weights(ta_path)
                logs.append(f"📂 Task Arithmeticベース: {ta_path.name}")

            if use_partial and group_methods:
                logs.append("⏳ 部分マージ実行中...")
                merged_weights, fk = partial_merge(baked_a, baked_b, group_methods, w_base_ta)
                if fk:
                    logs.append(f"⚠️ SLERPフォールバック: {len(fk)} パラメータを代替しました。")
                logs.append("✅ 部分マージ完了")
                method_label = "lora_bake_partial"
            elif post_merge_method == "weighted_average":
                merged_weights = weighted_average(baked_a, baked_b, post_alpha)
                logs.append(f"✅ Weighted Average 完了 (alpha={post_alpha:.3f})")
                method_label = "lora_bake_wa"
            elif post_merge_method == "slerp":
                merged_weights, fk = slerp(baked_a, baked_b, post_alpha)
                if fk:
                    logs.append(f"⚠️ SLERPフォールバック: {len(fk)} パラメータを代替しました。")
                logs.append(f"✅ SLERP 完了 (alpha={post_alpha:.3f})")
                method_label = "lora_bake_slerp"
            elif post_merge_method == "task_arithmetic":
                total = post_lambda_a + post_lambda_b
                if total <= 0:
                    return False, "❌ Task Arithmetic: lambda が 0 以下です。"
                lam_a, lam_b = post_lambda_a / total, post_lambda_b / total
                merged_weights = task_arithmetic(w_base_ta, baked_a, baked_b, lam_a, lam_b)
                logs.append(f"✅ Task Arithmetic 完了 (λA={lam_a:.3f}, λB={lam_b:.3f})")
                method_label = "lora_bake_ta"
            else:
                return False, f"❌ 未知の手法: {post_merge_method}"
        else:
            merged_weights = baked_a
            logs.append("✅ LoRA焼き込み完了")

        suffix   = ".safetensors" if output_format == "safetensors" else ".pt"
        out_dir  = Path(output_dir) if output_dir else CHECKPOINTS_DIR / "lora_merged"
        out_path = out_dir / _make_output_filename(method_label, suffix)
        logs.append(f"⏳ 保存中: {out_path} ...")
        save_merged(merged_weights, cfg_base, out_path)
        logs.append(f"💾 保存先: {out_path}")
        logs.append(f"📊 総パラメータ数: {sum(int(t.numel()) for t in merged_weights.values()):,}")
        logs.append(f"🏷️  モデルバージョン: {base_ver}")
        return True, "\n".join(logs)

    except Exception as e:
        import traceback
        return False, f"❌ エラー: {e}\n\n{traceback.format_exc()}"


# ─────────────────────────────────────────────────────────────────────────────
# ユーティリティ
# ─────────────────────────────────────────────────────────────────────────────

def scan_lora_adapters_for_merge() -> list[str]:
    """lora/ 配下のアダプタフォルダを列挙する。"""
    lora_dir = Path(__file__).resolve().parent / "lora"
    lora_dir.mkdir(parents=True, exist_ok=True)
    result = []
    for p in sorted(lora_dir.rglob("adapter_config.json")):
        parent = p.parent
        if any((parent / s).is_file() for s in _ADAPTER_STATES):
            result.append(str(parent))
    return result


def peek_adapter_version(adapter_dir: str) -> str:
    """アダプタのバージョンを推定して返す（UI表示用）。"""
    return _infer_adapter_version(Path(adapter_dir))


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    import argparse, sys
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd")

    # lora-merge サブコマンド
    p_ll = sub.add_parser("lora-merge", help="LoRAアダプタ同士をマージ")
    p_ll.add_argument("--adapter-a", required=True)
    p_ll.add_argument("--adapter-b", required=True)
    p_ll.add_argument("--method", choices=["weighted_average","slerp","task_arithmetic"], default="weighted_average")
    p_ll.add_argument("--alpha", type=float, default=0.5)
    p_ll.add_argument("--lambda-a", type=float, default=0.5)
    p_ll.add_argument("--lambda-b", type=float, default=0.5)
    p_ll.add_argument("--base-adapter", default=None)
    p_ll.add_argument("--output-dir", default=None)

    # bake サブコマンド
    p_bk = sub.add_parser("bake", help="LoRAをベースモデルに焼き込み")
    p_bk.add_argument("--base-model", required=True)
    p_bk.add_argument("--adapter-a", action="append", dest="adapters_a", default=[])
    p_bk.add_argument("--scale-a",   action="append", dest="scales_a", type=float, default=[])
    p_bk.add_argument("--output-format", choices=["safetensors","pt"], default="safetensors")
    p_bk.add_argument("--output-dir", default=None)

    args = parser.parse_args()
    if args.cmd == "lora-merge":
        ok, msg = run_lora_lora_merge(
            args.adapter_a, args.adapter_b, args.method,
            args.alpha, args.lambda_a, args.lambda_b, args.base_adapter,
            output_dir=args.output_dir,
        )
    elif args.cmd == "bake":
        ok, msg = run_lora_merge(
            args.base_model, args.adapters_a,
            args.scales_a or None,
            output_format=args.output_format, output_dir=args.output_dir,
        )
    else:
        parser.print_help(); sys.exit(1)
    print(msg)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
