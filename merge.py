#!/usr/bin/env python3
"""
merge.py - Irodori-TTS モデルマージユーティリティ

対応機能:
  - Weighted Average
  - SLERP（球面線形補間）
  - Task Arithmetic
  - LoRA的差分注入
  - 部分マージ（グループごとに手法を独立選択）

対応形式: _ema.pt / .safetensors（推論用のみ）
"""
from __future__ import annotations

import json
import warnings
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any

import torch

# ─────────────────────────────────────────────────────────────────────────────
# 定数
# ─────────────────────────────────────────────────────────────────────────────

BASE_DIR        = Path(__file__).resolve().parent
CHECKPOINTS_DIR = BASE_DIR / "checkpoints"

# safetensors メタデータキー（convert_checkpoint_to_safetensors.py と統一）
CONFIG_META_KEY = "config_json"

# model_config の互換性チェック対象キー（アーキテクチャに影響するもののみ）
ARCH_CRITICAL_KEYS = {
    "model_dim", "num_layers", "num_heads",
    "text_dim", "text_layers", "text_heads",
    "speaker_dim", "speaker_layers", "speaker_heads",
    "latent_dim", "latent_patch_size",
    # v3 対応追加: 重みshapeに直結するフィールド
    "adaln_rank",
    "duration_architecture",
    "duration_aux_dim",
    "mlp_ratio",
    "text_mlp_ratio",
    "speaker_mlp_ratio",
    "speaker_patch_size",
    "timestep_embed_dim",
    "use_caption_condition",
}

# SLERP ゼロベクトル判定閾値
SLERP_NORM_THRESHOLD = 1e-6

# Task Arithmetic デフォルトベースモデルパス
DEFAULT_BASE_PATH = CHECKPOINTS_DIR / "Aratako_Irodori-TTS-500M-v3" / "model.safetensors"

# 部分マージ グループ定義（キープレフィックス）
LAYER_GROUPS: dict[str, list[str]] = {
    "text": [
        "text_encoder.",
        "text_norm.",
    ],
    "speaker": [
        "speaker_encoder.",
        "speaker_norm.",
    ],
    "diffusion_core": [
        "blocks.",
        "cond_module.",
    ],
    "io": [
        "in_proj.",
        "out_norm.",
        "out_proj.",
    ],
}

# JointAttention 内のテキスト／話者専用キーサフィックス
TEXT_SPECIFIC_SUFFIXES    = (".attention.wk_text", ".attention.wv_text")
SPEAKER_SPECIFIC_SUFFIXES = (".attention.wk_speaker", ".attention.wv_speaker")

# ─────────────────────────────────────────────────────────────────────────────
# モデルロード
# ─────────────────────────────────────────────────────────────────────────────

def _load_weights(path: Path) -> dict[str, torch.Tensor]:
    """重み dict を返す。.pt / .safetensors 両対応。"""
    if path.suffix == ".safetensors":
        from safetensors.torch import load_file
        return load_file(str(path), device="cpu")
    else:
        ckpt = torch.load(str(path), map_location="cpu", weights_only=True)
        if "model" in ckpt:
            return ckpt["model"]
        # フラット tensor dict の場合
        if all(isinstance(v, torch.Tensor) for v in ckpt.values()):
            return ckpt
        raise ValueError(f"'model' キーが見つかりません: {path}")


def _load_model_config(path: Path) -> dict[str, Any]:
    """model_config dict を返す。.pt / .safetensors 両対応。"""
    if path.suffix == ".safetensors":
        from safetensors import safe_open
        with safe_open(str(path), framework="pt") as f:
            meta = f.metadata() or {}
        if CONFIG_META_KEY not in meta:
            raise ValueError(
                f"safetensors メタデータに '{CONFIG_META_KEY}' が見つかりません: {path}"
            )
        flat = json.loads(meta[CONFIG_META_KEY])
        # 推論用フィールド（アーキテクチャに無関係）を除去
        for k in ("max_text_len", "fixed_target_latent_steps"):
            flat.pop(k, None)
        return flat
    else:
        ckpt = torch.load(str(path), map_location="cpu", weights_only=True)
        if "model_config" not in ckpt:
            raise ValueError(f"'model_config' キーが見つかりません: {path}")
        return dict(ckpt["model_config"])


# ─────────────────────────────────────────────────────────────────────────────
# model_config 互換性チェック
# ─────────────────────────────────────────────────────────────────────────────

def check_config_compatibility(
    cfg_a: dict[str, Any],
    cfg_b: dict[str, Any],
    label_a: str = "モデルA",
    label_b: str = "モデルB",
) -> tuple[bool, list[str]]:
    """
    ARCH_CRITICAL_KEYS のみを比較する。
    Returns (is_compatible, list_of_mismatch_messages).
    """
    mismatches: list[str] = []
    for key in ARCH_CRITICAL_KEYS:
        val_a = cfg_a.get(key)
        val_b = cfg_b.get(key)
        if val_a != val_b:
            mismatches.append(
                f"  {key}: {label_a}={val_a!r}  vs  {label_b}={val_b!r}"
            )
    return (len(mismatches) == 0), mismatches


# ─────────────────────────────────────────────────────────────────────────────
# グループ判定
# ─────────────────────────────────────────────────────────────────────────────

def _key_group(key: str) -> str:
    """state_dict キーがどのグループに属するかを返す。"""
    # blocks.* 内のテキスト専用 / 話者専用キーはグループを細分化
    if key.startswith("blocks."):
        for sfx in TEXT_SPECIFIC_SUFFIXES:
            if key.endswith(sfx):
                return "text"
        for sfx in SPEAKER_SPECIFIC_SUFFIXES:
            if key.endswith(sfx):
                return "speaker"
        return "diffusion_core"
    for group, prefixes in LAYER_GROUPS.items():
        for pfx in prefixes:
            if key.startswith(pfx):
                return group
    return "io"  # フォールバック


# ─────────────────────────────────────────────────────────────────────────────
# マージ手法
# ─────────────────────────────────────────────────────────────────────────────

def weighted_average(
    w_a: dict[str, torch.Tensor],
    w_b: dict[str, torch.Tensor],
    alpha: float,
) -> dict[str, torch.Tensor]:
    """merged = alpha * A + (1 - alpha) * B"""
    result: dict[str, torch.Tensor] = {}
    for key in w_a:
        if key not in w_b:
            result[key] = w_a[key].clone()
            continue
        result[key] = alpha * w_a[key].float() + (1.0 - alpha) * w_b[key].float()
        result[key] = result[key].to(w_a[key].dtype)
    return result


def _slerp_tensor(
    t_a: torch.Tensor,
    t_b: torch.Tensor,
    alpha: float,
) -> tuple[torch.Tensor, bool]:
    """
    1D化してSLERP補間。ノルムが閾値以下の場合は (Weighted Average結果, True) を返す。
    Returns (tensor, fell_back).
    """
    orig_shape = t_a.shape
    orig_dtype = t_a.dtype
    a = t_a.float().flatten()
    b = t_b.float().flatten()

    norm_a = torch.norm(a)
    norm_b = torch.norm(b)

    if norm_a < SLERP_NORM_THRESHOLD or norm_b < SLERP_NORM_THRESHOLD:
        result = (alpha * a + (1.0 - alpha) * b).reshape(orig_shape).to(orig_dtype)
        return result, True

    a_unit = a / norm_a
    b_unit = b / norm_b
    dot = torch.clamp(torch.dot(a_unit, b_unit), -1.0, 1.0)
    theta = torch.acos(dot)

    if torch.abs(theta) < SLERP_NORM_THRESHOLD:
        result = (alpha * a + (1.0 - alpha) * b).reshape(orig_shape).to(orig_dtype)
        return result, False

    sin_theta = torch.sin(theta)
    coeff_a = torch.sin(alpha * theta) / sin_theta
    coeff_b = torch.sin((1.0 - alpha) * theta) / sin_theta

    # ノルムも補間
    merged_norm = alpha * norm_a + (1.0 - alpha) * norm_b
    merged = (coeff_a * a_unit + coeff_b * b_unit) * merged_norm
    return merged.reshape(orig_shape).to(orig_dtype), False


def slerp(
    w_a: dict[str, torch.Tensor],
    w_b: dict[str, torch.Tensor],
    alpha: float,
) -> tuple[dict[str, torch.Tensor], list[str]]:
    """
    SLERP マージ。
    Returns (merged_weights, fallback_keys).
    """
    result: dict[str, torch.Tensor] = {}
    fallback_keys: list[str] = []

    for key in w_a:
        if key not in w_b:
            result[key] = w_a[key].clone()
            continue
        t_a = w_a[key]
        t_b = w_b[key]
        if t_a.ndim == 0:
            # スカラーは線形補間
            result[key] = (alpha * t_a.float() + (1.0 - alpha) * t_b.float()).to(t_a.dtype)
            continue
        merged, fell_back = _slerp_tensor(t_a, t_b, alpha)
        result[key] = merged
        if fell_back:
            fallback_keys.append(key)

    return result, fallback_keys


def task_arithmetic(
    w_base: dict[str, torch.Tensor],
    w_a: dict[str, torch.Tensor],
    w_b: dict[str, torch.Tensor],
    lambda_a: float,
    lambda_b: float,
) -> dict[str, torch.Tensor]:
    """
    merged = base + lambda_a * (A - base) + lambda_b * (B - base)
    lambda_a + lambda_b は呼び出し前に正規化済みを想定。
    """
    result: dict[str, torch.Tensor] = {}
    for key in w_base:
        base = w_base[key].float()
        a = w_a.get(key, base).float()
        b = w_b.get(key, base).float()
        merged = base + lambda_a * (a - base) + lambda_b * (b - base)
        result[key] = merged.to(w_base[key].dtype)
    return result


def lora_inject(
    w_base: dict[str, torch.Tensor],
    w_donor: dict[str, torch.Tensor],
    scale: float,
    target_groups: list[str],
) -> dict[str, torch.Tensor]:
    """
    LoRA的差分注入:
    result[key] = base[key] + scale * (donor[key] - base[key])
    target_groups に含まれるグループのキーのみ差分注入し、それ以外は base をそのまま使う。
    """
    result: dict[str, torch.Tensor] = {}
    for key in w_base:
        base = w_base[key].float()
        if _key_group(key) in target_groups and key in w_donor:
            donor = w_donor[key].float()
            merged = base + scale * (donor - base)
            result[key] = merged.to(w_base[key].dtype)
        else:
            result[key] = w_base[key].clone()
    return result


# ─────────────────────────────────────────────────────────────────────────────
# 部分マージ（グループごとに手法を選択）
# ─────────────────────────────────────────────────────────────────────────────

def partial_merge(
    w_a: dict[str, torch.Tensor],
    w_b: dict[str, torch.Tensor],
    group_methods: dict[str, dict],
    w_base: dict[str, torch.Tensor] | None = None,
) -> tuple[dict[str, torch.Tensor], list[str]]:
    """
    グループごとに異なる手法でマージする。

    group_methods の構造:
    {
        "text":          {"method": "weighted_average", "alpha": 0.8},
        "speaker":       {"method": "slerp",            "alpha": 0.3},
        "diffusion_core":{"method": "task_arithmetic",  "lambda_a": 0.6, "lambda_b": 0.4},
        "io":            {"method": "weighted_average", "alpha": 0.5},
    }

    Returns (merged_weights, fallback_keys).
    """
    result: dict[str, torch.Tensor] = {}
    all_fallback_keys: list[str] = []

    for key in w_a:
        group = _key_group(key)
        cfg   = group_methods.get(group, {"method": "weighted_average", "alpha": 0.5})
        method = cfg.get("method", "weighted_average")

        if key not in w_b:
            result[key] = w_a[key].clone()
            continue

        if method == "weighted_average":
            alpha = float(cfg.get("alpha", 0.5))
            t = alpha * w_a[key].float() + (1.0 - alpha) * w_b[key].float()
            result[key] = t.to(w_a[key].dtype)

        elif method == "slerp":
            alpha = float(cfg.get("alpha", 0.5))
            if w_a[key].ndim == 0:
                t = alpha * w_a[key].float() + (1.0 - alpha) * w_b[key].float()
                result[key] = t.to(w_a[key].dtype)
            else:
                merged, fell_back = _slerp_tensor(w_a[key], w_b[key], alpha)
                result[key] = merged
                if fell_back:
                    all_fallback_keys.append(key)

        elif method == "task_arithmetic":
            if w_base is None:
                raise ValueError("Task Arithmetic にはベースモデルが必要です。")
            base = w_base.get(key, w_a[key]).float()
            lam_a = float(cfg.get("lambda_a", 0.5))
            lam_b = float(cfg.get("lambda_b", 0.5))
            t = base + lam_a * (w_a[key].float() - base) + lam_b * (w_b[key].float() - base)
            result[key] = t.to(w_a[key].dtype)

        else:
            # 未知の手法はフォールバック
            t = 0.5 * w_a[key].float() + 0.5 * w_b[key].float()
            result[key] = t.to(w_a[key].dtype)

    return result, all_fallback_keys


# ─────────────────────────────────────────────────────────────────────────────
# 保存
# ─────────────────────────────────────────────────────────────────────────────

def _build_metadata(model_config: dict[str, Any]) -> dict[str, str]:
    """safetensors 保存用メタデータを構築する。"""
    return {
        CONFIG_META_KEY: json.dumps(model_config, ensure_ascii=False, separators=(",", ":")),
    }


def save_merged(
    weights: dict[str, torch.Tensor],
    model_config: dict[str, Any],
    output_path: Path,
) -> None:
    """マージ済み重みを保存する。.pt / .safetensors 自動判定。"""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # contiguous 化
    weights = {k: v.contiguous() for k, v in weights.items()}

    if output_path.suffix == ".safetensors":
        from safetensors.torch import save_file
        metadata = _build_metadata(model_config)
        save_file(weights, str(output_path), metadata=metadata)
    else:
        payload = {
            "model": weights,
            "model_config": model_config,
        }
        torch.save(payload, str(output_path))


def _make_output_filename(method_name: str, suffix: str) -> str:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"merged_{method_name}_{stamp}{suffix}"


# ─────────────────────────────────────────────────────────────────────────────
# エラーメッセージ整形
# ─────────────────────────────────────────────────────────────────────────────

def _format_compat_error(
    mismatches: list[str],
    cfg_a: dict[str, Any],
    cfg_b: dict[str, Any],
    label_a: str = "モデルA",
    label_b: str = "モデルB",
    context: str = "",
) -> str:
    """
    互換性エラーメッセージを生成する。
    latent_dim の不一致が含まれる場合は v1/v2 の混在であることを明示する。
    """
    # v2/v3 は同じ dim32 のため latent_dim のみでは区別不可
    _DIM_TO_VERSION = {32: "v2/v3 (dim32)", 128: "v1 (dim128)"}

    dim_a = cfg_a.get("latent_dim")
    dim_b = cfg_b.get("latent_dim")
    has_dim_mismatch = (dim_a is not None and dim_b is not None and dim_a != dim_b)

    ctx = f"（{context}）" if context else ""

    if has_dim_mismatch:
        ver_a = _DIM_TO_VERSION.get(int(dim_a), f"unknown(dim={dim_a})")
        ver_b = _DIM_TO_VERSION.get(int(dim_b), f"unknown(dim={dim_b})")
        lines = [
            f"❌ モデルバージョン互換性エラー{ctx}: v1 / v2・v3 混在のためマージ不可",
            f"   {label_a}: {ver_a}  /  {label_b}: {ver_b}",
            "   同じバージョン（どちらも v2/v3(dim32) または どちらも v1(dim128)）のモデル同士を選択してください。",
        ]
        other = [m for m in mismatches if "latent_dim" not in m]
        if other:
            lines.append("   その他の不一致:")
            lines.extend(f"  {m}" for m in other)
    else:
        lines = [f"❌ model_config 互換性エラー{ctx}（アーキテクチャが異なるためマージ不可）:"]
        lines.extend(mismatches)

    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# メインエントリ（Gradio から呼び出す）
# ─────────────────────────────────────────────────────────────────────────────

def run_merge(
    # ── 入力モデル ──
    path_a: str,
    path_b: str,
    # ── マージ手法 ──
    method: str,                        # "weighted_average" | "slerp" | "task_arithmetic"
    alpha: float = 0.5,                 # WA / SLERP 用
    lambda_a: float = 0.5,              # Task Arithmetic 用
    lambda_b: float = 0.5,              # Task Arithmetic 用（正規化は内部で実施）
    base_path: str | None = None,       # Task Arithmetic ベース
    # ── 部分マージ ──
    use_partial: bool = False,
    group_methods: dict[str, dict] | None = None,
    # ── LoRA的差分注入 ──
    use_lora_inject: bool = False,
    lora_base_path: str | None = None,
    lora_donor_path: str | None = None,
    lora_scale: float = 0.5,
    lora_target_groups: list[str] | None = None,
    # ── 出力 ──
    output_format: str = "safetensors",  # "safetensors" | "pt"
    output_dir: str | None = None,
) -> tuple[bool, str]:
    """
    マージを実行して結果ファイルを保存する。

    Returns:
        (success: bool, message: str)
    """
    logs: list[str] = []

    try:
        p_a = Path(path_a)
        p_b = Path(path_b)

        # ── LoRA的差分注入モード ──────────────────────────────────────
        if use_lora_inject:
            if not lora_base_path or not lora_donor_path:
                return False, "❌ LoRA注入: ベースパスとドナーパスを指定してください。"

            p_base  = Path(lora_base_path)
            p_donor = Path(lora_donor_path)

            logs.append(f"📂 LoRAベース: {p_base.name}")
            logs.append(f"📂 LoRAドナー: {p_donor.name}")

            cfg_base  = _load_model_config(p_base)
            cfg_donor = _load_model_config(p_donor)
            ok, mismatches = check_config_compatibility(cfg_base, cfg_donor, "ベース", "ドナー")
            if not ok:
                msg = _format_compat_error(mismatches, cfg_base, cfg_donor, "ベース", "ドナー", "LoRA注入")
                return False, msg

            w_base  = _load_weights(p_base)
            w_donor = _load_weights(p_donor)
            target_groups = lora_target_groups or list(LAYER_GROUPS.keys())
            weights = lora_inject(w_base, w_donor, lora_scale, target_groups)
            model_config = cfg_base

            suffix = ".safetensors" if output_format == "safetensors" else ".pt"
            out_dir = Path(output_dir) if output_dir else CHECKPOINTS_DIR / "merged"
            out_path = out_dir / _make_output_filename("lora_inject", suffix)
            save_merged(weights, model_config, out_path)
            logs.append(f"✅ LoRA注入完了 → {out_path}")
            return True, "\n".join(logs)

        # ── 通常マージモード ──────────────────────────────────────────
        logs.append(f"📂 モデルA: {p_a.name}")
        logs.append(f"📂 モデルB: {p_b.name}")

        # model_config 読み込みと互換性チェック
        cfg_a = _load_model_config(p_a)
        cfg_b = _load_model_config(p_b)
        ok, mismatches = check_config_compatibility(cfg_a, cfg_b, "モデルA", "モデルB")
        if not ok:
            msg = _format_compat_error(mismatches, cfg_a, cfg_b, "モデルA", "モデルB")
            return False, msg

        logs.append("✅ model_config 互換性チェック: OK")

        # 重み読み込み
        w_a = _load_weights(p_a)
        w_b = _load_weights(p_b)

        # ── 部分マージ ────────────────────────────────────────────────
        if use_partial and group_methods:
            # Task Arithmetic を使うグループがあればベースモデルが必要
            needs_base = any(
                v.get("method") == "task_arithmetic" for v in group_methods.values()
            )
            w_base_partial: dict[str, torch.Tensor] | None = None
            if needs_base:
                bp = Path(base_path) if base_path else DEFAULT_BASE_PATH
                if not bp.exists():
                    return False, f"❌ ベースモデルが見つかりません: {bp}"
                w_base_partial = _load_weights(bp)
                logs.append(f"📂 ベース（部分マージ用）: {bp.name}")

            weights, fallback_keys = partial_merge(w_a, w_b, group_methods, w_base_partial)
            if fallback_keys:
                logs.append(
                    f"⚠️ SLERP フォールバック（ゼロベクトル検出）: {len(fallback_keys)} パラメータを Weighted Average で代替しました。"
                )
            logs.append("✅ 部分マージ完了")
            method_label = "partial"

        # ── 通常マージ手法 ────────────────────────────────────────────
        elif method == "weighted_average":
            weights = weighted_average(w_a, w_b, alpha)
            logs.append(f"✅ Weighted Average 完了 (alpha={alpha:.3f})")
            method_label = "wa"

        elif method == "slerp":
            weights, fallback_keys = slerp(w_a, w_b, alpha)
            if fallback_keys:
                logs.append(
                    f"⚠️ SLERP フォールバック（ゼロベクトル検出）: {len(fallback_keys)} パラメータを Weighted Average で代替しました。"
                )
            logs.append(f"✅ SLERP 完了 (alpha={alpha:.3f})")
            method_label = "slerp"

        elif method == "task_arithmetic":
            bp = Path(base_path) if base_path else DEFAULT_BASE_PATH
            if not bp.exists():
                return False, f"❌ Task Arithmetic: ベースモデルが見つかりません: {bp}"

            # λ 正規化
            total = lambda_a + lambda_b
            if total <= 0:
                return False, "❌ Task Arithmetic: lambda_a + lambda_b が 0 以下です。"
            lam_a_norm = lambda_a / total
            lam_b_norm = lambda_b / total

            logs.append(
                f"📂 ベース: {bp.name} / λA={lam_a_norm:.3f} λB={lam_b_norm:.3f} "
                f"（正規化前: {lambda_a:.3f} + {lambda_b:.3f}）"
            )

            # ベース model_config との互換性チェック
            cfg_base = _load_model_config(bp)
            ok_b, mm_b = check_config_compatibility(cfg_a, cfg_base, "モデルA", "ベース")
            if not ok_b:
                return False, _format_compat_error(mm_b, cfg_a, cfg_base, "モデルA", "ベース", "Task Arithmetic ベース")

            w_base = _load_weights(bp)
            weights = task_arithmetic(w_base, w_a, w_b, lam_a_norm, lam_b_norm)
            logs.append(f"✅ Task Arithmetic 完了")
            method_label = "ta"

        else:
            return False, f"❌ 未知のマージ手法: {method}"

        # 保存
        suffix = ".safetensors" if output_format == "safetensors" else ".pt"
        out_dir = Path(output_dir) if output_dir else CHECKPOINTS_DIR / "merged"
        out_path = out_dir / _make_output_filename(method_label, suffix)
        save_merged(weights, cfg_a, out_path)

        total_params = sum(int(t.numel()) for t in weights.values())
        logs.append(f"💾 保存先: {out_path}")
        logs.append(f"📊 総パラメータ数: {total_params:,}")
        return True, "\n".join(logs)

    except Exception as e:
        import traceback
        return False, f"❌ エラー: {e}\n\n{traceback.format_exc()}"


# ─────────────────────────────────────────────────────────────────────────────
# ユーティリティ（Gradio UI から使用）
# ─────────────────────────────────────────────────────────────────────────────

def scan_checkpoints_for_merge() -> list[str]:
    """checkpoints/ 配下の .pt / .safetensors を列挙（codecs・tokenizers 除外）。"""
    CHECKPOINTS_DIR.mkdir(parents=True, exist_ok=True)
    candidates = sorted([
        *CHECKPOINTS_DIR.glob("**/*.pt"),
        *CHECKPOINTS_DIR.glob("**/*.safetensors"),
    ])
    result = []
    for p in candidates:
        parts = p.relative_to(CHECKPOINTS_DIR).parts
        if parts[0] in {"codecs", "tokenizers"}:
            continue
        result.append(str(p))
    return result


def get_default_base_path() -> str:
    return str(DEFAULT_BASE_PATH)
