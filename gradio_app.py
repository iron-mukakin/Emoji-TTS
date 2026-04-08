#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import threading
import time
import warnings
from datetime import datetime
from pathlib import Path

import gradio as gr
import yaml

# torch.nn.utils.weight_norm deprecation warning (from upstream deps) is noisy
# but currently harmless for inference.
warnings.filterwarnings(
    "ignore",
    message=r"`torch\.nn\.utils\.weight_norm` is deprecated in favor of `torch\.nn\.utils\.parametrizations\.weight_norm`\.",
    category=FutureWarning,
    module=r"torch\.nn\.utils\.weight_norm",
)

try:
    import pandas as pd
    _PANDAS_AVAILABLE = True
except ImportError:
    _PANDAS_AVAILABLE = False
    pd = None  # type: ignore

from merge import (
    run_merge,
    scan_checkpoints_for_merge,
    get_default_base_path,
    LAYER_GROUPS,
)

from lora_merge import (
    run_lora_merge,
    run_lora_lora_merge,
    scan_lora_adapters_for_merge,
    peek_adapter_version,
)

from irodori_tts.inference_runtime import (
    RuntimeKey,
    SamplingRequest,
    clear_cached_runtime,
    default_runtime_device,
    get_cached_runtime,
    list_available_runtime_devices,
    list_available_runtime_precisions,
    save_wav,
)

# ─────────────────────────────────────────────────────────────────────────────
# パス定数
# ─────────────────────────────────────────────────────────────────────────────
BASE_DIR         = Path(__file__).resolve().parent
CHECKPOINTS_DIR  = BASE_DIR / "checkpoints"
CONFIGS_DIR      = BASE_DIR / "configs"
LOGS_DIR         = BASE_DIR / "logs"
OUTPUTS_DIR      = BASE_DIR / "gradio_outputs"
LORA_DIR         = BASE_DIR / "lora"
DEFAULT_HF_REPO  = "Aratako/Irodori-TTS-500M-v2"
DEFAULT_CONFIG   = "train_v2.yaml"
DEFAULT_PREPARE_CODEC_REPO = "Aratako/Semantic-DACVAE-Japanese-32dim"
PREPARE_CODEC_REPO_CHOICES = [
    "Aratako/Semantic-DACVAE-Japanese-32dim",  # v2 (dim32)
    "facebook/dacvae-watermarked",             # v1 (dim128)
]
FIXED_SECONDS    = 30.0
DATASET_TOOLS    = BASE_DIR / "dataset_tools.py"
DEFAULT_DATASET_DIR = BASE_DIR / "my_dataset"
SPEAKERS_DIR        = BASE_DIR / "speakers"

# ─────────────────────────────────────────────────────────────────────────────
# グローバルプロセス管理（学習・前処理の排他制御）
# ─────────────────────────────────────────────────────────────────────────────
_proc_lock  = threading.Lock()
_active_proc: subprocess.Popen | None = None
_active_log_path: Path | None = None


# ─────────────────────────────────────────────────────────────────────────────
# 共通ユーティリティ
# ─────────────────────────────────────────────────────────────────────────────

def _scan_checkpoints() -> list[str]:
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


def _scan_configs() -> list[str]:
    CONFIGS_DIR.mkdir(parents=True, exist_ok=True)
    return sorted(str(p) for p in CONFIGS_DIR.glob("*.yaml")) + \
           sorted(str(p) for p in CONFIGS_DIR.glob("*.yml"))


def _scan_manifests() -> list[str]:
    return sorted(str(p) for p in BASE_DIR.glob("**/*.jsonl"))


def _scan_train_checkpoints() -> list[str]:
    result = []
    for p in BASE_DIR.glob("**/*.pt"):
        if p.stat().st_size > 1024 * 1024:
            result.append(str(p))
    return sorted(result)


def _scan_lora_adapters() -> list[str]:
    """adapter_config.json と adapter_model.safetensors/.bin の両方が存在するフォルダを列挙。"""
    LORA_DIR.mkdir(parents=True, exist_ok=True)
    _ADAPTER_STATES = ("adapter_model.safetensors", "adapter_model.bin")
    result = []
    for p in sorted(LORA_DIR.rglob("adapter_config.json")):
        parent = p.parent
        if any((parent / s).is_file() for s in _ADAPTER_STATES):
            result.append(str(parent))
    return result


def _scan_lora_full_adapters() -> list[str]:
    LORA_DIR.mkdir(parents=True, exist_ok=True)
    result = []
    for p in sorted(LORA_DIR.rglob("adapter_config.json")):
        if p.parent.name.endswith("_full"):
            result.append(str(p.parent))
    return result


# ─────────────────────────────────────────────────────────────────────────────
# スピーカーライブラリ ユーティリティ
# ─────────────────────────────────────────────────────────────────────────────

def _scan_speakers() -> list[str]:
    """speakers/ 配下のキャラクター名を列挙（ref.pt が存在するフォルダのみ）。"""
    SPEAKERS_DIR.mkdir(parents=True, exist_ok=True)
    return ["（使用しない）"] + sorted(
        d.name for d in SPEAKERS_DIR.iterdir()
        if d.is_dir() and (d / "ref.pt").exists()
    )


def _run_create_speaker(
    char_name: str,
    wav_path: str,
    checkpoint: str,
    model_device: str,
    model_precision: str,
    codec_device: str,
    codec_precision: str,
) -> str:
    """参照WAVをDACVAEエンコードして speakers/{char_name}/ に3ファイルを生成する。"""
    import shutil
    import torch
    import torchaudio

    char_name = str(char_name).strip()
    if not char_name:
        return "エラー: キャラクター名を入力してください。"
    if not wav_path or not Path(wav_path).is_file():
        return "エラー: WAVファイルを選択してください。"

    # ── ガードレール: キャッシュ済み runtime を直接使用 ──────────────
    # get_cached_runtime() による暗黙の再ロードを防止する。
    # キャッシュ未ロードの場合はエラーを返す。
    # キャッシュ済みの場合は codec_repo も含めてそのまま流用する。
    from irodori_tts.inference_runtime import _RUNTIME_CACHE_KEY, _RUNTIME_CACHE_VALUE
    _cached_runtime = _RUNTIME_CACHE_VALUE
    _cached_key = _RUNTIME_CACHE_KEY

    if _cached_runtime is None:
        return (
            "エラー: モデルが読み込まれていません。\n"
            "先に「📥 モデル読み込み」ボタンを押してください。"
        )

    # checkpoint / device / precision の一致確認（codec_repo はキャッシュから自動使用）
    try:
        runtime_key = _build_runtime_key(
            checkpoint, model_device, model_precision,
            codec_device, codec_precision, False, "（なし）",
            codec_repo=_cached_key.codec_repo,
        )
    except Exception as e:
        return f"エラー: チェックポイントパスが無効です。\n{e}"

    _codec_fields = ("checkpoint", "model_device", "codec_repo", "model_precision",
                     "codec_device", "codec_precision", "enable_watermark")
    _mismatch = [
        f for f in _codec_fields
        if getattr(_cached_key, f, None) != getattr(runtime_key, f, None)
    ]
    if _mismatch:
        return (
            f"エラー: 現在読み込み中のモデル設定と登録パネルの設定が一致しません。\n"
            f"不一致フィールド: {', '.join(_mismatch)}\n"
            "推論タブの設定と一致させてから「📥 モデル読み込み」を実行してください。"
        )

    codec = _cached_runtime.codec

    # モデルバージョン情報を取得してログに含める
    ldim = int(_cached_runtime.model_cfg.latent_dim)
    version_label = "v2" if ldim == 32 else ("v1" if ldim == 128 else f"unknown(dim={ldim})")

    try:
        wav, sr = torchaudio.load(wav_path)
    except Exception as e:
        return f"エラー: WAV読み込み失敗: {e}"

    # モノラル化
    if wav.shape[0] > 1:
        wav = wav.mean(dim=0, keepdim=True)

    # 最大30秒トリム
    max_samples = int(30.0 * sr)
    trimmed = wav.shape[1] > max_samples
    if trimmed:
        wav = wav[:, :max_samples]

    duration_sec = round(wav.shape[1] / sr, 2)

    try:
        with torch.inference_mode():
            latent = codec.encode_waveform(wav.unsqueeze(0), sample_rate=sr).cpu()
    except Exception as e:
        return f"エラー: DACVAEエンコード失敗: {e}"

    out_dir = SPEAKERS_DIR / char_name
    out_dir.mkdir(parents=True, exist_ok=True)

    shutil.copy2(wav_path, out_dir / "ref.wav")
    torch.save(latent, out_dir / "ref.pt")
    (out_dir / "profile.json").write_text(
        json.dumps(
            {
                "name": char_name,
                "duration_sec": duration_sec,
                "latent_shape": list(latent.shape),
                "source_wav": str(wav_path),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    msg = f"✅ 登録完了: speakers/{char_name}/\n"
    msg += f"  ref.wav / ref.pt / profile.json\n"
    msg += f"  潜在 shape: {tuple(latent.shape)}  ({duration_sec}秒)\n"
    msg += f"  使用モデル: {version_label} (latent_dim={ldim}, codec={_cached_key.codec_repo})"
    if trimmed:
        msg += "\n  （30秒にトリム済み）"
    return msg


# ─────────────────────────────────────────────────────────────────────────────
# LoRAプリセット用ユーティリティ  ← 追加
# ─────────────────────────────────────────────────────────────────────────────

def _scan_lora_configs() -> list[str]:
    """configs/ 配下のYAMLのうち 'lora' セクションを持つものを列挙。"""
    CONFIGS_DIR.mkdir(parents=True, exist_ok=True)
    result = []
    for p in sorted(CONFIGS_DIR.glob("*.yaml")) + sorted(CONFIGS_DIR.glob("*.yml")):
        try:
            data = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
            if "lora" in data:
                result.append(str(p))
        except Exception:
            pass
    return result


def _lora_config_from_ui(
    base_model, manifest, output_dir, run_name,
    lora_rank, lora_alpha, lora_dropout, target_modules,
    save_mode, attention_backend,
    use_early_stopping, es_patience, es_min_delta,
    use_ema, ema_decay,
    resume_enabled, resume_lora_path,
    batch_size, grad_accum, lr, optimizer,
    lr_scheduler, warmup_steps,
    max_steps, save_every, log_every,
    valid_ratio, valid_every,
    wandb_enabled, wandb_project, wandb_run_name,
    seed,
) -> dict:
    return {
        "lora": {
            "lora_rank": int(lora_rank),
            "lora_alpha": float(lora_alpha),
            "lora_dropout": float(lora_dropout),
            "target_modules": str(target_modules),
            "save_mode": str(save_mode),
            "attention_backend": str(attention_backend),
            "use_early_stopping": bool(use_early_stopping),
            "es_patience": int(es_patience),
            "es_min_delta": float(es_min_delta),
            "use_ema": bool(use_ema),
            "ema_decay": float(ema_decay),
            "batch_size": int(batch_size),
            "grad_accum": int(grad_accum),
            "lr": float(lr),
            "optimizer": str(optimizer),
            "lr_scheduler": str(lr_scheduler),
            "warmup_steps": int(warmup_steps),
            "max_steps": int(max_steps),
            "save_every": int(save_every),
            "log_every": int(log_every),
            "valid_ratio": float(valid_ratio),
            "valid_every": int(valid_every),
            "wandb_enabled": bool(wandb_enabled),
            "wandb_project": str(wandb_project) if wandb_project else "",
            "wandb_run_name": str(wandb_run_name) if wandb_run_name else "",
            "seed": int(seed),
        }
    }


def _save_lora_config(config_name: str, data: dict) -> str:
    CONFIGS_DIR.mkdir(parents=True, exist_ok=True)
    p = Path(config_name)
    if not p.suffix:
        p = p.with_suffix(".yaml")
    if not p.is_absolute():
        p = CONFIGS_DIR / p.name
    with open(p, "w", encoding="utf-8") as f:
        yaml.dump(data, f, allow_unicode=True, default_flow_style=False, sort_keys=False)
    return f"保存しました: {p}"


def _load_lora_config(config_path: str) -> dict:
    if not config_path:
        return {}
    p = Path(config_path)
    if not p.is_file():
        return {}
    data = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
    return data.get("lora", {})


def _load_lora_preset(config_path: str):
    if not config_path:
        return (
            16, 32.0, 0.05, "wq,wk,wv,wo",
            "EMAのみ", "sdpa",
            False, 3, 0.01,
            True, 0.9999,
            4, 1, 1e-4, "adamw",
            "none", 0,
            1000, 100, 10,
            0.0, 100,
            False, "", "",
            0,
        )
    cfg = _load_lora_config(config_path)
    def g(k, fb):
        return cfg.get(k, fb)
    return (
        g("lora_rank", 16),
        g("lora_alpha", 32.0),
        g("lora_dropout", 0.05),
        g("target_modules", "wq,wk,wv,wo"),
        g("save_mode", "EMAのみ"),
        g("attention_backend", "sdpa"),
        g("use_early_stopping", False),
        g("es_patience", 3),
        g("es_min_delta", 0.01),
        g("use_ema", True),
        g("ema_decay", 0.9999),
        g("batch_size", 4),
        g("grad_accum", 1),
        g("lr", 1e-4),
        g("optimizer", "adamw"),
        g("lr_scheduler", "none"),
        g("warmup_steps", 0),
        g("max_steps", 1000),
        g("save_every", 100),
        g("log_every", 10),
        g("valid_ratio", 0.0),
        g("valid_every", 100),
        g("wandb_enabled", False),
        g("wandb_project", "") or "",
        g("wandb_run_name", "") or "",
        g("seed", 0),
    )


def _save_lora_preset(name: str, *cfg_args):
    cfg_data = _lora_config_from_ui(*cfg_args)
    return _save_lora_config(name, cfg_data)


# ─────────────────────────────────────────────────────────────────────────────
# 以降は元のコードと同じ（_ensure_default_model から build_ui まで）
# ─────────────────────────────────────────────────────────────────────────────

def _detect_model_version_from_runtime() -> tuple[str, str, int] | None:
    """キャッシュ済み runtime からモデルバージョン情報を取得する。
    戻り値: (version_label, codec_repo, latent_dim) または None"""
    from irodori_tts.inference_runtime import _RUNTIME_CACHE_VALUE
    runtime = _RUNTIME_CACHE_VALUE
    if runtime is None:
        return None
    ldim = int(runtime.model_cfg.latent_dim)
    version = "v2" if ldim == 32 else ("v1" if ldim == 128 else f"unknown(dim={ldim})")
    return version, runtime.key.codec_repo, ldim


def _runtime_uses_voice_design() -> bool:
    from irodori_tts.inference_runtime import _RUNTIME_CACHE_VALUE

    runtime = _RUNTIME_CACHE_VALUE
    if runtime is None:
        return False
    return bool(getattr(runtime.model_cfg, "use_caption_condition", False))


def _codec_repo_for_latent_dim(latent_dim: int) -> str:
    """latent_dim からデフォルト codec_repo を返す。"""
    if latent_dim == 32:
        return "Aratako/Semantic-DACVAE-Japanese-32dim"
    return "facebook/dacvae-watermarked"


def _validate_lora_compat_ui(lora_adapter: str) -> str:
    """
    LoRAアダプタとロード済みモデルの互換性をUIから検証する。
    戻り値: 状態メッセージ文字列（エラー時は ❌ プレフィックス）
    """
    import json as _json
    if not lora_adapter or lora_adapter.strip() in ("", "（なし）"):
        return ""

    lp = Path(lora_adapter.strip())
    if not lp.is_dir():
        return f"❌ フォルダが存在しません: {lp}"

    adapter_config_path = lp / "adapter_config.json"
    if not adapter_config_path.is_file():
        return "❌ adapter_config.json が見つかりません。"

    _ADAPTER_STATES = ("adapter_model.safetensors", "adapter_model.bin")
    if not any((lp / s).is_file() for s in _ADAPTER_STATES):
        return "❌ adapter_model.safetensors / adapter_model.bin が見つかりません。"

    # 本家版 vs フォーク版の識別
    try:
        adapter_cfg = _json.loads(adapter_config_path.read_text(encoding="utf-8"))
    except Exception as e:
        return f"❌ adapter_config.json の読み込みに失敗: {e}"

    has_metadata = (lp / "irodori_lora_metadata.json").is_file()
    target_modules = adapter_cfg.get("target_modules")
    is_upstream = has_metadata or (
        isinstance(target_modules, str) and target_modules.startswith("^")
    )
    origin_label = "本家版" if is_upstream else "フォーク版"

    # ロード済みモデルとの latent_dim 照合
    info = _detect_model_version_from_runtime()
    if info is None:
        return f"⚠️ {origin_label}アダプタ検出。モデル未読み込みのため latent_dim 照合をスキップ。"

    version_label, _, ldim = info
    adapter_st_path = lp / "adapter_model.safetensors"
    if adapter_st_path.is_file():
        try:
            from safetensors import safe_open as _safe_open
            from irodori_tts.inference_runtime import _RUNTIME_CACHE_VALUE
            runtime = _RUNTIME_CACHE_VALUE
            expected_patched = ldim * int(runtime.model_cfg.latent_patch_size)

            with _safe_open(str(adapter_st_path), framework="pt", device="cpu") as _h:
                all_keys = list(_h.keys())
            in_proj_key = next(
                (k for k in all_keys if "in_proj" in k and "lora_A" in k), None
            )
            if in_proj_key is not None:
                with _safe_open(str(adapter_st_path), framework="pt", device="cpu") as _h:
                    t = _h.get_tensor(in_proj_key)
                adapter_in = int(t.shape[1])
                if adapter_in != expected_patched:
                    return (
                        f"❌ 互換性エラー ({origin_label}): "
                        f"アダプタ in_features={adapter_in} ≠ "
                        f"モデル patched_latent_dim={expected_patched}。"
                        f"このLoRAはモデル {version_label} と非互換です。"
                    )
        except Exception as e:
            return f"⚠️ {origin_label}アダプタ / shape 検証スキップ: {e}"

    return f"✅ {origin_label}アダプタ ({version_label} モデルと互換)"


def _ensure_default_model() -> None:
    if _scan_checkpoints():
        return
    print(f"[gradio] モデル未検出。{DEFAULT_HF_REPO} を自動ダウンロードします...", flush=True)
    try:
        from huggingface_hub import hf_hub_download
        safe_name = DEFAULT_HF_REPO.replace("/", "_")
        local_dir = CHECKPOINTS_DIR / safe_name
        local_dir.mkdir(parents=True, exist_ok=True)
        downloaded = hf_hub_download(
            repo_id=DEFAULT_HF_REPO,
            filename="model.safetensors",
            local_dir=str(local_dir),
        )
        print(f"[gradio] 自動ダウンロード完了: {downloaded}", flush=True)
    except Exception as e:
        print(f"[gradio] 自動ダウンロード失敗: {e}", flush=True)


def _download_from_hf(repo_id_input: str) -> tuple[gr.Dropdown, str]:
    repo_id = str(repo_id_input).strip()
    if not repo_id:
        return gr.Dropdown(), "エラー: repo_id を入力してください。"
    try:
        from huggingface_hub import hf_hub_download
        safe_name = repo_id.replace("/", "_")
        local_dir = CHECKPOINTS_DIR / safe_name
        local_dir.mkdir(parents=True, exist_ok=True)
        dest = local_dir / "model.safetensors"
        if dest.is_file():
            msg = f"スキップ: すでに存在します\n{dest}"
        else:
            downloaded = hf_hub_download(
                repo_id=repo_id, filename="model.safetensors", local_dir=str(local_dir),
            )
            dest = Path(downloaded)
            msg = f"ダウンロード完了:\n{dest}"
        checkpoints = _scan_checkpoints()
        return gr.Dropdown(choices=checkpoints, value=str(dest)), msg
    except Exception as e:
        checkpoints = _scan_checkpoints()
        return gr.Dropdown(choices=checkpoints), f"エラー: {e}"


# ─────────────────────────────────────────────────────────────────────────────
# 推論タブ ロジック
# ─────────────────────────────────────────────────────────────────────────────

def _default_model_device() -> str:
    return default_runtime_device()

def _precision_choices_for_device(device: str) -> list[str]:
    return list_available_runtime_precisions(device)

def _on_model_device_change(device: str) -> gr.Dropdown:
    choices = _precision_choices_for_device(device)
    return gr.Dropdown(choices=choices, value=choices[0])

def _on_codec_device_change(device: str) -> gr.Dropdown:
    choices = _precision_choices_for_device(device)
    return gr.Dropdown(choices=choices, value=choices[0])

def _parse_optional_float(raw: str | None, label: str) -> float | None:
    if raw is None: return None
    text = str(raw).strip()
    if text == "" or text.lower() == "none": return None
    try: return float(text)
    except ValueError as exc: raise ValueError(f"{label} must be a float or blank.") from exc

def _parse_optional_int(raw: str | None, label: str) -> int | None:
    if raw is None: return None
    text = str(raw).strip()
    if text == "" or text.lower() == "none": return None
    try: return int(text)
    except ValueError as exc: raise ValueError(f"{label} must be an int or blank.") from exc

def _format_timings(stage_timings: list[tuple[str, float]], total_to_decode: float) -> str:
    lines = [
        "[timing] ---- request ----",
        *[f"[timing] {name}: {sec * 1000.0:.1f} ms" for name, sec in stage_timings],
        f"[timing] total_to_decode: {total_to_decode:.3f} s",
    ]
    return "\n".join(lines)

def _resolve_checkpoint_path_infer(raw_checkpoint: str) -> str:
    checkpoint = str(raw_checkpoint).strip()
    if checkpoint == "":
        raise ValueError("チェックポイントが選択されていません。")
    suffix = Path(checkpoint).suffix.lower()
    if suffix in {".pt", ".safetensors"}:
        if not Path(checkpoint).is_file():
            raise FileNotFoundError(f"チェックポイントファイルが見つかりません: {checkpoint}")
        return checkpoint
    raise ValueError(f"サポートされていないファイル形式: {suffix}")

def _peek_latent_dim_from_checkpoint(checkpoint_path: str) -> int | None:
    """チェックポイントを軽量に読み取りlatent_dimを返す。失敗時はNone。"""
    try:
        from pathlib import Path as _Path
        import json as _json
        p = _Path(checkpoint_path)
        if p.suffix.lower() == ".safetensors":
            from safetensors import safe_open as _safe_open
            with _safe_open(str(p), framework="pt", device="cpu") as h:
                meta = h.metadata() or {}
            cfg_raw = meta.get("config_json")
            if cfg_raw:
                cfg = _json.loads(cfg_raw)
                return int(cfg["latent_dim"])
        else:
            import torch as _torch
            ckpt = _torch.load(str(p), map_location="cpu", weights_only=True)
            model_cfg = ckpt.get("model_config", {})
            if "latent_dim" in model_cfg:
                return int(model_cfg["latent_dim"])
    except Exception:
        pass
    return None

def _build_runtime_key(checkpoint, model_device, model_precision, codec_device, codec_precision, enable_watermark, lora_adapter="（なし）", codec_repo="Aratako/Semantic-DACVAE-Japanese-32dim"):
    checkpoint_path = _resolve_checkpoint_path_infer(checkpoint)
    lora_path = None
    if str(lora_adapter).strip() and str(lora_adapter).strip() != "（なし）":
        lp = Path(lora_adapter)
        if lp.is_dir() and (lp / "adapter_config.json").exists():
            lora_path = str(lp)
    return RuntimeKey(
        checkpoint=checkpoint_path,
        model_device=str(model_device),
        codec_repo=str(codec_repo),
        model_precision=str(model_precision),
        codec_device=str(codec_device),
        codec_precision=str(codec_precision),
        codec_deterministic_encode=True,
        codec_deterministic_decode=True,
        enable_watermark=bool(enable_watermark),
        compile_model=False,
        compile_dynamic=False,
        lora_path=lora_path,
    )

def _load_model(checkpoint, model_device, model_precision, codec_device, codec_precision, enable_watermark, lora_adapter="（なし）") -> tuple[str, str, bool]:
    """モデルをロードしてステータスと自動選択された codec_repo を返す。"""
    # ロード前にlatent_dimを先読みして正しいcodec_repoを決定する
    _raw_cp = _resolve_checkpoint_path_infer(str(checkpoint).strip())
    _ldim_pre = _peek_latent_dim_from_checkpoint(_raw_cp)
    _initial_codec_repo = _codec_repo_for_latent_dim(_ldim_pre) if _ldim_pre is not None else "Aratako/Semantic-DACVAE-Japanese-32dim"
    runtime_key = _build_runtime_key(checkpoint, model_device, model_precision, codec_device, codec_precision, enable_watermark, lora_adapter, codec_repo=_initial_codec_repo)
    _, reloaded = get_cached_runtime(runtime_key)
    status = "モデルを読み込みました" if reloaded else "モデルは既にロード済みです（再利用）"

    # ロード済みモデルからバージョン情報を取得
    info = _detect_model_version_from_runtime()
    version_str = ""
    auto_codec_repo = runtime_key.codec_repo
    if info is not None:
        version_label, codec_repo_used, ldim = info
        version_str = f"\nモデルバージョン: {version_label} (latent_dim={ldim})"
        auto_codec_repo = codec_repo_used

    lora_info = f"\nLoRAアダプタ: {runtime_key.lora_path}" if runtime_key.lora_path else ""
    voice_design_enabled = _runtime_uses_voice_design()
    vd_line = "\nvoice_design: enabled (caption conditioning)" if voice_design_enabled else "\nvoice_design: disabled"
    status_text = (
        f"{status}\n"
        f"checkpoint: {runtime_key.checkpoint}"
        f"{version_str}\n"
        f"model_device: {runtime_key.model_device} / {runtime_key.model_precision}\n"
        f"codec_device: {runtime_key.codec_device} / {runtime_key.codec_precision}\n"
        f"codec_repo: {auto_codec_repo}"
        f"{vd_line}"
        f"{lora_info}"
    )
    return status_text, auto_codec_repo, voice_design_enabled

def _clear_runtime_cache() -> str:
    clear_cached_runtime()
    return "モデルをメモリから解放しました"


def _clear_runtime_cache_ui():
    return (
        _clear_runtime_cache(),
        gr.update(visible=False, open=False),
        gr.update(visible=True, open=False),
    )

def _run_generation(
    checkpoint, model_device, model_precision, codec_device, codec_precision, enable_watermark,
    lora_adapter, lora_scale, lora_disabled_modules_raw,
    text, caption_text, uploaded_audio, spk_ref_latent_path,
    num_steps, seed_raw, cfg_guidance_mode, cfg_scale_text, cfg_scale_speaker,
    cfg_scale_caption,
    cfg_scale_raw, cfg_min_t, cfg_max_t, context_kv_cache,
    max_caption_len_raw,
    truncation_factor_raw, rescale_k_raw, rescale_sigma_raw,
    speaker_kv_scale_raw, speaker_kv_min_t_raw, speaker_kv_max_layers_raw,
    num_candidates: int = 1,
    multiline_mode: str = "デフォルト",
    silence_sec: float = 0.1,
) -> tuple[list[tuple[str, str]], str, str]:
    def stdout_log(msg: str) -> None:
        print(msg, flush=True)

    # ロード済みモデルの codec_repo を優先使用（v1/v2 自動対応）
    info = _detect_model_version_from_runtime()
    auto_codec_repo = (
        info[1] if info is not None
        else _codec_repo_for_latent_dim(32)  # フォールバック: v2
    )
    runtime_key = _build_runtime_key(
        checkpoint, model_device, model_precision,
        codec_device, codec_precision, enable_watermark,
        lora_adapter, codec_repo=auto_codec_repo,
    )
    if str(text).strip() == "":
        raise ValueError("テキストを入力してください。")

    cfg_scale        = _parse_optional_float(cfg_scale_raw, "cfg_scale")
    max_caption_len  = _parse_optional_int(max_caption_len_raw, "max_caption_len")
    truncation_factor= _parse_optional_float(truncation_factor_raw, "truncation_factor")
    rescale_k        = _parse_optional_float(rescale_k_raw, "rescale_k")
    rescale_sigma    = _parse_optional_float(rescale_sigma_raw, "rescale_sigma")
    speaker_kv_scale = _parse_optional_float(speaker_kv_scale_raw, "speaker_kv_scale")
    speaker_kv_min_t = _parse_optional_float(speaker_kv_min_t_raw, "speaker_kv_min_t")
    speaker_kv_max_layers = _parse_optional_int(speaker_kv_max_layers_raw, "speaker_kv_max_layers")
    seed = _parse_optional_int(seed_raw, "seed")

    # lora_disabled_modules: カンマ区切り文字列 → tuple[str, ...]
    _disabled_raw = str(lora_disabled_modules_raw).strip() if lora_disabled_modules_raw else ""
    lora_disabled_modules: tuple[str, ...] = (
        tuple(m.strip() for m in _disabled_raw.split(",") if m.strip())
        if _disabled_raw else ()
    )

    # 参照音声の優先順位: スピーカーライブラリ > 直接アップロード > no-reference
    _spk_pt = str(spk_ref_latent_path).strip() if spk_ref_latent_path else ""
    ref_latent_path: str | None = None
    ref_wav: str | None = None

    if _spk_pt and Path(_spk_pt).is_file():
        ref_latent_path = _spk_pt
        no_ref = False
    elif uploaded_audio and str(uploaded_audio).strip():
        ref_wav = str(uploaded_audio)
        no_ref = False
    else:
        no_ref = True

    # ── 改行分割モードの判定 ────────────────────────────────────────
    _multiline_mode = str(multiline_mode).strip()
    _use_multiline = _multiline_mode in (
        "改行ごとに連続生成で終了",
        "改行ごとに連続生成後に連結",
    )
    _use_concat = _multiline_mode == "改行ごとに連続生成後に連結"

    # 改行分割モード時は候補数を強制的に1に固定
    if _use_multiline:
        num_candidates = 1
    else:
        num_candidates = max(1, int(num_candidates))

    runtime, reloaded = get_cached_runtime(runtime_key)
    stdout_log(f"[gradio] runtime: {'reloaded' if reloaded else 'reused'}")
    use_voice_design = bool(getattr(runtime.model_cfg, "use_caption_condition", False))
    caption_value = str(caption_text).strip() if use_voice_design and caption_text is not None else ""

    # モデルバージョン情報をログに記録
    _ver_info = _detect_model_version_from_runtime()
    _ver_str = f"{_ver_info[0]} (latent_dim={_ver_info[2]})" if _ver_info else "unknown"
    stdout_log(f"[gradio] model_version: {_ver_str}")

    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

    gallery_items: list[tuple[str, str]] = []
    all_detail_lines: list[str] = [
        "runtime: reloaded" if reloaded else "runtime: reused",
        f"model_version: {_ver_str}",
        f"multiline_mode: {_multiline_mode}",
    ]
    if runtime_key.lora_path:
        all_detail_lines.append(f"lora: {Path(runtime_key.lora_path).name}")
    last_timing_text = ""

    # ── 共通のsynthesize呼び出しヘルパー ───────────────────────────
    def _synthesize_line(line_text: str, line_seed) -> object:
        return runtime.synthesize(
            SamplingRequest(
                text=str(line_text), ref_wav=ref_wav, ref_latent=ref_latent_path, no_ref=bool(no_ref),
                seconds=FIXED_SECONDS, max_ref_seconds=30.0, max_text_len=None,
                caption=caption_value or None,
                max_caption_len=max_caption_len,
                num_steps=int(num_steps),
                seed=line_seed,
                cfg_guidance_mode=str(cfg_guidance_mode),
                cfg_scale_text=float(cfg_scale_text),
                cfg_scale_caption=float(cfg_scale_caption),
                cfg_scale_speaker=float(cfg_scale_speaker),
                cfg_scale=cfg_scale, cfg_min_t=float(cfg_min_t), cfg_max_t=float(cfg_max_t),
                truncation_factor=truncation_factor, rescale_k=rescale_k, rescale_sigma=rescale_sigma,
                context_kv_cache=bool(context_kv_cache),
                speaker_kv_scale=speaker_kv_scale, speaker_kv_min_t=speaker_kv_min_t,
                speaker_kv_max_layers=speaker_kv_max_layers, trim_tail=True,
                lora_scale=float(lora_scale) if runtime_key.lora_path else 1.0,
                lora_disabled_modules=lora_disabled_modules if runtime_key.lora_path else (),
            ),
            log_fn=stdout_log,
        )

    # ── 改行分割モード ──────────────────────────────────────────────
    if _use_multiline:
        import torch as _torch

        lines = [ln for ln in str(text).split("\n") if ln.strip()]
        if not lines:
            raise ValueError("有効なテキスト行がありません。")

        all_detail_lines.append(f"分割行数: {len(lines)}")
        if _use_concat:
            all_detail_lines.append(f"無音区間: {silence_sec:.1f}秒")

        line_results = []
        for li, line in enumerate(lines):
            line_seed = None if seed is None else (seed + li)
            stdout_log(f"[gradio] generating line {li + 1}/{len(lines)}: {line[:40]!r} ...")
            result = _synthesize_line(line, line_seed)
            line_results.append(result)
            all_detail_lines.append(
                f"[行 {li + 1}] seed={result.used_seed}  text={line[:40]!r}"
            )
            for msg in result.messages:
                all_detail_lines.append(f"  {msg}")
            last_timing_text = _format_timings(result.stage_timings, result.total_to_decode)

        if _use_concat:
            # ── 連結モード: 無音区間を挟んで1ファイルに結合 ──────────
            sample_rate = line_results[0].sample_rate
            silence_samples = int(silence_sec * sample_rate)

            audio_segments = []
            for li, result in enumerate(line_results):
                audio_segments.append(result.audio.float())
                if li < len(line_results) - 1 and silence_samples > 0:
                    silence_tensor = _torch.zeros(
                        result.audio.shape[0] if result.audio.dim() > 1 else 1,
                        silence_samples,
                        dtype=_torch.float32,
                    ) if result.audio.dim() > 1 else _torch.zeros(
                        silence_samples,
                        dtype=_torch.float32,
                    )
                    audio_segments.append(silence_tensor)

            if line_results[0].audio.dim() > 1:
                concatenated = _torch.cat(audio_segments, dim=-1)
            else:
                concatenated = _torch.cat(audio_segments, dim=0)

            stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            out_path = save_wav(
                OUTPUTS_DIR / f"sample_{stamp}_concat.wav",
                concatenated,
                sample_rate,
            )
            caption = f"連結音声  {len(lines)}行  無音{silence_sec:.1f}秒"
            gallery_items.append((str(out_path), caption))
            all_detail_lines.append(f"連結保存: {out_path}")
            stdout_log(f"[gradio] concatenated saved: {out_path}")

        else:
            # ── 個別保存モード: 行ごとに別ファイルとして保存 ──────────
            stamp_base = datetime.now().strftime("%Y%m%d_%H%M%S")
            for li, result in enumerate(line_results):
                out_path = save_wav(
                    OUTPUTS_DIR / f"sample_{stamp_base}_line{li + 1}.wav",
                    result.audio.float(),
                    result.sample_rate,
                )
                caption = f"行 {li + 1}  seed={result.used_seed}"
                gallery_items.append((str(out_path), caption))
                all_detail_lines.append(f"[行 {li + 1}] saved={out_path}")
                stdout_log(f"[gradio] line {li + 1} saved: {out_path}")

    # ── 通常モード（デフォルト） ────────────────────────────────────
    else:
        for i in range(num_candidates):
            candidate_seed = None if seed is None else (seed + i)
            stdout_log(f"[gradio] generating candidate {i + 1}/{num_candidates} ...")

            result = _synthesize_line(str(text), candidate_seed)

            stamp    = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            out_path = save_wav(
                OUTPUTS_DIR / f"sample_{stamp}_c{i + 1}.wav",
                result.audio.float(),
                result.sample_rate,
            )
            caption = f"候補 {i + 1}  seed={result.used_seed}"
            gallery_items.append((str(out_path), caption))

            all_detail_lines.append(
                f"[候補 {i + 1}] seed={result.used_seed}  saved={out_path}"
            )
            for msg in result.messages:
                all_detail_lines.append(f"  {msg}")

            last_timing_text = _format_timings(result.stage_timings, result.total_to_decode)
            stdout_log(f"[gradio] candidate {i + 1} saved: {out_path}")

    detail_text = "\n".join(all_detail_lines)
    return gallery_items, detail_text, last_timing_text


# ─────────────────────────────────────────────────────────────────────────────
# Prepare Manifest タブ ロジック
# ─────────────────────────────────────────────────────────────────────────────

def _read_csv_headers(file_path: str) -> list[str]:
    import csv as _csv
    import json as _json

    p = Path(file_path.strip()) if file_path else None
    if not p:
        return []
    try:
        if p.is_dir():
            for name in ("metadata.csv", "metadata.jsonl", "metadata.json"):
                candidate = p / name
                if candidate.is_file():
                    return _read_csv_headers(str(candidate))
            return []

        if not p.is_file():
            return []

        suffix = p.suffix.lower()
        if suffix == ".csv":
            with open(p, encoding="utf-8", errors="replace", newline="") as f:
                headers = next(_csv.reader(f), [])
            return [h.strip() for h in headers if h.strip()]

        elif suffix in {".jsonl", ".json"}:
            lines = p.read_text(encoding="utf-8", errors="replace").splitlines()
            for line in lines:
                line = line.strip()
                if line:
                    obj = _json.loads(line)
                    return list(obj.keys())
    except Exception:
        pass
    return []


def _preview_dataset(dataset: str, split: str, audio_col: str, text_col: str) -> str:
    dataset = str(dataset).strip()
    if not dataset:
        return "データセットを入力してください。"

    p = Path(dataset)
    if p.is_file() and p.suffix in {".jsonl", ".json"}:
        try:
            lines = p.read_text(encoding="utf-8").strip().splitlines()
            count = len(lines)
            previews = []
            for line in lines[:3]:
                try:
                    obj = json.loads(line)
                    text = obj.get(text_col, "（テキストなし）")
                    previews.append(f"  • {str(text)[:80]}")
                except Exception:
                    previews.append("  • （パース失敗）")
            preview_str = "\n".join(previews)
            return f"✅ ローカルJSONL: {count} 件\n\n【サンプル（最大3件）】\n{preview_str}"
        except Exception as e:
            return f"❌ ファイル読み込みエラー: {e}"

    if p.is_dir():
        audio_exts = {".wav", ".mp3", ".flac", ".ogg", ".m4a"}
        files = [f for f in p.rglob("*") if f.suffix.lower() in audio_exts]
        count = len(files)
        previews = [f"  • {f.name}" for f in sorted(files)[:3]]
        preview_str = "\n".join(previews) if previews else "  （ファイルなし）"
        return f"✅ ローカルフォルダ: 音声ファイル {count} 件\n\n【サンプル（最大3件）】\n{preview_str}"

    try:
        from datasets import load_dataset_builder
        builder = load_dataset_builder(dataset)
        info = builder.info
        splits_info = info.splits or {}
        split_info = splits_info.get(split)
        count = split_info.num_examples if split_info else "不明"
        name = info.dataset_name or dataset
        desc = (info.description or "")[:120]
        return (
            f"✅ HuggingFace Dataset: {name}\n"
            f"スプリット '{split}': {count} 件\n\n"
            f"【説明】\n{desc}..."
        )
    except ImportError:
        return "⚠️ `datasets` ライブラリが未インストールです。\n`pip install datasets` を実行してください。"
    except Exception as e:
        return f"⚠️ データセット情報の取得に失敗しました:\n{e}"


def _build_manifest_command(
    data_source_mode,
    dataset,
    split,
    prepare_mode,
    audio_col, text_col, speaker_col, caption_col,
    output_manifest, latent_dir, device, codec_repo,
) -> list[str]:
    def _s(val, fallback=""):
        if val is None or isinstance(val, (dict, list)):
            return fallback
        s = str(val).strip()
        return s if s else fallback

    cmd = [sys.executable, str(BASE_DIR / "prepare_manifest.py")]

    if data_source_mode == "local_csv":
        csv_path = Path(_s(dataset))
        folder = str(csv_path.parent if csv_path.suffix.lower() == ".csv" else csv_path)
        cmd += ["--dataset", "audiofolder",
                "--data-files", folder,
                "--split", "train"]
    elif data_source_mode == "local_jsonl":
        cmd += ["--dataset", "json",
                "--data-files", _s(dataset),
                "--split", "train"]
    else:
        cmd += ["--dataset", _s(dataset),
                "--split",   _s(split, "train")]

    mode = str(prepare_mode).strip().lower()
    auto_codec_repo = str(codec_repo).strip()
    if mode in {"model_v2", "voice_design"}:
        auto_codec_repo = "Aratako/Semantic-DACVAE-Japanese-32dim"
    elif mode == "model_v1":
        auto_codec_repo = "facebook/dacvae-watermarked"
    if not auto_codec_repo:
        auto_codec_repo = DEFAULT_PREPARE_CODEC_REPO

    cmd += [
        "--audio-column",    _s(audio_col, "audio"),
        "--text-column",     _s(text_col, "text"),
        "--output-manifest", _s(output_manifest),
        "--latent-dir",      _s(latent_dir),
        "--device",          _s(device, "cpu"),
        "--codec-repo",      auto_codec_repo,
    ]
    if mode == "voice_design":
        cap = _s(caption_col)
        if cap:
            cmd += ["--caption-column", cap]
    else:
        spk = _s(speaker_col)
        if spk:
            cmd += ["--speaker-column", spk]
    return cmd


def _manifest_cmd_preview(
    data_source_mode, dataset, split, prepare_mode, audio_col, text_col, speaker_col, caption_col,
    output_manifest, latent_dir, device, codec_repo,
) -> str:
    return " ".join(_build_manifest_command(
        data_source_mode, dataset, split, prepare_mode, audio_col, text_col, speaker_col, caption_col,
        output_manifest, latent_dir, device, codec_repo,
    ))


def _run_manifest(
    data_source_mode, dataset, split, prepare_mode, audio_col, text_col, speaker_col, caption_col,
    output_manifest, latent_dir, device, codec_repo,
) -> tuple[str, str]:
    global _active_proc, _active_log_path
    cmd_list = _build_manifest_command(
        data_source_mode, dataset, split, prepare_mode, audio_col, text_col, speaker_col, caption_col,
        output_manifest, latent_dir, device, codec_repo,
    )
    cmd_str = " ".join(cmd_list)
    with _proc_lock:
        if _active_proc is not None and _active_proc.poll() is None:
            return "別のプロセスが実行中です。停止してから再実行してください。", cmd_str

        LOGS_DIR.mkdir(parents=True, exist_ok=True)
        stamp    = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_path = LOGS_DIR / f"manifest_{stamp}.log"
        _active_log_path = log_path

        env = os.environ.copy()
        env["PYTHONUTF8"] = "1"
        env["PYTHONIOENCODING"] = "utf-8"
        proc = subprocess.Popen(
            cmd_list,
            shell=False,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, bufsize=1, encoding="utf-8", errors="replace", env=env,
        )
        _active_proc = proc

    def _stream():
        with open(log_path, "w", encoding="utf-8") as f:
            for line in proc.stdout:
                f.write(line)
                f.flush()
        proc.wait()

    threading.Thread(target=_stream, daemon=True).start()
    return f"実行開始 (PID {proc.pid})\nログ: {log_path}", cmd_str


def _read_manifest_log() -> str:
    global _active_log_path, _active_proc
    if _active_log_path is None or not _active_log_path.exists():
        return ""
    text = _active_log_path.read_text(encoding="utf-8", errors="replace")
    if _active_proc is not None and _active_proc.poll() is not None:
        rc = _active_proc.returncode
        text += f"\n\n--- プロセス終了 (returncode={rc}) ---"
    return text


def _stop_process() -> str:
    global _active_proc
    with _proc_lock:
        if _active_proc is None or _active_proc.poll() is not None:
            return "実行中のプロセスはありません。"
        _active_proc.terminate()
        return f"プロセス (PID {_active_proc.pid}) に停止シグナルを送信しました。"


# ─────────────────────────────────────────────────────────────────────────────
# 学習タブ ロジック
# ─────────────────────────────────────────────────────────────────────────────

_TRAIN_LOG_PATH: Path | None = None
_TRAIN_PROC: subprocess.Popen | None = None
_TRAIN_LOG_LOCK = threading.Lock()
# ETA推定用: {"speed": steps/sec, "eta_sec": float, "step": int, "max_steps": int}
_TRAIN_ETA_INFO: dict = {}

_LORA_TRAIN_PROC: subprocess.Popen | None = None
_LORA_TRAIN_LOG_PATH: Path | None = None
_LORA_TRAIN_LOG_LOCK = threading.Lock()
# ETA推定用: {"speed": steps/sec, "eta_sec": float, "step": int, "max_steps": int}
_LORA_ETA_INFO: dict = {}


def _load_yaml_config(config_path: str) -> dict:
    p = Path(config_path)
    if not p.is_file():
        return {}
    with open(p, encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _save_yaml_config(config_path: str, data: dict) -> str:
    CONFIGS_DIR.mkdir(parents=True, exist_ok=True)
    p = Path(config_path)
    if not p.suffix:
        p = p.with_suffix(".yaml")
    if not p.is_absolute():
        p = CONFIGS_DIR / p.name
    with open(p, "w", encoding="utf-8") as f:
        yaml.dump(data, f, allow_unicode=True, default_flow_style=False, sort_keys=False)
    return f"保存しました: {p}"


def _config_from_ui(
    manifest, output_dir,
    batch_size, grad_accum, num_workers, persistent_workers, prefetch_factor,
    allow_tf32, compile_model, precision,
    optimizer, muon_momentum, learning_rate, weight_decay,
    adam_beta1, adam_beta2, adam_eps,
    lr_scheduler, warmup_steps, stable_steps, min_lr_scale,
    max_steps, max_text_len,
    text_dropout, speaker_dropout, timestep_stratified,
    max_latent_steps, fixed_target_latent_steps, fixed_target_full_mask,
    log_every, save_every,
    wandb_enabled, wandb_project, wandb_run_name,
    valid_ratio, valid_every,
    early_stopping, es_patience, es_min_delta,
    use_ema, ema_decay,
    seed,
) -> dict:
    return {
        "train": {
            "batch_size": int(batch_size),
            "gradient_accumulation_steps": int(grad_accum),
            "num_workers": int(num_workers),
            "dataloader_persistent_workers": bool(persistent_workers),
            "dataloader_prefetch_factor": int(prefetch_factor),
            "allow_tf32": bool(allow_tf32),
            "compile_model": bool(compile_model),
            "precision": str(precision),
            "optimizer": str(optimizer),
            "muon_momentum": float(muon_momentum),
            "learning_rate": float(learning_rate),
            "weight_decay": float(weight_decay),
            "adam_beta1": float(adam_beta1),
            "adam_beta2": float(adam_beta2),
            "adam_eps": float(adam_eps),
            "lr_scheduler": str(lr_scheduler),
            "warmup_steps": int(warmup_steps),
            "stable_steps": int(stable_steps),
            "min_lr_scale": float(min_lr_scale),
            "max_steps": int(max_steps),
            "max_text_len": int(max_text_len),
            "text_condition_dropout": float(text_dropout),
            "speaker_condition_dropout": float(speaker_dropout),
            "timestep_stratified": bool(timestep_stratified),
            "max_latent_steps": int(max_latent_steps),
            "fixed_target_latent_steps": int(fixed_target_latent_steps),
            "fixed_target_full_mask": bool(fixed_target_full_mask),
            "log_every": int(log_every),
            "save_every": int(save_every),
            "wandb_enabled": bool(wandb_enabled),
            "wandb_project": str(wandb_project) if wandb_project else None,
            "wandb_run_name": str(wandb_run_name) if wandb_run_name else None,
            "valid_ratio": float(valid_ratio),
            "valid_every": int(valid_every),
            "seed": int(seed),
        }
    }


def _build_train_command(
    manifest, output_dir, config_path,
    use_early_stopping, es_patience, es_min_delta,
    use_ema, ema_decay,
    resume_enabled, resume_checkpoint,
    save_mode,
    num_gpus,
    attention_backend="sdpa",
) -> list[str]:
    if int(num_gpus) > 1:
        cmd = [sys.executable, "-m", "torch.distributed.run",
               f"--nproc_per_node={num_gpus}", str(BASE_DIR / "train.py")]
    else:
        cmd = [sys.executable, str(BASE_DIR / "train.py")]

    cmd += [
        "--config", str(config_path),
        "--manifest", str(manifest),
        "--output-dir", str(output_dir),
    ]
    if use_early_stopping:
        cmd += ["--early-stopping",
                "--early-stopping-patience", str(es_patience),
                "--early-stopping-min-delta", str(es_min_delta)]
    if use_ema:
        cmd += ["--ema-decay", str(ema_decay)]

    if resume_enabled and str(resume_checkpoint).strip():
        cmd += ["--resume", str(resume_checkpoint)]

    if str(save_mode) in ("Fullのみ", "EMA + Full両方"):
        cmd += ["--save-full"]

    if str(attention_backend) and str(attention_backend) != "sdpa":
        cmd += ["--attention-backend", str(attention_backend)]

    return cmd


def _start_train(
    manifest, output_dir, config_path,
    use_early_stopping, es_patience, es_min_delta,
    use_ema, ema_decay, resume_enabled, resume_checkpoint, save_mode, num_gpus,
    attention_backend="sdpa",
    *ui_cfg_args,
) -> tuple[str, str]:
    global _TRAIN_LOG_PATH, _TRAIN_PROC, _TRAIN_ETA_INFO

    with _TRAIN_LOG_LOCK:
        if _TRAIN_PROC is not None and _TRAIN_PROC.poll() is None:
            return "学習が既に実行中です。停止してから再実行してください。", ""

    cfg_data = _config_from_ui(*ui_cfg_args)
    tmp_config = CONFIGS_DIR / "_train_tmp.yaml"
    CONFIGS_DIR.mkdir(parents=True, exist_ok=True)

    base_cfg = _load_yaml_config(str(config_path)) if Path(config_path).is_file() else {}
    base_cfg.update(cfg_data)
    with open(tmp_config, "w", encoding="utf-8") as f:
        yaml.dump(base_cfg, f, allow_unicode=True, default_flow_style=False, sort_keys=False)

    cmd_list = _build_train_command(
        manifest, output_dir, tmp_config,
        use_early_stopping, es_patience, es_min_delta,
        use_ema, ema_decay, resume_enabled, resume_checkpoint, save_mode, num_gpus,
        attention_backend,
    )
    cmd = " ".join(cmd_list)

    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    stamp    = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = LOGS_DIR / f"train_{stamp}.log"

    _TRAIN_ETA_INFO.clear()

    with _TRAIN_LOG_LOCK:
        _TRAIN_LOG_PATH = log_path
        env = os.environ.copy()
        env["PYTHONUTF8"] = "1"
        env["PYTHONIOENCODING"] = "utf-8"
        proc = subprocess.Popen(
            cmd_list, shell=False, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, bufsize=1, encoding="utf-8", errors="replace", env=env,
        )
        _TRAIN_PROC = proc

    import re as _re_train_eta
    _TRAIN_STEP_RE = _re_train_eta.compile(r"step=(\d+)")
    _TRAIN_SPEED_RE = _re_train_eta.compile(r"speed=([0-9.]+)steps/s")
    _TRAIN_ETA_RE = _re_train_eta.compile(r"eta=(.+)")

    def _train_eta_str_to_sec(eta_str: str) -> float:
        import re as _r
        s = eta_str.strip()
        total = 0.0
        m = _r.search(r"(\d+)時間", s)
        if m:
            total += int(m.group(1)) * 3600
        m = _r.search(r"(\d+)分", s)
        if m:
            total += int(m.group(1)) * 60
        m = _r.search(r"(\d+)秒", s)
        if m:
            total += int(m.group(1))
        return total

    def _stream():
        with open(log_path, "w", encoding="utf-8") as f:
            for line in proc.stdout:
                f.write(line)
                f.flush()
                # train.py のログ行から speed= / eta= を直接パース
                if "step=" in line and "loss=" in line:
                    m_step = _TRAIN_STEP_RE.search(line)
                    m_speed = _TRAIN_SPEED_RE.search(line)
                    m_eta = _TRAIN_ETA_RE.search(line)
                    if m_step:
                        current_step = int(m_step.group(1))
                        speed = float(m_speed.group(1)) if m_speed else 0.0
                        eta_sec = _train_eta_str_to_sec(m_eta.group(1)) if m_eta else 0.0
                        _TRAIN_ETA_INFO.update({
                            "step": current_step,
                            "speed": speed,
                            "eta_sec": eta_sec,
                        })
        proc.wait()
        _write_tensorboard_events(log_path)

    threading.Thread(target=_stream, daemon=True).start()
    return f"学習開始 (PID {proc.pid})\nログ: {log_path}", cmd


def _stop_train() -> str:
    global _TRAIN_PROC
    import signal as _signal
    with _TRAIN_LOG_LOCK:
        if _TRAIN_PROC is None or _TRAIN_PROC.poll() is not None:
            return "実行中の学習プロセスはありません。"
        pid = _TRAIN_PROC.pid
        proc = _TRAIN_PROC
    try:
        import os as _os
        _os.kill(pid, _signal.SIGINT)
    except (ProcessLookupError, PermissionError, OSError):
        pass
    def _deferred_kill():
        import time as _t
        _t.sleep(5)
        if proc.poll() is None:
            try:
                proc.kill()
            except Exception:
                pass
    import threading as _thr
    _thr.Thread(target=_deferred_kill, daemon=True).start()
    return f"学習プロセス (PID {pid}) に停止シグナルを送信しました（最大5秒でシャットダウン）。"


def _read_train_log() -> str:
    with _TRAIN_LOG_LOCK:
        path = _TRAIN_LOG_PATH
        proc = _TRAIN_PROC
    if path is None or not path.exists():
        return ""
    text = path.read_text(encoding="utf-8", errors="replace")
    if proc is not None and proc.poll() is not None:
        rc = proc.returncode
        text += f"\n\n--- 学習終了 (returncode={rc}) ---"
    lines = text.splitlines()
    if len(lines) > 200:
        text = f"... （先頭省略、末尾200行表示）\n" + "\n".join(lines[-200:])
    # ETA情報を末尾に付加（学習中のみ）
    eta = _TRAIN_ETA_INFO
    if eta and proc is not None and proc.poll() is None:
        step = eta.get("step", 0)
        speed = eta.get("speed", 0.0)
        eta_sec = int(eta.get("eta_sec", 0.0))
        h, rem = divmod(eta_sec, 3600)
        m, s = divmod(rem, 60)
        if h > 0:
            eta_str = f"{h}時間{m}分"
        elif m > 0:
            eta_str = f"{m}分{s}秒"
        else:
            eta_str = f"{s}秒"
        text += (
            f"\n\n--- ETA: 残り約 {eta_str}"
            f"  (step={step}, {speed:.3f} steps/sec) ---"
        )
    return text


def _parse_train_log_metrics():
    if not _PANDAS_AVAILABLE:
        return None
    with _TRAIN_LOG_LOCK:
        path = _TRAIN_LOG_PATH
    if path is None or not path.exists():
        return pd.DataFrame({"step": [], "loss": [], "lr": []})

    import re as _re_metrics
    # 各フィールドを個別に正規表現で抽出（speed= や eta= が混在しても壊れない）
    _RE_STEP = _re_metrics.compile(r"\bstep=(\d+)")
    _RE_LOSS = _re_metrics.compile(r"\bloss=([0-9.eE+\-]+)")
    _RE_LR   = _re_metrics.compile(r"\blr=([0-9.eE+\-]+)")

    rows = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if "step=" not in line or "loss=" not in line:
            continue
        # valid行・EarlyStopping行などのメトリクス以外の行を除外
        stripped = line.lstrip()
        if stripped.startswith("valid") or stripped.startswith("EarlyStopping"):
            continue
        try:
            m_step = _RE_STEP.search(line)
            m_loss = _RE_LOSS.search(line)
            m_lr   = _RE_LR.search(line)
            if not m_step or not m_loss:
                continue
            step = int(m_step.group(1))
            loss = float(m_loss.group(1))
            lr   = float(m_lr.group(1)) if m_lr else 0.0
            rows.append({"step": step, "loss": loss, "lr": lr})
        except (ValueError, AttributeError):
            continue
    if not rows:
        return pd.DataFrame({"step": [], "loss": [], "lr": []})
    return pd.DataFrame(rows)


def _write_tensorboard_events(log_path: Path) -> None:
    try:
        from torch.utils.tensorboard import SummaryWriter
        tb_dir = LOGS_DIR / "tensorboard" / log_path.stem
        tb_dir.mkdir(parents=True, exist_ok=True)
        writer = SummaryWriter(log_dir=str(tb_dir))
        df = _parse_train_log_metrics()
        for _, row in df.iterrows():
            writer.add_scalar("train/loss", row["loss"], int(row["step"]))
            writer.add_scalar("train/lr",   row["lr"],   int(row["step"]))
        writer.close()
        print(f"[gradio] TensorBoardイベント保存: {tb_dir}", flush=True)
    except ImportError:
        pass
    finally:
        df = _parse_train_log_metrics()
        if not df.empty:
            csv_path = LOGS_DIR / f"{log_path.stem}_metrics.csv"
            df.to_csv(csv_path, index=False)
            print(f"[gradio] メトリクスCSV保存: {csv_path}", flush=True)


# ─────────────────────────────────────────────────────────────────────────────
# モデルマージ タブ ロジック
# ─────────────────────────────────────────────────────────────────────────────

def _merge_scan() -> list[str]:
    return scan_checkpoints_for_merge()


def _run_merge_ui(
    path_a, path_b,
    method,
    alpha,
    lambda_a, lambda_b,
    base_path_ta,
    use_partial,
    text_method, text_alpha, text_lam_a, text_lam_b,
    speaker_method, speaker_alpha, speaker_lam_a, speaker_lam_b,
    diffusion_method, diffusion_alpha, diffusion_lam_a, diffusion_lam_b,
    io_method, io_alpha, io_lam_a, io_lam_b,
    use_lora,
    lora_base, lora_donor, lora_scale,
    lora_grp_text, lora_grp_speaker, lora_grp_diffusion, lora_grp_io,
    output_format,
    output_dir,
) -> str:
    def _norm(a, b):
        t = float(a) + float(b)
        return (float(a) / t, float(b) / t) if t > 0 else (0.5, 0.5)

    group_methods = None
    if use_partial:
        def _group_cfg(meth, al, la, lb):
            if meth == "task_arithmetic":
                na, nb = _norm(la, lb)
                return {"method": meth, "lambda_a": na, "lambda_b": nb}
            return {"method": meth, "alpha": float(al)}
        group_methods = {
            "text":          _group_cfg(text_method,     text_alpha,     text_lam_a,     text_lam_b),
            "speaker":       _group_cfg(speaker_method,  speaker_alpha,  speaker_lam_a,  speaker_lam_b),
            "diffusion_core":_group_cfg(diffusion_method,diffusion_alpha,diffusion_lam_a,diffusion_lam_b),
            "io":            _group_cfg(io_method,       io_alpha,       io_lam_a,       io_lam_b),
        }

    lora_targets = []
    if lora_grp_text:      lora_targets.append("text")
    if lora_grp_speaker:   lora_targets.append("speaker")
    if lora_grp_diffusion: lora_targets.append("diffusion_core")
    if lora_grp_io:        lora_targets.append("io")

    la_norm, lb_norm = _norm(lambda_a, lambda_b)

    success, message = run_merge(
        path_a=str(path_a),
        path_b=str(path_b),
        method=str(method),
        alpha=float(alpha),
        lambda_a=la_norm,
        lambda_b=lb_norm,
        base_path=str(base_path_ta) if base_path_ta else None,
        use_partial=bool(use_partial),
        group_methods=group_methods,
        use_lora_inject=bool(use_lora),
        lora_base_path=str(lora_base) if lora_base else None,
        lora_donor_path=str(lora_donor) if lora_donor else None,
        lora_scale=float(lora_scale),
        lora_target_groups=lora_targets if lora_targets else None,
        output_format="safetensors" if output_format == ".safetensors" else "pt",
        output_dir=str(output_dir) if output_dir else None,
    )
    return message

# ─────────────────────────────────────────────────────────────────────────────
# Convert タブ ロジック
# ─────────────────────────────────────────────────────────────────────────────

def _run_convert(input_pt: str) -> str:
    if not str(input_pt).strip():
        return "エラー: 変換対象の .pt ファイルを選択してください。"
    p = Path(input_pt)
    if not p.is_file():
        return f"エラー: ファイルが見つかりません: {p}"
    cmd = f"{sys.executable} {BASE_DIR / 'convert_checkpoint_to_safetensors.py'} {p}"
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=300)
        out = result.stdout + result.stderr
        if result.returncode == 0:
            safetensors_path = p.with_suffix(".safetensors")
            return f"変換完了: {safetensors_path}\n\n{out}"
        else:
            return f"変換失敗 (returncode={result.returncode}):\n{out}"
    except subprocess.TimeoutExpired:
        return "エラー: タイムアウト (300秒超過)"
    except Exception as e:
        return f"エラー: {e}"


# ─────────────────────────────────────────────────────────────────────────────
# Dataset Tools タブ ロジック
# ─────────────────────────────────────────────────────────────────────────────

_DS_LOG_PATH: Path | None = None
_DS_PROC: subprocess.Popen | None = None
_DS_LOG_LOCK = threading.Lock()


def _build_dataset_command(
    mode: str,
    input_path: str,
    slice_output: str,
    min_sec: float,
    max_sec: float,
    threshold: float,
    min_silence_ms: int,
    speech_pad_ms: int,
    target_sr_enabled: bool,
    target_sr: int,
    recursive_slice: bool,
    caption_input: str,
    manifest_output_dir: str,
    manifest_filename: str,
    output_format: str,
    whisper_model: str,
    language: str,
    speaker_id: str,
    speaker_field_mode: str,
    recursive_caption: bool,
    device: str,
    model_cache_dir: str,
) -> list[str]:
    fmt_ext = "csv" if output_format == "CSV" else "jsonl"
    manifest_path = str(Path(manifest_output_dir) / f"{manifest_filename}.{fmt_ext}")

    if mode == "スライスのみ":
        cmd = [sys.executable, str(DATASET_TOOLS), "slice",
               "--input", input_path,
               "--output", slice_output,
               "--min-sec", str(min_sec),
               "--max-sec", str(max_sec),
               "--threshold", str(threshold),
               "--min-silence-ms", str(int(min_silence_ms)),
               "--speech-pad-ms",  str(int(speech_pad_ms)),
               ]
        if target_sr_enabled and int(target_sr) > 0:
            cmd += ["--target-sr", str(int(target_sr))]
        if str(device).strip() and device != "自動":
            cmd += ["--device", str(device).strip()]
        if recursive_slice:
            cmd += ["--recursive"]

    elif mode == "キャプションのみ":
        lang = "" if language in ("自動検出", "auto", "") else language
        cmd = [sys.executable, str(DATASET_TOOLS), "caption",
               "--input", caption_input,
               "--output-manifest", manifest_path,
               "--format", fmt_ext,
               "--model", whisper_model,
               ]
        if lang:
            cmd += ["--language", lang]
        speaker_value = str(speaker_id).strip()
        is_voice_design = str(speaker_field_mode).strip() == "caption"
        if is_voice_design:
            cmd += ["--voice-design"]
            if speaker_value:
                cmd += ["--voice-design-caption", speaker_value]
        elif speaker_value:
            cmd += ["--speaker-id", speaker_value]
        if recursive_caption:
            cmd += ["--recursive"]
        if str(device).strip() and device != "自動":
            cmd += ["--device", str(device).strip()]
        if str(model_cache_dir).strip():
            cmd += ["--model-cache-dir", str(model_cache_dir).strip()]

    else:
        lang = "" if language in ("自動検出", "auto", "") else language
        cmd = [sys.executable, str(DATASET_TOOLS), "pipeline",
               "--input", input_path,
               "--slice-output", slice_output,
               "--output-manifest", manifest_path,
               "--format", fmt_ext,
               "--min-sec", str(min_sec),
               "--max-sec", str(max_sec),
               "--threshold", str(threshold),
               "--min-silence-ms", str(int(min_silence_ms)),
               "--speech-pad-ms",  str(int(speech_pad_ms)),
               "--model", whisper_model,
               ]
        if target_sr_enabled and int(target_sr) > 0:
            cmd += ["--target-sr", str(int(target_sr))]
        if lang:
            cmd += ["--language", lang]
        speaker_value = str(speaker_id).strip()
        is_voice_design = str(speaker_field_mode).strip() == "caption"
        if is_voice_design:
            cmd += ["--voice-design"]
            if speaker_value:
                cmd += ["--voice-design-caption", speaker_value]
        elif speaker_value:
            cmd += ["--speaker-id", speaker_value]
        if str(device).strip() and device != "自動":
            cmd += ["--device", str(device).strip()]
        if str(model_cache_dir).strip():
            cmd += ["--model-cache-dir", str(model_cache_dir).strip()]

    return cmd


def _start_dataset_job(*args) -> tuple[str, str]:
    global _DS_LOG_PATH, _DS_PROC

    with _DS_LOG_LOCK:
        if _DS_PROC is not None and _DS_PROC.poll() is None:
            return "別のジョブが実行中です。停止してから再実行してください。", ""

    cmd_list = _build_dataset_command(*args)
    cmd_str  = " ".join(cmd_list)

    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    stamp    = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = LOGS_DIR / f"dataset_{stamp}.log"

    with _DS_LOG_LOCK:
        _DS_LOG_PATH = log_path
        env = os.environ.copy()
        env["PYTHONUTF8"] = "1"
        env["PYTHONIOENCODING"] = "utf-8"
        proc = subprocess.Popen(
            cmd_list, shell=False, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, bufsize=1, encoding="utf-8", errors="replace", env=env,
        )
        _DS_PROC = proc

    def _stream():
        with open(log_path, "w", encoding="utf-8") as f:
            for line in proc.stdout:
                f.write(line)
                f.flush()
        proc.wait()

    threading.Thread(target=_stream, daemon=True).start()
    return f"実行開始 (PID {proc.pid})\nログ: {log_path}", cmd_str


def _stop_dataset_job() -> str:
    global _DS_PROC
    with _DS_LOG_LOCK:
        if _DS_PROC is None or _DS_PROC.poll() is not None:
            return "実行中のジョブはありません。"
        _DS_PROC.terminate()
        return f"ジョブ (PID {_DS_PROC.pid}) に停止シグナルを送信しました。"


def _read_dataset_log() -> str:
    with _DS_LOG_LOCK:
        path = _DS_LOG_PATH
        proc = _DS_PROC
    if path is None or not path.exists():
        return ""
    text = path.read_text(encoding="utf-8", errors="replace")
    if proc is not None and proc.poll() is not None:
        rc = proc.returncode
        text += f"\n\n--- 完了 (returncode={rc}) ---"
    lines = text.splitlines()
    if len(lines) > 300:
        text = "... （先頭省略、末尾300行表示）\n" + "\n".join(lines[-300:])
    return text


# ─────────────────────────────────────────────────────────────────────────────
# 絵文字キャプション ロジック
# ─────────────────────────────────────────────────────────────────────────────

_EMOJI_API_KEYS = {
    "LM Studio（ローカル）": "lm_studio",
    "Groq": "groq",
    "OpenAI（ChatGPT）": "openai",
    "Together AI": "together",
}
_EMOJI_DEFAULT_MODELS = {
    "lm_studio": "",
    "groq": "llama-3.3-70b-versatile",
    "openai": "gpt-4o-mini",
    "together": "meta-llama/Llama-3.3-70B-Instruct-Turbo",
}


def _append_emoji_to_ds_log(log_path: Path, message: str) -> None:
    try:
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(message + "\n")
            f.flush()
    except Exception:
        pass


def _run_emoji_caption_inline(
    csv_path: str,
    wav_dir: str,
    api_label: str,
    api_key: str = "",
    voice_design: bool = False,
) -> None:
    global _DS_PROC, _DS_LOG_PATH

    api_key_str = _EMOJI_API_KEYS.get(api_label, "lm_studio")

    cmd = [
        sys.executable, str(DATASET_TOOLS), "emoji_caption",
        "--csv",     str(csv_path).strip(),
        "--wav-dir", str(wav_dir).strip(),
        "--api",     api_key_str,
    ]
    if api_key_str != "lm_studio" and str(api_key).strip():
        cmd += ["--api-key", str(api_key).strip()]
    if voice_design:
        cmd += ["--voice-design"]

    env = os.environ.copy()
    env["PYTHONUTF8"] = "1"
    env["PYTHONIOENCODING"] = "utf-8"

    with _DS_LOG_LOCK:
        log_path = _DS_LOG_PATH
        _append_emoji_to_ds_log(log_path, f"\n{'='*60}")
        _append_emoji_to_ds_log(log_path, "🎭 絵文字キャプション開始")
        _append_emoji_to_ds_log(log_path, f"{'='*60}")
        proc = subprocess.Popen(
            cmd, shell=False,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, bufsize=1, encoding="utf-8", errors="replace", env=env,
        )
        _DS_PROC = proc

    def _stream():
        with open(log_path, "a", encoding="utf-8") as f:
            for line in proc.stdout:
                f.write(line)
                f.flush()
        proc.wait()
        rc = proc.returncode
        _append_emoji_to_ds_log(log_path, f"\n--- 絵文字キャプション完了 (returncode={rc}) ---")

    threading.Thread(target=_stream, daemon=True).start()


# ─────────────────────────────────────────────────────────────────────────────
# LoRA 学習タブ ロジック
# ─────────────────────────────────────────────────────────────────────────────

def _build_lora_train_command(
    base_model, manifest, output_dir, run_name,
    lora_rank, lora_alpha, lora_dropout, target_modules,
    save_mode, attention_backend,
    use_early_stopping, es_patience, es_min_delta,
    use_ema, ema_decay,
    resume_enabled, resume_lora_path,
    batch_size, grad_accum, lr, optimizer, lr_scheduler, warmup_steps,
    max_steps, save_every, log_every,
    valid_ratio, valid_every,
    wandb_enabled, wandb_project, wandb_run_name,
    seed,
) -> list[str]:
    cmd = [sys.executable, str(BASE_DIR / "lora_train.py")]
    cmd += ["--base-model", str(base_model)]
    cmd += ["--manifest", str(manifest)]

    _run_name = str(run_name).strip() if str(run_name).strip() else ""
    if _run_name:
        cmd += ["--run-name", _run_name]

    if str(output_dir).strip():
        cmd += ["--output-dir", str(output_dir).strip()]

    cmd += [
        "--lora-rank", str(int(lora_rank)),
        "--lora-alpha", str(float(lora_alpha)),
        "--lora-dropout", str(float(lora_dropout)),
        "--target-modules", str(target_modules).strip(),
        "--batch-size", str(int(batch_size)),
        "--gradient-accumulation-steps", str(int(grad_accum)),
        "--lr", str(float(lr)),
        "--optimizer", str(optimizer),
        "--lr-scheduler", str(lr_scheduler),
        "--warmup-steps", str(int(warmup_steps)),
        "--max-steps", str(int(max_steps)),
        "--save-every", str(int(save_every)),
        "--log-every", str(int(log_every)),
        "--seed", str(int(seed)),
    ]

    if str(attention_backend) != "sdpa":
        cmd += ["--attention-backend", str(attention_backend)]

    if str(save_mode) == "EMA + Full両方":
        cmd += ["--save-full"]

    if use_ema:
        cmd += ["--ema-decay", str(float(ema_decay))]

    if float(valid_ratio) > 0.0:
        cmd += ["--valid-ratio", str(float(valid_ratio))]
        if int(valid_every) > 0:
            cmd += ["--valid-every", str(int(valid_every))]

    if use_early_stopping and float(valid_ratio) > 0.0:
        cmd += [
            "--early-stopping",
            "--early-stopping-patience", str(int(es_patience)),
            "--early-stopping-min-delta", str(float(es_min_delta)),
        ]

    if wandb_enabled:
        cmd += ["--wandb"]
        if str(wandb_project).strip():
            cmd += ["--wandb-project", str(wandb_project).strip()]
        if str(wandb_run_name).strip():
            cmd += ["--wandb-run-name", str(wandb_run_name).strip()]

    if resume_enabled and str(resume_lora_path).strip():
        cmd += ["--resume-lora", str(resume_lora_path).strip()]

    return cmd


def _start_lora_train(*args) -> tuple[str, str]:
    global _LORA_TRAIN_PROC, _LORA_TRAIN_LOG_PATH, _LORA_ETA_INFO

    with _LORA_TRAIN_LOG_LOCK:
        if _LORA_TRAIN_PROC is not None and _LORA_TRAIN_PROC.poll() is None:
            return "LoRA学習が既に実行中です。停止してから再実行してください。", ""

    cmd_list = _build_lora_train_command(*args)
    cmd_str = " ".join(cmd_list)

    # _build_lora_train_command のシグネチャ順 (0-indexed):
    # 0:base_model 1:manifest 2:output_dir 3:run_name 4:lora_rank 5:lora_alpha
    # 6:lora_dropout 7:target_modules 8:save_mode 9:attention_backend
    # 10:use_early_stopping 11:es_patience 12:es_min_delta 13:use_ema 14:ema_decay
    # 15:resume_enabled 16:resume_lora_path 17:batch_size 18:grad_accum 19:lr
    # 20:optimizer 21:lr_scheduler 22:warmup_steps 23:max_steps ...
    try:
        _max_steps = int(args[23])
    except (IndexError, ValueError, TypeError):
        _max_steps = 0

    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = LOGS_DIR / f"lora_train_{stamp}.log"

    _LORA_ETA_INFO.clear()

    with _LORA_TRAIN_LOG_LOCK:
        _LORA_TRAIN_LOG_PATH = log_path
        env = os.environ.copy()
        env["PYTHONUTF8"] = "1"
        env["PYTHONIOENCODING"] = "utf-8"
        env["PYTHONUNBUFFERED"] = "1"
        proc = subprocess.Popen(
            cmd_list, shell=False, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, bufsize=1, encoding="utf-8", errors="replace", env=env,
        )
        _LORA_TRAIN_PROC = proc

    import re as _re_eta
    _STEP_RE = _re_eta.compile(r"step=(\d+)")
    import re as _re_speed
    _SPEED_RE = _re_speed.compile(r"speed=([0-9.]+)steps/s")
    _ETA_STR_RE = _re_speed.compile(r"eta=(.+)")

    def _eta_str_to_sec(eta_str: str) -> float:
        """lora_train.py が出力する eta= 文字列を秒数に変換する。
        フォーマット: 〇時間〇分 / 〇分〇秒 / 〇秒
        """
        import re as _r
        s = eta_str.strip()
        total = 0.0
        m = _r.search(r"(\d+)時間", s)
        if m:
            total += int(m.group(1)) * 3600
        m = _r.search(r"(\d+)分", s)
        if m:
            total += int(m.group(1)) * 60
        m = _r.search(r"(\d+)秒", s)
        if m:
            total += int(m.group(1))
        return total

    def _stream():
        with open(log_path, "w", encoding="utf-8") as f:
            for line in proc.stdout:
                f.write(line)
                f.flush()
                # ステップ行からstep/speed/etaを直接パースして更新
                # lora_train.py が計算した値をそのまま使うことで二重計算のズレを排除
                if "step=" in line and "loss=" in line:
                    m_step = _STEP_RE.search(line)
                    m_speed = _SPEED_RE.search(line)
                    m_eta = _ETA_STR_RE.search(line)
                    if m_step and _max_steps > 0:
                        current_step = int(m_step.group(1))
                        speed = float(m_speed.group(1)) if m_speed else 0.0
                        eta_sec = _eta_str_to_sec(m_eta.group(1)) if m_eta else 0.0
                        _LORA_ETA_INFO.update({
                            "step": current_step,
                            "max_steps": _max_steps,
                            "speed": speed,
                            "eta_sec": eta_sec,
                        })
        proc.wait()

    threading.Thread(target=_stream, daemon=True).start()

    warning = ""
    if bool(args[10]) and float(args[26]) <= 0.0:
        warning = "\n⚠️ Early Stopping は valid_ratio=0 のため無効化されました。"
    return f"LoRA学習開始 (PID {proc.pid})\nログ: {log_path}{warning}", cmd_str


def _stop_lora_train() -> str:
    global _LORA_TRAIN_PROC
    import signal as _signal
    with _LORA_TRAIN_LOG_LOCK:
        if _LORA_TRAIN_PROC is None or _LORA_TRAIN_PROC.poll() is not None:
            return "実行中のLoRA学習プロセスはありません。"
        pid = _LORA_TRAIN_PROC.pid
        proc = _LORA_TRAIN_PROC
    # SIGINTでグレースフルシャットダウンを試みる（DataLoaderが正常終了できる）
    try:
        import os as _os
        _os.kill(pid, _signal.SIGINT)
    except (ProcessLookupError, PermissionError, OSError):
        pass
    # 最大5秒待機し、まだ生きていれば強制終了
    def _deferred_kill():
        import time as _t
        _t.sleep(5)
        if proc.poll() is None:
            try:
                proc.kill()
            except Exception:
                pass
    import threading as _thr
    _thr.Thread(target=_deferred_kill, daemon=True).start()
    return f"LoRA学習プロセス (PID {pid}) に停止シグナルを送信しました（最大5秒でシャットダウン）。"


def _read_lora_train_log() -> str:
    with _LORA_TRAIN_LOG_LOCK:
        path = _LORA_TRAIN_LOG_PATH
        proc = _LORA_TRAIN_PROC
    if path is None or not path.exists():
        return ""
    text = path.read_text(encoding="utf-8", errors="replace")
    if proc is not None and proc.poll() is not None:
        text += f"\n\n--- LoRA学習終了 (returncode={proc.returncode}) ---"
    lines = text.splitlines()
    if len(lines) > 200:
        text = "... （先頭省略、末尾200行表示）\n" + "\n".join(lines[-200:])
    # ETA情報を末尾に付加（学習中のみ）
    eta = _LORA_ETA_INFO
    if eta and proc is not None and proc.poll() is None:
        step = eta.get("step", 0)
        max_steps = eta.get("max_steps", 0)
        speed = eta.get("speed", 0.0)
        eta_sec = int(eta.get("eta_sec", 0.0))
        h, rem = divmod(eta_sec, 3600)
        m, s = divmod(rem, 60)
        if h > 0:
            eta_str = f"{h}時間{m}分"
        elif m > 0:
            eta_str = f"{m}分{s}秒"
        else:
            eta_str = f"{s}秒"
        progress_pct = (step / max_steps * 100) if max_steps > 0 else 0.0
        text += (
            f"\n\n--- ETA: 残り約 {eta_str}"
            f"  ({step}/{max_steps} steps, {progress_pct:.1f}%,"
            f" {speed:.3f} steps/sec) ---"
        )
    return text


def _run_lora_convert(input_full_dir: str, force: bool = False) -> str:
    if not str(input_full_dir).strip():
        return "エラー: 変換対象の _full フォルダを選択してください。"
    p = Path(input_full_dir)
    if not p.is_dir():
        return f"エラー: フォルダが存在しません: {p}"
    cmd = [sys.executable, str(BASE_DIR / "convert_lora_checkpoint.py"), str(p)]
    if force:
        cmd += ["--force"]
    result = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")
    out = (result.stdout + result.stderr).strip()
    if result.returncode == 0:
        return f"変換完了\n\n{out}"
    else:
        return f"変換失敗 (returncode={result.returncode}):\n{out}"


# ─────────────────────────────────────────────────────────────────────────────
# UI 構築
# ─────────────────────────────────────────────────────────────────────────────

# ─────────────────────────────────────────────────────────────────────────────
# ダークモード切り替え用 CSS / JS（モジュールトップレベル）
# ─────────────────────────────────────────────────────────────────────────────
_DARK_CSS = """
/* ── ダークモード変数 ── */
.dark-mode {
    --bg-primary:    #1a1a1a;
    --bg-secondary:  #2a2a2a;
    --bg-tertiary:   #333333;
    --text-primary:  #e8e8e8;
    --text-secondary:#aaaaaa;
    --border-color:  #444444;
    --accent:        #7c9cbf;
}
.dark-mode .gradio-container,
.dark-mode body {
    background-color: var(--bg-primary) !important;
    color: var(--text-primary) !important;
}
.dark-mode .gr-box,
.dark-mode .gr-panel,
.dark-mode .gr-form,
.dark-mode .gr-block,
.dark-mode .block,
.dark-mode .panel,
.dark-mode fieldset {
    background-color: var(--bg-secondary) !important;
    border-color: var(--border-color) !important;
}
.dark-mode .tab-nav button,
.dark-mode .tabs > .tab-nav > button {
    background-color: var(--bg-tertiary) !important;
    color: var(--text-primary) !important;
    border-color: var(--border-color) !important;
}
.dark-mode .tab-nav button.selected {
    background-color: var(--accent) !important;
    color: #ffffff !important;
}
.dark-mode input,
.dark-mode textarea,
.dark-mode select,
.dark-mode .gr-textbox textarea,
.dark-mode .gr-textbox input {
    background-color: var(--bg-tertiary) !important;
    color: var(--text-primary) !important;
    border-color: var(--border-color) !important;
}
.dark-mode label,
.dark-mode .gr-label,
.dark-mode .label-wrap span {
    color: var(--text-secondary) !important;
}
.dark-mode .gr-button-secondary,
.dark-mode button.secondary {
    background-color: var(--bg-tertiary) !important;
    color: var(--text-primary) !important;
    border-color: var(--border-color) !important;
}
.dark-mode .gr-accordion,
.dark-mode details,
.dark-mode details summary {
    background-color: var(--bg-secondary) !important;
    color: var(--text-primary) !important;
    border-color: var(--border-color) !important;
}
.dark-mode .gr-slider input[type=range] {
    accent-color: var(--accent) !important;
}
.dark-mode .gr-dropdown,
.dark-mode .wrap {
    background-color: var(--bg-tertiary) !important;
    color: var(--text-primary) !important;
    border-color: var(--border-color) !important;
}
#dark-mode-toggle-btn {
    position: fixed;
    top: 12px;
    right: 18px;
    z-index: 9999;
    padding: 5px 14px;
    border-radius: 20px;
    border: 1px solid #888;
    background: #f0f0f0;
    color: #333;
    font-size: 13px;
    cursor: pointer;
    transition: background 0.2s, color 0.2s;
    box-shadow: 0 2px 6px rgba(0,0,0,0.15);
}
.dark-mode #dark-mode-toggle-btn {
    background: #444 !important;
    color: #eee !important;
    border-color: #666 !important;
}
"""

_DARK_JS = """
function() {
    if (!document.getElementById('dark-mode-toggle-btn')) {
        var btn = document.createElement('button');
        btn.id = 'dark-mode-toggle-btn';
        btn.textContent = '🌙 ダーク';
        document.body.appendChild(btn);
        if (localStorage.getItem('irodori_dark') === '1') {
            document.body.classList.add('dark-mode');
            btn.textContent = '☀️ ライト';
        }
        btn.addEventListener('click', function() {
            var isDark = document.body.classList.toggle('dark-mode');
            btn.textContent = isDark ? '☀️ ライト' : '🌙 ダーク';
            localStorage.setItem('irodori_dark', isDark ? '1' : '0');
        });
    }
}
"""

def build_ui() -> gr.Blocks:
    _ensure_default_model()

    initial_checkpoints      = _scan_checkpoints()
    default_checkpoint       = initial_checkpoints[-1] if initial_checkpoints else ""
    default_model_device     = _default_model_device()
    default_codec_device     = _default_model_device()
    device_choices           = list_available_runtime_devices()
    model_precision_choices  = _precision_choices_for_device(default_model_device)
    codec_precision_choices  = _precision_choices_for_device(default_codec_device)
    initial_configs          = _scan_configs()
    default_config           = next((c for c in initial_configs if DEFAULT_CONFIG in c), initial_configs[-1] if initial_configs else "")
    initial_manifests        = _scan_manifests()
    initial_train_ckpts      = _scan_train_checkpoints()

    default_cfg = _load_yaml_config(default_config).get("train", {}) if default_config else {}

    def _v(key, fallback=None):
        return default_cfg.get(key, fallback)

    # ダークモードCSS/JSはモジュールトップレベル (_DARK_CSS/_DARK_JS) で定義

    with gr.Blocks(title="Irodori-TTS GUI") as demo:
        gr.Markdown("# 🎙️ Irodori-TTS GUI")

        with gr.Tabs():

            # ═══════════════════════════════════════════════════════════════
            # タブ1: 推論
            # ═══════════════════════════════════════════════════════════════
            with gr.Tab("🔊 推論"):
                gr.Markdown("## モデル設定")

                with gr.Accordion("⬇️ HuggingFace からモデルをダウンロード", open=not bool(initial_checkpoints)):
                    with gr.Row():
                        hf_repo_id = gr.Textbox(
                            label="HuggingFace repo id", value=DEFAULT_HF_REPO,
                            placeholder="org/model-name", scale=4,
                        )
                        hf_dl_btn = gr.Button("ダウンロード", scale=1)
                    hf_dl_status = gr.Textbox(label="ダウンロード状況", interactive=False, lines=2)

                with gr.Row():
                    infer_checkpoint = gr.Dropdown(
                        label="チェックポイント (.pt / .safetensors)",
                        choices=initial_checkpoints,
                        value=default_checkpoint or None,
                        scale=4, allow_custom_value=False,
                    )
                    infer_refresh_btn = gr.Button("🔄 更新", scale=1)

                with gr.Row():
                    infer_lora_adapter = gr.Dropdown(
                        label="LoRAアダプタ（なし=ベースモデルのみ）",
                        choices=["（なし）"] + _scan_lora_adapters(),
                        value="（なし）",
                        scale=4, allow_custom_value=False,
                    )
                    infer_lora_refresh_btn = gr.Button("🔄", scale=1)
                infer_lora_compat_status = gr.Textbox(
                    label="LoRA互換性チェック",
                    value="",
                    interactive=False,
                    lines=1,
                    visible=False,
                )
                infer_lora_scale = gr.Slider(
                    label="LoRAスケール（0.0=LoRA無効 / 1.0=通常 / >1.0=強調）",
                    minimum=0.0, maximum=2.0, value=1.0, step=0.05, visible=False,
                )
                infer_lora_disabled_modules = gr.Textbox(
                    label="LoRA無効モジュール（カンマ区切り、空=全て有効）",
                    value="",
                    placeholder="例: blocks.0.attention, blocks.1.attention",
                    visible=False,
                    info="指定したモジュールのLoRAをスケール0で無効化します。",
                )

                with gr.Row():
                    model_device = gr.Dropdown(label="モデルデバイス", choices=device_choices, value=default_model_device, scale=1)
                    model_precision = gr.Dropdown(label="モデル精度", choices=model_precision_choices, value=model_precision_choices[0], scale=1)
                    codec_device = gr.Dropdown(label="コーデックデバイス", choices=device_choices, value=default_codec_device, scale=1)
                    codec_precision = gr.Dropdown(label="コーデック精度", choices=codec_precision_choices, value=codec_precision_choices[0], scale=1)
                    enable_watermark = gr.Checkbox(label="ウォーターマーク", value=False, scale=1)

                # codec_repo 選択（モデルロード後に自動更新）
                infer_codec_repo = gr.Dropdown(
                    label="コーデックリポジトリ（モデル読み込み時に自動設定）",
                    choices=PREPARE_CODEC_REPO_CHOICES,
                    value="Aratako/Semantic-DACVAE-Japanese-32dim",
                    info="v2(dim32) / v1(dim128) — モデル読み込み後に自動切替されます。",
                    interactive=True,
                )

                with gr.Row():
                    load_model_btn  = gr.Button("📥 モデル読み込み", variant="secondary")
                    unload_model_btn= gr.Button("🗑️ メモリ解放",    variant="secondary")
                model_status = gr.Textbox(label="モデルステータス", interactive=False, lines=4)

                gr.Markdown("## 音声生成")

                with gr.Accordion("🎤 参照音声（省略するとno-referenceモードで生成）", open=False) as infer_ref_accordion:
                    # スピーカーライブラリ選択時のref_latentパスを保持する隠しフィールド
                    spk_ref_latent_path = gr.Textbox(visible=False, value="")

                    with gr.Tabs():
                        # ── タブ1: 直接アップロード ──────────────────────
                        with gr.Tab("🎙️ 直接アップロード"):
                            infer_audio = gr.Audio(label="参照音声", type="filepath")
                            # アップロード時はスピーカーライブラリの選択をクリア
                            infer_audio.change(
                                lambda v: "",
                                inputs=[infer_audio],
                                outputs=[spk_ref_latent_path],
                            )

                        # ── タブ2: スピーカーライブラリ ──────────────────
                        with gr.Tab("🎭 スピーカーライブラリ"):
                            with gr.Row():
                                spk_select = gr.Dropdown(
                                    label="キャラクター",
                                    choices=_scan_speakers(),
                                    value="（使用しない）",
                                    scale=4,
                                )
                                spk_lib_refresh = gr.Button("🔄", scale=1)
                            spk_info = gr.Textbox(
                                label="登録情報", interactive=False, lines=2
                            )

                            def _on_spk_select(name):
                                if not name or name == "（使用しない）":
                                    return "", ""
                                pt = SPEAKERS_DIR / name / "ref.pt"
                                profile_p = SPEAKERS_DIR / name / "profile.json"
                                pt_str = str(pt) if pt.exists() else ""
                                info = ""
                                if profile_p.exists():
                                    try:
                                        p = json.loads(profile_p.read_text(encoding="utf-8"))
                                        info = (
                                            f"duration: {p.get('duration_sec', '?')}秒  "
                                            f"latent: {p.get('latent_shape', '?')}"
                                        )
                                    except Exception:
                                        pass
                                return pt_str, info

                            spk_select.change(
                                _on_spk_select,
                                inputs=[spk_select],
                                outputs=[spk_ref_latent_path, spk_info],
                            )
                            spk_lib_refresh.click(
                                lambda: gr.Dropdown(choices=_scan_speakers()),
                                outputs=[spk_select],
                            )

                        # ── タブ3: スピーカー登録 ─────────────────────────
                        with gr.Tab("➕ スピーカー登録"):
                            gr.Markdown(
                                "参照WAVをDACVAEエンコードして `speakers/{名前}/` に\n"
                                "`ref.wav` / `ref.pt` / `profile.json` の3ファイルを生成します。\n\n"
                                "> **事前条件**: 上部の「📥 モデル読み込み」を完了してから実行してください。"
                            )
                            spk_reg_name = gr.Textbox(
                                label="キャラクター名",
                                placeholder="alice",
                            )
                            spk_reg_wav = gr.Audio(
                                label="参照WAV（5〜30秒推奨、雑音なし）",
                                type="filepath",
                            )
                            spk_reg_btn = gr.Button(
                                "💾 登録", variant="primary"
                            )
                            spk_reg_status = gr.Textbox(
                                label="結果", interactive=False, lines=4
                            )

                            spk_reg_btn.click(
                                _run_create_speaker,
                                inputs=[
                                    spk_reg_name, spk_reg_wav,
                                    infer_checkpoint,
                                    model_device, model_precision,
                                    codec_device, codec_precision,
                                ],
                                outputs=[spk_reg_status],
                            )
                            # 登録完了後にライブラリDropdownを自動更新
                            spk_reg_btn.click(
                                lambda: gr.Dropdown(choices=_scan_speakers()),
                                outputs=[spk_select],
                            )

                # ── 感情スタイルプリセット ──────────────────────────────
                with gr.Accordion("🎭 感情スタイル", open=True):
                    gr.Markdown(
                        "プリセットボタンを押すと、下の各パラメータが自動設定されます。"
                        "その後スライダーを手動調整することも可能です。"
                    )
                    with gr.Row():
                        preset_normal  = gr.Button("😐 ノーマル",   variant="secondary", scale=1)
                        preset_strong  = gr.Button("😤 力強く",     variant="secondary", scale=1)
                        preset_calm    = gr.Button("😌 おとなしく", variant="secondary", scale=1)
                        preset_bright  = gr.Button("😊 明るく",     variant="secondary", scale=1)
                        preset_whisper = gr.Button("🤫 ひそやかに", variant="secondary", scale=1)

                    gr.Markdown("##### スタイル調整パラメータ")
                    with gr.Row():
                        style_cfg_text = gr.Slider(
                            label="テキスト表現力（低：棒読み ↔ 高：抑揚強調）",
                            minimum=0.0, maximum=10.0, value=3.0, step=0.1, scale=2,
                        )
                        style_cfg_speaker = gr.Slider(
                            label="感情の強さ（低：ニュートラル ↔ 高：スタイル強調）",
                            minimum=0.0, maximum=10.0, value=5.0, step=0.1, scale=2,
                        )
                    with gr.Row():
                        style_kv_scale = gr.Slider(
                            label="話者密着度（1.0=標準、高いほど参照音声の特徴を強く反映）",
                            minimum=1.0, maximum=4.0, value=1.0, step=0.1, scale=2,
                        )
                        style_trunc = gr.Slider(
                            label="表現の振れ幅（低：安定・平坦 ↔ 高：ダイナミック・不安定）",
                            minimum=0.7, maximum=1.0, value=1.0, step=0.01, scale=2,
                        )

                # ── サンプリング設定 ────────────────────────────────────
                with gr.Accordion("🎛️ サンプリング設定", open=True):
                    with gr.Row():
                        num_steps = gr.Slider(
                            label="ステップ数（多いほど品質向上・低速）",
                            minimum=1, maximum=120, value=40, step=1,
                        )
                        seed_raw = gr.Textbox(label="シード（空白=ランダム）", value="")

                # ── CFG設定 ─────────────────────────────────────────────
                with gr.Accordion("⚙️ CFG設定", open=False):
                    gr.Markdown(
                        "**CFG（Classifier-Free Guidance）** はモデルが条件（テキスト・話者）をどれだけ"
                        "強く守るかを制御します。値が大きいほど条件に忠実になりますが、高すぎると"
                        "不自然になる場合があります。"
                    )
                    with gr.Row():
                        cfg_guidance_mode = gr.Dropdown(
                            label="ガイダンスモード",
                            choices=["independent", "joint", "alternating"],
                            value="independent",
                            info="independent=高品質・低速（推奨） / joint=バランス / alternating=高速",
                        )
                        cfg_scale_text = gr.Slider(
                            label="テキストCFG強度",
                            minimum=0.0, maximum=10.0, value=3.0, step=0.1,
                            info="テキスト内容への忠実度。感情スタイルと連動します。",
                        )
                        cfg_scale_speaker = gr.Slider(
                            label="話者CFG強度",
                            minimum=0.0, maximum=10.0, value=5.0, step=0.1,
                            info="参照音声の声質への忠実度。感情スタイルと連動します。",
                        )
                # ── 詳細設定 ────────────────────────────────────────────
                with gr.Accordion("🔬 詳細設定（上級者向け）", open=False):
                    gr.Markdown(
                        "通常は変更不要です。動作確認・実験用途向けの項目です。"
                    )
                    cfg_scale_raw = gr.Textbox(
                        label="CFGスケール一括上書き（テキスト・話者を同値に設定。空=無効）",
                        value="",
                    )
                    with gr.Row():
                        cfg_min_t = gr.Number(
                            label="CFG適用開始タイムステップ",
                            value=0.5,
                            info="拡散過程のどの時点からCFGを適用するか（0.0〜1.0）",
                        )
                        cfg_max_t = gr.Number(
                            label="CFG適用終了タイムステップ",
                            value=1.0,
                            info="拡散過程のどの時点までCFGを適用するか（0.0〜1.0）",
                        )
                        context_kv_cache = gr.Checkbox(
                            label="コンテキストKVキャッシュ（推論高速化）",
                            value=True,
                            info="テキスト・話者のKV射影を事前計算してステップ間で再利用します",
                        )
                    with gr.Row():
                        rescale_k_raw = gr.Textbox(
                            label="スコア再スケールk（空=無効）",
                            value="",
                            info="Xu et al. 2025 の時間的スコアリスケール係数k",
                        )
                        rescale_sigma_raw = gr.Textbox(
                            label="スコア再スケールsigma（空=無効）",
                            value="",
                            info="rescale_k と合わせて設定します",
                        )
                    with gr.Row():
                        speaker_kv_min_t_raw = gr.Textbox(
                            label="話者KVスケール適用閾値（デフォルト0.9）",
                            value="0.9",
                            info="この値以上のタイムステップでのみ話者KV強調を適用します",
                        )
                        speaker_kv_max_layers_raw = gr.Textbox(
                            label="話者KVスケール適用レイヤー数上限（空=全レイヤー）",
                            value="",
                            info="拡散ブロックの先頭N層にのみ話者KV強調を適用します",
                        )

                    # 感情スタイルパネルと詳細設定の内部変数をつなぐ隠しフィールド
                    # truncation_factor と speaker_kv_scale はスタイルパネルから制御するため
                    # 詳細設定には表示しない（テキストボックスは後処理用に内部保持）
                    truncation_factor_raw = gr.Textbox(visible=False, value="")
                    speaker_kv_scale_raw  = gr.Textbox(visible=False, value="")

                # ── 改行分割生成オプション ───────────────────────────────
                with gr.Accordion("📝 改行分割生成オプション", open=True):
                    gr.Markdown(
                        "プロンプトの改行ごとに生成を区切り、連続生成・連結するオプションです。\n"
                        "- **デフォルト**: テキスト全体を1回で生成します（従来の動作）\n"
                        "- **改行ごとに連続生成で終了**: 改行ごとに個別ファイルをギャラリーに出力します\n"
                        "- **改行ごとに連続生成後に連結**: 改行ごとに生成後、無音区間を挟んで1ファイルに連結します\n\n"
                        "> 改行分割モード選択時は「生成候補数」が自動的に1に固定されます。"
                    )
                    multiline_mode = gr.Dropdown(
                        label="改行分割生成モード",
                        choices=[
                            "デフォルト",
                            "改行ごとに連続生成で終了",
                            "改行ごとに連続生成後に連結",
                        ],
                        value="デフォルト",
                        info="デフォルト=通常生成 / 連続生成=行ごとに個別出力 / 連結=無音区間を挟んで1ファイルに結合",
                    )
                    silence_sec = gr.Slider(
                        label="無音区間（秒）",
                        minimum=0.1, maximum=3.0, value=0.1, step=0.1,
                        interactive=False,
                        info="連結モード時に行間に挿入する無音の長さ（連結モード以外では無効）",
                    )

                    def _on_multiline_mode_change(mode: str):
                        is_concat = mode == "改行ごとに連続生成後に連結"
                        return gr.Slider(interactive=is_concat)

                    multiline_mode.change(
                        _on_multiline_mode_change,
                        inputs=[multiline_mode],
                        outputs=[silence_sec],
                    )

                # ── 候補数設定 ──────────────────────────────────────────
                num_candidates = gr.Slider(
                    label="生成候補数 (Num Candidates)",
                    minimum=1, maximum=8, value=1, step=1,
                    info="1回の生成で作成する候補音声の数。シード指定時は seed, seed+1, seed+2... が使われます。改行分割モード時は1固定。",
                )

                # ── テキスト入力（生成ボタン直上） ─────────────────────
                infer_text = gr.Textbox(label="テキスト（合成したい文章）", lines=4)
                with gr.Accordion("🎨 Caption (Voice Design)", open=False, visible=False) as caption_vd_accordion:
                    caption_input_vd = gr.Textbox(
                        label="Caption (Voice Design)",
                        value="",
                        lines=2,
                        placeholder="e.g. calm, bright, energetic, whispering",
                        info="VoiceDesign モデル読み込み時のみ表示されます。",
                    )
                    with gr.Row():
                        cfg_scale_caption = gr.Slider(
                            label="Caption CFG Scale",
                            minimum=0.0, maximum=10.0, value=3.0, step=0.1,
                        )
                        max_caption_len_raw = gr.Textbox(
                            label="Max Caption Len (optional)",
                            value="",
                        )

                generate_btn = gr.Button("🎵 生成", variant="primary", size="lg")

                # ── 候補リスト（最大8候補）──────────────────────────
                _MAX_CANDIDATES = 8
                gr.Markdown("### 生成結果")
                gr.Markdown(
                    "各候補の再生ボタンで試聴できます。"
                    "ファイルは `gradio_outputs/` フォルダに保存されています。"
                )

                # 候補ごとに (ラベルTextbox + Audioプレイヤー) を最大8セット用意し
                # 生成数に応じて visible を切り替える
                _cand_labels = []
                _cand_audios = []
                for _ci in range(_MAX_CANDIDATES):
                    with gr.Row(visible=False) as _row:
                        pass
                    _lbl = gr.Textbox(
                        value="",
                        label=f"候補 {_ci + 1}",
                        interactive=False,
                        visible=False,
                        max_lines=1,
                        show_label=True,
                        scale=1,
                    )
                    _aud = gr.Audio(
                        value=None,
                        label=f"候補 {_ci + 1} の音声",
                        type="filepath",
                        interactive=False,
                        visible=False,
                        scale=3,
                    )
                    _cand_labels.append(_lbl)
                    _cand_audios.append(_aud)

                out_log    = gr.Textbox(label="実行ログ", lines=6)
                out_timing = gr.Textbox(label="タイミング情報", lines=6)

                # _run_generation の戻り値 (gallery_items, detail, timing) を
                # 候補ラベル×8 + 候補Audio×8 + log + timing に展開するラッパー
                def _run_generation_ui(*args):
                    gallery_items, detail_text, timing_text = _run_generation(*args)
                    label_updates = []
                    audio_updates = []
                    for i in range(_MAX_CANDIDATES):
                        if i < len(gallery_items):
                            path, caption = gallery_items[i]
                            label_updates.append(gr.update(value=caption, visible=True))
                            audio_updates.append(gr.update(value=path, visible=True))
                        else:
                            label_updates.append(gr.update(value="", visible=False))
                            audio_updates.append(gr.update(value=None, visible=False))
                    return label_updates + audio_updates + [detail_text, timing_text]

                # ── プリセット定義 ──────────────────────────────────────
                # (cfg_text, cfg_speaker, kv_scale, trunc)
                _PRESETS = {
                    "normal":  (3.0, 5.0, 1.0, 1.0),
                    "strong":  (5.0, 7.0, 1.8, 1.0),
                    "calm":    (2.0, 3.0, 1.0, 0.80),
                    "bright":  (4.5, 6.0, 1.5, 0.95),
                    "whisper": (2.0, 2.0, 1.0, 0.75),
                }

                def _apply_preset(name):
                    ct, cs, kv, tr = _PRESETS[name]
                    # kv_scale: 1.0のときは空文字（無効）、それ以外は文字列で返す
                    kv_str = "" if kv == 1.0 else str(kv)
                    # trunc: 1.0のときは空文字（無効）、それ以外は文字列で返す
                    tr_str = "" if tr == 1.0 else str(tr)
                    return ct, cs, ct, cs, kv, tr, kv_str, tr_str

                # プリセットが更新するコンポーネントのリスト
                _preset_outputs = [
                    style_cfg_text, style_cfg_speaker,
                    cfg_scale_text, cfg_scale_speaker,
                    style_kv_scale, style_trunc,
                    speaker_kv_scale_raw, truncation_factor_raw,
                ]

                preset_normal.click(
                    lambda: _apply_preset("normal"), outputs=_preset_outputs,
                )
                preset_strong.click(
                    lambda: _apply_preset("strong"), outputs=_preset_outputs,
                )
                preset_calm.click(
                    lambda: _apply_preset("calm"), outputs=_preset_outputs,
                )
                preset_bright.click(
                    lambda: _apply_preset("bright"), outputs=_preset_outputs,
                )
                preset_whisper.click(
                    lambda: _apply_preset("whisper"), outputs=_preset_outputs,
                )

                # スタイルスライダー変更 → CFGスライダーと内部テキストへ反映
                def _sync_style_to_cfg(ct, cs, kv, tr):
                    kv_str = "" if kv <= 1.0 else str(round(kv, 2))
                    tr_str = "" if tr >= 1.0 else str(round(tr, 2))
                    return ct, cs, kv_str, tr_str

                style_cfg_text.change(
                    lambda v, cs, kv, tr: _sync_style_to_cfg(v, cs, kv, tr),
                    inputs=[style_cfg_text, style_cfg_speaker, style_kv_scale, style_trunc],
                    outputs=[cfg_scale_text, cfg_scale_speaker, speaker_kv_scale_raw, truncation_factor_raw],
                )
                style_cfg_speaker.change(
                    lambda ct, v, kv, tr: _sync_style_to_cfg(ct, v, kv, tr),
                    inputs=[style_cfg_text, style_cfg_speaker, style_kv_scale, style_trunc],
                    outputs=[cfg_scale_text, cfg_scale_speaker, speaker_kv_scale_raw, truncation_factor_raw],
                )
                style_kv_scale.change(
                    lambda ct, cs, v, tr: _sync_style_to_cfg(ct, cs, v, tr),
                    inputs=[style_cfg_text, style_cfg_speaker, style_kv_scale, style_trunc],
                    outputs=[cfg_scale_text, cfg_scale_speaker, speaker_kv_scale_raw, truncation_factor_raw],
                )
                style_trunc.change(
                    lambda ct, cs, kv, v: _sync_style_to_cfg(ct, cs, kv, v),
                    inputs=[style_cfg_text, style_cfg_speaker, style_kv_scale, style_trunc],
                    outputs=[cfg_scale_text, cfg_scale_speaker, speaker_kv_scale_raw, truncation_factor_raw],
                )

                # CFGスライダー変更 → スタイルスライダーへ反映（逆同期）
                cfg_scale_text.change(
                    lambda v: v, inputs=[cfg_scale_text], outputs=[style_cfg_text],
                )
                cfg_scale_speaker.change(
                    lambda v: v, inputs=[cfg_scale_speaker], outputs=[style_cfg_speaker],
                )

                hf_dl_btn.click(_download_from_hf, inputs=[hf_repo_id], outputs=[infer_checkpoint, hf_dl_status])
                infer_refresh_btn.click(
                    lambda: gr.Dropdown(choices=_scan_checkpoints(), value=(_scan_checkpoints() or [None])[-1]),
                    outputs=[infer_checkpoint],
                )
                infer_lora_refresh_btn.click(
                    lambda: gr.Dropdown(choices=["（なし）"] + _scan_lora_adapters()),
                    outputs=[infer_lora_adapter],
                )

                def _on_lora_adapter_change(v):
                    is_active = str(v).strip() not in ("", "（なし）")
                    compat_msg = _validate_lora_compat_ui(v) if is_active else ""
                    return (
                        gr.Slider(visible=is_active),
                        gr.Textbox(visible=is_active),
                        gr.Textbox(value=compat_msg, visible=is_active),
                    )

                infer_lora_adapter.change(
                    _on_lora_adapter_change,
                    inputs=[infer_lora_adapter],
                    outputs=[infer_lora_scale, infer_lora_disabled_modules, infer_lora_compat_status],
                )

                model_device.change(_on_model_device_change, inputs=[model_device], outputs=[model_precision])
                codec_device.change(_on_codec_device_change, inputs=[codec_device], outputs=[codec_precision])

                def _load_model_ui(checkpoint, model_device, model_precision, codec_device, codec_precision, enable_watermark, lora_adapter, cur_lora_adapter):
                    """モデルロード後にステータス・codec_repo・LoRA互換チェックを更新。"""
                    status_text, auto_codec, voice_design_enabled = _load_model(
                        checkpoint, model_device, model_precision,
                        codec_device, codec_precision, enable_watermark, lora_adapter,
                    )
                    # LoRAアダプタが選択中なら再チェック
                    compat_msg = _validate_lora_compat_ui(cur_lora_adapter) if (
                        str(cur_lora_adapter).strip() not in ("", "（なし）")
                    ) else ""
                    return (
                        status_text,
                        gr.Dropdown(value=auto_codec),
                        gr.Textbox(value=compat_msg),
                        gr.update(visible=voice_design_enabled, open=voice_design_enabled),
                        gr.update(visible=not voice_design_enabled, open=False),
                    )

                load_model_btn.click(
                    _load_model_ui,
                    inputs=[infer_checkpoint, model_device, model_precision,
                            codec_device, codec_precision, enable_watermark,
                            infer_lora_adapter, infer_lora_adapter],
                    outputs=[model_status, infer_codec_repo, infer_lora_compat_status, caption_vd_accordion, infer_ref_accordion],
                )
                unload_model_btn.click(_clear_runtime_cache_ui, outputs=[model_status, caption_vd_accordion, infer_ref_accordion])
                _ui_outputs = _cand_labels + _cand_audios + [out_log, out_timing]
                generate_btn.click(_run_generation_ui,
                    inputs=[
                        infer_checkpoint, model_device, model_precision, codec_device, codec_precision, enable_watermark,
                        infer_lora_adapter, infer_lora_scale, infer_lora_disabled_modules,
                        infer_text, caption_input_vd, infer_audio, spk_ref_latent_path,
                        num_steps, seed_raw, cfg_guidance_mode,
                        cfg_scale_text, cfg_scale_speaker, cfg_scale_caption, cfg_scale_raw, cfg_min_t, cfg_max_t,
                        context_kv_cache, max_caption_len_raw, truncation_factor_raw, rescale_k_raw, rescale_sigma_raw,
                        speaker_kv_scale_raw, speaker_kv_min_t_raw, speaker_kv_max_layers_raw,
                        num_candidates,
                        multiline_mode,
                        silence_sec,
                    ],
                    outputs=_ui_outputs,
                )

            # ═══════════════════════════════════════════════════════════════
            # タブ2: Prepare Manifest
            # ═══════════════════════════════════════════════════════════════
            with gr.Tab("📂 Prepare Manifest"):
                gr.Markdown(
                    "## データセット前処理\n"
                    "音声データセットをDACVAEラテントに変換し、学習用マニフェスト（JSONL）を生成します。\n\n"
                    "> **ローカルCSV/JSONLを使う場合**：Dataset作成タブで生成した `metadata.csv` または `metadata.jsonl` を直接指定できます。"
                )

                pm_data_source = gr.Radio(
                    label="データソース",
                    choices=["ローカルCSV", "ローカルJSONL", "HuggingFaceデータセット"],
                    value="ローカルCSV",
                )
                pm_prepare_mode = gr.Dropdown(
                    label="モード",
                    choices=["model_v1", "model_v2", "voice_design"],
                    value="model_v2",
                    info="model_v1=dim128, model_v2=dim32, voice_design=dim32 + caption列",
                )

                with gr.Group() as pm_local_group:
                    pm_dataset = gr.Textbox(
                        label="ファイルパス（CSV または JSONL）",
                        value=str(DEFAULT_DATASET_DIR / "metadata.csv"),
                        placeholder=str(DEFAULT_DATASET_DIR / "metadata.csv"),
                        info="Dataset作成タブで生成したmetadata.csv / metadata.jsonlを指定",
                    )

                with gr.Group() as pm_hf_group:
                    with gr.Row():
                        pm_hf_name = gr.Textbox(
                            label="HuggingFaceデータセット名",
                            placeholder="例: myorg/my_dataset",
                            visible=False,
                        )
                        pm_split = gr.Textbox(
                            label="スプリット名",
                            value="train",
                            visible=False,
                        )

                with gr.Accordion("📋 列名設定", open=True):
                    gr.Markdown(
                        "**ローカルCSV / JSONL（audiofolder形式）の列名**\n"
                        "- 音声列: `audio` 固定（CSV・JSONL の `file_name` 列を audiofolder が読み込み時に `audio` へ自動置換）\n"
                        "- テキスト列・話者ID列: ファイルパス入力後にヘッダーを自動取得してドロップダウンに反映します。\n\n"
                        "HuggingFace データセットの場合はそれぞれの列名を手動で入力してください。"
                    )
                    with gr.Row():
                        pm_audio_col   = gr.Dropdown(label="音声列名", value="audio",
                                                     choices=["audio"], allow_custom_value=True)
                        pm_text_col    = gr.Dropdown(label="テキスト列名", value="text",
                                                     choices=["text"], allow_custom_value=True)
                        pm_speaker_col = gr.Dropdown(label="話者ID列名（省略可）", value="",
                                                     choices=[""], allow_custom_value=True)
                        pm_caption_col = gr.Dropdown(label="Caption列名（Voice Design）", value="caption",
                                                     choices=["caption"], allow_custom_value=True, visible=False)
                    pm_col_status = gr.Textbox(label="列名取得状況", interactive=False, lines=1)

                with gr.Row():
                    pm_output_manifest = gr.Textbox(
                        label="出力マニフェストパス（.jsonl）",
                        value=str(BASE_DIR / "data" / "train_manifest.jsonl"),
                    )
                    pm_latent_dir = gr.Textbox(
                        label="ラテント保存フォルダ",
                        value=str(BASE_DIR / "data" / "latents"),
                    )
                    pm_device = gr.Dropdown(
                        label="使用デバイス", choices=device_choices, value=default_model_device,
                    )

                pm_codec_repo = gr.Dropdown(
                    label="DACVAE codec",
                    choices=PREPARE_CODEC_REPO_CHOICES,
                    value=DEFAULT_PREPARE_CODEC_REPO,
                    allow_custom_value=True,
                    info="v2(dim32) default / switch to v1(dim128) as needed.",
                )

                def _on_pm_source_change(mode):
                    is_hf = mode == "HuggingFaceデータセット"
                    return (
                        gr.update(visible=not is_hf),
                        gr.update(visible=is_hf),
                        gr.update(visible=is_hf),
                        gr.update(value="audio"),
                    )
                pm_data_source.change(
                    _on_pm_source_change, inputs=[pm_data_source],
                    outputs=[pm_dataset, pm_hf_name, pm_split, pm_audio_col],
                )

                def _on_pm_prepare_mode_change(mode: str):
                    mode_key = str(mode).strip().lower()
                    is_voice = mode_key == "voice_design"
                    codec_repo = (
                        "facebook/dacvae-watermarked"
                        if mode_key == "model_v1"
                        else "Aratako/Semantic-DACVAE-Japanese-32dim"
                    )
                    status = (
                        "voice_design: caption列を使用 / codec dim32"
                        if is_voice
                        else ("model_v1: speaker_id列を使用 / codec dim128" if mode_key == "model_v1"
                              else "model_v2: speaker_id列を使用 / codec dim32")
                    )
                    return (
                        gr.update(visible=not is_voice, interactive=not is_voice, value="" if is_voice else None),
                        gr.update(visible=is_voice),
                        gr.update(value=codec_repo),
                        status,
                    )

                pm_prepare_mode.change(
                    _on_pm_prepare_mode_change,
                    inputs=[pm_prepare_mode],
                    outputs=[pm_speaker_col, pm_caption_col, pm_codec_repo, pm_col_status],
                )

                def _auto_fill_columns(file_path: str, mode: str):
                    if mode == "HuggingFaceデータセット":
                        return gr.update(), gr.update(), gr.update(), gr.update(), "HFデータセット: 列名を手動で入力してください。"
                    headers = _read_csv_headers(file_path)
                    if not headers:
                        return gr.update(), gr.update(), gr.update(), gr.update(), "⚠️ 列名を取得できませんでした。ファイルパスを確認してください。"
                    audio_choices = ["audio"] + [h for h in headers if h not in {"audio", "file_name"}]
                    exclude = {"file_name", "audio", "speaker_id", "speaker"}
                    text_choices = [h for h in headers if h not in {"file_name", "audio"}]
                    text_guess = next((h for h in headers if h not in exclude), text_choices[0] if text_choices else "text")
                    spk_choices = [""] + [h for h in headers if h not in {"file_name", "audio", text_guess}]
                    cap_choices = [""] + [h for h in headers if h not in {"file_name", "audio", text_guess}]
                    spk_default = "speaker_id" if "speaker_id" in headers else ""
                    cap_default = "caption" if "caption" in headers else ""
                    status = f"✅ 列名を取得しました: {headers}"
                    return (
                        gr.update(choices=audio_choices, value="audio"),
                        gr.update(choices=text_choices, value=text_guess),
                        gr.update(choices=spk_choices, value=spk_default),
                        gr.update(choices=cap_choices, value=cap_default),
                        status,
                    )

                pm_dataset.change(
                    _auto_fill_columns, inputs=[pm_dataset, pm_data_source],
                    outputs=[pm_audio_col, pm_text_col, pm_speaker_col, pm_caption_col, pm_col_status],
                )
                pm_data_source.change(
                    _auto_fill_columns, inputs=[pm_dataset, pm_data_source],
                    outputs=[pm_audio_col, pm_text_col, pm_speaker_col, pm_caption_col, pm_col_status],
                )

                pm_cmd_preview = gr.Textbox(label="📋 実行コマンドプレビュー", interactive=False, lines=3)

                def _get_pm_inputs_values(mode, local_path, hf_name, split,
                                          prepare_mode, audio_col, text_col, speaker_col, caption_col,
                                          output_manifest, latent_dir, device, codec_repo):
                    src_mode = {"ローカルCSV": "local_csv",
                                "ローカルJSONL": "local_jsonl",
                                "HuggingFaceデータセット": "hf_dataset"}.get(mode, "local_csv")
                    dataset = hf_name if mode == "HuggingFaceデータセット" else local_path
                    return (
                        src_mode, dataset, split, prepare_mode,
                        audio_col, text_col, speaker_col, caption_col,
                        output_manifest, latent_dir, device, codec_repo,
                    )

                _pm_all_inputs = [pm_data_source, pm_dataset, pm_hf_name, pm_split,
                                  pm_prepare_mode, pm_audio_col, pm_text_col, pm_speaker_col, pm_caption_col,
                                  pm_output_manifest, pm_latent_dir, pm_device, pm_codec_repo]

                def _update_pm_cmd(mode, local_path, hf_name, split,
                                   prepare_mode, audio_col, text_col, speaker_col, caption_col,
                                   output_manifest, latent_dir, device, codec_repo):
                    args = _get_pm_inputs_values(mode, local_path, hf_name, split,
                                                 prepare_mode, audio_col, text_col, speaker_col, caption_col,
                                                 output_manifest, latent_dir, device, codec_repo)
                    return _manifest_cmd_preview(*args)

                for comp in _pm_all_inputs:
                    comp.change(_update_pm_cmd, inputs=_pm_all_inputs, outputs=[pm_cmd_preview])

                with gr.Row():
                    pm_run_btn  = gr.Button("▶️ 実行", variant="primary")
                    pm_stop_btn = gr.Button("⏹️ 停止", variant="stop")
                    pm_log_btn  = gr.Button("🔄 ログ更新")

                pm_status = gr.Textbox(label="実行状況", interactive=False, lines=2)
                pm_log    = gr.Textbox(label="ログ出力", interactive=False, lines=20, max_lines=20)

                def _run_manifest_ui(mode, local_path, hf_name, split,
                                     prepare_mode, audio_col, text_col, speaker_col, caption_col,
                                     output_manifest, latent_dir, device, codec_repo):
                    args = _get_pm_inputs_values(mode, local_path, hf_name, split,
                                                 prepare_mode, audio_col, text_col, speaker_col, caption_col,
                                                 output_manifest, latent_dir, device, codec_repo)
                    return _run_manifest(*args)

                pm_run_btn.click(_run_manifest_ui, inputs=_pm_all_inputs,
                                 outputs=[pm_status, pm_cmd_preview])
                pm_stop_btn.click(_stop_process, outputs=[pm_status])
                pm_log_btn.click(_read_manifest_log, outputs=[pm_log])

            # ═══════════════════════════════════════════════════════════════
            # タブ3: 学習
            # ═══════════════════════════════════════════════════════════════
            with gr.Tab("🏋️ 学習"):
                gr.Markdown("## 学習設定")

                with gr.Accordion("💾 プリセット管理（configs/ フォルダ）", open=True):
                    with gr.Row():
                        preset_dropdown = gr.Dropdown(
                            label="プリセット選択", choices=initial_configs,
                            value=default_config or None, scale=3,
                        )
                        preset_refresh_btn = gr.Button("🔄 更新", scale=1)
                    with gr.Row():
                        preset_name_input = gr.Textbox(
                            label="保存ファイル名（例: my_config.yaml）",
                            value="my_config.yaml", scale=3,
                        )
                        preset_save_btn = gr.Button("💾 保存", scale=1)
                    preset_status = gr.Textbox(label="プリセット操作結果", interactive=False, lines=1)

                with gr.Row():
                    train_manifest = gr.Dropdown(
                        label="マニフェストファイル (.jsonl)",
                        choices=initial_manifests,
                        value=initial_manifests[-1] if initial_manifests else None,
                        allow_custom_value=True, scale=3,
                    )
                    train_manifest_refresh = gr.Button("🔄", scale=1)
                train_output_dir = gr.Textbox(
                    label="学習出力フォルダ（チェックポイント保存先）",
                    value=str(BASE_DIR / "outputs" / "irodori_tts"),
                )

                with gr.Row():
                    num_gpus = gr.Slider(label="GPU数（1=単体, 複数=DDP）", minimum=1, maximum=8, value=1, step=1)
                    save_mode = gr.Dropdown(
                        label="保存ファイル形式",
                        choices=["EMAのみ", "Fullのみ", "EMA + Full両方"],
                        value="EMAのみ",
                        info="EMA=推論用軽量版、Full=追加学習用（optimizer状態含む）",
                    )
                    train_attention_backend = gr.Dropdown(
                        label="Attention Backend",
                        choices=["sdpa", "flash2", "sage", "eager"],
                        value="sdpa",
                        info="sdpa=推奨 / flash2=FlashAttention2要インストール / sage=SageAttention要インストール",
                    )

                with gr.Accordion("🔁 ベースモデル・追加学習設定", open=True):
                    _default_safetensors = str(
                        CHECKPOINTS_DIR / "Aratako_Irodori-TTS-500M-v2" / "model.safetensors"
                    )
                    gr.Markdown(
                        "**--resume オプション設定**\n\n"
                        "- **オフ（スクラッチ学習）**: モデルを最初からランダム初期化して学習します。\n"
                        "- **オン・パス未入力**: `checkpoints/Aratako_Irodori-TTS-500M-v2/model.safetensors` が"
                        "存在すれば自動でロードして追加学習します（デフォルト動作）。\n"
                        "- **オン・パス入力**: 指定したファイルをベースに追加学習します。"
                        "`.safetensors` を指定するとstep=0から学習開始、`.pt` チェックポイントを指定すると"
                        "step/optimizer状態を引き継いで再開します。"
                    )
                    with gr.Row():
                        resume_enabled = gr.Checkbox(
                            label="--resume を有効にする（追加学習 / チェックポイント再開）",
                            value=True,
                            scale=1,
                        )
                    resume_checkpoint = gr.Textbox(
                        label="ベースモデルパス（空欄 = デフォルト自動参照）",
                        value="",
                        placeholder=_default_safetensors,
                        info=f"空欄の場合、resume有効時は {_default_safetensors} を自動参照します。",
                    )

                with gr.Accordion("⚙️ バッチ・精度設定", open=True):
                    gr.Markdown("*バッチサイズと勾配蓄積ステップの積が実効バッチサイズになります。*")
                    with gr.Row():
                        t_batch_size  = gr.Slider(label="バッチサイズ（GPUメモリに合わせて調整）", minimum=1, maximum=64, value=_v("batch_size", 4), step=1)
                        t_grad_accum  = gr.Slider(label="勾配蓄積ステップ数（実効バッチを増やす）", minimum=1, maximum=32, value=_v("gradient_accumulation_steps", 2), step=1)
                        t_num_workers = gr.Slider(label="DataLoaderワーカー数", minimum=0, maximum=16, value=_v("num_workers", 4), step=1)
                    with gr.Row():
                        t_persistent_workers = gr.Checkbox(label="ワーカーの永続化（起動高速化）", value=_v("dataloader_persistent_workers", True))
                        t_prefetch_factor    = gr.Slider(label="プリフェッチ係数", minimum=1, maximum=8, value=_v("dataloader_prefetch_factor", 2), step=1)
                        t_allow_tf32         = gr.Checkbox(label="TF32を許可（Ampere以降GPU向け高速化）", value=_v("allow_tf32", True))
                        t_compile_model      = gr.Checkbox(label="torch.compileを使用（初回遅延あり）", value=_v("compile_model", False))
                        t_precision          = gr.Dropdown(label="学習精度", choices=["bf16", "fp32", "fp16"], value=_v("precision", "bf16"))

                with gr.Accordion("🔧 オプティマイザ設定", open=True):
                    gr.Markdown("*Muon: 行列重み向けの高性能オプティマイザ。AdamW: 汎用的で安定。*")
                    with gr.Row():
                        t_optimizer    = gr.Dropdown(label="オプティマイザ", choices=["muon", "adamw", "lion", "ademamix", "sgd"], value=_v("optimizer", "muon"))
                        t_learning_rate= gr.Number(label="学習率", value=_v("learning_rate", 3e-4))
                        t_weight_decay = gr.Number(label="重み減衰（L2正則化）", value=_v("weight_decay", 0.01))
                    with gr.Row():
                        t_muon_momentum= gr.Number(label="Muonモメンタム（Muon使用時のみ有効）", value=_v("muon_momentum", 0.95))
                        t_adam_beta1   = gr.Number(label="Adam β1（AdamW使用時）", value=_v("adam_beta1", 0.9))
                        t_adam_beta2   = gr.Number(label="Adam β2（AdamW使用時）", value=_v("adam_beta2", 0.999))
                        t_adam_eps     = gr.Number(label="Adam ε（AdamW使用時）", value=_v("adam_eps", 1e-8))

                with gr.Accordion("📈 学習率スケジューラ", open=True):
                    gr.Markdown("*wsd: warmup→stable→decay の3段階スケジュール。cosine: コサインアニーリング。*")
                    with gr.Row():
                        t_lr_scheduler  = gr.Dropdown(label="スケジューラ種別", choices=["wsd", "cosine", "none"], value=_v("lr_scheduler", "wsd"))
                        t_warmup_steps  = gr.Number(label="ウォームアップステップ数", value=_v("warmup_steps", 300), precision=0)
                        t_stable_steps  = gr.Number(label="安定期ステップ数（wsdのみ）", value=_v("stable_steps", 2100), precision=0)
                        t_min_lr_scale  = gr.Number(label="最小学習率スケール比率（0〜1）", value=_v("min_lr_scale", 0.01))

                with gr.Accordion("🔢 学習ステップ・テキスト設定", open=True):
                    with gr.Row():
                        t_max_steps             = gr.Number(label="最大学習ステップ数", value=_v("max_steps", 3000), precision=0)
                        t_max_text_len          = gr.Number(label="テキスト最大トークン長", value=_v("max_text_len", 256), precision=0)
                        t_max_latent_steps      = gr.Number(label="ラテント最大フレーム数", value=_v("max_latent_steps", 750), precision=0)
                        t_fixed_target_latent_steps = gr.Number(label="固定ターゲットラテント長", value=_v("fixed_target_latent_steps", 750), precision=0)
                        t_fixed_target_full_mask= gr.Checkbox(label="固定ターゲット全マスク", value=_v("fixed_target_full_mask", True))

                with gr.Accordion("🎲 Conditioningドロップアウト・タイムステップ", open=False):
                    gr.Markdown("*ドロップアウト率を高めると過学習防止。小データセットでは0.1〜0.2推奨。*")
                    with gr.Row():
                        t_text_dropout      = gr.Slider(label="テキスト条件ドロップアウト率（0=無効）", minimum=0.0, maximum=0.5, value=_v("text_condition_dropout", 0.15), step=0.01)
                        t_speaker_dropout   = gr.Slider(label="話者条件ドロップアウト率（0=無効）", minimum=0.0, maximum=0.5, value=_v("speaker_condition_dropout", 0.15), step=0.01)
                        t_timestep_stratified= gr.Checkbox(label="タイムステップ層化サンプリング（安定化に有効）", value=_v("timestep_stratified", True))

                with gr.Accordion("💾 グラフ更新・チェックポイント保存設定", open=False):
                    with gr.Row():
                        t_log_every  = gr.Number(label="グラフ描画間隔（ステップ数）", value=_v("log_every", 10), precision=0,
                                                 info="この間隔でloss/lrをログ出力→グラフに反映。ファイル保存とは無関係。")
                        t_save_every = gr.Number(label="チェックポイント保存間隔（ステップ数）", value=_v("save_every", 100), precision=0)

                with gr.Accordion("📊 Weights & Biases 設定", open=False):
                    gr.Markdown("*wandb_enabledをオンにするとクラウドでリアルタイム学習曲線を確認できます。*")
                    with gr.Row():
                        t_wandb_enabled  = gr.Checkbox(label="W&B を有効化", value=_v("wandb_enabled", False))
                        t_wandb_project  = gr.Textbox(label="W&B プロジェクト名", value=_v("wandb_project", "") or "")
                        t_wandb_run_name = gr.Textbox(label="W&B 実行名（省略可）", value=_v("wandb_run_name", "") or "")

                with gr.Accordion("✅ バリデーション設定", open=False):
                    gr.Markdown("*valid_ratioを0より大きくするとバリデーションlossを監視できます。early_stoppingには必須。*")
                    with gr.Row():
                        t_valid_ratio= gr.Slider(label="バリデーション分割比率（0=無効）", minimum=0.0, maximum=0.5, value=_v("valid_ratio", 0.0), step=0.01)
                        t_valid_every= gr.Number(label="バリデーション実行間隔（ステップ数）", value=_v("valid_every", 100), precision=0)

                with gr.Accordion("🔀 オプション機能", open=False):
                    gr.Markdown("*Early Stoppingはvalid_ratio > 0 のときのみ有効。EMAは推論品質向上に有効。*")
                    with gr.Row():
                        t_early_stopping = gr.Checkbox(label="Early Stopping を有効化（valid lossが改善しなくなったら自動停止）", value=False)
                        t_es_patience    = gr.Number(label="Early Stopping: 悪化を許容する回数", value=3, precision=0)
                        t_es_min_delta   = gr.Number(label="Early Stopping: カウント最小悪化量", value=0.01)
                    with gr.Row():
                        t_use_ema  = gr.Checkbox(label="EMA（指数移動平均）を有効化（推論品質向上）", value=False)
                        t_ema_decay= gr.Number(label="EMA減衰率（0に近いほど追従速度が速い）", value=0.9999)
                    t_seed = gr.Number(label="乱数シード（再現性のために固定推奨）", value=_v("seed", 0), precision=0)

                gr.Markdown("### 📋 実行コマンドプレビュー")
                train_cmd_preview = gr.Textbox(label="コマンドライン（確認用）", interactive=False, lines=3)

                with gr.Row():
                    train_start_btn = gr.Button("▶️ 学習開始", variant="primary", size="lg")
                    train_stop_btn  = gr.Button("⏹️ 学習停止", variant="stop")
                train_status = gr.Textbox(label="実行状況", interactive=False, lines=2)

                gr.Markdown("### 📈 学習ログ・グラフ")
                with gr.Row():
                    auto_refresh_interval = gr.Slider(
                        label="自動更新間隔（秒）",
                        minimum=2, maximum=60, value=5, step=1,
                        info="学習中にログ・グラフを自動更新する間隔です。",
                        scale=3,
                    )
                    train_log_refresh_btn = gr.Button("🔄 手動更新", scale=1)

                train_log_text = gr.Textbox(label="学習ログ（末尾200行）", interactive=False, lines=15, max_lines=15, elem_id="train_log_text")

                gr.HTML("""
<script>
(function() {
    function attachScrollWatcher() {
        var el = document.getElementById('train_log_text');
        if (!el) { setTimeout(attachScrollWatcher, 500); return; }
        var ta = el.querySelector('textarea');
        if (!ta) { setTimeout(attachScrollWatcher, 500); return; }
        var lastVal = ta.value;
        setInterval(function() {
            if (ta.value !== lastVal) {
                lastVal = ta.value;
                ta.scrollTop = ta.scrollHeight;
            }
        }, 300);
    }
    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', attachScrollWatcher);
    } else {
        attachScrollWatcher();
    }
})();
</script>
""")

                if _PANDAS_AVAILABLE:
                    gr.Markdown("*ログ・グラフは自動更新されます（手動更新ボタンでも即時反映可）。*")
                    _empty_df = pd.DataFrame({"step": [], "loss": [], "lr": []})
                    with gr.Row():
                        loss_plot = gr.LinePlot(
                            value=_empty_df,
                            label="Loss曲線",
                            x="step", y="loss",
                            height=300,
                        )
                        lr_plot = gr.LinePlot(
                            value=_empty_df,
                            label="学習率曲線",
                            x="step", y="lr",
                            height=300,
                        )

                    def _do_refresh():
                        log = _read_train_log()
                        df = _parse_train_log_metrics()
                        return log, df, df

                    train_log_refresh_btn.click(_do_refresh, outputs=[train_log_text, loss_plot, lr_plot])

                    _auto_timer = gr.Timer(value=5, active=True)
                    _auto_timer.tick(_do_refresh, outputs=[train_log_text, loss_plot, lr_plot])
                    auto_refresh_interval.change(
                        lambda v: float(v),
                        inputs=[auto_refresh_interval],
                        outputs=[_auto_timer],
                    )

                else:
                    gr.Markdown(
                        "⚠️ **グラフ表示には `pandas` が必要です。**\n"
                        "`pip install pandas` または `uv add pandas` を実行後に再起動してください。"
                    )
                    metrics_text = gr.Textbox(label="メトリクス（step / loss / lr）", interactive=False, lines=6)

                    def _do_refresh_nopd():
                        log = _read_train_log()
                        df = _parse_train_log_metrics()
                        if df is None:
                            metrics = "（pandas未インストールのためグラフ非表示）"
                        elif df.empty:
                            metrics = "（データなし）"
                        else:
                            lines = [f"step={int(r['step'])}  loss={r['loss']:.4f}  lr={r['lr']:.2e}"
                                     for _, r in df.tail(10).iterrows()]
                            metrics = "\n".join(lines)
                        return log, metrics

                    train_log_refresh_btn.click(_do_refresh_nopd, outputs=[train_log_text, metrics_text])

                    _auto_timer = gr.Timer(value=5, active=True)
                    _auto_timer.tick(_do_refresh_nopd, outputs=[train_log_text, metrics_text])
                    auto_refresh_interval.change(
                        lambda v: float(v),
                        inputs=[auto_refresh_interval],
                        outputs=[_auto_timer],
                    )

                _train_cfg_inputs = [
                    train_manifest, train_output_dir,
                    t_batch_size, t_grad_accum, t_num_workers, t_persistent_workers, t_prefetch_factor,
                    t_allow_tf32, t_compile_model, t_precision,
                    t_optimizer, t_muon_momentum, t_learning_rate, t_weight_decay,
                    t_adam_beta1, t_adam_beta2, t_adam_eps,
                    t_lr_scheduler, t_warmup_steps, t_stable_steps, t_min_lr_scale,
                    t_max_steps, t_max_text_len,
                    t_text_dropout, t_speaker_dropout, t_timestep_stratified,
                    t_max_latent_steps, t_fixed_target_latent_steps, t_fixed_target_full_mask,
                    t_log_every, t_save_every,
                    t_wandb_enabled, t_wandb_project, t_wandb_run_name,
                    t_valid_ratio, t_valid_every,
                    t_early_stopping, t_es_patience, t_es_min_delta,
                    t_use_ema, t_ema_decay, t_seed,
                ]
                _train_exec_inputs = [
                    train_manifest, train_output_dir, preset_dropdown,
                    t_early_stopping, t_es_patience, t_es_min_delta,
                    t_use_ema, t_ema_decay,
                    resume_enabled, resume_checkpoint, save_mode,
                    num_gpus,
                    train_attention_backend,
                ] + _train_cfg_inputs

                def _update_train_cmd(manifest, output_dir, config_path,
                                      use_early_stopping, es_patience, es_min_delta,
                                      use_ema, ema_decay,
                                      resume_enabled, resume_checkpoint, save_mode,
                                      num_gpus, attention_backend, *_rest):
                    return _build_train_command(manifest, output_dir, config_path,
                                               use_early_stopping, es_patience, es_min_delta,
                                               use_ema, ema_decay,
                                               resume_enabled, resume_checkpoint, save_mode,
                                               num_gpus, attention_backend)

                for comp in [train_manifest, train_output_dir, preset_dropdown,
                              t_early_stopping, t_es_patience, t_es_min_delta,
                              t_use_ema, t_ema_decay,
                              resume_enabled, resume_checkpoint, save_mode,
                              num_gpus, train_attention_backend]:
                    comp.change(_update_train_cmd, inputs=_train_exec_inputs, outputs=[train_cmd_preview])

                def _load_preset(config_path: str):
                    cfg = _load_yaml_config(config_path).get("train", {})
                    def g(k, fb): return cfg.get(k, fb)
                    return (
                        g("batch_size", 4), g("gradient_accumulation_steps", 2),
                        g("num_workers", 4), g("dataloader_persistent_workers", True),
                        g("dataloader_prefetch_factor", 2), g("allow_tf32", True),
                        g("compile_model", False), g("precision", "bf16"),
                        g("optimizer", "muon"), g("muon_momentum", 0.95),
                        g("learning_rate", 3e-4), g("weight_decay", 0.01),
                        g("adam_beta1", 0.9), g("adam_beta2", 0.999), g("adam_eps", 1e-8),
                        g("lr_scheduler", "wsd"), g("warmup_steps", 300),
                        g("stable_steps", 2100), g("min_lr_scale", 0.01),
                        g("max_steps", 3000), g("max_text_len", 256),
                        g("text_condition_dropout", 0.15), g("speaker_condition_dropout", 0.15),
                        g("timestep_stratified", True),
                        g("max_latent_steps", 750), g("fixed_target_latent_steps", 750),
                        g("fixed_target_full_mask", True),
                        g("log_every", 10), g("save_every", 100),
                        g("wandb_enabled", False), g("wandb_project", "") or "",
                        g("wandb_run_name", "") or "",
                        g("valid_ratio", 0.0), g("valid_every", 100),
                        False, 3, 0.01, False, 0.9999,
                        g("seed", 0),
                    )

                _preset_outputs = [
                    t_batch_size, t_grad_accum, t_num_workers, t_persistent_workers, t_prefetch_factor,
                    t_allow_tf32, t_compile_model, t_precision,
                    t_optimizer, t_muon_momentum, t_learning_rate, t_weight_decay,
                    t_adam_beta1, t_adam_beta2, t_adam_eps,
                    t_lr_scheduler, t_warmup_steps, t_stable_steps, t_min_lr_scale,
                    t_max_steps, t_max_text_len,
                    t_text_dropout, t_speaker_dropout, t_timestep_stratified,
                    t_max_latent_steps, t_fixed_target_latent_steps, t_fixed_target_full_mask,
                    t_log_every, t_save_every,
                    t_wandb_enabled, t_wandb_project, t_wandb_run_name,
                    t_valid_ratio, t_valid_every,
                    t_early_stopping, t_es_patience, t_es_min_delta,
                    t_use_ema, t_ema_decay, t_seed,
                ]

                preset_dropdown.change(_load_preset, inputs=[preset_dropdown], outputs=_preset_outputs)
                preset_refresh_btn.click(
                    lambda: gr.Dropdown(choices=_scan_configs(), value=default_config or None),
                    outputs=[preset_dropdown],
                )

                def _save_preset(name, *cfg_args):
                    cfg_data = _config_from_ui(*cfg_args)
                    return _save_yaml_config(name, cfg_data)

                preset_save_btn.click(
                    _save_preset,
                    inputs=[preset_name_input] + _train_cfg_inputs,
                    outputs=[preset_status],
                )

                train_manifest_refresh.click(
                    lambda: gr.Dropdown(choices=_scan_manifests(), value=(_scan_manifests() or [None])[-1]),
                    outputs=[train_manifest],
                )

                train_start_btn.click(
                    _start_train, inputs=_train_exec_inputs,
                    outputs=[train_status, train_cmd_preview],
                )
                train_stop_btn.click(_stop_train, outputs=[train_status])

            # ═══════════════════════════════════════════════════════════════
            # タブ4: LoRA学習
            # ═══════════════════════════════════════════════════════════════
            with gr.Tab("🚀 LoRA学習"):
                gr.Markdown(
                    "## LoRA 差分学習\n"
                    "ベースモデルに対して LoRA アダプタを学習します。\n\n"
                    "> **必要ライブラリ**: `pip install peft`"
                )

                # ── プリセット管理 ── ← 追加
                with gr.Accordion("💾 プリセット管理（configs/ フォルダ）", open=True):
                    with gr.Row():
                        lora_preset_dropdown = gr.Dropdown(
                            label="プリセット選択",
                            choices=_scan_lora_configs(),
                            value=None,
                            scale=3,
                        )
                        lora_preset_refresh_btn = gr.Button("🔄 更新", scale=1)
                    with gr.Row():
                        lora_preset_name_input = gr.Textbox(
                            label="保存ファイル名（例: my_lora.yaml）",
                            value="my_lora.yaml",
                            scale=3,
                        )
                        lora_preset_save_btn = gr.Button("💾 保存", scale=1)
                    lora_preset_status = gr.Textbox(label="プリセット操作結果", interactive=False, lines=1)

                # ── ベースモデル ──
                with gr.Row():
                    lora_base_model = gr.Dropdown(
                        label="ベースモデル (.pt / .safetensors)",
                        choices=initial_checkpoints,
                        value=(
                            str(CHECKPOINTS_DIR / "Aratako_Irodori-TTS-500M-v2" / "model.safetensors")
                            if (CHECKPOINTS_DIR / "Aratako_Irodori-TTS-500M-v2" / "model.safetensors").exists()
                            else (initial_checkpoints[-1] if initial_checkpoints else None)
                        ),
                        allow_custom_value=True, scale=4,
                    )
                    lora_base_refresh_btn = gr.Button("🔄 更新", scale=1)

                # ── マニフェスト ──
                with gr.Row():
                    lora_manifest = gr.Dropdown(
                        label="マニフェストファイル (.jsonl)",
                        choices=initial_manifests,
                        value=initial_manifests[-1] if initial_manifests else None,
                        allow_custom_value=True, scale=4,
                    )
                    lora_manifest_refresh_btn = gr.Button("🔄", scale=1)

                # ── 実行名・保存先 ──
                with gr.Row():
                    lora_run_name = gr.Textbox(
                        label="実行名 (run_name)（空欄=タイムスタンプ自動生成）",
                        value="", placeholder="my_lora_run", scale=2,
                    )
                    lora_output_dir = gr.Textbox(
                        label="LoRA出力フォルダ（空欄=lora/{run_name}/）",
                        value="", placeholder=str(LORA_DIR / "{run_name}"), scale=2,
                    )

                # ── 保存モード・Attention ──
                with gr.Row():
                    lora_save_mode = gr.Dropdown(
                        label="保存モード",
                        choices=["EMAのみ", "EMA + Full両方"],
                        value="EMAのみ",
                        info="EMAのみ=推論専用 / EMA+Full=Resume前提",
                    )
                    lora_attention_backend = gr.Dropdown(
                        label="Attention Backend",
                        choices=["sdpa", "flash2", "sage", "eager"],
                        value="sdpa",
                        info="sdpa=推奨 / flash2・sage=別途インストール必要",
                    )

                # ── LoRA設定 ──
                with gr.Accordion("🔧 LoRA設定", open=True):
                    with gr.Row():
                        lora_rank = gr.Slider(label="LoRAランク", minimum=1, maximum=128, value=16, step=1)
                        lora_alpha = gr.Number(label="lora_alpha", value=32.0)
                        lora_dropout = gr.Slider(label="lora_dropout", minimum=0.0, maximum=0.5, value=0.05, step=0.01)
                    lora_target_modules = gr.Textbox(
                        label="ターゲットモジュール（カンマ区切り）",
                        value="wq,wk,wv,wo",
                        info="デフォルト: wq,wk,wv,wo / 拡張: wq,wk,wv,wo,wk_text,wv_text,wk_speaker,wv_speaker,w1,w2,w3",
                    )

                # ── Resume設定 ──
                with gr.Accordion("🔄 Resume設定", open=False):
                    lora_resume_enabled = gr.Checkbox(label="Resume（既存LoRAから再開）", value=False)
                    with gr.Row():
                        lora_resume_path = gr.Dropdown(
                            label="既存LoRAフォルダ（_full推奨）",
                            choices=_scan_lora_adapters(),
                            value=None, allow_custom_value=True, scale=4,
                        )
                        lora_resume_refresh_btn = gr.Button("🔄", scale=1)
                    lora_resume_warning = gr.Markdown(visible=False)

                    def _on_lora_resume_path_change(path):
                        if path and "_ema" in str(path):
                            return gr.update(visible=True, value=(
                                "⚠️ **_ema フォルダを選択しています。**\n\n"
                                "EMA版にはoptimizer状態・step数が含まれないため、"
                                "学習は step=0 から再スタートします。\n"
                                "学習率ウォームアップが再度かかり、学習曲線が不連続になります。\n\n"
                                "中断した学習を完全に再開する場合は **_full フォルダ** を選択してください。"
                            ))
                        return gr.update(visible=False)

                    lora_resume_path.change(
                        _on_lora_resume_path_change,
                        inputs=[lora_resume_path], outputs=[lora_resume_warning],
                    )
                    lora_resume_refresh_btn.click(
                        lambda: gr.Dropdown(choices=_scan_lora_adapters()),
                        outputs=[lora_resume_path],
                    )

                # ── 学習パラメータ ──
                with gr.Accordion("⚙️ 学習パラメータ", open=True):
                    with gr.Row():
                        lora_batch_size = gr.Slider(label="バッチサイズ", minimum=1, maximum=32, value=4, step=1)
                        lora_grad_accum = gr.Slider(label="勾配蓄積ステップ", minimum=1, maximum=16, value=1, step=1)
                    with gr.Row():
                        lora_lr = gr.Number(label="学習率", value=1e-4)
                        lora_optimizer = gr.Dropdown(
                            label="オプティマイザ", choices=["adamw", "muon", "lion", "ademamix"],
                            value="adamw",
                        )
                        lora_lr_scheduler = gr.Dropdown(
                            label="スケジューラ", choices=["none", "cosine", "wsd"], value="none",
                        )
                        lora_warmup_steps = gr.Number(label="ウォームアップステップ", value=0, precision=0)
                    with gr.Row():
                        lora_max_steps = gr.Number(label="最大学習ステップ", value=1000, precision=0)
                        lora_save_every = gr.Number(label="保存間隔", value=100, precision=0)
                        lora_log_every = gr.Number(label="ログ間隔", value=10, precision=0)

                # ── EMA設定 ──
                with gr.Accordion("📊 EMA設定", open=False):
                    with gr.Row():
                        lora_use_ema = gr.Checkbox(label="EMAを有効化", value=True)
                        lora_ema_decay = gr.Number(label="EMA減衰率", value=0.9999)

                # ── バリデーション設定 ──
                with gr.Accordion("✅ バリデーション設定", open=False):
                    with gr.Row():
                        lora_valid_ratio = gr.Slider(label="バリデーション分割比率", minimum=0.0, maximum=0.5, value=0.0, step=0.01)
                        lora_valid_every = gr.Number(label="バリデーション実行間隔", value=100, precision=0)

                # ── Early Stopping ──
                with gr.Accordion("🛑 Early Stopping設定", open=False):
                    with gr.Row():
                        lora_early_stopping = gr.Checkbox(label="Early Stoppingを有効化", value=False)
                        lora_es_patience = gr.Number(label="パティエンス", value=3, precision=0)
                        lora_es_min_delta = gr.Number(label="最小悪化量", value=0.01)

                # ── W&B設定 ──
                with gr.Accordion("📈 W&B設定", open=False):
                    with gr.Row():
                        lora_wandb_enabled = gr.Checkbox(label="W&Bを有効化", value=False)
                        lora_wandb_project = gr.Textbox(label="W&Bプロジェクト名", value="")
                        lora_wandb_run_name = gr.Textbox(label="W&B実行名（省略可）", value="")

                lora_seed = gr.Number(label="乱数シード", value=0, precision=0)

                gr.Markdown("### 📋 実行コマンドプレビュー")
                lora_cmd_preview = gr.Textbox(label="コマンドライン（確認用）", interactive=False, lines=3)

                with gr.Row():
                    lora_start_btn = gr.Button("▶️ LoRA学習開始", variant="primary", size="lg")
                    lora_stop_btn = gr.Button("⏹️ 停止", variant="stop")
                lora_train_status = gr.Textbox(label="実行状況", interactive=False, lines=2)

                gr.Markdown("### 📋 学習ログ")
                with gr.Row():
                    lora_log_interval = gr.Slider(label="自動更新間隔（秒）", minimum=2, maximum=60, value=5, step=1, scale=3)
                    lora_log_refresh_btn = gr.Button("🔄 手動更新", scale=1)
                lora_log_text = gr.Textbox(label="LoRA学習ログ（末尾200行）", interactive=False, lines=15, max_lines=15)

                # ── イベント配線 ──
                _lora_exec_inputs = [
                    lora_base_model, lora_manifest, lora_output_dir, lora_run_name,
                    lora_rank, lora_alpha, lora_dropout, lora_target_modules,
                    lora_save_mode, lora_attention_backend,
                    lora_early_stopping, lora_es_patience, lora_es_min_delta,
                    lora_use_ema, lora_ema_decay,
                    lora_resume_enabled, lora_resume_path,
                    lora_batch_size, lora_grad_accum, lora_lr, lora_optimizer,
                    lora_lr_scheduler, lora_warmup_steps,
                    lora_max_steps, lora_save_every, lora_log_every,
                    lora_valid_ratio, lora_valid_every,
                    lora_wandb_enabled, lora_wandb_project, lora_wandb_run_name,
                    lora_seed,
                ]

                # プリセット読み込み・保存対象コンポーネント
                # （base_model / manifest / output_dir / run_name はパス依存のため読み込み対象外）
                _lora_preset_outputs = [
                    lora_rank, lora_alpha, lora_dropout, lora_target_modules,
                    lora_save_mode, lora_attention_backend,
                    lora_early_stopping, lora_es_patience, lora_es_min_delta,
                    lora_use_ema, lora_ema_decay,
                    lora_batch_size, lora_grad_accum, lora_lr, lora_optimizer,
                    lora_lr_scheduler, lora_warmup_steps,
                    lora_max_steps, lora_save_every, lora_log_every,
                    lora_valid_ratio, lora_valid_every,
                    lora_wandb_enabled, lora_wandb_project, lora_wandb_run_name,
                    lora_seed,
                ]

                def _update_lora_cmd(*args):
                    try:
                        return " ".join(_build_lora_train_command(*args))
                    except Exception as e:
                        return f"(プレビュー生成エラー: {e})"

                for comp in [lora_base_model, lora_manifest, lora_output_dir, lora_run_name,
                              lora_rank, lora_alpha, lora_dropout, lora_target_modules,
                              lora_save_mode, lora_attention_backend,
                              lora_use_ema, lora_ema_decay, lora_max_steps]:
                    comp.change(_update_lora_cmd, inputs=_lora_exec_inputs, outputs=[lora_cmd_preview])

                # プリセット読み込み・更新・保存
                lora_preset_dropdown.change(
                    _load_lora_preset,
                    inputs=[lora_preset_dropdown],
                    outputs=_lora_preset_outputs,
                )
                lora_preset_refresh_btn.click(
                    lambda: gr.Dropdown(choices=_scan_lora_configs(), value=None),
                    outputs=[lora_preset_dropdown],
                )
                lora_preset_save_btn.click(
                    _save_lora_preset,
                    inputs=[lora_preset_name_input] + _lora_exec_inputs,
                    outputs=[lora_preset_status],
                )

                lora_base_refresh_btn.click(
                    lambda: gr.Dropdown(choices=_scan_checkpoints(), value=(_scan_checkpoints() or [None])[-1]),
                    outputs=[lora_base_model],
                )
                lora_manifest_refresh_btn.click(
                    lambda: gr.Dropdown(choices=_scan_manifests(), value=(_scan_manifests() or [None])[-1]),
                    outputs=[lora_manifest],
                )
                lora_start_btn.click(_start_lora_train, inputs=_lora_exec_inputs,
                                     outputs=[lora_train_status, lora_cmd_preview])
                lora_stop_btn.click(_stop_lora_train, outputs=[lora_train_status])
                lora_log_refresh_btn.click(_read_lora_train_log, outputs=[lora_log_text])
                _lora_timer = gr.Timer(value=5, active=True)
                _lora_timer.tick(_read_lora_train_log, outputs=[lora_log_text])
                lora_log_interval.change(lambda v: float(v), inputs=[lora_log_interval], outputs=[_lora_timer])

            # ═══════════════════════════════════════════════════════════════
            # タブ5: Dataset作成
            # ═══════════════════════════════════════════════════════════════
            with gr.Tab("🎙️ Dataset作成"):
                gr.Markdown(
                    "## データセット作成\n"
                    "長尺音声の**無音区間スライス**と**Whisperキャプション**を行い、"
                    "学習用 manifest（CSV / JSONL）を生成します。\n\n"
                    "> 必要ライブラリ: `pip install librosa soundfile faster-whisper`"
                )

                ds_mode = gr.Radio(
                    label="実行モード",
                    choices=["スライスのみ", "キャプションのみ", "パイプライン（スライス→キャプション）"],
                    value="パイプライン（スライス→キャプション）",
                )

                with gr.Accordion("✂️ スライス設定", open=True) as slice_accordion:
                    gr.Markdown("*Silero VAD（ニューラルネット音声活動検出）で発話区間を検出してスライスします。連続発話・キャラクター音声に対応。*")
                    with gr.Row():
                        ds_input = gr.Textbox(
                            label="入力パス（ファイルまたはフォルダ）",
                            value=str(BASE_DIR / "input"),
                            placeholder="/path/to/audio_or_folder",
                            scale=3,
                        )
                        ds_recursive_slice = gr.Checkbox(label="サブフォルダも検索", value=False, scale=1)
                    ds_slice_output = gr.Textbox(
                        label="スライス済み音声の保存先フォルダ",
                        value=str(DEFAULT_DATASET_DIR),
                        placeholder=str(DEFAULT_DATASET_DIR),
                    )
                    with gr.Row():
                        ds_min_sec      = gr.Number(label="最小セグメント長（秒）", value=2.0,
                                                    info="これより短いセグメントは破棄")
                        ds_max_sec      = gr.Number(label="最大セグメント長（秒）", value=30.0,
                                                    info="超えた場合は最近傍の無音点で分割")
                        ds_top_db       = gr.Slider(label="VAD 発話判定閾値（threshold）",
                                                    minimum=0.1, maximum=0.9, value=0.5, step=0.05,
                                                    info="大きいほど厳しく検出（0.5推奨）")
                    with gr.Row():
                        ds_frame_length = gr.Number(label="無音最短継続時間（ms）", value=300, precision=0,
                                                    info="この時間以上の無音でないと区切らない")
                        ds_hop_length   = gr.Number(label="発話前後パディング（ms）", value=30, precision=0,
                                                    info="発話区間の前後に追加する余白")
                    with gr.Row():
                        ds_target_sr_enabled = gr.Checkbox(label="リサンプルを有効化", value=False)
                        ds_target_sr         = gr.Number(label="リサンプル先サンプリングレート（Hz）", value=44100, precision=0)

                with gr.Accordion("🗣️ キャプション設定", open=True) as caption_accordion:
                    gr.Markdown("*faster-whisper で音声を文字起こしします。精度重視設定（large-v3 + beam=5）がデフォルトです。*")
                    ds_caption_input = gr.Textbox(
                        label="キャプション対象フォルダ（キャプションのみモード時に使用）",
                        value=str(DEFAULT_DATASET_DIR),
                        placeholder=str(DEFAULT_DATASET_DIR),
                        info="パイプラインモード時はスライス出力先が自動的に使われます。",
                    )
                    with gr.Row():
                        ds_whisper_model = gr.Dropdown(
                            label="Whisperモデル",
                            choices=["large-v3", "large-v2", "large", "medium", "small", "base", "tiny"],
                            value="medium",
                            info="精度: large-v3 > large-v2 > medium > small（VRAMも同順で多く必要）",
                        )
                        ds_language = gr.Dropdown(
                            label="言語",
                            choices=["ja", "en", "zh", "ko", "自動検出"],
                            value="ja",
                        )
                        ds_device = gr.Dropdown(
                            label="使用デバイス",
                            choices=["自動", "cuda", "cpu"],
                            value="自動",
                        )
                    with gr.Row():
                        ds_speaker_field_mode = gr.Radio(
                            label="話者ID欄の用途",
                            choices=["speaker", "caption"],
                            value="speaker",
                            info="speaker=通常モード（speaker_id列） / caption=VoiceDesignモード（caption列）",
                            scale=2,
                        )
                    with gr.Row():
                        ds_speaker_id = gr.Textbox(
                            label="話者ID（省略可・全ファイルに付与）",
                            value="",
                            placeholder="例: SPEAKER_A",
                            scale=2,
                            visible=True,
                        )
                        ds_recursive_caption = gr.Checkbox(label="サブフォルダも検索", value=False, scale=1)
                    def _on_speaker_field_mode_change(mode):
                        is_voice_design = str(mode).strip() == "caption"
                        return gr.update(visible=not is_voice_design)
                    ds_speaker_field_mode.change(
                        _on_speaker_field_mode_change,
                        inputs=[ds_speaker_field_mode],
                        outputs=[ds_speaker_id],
                    )
                    ds_model_cache_dir = gr.Textbox(
                        label="Whisperモデルキャッシュフォルダ",
                        value=str(CHECKPOINTS_DIR / "whisper"),
                        info="モデルが存在しない場合は自動ダウンロードされます。空欄にするとHFデフォルト (~/.cache/huggingface/hub) に保存されます。",
                    )

                with gr.Accordion("🎭 絵文字キャプション設定（オプション）", open=False) as ec_accordion:
                    gr.Markdown(
                        "Whisperキャプション完了後に音響特徴量とLLMを使って自動処理します。\n\n"
                        "- **text列 （全モード）**: 絵文字キャプション（`EMOJI_ANNOTATIONS` ルール）\n"
                        "- **caption列 （VoiceDesignモードのみ）**: 音声分析キャプション（`VOICE_ANALYSIS_CAPTION_RULES`・音響パラメーターのみ・読み上げ内容含まず）\n\n"
                        "> 必要ライブラリ: `pip install librosa openai`  \n"
                        "> Manifest出力設定が **CSV形式** の場合のみ動作します。"
                    )
                    with gr.Row():
                        ec_enabled = gr.Checkbox(
                            label="🎭 絵文字キャプションを有効にする",
                            value=False,
                            scale=1,
                            info="チェックを入れるとWhisperキャプション完了後に続けて実行します。",
                        )
                        ec_api = gr.Dropdown(
                            label="APIプロバイダー",
                            choices=[
                                "LM Studio（ローカル）",
                                "Groq",
                                "OpenAI（ChatGPT）",
                                "Together AI",
                            ],
                            value="LM Studio（ローカル）",
                            scale=2,
                            interactive=False,
                            info="LM StudioはAPIキー不要。起動してモデルをロードしておいてください。",
                        )

                    ec_api_key = gr.Textbox(
                        label="APIキー（Groq / OpenAI / Together AI）",
                        value="",
                        placeholder="sk-... または gsk_...",
                        type="password",
                        visible=False,
                        info="LM Studio以外を選択した場合に必要です。",
                    )

                    def _ec_on_enabled(checked):
                        return gr.update(interactive=checked)

                    def _ec_on_api_change(api_label):
                        need_key = api_label not in ("LM Studio（ローカル）",)
                        return gr.update(visible=need_key)

                    ec_enabled.change(_ec_on_enabled, inputs=[ec_enabled], outputs=[ec_api])
                    ec_api.change(_ec_on_api_change, inputs=[ec_api], outputs=[ec_api_key])

                with gr.Accordion("📄 Manifest出力設定", open=True):
                    with gr.Row():
                        ds_manifest_output_dir = gr.Textbox(
                            label="manifest保存先フォルダ",
                            value=str(DEFAULT_DATASET_DIR),
                            placeholder=str(DEFAULT_DATASET_DIR),
                            scale=3,
                        )
                        ds_manifest_filename = gr.Textbox(
                            label="ファイル名（拡張子なし）",
                            value="metadata",
                            scale=2,
                        )
                        ds_output_format = gr.Dropdown(
                            label="出力フォーマット",
                            choices=["CSV", "JSONL"],
                            value="CSV",
                            scale=1,
                        )
                    gr.Markdown(
                        "📌 **フォーマット補足**\n"
                        "- **CSV（speakerモード）**: `file_name,text,speaker`\n"
                        "- **CSV（caption/VoiceDesignモード）**: `file_name,text,caption`\n"
                        "- **JSONL**: `{\"text\":\"...\",\"audio_path\":\"...\"}` — `prepare_manifest.py` への入力前段として使用可能"
                    )

                gr.Markdown("### 📋 実行コマンドプレビュー")
                ds_cmd_preview = gr.Textbox(label="コマンドライン（確認用）", interactive=False, lines=3)

                with gr.Row():
                    ds_start_btn = gr.Button("▶️ 実行", variant="primary", size="lg")
                    ds_stop_btn  = gr.Button("⏹️ 停止", variant="stop")
                ds_status = gr.Textbox(label="実行状況", interactive=False, lines=2)

                gr.Markdown("### 📋 実行ログ")
                with gr.Row():
                    ds_log_interval = gr.Slider(
                        label="自動更新間隔（秒）", minimum=2, maximum=30, value=3, step=1, scale=3,
                    )
                    ds_log_refresh_btn = gr.Button("🔄 手動更新", scale=1)
                ds_log_text = gr.Textbox(
                    label="ログ出力", interactive=False, lines=20, max_lines=20,
                    elem_id="ds_log_text",
                )
                gr.HTML("""
<script>
(function() {
    function attachDsLogScroll() {
        var el = document.getElementById('ds_log_text');
        if (!el) { setTimeout(attachDsLogScroll, 500); return; }
        var ta = el.querySelector('textarea');
        if (!ta) { setTimeout(attachDsLogScroll, 500); return; }
        var lastVal = ta.value;
        setInterval(function() {
            if (ta.value !== lastVal) { lastVal = ta.value; ta.scrollTop = ta.scrollHeight; }
        }, 300);
    }
    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', attachDsLogScroll);
    } else { attachDsLogScroll(); }
})();
</script>
""")

                def _on_mode_change(mode):
                    show_slice   = mode in ("スライスのみ", "パイプライン（スライス→キャプション）")
                    show_caption = mode in ("キャプションのみ", "パイプライン（スライス→キャプション）")
                    return (
                        gr.update(visible=show_slice, open=show_slice),
                        gr.update(visible=show_caption, open=show_caption),
                    )

                ds_mode.change(
                    _on_mode_change, inputs=[ds_mode],
                    outputs=[slice_accordion, caption_accordion],
                )

                _ds_all_inputs = [
                    ds_mode, ds_input, ds_slice_output,
                    ds_min_sec, ds_max_sec, ds_top_db, ds_frame_length, ds_hop_length,
                    ds_target_sr_enabled, ds_target_sr, ds_recursive_slice,
                    ds_caption_input, ds_manifest_output_dir, ds_manifest_filename,
                    ds_output_format, ds_whisper_model, ds_language, ds_speaker_id,
                    ds_speaker_field_mode,
                    ds_recursive_caption, ds_device, ds_model_cache_dir,
                ]

                def _update_ds_cmd(*args):
                    try:
                        return " ".join(_build_dataset_command(*args))
                    except Exception as e:
                        return f"(プレビュー生成エラー: {e})"

                for comp in _ds_all_inputs:
                    comp.change(_update_ds_cmd, inputs=_ds_all_inputs, outputs=[ds_cmd_preview])

                _ds_exec_inputs = _ds_all_inputs + [ec_enabled, ec_api, ec_api_key]

                def _start_dataset_job_with_emoji(*args):
                    base_args   = args[:-3]
                    ec_enabled_ = bool(args[-3])
                    ec_api_     = str(args[-2])
                    ec_api_key_ = str(args[-1]).strip()

                    # args[18] = ds_speaker_field_mode（_ds_all_inputs の順序に対応）
                    speaker_field_mode_ = str(args[18]).strip()
                    is_voice_design_    = speaker_field_mode_ == "caption"

                    status, cmd = _start_dataset_job(*base_args)

                    if ec_enabled_:
                        # 絵文字キャプション（text列）は両モードで実行
                        # 音声分析キャプション（caption列）はVoiceDesignモード時のみ実行
                        output_fmt = str(args[14]).strip()
                        if output_fmt != "CSV":
                            status += "\n⚠️ 絵文字キャプションはCSV形式のみ対応です。出力形式をCSVに変更してください。"
                        else:
                            manifest_dir  = str(args[12]).strip()
                            manifest_name = str(args[13]).strip()
                            csv_path      = str(Path(manifest_dir) / f"{manifest_name}.csv")
                            mode_         = str(args[0])
                            wav_dir_      = str(args[11]).strip() if mode_ == "キャプションのみ" else str(args[2]).strip()

                            def _wait_and_emoji(csv_path=csv_path, wav_dir_=wav_dir_,
                                                ec_api_=ec_api_, ec_api_key_=ec_api_key_,
                                                vd=is_voice_design_):
                                while True:
                                    with _DS_LOG_LOCK:
                                        proc = _DS_PROC
                                    if proc is None or proc.poll() is not None:
                                        break
                                    time.sleep(1)
                                with _DS_LOG_LOCK:
                                    rc = _DS_PROC.returncode if _DS_PROC else -1
                                if rc == 0:
                                    _run_emoji_caption_inline(csv_path, wav_dir_, ec_api_, ec_api_key_, voice_design=vd)

                            threading.Thread(target=_wait_and_emoji, daemon=True).start()
                            if is_voice_design_:
                                status += f"\n🎭 絵文字キャプション（text列）＋音声分析キャプション（caption列）: 完了後に自動実行します（API: {ec_api_}）"
                            else:
                                status += f"\n🎭 絵文字キャプション（text列のみ）: 完了後に自動実行します（API: {ec_api_}）\n   ※ speaker列はそのまま維持されます。"

                    return status, cmd

                ds_start_btn.click(
                    _start_dataset_job_with_emoji, inputs=_ds_exec_inputs,
                    outputs=[ds_status, ds_cmd_preview],
                )
                ds_stop_btn.click(_stop_dataset_job, outputs=[ds_status])

                def _ds_refresh():
                    return _read_dataset_log()

                ds_log_refresh_btn.click(_ds_refresh, outputs=[ds_log_text])

                _ds_timer = gr.Timer(value=3, active=True)
                _ds_timer.tick(_ds_refresh, outputs=[ds_log_text])
                ds_log_interval.change(
                    lambda v: float(v),
                    inputs=[ds_log_interval], outputs=[_ds_timer],
                )

            # ═══════════════════════════════════════════════════════════════
            # タブ6: チェックポイント変換
            # ═══════════════════════════════════════════════════════════════
            with gr.Tab("🔄 チェックポイント変換"):
                with gr.Tab("📦 通常チェックポイント変換"):
                    gr.Markdown(
                        "## .pt → .safetensors 変換\n"
                        "学習チェックポイント（`.pt`）を推論用の `.safetensors` 形式に変換します。\n"
                        "変換後のファイルは元の `.pt` と同じフォルダに保存されます。"
                    )

                    with gr.Row():
                        conv_input = gr.Dropdown(
                            label="変換対象の .pt ファイル",
                            choices=initial_train_ckpts,
                            value=initial_train_ckpts[-1] if initial_train_ckpts else None,
                            allow_custom_value=True, scale=4,
                        )
                        conv_refresh_btn = gr.Button("🔄 更新", scale=1)

                    conv_btn    = gr.Button("⚙️ 変換実行", variant="primary", size="lg")
                    conv_status = gr.Textbox(label="変換結果", interactive=False, lines=6)

                    conv_refresh_btn.click(
                        lambda: gr.Dropdown(choices=_scan_train_checkpoints(), value=(_scan_train_checkpoints() or [None])[-1]),
                        outputs=[conv_input],
                    )
                    conv_btn.click(_run_convert, inputs=[conv_input], outputs=[conv_status])

                with gr.Tab("🚀 LoRA変換"):
                    gr.Markdown(
                        "## LoRA Full版 → EMA版 変換\n"
                        "`_full` フォルダの EMA shadow 重みから、推論用の `_ema` フォルダを生成します。\n\n"
                        "> **必要条件**: LoRA学習時に `--save-full` と `--ema-decay` を指定して保存したチェックポイント"
                    )

                    with gr.Row():
                        lora_conv_input = gr.Dropdown(
                            label="変換対象の _full フォルダ",
                            choices=_scan_lora_full_adapters(),
                            value=(_scan_lora_full_adapters() or [None])[-1],
                            allow_custom_value=True, scale=4,
                        )
                        lora_conv_refresh_btn = gr.Button("🔄 更新", scale=1)

                    lora_conv_force = gr.Checkbox(label="既存の出力を上書き (--force)", value=False)
                    lora_conv_btn = gr.Button("⚙️ LoRA変換実行", variant="primary", size="lg")
                    lora_conv_status = gr.Textbox(label="変換結果", interactive=False, lines=8)

                    lora_conv_refresh_btn.click(
                        lambda: gr.Dropdown(choices=_scan_lora_full_adapters(), value=(_scan_lora_full_adapters() or [None])[-1]),
                        outputs=[lora_conv_input],
                    )
                    lora_conv_btn.click(
                        _run_lora_convert,
                        inputs=[lora_conv_input, lora_conv_force],
                        outputs=[lora_conv_status],
                    )

            # ═══════════════════════════════════════════════════════════════
            # タブ7: モデルマージ
            # ═══════════════════════════════════════════════════════════════
            with gr.Tab("🔀 モデルマージ"):
                gr.Markdown(
                    "## モデルマージ\n"
                    "推論用モデル（EMA .pt / .safetensors）同士をマージして新しいモデルを生成します。\n\n"
                    "> **対応形式**: `_ema.pt` / `.safetensors`（推論用のみ）"
                )

                initial_merge_ckpts = _merge_scan()
                default_base_path   = get_default_base_path()

                with gr.Row():
                    merge_ckpt_a = gr.Dropdown(
                        label="モデルA",
                        choices=initial_merge_ckpts,
                        value=initial_merge_ckpts[-1] if initial_merge_ckpts else None,
                        allow_custom_value=True, scale=4,
                    )
                    merge_refresh_a = gr.Button("🔄", scale=1)

                with gr.Row():
                    merge_ckpt_b = gr.Dropdown(
                        label="モデルB",
                        choices=initial_merge_ckpts,
                        value=initial_merge_ckpts[0] if len(initial_merge_ckpts) > 1 else None,
                        allow_custom_value=True, scale=4,
                    )
                    merge_refresh_b = gr.Button("🔄", scale=1)

                with gr.Accordion("⚙️ 基本マージ設定", open=True):
                    merge_method = gr.Dropdown(
                        label="マージ手法",
                        choices=["weighted_average", "slerp", "task_arithmetic"],
                        value="weighted_average",
                        info="weighted_average: 安定・高速 / slerp: ノルム保持 / task_arithmetic: ベースモデル必要",
                    )

                    with gr.Row():
                        merge_alpha = gr.Slider(
                            label="α（モデルAの割合）",
                            minimum=0.0, maximum=1.0, value=0.5, step=0.01,
                            info="Weighted Average / SLERP で使用",
                        )

                    with gr.Group() as ta_group:
                        gr.Markdown("**Task Arithmetic 設定**")
                        with gr.Row():
                            merge_lambda_a = gr.Slider(
                                label="λA（モデルAタスクベクトルの重み）",
                                minimum=0.0, maximum=1.0, value=0.5, step=0.01,
                            )
                            merge_lambda_b = gr.Slider(
                                label="λB（モデルBタスクベクトルの重み）",
                                minimum=0.0, maximum=1.0, value=0.5, step=0.01,
                                info="λA + λB は自動的に合計1.0に正規化されます",
                            )
                        with gr.Row():
                            merge_base_ta = gr.Dropdown(
                                label="ベースモデル（Task Arithmetic用）",
                                choices=initial_merge_ckpts,
                                value=default_base_path if default_base_path in initial_merge_ckpts else (initial_merge_ckpts[0] if initial_merge_ckpts else None),
                                allow_custom_value=True, scale=4,
                            )
                            merge_refresh_base = gr.Button("🔄", scale=1)

                    def _on_method_change(method):
                        visible = method == "task_arithmetic"
                        return gr.update(visible=visible)
                    merge_method.change(_on_method_change, inputs=[merge_method], outputs=[ta_group])

                with gr.Accordion("🧩 部分マージ（グループごとに手法を選択）", open=False):
                    gr.Markdown(
                        "有効にすると、レイヤーグループごとに異なるマージ手法を設定できます。\n"
                        "- **text**: テキストエンコーダ・TextBlock・JointAttentionのテキストKV\n"
                        "- **speaker**: 話者エンコーダ・JointAttentionの話者KV\n"
                        "- **diffusion_core**: DiffusionBlock本体（Attention/MLP/AdaLN）・cond_module\n"
                        "- **io**: in_proj / out_norm / out_proj"
                    )
                    use_partial = gr.Checkbox(label="部分マージを有効にする", value=False)

                    _method_choices = ["weighted_average", "slerp", "task_arithmetic"]

                    with gr.Group():
                        gr.Markdown("#### テキスト応答性グループ（text）")
                        with gr.Row():
                            pg_text_method  = gr.Dropdown(choices=_method_choices, value="weighted_average", label="手法")
                            pg_text_alpha   = gr.Slider(minimum=0.0, maximum=1.0, value=0.5, step=0.01, label="α")
                            pg_text_lam_a   = gr.Slider(minimum=0.0, maximum=1.0, value=0.5, step=0.01, label="λA")
                            pg_text_lam_b   = gr.Slider(minimum=0.0, maximum=1.0, value=0.5, step=0.01, label="λB")

                        gr.Markdown("#### 話者表現グループ（speaker）")
                        with gr.Row():
                            pg_spk_method   = gr.Dropdown(choices=_method_choices, value="weighted_average", label="手法")
                            pg_spk_alpha    = gr.Slider(minimum=0.0, maximum=1.0, value=0.5, step=0.01, label="α")
                            pg_spk_lam_a    = gr.Slider(minimum=0.0, maximum=1.0, value=0.5, step=0.01, label="λA")
                            pg_spk_lam_b    = gr.Slider(minimum=0.0, maximum=1.0, value=0.5, step=0.01, label="λB")

                        gr.Markdown("#### 拡散コアグループ（diffusion_core）")
                        with gr.Row():
                            pg_diff_method  = gr.Dropdown(choices=_method_choices, value="weighted_average", label="手法")
                            pg_diff_alpha   = gr.Slider(minimum=0.0, maximum=1.0, value=0.5, step=0.01, label="α")
                            pg_diff_lam_a   = gr.Slider(minimum=0.0, maximum=1.0, value=0.5, step=0.01, label="λA")
                            pg_diff_lam_b   = gr.Slider(minimum=0.0, maximum=1.0, value=0.5, step=0.01, label="λB")

                        gr.Markdown("#### 入出力グループ（io）")
                        with gr.Row():
                            pg_io_method    = gr.Dropdown(choices=_method_choices, value="weighted_average", label="手法")
                            pg_io_alpha     = gr.Slider(minimum=0.0, maximum=1.0, value=0.5, step=0.01, label="α")
                            pg_io_lam_a     = gr.Slider(minimum=0.0, maximum=1.0, value=0.5, step=0.01, label="λA")
                            pg_io_lam_b     = gr.Slider(minimum=0.0, maximum=1.0, value=0.5, step=0.01, label="λB")

                with gr.Accordion("💉 LoRA的差分注入（オプション）", open=False):
                    gr.Markdown(
                        "ベースモデルに対してドナーモデルの差分を指定スケールで注入します。\n"
                        "`result = base + scale × (donor − base)`\n\n"
                        "注入対象グループを選択してください。"
                    )
                    use_lora = gr.Checkbox(label="LoRA的差分注入を有効にする", value=False)

                    with gr.Row():
                        lora_base = gr.Dropdown(
                            label="ベースモデル",
                            choices=initial_merge_ckpts,
                            value=default_base_path if default_base_path in initial_merge_ckpts else (initial_merge_ckpts[0] if initial_merge_ckpts else None),
                            allow_custom_value=True, scale=4,
                        )
                        lora_refresh_base = gr.Button("🔄", scale=1)

                    with gr.Row():
                        lora_donor = gr.Dropdown(
                            label="ドナーモデル（差分元）",
                            choices=initial_merge_ckpts,
                            value=initial_merge_ckpts[-1] if initial_merge_ckpts else None,
                            allow_custom_value=True, scale=4,
                        )
                        lora_refresh_donor = gr.Button("🔄", scale=1)

                    lora_scale = gr.Slider(
                        label="注入スケール（0=ベースのみ、1=ドナーに完全置換）",
                        minimum=0.0, maximum=1.0, value=0.3, step=0.01,
                    )

                    gr.Markdown("**注入対象グループ**")
                    with gr.Row():
                        lora_grp_text     = gr.Checkbox(label="text（テキスト応答性）",  value=True)
                        lora_grp_speaker  = gr.Checkbox(label="speaker（話者表現）",     value=True)
                        lora_grp_diffusion= gr.Checkbox(label="diffusion_core（拡散コア）", value=False)
                        lora_grp_io       = gr.Checkbox(label="io（入出力）",            value=False)

                with gr.Accordion("💾 出力設定", open=True):
                    with gr.Row():
                        merge_output_format = gr.Dropdown(
                            label="保存形式",
                            choices=[".safetensors", ".pt"],
                            value=".safetensors",
                            info=".safetensors=推論用（推奨）/ .pt=PyTorch標準形式",
                            scale=1,
                        )
                        merge_output_dir = gr.Textbox(
                            label="保存先フォルダ（空欄=checkpoints/merged/）",
                            value="",
                            placeholder=str(CHECKPOINTS_DIR / "merged"),
                            scale=3,
                        )

                merge_run_btn = gr.Button("🔀 マージ実行", variant="primary", size="lg")
                merge_status  = gr.Textbox(label="実行結果", interactive=False, lines=10)

                def _rescan_merge():
                    ckpts = _merge_scan()
                    val = ckpts[-1] if ckpts else None
                    return gr.Dropdown(choices=ckpts, value=val)

                merge_refresh_a.click(_rescan_merge, outputs=[merge_ckpt_a])
                merge_refresh_b.click(_rescan_merge, outputs=[merge_ckpt_b])
                merge_refresh_base.click(_rescan_merge, outputs=[merge_base_ta])
                lora_refresh_base.click(_rescan_merge, outputs=[lora_base])
                lora_refresh_donor.click(_rescan_merge, outputs=[lora_donor])

                _merge_inputs = [
                    merge_ckpt_a, merge_ckpt_b,
                    merge_method, merge_alpha,
                    merge_lambda_a, merge_lambda_b, merge_base_ta,
                    use_partial,
                    pg_text_method,  pg_text_alpha,  pg_text_lam_a,  pg_text_lam_b,
                    pg_spk_method,   pg_spk_alpha,   pg_spk_lam_a,   pg_spk_lam_b,
                    pg_diff_method,  pg_diff_alpha,  pg_diff_lam_a,  pg_diff_lam_b,
                    pg_io_method,    pg_io_alpha,    pg_io_lam_a,    pg_io_lam_b,
                    use_lora,
                    lora_base, lora_donor, lora_scale,
                    lora_grp_text, lora_grp_speaker, lora_grp_diffusion, lora_grp_io,
                    merge_output_format, merge_output_dir,
                ]
                merge_run_btn.click(_run_merge_ui, inputs=_merge_inputs, outputs=[merge_status])


            # ═══════════════════════════════════════════════════════════════
            # タブ8: LoRAマージ
            # ═══════════════════════════════════════════════════════════════
            with gr.Tab("🧬 LoRAマージ"):
                gr.Markdown("## LoRAマージ")

                initial_lm_adapters = scan_lora_adapters_for_merge()
                initial_lm_ckpts    = _merge_scan()
                default_lm_base     = get_default_base_path()

                with gr.Tabs():

                    # ─────────────────────────────────────────────────────
                    # サブタブ1: 通常LoRAマージ（アダプタ同士）
                    # ─────────────────────────────────────────────────────
                    with gr.Tab("🔀 通常LoRAマージ"):
                        gr.Markdown(
                            "## 通常LoRAマージ\n"
                            "LoRAアダプタ同士をマージして新しいLoRAアダプタを生成します。\n"
                            "ベースモデルへの焼き込みは行いません。\n\n"
                            "> **出力**: `lora/lora_merged_*/` フォルダ（推論タブで直接使用可）"
                        )

                        with gr.Row():
                            ll_adapter_a = gr.Dropdown(
                                label="アダプタA",
                                choices=["（なし）"] + initial_lm_adapters,
                                value=initial_lm_adapters[0] if initial_lm_adapters else "（なし）",
                                allow_custom_value=True, scale=4,
                            )
                            ll_ver_a   = gr.Textbox(label="バージョン", interactive=False, scale=1, max_lines=1)
                            ll_ref_a   = gr.Button("🔄", scale=1)

                        with gr.Row():
                            ll_adapter_b = gr.Dropdown(
                                label="アダプタB",
                                choices=["（なし）"] + initial_lm_adapters,
                                value=initial_lm_adapters[1] if len(initial_lm_adapters) > 1 else "（なし）",
                                allow_custom_value=True, scale=4,
                            )
                            ll_ver_b   = gr.Textbox(label="バージョン", interactive=False, scale=1, max_lines=1)
                            ll_ref_b   = gr.Button("🔄", scale=1)

                        with gr.Accordion("⚙️ マージ設定", open=True):
                            ll_method = gr.Dropdown(
                                label="マージ手法",
                                choices=["weighted_average", "slerp", "task_arithmetic"],
                                value="weighted_average",
                                info="weighted_average: 安定 / slerp: ノルム保持 / task_arithmetic: ベースアダプタ必要",
                            )
                            with gr.Row():
                                ll_alpha    = gr.Slider(label="α（アダプタAの割合）", minimum=0.0, maximum=1.0, value=0.5, step=0.01)
                            with gr.Group() as ll_ta_group:
                                gr.Markdown("**Task Arithmetic 設定**")
                                with gr.Row():
                                    ll_lambda_a = gr.Slider(label="λA", minimum=0.0, maximum=1.0, value=0.5, step=0.01)
                                    ll_lambda_b = gr.Slider(label="λB（自動正規化）", minimum=0.0, maximum=1.0, value=0.5, step=0.01)
                                with gr.Row():
                                    ll_base_adapter = gr.Dropdown(
                                        label="ベースアダプタ（Task Arithmetic用）",
                                        choices=["（なし）"] + initial_lm_adapters,
                                        value="（なし）",
                                        allow_custom_value=True, scale=4,
                                    )
                                    ll_ref_base = gr.Button("🔄", scale=1)

                        with gr.Accordion("🧩 部分マージ（グループ別手法）", open=False):
                            gr.Markdown(
                                "有効にすると、レイヤーグループごとに異なる手法でマージできます。\n"
                                "上の「マージ設定」より優先されます。"
                            )
                            ll_use_partial = gr.Checkbox(label="部分マージを有効にする", value=False)
                            _ll_mc = ["weighted_average", "slerp", "task_arithmetic"]
                            with gr.Group():
                                gr.Markdown("#### text グループ")
                                with gr.Row():
                                    ll_pg_text_m = gr.Dropdown(choices=_ll_mc, value="weighted_average", label="手法")
                                    ll_pg_text_a = gr.Slider(minimum=0.0, maximum=1.0, value=0.5, step=0.01, label="α")
                                    ll_pg_text_la= gr.Slider(minimum=0.0, maximum=1.0, value=0.5, step=0.01, label="λA")
                                    ll_pg_text_lb= gr.Slider(minimum=0.0, maximum=1.0, value=0.5, step=0.01, label="λB")
                                gr.Markdown("#### speaker グループ")
                                with gr.Row():
                                    ll_pg_spk_m  = gr.Dropdown(choices=_ll_mc, value="weighted_average", label="手法")
                                    ll_pg_spk_a  = gr.Slider(minimum=0.0, maximum=1.0, value=0.5, step=0.01, label="α")
                                    ll_pg_spk_la = gr.Slider(minimum=0.0, maximum=1.0, value=0.5, step=0.01, label="λA")
                                    ll_pg_spk_lb = gr.Slider(minimum=0.0, maximum=1.0, value=0.5, step=0.01, label="λB")
                                gr.Markdown("#### diffusion_core グループ")
                                with gr.Row():
                                    ll_pg_diff_m = gr.Dropdown(choices=_ll_mc, value="weighted_average", label="手法")
                                    ll_pg_diff_a = gr.Slider(minimum=0.0, maximum=1.0, value=0.5, step=0.01, label="α")
                                    ll_pg_diff_la= gr.Slider(minimum=0.0, maximum=1.0, value=0.5, step=0.01, label="λA")
                                    ll_pg_diff_lb= gr.Slider(minimum=0.0, maximum=1.0, value=0.5, step=0.01, label="λB")
                                gr.Markdown("#### io グループ")
                                with gr.Row():
                                    ll_pg_io_m   = gr.Dropdown(choices=_ll_mc, value="weighted_average", label="手法")
                                    ll_pg_io_a   = gr.Slider(minimum=0.0, maximum=1.0, value=0.5, step=0.01, label="α")
                                    ll_pg_io_la  = gr.Slider(minimum=0.0, maximum=1.0, value=0.5, step=0.01, label="λA")
                                    ll_pg_io_lb  = gr.Slider(minimum=0.0, maximum=1.0, value=0.5, step=0.01, label="λB")

                        with gr.Accordion("💾 出力設定", open=True):
                            ll_output_dir = gr.Textbox(
                                label="保存先フォルダ（空欄=lora/lora_merged_*/）",
                                value="",
                                placeholder=str(Path(__file__).resolve().parent / "lora" / "lora_merged_*"),
                            )

                        ll_run_btn = gr.Button("🔀 LoRAマージ実行", variant="primary", size="lg")
                        ll_status  = gr.Textbox(label="実行結果", interactive=False, lines=10)

                        # ── イベント ──
                        def _ll_rescan():
                            ads = scan_lora_adapters_for_merge()
                            return gr.Dropdown(choices=["（なし）"] + ads)

                        def _ll_ver(v):
                            if not v or str(v).strip() in ("", "（なし）"):
                                return ""
                            return peek_adapter_version(str(v).strip())

                        def _ll_method_change(m):
                            return gr.update(visible=(m == "task_arithmetic"))

                        ll_ref_a.click(_ll_rescan, outputs=[ll_adapter_a])
                        ll_ref_b.click(_ll_rescan, outputs=[ll_adapter_b])
                        ll_ref_base.click(_ll_rescan, outputs=[ll_base_adapter])
                        ll_adapter_a.change(_ll_ver, inputs=[ll_adapter_a], outputs=[ll_ver_a])
                        ll_adapter_b.change(_ll_ver, inputs=[ll_adapter_b], outputs=[ll_ver_b])
                        ll_method.change(_ll_method_change, inputs=[ll_method], outputs=[ll_ta_group])

                        def _run_ll_merge_ui(
                            adapter_a, adapter_b,
                            method, alpha, lambda_a, lambda_b, base_adapter,
                            use_partial,
                            pg_text_m, pg_text_a, pg_text_la, pg_text_lb,
                            pg_spk_m,  pg_spk_a,  pg_spk_la,  pg_spk_lb,
                            pg_diff_m, pg_diff_a, pg_diff_la, pg_diff_lb,
                            pg_io_m,   pg_io_a,   pg_io_la,   pg_io_lb,
                            output_dir,
                        ) -> str:
                            if str(adapter_a).strip() in ("", "（なし）"):
                                return "❌ アダプタAを選択してください。"
                            if str(adapter_b).strip() in ("", "（なし）"):
                                return "❌ アダプタBを選択してください。"

                            def _norm(a, b):
                                t = float(a) + float(b)
                                return (float(a)/t, float(b)/t) if t > 0 else (0.5, 0.5)

                            gm = None
                            if use_partial:
                                def _gc(m, al, la, lb):
                                    if m == "task_arithmetic":
                                        na, nb = _norm(la, lb)
                                        return {"method": m, "lambda_a": na, "lambda_b": nb}
                                    return {"method": m, "alpha": float(al)}
                                gm = {
                                    "text":          _gc(pg_text_m, pg_text_a, pg_text_la, pg_text_lb),
                                    "speaker":       _gc(pg_spk_m,  pg_spk_a,  pg_spk_la,  pg_spk_lb),
                                    "diffusion_core":_gc(pg_diff_m, pg_diff_a, pg_diff_la, pg_diff_lb),
                                    "io":            _gc(pg_io_m,   pg_io_a,   pg_io_la,   pg_io_lb),
                                }

                            ba = str(base_adapter).strip()
                            ba = None if ba in ("", "（なし）") else ba
                            la_n, lb_n = _norm(lambda_a, lambda_b)

                            _, msg = run_lora_lora_merge(
                                adapter_dir_a=str(adapter_a).strip(),
                                adapter_dir_b=str(adapter_b).strip(),
                                method=str(method),
                                alpha=float(alpha),
                                lambda_a=la_n,
                                lambda_b=lb_n,
                                base_adapter_dir=ba,
                                use_partial=bool(use_partial),
                                group_methods=gm,
                                output_dir=str(output_dir).strip() or None,
                            )
                            return msg

                        _ll_inputs = [
                            ll_adapter_a, ll_adapter_b,
                            ll_method, ll_alpha, ll_lambda_a, ll_lambda_b, ll_base_adapter,
                            ll_use_partial,
                            ll_pg_text_m, ll_pg_text_a, ll_pg_text_la, ll_pg_text_lb,
                            ll_pg_spk_m,  ll_pg_spk_a,  ll_pg_spk_la,  ll_pg_spk_lb,
                            ll_pg_diff_m, ll_pg_diff_a, ll_pg_diff_la, ll_pg_diff_lb,
                            ll_pg_io_m,   ll_pg_io_a,   ll_pg_io_la,   ll_pg_io_lb,
                            ll_output_dir,
                        ]
                        ll_run_btn.click(_run_ll_merge_ui, inputs=_ll_inputs, outputs=[ll_status])

                    # ─────────────────────────────────────────────────────
                    # サブタブ2: 本体モデルマージ（焼き込み）
                    # ─────────────────────────────────────────────────────
                    with gr.Tab("🔥 本体モデルマージ（焼き込み）"):
                        gr.Markdown(
                            "## 本体モデルマージ（焼き込み）\n"
                            "LoRAアダプタをベースモデルに焼き込み、マージ済みモデルを生成します。\n\n"
                            "> **出力**: `checkpoints/lora_merged/` フォルダ（推論タブで直接使用可）"
                        )

                        with gr.Row():
                            lm_base_model = gr.Dropdown(
                                label="ベースモデル (.pt / .safetensors)",
                                choices=initial_lm_ckpts,
                                value=default_lm_base if default_lm_base in initial_lm_ckpts else (initial_lm_ckpts[-1] if initial_lm_ckpts else None),
                                allow_custom_value=True, scale=4,
                            )
                            lm_refresh_base = gr.Button("🔄", scale=1)

                        with gr.Accordion("🅰️ アダプタA（必須）", open=True):
                            with gr.Row():
                                lm_adapter_a1 = gr.Dropdown(
                                    label="アダプタA-1",
                                    choices=["（なし）"] + initial_lm_adapters,
                                    value=initial_lm_adapters[0] if initial_lm_adapters else "（なし）",
                                    allow_custom_value=True, scale=4,
                                )
                                lm_scale_a1 = gr.Slider(label="scale", minimum=0.0, maximum=2.0, value=1.0, step=0.05, scale=2)
                                lm_ver_a1   = gr.Textbox(label="バージョン", interactive=False, scale=1, max_lines=1)
                                lm_ref_a1   = gr.Button("🔄", scale=1)
                            with gr.Row():
                                lm_adapter_a2 = gr.Dropdown(
                                    label="アダプタA-2（省略可）",
                                    choices=["（なし）"] + initial_lm_adapters,
                                    value="（なし）",
                                    allow_custom_value=True, scale=4,
                                )
                                lm_scale_a2 = gr.Slider(label="scale", minimum=0.0, maximum=2.0, value=1.0, step=0.05, scale=2)
                                lm_ver_a2   = gr.Textbox(label="バージョン", interactive=False, scale=1, max_lines=1)
                                lm_ref_a2   = gr.Button("🔄", scale=1)
                            with gr.Accordion("🧩 部分焼き込みA", open=False):
                                lm_use_pbake_a = gr.Checkbox(label="部分焼き込みAを有効にする", value=False)
                                with gr.Row():
                                    lm_pbake_a_text  = gr.Checkbox(label="text",           value=True)
                                    lm_pbake_a_spk   = gr.Checkbox(label="speaker",        value=True)
                                    lm_pbake_a_diff  = gr.Checkbox(label="diffusion_core", value=True)
                                    lm_pbake_a_io    = gr.Checkbox(label="io",             value=True)

                        with gr.Accordion("🔀 焼き込み後マージ（オプション）", open=False):
                            lm_post_method = gr.Dropdown(
                                label="マージ手法",
                                choices=["none", "weighted_average", "slerp", "task_arithmetic"],
                                value="none",
                                info="none=焼き込みのみ",
                            )
                            with gr.Group() as lm_post_group:
                                with gr.Row():
                                    lm_post_alpha = gr.Slider(label="α（アダプタA側の割合）", minimum=0.0, maximum=1.0, value=0.5, step=0.01)
                                with gr.Row():
                                    lm_post_lam_a = gr.Slider(label="λA", minimum=0.0, maximum=1.0, value=0.5, step=0.01)
                                    lm_post_lam_b = gr.Slider(label="λB", minimum=0.0, maximum=1.0, value=0.5, step=0.01)
                                with gr.Row():
                                    lm_post_base = gr.Dropdown(
                                        label="Task Arithmetic用ベースモデル（省略時=ベースモデルと同じ）",
                                        choices=["（省略）"] + initial_lm_ckpts,
                                        value="（省略）",
                                        allow_custom_value=True, scale=4,
                                    )
                                    lm_ref_post_base = gr.Button("🔄", scale=1)

                            with gr.Group() as lm_adp_b_group:
                                gr.Markdown("**アダプタB**")
                                with gr.Row():
                                    lm_adapter_b1 = gr.Dropdown(
                                        label="アダプタB-1",
                                        choices=["（なし）"] + initial_lm_adapters,
                                        value="（なし）",
                                        allow_custom_value=True, scale=4,
                                    )
                                    lm_scale_b1 = gr.Slider(label="scale", minimum=0.0, maximum=2.0, value=1.0, step=0.05, scale=2)
                                    lm_ver_b1   = gr.Textbox(label="バージョン", interactive=False, scale=1, max_lines=1)
                                    lm_ref_b1   = gr.Button("🔄", scale=1)
                                with gr.Row():
                                    lm_adapter_b2 = gr.Dropdown(
                                        label="アダプタB-2（省略可）",
                                        choices=["（なし）"] + initial_lm_adapters,
                                        value="（なし）",
                                        allow_custom_value=True, scale=4,
                                    )
                                    lm_scale_b2 = gr.Slider(label="scale", minimum=0.0, maximum=2.0, value=1.0, step=0.05, scale=2)
                                    lm_ver_b2   = gr.Textbox(label="バージョン", interactive=False, scale=1, max_lines=1)
                                    lm_ref_b2   = gr.Button("🔄", scale=1)
                                with gr.Accordion("🧩 部分焼き込みB", open=False):
                                    lm_use_pbake_b = gr.Checkbox(label="部分焼き込みBを有効にする", value=False)
                                    with gr.Row():
                                        lm_pbake_b_text = gr.Checkbox(label="text",           value=True)
                                        lm_pbake_b_spk  = gr.Checkbox(label="speaker",        value=True)
                                        lm_pbake_b_diff = gr.Checkbox(label="diffusion_core", value=True)
                                        lm_pbake_b_io   = gr.Checkbox(label="io",             value=True)

                            with gr.Accordion("🧩 部分マージ（焼き込み後・グループ別）", open=False):
                                lm_use_partial = gr.Checkbox(label="部分マージを有効にする", value=False)
                                _lm_mc = ["weighted_average", "slerp", "task_arithmetic"]
                                with gr.Group():
                                    gr.Markdown("#### text グループ")
                                    with gr.Row():
                                        lm_pg_text_m = gr.Dropdown(choices=_lm_mc, value="weighted_average", label="手法")
                                        lm_pg_text_a = gr.Slider(minimum=0.0, maximum=1.0, value=0.5, step=0.01, label="α")
                                        lm_pg_text_la= gr.Slider(minimum=0.0, maximum=1.0, value=0.5, step=0.01, label="λA")
                                        lm_pg_text_lb= gr.Slider(minimum=0.0, maximum=1.0, value=0.5, step=0.01, label="λB")
                                    gr.Markdown("#### speaker グループ")
                                    with gr.Row():
                                        lm_pg_spk_m  = gr.Dropdown(choices=_lm_mc, value="weighted_average", label="手法")
                                        lm_pg_spk_a  = gr.Slider(minimum=0.0, maximum=1.0, value=0.5, step=0.01, label="α")
                                        lm_pg_spk_la = gr.Slider(minimum=0.0, maximum=1.0, value=0.5, step=0.01, label="λA")
                                        lm_pg_spk_lb = gr.Slider(minimum=0.0, maximum=1.0, value=0.5, step=0.01, label="λB")
                                    gr.Markdown("#### diffusion_core グループ")
                                    with gr.Row():
                                        lm_pg_diff_m = gr.Dropdown(choices=_lm_mc, value="weighted_average", label="手法")
                                        lm_pg_diff_a = gr.Slider(minimum=0.0, maximum=1.0, value=0.5, step=0.01, label="α")
                                        lm_pg_diff_la= gr.Slider(minimum=0.0, maximum=1.0, value=0.5, step=0.01, label="λA")
                                        lm_pg_diff_lb= gr.Slider(minimum=0.0, maximum=1.0, value=0.5, step=0.01, label="λB")
                                    gr.Markdown("#### io グループ")
                                    with gr.Row():
                                        lm_pg_io_m   = gr.Dropdown(choices=_lm_mc, value="weighted_average", label="手法")
                                        lm_pg_io_a   = gr.Slider(minimum=0.0, maximum=1.0, value=0.5, step=0.01, label="α")
                                        lm_pg_io_la  = gr.Slider(minimum=0.0, maximum=1.0, value=0.5, step=0.01, label="λA")
                                        lm_pg_io_lb  = gr.Slider(minimum=0.0, maximum=1.0, value=0.5, step=0.01, label="λB")

                        with gr.Accordion("💾 出力設定", open=True):
                            with gr.Row():
                                lm_output_format = gr.Dropdown(
                                    label="保存形式",
                                    choices=[".safetensors", ".pt"],
                                    value=".safetensors",
                                    scale=1,
                                )
                                lm_output_dir = gr.Textbox(
                                    label="保存先フォルダ（空欄=checkpoints/lora_merged/）",
                                    value="",
                                    placeholder=str(CHECKPOINTS_DIR / "lora_merged"),
                                    scale=3,
                                )

                        lm_run_btn = gr.Button("🔥 焼き込みマージ実行", variant="primary", size="lg")
                        lm_status  = gr.Textbox(label="実行結果", interactive=False, lines=12)

                        # ── イベント ──
                        def _lm_rescan_ckpts():
                            ckpts = _merge_scan()
                            return gr.Dropdown(choices=ckpts, value=ckpts[-1] if ckpts else None)

                        def _lm_rescan_adapters():
                            ads = scan_lora_adapters_for_merge()
                            return gr.Dropdown(choices=["（なし）"] + ads)

                        def _lm_ver(v):
                            if not v or str(v).strip() in ("", "（なし）"):
                                return ""
                            return peek_adapter_version(str(v).strip())

                        def _lm_post_method_change(m):
                            vis = m != "none"
                            return gr.update(visible=vis), gr.update(visible=vis)

                        lm_refresh_base.click(_lm_rescan_ckpts, outputs=[lm_base_model])
                        lm_ref_a1.click(_lm_rescan_adapters, outputs=[lm_adapter_a1])
                        lm_ref_a2.click(_lm_rescan_adapters, outputs=[lm_adapter_a2])
                        lm_ref_b1.click(_lm_rescan_adapters, outputs=[lm_adapter_b1])
                        lm_ref_b2.click(_lm_rescan_adapters, outputs=[lm_adapter_b2])
                        lm_ref_post_base.click(_lm_rescan_ckpts, outputs=[lm_post_base])
                        lm_adapter_a1.change(_lm_ver, inputs=[lm_adapter_a1], outputs=[lm_ver_a1])
                        lm_adapter_a2.change(_lm_ver, inputs=[lm_adapter_a2], outputs=[lm_ver_a2])
                        lm_adapter_b1.change(_lm_ver, inputs=[lm_adapter_b1], outputs=[lm_ver_b1])
                        lm_adapter_b2.change(_lm_ver, inputs=[lm_adapter_b2], outputs=[lm_ver_b2])
                        lm_post_method.change(
                            _lm_post_method_change, inputs=[lm_post_method],
                            outputs=[lm_post_group, lm_adp_b_group],
                        )

                        def _run_lm_bake_ui(
                            base_model,
                            adapter_a1, scale_a1, adapter_a2, scale_a2,
                            use_pbake_a, pbake_a_text, pbake_a_spk, pbake_a_diff, pbake_a_io,
                            post_method, post_alpha, post_lam_a, post_lam_b, post_base,
                            adapter_b1, scale_b1, adapter_b2, scale_b2,
                            use_pbake_b, pbake_b_text, pbake_b_spk, pbake_b_diff, pbake_b_io,
                            use_partial,
                            pg_text_m, pg_text_a, pg_text_la, pg_text_lb,
                            pg_spk_m,  pg_spk_a,  pg_spk_la,  pg_spk_lb,
                            pg_diff_m, pg_diff_a, pg_diff_la, pg_diff_lb,
                            pg_io_m,   pg_io_a,   pg_io_la,   pg_io_lb,
                            output_format, output_dir,
                        ) -> str:
                            def _norm(a, b):
                                t = float(a) + float(b)
                                return (float(a)/t, float(b)/t) if t > 0 else (0.5, 0.5)

                            dirs_a, scales_a = [], []
                            for ad, sc in [(adapter_a1, scale_a1), (adapter_a2, scale_a2)]:
                                if str(ad).strip() not in ("", "（なし）"):
                                    dirs_a.append(str(ad).strip()); scales_a.append(float(sc))
                            if not dirs_a:
                                return "❌ アダプタA-1を選択してください。"

                            dirs_b, scales_b = [], []
                            for ad, sc in [(adapter_b1, scale_b1), (adapter_b2, scale_b2)]:
                                if str(ad).strip() not in ("", "（なし）"):
                                    dirs_b.append(str(ad).strip()); scales_b.append(float(sc))

                            gb_a = {
                                "text": bool(pbake_a_text), "speaker": bool(pbake_a_spk),
                                "diffusion_core": bool(pbake_a_diff), "io": bool(pbake_a_io),
                            } if use_pbake_a else None
                            gb_b = {
                                "text": bool(pbake_b_text), "speaker": bool(pbake_b_spk),
                                "diffusion_core": bool(pbake_b_diff), "io": bool(pbake_b_io),
                            } if use_pbake_b else None

                            gm = None
                            if use_partial:
                                def _gc(m, al, la, lb):
                                    if m == "task_arithmetic":
                                        na, nb = _norm(la, lb)
                                        return {"method": m, "lambda_a": na, "lambda_b": nb}
                                    return {"method": m, "alpha": float(al)}
                                gm = {
                                    "text":          _gc(pg_text_m, pg_text_a, pg_text_la, pg_text_lb),
                                    "speaker":       _gc(pg_spk_m,  pg_spk_a,  pg_spk_la,  pg_spk_lb),
                                    "diffusion_core":_gc(pg_diff_m, pg_diff_a, pg_diff_la, pg_diff_lb),
                                    "io":            _gc(pg_io_m,   pg_io_a,   pg_io_la,   pg_io_lb),
                                }

                            pb_str = str(post_base).strip()
                            pb = None if pb_str in ("", "（省略）") else pb_str
                            lam_a_n, lam_b_n = _norm(post_lam_a, post_lam_b)

                            _, msg = run_lora_merge(
                                base_path=str(base_model),
                                adapter_dirs_a=dirs_a, adapter_scales_a=scales_a,
                                adapter_dirs_b=dirs_b or None,
                                adapter_scales_b=scales_b or None,
                                post_merge_method=str(post_method),
                                post_alpha=float(post_alpha),
                                post_lambda_a=lam_a_n, post_lambda_b=lam_b_n,
                                post_base_path=pb,
                                use_partial=bool(use_partial), group_methods=gm,
                                use_partial_bake_a=bool(use_pbake_a), group_bake_a=gb_a,
                                use_partial_bake_b=bool(use_pbake_b), group_bake_b=gb_b,
                                output_format="safetensors" if output_format == ".safetensors" else "pt",
                                output_dir=str(output_dir).strip() or None,
                            )
                            return msg

                        _lm_inputs = [
                            lm_base_model,
                            lm_adapter_a1, lm_scale_a1, lm_adapter_a2, lm_scale_a2,
                            lm_use_pbake_a, lm_pbake_a_text, lm_pbake_a_spk, lm_pbake_a_diff, lm_pbake_a_io,
                            lm_post_method, lm_post_alpha, lm_post_lam_a, lm_post_lam_b, lm_post_base,
                            lm_adapter_b1, lm_scale_b1, lm_adapter_b2, lm_scale_b2,
                            lm_use_pbake_b, lm_pbake_b_text, lm_pbake_b_spk, lm_pbake_b_diff, lm_pbake_b_io,
                            lm_use_partial,
                            lm_pg_text_m, lm_pg_text_a, lm_pg_text_la, lm_pg_text_lb,
                            lm_pg_spk_m,  lm_pg_spk_a,  lm_pg_spk_la,  lm_pg_spk_lb,
                            lm_pg_diff_m, lm_pg_diff_a, lm_pg_diff_la, lm_pg_diff_lb,
                            lm_pg_io_m,   lm_pg_io_a,   lm_pg_io_la,   lm_pg_io_lb,
                            lm_output_format, lm_output_dir,
                        ]
                        lm_run_btn.click(_run_lm_bake_ui, inputs=_lm_inputs, outputs=[lm_status])


    return demo


def main() -> None:
    parser = argparse.ArgumentParser(description="Irodori-TTS GUI")
    parser.add_argument("--server-name", default="127.0.0.1")
    parser.add_argument("--server-port", type=int, default=7860)
    parser.add_argument("--share", action="store_true")
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    demo = build_ui()
    demo.queue(default_concurrency_limit=1)
    demo.launch(
        server_name=args.server_name,
        server_port=args.server_port,
        share=bool(args.share),
        debug=bool(args.debug),
        theme=gr.themes.Soft(),
        css=_DARK_CSS,
        js=_DARK_JS,
    )


if __name__ == "__main__":
    main()
