#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import threading
import time
from datetime import datetime
from pathlib import Path

import gradio as gr
import yaml

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
DEFAULT_HF_REPO  = "Aratako/Irodori-TTS-500M"
DEFAULT_CONFIG   = "train_v1.yaml"
FIXED_SECONDS    = 30.0
DATASET_TOOLS    = BASE_DIR / "dataset_tools.py"
DEFAULT_DATASET_DIR = BASE_DIR / "my_dataset"

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
    LORA_DIR.mkdir(parents=True, exist_ok=True)
    result = []
    for p in sorted(LORA_DIR.rglob("adapter_config.json")):
        result.append(str(p.parent))
    return result


def _scan_lora_full_adapters() -> list[str]:
    LORA_DIR.mkdir(parents=True, exist_ok=True)
    result = []
    for p in sorted(LORA_DIR.rglob("adapter_config.json")):
        if p.parent.name.endswith("_full"):
            result.append(str(p.parent))
    return result


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

def _build_runtime_key(checkpoint, model_device, model_precision, codec_device, codec_precision, enable_watermark, lora_adapter="（なし）"):
    checkpoint_path = _resolve_checkpoint_path_infer(checkpoint)
    lora_path = None
    if str(lora_adapter).strip() and str(lora_adapter).strip() != "（なし）":
        lp = Path(lora_adapter)
        if lp.is_dir() and (lp / "adapter_config.json").exists():
            lora_path = str(lp)
    return RuntimeKey(
        checkpoint=checkpoint_path,
        model_device=str(model_device),
        codec_repo="facebook/dacvae-watermarked",
        model_precision=str(model_precision),
        codec_device=str(codec_device),
        codec_precision=str(codec_precision),
        enable_watermark=bool(enable_watermark),
        compile_model=False,
        compile_dynamic=False,
        lora_path=lora_path,
    )

def _load_model(checkpoint, model_device, model_precision, codec_device, codec_precision, enable_watermark, lora_adapter="（なし）") -> str:
    runtime_key = _build_runtime_key(checkpoint, model_device, model_precision, codec_device, codec_precision, enable_watermark, lora_adapter)
    _, reloaded = get_cached_runtime(runtime_key)
    status = "モデルを読み込みました" if reloaded else "モデルは既にロード済みです（再利用）"
    lora_info = f"\nlora: {runtime_key.lora_path}" if runtime_key.lora_path else ""
    return (
        f"{status}\n"
        f"checkpoint: {runtime_key.checkpoint}\n"
        f"model_device: {runtime_key.model_device} / {runtime_key.model_precision}\n"
        f"codec_device: {runtime_key.codec_device} / {runtime_key.codec_precision}"
        f"{lora_info}"
    )

def _clear_runtime_cache() -> str:
    clear_cached_runtime()
    return "モデルをメモリから解放しました"

def _run_generation(
    checkpoint, model_device, model_precision, codec_device, codec_precision, enable_watermark,
    lora_adapter, lora_scale,
    text, uploaded_audio, num_steps, seed_raw, cfg_guidance_mode, cfg_scale_text, cfg_scale_speaker,
    cfg_scale_raw, cfg_min_t, cfg_max_t, context_kv_cache,
    truncation_factor_raw, rescale_k_raw, rescale_sigma_raw,
    speaker_kv_scale_raw, speaker_kv_min_t_raw, speaker_kv_max_layers_raw,
) -> tuple[str | None, str, str]:
    def stdout_log(msg: str) -> None:
        print(msg, flush=True)

    runtime_key = _build_runtime_key(checkpoint, model_device, model_precision, codec_device, codec_precision, enable_watermark, lora_adapter)
    if str(text).strip() == "":
        raise ValueError("テキストを入力してください。")

    cfg_scale        = _parse_optional_float(cfg_scale_raw, "cfg_scale")
    truncation_factor= _parse_optional_float(truncation_factor_raw, "truncation_factor")
    rescale_k        = _parse_optional_float(rescale_k_raw, "rescale_k")
    rescale_sigma    = _parse_optional_float(rescale_sigma_raw, "rescale_sigma")
    speaker_kv_scale = _parse_optional_float(speaker_kv_scale_raw, "speaker_kv_scale")
    speaker_kv_min_t = _parse_optional_float(speaker_kv_min_t_raw, "speaker_kv_min_t")
    speaker_kv_max_layers = _parse_optional_int(speaker_kv_max_layers_raw, "speaker_kv_max_layers")
    seed = _parse_optional_int(seed_raw, "seed")

    ref_wav = str(uploaded_audio) if uploaded_audio and str(uploaded_audio).strip() else None
    no_ref  = ref_wav is None

    runtime, reloaded = get_cached_runtime(runtime_key)
    stdout_log(f"[gradio] runtime: {'reloaded' if reloaded else 'reused'}")

    result = runtime.synthesize(
        SamplingRequest(
            text=str(text), ref_wav=ref_wav, ref_latent=None, no_ref=bool(no_ref),
            seconds=FIXED_SECONDS, max_ref_seconds=30.0, max_text_len=None,
            num_steps=int(num_steps), seed=None if seed is None else int(seed),
            cfg_guidance_mode=str(cfg_guidance_mode),
            cfg_scale_text=float(cfg_scale_text), cfg_scale_speaker=float(cfg_scale_speaker),
            cfg_scale=cfg_scale, cfg_min_t=float(cfg_min_t), cfg_max_t=float(cfg_max_t),
            truncation_factor=truncation_factor, rescale_k=rescale_k, rescale_sigma=rescale_sigma,
            context_kv_cache=bool(context_kv_cache),
            speaker_kv_scale=speaker_kv_scale, speaker_kv_min_t=speaker_kv_min_t,
            speaker_kv_max_layers=speaker_kv_max_layers, trim_tail=True,
            lora_scale=float(lora_scale) if runtime_key.lora_path else 1.0,
        ),
        log_fn=stdout_log,
    )

    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    stamp    = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    out_path = save_wav(OUTPUTS_DIR / f"sample_{stamp}.wav", result.audio.float(), result.sample_rate)

    detail_text = "\n".join([
        "runtime: reloaded" if reloaded else "runtime: reused",
        f"seed_used: {result.used_seed}", f"saved: {out_path}", *result.messages,
    ])
    timing_text = _format_timings(result.stage_timings, result.total_to_decode)
    stdout_log(f"[gradio] saved: {out_path}")
    return str(out_path), detail_text, timing_text


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
    audio_col, text_col, speaker_col,
    output_manifest, latent_dir, device,
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

    cmd += [
        "--audio-column",    _s(audio_col, "audio"),
        "--text-column",     _s(text_col, "text"),
        "--output-manifest", _s(output_manifest),
        "--latent-dir",      _s(latent_dir),
        "--device",          _s(device, "cpu"),
    ]
    spk = _s(speaker_col)
    if spk:
        cmd += ["--speaker-column", spk]
    return cmd


def _manifest_cmd_preview(
    data_source_mode, dataset, split, audio_col, text_col, speaker_col,
    output_manifest, latent_dir, device,
) -> str:
    return " ".join(_build_manifest_command(
        data_source_mode, dataset, split, audio_col, text_col, speaker_col,
        output_manifest, latent_dir, device,
    ))


def _run_manifest(
    data_source_mode, dataset, split, audio_col, text_col, speaker_col,
    output_manifest, latent_dir, device,
) -> tuple[str, str]:
    global _active_proc, _active_log_path
    cmd_list = _build_manifest_command(
        data_source_mode, dataset, split, audio_col, text_col, speaker_col,
        output_manifest, latent_dir, device,
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

_LORA_TRAIN_PROC: subprocess.Popen | None = None
_LORA_TRAIN_LOG_PATH: Path | None = None
_LORA_TRAIN_LOG_LOCK = threading.Lock()


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
    global _TRAIN_LOG_PATH, _TRAIN_PROC

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

    def _stream():
        with open(log_path, "w", encoding="utf-8") as f:
            for line in proc.stdout:
                f.write(line)
                f.flush()
        proc.wait()
        _write_tensorboard_events(log_path)

    threading.Thread(target=_stream, daemon=True).start()
    return f"学習開始 (PID {proc.pid})\nログ: {log_path}", cmd


def _stop_train() -> str:
    global _TRAIN_PROC
    with _TRAIN_LOG_LOCK:
        if _TRAIN_PROC is None or _TRAIN_PROC.poll() is not None:
            return "実行中の学習プロセスはありません。"
        _TRAIN_PROC.terminate()
        return f"学習プロセス (PID {_TRAIN_PROC.pid}) に停止シグナルを送信しました。"


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
    return text


def _parse_train_log_metrics():
    if not _PANDAS_AVAILABLE:
        return None
    with _TRAIN_LOG_LOCK:
        path = _TRAIN_LOG_PATH
    if path is None or not path.exists():
        return pd.DataFrame({"step": [], "loss": [], "lr": []})

    rows = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if "step=" not in line or "loss=" not in line:
            continue
        try:
            parts = {k: v for k, v in (p.split("=") for p in line.split() if "=" in p)}
            step = int(parts["step"])
            loss = float(parts["loss"])
            lr   = float(parts.get("lr", 0.0))
            rows.append({"step": step, "loss": loss, "lr": lr})
        except (ValueError, KeyError):
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
        if str(speaker_id).strip():
            cmd += ["--speaker-id", str(speaker_id).strip()]
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
        if str(speaker_id).strip():
            cmd += ["--speaker-id", str(speaker_id).strip()]
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
    global _LORA_TRAIN_PROC, _LORA_TRAIN_LOG_PATH

    with _LORA_TRAIN_LOG_LOCK:
        if _LORA_TRAIN_PROC is not None and _LORA_TRAIN_PROC.poll() is None:
            return "LoRA学習が既に実行中です。停止してから再実行してください。", ""

    cmd_list = _build_lora_train_command(*args)
    cmd_str = " ".join(cmd_list)

    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = LOGS_DIR / f"lora_train_{stamp}.log"

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

    def _stream():
        with open(log_path, "w", encoding="utf-8") as f:
            for line in proc.stdout:
                f.write(line)
                f.flush()
        proc.wait()

    threading.Thread(target=_stream, daemon=True).start()

    warning = ""
    if bool(args[10]) and float(args[26]) <= 0.0:
        warning = "\n⚠️ Early Stopping は valid_ratio=0 のため無効化されました。"
    return f"LoRA学習開始 (PID {proc.pid})\nログ: {log_path}{warning}", cmd_str


def _stop_lora_train() -> str:
    global _LORA_TRAIN_PROC
    with _LORA_TRAIN_LOG_LOCK:
        if _LORA_TRAIN_PROC is None or _LORA_TRAIN_PROC.poll() is not None:
            return "実行中のLoRA学習プロセスはありません。"
        _LORA_TRAIN_PROC.terminate()
        return f"LoRA学習プロセス (PID {_LORA_TRAIN_PROC.pid}) に停止シグナルを送信しました。"


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
                infer_lora_scale = gr.Slider(
                    label="LoRAスケール（0.0=LoRA無効 / 1.0=通常 / >1.0=強調）",
                    minimum=0.0, maximum=2.0, value=1.0, step=0.05, visible=False,
                )

                with gr.Row():
                    model_device = gr.Dropdown(label="モデルデバイス", choices=device_choices, value=default_model_device, scale=1)
                    model_precision = gr.Dropdown(label="モデル精度", choices=model_precision_choices, value=model_precision_choices[0], scale=1)
                    codec_device = gr.Dropdown(label="コーデックデバイス", choices=device_choices, value=default_codec_device, scale=1)
                    codec_precision = gr.Dropdown(label="コーデック精度", choices=codec_precision_choices, value=codec_precision_choices[0], scale=1)
                    enable_watermark = gr.Checkbox(label="ウォーターマーク", value=False, scale=1)

                with gr.Row():
                    load_model_btn  = gr.Button("📥 モデル読み込み", variant="secondary")
                    unload_model_btn= gr.Button("🗑️ メモリ解放",    variant="secondary")
                model_status = gr.Textbox(label="モデルステータス", interactive=False, lines=3)

                gr.Markdown("## 音声生成")
                infer_text = gr.Textbox(label="テキスト（合成したい文章）", lines=4)
                infer_audio = gr.Audio(label="参照音声（省略するとno-referenceモードになります）", type="filepath")

                with gr.Accordion("🎛️ サンプリング設定", open=True):
                    with gr.Row():
                        num_steps = gr.Slider(label="ステップ数（多いほど品質向上、遅くなる）", minimum=1, maximum=120, value=40, step=1)
                        seed_raw  = gr.Textbox(label="シード（空白=ランダム）", value="")
                    with gr.Row():
                        cfg_guidance_mode = gr.Dropdown(
                            label="CFGガイダンスモード（independent=高品質/遅い、joint=バランス、alternating=高速）",
                            choices=["independent", "joint", "alternating"], value="independent",
                        )
                        cfg_scale_text    = gr.Slider(label="CFGスケール: テキスト条件の強度", minimum=0.0, maximum=10.0, value=3.0, step=0.1)
                        cfg_scale_speaker = gr.Slider(label="CFGスケール: 話者条件の強度", minimum=0.0, maximum=10.0, value=5.0, step=0.1)

                with gr.Accordion("🔬 詳細設定（上級者向け）", open=False):
                    cfg_scale_raw = gr.Textbox(label="CFGスケール一括上書き（テキスト・話者を同値に設定）", value="")
                    with gr.Row():
                        cfg_min_t       = gr.Number(label="CFG適用開始タイムステップ", value=0.5)
                        cfg_max_t       = gr.Number(label="CFG適用終了タイムステップ", value=1.0)
                        context_kv_cache= gr.Checkbox(label="コンテキストKVキャッシュ（推論高速化）", value=True)
                    with gr.Row():
                        truncation_factor_raw= gr.Textbox(label="切り捨て係数（0.8〜0.9で平坦化。空=無効）", value="")
                        rescale_k_raw        = gr.Textbox(label="時間的スコア再スケールk（空=無効）", value="")
                        rescale_sigma_raw    = gr.Textbox(label="時間的スコア再スケールsigma（空=無効）", value="")
                    with gr.Row():
                        speaker_kv_scale_raw    = gr.Textbox(label="話者KVスケール（>1で話者性強調。空=無効）", value="")
                        speaker_kv_min_t_raw    = gr.Textbox(label="話者KVスケール適用閾値（デフォルト0.9）", value="0.9")
                        speaker_kv_max_layers_raw= gr.Textbox(label="話者KVスケール適用レイヤー数上限（空=全レイヤー）", value="")

                generate_btn = gr.Button("🎵 生成", variant="primary", size="lg")
                out_audio   = gr.Audio(label="生成音声", type="filepath")
                out_log     = gr.Textbox(label="実行ログ", lines=6)
                out_timing  = gr.Textbox(label="タイミング情報", lines=6)

                hf_dl_btn.click(_download_from_hf, inputs=[hf_repo_id], outputs=[infer_checkpoint, hf_dl_status])
                infer_refresh_btn.click(
                    lambda: gr.Dropdown(choices=_scan_checkpoints(), value=(_scan_checkpoints() or [None])[-1]),
                    outputs=[infer_checkpoint],
                )
                infer_lora_refresh_btn.click(
                    lambda: gr.Dropdown(choices=["（なし）"] + _scan_lora_adapters()),
                    outputs=[infer_lora_adapter],
                )
                infer_lora_adapter.change(
                    lambda v: gr.Slider(visible=(str(v).strip() not in ("", "（なし）"))),
                    inputs=[infer_lora_adapter], outputs=[infer_lora_scale],
                )
                model_device.change(_on_model_device_change, inputs=[model_device], outputs=[model_precision])
                codec_device.change(_on_codec_device_change, inputs=[codec_device], outputs=[codec_precision])
                load_model_btn.click(_load_model,
                    inputs=[infer_checkpoint, model_device, model_precision, codec_device, codec_precision, enable_watermark, infer_lora_adapter],
                    outputs=[model_status])
                unload_model_btn.click(_clear_runtime_cache, outputs=[model_status])
                generate_btn.click(_run_generation,
                    inputs=[
                        infer_checkpoint, model_device, model_precision, codec_device, codec_precision, enable_watermark,
                        infer_lora_adapter, infer_lora_scale,
                        infer_text, infer_audio, num_steps, seed_raw, cfg_guidance_mode,
                        cfg_scale_text, cfg_scale_speaker, cfg_scale_raw, cfg_min_t, cfg_max_t,
                        context_kv_cache, truncation_factor_raw, rescale_k_raw, rescale_sigma_raw,
                        speaker_kv_scale_raw, speaker_kv_min_t_raw, speaker_kv_max_layers_raw,
                    ],
                    outputs=[out_audio, out_log, out_timing],
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

                def _auto_fill_columns(file_path: str, mode: str):
                    if mode == "HuggingFaceデータセット":
                        return gr.update(), gr.update(), gr.update(), "HFデータセット: 列名を手動で入力してください。"
                    headers = _read_csv_headers(file_path)
                    if not headers:
                        return gr.update(), gr.update(), gr.update(), "⚠️ 列名を取得できませんでした。ファイルパスを確認してください。"
                    audio_choices = ["audio"] + [h for h in headers if h not in {"audio", "file_name"}]
                    exclude = {"file_name", "audio", "speaker_id", "speaker"}
                    text_choices = [h for h in headers if h not in {"file_name", "audio"}]
                    text_guess = next((h for h in headers if h not in exclude), text_choices[0] if text_choices else "text")
                    spk_choices = [""] + [h for h in headers if h not in {"file_name", "audio", text_guess}]
                    status = f"✅ 列名を取得しました: {headers}"
                    return (
                        gr.update(choices=audio_choices, value="audio"),
                        gr.update(choices=text_choices, value=text_guess),
                        gr.update(choices=spk_choices, value=""),
                        status,
                    )

                pm_dataset.change(
                    _auto_fill_columns, inputs=[pm_dataset, pm_data_source],
                    outputs=[pm_audio_col, pm_text_col, pm_speaker_col, pm_col_status],
                )
                pm_data_source.change(
                    _auto_fill_columns, inputs=[pm_dataset, pm_data_source],
                    outputs=[pm_audio_col, pm_text_col, pm_speaker_col, pm_col_status],
                )

                pm_cmd_preview = gr.Textbox(label="📋 実行コマンドプレビュー", interactive=False, lines=3)

                def _get_pm_inputs_values(mode, local_path, hf_name, split,
                                          audio_col, text_col, speaker_col,
                                          output_manifest, latent_dir, device):
                    src_mode = {"ローカルCSV": "local_csv",
                                "ローカルJSONL": "local_jsonl",
                                "HuggingFaceデータセット": "hf_dataset"}.get(mode, "local_csv")
                    dataset = hf_name if mode == "HuggingFaceデータセット" else local_path
                    return src_mode, dataset, split, audio_col, text_col, speaker_col, output_manifest, latent_dir, device

                _pm_all_inputs = [pm_data_source, pm_dataset, pm_hf_name, pm_split,
                                  pm_audio_col, pm_text_col, pm_speaker_col,
                                  pm_output_manifest, pm_latent_dir, pm_device]

                def _update_pm_cmd(mode, local_path, hf_name, split,
                                   audio_col, text_col, speaker_col,
                                   output_manifest, latent_dir, device):
                    args = _get_pm_inputs_values(mode, local_path, hf_name, split,
                                                 audio_col, text_col, speaker_col,
                                                 output_manifest, latent_dir, device)
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
                                     audio_col, text_col, speaker_col,
                                     output_manifest, latent_dir, device):
                    args = _get_pm_inputs_values(mode, local_path, hf_name, split,
                                                 audio_col, text_col, speaker_col,
                                                 output_manifest, latent_dir, device)
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
                        CHECKPOINTS_DIR / "Aratako_Irodori-TTS-500M" / "model.safetensors"
                    )
                    gr.Markdown(
                        "**--resume オプション設定**\n\n"
                        "- **オフ（スクラッチ学習）**: モデルを最初からランダム初期化して学習します。\n"
                        "- **オン・パス未入力**: `checkpoints/Aratako_Irodori-TTS-500M/model.safetensors` が"
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
                            str(CHECKPOINTS_DIR / "Aratako_Irodori-TTS-500M" / "model.safetensors")
                            if (CHECKPOINTS_DIR / "Aratako_Irodori-TTS-500M" / "model.safetensors").exists()
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
                        ds_speaker_id = gr.Textbox(
                            label="話者ID（省略可・全ファイルに付与）",
                            value="",
                            placeholder="例: SPEAKER_A",
                            scale=2,
                        )
                        ds_recursive_caption = gr.Checkbox(label="サブフォルダも検索", value=False, scale=1)
                    ds_model_cache_dir = gr.Textbox(
                        label="Whisperモデルキャッシュフォルダ",
                        value=str(CHECKPOINTS_DIR / "whisper"),
                        info="モデルが存在しない場合は自動ダウンロードされます。空欄にするとHFデフォルト (~/.cache/huggingface/hub) に保存されます。",
                    )

                with gr.Accordion("🎭 絵文字キャプション設定（オプション）", open=False):
                    gr.Markdown(
                        "有効にすると、Whisperキャプション完了後に音響特徴量とLLMを使って"
                        "**Irodori-TTS互換の絵文字キャプション**を自動生成します。\n\n"
                        "> 必要ライブラリ: `pip install librosa openai`  \n"
                        "> Manifest出力設定が **CSV形式** の場合のみ動作します。"
                    )
                    with gr.Row():
                        ec_enabled = gr.Checkbox(
                            label="🎭 絵文字キャプションを有効にする",
                            value=False,
                            scale=1,
                            info="チェックを入れると通常キャプション完了後に絵文字キャプションを続けて実行します。",
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
                        "- **CSV**: `audio_path,text,speaker_id` — Excelや各種ツールで開きやすい汎用形式\n"
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
                    return gr.update(open=show_slice), gr.update(open=show_caption)

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

                    status, cmd = _start_dataset_job(*base_args)

                    if ec_enabled_:
                        output_fmt    = str(args[14]).strip()
                        if output_fmt != "CSV":
                            status += "\n⚠️ 絵文字キャプションはCSV形式のみ対応です。出力形式をCSVに変更してください。"
                        else:
                            manifest_dir  = str(args[12]).strip()
                            manifest_name = str(args[13]).strip()
                            csv_path      = str(Path(manifest_dir) / f"{manifest_name}.csv")
                            mode_         = str(args[0])
                            wav_dir_      = str(args[11]).strip() if mode_ == "キャプションのみ" else str(args[2]).strip()

                            def _wait_and_emoji(csv_path=csv_path, wav_dir_=wav_dir_, ec_api_=ec_api_, ec_api_key_=ec_api_key_):
                                while True:
                                    with _DS_LOG_LOCK:
                                        proc = _DS_PROC
                                    if proc is None or proc.poll() is not None:
                                        break
                                    time.sleep(1)
                                with _DS_LOG_LOCK:
                                    rc = _DS_PROC.returncode if _DS_PROC else -1
                                if rc == 0:
                                    _run_emoji_caption_inline(csv_path, wav_dir_, ec_api_, ec_api_key_)

                            threading.Thread(target=_wait_and_emoji, daemon=True).start()
                            status += f"\n🎭 絵文字キャプション: 通常処理完了後に自動実行します（API: {ec_api_}）"

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
    )


if __name__ == "__main__":
    main()
