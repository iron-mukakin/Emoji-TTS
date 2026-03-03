#!/usr/bin/env python3
"""LoRA差分学習スクリプト for Irodori-TTS.

train.py をベースとして LoRA 差分学習に特化。
peft ライブラリを使用し、操作面・引数仕様を train.py に最大限準拠。
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import os
import random
import re
import sys
from contextlib import nullcontext
from dataclasses import asdict, replace
from datetime import datetime
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from irodori_tts.config import (
    ModelConfig,
    TrainConfig,
    dump_configs,
    load_experiment_yaml,
    merge_dataclass_overrides,
)
from irodori_tts.dataset import LatentTextDataset, TTSCollator
from irodori_tts.model import TextToLatentRFDiT
from irodori_tts.optim import build_optimizer_extended, build_scheduler, current_lr
from irodori_tts.progress import TrainProgress
from irodori_tts.rf import (
    rf_interpolate,
    rf_velocity_target,
    sample_logit_normal_t,
    sample_stratified_logit_normal_t,
)
from irodori_tts.tokenizer import PretrainedTextTokenizer

# train.py から共通関数を流用
from train import (
    EMAModel,
    EarlyStopping,
    apply_attention_backend,
    apply_gradient_checkpointing,
    build_text_tokenizer,
    cli_provided,
    echo_style_masked_mse,
    enforce_periodic_checkpoint_limit,
    list_best_val_loss_checkpoints,
    maybe_save_best_val_loss_checkpoint,
    prune_best_val_loss_checkpoints,
    run_validation,
    set_seed,
    split_train_valid_indices,
    validate_text_backbone_dim,
)

WANDB_MODES = {"online", "offline", "disabled"}

# LoRAデフォルト出力先（このスクリプトと同階層の lora/ フォルダ）
_DEFAULT_LORA_DIR = Path(__file__).resolve().parent / "lora"

# pyファイル基準のcheckpointsフォルダ
_PROJECT_CHECKPOINTS_DIR = Path(__file__).resolve().parent / "checkpoints"
_HF_TOKENIZER_CACHE_DIR = _PROJECT_CHECKPOINTS_DIR / "tokenizers"

# LoRAターゲットモジュール
DEFAULT_TARGET_MODULES = ["wq", "wk", "wv", "wo"]
EXTENDED_TARGET_MODULES = [
    "wq", "wk", "wv", "wo",         # JointAttention コア
    "wk_text", "wv_text",           # テキスト KV 投影
    "wk_speaker", "wv_speaker",     # 話者 KV 投影
    "w1", "w2", "w3",               # SwiGLU MLP
]

LORA_CHECKPOINT_STEP_RE = re.compile(r"^lora_checkpoint_(\d+)_(ema|full)$")


# ---------------------------------------------------------------------------
# LoRA チェックポイント保存
# ---------------------------------------------------------------------------

def save_lora_checkpoint(
    output_dir: Path,
    model,  # PeftModel
    optimizer: torch.optim.Optimizer,
    scheduler,
    step: int,
    base_model_path: str,
    base_model_cfg: dict,
    lora_cfg_dict: dict,
    train_cfg: TrainConfig,
    ema_model: EMAModel | None = None,
    save_full: bool = False,
) -> None:
    """LoRAチェックポイントを保存する。

    保存方針:
    - EMAが有効な場合:
        lora_checkpoint_XXXXXXX_ema/ ... EMA版（推論用）常に保存
        lora_checkpoint_XXXXXXX_full/ ... フル版（再開用）--save-full時のみ
    - EMAが無効な場合:
        lora_checkpoint_XXXXXXX_ema/ ... 生重みを EMA版フォルダに保存
    """
    from safetensors.torch import save_file as save_safetensors

    ema_dir = output_dir / f"lora_checkpoint_{step:07d}_ema"
    ema_dir.mkdir(parents=True, exist_ok=True)

    # adapter_config.json は peft が自動生成するが、手動でコピーしておく
    # EMA版: EMA重みで adapter_model.safetensors を保存
    if ema_model is not None:
        # EMA適用してアダプタ重みを取得
        ema_model.apply_shadow(model)
        _save_lora_adapter_safetensors(model, ema_dir)
        ema_model.restore(model)
    else:
        _save_lora_adapter_safetensors(model, ema_dir)

    # フル版保存 (EMA + optimizer + scheduler + ema_shadow)
    if save_full:
        full_dir = output_dir / f"lora_checkpoint_{step:07d}_full"
        full_dir.mkdir(parents=True, exist_ok=True)

        # 生重みで adapter_model.safetensors を保存
        _save_lora_adapter_safetensors(model, full_dir)

        # optimizer / scheduler
        torch.save(optimizer.state_dict(), full_dir / "optimizer.pt")
        if scheduler is not None:
            torch.save(scheduler.state_dict(), full_dir / "scheduler.pt")

        # EMA shadow重みを独立保存（JSON肥大化防止）
        if ema_model is not None:
            torch.save(ema_model.shadow, full_dir / "ema_shadow.pt")

        # train_state.json
        import hashlib
        base_sha256 = ""
        try:
            bp = Path(base_model_path)
            if bp.is_file():
                h = hashlib.sha256()
                with open(bp, "rb") as f:
                    for chunk in iter(lambda: f.read(1 << 20), b""):
                        h.update(chunk)
                base_sha256 = h.hexdigest()
        except Exception:
            pass

        train_state = {
            "step": step,
            "base_model_path": base_model_path,
            "base_model_sha256": base_sha256,
            "base_model_config": base_model_cfg,
            "lora_config": lora_cfg_dict,
            "train_config": asdict(train_cfg),
            "ema_decay": ema_model.decay if ema_model is not None else None,
        }
        (full_dir / "train_state.json").write_text(
            json.dumps(train_state, ensure_ascii=False, indent=2), encoding="utf-8"
        )


def _save_lora_adapter_safetensors(model, out_dir: Path) -> None:
    """PeftModel から LoRA差分アダプタを safetensors 形式で保存する。"""
    from safetensors.torch import save_file as save_safetensors

    # adapter_config.json を保存
    model.save_pretrained(str(out_dir))

    # save_pretrained が生成した adapter_model.bin を safetensors に変換
    # peft >= 0.5 は safetensors を自動で吐くが、念のため明示的に処理
    bin_path = out_dir / "adapter_model.bin"
    safe_path = out_dir / "adapter_model.safetensors"
    if bin_path.exists() and not safe_path.exists():
        weights = torch.load(bin_path, map_location="cpu", weights_only=True)
        tensors = {k: v.contiguous().cpu() for k, v in weights.items()}
        save_safetensors(tensors, str(safe_path))
        bin_path.unlink()


def save_lora_final(
    output_dir: Path,
    model,
    ema_model: EMAModel | None,
) -> None:
    """学習完了時に lora_checkpoint_final_ema/ を保存する。"""
    final_dir = output_dir / "lora_checkpoint_final_ema"
    final_dir.mkdir(parents=True, exist_ok=True)
    if ema_model is not None:
        ema_model.apply_shadow(model)
        _save_lora_adapter_safetensors(model, final_dir)
        ema_model.restore(model)
    else:
        _save_lora_adapter_safetensors(model, final_dir)


def save_lora_best_val(
    output_dir: Path,
    model,
    ema_model: EMAModel | None,
    step: int,
    val_loss: float,
) -> Path:
    """ベストバリデーション損失時の LoRA チェックポイントを保存する。"""
    dir_name = f"lora_checkpoint_best_val_loss_{step:07d}_{val_loss:.6f}_ema"
    best_dir = output_dir / dir_name
    best_dir.mkdir(parents=True, exist_ok=True)
    if ema_model is not None:
        ema_model.apply_shadow(model)
        _save_lora_adapter_safetensors(model, best_dir)
        ema_model.restore(model)
    else:
        _save_lora_adapter_safetensors(model, best_dir)
    return best_dir


def _load_base_model(base_model_path: str, device: torch.device) -> tuple[TextToLatentRFDiT, dict]:
    """ベースモデルをロードして (model, model_cfg_dict) を返す。"""
    p = Path(base_model_path)
    if not p.is_file():
        raise FileNotFoundError(f"Base model not found: {base_model_path}")

    if p.suffix.lower() == ".safetensors":
        from safetensors.torch import load_file
        from safetensors import safe_open
        weights = load_file(str(p), device="cpu")
        with safe_open(str(p), framework="pt", device="cpu") as handle:
            metadata = handle.metadata() or {}
        cfg_json = metadata.get("config_json")
        if cfg_json is None:
            raise ValueError(f"Safetensors file has no 'config_json' metadata: {p}")
        flat_cfg = json.loads(cfg_json)
        # inference config keys を除外してモデル設定を取得
        _INF_KEYS = {"max_text_len", "fixed_target_latent_steps"}
        model_cfg_dict = {k: v for k, v in flat_cfg.items() if k not in _INF_KEYS}
        model_cfg = ModelConfig(**model_cfg_dict)
        model = TextToLatentRFDiT(model_cfg).to(device)
        missing, unexpected = model.load_state_dict(weights, strict=False)
        if missing:
            print(f"  Base model missing keys: {len(missing)}")
        if unexpected:
            print(f"  Base model unexpected keys: {len(unexpected)}")
    else:
        ckpt = torch.load(str(p), map_location=device, weights_only=True)
        model_cfg_dict = ckpt.get("model_config")
        if not isinstance(model_cfg_dict, dict):
            raise ValueError(f"Checkpoint missing model_config: {p}")
        model_cfg = ModelConfig(**model_cfg_dict)
        model = TextToLatentRFDiT(model_cfg).to(device)
        model.load_state_dict(ckpt["model"])

    return model, model_cfg_dict


# ---------------------------------------------------------------------------
# メイン
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="LoRA fine-tuning for Irodori-TTS.")

    # ── LoRA固有引数 ──────────────────────────────────────────────
    parser.add_argument("--base-model", required=True,
                        help="ベースモデルパス (.pt または .safetensors)。必須。")
    parser.add_argument("--lora-rank", type=int, default=16,
                        help="LoRA ランク (デフォルト: 16)。")
    parser.add_argument("--lora-alpha", type=float, default=32.0,
                        help="LoRA スケール (デフォルト: 32)。")
    parser.add_argument("--lora-dropout", type=float, default=0.05,
                        help="LoRA ドロップアウト率 (デフォルト: 0.05)。")
    parser.add_argument("--target-modules", default="wq,wk,wv,wo",
                        help="LoRA適用モジュール カンマ区切り (デフォルト: wq,wk,wv,wo)。")
    parser.add_argument("--run-name", default=None,
                        help="出力サブフォルダ名 (デフォルト: タイムスタンプ自動生成)。")
    parser.add_argument("--resume-lora", default=None,
                        help="既存 _full フォルダパスを指定して Resume。")

    # ── train.py 共通引数 ─────────────────────────────────────────
    parser.add_argument("--manifest", required=True,
                        help="JSONL マニフェストファイルパス。")
    parser.add_argument("--output-dir", default=None,
                        help=f"LoRA出力フォルダ (デフォルト: lora/{{run_name}}/)。")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--precision", choices=["fp32", "bf16"], default="bf16")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--max-text-len", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--optimizer", choices=["adamw", "muon", "lion", "ademamix", "sgd"], default="adamw")
    parser.add_argument("--adam-beta1", type=float, default=0.9)
    parser.add_argument("--adam-beta2", type=float, default=0.999)
    parser.add_argument("--adam-eps", type=float, default=1e-8)
    parser.add_argument("--muon-momentum", type=float, default=0.95)
    parser.add_argument("--lr-scheduler", choices=["none", "cosine", "wsd"], default="none")
    parser.add_argument("--warmup-steps", type=int, default=0)
    parser.add_argument("--stable-steps", type=int, default=0)
    parser.add_argument("--min-lr-scale", type=float, default=0.1)
    parser.add_argument("--max-steps", type=int, default=1000)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--save-every", type=int, default=100)
    parser.add_argument("--valid-ratio", type=float, default=0.0)
    parser.add_argument("--valid-every", type=int, default=0)
    parser.add_argument("--early-stopping", action="store_true", default=False)
    parser.add_argument("--early-stopping-patience", type=int, default=3)
    parser.add_argument("--early-stopping-min-delta", type=float, default=0.01)
    parser.add_argument("--ema-decay", type=float, default=None)
    parser.add_argument("--clip-grad-norm", type=float, default=1.0)
    parser.add_argument("--save-full", action="store_true", default=False,
                        help="EMA版に加えてフル版 (_full/) も保存する。Resume前提の学習時に指定。")
    parser.add_argument("--attention-backend",
                        choices=["sdpa", "flash2", "sage", "eager"], default="sdpa")
    parser.add_argument("--grad-checkpoint", action="store_true", default=False)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--text-condition-dropout", type=float, default=0.1)
    parser.add_argument("--speaker-condition-dropout", type=float, default=0.1)
    parser.add_argument("--timestep-stratified", action="store_true")
    parser.add_argument("--max-latent-steps", type=int, default=750)
    parser.add_argument("--fixed-target-latent-steps", type=int, default=None)
    parser.add_argument("--fixed-target-full-mask", action="store_true")
    parser.add_argument("--wandb", dest="wandb_enabled", action="store_true", default=False)
    parser.add_argument("--wandb-project", default=None)
    parser.add_argument("--wandb-entity", default=None)
    parser.add_argument("--wandb-run-name", default=None)
    parser.add_argument("--wandb-mode", choices=sorted(WANDB_MODES), default="online")
    parser.add_argument("--lion-beta1", type=float, default=0.9)
    parser.add_argument("--lion-beta2", type=float, default=0.99)
    parser.add_argument("--ademamix-alpha", type=float, default=5.0)
    parser.add_argument("--ademamix-beta3", type=float, default=0.9999)

    args = parser.parse_args()

    # run_name 決定
    run_name = args.run_name if args.run_name else datetime.now().strftime("lora_%Y%m%d_%H%M%S")

    # 出力先決定
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = _DEFAULT_LORA_DIR / run_name
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── 基本設定 ──────────────────────────────────────────────────
    device = torch.device(args.device)
    set_seed(args.seed)

    precision = args.precision
    if precision == "bf16" and device.type != "cuda":
        print("warning: bf16 on non-CUDA. Falling back to fp32.")
        precision = "fp32"
    if precision == "bf16" and not torch.cuda.is_bf16_supported():
        print("warning: bf16 not supported. Falling back to fp32.")
        precision = "fp32"
    use_bf16 = precision == "bf16"

    print(f"LoRA学習開始: run_name={run_name}")
    print(f"出力先: {output_dir}")
    print(f"ベースモデル: {args.base_model}")
    print(f"デバイス: {device} / 精度: {precision}")

    # ── ベースモデルロード ─────────────────────────────────────────
    print("ベースモデルをロード中...")
    raw_model, base_model_cfg_dict = _load_base_model(args.base_model, device)
    model_cfg = ModelConfig(**base_model_cfg_dict)
    print(f"ベースモデルロード完了: {args.base_model}")

    # ── Attention Backend ──────────────────────────────────────────
    if device.type == "cuda":
        apply_attention_backend(raw_model, args.attention_backend)

    # ── 勾配チェックポイント ───────────────────────────────────────
    if args.grad_checkpoint:
        apply_gradient_checkpointing(raw_model)

    # ── LoRA 設定 ──────────────────────────────────────────────────
    try:
        from peft import LoraConfig, get_peft_model, PeftModel
    except ImportError as exc:
        raise RuntimeError(
            "peft ライブラリが必要です。`pip install peft` を実行してください。"
        ) from exc

    target_modules = [m.strip() for m in args.target_modules.split(",") if m.strip()]
    lora_cfg_dict = {
        "r": args.lora_rank,
        "lora_alpha": args.lora_alpha,
        "lora_dropout": args.lora_dropout,
        "target_modules": target_modules,
    }

    step = 0
    optimizer_state = None
    scheduler_state = None

    if args.resume_lora:
        # _full フォルダから Resume
        resume_dir = Path(args.resume_lora)
        print(f"LoRA Resume: {resume_dir}")
        model = PeftModel.from_pretrained(raw_model, str(resume_dir), is_trainable=True)

        train_state_path = resume_dir / "train_state.json"
        if train_state_path.exists():
            train_state = json.loads(train_state_path.read_text(encoding="utf-8"))
            step = int(train_state.get("step", 0))
            print(f"Resume: step={step} を復元")

        opt_path = resume_dir / "optimizer.pt"
        if opt_path.exists():
            optimizer_state = torch.load(str(opt_path), map_location="cpu", weights_only=True)

        sched_path = resume_dir / "scheduler.pt"
        if sched_path.exists():
            scheduler_state = torch.load(str(sched_path), map_location="cpu", weights_only=True)
    else:
        # 新規LoRA学習
        lora_config = LoraConfig(
            r=args.lora_rank,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            target_modules=target_modules,
            bias="none",
        )
        model = get_peft_model(raw_model, lora_config)

    model.print_trainable_parameters()
    model = model.to(device)
    if use_bf16:
        model = model.to(dtype=torch.bfloat16)
    model.train()

    # ── トークナイザー ──────────────────────────────────────────────
    _HF_TOKENIZER_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    tokenizer = build_text_tokenizer(model_cfg, local_files_only=False)

    # ── データセット ────────────────────────────────────────────────
    full_dataset = LatentTextDataset(
        manifest_path=args.manifest,
        latent_dim=model_cfg.latent_dim,
        max_latent_steps=args.max_latent_steps,
    )

    train_dataset = full_dataset
    valid_dataset = None
    if args.valid_ratio > 0.0:
        train_indices, valid_indices = split_train_valid_indices(
            num_samples=len(full_dataset),
            valid_ratio=args.valid_ratio,
            seed=args.seed,
        )
        train_dataset = LatentTextDataset(
            manifest_path=args.manifest,
            latent_dim=model_cfg.latent_dim,
            max_latent_steps=args.max_latent_steps,
            subset_indices=train_indices,
        )
        valid_dataset = LatentTextDataset(
            manifest_path=args.manifest,
            latent_dim=model_cfg.latent_dim,
            max_latent_steps=args.max_latent_steps,
            subset_indices=valid_indices,
        )
        print(f"バリデーション分割: train={len(train_dataset)} valid={len(valid_dataset)}")

    collator = TTSCollator(
        tokenizer=tokenizer,
        latent_dim=model_cfg.latent_dim,
        latent_patch_size=model_cfg.latent_patch_size,
        fixed_target_latent_steps=args.fixed_target_latent_steps,
        fixed_target_full_mask=args.fixed_target_full_mask,
        max_text_len=args.max_text_len,
    )

    drop_last = len(train_dataset) >= args.batch_size
    loader = DataLoader(
        dataset=train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        collate_fn=collator,
        drop_last=drop_last,
        persistent_workers=(args.num_workers > 0),
    )

    valid_loader = None
    if valid_dataset is not None:
        valid_loader = DataLoader(
            dataset=valid_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=(device.type == "cuda"),
            collate_fn=collator,
            drop_last=False,
        )

    # ── オプティマイザ（LoRAパラメータのみ） ────────────────────────
    # TrainConfig を仮生成してbuild_optimizer_extendedに渡す
    train_cfg = TrainConfig(
        manifest_path=args.manifest,
        output_dir=str(output_dir),
        batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        num_workers=args.num_workers,
        precision=precision,
        optimizer=args.optimizer,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        adam_beta1=args.adam_beta1,
        adam_beta2=args.adam_beta2,
        adam_eps=args.adam_eps,
        muon_momentum=args.muon_momentum,
        lr_scheduler=args.lr_scheduler,
        warmup_steps=args.warmup_steps,
        stable_steps=args.stable_steps,
        min_lr_scale=args.min_lr_scale,
        max_steps=args.max_steps,
        max_text_len=args.max_text_len,
        log_every=args.log_every,
        save_every=args.save_every,
        valid_ratio=args.valid_ratio,
        valid_every=args.valid_every,
        seed=args.seed,
        text_condition_dropout=args.text_condition_dropout,
        speaker_condition_dropout=args.speaker_condition_dropout,
        timestep_stratified=args.timestep_stratified,
        max_latent_steps=args.max_latent_steps,
        fixed_target_latent_steps=args.fixed_target_latent_steps,
        fixed_target_full_mask=args.fixed_target_full_mask,
        wandb_enabled=args.wandb_enabled,
        wandb_project=args.wandb_project,
        wandb_entity=args.wandb_entity,
        wandb_run_name=args.wandb_run_name,
        wandb_mode=args.wandb_mode,
    )

    # LoRAパラメータのみをオプティマイザに渡す
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    print(f"学習対象パラメータ数: {sum(p.numel() for p in trainable_params):,}")

    optimizer = build_optimizer_extended(
        model,
        train_cfg,
        optimizer_name=args.optimizer,
        lion_betas=(args.lion_beta1, args.lion_beta2),
        ademamix_alpha=args.ademamix_alpha,
        ademamix_beta3=args.ademamix_beta3,
        trainable_params_override=trainable_params,
    )

    if optimizer_state is not None:
        try:
            optimizer.load_state_dict(optimizer_state)
            print("オプティマイザ状態を復元しました。")
        except Exception as e:
            print(f"warning: オプティマイザ状態の復元に失敗: {e}")

    scheduler = build_scheduler(optimizer, train_cfg)
    if scheduler_state is not None and scheduler is not None:
        try:
            scheduler.load_state_dict(scheduler_state)
            print("スケジューラ状態を復元しました。")
        except Exception as e:
            print(f"warning: スケジューラ状態の復元に失敗: {e}")

    print(f"Optimizer={args.optimizer} lr={current_lr(optimizer):.3e}")

    # ── EMA ──────────────────────────────────────────────────────
    ema_model: EMAModel | None = None
    if args.ema_decay is not None:
        # LoRAパラメータのみを対象にEMAを構築
        class LoRAEMAModel(EMAModel):
            def __init__(self, model, decay):
                self.decay = decay
                self.shadow: dict[str, torch.Tensor] = {}
                self.backup: dict[str, torch.Tensor] = {}
                for name, param in model.named_parameters():
                    if param.requires_grad:
                        self.shadow[name] = param.data.clone().float()

        ema_model = LoRAEMAModel(model, decay=args.ema_decay)

        # Resume時にema_shadow.ptがあれば復元
        if args.resume_lora:
            shadow_path = Path(args.resume_lora) / "ema_shadow.pt"
            if shadow_path.exists():
                ema_model.shadow = torch.load(str(shadow_path), map_location="cpu", weights_only=True)
                print("EMA shadow重みを復元しました。")

        print(f"EMA有効 (decay={args.ema_decay})")

    # ── Early Stopping ────────────────────────────────────────────
    early_stopper: EarlyStopping | None = None
    if args.early_stopping:
        if args.valid_ratio <= 0.0:
            raise ValueError("--early-stopping には --valid-ratio > 0 が必要です。")
        early_stopper = EarlyStopping(
            patience=args.early_stopping_patience,
            min_delta=args.early_stopping_min_delta,
            mode="min",
        )
        print(f"Early Stopping: patience={args.early_stopping_patience}")

    # ── W&B ──────────────────────────────────────────────────────
    wandb_run = None
    if args.wandb_enabled:
        try:
            import wandb
            wandb_run = wandb.init(
                project=args.wandb_project,
                entity=args.wandb_entity,
                name=args.wandb_run_name or run_name,
                mode=args.wandb_mode,
                dir=str(output_dir),
                config={
                    "base_model": args.base_model,
                    "lora": lora_cfg_dict,
                    "train": asdict(train_cfg),
                    "script": "lora_train.py",
                },
            )
            print(f"W&B 有効: project={args.wandb_project}")
        except ImportError as exc:
            raise RuntimeError("wandb が未インストールです。`pip install wandb` を実行してください。") from exc

    # ── 学習ループ ─────────────────────────────────────────────────
    accum_steps = args.gradient_accumulation_steps
    clip_grad_norm = args.clip_grad_norm

    has_validation = valid_loader is not None and args.valid_every > 0
    best_val_checkpoints: list[tuple[float, int, Path]] = []

    if scheduler is not None and step == 0:
        scheduler.step()
    optimizer.zero_grad(set_to_none=True)

    accum_micro_steps = 0
    accum_loss = torch.zeros((), device=device, dtype=torch.float32)
    epoch = 0

    print(f"学習開始: max_steps={args.max_steps} save_every={args.save_every}")

    try:
        while step < args.max_steps and not (early_stopper is not None and early_stopper.should_stop):
            epoch += 1
            for batch in loader:
                accum_micro_steps += 1

                text_ids = batch["text_ids"].to(device, non_blocking=True)
                text_mask = batch["text_mask"].to(device, non_blocking=True)
                x0 = batch["latent_patched"].to(device, non_blocking=True)
                x_mask = batch["latent_mask_patched"].to(device, non_blocking=True)
                x_mask_valid = batch["latent_mask_valid_patched"].to(device, non_blocking=True)
                ref_latent = batch["ref_latent_patched"].to(device, non_blocking=True)
                ref_mask = batch["ref_latent_mask_patched"].to(device, non_blocking=True)
                has_speaker = batch["has_speaker"].to(device, non_blocking=True)

                bsz = x0.shape[0]
                if args.timestep_stratified:
                    t = sample_stratified_logit_normal_t(
                        batch_size=bsz, device=device,
                        mean=train_cfg.timestep_logit_mean, std=train_cfg.timestep_logit_std,
                        t_min=train_cfg.timestep_min, t_max=train_cfg.timestep_max,
                    )
                else:
                    t = sample_logit_normal_t(
                        batch_size=bsz, device=device,
                        mean=train_cfg.timestep_logit_mean, std=train_cfg.timestep_logit_std,
                        t_min=train_cfg.timestep_min, t_max=train_cfg.timestep_max,
                    )

                noise = torch.randn_like(x0)
                x_t = rf_interpolate(x0, noise, t)
                v_target = rf_velocity_target(x0, noise)

                text_cond_drop = torch.rand(bsz, device=device) < args.text_condition_dropout
                if text_cond_drop.any():
                    text_mask = text_mask.clone()
                    text_mask[text_cond_drop] = False

                speaker_cond_drop = torch.rand(bsz, device=device) < args.speaker_condition_dropout
                use_speaker = has_speaker & (~speaker_cond_drop)
                ref_mask = ref_mask & use_speaker[:, None]
                ref_latent = ref_latent * use_speaker[:, None, None].to(ref_latent.dtype)

                should_step = (accum_micro_steps % accum_steps) == 0

                with (
                    torch.autocast(device_type="cuda", dtype=torch.bfloat16)
                    if use_bf16 else nullcontext()
                ):
                    v_pred = model(
                        x_t=x_t, t=t,
                        text_input_ids=text_ids, text_mask=text_mask,
                        ref_latent=ref_latent, ref_mask=ref_mask,
                        latent_mask=x_mask,
                    )

                v_pred = v_pred.float()
                loss = echo_style_masked_mse(
                    v_pred, v_target.float(),
                    loss_mask=x_mask, valid_mask=x_mask_valid,
                )
                (loss / float(accum_steps)).backward()
                accum_loss += loss.detach()

                if not should_step:
                    continue

                step_loss = (accum_loss / float(accum_steps)).item()
                accum_loss.zero_()

                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    clip_grad_norm if clip_grad_norm > 0 else float("inf"),
                )
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                if scheduler is not None:
                    scheduler.step()
                if ema_model is not None:
                    ema_model.update(model)
                step += 1

                if step == 1 or step % args.log_every == 0:
                    lr_val = current_lr(optimizer)
                    print(f"step={step} loss={step_loss:.6f} lr={lr_val:.3e}")
                    if wandb_run is not None:
                        wandb_run.log({"train/loss": step_loss, "train/lr": lr_val}, step=step)

                if step % args.save_every == 0:
                    print(f"チェックポイント保存中 (step={step})...")
                    save_lora_checkpoint(
                        output_dir=output_dir, model=model,
                        optimizer=optimizer, scheduler=scheduler, step=step,
                        base_model_path=args.base_model,
                        base_model_cfg=base_model_cfg_dict,
                        lora_cfg_dict=lora_cfg_dict,
                        train_cfg=train_cfg, ema_model=ema_model,
                        save_full=args.save_full,
                    )
                    print(f"チェックポイント保存完了: step={step}")

                if has_validation and step % args.valid_every == 0:
                    valid_metrics = run_validation(
                        model=model, loader=valid_loader, train_cfg=train_cfg,
                        device=device, use_bf16=use_bf16, distributed=False,
                    )
                    v_loss = valid_metrics["loss"]
                    print(f"valid step={step} loss={v_loss:.6f}")
                    if wandb_run is not None:
                        wandb_run.log({"valid/loss": v_loss}, step=step)

                    # ベストval保存
                    best_dir = save_lora_best_val(output_dir, model, ema_model, step, v_loss)
                    print(f"best val checkpoint: {best_dir.name} (loss={v_loss:.6f})")

                    if early_stopper is not None:
                        if early_stopper.step(score=v_loss, current_step=step):
                            print(f"Early Stopping: best step={early_stopper.best_step} loss={early_stopper.best_score:.6f}")
                            break

                if step >= args.max_steps:
                    break
                if early_stopper is not None and early_stopper.should_stop:
                    break

        # 最終チェックポイント保存
        print("最終チェックポイントを保存中...")
        save_lora_final(output_dir, model, ema_model)
        print(f"学習完了 (step={step}): {output_dir / 'lora_checkpoint_final_ema'}")

        if wandb_run is not None:
            wandb_run.summary["train/final_step"] = step

    finally:
        if wandb_run is not None:
            wandb_run.finish()


if __name__ == "__main__":
    main()
