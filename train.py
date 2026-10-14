#!/usr/bin/env python3
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
import time as _time
from pathlib import Path

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler

from irodori_tts.config import (
    ModelConfig,
    TrainConfig,
    dump_configs,
    load_experiment_yaml,
    merge_dataclass_overrides,
)
from irodori_tts.dataset import LatentTextDataset, TTSCollator
from irodori_tts.duration import set_duration_has_speaker_feature
from irodori_tts.meanflow import (
    EncodedConditions,
    adaptive_meanflow_loss,
    apply_condition_profile,
    compute_teacher_meanflow_target,
    conditions_to_device,
    index_conditions,
    sample_meanflow_interval,
)
from irodori_tts.model import TextToLatentRFDiT
from irodori_tts.optim import build_optimizer, build_optimizer_extended, build_scheduler, current_lr
from irodori_tts.progress import TrainProgress
from irodori_tts.rf import (
    rf_interpolate,
    rf_velocity_target,
    sample_logit_normal_t,
    sample_stratified_logit_normal_t,
)
from irodori_tts.tokenizer import PretrainedTextTokenizer

WANDB_MODES = {"online", "offline", "disabled"}
TRAIN_MODES = {"rf", "duration_only", "meanflow_distill"}
MEANFLOW_RESUME_OBJECTIVE_FIELDS = (
    "teacher_checkpoint",
    "teacher_steps",
    "meanflow_anchor_prob",
    "meanflow_time_logit_mean",
    "meanflow_time_logit_std",
    "meanflow_adaptive_weight_power",
    "meanflow_adaptive_weight_eps",
    "meanflow_cfg_text_scale",
    "meanflow_cfg_speaker_scale",
    "meanflow_cfg_caption_scale",
    "meanflow_cfg_min_t",
    "meanflow_cfg_max_t",
    "text_condition_dropout",
    "speaker_condition_dropout",
    "caption_condition_dropout",
)
CHECKPOINT_STEP_RE = re.compile(r"^checkpoint_(\d+)(?:\.pt)?$")
CHECKPOINT_BEST_VAL_LOSS_RE = re.compile(
    r"^checkpoint_best_val_loss_(\d+)_(-?\d+(?:\.\d+)?)(?:\.pt)?$"
)
SAFETENSORS_CONFIG_META_KEY = "config_json"
SAFETENSORS_TEXT_ENCODER_CONFIG_META_KEY = "text_encoder_config_json"
SAFETENSORS_INFERENCE_CONFIG_KEYS = {"max_text_len", "max_caption_len", "fixed_target_latent_steps", "ref_max_seconds"}

# pyファイル基準のcheckpointsフォルダ（トークナイザー等のHFキャッシュ先）
_PROJECT_CHECKPOINTS_DIR = Path(__file__).resolve().parent / "checkpoints"
_HF_TOKENIZER_CACHE_DIR = _PROJECT_CHECKPOINTS_DIR / "tokenizers"


# ---------------------------------------------------------------------------
# EMA (Exponential Moving Average)
# ---------------------------------------------------------------------------
class EMAModel:
    """モデルパラメータの指数移動平均を管理するクラス。
    推論品質向上のため、学習中の重みの平滑化コピーを保持する。"""

    def __init__(self, model: torch.nn.Module, decay: float = 0.9999):
        self.decay = decay
        self.step_count: int = 0  # bias correction 用ステップカウンタ
        self.shadow: dict[str, torch.Tensor] = {}
        self.backup: dict[str, torch.Tensor] = {}
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone().float()

    @torch.no_grad()
    def update(self, model: torch.nn.Module) -> None:
        self.step_count += 1
        for name, param in model.named_parameters():
            if param.requires_grad and name in self.shadow:
                self.shadow[name].mul_(self.decay).add_(
                    param.data.float(), alpha=1.0 - self.decay
                )

    def apply_shadow(self, model: torch.nn.Module) -> None:
        """EMAの重みをモデルに適用（チェックポイント保存用）。

        step_count > 0 の場合は bias correction を適用し、
        初期値への引きずりを除去した正確な EMA 値を書き込む。
        bias correction: corrected = shadow / (1 - decay^step_count)
        """
        if self.step_count > 0:
            correction = 1.0 - self.decay ** self.step_count
        else:
            correction = None  # update 未実施: shadowは初期値そのもの
        for name, param in model.named_parameters():
            if param.requires_grad and name in self.shadow:
                self.backup[name] = param.data.clone()
                if correction is not None:
                    corrected = self.shadow[name] / correction
                else:
                    corrected = self.shadow[name]
                param.data.copy_(corrected.to(param.dtype))

    def restore(self, model: torch.nn.Module) -> None:
        """学習用の元の重みに戻す"""
        for name, param in model.named_parameters():
            if param.requires_grad and name in self.backup:
                param.data.copy_(self.backup[name])
        self.backup.clear()

    def state_dict(self) -> dict:
        return {"decay": self.decay, "shadow": self.shadow, "step_count": self.step_count}

    def load_state_dict(self, state: dict) -> None:
        self.decay = state["decay"]
        self.shadow = state["shadow"]
        # step_count は旧チェックポイントに存在しない場合がある（後方互換）
        self.step_count = int(state.get("step_count", 0))


# ---------------------------------------------------------------------------
# Attention backend
# ---------------------------------------------------------------------------
def apply_attention_backend(model: torch.nn.Module, backend: str) -> None:
    """モデル内のattentionバックエンドを設定する。
    backend: 'sdpa'(デフォルト), 'flash2'(FlashAttention2), 'eager'
    """
    if backend == "eager":
        # sdpaを無効化してeagerモードを強制
        torch.backends.cuda.enable_flash_sdp(False)
        torch.backends.cuda.enable_mem_efficient_sdp(False)
        torch.backends.cuda.enable_math_sdp(True)
        print("Attention backend: eager (math)")
    elif backend == "flash2":
        try:
            import flash_attn  # noqa: F401
            torch.backends.cuda.enable_flash_sdp(True)
            torch.backends.cuda.enable_mem_efficient_sdp(False)
            print("Attention backend: FlashAttention2")
        except ImportError:
            print("warning: flash_attn not installed, falling back to sdpa.")
            backend = "sdpa"
    if backend == "sdpa":
        torch.backends.cuda.enable_flash_sdp(True)
        torch.backends.cuda.enable_mem_efficient_sdp(True)
        print("Attention backend: sdpa (PyTorch scaled_dot_product_attention)")


# ---------------------------------------------------------------------------
# Gradient checkpointing
# ---------------------------------------------------------------------------
def apply_gradient_checkpointing(model: torch.nn.Module) -> None:
    """TextBlock（テキスト/キャプションエンコーダ）に勾配チェックポイントを適用する。

    DiffusionBlock と事前学習テキストエンコーダは model.py 内蔵の
    set_gradient_checkpointing() で処理するため、ここでは TextBlock のみ扱う。
    VRAMを最大40%削減できるが、再計算のため学習速度が約20%低下する。

      - TextBlock : forward(x, mask, freqs_cis)
    """
    from torch.utils.checkpoint import checkpoint as ckpt_fn

    TARGET_CLASSES = ("TextBlock",)
    count = 0

    for module in model.modules():
        cls_name = type(module).__name__
        if cls_name not in TARGET_CLASSES:
            continue

        # クロージャでmoduleをキャプチャし元のforwardを保持
        orig_forward = module.forward

        if cls_name == "TextBlock":
            def make_text_ckpt(orig):
                def checkpointed(x, mask, freqs_cis):
                    def _fn(x_, fc_):
                        return orig(x_, mask, fc_)
                    return ckpt_fn(_fn, x, freqs_cis, use_reentrant=False)
                return checkpointed
            module.forward = make_text_ckpt(orig_forward)

        count += 1

    if count > 0:
        print(f"Gradient checkpointing (TextBlock): enabled ({count}ブロックに適用, VRAM削減, 速度約20%低下)")


# ---------------------------------------------------------------------------
# Early Stopping
# ---------------------------------------------------------------------------
class EarlyStopping:
    """valid lossを監視し、改善が見られない場合に学習を停止するクラス。

    Parameters
    ----------
    patience  : 悪化を何回連続で許容するか（1回 = valid_every step分の猶予）
    min_delta : この値以上悪化した場合のみ「悪化」とカウントする（微小な揺れを無視）
    mode      : 'min'(lossが小さいほど良い) / 'max'(accuracyなど大きいほど良い場合)
    """

    def __init__(self, patience: int = 3, min_delta: float = 0.0, mode: str = "min"):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score: float | None = None
        self.best_step: int = 0
        self.should_stop = False

    def _is_improvement(self, score: float) -> bool:
        if self.best_score is None:
            return True
        if self.mode == "min":
            return score < self.best_score - self.min_delta
        else:
            return score > self.best_score + self.min_delta

    def step(self, score: float, current_step: int) -> bool:
        """
        スコアを評価して早期停止すべきか判定する。
        Returns True if training should stop.
        """
        if self._is_improvement(score):
            self.best_score = score
            self.best_step = current_step
            self.counter = 0
        else:
            self.counter += 1

        if self.counter >= self.patience:
            self.should_stop = True

        return self.should_stop

    def status(self) -> str:
        return (
            f"EarlyStopping: best={self.best_score:.6f} at step={self.best_step} "
            f"patience={self.counter}/{self.patience}"
        )


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def echo_style_masked_mse(
    pred: torch.Tensor,
    target: torch.Tensor,
    loss_mask: torch.Tensor,
    valid_mask: torch.Tensor,
) -> torch.Tensor:
    """
    Echo/JAX-style diffusion loss:
    - take mean squared error over loss_masked tokens
    - divide by mean valid-token ratio (short samples get up-weighted)

    If loss_mask == valid_mask, this reduces to standard masked MSE.
    """
    diff = (pred - target) ** 2
    diff = diff.mean(dim=-1)  # (B, S)
    loss_weight = loss_mask.float()
    valid_weight = valid_mask.float()

    # Keep normalization stable for degenerate samples with no valid target tokens.
    has_valid = (valid_weight.sum(dim=-1) > 0).float()[:, None]
    denom = (loss_weight * valid_weight * has_valid).mean().clamp_min(1e-6)
    return (diff * loss_weight).mean() / denom


def utterance_mean_masked_mse(
    pred: torch.Tensor,
    target: torch.Tensor,
    valid_mask: torch.Tensor,
) -> torch.Tensor:
    diff = (pred - target) ** 2
    diff = diff.mean(dim=-1)
    weight = valid_mask.float()
    per_sample = (diff * weight).sum(dim=-1) / weight.sum(dim=-1).clamp_min(1.0)
    return per_sample.mean()


def compute_rf_loss(
    *,
    pred: torch.Tensor,
    target: torch.Tensor,
    loss_mask: torch.Tensor,
    valid_mask: torch.Tensor,
    mode: str,
) -> torch.Tensor:
    mode = str(mode).strip().lower()
    if mode == "echo":
        return echo_style_masked_mse(pred, target, loss_mask=loss_mask, valid_mask=valid_mask)
    if mode == "utterance_mean":
        return utterance_mean_masked_mse(pred, target, valid_mask=valid_mask)
    raise ValueError(f"Unsupported rf_loss_mode={mode!r}. Expected 'echo' or 'utterance_mean'.")


def freeze_for_meanflow(model: TextToLatentRFDiT) -> tuple[int, int]:
    """Freeze non-DiT modules; MeanFlow updates the complete student DiT only."""
    dit_prefixes = (
        "cond_module.",
        "delta_cond_module.",
        "in_proj.",
        "blocks.",
        "out_norm.",
        "out_proj.",
    )
    trainable = 0
    frozen = 0
    for name, parameter in model.named_parameters():
        should_train = name.startswith(dit_prefixes)
        parameter.requires_grad_(should_train)
        if should_train:
            trainable += parameter.numel()
        else:
            frozen += parameter.numel()
    return trainable, frozen


def set_meanflow_frozen_modules_eval(model: TextToLatentRFDiT) -> None:
    for name in (
        "pretrained_text_backbone",
        "text_encoder",
        "text_norm",
        "speaker_encoder",
        "speaker_norm",
        "caption_encoder",
        "caption_norm",
        "duration_predictor",
    ):
        module = getattr(model, name, None)
        if module is not None:
            module.eval()


def resolve_meanflow_teacher_device(
    *,
    student_device: torch.device,
    local_rank: int,
    local_world_size: int,
    device_offset: int,
    visible_cuda_devices: int | None = None,
) -> torch.device:
    if device_offset < 0:
        raise ValueError("MeanFlow teacher device offset must be non-negative.")
    if device_offset == 0:
        return student_device
    if student_device.type != "cuda":
        raise ValueError("A separate MeanFlow teacher device requires CUDA.")
    if local_world_size <= 0:
        raise ValueError("MeanFlow local world size must be positive.")
    if not 0 <= local_rank < local_world_size:
        raise ValueError(
            f"MeanFlow local_rank={local_rank} is outside local_world_size={local_world_size}."
        )
    if device_offset < local_world_size:
        raise ValueError(
            "Paired MeanFlow teacher GPUs must not overlap student GPUs: "
            f"offset={device_offset} local_world_size={local_world_size}."
        )
    device_count = (
        torch.cuda.device_count() if visible_cuda_devices is None else visible_cuda_devices
    )
    teacher_index = local_rank + device_offset
    if device_offset + local_world_size > device_count or teacher_index >= device_count:
        raise ValueError(
            "Paired MeanFlow teacher GPU is unavailable: "
            f"local_rank={local_rank} offset={device_offset} "
            f"local_world_size={local_world_size} visible_cuda_devices={device_count}."
        )
    return torch.device(f"cuda:{teacher_index}")


def compute_meanflow_distillation_loss(
    *,
    model,
    teacher: TextToLatentRFDiT,
    train_cfg: TrainConfig,
    x0: torch.Tensor,
    latent_mask: torch.Tensor,
    valid_mask: torch.Tensor,
    text_ids: torch.Tensor,
    text_mask: torch.Tensor,
    ref_latent: torch.Tensor | None,
    ref_mask: torch.Tensor | None,
    caption_ids: torch.Tensor | None,
    caption_mask: torch.Tensor | None,
    text_drop: torch.Tensor,
    speaker_drop: torch.Tensor | None,
    caption_drop: torch.Tensor | None,
    use_bf16: bool,
) -> torch.Tensor:
    batch_size = x0.shape[0]
    r, t, delta_t, anchor_mask = sample_meanflow_interval(
        batch_size=batch_size,
        device=x0.device,
        logit_mean=train_cfg.meanflow_time_logit_mean,
        logit_std=train_cfg.meanflow_time_logit_std,
        anchor_prob=train_cfg.meanflow_anchor_prob,
    )
    noise = torch.randn_like(x0)
    x_t = rf_interpolate(x0, noise, t)
    teacher_device = next(teacher.parameters()).device
    cross_device_teacher = teacher_device != x0.device

    def to_teacher(value: torch.Tensor | None) -> torch.Tensor | None:
        if value is None:
            return None
        return value.to(device=teacher_device, non_blocking=not cross_device_teacher)

    with torch.no_grad(), torch.autocast(device_type=teacher_device.type, enabled=False):
        encoded = EncodedConditions(
            *teacher.encode_conditions(
                text_input_ids=to_teacher(text_ids),
                text_mask=to_teacher(text_mask),
                ref_latent=to_teacher(ref_latent),
                ref_mask=to_teacher(ref_mask),
                caption_input_ids=to_teacher(caption_ids),
                caption_mask=to_teacher(caption_mask),
            )
        )
        base_conditions = apply_condition_profile(
            encoded,
            text_drop=to_teacher(text_drop),
            speaker_drop=to_teacher(speaker_drop),
            caption_drop=to_teacher(caption_drop),
        )
        formula = rf_velocity_target(x0, noise).float()
        target = formula.clone()
        non_anchor_index = torch.nonzero(~anchor_mask, as_tuple=False).squeeze(-1)
        if non_anchor_index.numel() > 0:
            teacher_index = non_anchor_index.to(device=teacher_device)
            rollout_target = compute_teacher_meanflow_target(
                teacher=teacher,
                x_t=to_teacher(x_t.index_select(0, non_anchor_index)),
                t=to_teacher(t.index_select(0, non_anchor_index)),
                r=to_teacher(r.index_select(0, non_anchor_index)),
                conditions=index_conditions(base_conditions, teacher_index),
                latent_mask=to_teacher(latent_mask.index_select(0, non_anchor_index)),
                valid_mask=to_teacher(valid_mask.index_select(0, non_anchor_index)),
                teacher_steps=train_cfg.teacher_steps,
                cfg_text_scale=train_cfg.meanflow_cfg_text_scale,
                cfg_speaker_scale=train_cfg.meanflow_cfg_speaker_scale,
                cfg_caption_scale=train_cfg.meanflow_cfg_caption_scale,
                cfg_min_t=train_cfg.meanflow_cfg_min_t,
                cfg_max_t=train_cfg.meanflow_cfg_max_t,
                sample_chunk_size=train_cfg.meanflow_teacher_chunk_size,
                branch_batch_size=train_cfg.meanflow_teacher_branch_batch_size,
                fuse_base_branch=train_cfg.meanflow_teacher_fuse_base_branch,
            )
            target.index_copy_(
                0,
                non_anchor_index,
                rollout_target.to(
                    device=x0.device,
                    non_blocking=not cross_device_teacher,
                ),
            )
        target = target * valid_mask[:, :, None].float()
        student_conditions = conditions_to_device(
            base_conditions,
            x0.device,
            non_blocking=not cross_device_teacher,
        )

    with torch.autocast(device_type="cuda", dtype=torch.bfloat16) if use_bf16 else nullcontext():
        pred = model(
            x_t=x_t,
            t=t,
            delta_t=delta_t,
            text_input_ids=text_ids,
            text_mask=text_mask,
            ref_latent=ref_latent,
            ref_mask=ref_mask,
            caption_input_ids=caption_ids,
            caption_mask=caption_mask,
            latent_mask=latent_mask,
            encoded_conditions=(
                student_conditions.text_state,
                student_conditions.text_mask,
                student_conditions.speaker_state,
                student_conditions.speaker_mask,
                student_conditions.caption_state,
                student_conditions.caption_mask,
            ),
        )
    if not isinstance(pred, torch.Tensor):
        raise RuntimeError("MeanFlow student forward must return one velocity tensor.")
    return adaptive_meanflow_loss(
        pred,
        target,
        valid_mask,
        power=train_cfg.meanflow_adaptive_weight_power,
        eps=train_cfg.meanflow_adaptive_weight_eps,
    )


def freeze_for_speaker_inversion(model: torch.nn.Module) -> tuple[int, int]:
    """Speaker Inversion embedding 以外を凍結する（上流 train.py 相当）。"""
    trainable_params = 0
    frozen_params = 0
    for key, param in model.named_parameters():
        if str(key).startswith("speaker_inversion."):
            param.requires_grad_(True)
            trainable_params += param.numel()
        else:
            param.requires_grad_(False)
            frozen_params += param.numel()
    return trainable_params, frozen_params


PRETRAINED_PROJECTOR_PARAM_PREFIXES = ("text_encoder.", "caption_encoder.")


def _clear_non_pretrained_projector_grads(model: torch.nn.Module) -> None:
    """pretrained projector（text_encoder/caption_encoder）の勾配だけを残す（上流相当）。"""
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if name.startswith(PRETRAINED_PROJECTOR_PARAM_PREFIXES):
            continue
        if param.grad is not None:
            param.grad = None


def save_checkpoint(
    path: str | Path,
    model: TextToLatentRFDiT,
    optimizer: torch.optim.Optimizer,
    scheduler,
    step: int,
    model_cfg: ModelConfig,
    train_cfg: TrainConfig,
    ema_model: "EMAModel | None" = None,
    save_full: bool = False,
    text_encoder_config: dict | None = None,
) -> None:
    """チェックポイントを保存する。

    保存方針:
    - EMAが有効な場合:
        checkpoint_XXXXXXX_ema.pt  ... EMA版（推論用・軽量）常に保存
        checkpoint_XXXXXXX_full.pt ... フルサイズ版（追加学習用）--save-full指定時のみ
    - EMAが無効な場合:
        checkpoint_XXXXXXX.pt      ... 従来通りフルサイズのみ
    - Speaker Inversion 学習時は 追加で <path>.speaker.safetensors を保存する。
    - v4系では事前学習テキストエンコーダ設定(text_encoder_config)を .pt に埋め込む。
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    if train_cfg.speaker_inversion_enabled:
        from irodori_tts.speaker_inversion import save_speaker_inversion_checkpoint
        if ema_model is not None:
            ema_model.apply_shadow(model)
        save_speaker_inversion_checkpoint(
            path.with_name(path.stem + ".speaker.safetensors"), model=model
        )
        if ema_model is not None:
            ema_model.restore(model)

    if ema_model is not None:
        # EMA版を _ema.pt で保存（推論用・軽量）
        ema_path = path.with_name(path.stem + "_ema" + path.suffix)
        ema_model.apply_shadow(model)
        ema_payload = {
            "step": step,
            "model": model.state_dict(),
            "model_config": asdict(model_cfg),
        }
        if text_encoder_config is not None:
            ema_payload["text_encoder_config"] = text_encoder_config
        torch.save(ema_payload, ema_path)
        ema_model.restore(model)

        # フルサイズ版を _full.pt でオプション保存（追加学習用）
        if save_full:
            full_path = path.with_name(path.stem + "_full" + path.suffix)
            full_payload = {
                "step": step,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": None if scheduler is None else scheduler.state_dict(),
                "model_config": asdict(model_cfg),
                "train_config": asdict(train_cfg),
                "ema": ema_model.state_dict(),
            }
            if text_encoder_config is not None:
                full_payload["text_encoder_config"] = text_encoder_config
            torch.save(full_payload, full_path)
    else:
        # EMA無効時は従来通りフルサイズのみ保存
        payload = {
            "step": step,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": None if scheduler is None else scheduler.state_dict(),
            "model_config": asdict(model_cfg),
            "train_config": asdict(train_cfg),
        }
        if text_encoder_config is not None:
            payload["text_encoder_config"] = text_encoder_config
        torch.save(payload, path)


def _safe_unlink(path: Path) -> None:
    try:
        if path.is_dir():
            import shutil
            shutil.rmtree(path)
        else:
            path.unlink()
    except FileNotFoundError:
        return


def list_periodic_checkpoints(output_dir: Path) -> list[tuple[int, Path]]:
    checkpoints: list[tuple[int, Path]] = []
    for path in output_dir.glob("checkpoint_*"):
        match = CHECKPOINT_STEP_RE.match(path.name)
        if match is None:
            continue
        checkpoints.append((int(match.group(1)), path))
    checkpoints.sort(key=lambda item: item[0], reverse=True)
    return checkpoints


def enforce_periodic_checkpoint_limit(output_dir: Path, keep_count: int) -> None:
    if keep_count <= 0:
        return
    checkpoints = list_periodic_checkpoints(output_dir)
    for _, stale_path in checkpoints[keep_count:]:
        _safe_unlink(stale_path)


def list_best_val_loss_checkpoints(output_dir: Path) -> list[tuple[float, int, Path]]:
    checkpoints: list[tuple[float, int, Path]] = []
    for path in output_dir.glob("checkpoint_best_val_loss_*"):
        match = CHECKPOINT_BEST_VAL_LOSS_RE.match(path.name)
        if match is None:
            continue
        step = int(match.group(1))
        score = float(match.group(2))
        checkpoints.append((score, step, path))
    checkpoints.sort(key=lambda item: (item[0], item[1]))
    return checkpoints


def prune_best_val_loss_checkpoints(
    checkpoints: list[tuple[float, int, Path]],
    keep_best_n: int,
) -> list[tuple[float, int, Path]]:
    if keep_best_n <= 0:
        return checkpoints
    checkpoints = sorted(checkpoints, key=lambda item: (item[0], item[1]))
    while len(checkpoints) > keep_best_n:
        _, _, stale_path = checkpoints.pop()
        _safe_unlink(stale_path)
    return checkpoints


def maybe_save_best_val_loss_checkpoint(
    *,
    output_dir: Path,
    checkpoints: list[tuple[float, int, Path]],
    keep_best_n: int,
    val_loss: float,
    step: int,
    model: TextToLatentRFDiT,
    optimizer: torch.optim.Optimizer,
    scheduler,
    model_cfg: ModelConfig,
    train_cfg: TrainConfig,
    ema_model: "EMAModel | None" = None,
    save_full: bool = False,
    text_encoder_config: dict | None = None,
) -> tuple[list[tuple[float, int, Path]], Path | None]:
    if keep_best_n <= 0:
        return checkpoints, None

    checkpoints = sorted(checkpoints, key=lambda item: (item[0], item[1]))
    if len(checkpoints) >= keep_best_n:
        worst_score = checkpoints[-1][0]
        if val_loss >= worst_score:
            return checkpoints, None

    kept: list[tuple[float, int, Path]] = []
    for score, saved_step, path in checkpoints:
        if saved_step == step:
            _safe_unlink(path)
            continue
        kept.append((score, saved_step, path))
    checkpoints = kept

    path = output_dir / f"checkpoint_best_val_loss_{step:07d}_{val_loss:.6f}.pt"
    save_checkpoint(
        path=path,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        step=step,
        model_cfg=model_cfg,
        train_cfg=train_cfg,
        ema_model=ema_model,
        save_full=save_full,
        text_encoder_config=text_encoder_config,
    )
    checkpoints.append((float(val_loss), int(step), path))
    checkpoints = prune_best_val_loss_checkpoints(checkpoints, keep_best_n)
    return checkpoints, path


def cli_provided(argv: list[str], flag: str) -> bool:
    return any(x == flag or x.startswith(flag + "=") for x in argv)


def build_text_tokenizer(
    model_cfg: ModelConfig,
    *,
    local_files_only: bool = False,
) -> PretrainedTextTokenizer:
    _HF_TOKENIZER_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    tokenizer = PretrainedTextTokenizer.from_pretrained(
        repo_id=model_cfg.text_tokenizer_repo,
        add_bos=bool(model_cfg.text_add_bos),
        local_files_only=local_files_only,
        cache_dir=str(_HF_TOKENIZER_CACHE_DIR),
    )
    if tokenizer.vocab_size != model_cfg.text_vocab_size:
        raise ValueError(
            f"text_vocab_size mismatch: model text_vocab_size={model_cfg.text_vocab_size} but tokenizer "
            f"({model_cfg.text_tokenizer_repo}) vocab_size={tokenizer.vocab_size}."
        )
    return tokenizer


def validate_text_backbone_dim(
    model_cfg: ModelConfig,
    *,
    local_files_only: bool = False,
) -> int:
    try:
        from transformers import AutoConfig
    except ImportError as exc:
        raise RuntimeError(
            "transformers is required for pretrained text embedding initialization. "
            "Install with `pip install transformers sentencepiece`."
        ) from exc

    _HF_TOKENIZER_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    text_cfg = AutoConfig.from_pretrained(
        model_cfg.text_tokenizer_repo,
        trust_remote_code=False,
        local_files_only=local_files_only,
        cache_dir=str(_HF_TOKENIZER_CACHE_DIR),
    )
    hidden_size = getattr(text_cfg, "hidden_size", None)
    if hidden_size is None:
        raise ValueError(
            f"Could not read hidden_size from pretrained config: {model_cfg.text_tokenizer_repo}"
        )
    hidden_size = int(hidden_size)
    # use_pretrained_text_encoder の場合は PretrainedConditionProjector が
    # backbone hidden_size -> text_dim の次元変換を担うため、一致は要求しない。
    if not model_cfg.use_pretrained_text_encoder and hidden_size != model_cfg.text_dim:
        raise ValueError(
            f"text_dim mismatch: model text_dim={model_cfg.text_dim} but pretrained hidden_size={hidden_size} "
            f"for repo {model_cfg.text_tokenizer_repo}."
        )
    return hidden_size


def initialize_text_embedding_from_pretrained(
    model: TextToLatentRFDiT,
    model_cfg: ModelConfig,
    *,
    local_files_only: bool = False,
) -> None:
    try:
        from transformers import AutoModel
    except ImportError as exc:
        raise RuntimeError(
            "transformers is required for pretrained text embedding initialization. "
            "Install with `pip install transformers sentencepiece`."
        ) from exc

    _HF_TOKENIZER_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    text_backbone = AutoModel.from_pretrained(
        model_cfg.text_tokenizer_repo,
        trust_remote_code=False,
        dtype=torch.float32,
        low_cpu_mem_usage=True,
        local_files_only=local_files_only,
        cache_dir=str(_HF_TOKENIZER_CACHE_DIR),
    )
    pretrained_embedding = text_backbone.get_input_embeddings()
    if pretrained_embedding is None:
        raise ValueError(
            f"Pretrained model has no input embeddings: {model_cfg.text_tokenizer_repo}"
        )
    src_weight = pretrained_embedding.weight.detach().to(device="cpu", dtype=torch.float32)
    tgt_weight = model.text_encoder.text_embedding.weight
    src_vocab, src_dim = tuple(src_weight.shape)
    tgt_vocab, tgt_dim = tuple(tgt_weight.shape)
    if src_dim != tgt_dim:
        raise ValueError(
            f"Embedding hidden size mismatch: pretrained={src_dim} model={tgt_dim}. Check text_dim."
        )

    copy_rows = min(src_vocab, tgt_vocab)
    with torch.no_grad():
        tgt_weight[:copy_rows].copy_(
            src_weight[:copy_rows].to(device=tgt_weight.device, dtype=tgt_weight.dtype)
        )

    del text_backbone


def _normalize_checkpoint_path(path: str | Path) -> Path:
    return Path(os.path.abspath(str(Path(path).expanduser())))


def _validate_meanflow_resume_objective(
    train_cfg: TrainConfig,
    *,
    resume_train_cfg: dict | None,
    resume_model_cfg: dict | None,
) -> None:
    """Prevent a MeanFlow checkpoint and Muon state resuming onto a new target."""
    if not isinstance(resume_train_cfg, dict):
        raise ValueError("MeanFlow resume checkpoint is missing train_config provenance.")
    if str(resume_train_cfg.get("train_mode", "")).strip().lower() != "meanflow_distill":
        raise ValueError("MeanFlow training can only resume a meanflow_distill checkpoint.")
    if not isinstance(resume_model_cfg, dict):
        raise ValueError("MeanFlow resume checkpoint is missing model_config provenance.")
    if str(resume_model_cfg.get("flow_parameterization", "")).strip().lower() != "meanflow":
        raise ValueError("MeanFlow resume checkpoint must contain a MeanFlow student.")

    for field in MEANFLOW_RESUME_OBJECTIVE_FIELDS:
        if field not in resume_train_cfg:
            raise ValueError(
                f"MeanFlow resume checkpoint is missing objective field train.{field}."
            )
        current_value = getattr(train_cfg, field)
        resume_value = resume_train_cfg[field]
        if field == "teacher_checkpoint":
            current_value = str(_normalize_checkpoint_path(current_value))
            resume_value = str(_normalize_checkpoint_path(resume_value))
        if current_value != resume_value:
            raise ValueError(
                f"MeanFlow resume checkpoint expects train.{field}={resume_value!r}, "
                f"but current config requests {current_value!r}."
            )


def _load_model_state_from_checkpoint(
    path: Path,
) -> tuple[dict[str, torch.Tensor], dict | None, dict | None, dict | None]:
    """チェックポイントから (state, model_config, train_config, text_encoder_config) を読み出す。

    text_encoder_config は v4 系 (text_encoder_type='pretrained') の事前学習
    バックボーン設定で、safetensors では 'text_encoder_config_json' メタデータ、
    .pt では 'text_encoder_config' キーに保存される。無い場合は None。
    """
    if path.suffix.lower() == ".safetensors":
        from safetensors import safe_open
        from safetensors.torch import load_file as load_safetensors_file

        checkpoint_model_cfg = None
        text_encoder_config = None
        with safe_open(str(path), framework="pt", device="cpu") as handle:
            metadata = dict(handle.metadata() or {})
        from irodori_tts.quantization import parse_quantization_metadata
        if parse_quantization_metadata(metadata) is not None:
            raise ValueError(
                "Quantized checkpoints are inference-only and cannot be used with "
                "--init-checkpoint. Train LoRA against the matching full-precision "
                "base model, then merge and quantize it for inference."
            )
        config_json = metadata.get(SAFETENSORS_CONFIG_META_KEY)
        if config_json:
            parsed = json.loads(config_json)
            if isinstance(parsed, dict):
                checkpoint_model_cfg = {
                    key: value
                    for key, value in parsed.items()
                    if key not in SAFETENSORS_INFERENCE_CONFIG_KEYS
                }
        text_encoder_json = metadata.get(SAFETENSORS_TEXT_ENCODER_CONFIG_META_KEY)
        if text_encoder_json:
            parsed_te = json.loads(text_encoder_json)
            if isinstance(parsed_te, dict):
                text_encoder_config = parsed_te
        return load_safetensors_file(str(path), device="cpu"), checkpoint_model_cfg, None, text_encoder_config

    payload = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(payload, dict):
        raise ValueError(f"Checkpoint payload must be a dictionary, got {type(payload)!r}.")

    raw_model = payload.get("model")
    if raw_model is None and all(isinstance(v, torch.Tensor) for v in payload.values()):
        raw_model = payload
    if not isinstance(raw_model, dict):
        raise ValueError(f"Checkpoint does not contain a model state dictionary: {path}")

    checkpoint_model_cfg = payload.get("model_config")
    if checkpoint_model_cfg is not None and not isinstance(checkpoint_model_cfg, dict):
        raise ValueError(f"Checkpoint model_config must be a dictionary when present: {path}")
    checkpoint_train_cfg = payload.get("train_config")
    if checkpoint_train_cfg is not None and not isinstance(checkpoint_train_cfg, dict):
        raise ValueError(f"Checkpoint train_config must be a dictionary when present: {path}")
    text_encoder_config = payload.get("text_encoder_config")
    if text_encoder_config is not None and not isinstance(text_encoder_config, dict):
        raise ValueError(f"Checkpoint text_encoder_config must be a dictionary when present: {path}")
    return raw_model, checkpoint_model_cfg, checkpoint_train_cfg, text_encoder_config


def _check_model_config_compatibility(
    checkpoint_path: Path,
    checkpoint_model_cfg: dict | None,
    current_model_cfg: ModelConfig,
    *,
    require_caption_match: bool = False,
) -> None:
    if checkpoint_model_cfg is None:
        return

    checkpoint_cfg = merge_dataclass_overrides(
        ModelConfig(),
        checkpoint_model_cfg,
        section="checkpoint model_config",
    )

    comparisons: list[tuple[str, object, object]] = [
        ("latent_dim", checkpoint_cfg.latent_dim, current_model_cfg.latent_dim),
        ("latent_patch_size", checkpoint_cfg.latent_patch_size, current_model_cfg.latent_patch_size),
        ("model_dim", checkpoint_cfg.model_dim, current_model_cfg.model_dim),
        ("num_layers", checkpoint_cfg.num_layers, current_model_cfg.num_layers),
        ("num_heads", checkpoint_cfg.num_heads, current_model_cfg.num_heads),
        ("text_vocab_size", checkpoint_cfg.text_vocab_size, current_model_cfg.text_vocab_size),
        ("text_dim", checkpoint_cfg.text_dim, current_model_cfg.text_dim),
        ("adaln_rank", checkpoint_cfg.adaln_rank, current_model_cfg.adaln_rank),
    ]
    if require_caption_match:
        comparisons.extend([
            ("use_caption_condition", checkpoint_cfg.use_caption_condition, current_model_cfg.use_caption_condition),
            ("caption_vocab_size", checkpoint_cfg.caption_vocab_size_resolved, current_model_cfg.caption_vocab_size_resolved),
        ])

    for key, checkpoint_value, current_value in comparisons:
        if checkpoint_value != current_value:
            raise ValueError(
                f"Checkpoint/config mismatch for '{key}': checkpoint={checkpoint_value} "
                f"current={current_value} ({checkpoint_path})"
            )


def checkpoint_uses_caption_condition(
    checkpoint_model_cfg: dict | None,
    state_dict: dict[str, torch.Tensor],
) -> bool:
    """チェックポイントがcaption conditioningを持つか判別する。"""
    if checkpoint_model_cfg is not None:
        checkpoint_cfg = merge_dataclass_overrides(
            ModelConfig(), checkpoint_model_cfg, section="checkpoint model_config"
        )
        if checkpoint_cfg.use_caption_condition:
            return True
    return any(
        key.startswith("caption_encoder.")
        or key.startswith("caption_norm.")
        or ".wk_caption." in key
        or ".wv_caption." in key
        for key in state_dict
    )


def checkpoint_uses_duration_predictor(
    checkpoint_model_cfg: dict | None,
    state_dict: dict[str, torch.Tensor],
) -> bool:
    """チェックポイントがduration predictorを持つか判別する。"""
    if checkpoint_model_cfg is not None:
        checkpoint_cfg = merge_dataclass_overrides(
            ModelConfig(), checkpoint_model_cfg, section="checkpoint model_config"
        )
        if checkpoint_cfg.use_duration_predictor:
            return True
    return any(key.startswith("duration_predictor.") for key in state_dict)


def _summarize_missing_keys(missing, max_examples: int = 10) -> None:
    """欠けているキーをトップレベルのモジュール名ごとに集計して表示する。"""
    from collections import Counter as _Counter
    prefixes = _Counter(str(k).split(".")[0] for k in missing)
    for prefix, count in sorted(prefixes.items(), key=lambda x: (-x[1], x[0])):
        print(f"    - {prefix}: {count}")
    for example_key in list(missing)[:max_examples]:
        print(f"      e.g. {example_key}")


def load_model_state_partially(
    model: TextToLatentRFDiT,
    state_dict: dict[str, torch.Tensor],
) -> tuple[list[str], list[str], list[str]]:
    """shape不一致・余分なキーをスキップして部分ロードする。"""
    model_state = model.state_dict()
    filtered_state: dict[str, torch.Tensor] = {}
    skipped_shape: list[str] = []
    skipped_extra: list[str] = []

    for key, value in state_dict.items():
        target = model_state.get(key)
        if target is None:
            skipped_extra.append(key)
            continue
        if tuple(target.shape) != tuple(value.shape):
            skipped_shape.append(key)
            continue
        filtered_state[key] = value

    missing_keys, unexpected_keys = model.load_state_dict(filtered_state, strict=False)
    if unexpected_keys:
        skipped_extra.extend(unexpected_keys)
    return missing_keys, skipped_shape, skipped_extra



def resolve_dist_env() -> tuple[int, int, int]:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", str(rank)))
    return rank, world_size, local_rank


def setup_distributed(device_arg: str) -> tuple[int, int, int, bool, torch.device]:
    rank, world_size, local_rank = resolve_dist_env()
    distributed = world_size > 1
    if distributed:
        if not str(device_arg).startswith("cuda"):
            raise ValueError(
                f"WORLD_SIZE={world_size} detected, but --device={device_arg!r}. "
                "DDP multi-GPU training requires --device cuda."
            )
        if not torch.cuda.is_available():
            raise RuntimeError("WORLD_SIZE>1 detected, but CUDA is not available.")
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend="nccl")
        device = torch.device(f"cuda:{local_rank}")
    else:
        device = torch.device(device_arg)
    return rank, world_size, local_rank, distributed, device


def reduce_mean(value: torch.Tensor, world_size: int, distributed: bool) -> torch.Tensor:
    reduced = value.detach().clone()
    if not distributed:
        return reduced
    dist.all_reduce(reduced, op=dist.ReduceOp.SUM)
    reduced /= float(world_size)
    return reduced


def split_train_valid_indices(
    *,
    num_samples: int,
    valid_ratio: float,
    seed: int,
) -> tuple[list[int], list[int]]:
    if valid_ratio <= 0.0:
        return list(range(num_samples)), []
    if num_samples < 2:
        raise ValueError(
            f"Validation split requires at least 2 samples in manifest, got {num_samples}."
        )

    valid_count = int(num_samples * valid_ratio)
    valid_count = max(1, valid_count)
    if valid_count >= num_samples:
        valid_count = num_samples - 1

    generator = torch.Generator()
    generator.manual_seed(int(seed))
    perm = torch.randperm(num_samples, generator=generator).tolist()
    valid_indices = sorted(perm[:valid_count])
    train_indices = sorted(perm[valid_count:])
    if not train_indices or not valid_indices:
        raise ValueError(
            "Failed to create non-empty train/valid split. "
            f"num_samples={num_samples} valid_ratio={valid_ratio}"
        )
    return train_indices, valid_indices


def run_meanflow_validation(
    *,
    model,
    teacher: TextToLatentRFDiT,
    loader: DataLoader,
    train_cfg: TrainConfig,
    device: torch.device,
    use_bf16: bool,
    distributed: bool,
) -> dict[str, float]:
    was_training = model.training
    raw_model = model.module if isinstance(model, DDP) else model
    model.eval()
    teacher.eval()
    totals = torch.zeros(3, device=device, dtype=torch.float64)
    with torch.no_grad():
        for batch in loader:
            text_ids = batch["text_ids"].to(device, non_blocking=True)
            text_mask = batch["text_mask"].to(device, non_blocking=True)
            x0 = batch["latent_patched"].to(device, non_blocking=True)
            latent_mask = batch["latent_mask_patched"].to(device, non_blocking=True)
            valid_mask = batch["latent_mask_valid_patched"].to(device, non_blocking=True)
            batch_size = text_ids.shape[0]

            caption_ids = None
            caption_mask = None
            caption_drop = None
            if raw_model.cfg.use_caption_condition:
                caption_ids = batch["caption_ids"].to(device, non_blocking=True)
                caption_mask = batch["caption_mask"].to(device, non_blocking=True)
                has_caption = batch["has_caption"].to(device, non_blocking=True)
                sampled_drop = (
                    torch.rand(batch_size, device=device) < train_cfg.caption_condition_dropout
                )
                caption_drop = ~(has_caption & ~sampled_drop)

            ref_latent = None
            ref_mask = None
            speaker_drop = None
            if raw_model.cfg.use_speaker_condition_resolved:
                ref_latent = batch["ref_latent_patched"].to(device, non_blocking=True)
                ref_mask = batch["ref_latent_mask_patched"].to(device, non_blocking=True)
                has_speaker = batch["has_speaker"].to(device, non_blocking=True)
                sampled_drop = (
                    torch.rand(batch_size, device=device) < train_cfg.speaker_condition_dropout
                )
                speaker_drop = ~(has_speaker & ~sampled_drop)

            text_drop = torch.rand(batch_size, device=device) < train_cfg.text_condition_dropout
            loss = compute_meanflow_distillation_loss(
                model=model,
                teacher=teacher,
                train_cfg=train_cfg,
                x0=x0,
                latent_mask=latent_mask,
                valid_mask=valid_mask,
                text_ids=text_ids,
                text_mask=text_mask,
                ref_latent=ref_latent,
                ref_mask=ref_mask,
                caption_ids=caption_ids,
                caption_mask=caption_mask,
                text_drop=text_drop,
                speaker_drop=speaker_drop,
                caption_drop=caption_drop,
                use_bf16=use_bf16,
            )
            weight = float(batch_size)
            totals[0] += loss.detach().double() * weight
            totals[1] += batch["num_frames"].float().mean().double().to(device) * weight
            totals[2] += weight

    if distributed:
        dist.all_reduce(totals, op=dist.ReduceOp.SUM)
    denom = max(float(totals[2].item()), 1.0)
    if was_training:
        model.train()
        set_meanflow_frozen_modules_eval(raw_model)
    return {
        "loss": float(totals[0].item() / denom),
        "rf_loss": float(totals[0].item() / denom),
        "duration_loss": 0.0,
        "duration_mae_frames": 0.0,
        "target_frames_mean": float(totals[1].item() / denom),
        "num_samples": float(totals[2].item()),
    }


def run_validation(
    *,
    model,
    teacher: TextToLatentRFDiT | None,
    loader: DataLoader,
    train_cfg: TrainConfig,
    device: torch.device,
    use_bf16: bool,
    distributed: bool,
) -> dict[str, float]:
    if train_cfg.train_mode == "meanflow_distill":
        if teacher is None:
            raise RuntimeError("MeanFlow validation requires a teacher model.")
        return run_meanflow_validation(
            model=model,
            teacher=teacher,
            loader=loader,
            train_cfg=train_cfg,
            device=device,
            use_bf16=use_bf16,
            distributed=distributed,
        )
    was_training = model.training
    # PeftModel は .cfg を直接持たないため base_model.model 経由で取得する。
    # DDP > PeftModel > TextToLatentRFDiT の順にアンラップして cfg を取得する。
    def _unwrap_cfg(m) -> "ModelConfig":
        from torch.nn.parallel import DistributedDataParallel as _DDP
        inner = m.module if isinstance(m, _DDP) else m
        # peft.PeftModel / peft.PeftModelForFeatureExtraction など
        try:
            from peft import PeftModel as _PeftModel
            if isinstance(inner, _PeftModel):
                inner = inner.base_model.model
        except ImportError:
            pass
        return inner.cfg

    model_cfg: ModelConfig = _unwrap_cfg(model)
    duration_only = str(train_cfg.train_mode).strip().lower() == "duration_only"
    model.eval()
    totals = torch.zeros(6, device=device, dtype=torch.float64)

    with torch.no_grad():
        for batch in loader:
            text_ids = batch["text_ids"].to(device, non_blocking=True)
            text_mask = batch["text_mask"].to(device, non_blocking=True)
            caption_ids = None
            caption_mask = None
            if model_cfg.use_caption_condition:
                caption_ids = batch["caption_ids"].to(device, non_blocking=True)
                caption_mask = batch["caption_mask"].to(device, non_blocking=True)
                has_caption = batch["has_caption"].to(device, non_blocking=True)
            else:
                has_caption = None
            num_frames = batch["num_frames"].to(device, non_blocking=True)
            duration_features = batch["duration_features"].to(device, non_blocking=True)
            ref_latent = None
            ref_mask = None
            if model_cfg.use_speaker_condition_resolved:
                ref_latent = batch["ref_latent_patched"].to(device, non_blocking=True)
                ref_mask = batch["ref_latent_mask_patched"].to(device, non_blocking=True)
                has_speaker = batch["has_speaker"].to(device, non_blocking=True)
            else:
                has_speaker = None

            bsz = text_ids.shape[0]
            x0 = x_mask = x_mask_valid = x_t = t = v_target = None
            if not duration_only:
                x0 = batch["latent_patched"].to(device, non_blocking=True)
                x_mask = batch["latent_mask_patched"].to(device, non_blocking=True)
                x_mask_valid = batch["latent_mask_valid_patched"].to(device, non_blocking=True)
                if train_cfg.timestep_stratified:
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

            if model_cfg.use_speaker_condition_resolved:
                use_speaker = has_speaker
                duration_has_speaker = use_speaker
                duration_features = set_duration_has_speaker_feature(
                    duration_features, duration_has_speaker
                )
            else:
                duration_has_speaker = None
            if model_cfg.use_caption_condition:
                duration_has_caption = has_caption
            else:
                duration_has_caption = None

            with (
                torch.autocast(device_type="cuda", dtype=torch.bfloat16)
                if use_bf16
                else nullcontext()
            ):
                if duration_only:
                    duration_pred = model(
                        x_t=None, t=None,
                        text_input_ids=text_ids, text_mask=text_mask,
                        ref_latent=ref_latent, ref_mask=ref_mask,
                        caption_input_ids=caption_ids, caption_mask=caption_mask,
                        latent_mask=None,
                        duration_features=duration_features,
                        duration_has_speaker=duration_has_speaker,
                        duration_has_caption=duration_has_caption,
                        duration_only=True,
                    )
                    v_pred = None
                elif model_cfg.use_duration_predictor:
                    v_pred, duration_pred = model(
                        x_t=x_t, t=t,
                        text_input_ids=text_ids, text_mask=text_mask,
                        ref_latent=ref_latent, ref_mask=ref_mask,
                        caption_input_ids=caption_ids, caption_mask=caption_mask,
                        latent_mask=x_mask,
                        duration_features=duration_features,
                        duration_has_speaker=duration_has_speaker,
                        duration_has_caption=duration_has_caption,
                    )
                else:
                    # v2以前: speaker conditioningは手動マスク
                    if model_cfg.use_speaker_condition_resolved:
                        ref_mask = ref_mask & use_speaker[:, None]
                        ref_latent = ref_latent * use_speaker[:, None, None].to(ref_latent.dtype)
                    v_pred = model(
                        x_t=x_t, t=t,
                        text_input_ids=text_ids, text_mask=text_mask,
                        ref_latent=ref_latent, ref_mask=ref_mask,
                        caption_input_ids=caption_ids, caption_mask=caption_mask,
                        latent_mask=x_mask,
                    )
                    duration_pred = None

            rf_loss = torch.zeros((), device=device, dtype=torch.float32)
            if not duration_only:
                v_pred = v_pred.float()
                rf_loss = compute_rf_loss(
                    pred=v_pred, target=v_target.float(),
                    loss_mask=x_mask, valid_mask=x_mask_valid,
                    mode=train_cfg.rf_loss_mode,
                )
            duration_loss = torch.zeros((), device=device, dtype=torch.float32)
            if model_cfg.use_duration_predictor and duration_pred is not None:
                duration_target = torch.log1p(num_frames.float())
                duration_loss = F.huber_loss(
                    duration_pred.float(), duration_target,
                    delta=float(train_cfg.duration_huber_delta), reduction="mean",
                )
            if duration_only:
                loss = duration_loss
            else:
                loss = rf_loss + float(train_cfg.duration_loss_weight) * duration_loss

            weight = float(bsz)
            totals[0] += loss.detach().double() * weight
            totals[1] += rf_loss.detach().double() * weight
            totals[2] += duration_loss.detach().double() * weight
            totals[3] += weight

    if distributed:
        dist.all_reduce(totals, op=dist.ReduceOp.SUM)
    denom = max(float(totals[3].item()), 1.0)
    metrics = {
        "loss": float(totals[0].item() / denom),
        "rf_loss": float(totals[1].item() / denom),
        "duration_loss": float(totals[2].item() / denom),
        "num_samples": float(totals[3].item()),
    }
    if was_training:
        model.train()
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Train Irodori-TTS.")
    parser.add_argument("--config", default=None, help="YAML config path (model/train overrides)")
    parser.add_argument(
        "--manifest",
        required=True,
        help="JSONL manifest with text+latent_path (optional speaker_id for reference sampling).",
    )
    parser.add_argument("--output-dir", default="outputs/irodori_tts")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument(
        "--precision", choices=["fp32", "bf16"], default="bf16",
        help="Compute precision for model forward pass. Model weights and optimizer states remain FP32.",
    )
    parser.add_argument(
        "--tf32", dest="allow_tf32", action=argparse.BooleanOptionalAction, default=None,
        help="Enable TF32 matmul/cuDNN kernels on CUDA for speed.",
    )
    parser.add_argument(
        "--compile-model", dest="compile_model", action=argparse.BooleanOptionalAction, default=None,
        help="Enable torch.compile for the training model.",
    )
    parser.add_argument(
        "--train-mode", choices=sorted(TRAIN_MODES), default=None,
        help="Training objective: rf runs DiT/RF training; duration_only trains only the duration predictor.",
    )
    parser.add_argument(
        "--resume", default=None,
        help="Resume full training state from a training checkpoint (.pt). .safetensors は --init-checkpoint を使用。",
    )
    parser.add_argument(
        "--init-checkpoint", default=None,
        help="Initialize model weights from a checkpoint (.pt or .safetensors) and start a new run with fresh optimizer / scheduler state.",
    )
    parser.add_argument("--max-steps", type=int, default=200000)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument(
        "--gradient-accumulation-steps", type=int, default=1,
        help="Number of micro-batches to accumulate before optimizer.step(). 1 disables accumulation.",
    )
    parser.add_argument(
        "--max-text-len", type=int, default=256,
        help="Maximum token length for text conditioning (right-truncated).",
    )
    parser.add_argument(
        "--max-caption-len", type=int, default=None,
        help="Maximum token length for caption conditioning (defaults to max_text_len).",
    )
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--optimizer", choices=["adamw", "muon"], default="muon")
    parser.add_argument("--adam-beta1", type=float, default=0.9)
    parser.add_argument("--adam-beta2", type=float, default=0.999)
    parser.add_argument("--adam-eps", type=float, default=1e-8)
    parser.add_argument("--muon-momentum", type=float, default=0.95)
    parser.add_argument("--lr-scheduler", choices=["none", "cosine", "wsd"], default="none")
    parser.add_argument("--warmup-steps", type=int, default=0)
    parser.add_argument(
        "--caption-warmup", action=argparse.BooleanOptionalAction, default=None,
        help="During the first caption_warmup_steps optimizer steps, update only caption-only parameters.",
    )
    parser.add_argument(
        "--caption-warmup-steps", type=int, default=0,
        help="Number of optimizer steps to run caption-only warmup for when caption_warmup is enabled.",
    )
    parser.add_argument(
        "--pretrained-text-encoder-lr", type=float, default=None,
        help="Separate learning rate for the pretrained text backbone (v4, text_encoder_type='pretrained').",
    )
    parser.add_argument(
        "--pretrained-text-encoder-optimizer", choices=["adamw", "muon"], default=None,
        help="Optimizer used for the pretrained text backbone parameters (this fork: adamw only).",
    )
    parser.add_argument(
        "--pretrained-projector-warmup-steps", type=int, default=0,
        help="For the first N optimizer steps, update only the pretrained text/caption projectors.",
    )
    speaker_inversion_group = parser.add_mutually_exclusive_group()
    speaker_inversion_group.add_argument(
        "--speaker-inversion", dest="speaker_inversion_enabled", action="store_true",
        help="Train Speaker Inversion embedding tokens only (all other parameters frozen).",
    )
    speaker_inversion_group.add_argument(
        "--no-speaker-inversion", dest="speaker_inversion_enabled", action="store_false",
        help="Disable Speaker Inversion training.",
    )
    parser.set_defaults(speaker_inversion_enabled=None)
    parser.add_argument("--speaker-inversion-tokens", type=int, default=None,
                        help="Number of Speaker Inversion embedding tokens (default from config: 16).")
    parser.add_argument("--speaker-inversion-init-std", type=float, default=None,
                        help="Stddev for Speaker Inversion embedding initialization.")
    parser.add_argument("--speaker-inversion-init-embedding", default=None,
                        help="Path to a .speaker.safetensors embedding to initialize Speaker Inversion from.")
    parser.add_argument("--stable-steps", type=int, default=0)
    parser.add_argument("--min-lr-scale", type=float, default=0.1)
    parser.add_argument("--latent-dim", type=int, default=128)
    parser.add_argument("--latent-patch-size", type=int, default=1)
    parser.add_argument("--max-latent-steps", type=int, default=750)
    parser.add_argument(
        "--fixed-target-latent-steps", type=int, default=None,
        help="If set, always train on this fixed target latent length.",
    )
    parser.add_argument(
        "--fixed-target-full-mask", action="store_true",
        help="Use full target mask for fixed-length training.",
    )
    parser.add_argument(
        "--rf-loss-mode", choices=["echo", "utterance_mean"], default=None,
        help="RF loss normalization mode.",
    )
    parser.add_argument("--duration-loss-weight", type=float, default=None)
    parser.add_argument("--duration-speaker-dropout", type=float, default=None)
    parser.add_argument("--duration-huber-delta", type=float, default=None)
    parser.add_argument(
        "--text-condition-dropout", type=float, default=0.1,
        help="Probability of dropping text conditioning during training.",
    )
    parser.add_argument(
        "--caption-condition-dropout", type=float, default=0.1,
        help="Probability of dropping caption conditioning during training.",
    )
    parser.add_argument(
        "--speaker-condition-dropout", type=float, default=0.1,
        help="Probability of dropping speaker/reference conditioning during training.",
    )
    parser.add_argument(
        "--timestep-stratified", action="store_true",
        help="Use stratified logit-normal timestep sampling (Echo-style).",
    )
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--save-every", type=int, default=1000)
    parser.add_argument(
        "--checkpoint-best-n", type=int, default=0,
        help="Keep up to N best validation-loss checkpoints. Set 0 to disable.",
    )
    parser.add_argument(
        "--valid-ratio", type=float, default=0.0,
        help="Split ratio for validation set from the single manifest. 0 disables validation.",
    )
    parser.add_argument(
        "--valid-every", type=int, default=0,
        help="Run validation every N training steps. Set <=0 to disable validation.",
    )
    parser.add_argument(
        "--progress", action=argparse.BooleanOptionalAction, default=None,
        help="Enable tqdm progress bar.",
    )
    parser.add_argument(
        "--progress-all", action=argparse.BooleanOptionalAction, default=None,
        help="Show tqdm progress bars for all ranks in DDP mode (default: rank0 only).",
    )
    wandb_group = parser.add_mutually_exclusive_group()
    wandb_group.add_argument("--wandb", dest="wandb_enabled", action="store_true", help="Enable Weights & Biases logging.")
    wandb_group.add_argument("--no-wandb", dest="wandb_enabled", action="store_false", help="Disable Weights & Biases logging.")
    parser.set_defaults(wandb_enabled=None)
    parser.add_argument("--wandb-project", default=None)
    parser.add_argument("--wandb-entity", default=None)
    parser.add_argument("--wandb-run-name", default=None)
    parser.add_argument("--wandb-mode", choices=sorted(WANDB_MODES), default=None)
    parser.add_argument("--metrics-log-dir", default=None,
                        help="Directory to write metrics_log.jsonl (default: <output_dir>/logs).")
    parser.add_argument("--seed", type=int, default=0)
    # ── 拡張オプティマイザ引数（Irodori独自） ──────────────────────────────
    parser.add_argument("--lion-beta1", type=float, default=0.9)
    parser.add_argument("--lion-beta2", type=float, default=0.99)
    parser.add_argument("--ademamix-alpha", type=float, default=5.0)
    parser.add_argument("--ademamix-beta3", type=float, default=0.9999)
    # ── Attentionバックエンド ────────────────────────────────────────────
    parser.add_argument(
        "--attention-backend", choices=["sdpa", "flash2", "eager"], default="sdpa",
        help="Attention実装を選択。sdpa=推奨(デフォルト), flash2=FlashAttention2要インストール, eager=フォールバック。",
    )
    # ── 勾配チェックポイント ──────────────────────────────────────────────
    parser.add_argument(
        "--grad-checkpoint", action="store_true", default=False,
        help="勾配チェックポイントを有効化。VRAMを最大40%%削減(速度約20%%低下)。",
    )
    parser.add_argument(
        "--grad-checkpoint-store-every", type=int, default=0,
        help="勾配チェックポイントで1/Nブロックごとに活性化を保持(再計算削減)。0=全ブロック再計算。"
             "未指定時はtrain.gradient_checkpointing_store_every。",
    )
    # ── EMA ──────────────────────────────────────────────────────────────
    parser.add_argument(
        "--ema-decay", type=float, default=None,
        help="EMA(指数移動平均)のdecay値を指定すると有効化(例: 0.9999)。",
    )
    # ── 勾配クリッピング ──────────────────────────────────────────────────
    parser.add_argument(
        "--clip-grad-norm", type=float, default=1.0,
        help="勾配クリッピングのmax norm値(default: 1.0)。0で無効化。",
    )
    # ── Early Stopping ────────────────────────────────────────────────────
    parser.add_argument("--early-stopping", action="store_true", default=False)
    parser.add_argument("--early-stopping-patience", type=int, default=3)
    parser.add_argument("--early-stopping-min-delta", type=float, default=0.01)
    # ── チェックポイント保存方式 ──────────────────────────────────────────
    parser.add_argument(
        "--save-full", dest="save_full", action="store_true", default=False,
        help="EMA版に加えてフルサイズ版(_full.pt)も保存する。追加学習を行う場合に指定。",
    )
    ddp_group = parser.add_mutually_exclusive_group()
    ddp_group.add_argument("--ddp-find-unused-parameters", dest="ddp_find_unused_parameters", action="store_true")
    ddp_group.add_argument("--no-ddp-find-unused-parameters", dest="ddp_find_unused_parameters", action="store_false")
    parser.set_defaults(ddp_find_unused_parameters=None)
    args = parser.parse_args()

    # --resume に .safetensors は不可（v3以降の仕様）
    if args.resume is not None and Path(args.resume).suffix.lower() == ".safetensors":
        raise ValueError(
            "--resume expects a training checkpoint (.pt). "
            "Use --init-checkpoint for inference-only .safetensors weights."
        )
    if args.resume is not None and args.init_checkpoint is not None:
        raise ValueError("--resume and --init-checkpoint are mutually exclusive.")

    rank, world_size, local_rank, distributed, device = setup_distributed(args.device)
    is_main_process = rank == 0

    raw_argv = sys.argv[1:]
    exp_cfg = load_experiment_yaml(args.config) if args.config else {}
    unknown_root = sorted(set(exp_cfg) - {"model", "train"})
    if unknown_root:
        raise ValueError(f"Unknown top-level config keys: {unknown_root}")
    if args.config and is_main_process:
        print(f"Loaded config: {args.config}")
    model_cfg = merge_dataclass_overrides(ModelConfig(), exp_cfg.get("model"), section="model")
    train_cfg = merge_dataclass_overrides(TrainConfig(), exp_cfg.get("train"), section="train")
    default_train_cfg = TrainConfig()

    train_cfg = replace(train_cfg, manifest_path=args.manifest)
    if train_cfg.output_dir == default_train_cfg.output_dir and not cli_provided(raw_argv, "--output-dir"):
        train_cfg = replace(train_cfg, output_dir=args.output_dir)

    if cli_provided(raw_argv, "--output-dir"):
        train_cfg = replace(train_cfg, output_dir=args.output_dir)
    if cli_provided(raw_argv, "--precision"):
        train_cfg = replace(train_cfg, precision=args.precision)
    if args.allow_tf32 is not None:
        train_cfg = replace(train_cfg, allow_tf32=args.allow_tf32)
    if args.compile_model is not None:
        train_cfg = replace(train_cfg, compile_model=args.compile_model)
    if cli_provided(raw_argv, "--train-mode"):
        train_cfg = replace(train_cfg, train_mode=args.train_mode)
    if cli_provided(raw_argv, "--batch-size"):
        train_cfg = replace(train_cfg, batch_size=args.batch_size)
    if cli_provided(raw_argv, "--gradient-accumulation-steps"):
        train_cfg = replace(train_cfg, gradient_accumulation_steps=args.gradient_accumulation_steps)
    if cli_provided(raw_argv, "--max-text-len"):
        train_cfg = replace(train_cfg, max_text_len=args.max_text_len)
    if cli_provided(raw_argv, "--max-caption-len"):
        train_cfg = replace(train_cfg, max_caption_len=args.max_caption_len)
    if cli_provided(raw_argv, "--num-workers"):
        train_cfg = replace(train_cfg, num_workers=args.num_workers)
    if cli_provided(raw_argv, "--lr"):
        train_cfg = replace(train_cfg, learning_rate=args.lr)
    if cli_provided(raw_argv, "--weight-decay"):
        train_cfg = replace(train_cfg, weight_decay=args.weight_decay)
    if cli_provided(raw_argv, "--optimizer"):
        train_cfg = replace(train_cfg, optimizer=args.optimizer)
    if cli_provided(raw_argv, "--adam-beta1"):
        train_cfg = replace(train_cfg, adam_beta1=args.adam_beta1)
    if cli_provided(raw_argv, "--adam-beta2"):
        train_cfg = replace(train_cfg, adam_beta2=args.adam_beta2)
    if cli_provided(raw_argv, "--adam-eps"):
        train_cfg = replace(train_cfg, adam_eps=args.adam_eps)
    if cli_provided(raw_argv, "--muon-momentum"):
        train_cfg = replace(train_cfg, muon_momentum=args.muon_momentum)
    if cli_provided(raw_argv, "--lr-scheduler"):
        train_cfg = replace(train_cfg, lr_scheduler=args.lr_scheduler)
    if cli_provided(raw_argv, "--warmup-steps"):
        train_cfg = replace(train_cfg, warmup_steps=args.warmup_steps)
    if args.caption_warmup is not None:
        train_cfg = replace(train_cfg, caption_warmup=bool(args.caption_warmup))
    if cli_provided(raw_argv, "--caption-warmup-steps"):
        train_cfg = replace(train_cfg, caption_warmup_steps=args.caption_warmup_steps)
    if cli_provided(raw_argv, "--pretrained-text-encoder-lr"):
        train_cfg = replace(train_cfg, pretrained_text_encoder_learning_rate=args.pretrained_text_encoder_lr)
    if cli_provided(raw_argv, "--pretrained-text-encoder-optimizer"):
        train_cfg = replace(train_cfg, pretrained_text_encoder_optimizer=args.pretrained_text_encoder_optimizer)
    if cli_provided(raw_argv, "--pretrained-projector-warmup-steps"):
        train_cfg = replace(train_cfg, pretrained_projector_warmup_steps=args.pretrained_projector_warmup_steps)
    if args.speaker_inversion_enabled is not None:
        train_cfg = replace(train_cfg, speaker_inversion_enabled=bool(args.speaker_inversion_enabled))
    if cli_provided(raw_argv, "--speaker-inversion-tokens"):
        train_cfg = replace(train_cfg, speaker_inversion_tokens=args.speaker_inversion_tokens)
    if cli_provided(raw_argv, "--speaker-inversion-init-std"):
        train_cfg = replace(train_cfg, speaker_inversion_init_std=args.speaker_inversion_init_std)
    if cli_provided(raw_argv, "--speaker-inversion-init-embedding"):
        train_cfg = replace(train_cfg, speaker_inversion_init_embedding=args.speaker_inversion_init_embedding)
    if cli_provided(raw_argv, "--stable-steps"):
        train_cfg = replace(train_cfg, stable_steps=args.stable_steps)
    if cli_provided(raw_argv, "--min-lr-scale"):
        train_cfg = replace(train_cfg, min_lr_scale=args.min_lr_scale)
    if cli_provided(raw_argv, "--max-steps"):
        train_cfg = replace(train_cfg, max_steps=args.max_steps)
    if cli_provided(raw_argv, "--text-condition-dropout"):
        train_cfg = replace(train_cfg, text_condition_dropout=args.text_condition_dropout)
    if cli_provided(raw_argv, "--caption-condition-dropout"):
        train_cfg = replace(train_cfg, caption_condition_dropout=args.caption_condition_dropout)
    if cli_provided(raw_argv, "--speaker-condition-dropout"):
        train_cfg = replace(train_cfg, speaker_condition_dropout=args.speaker_condition_dropout)
    if cli_provided(raw_argv, "--timestep-stratified"):
        train_cfg = replace(train_cfg, timestep_stratified=True)
    if cli_provided(raw_argv, "--max-latent-steps"):
        train_cfg = replace(train_cfg, max_latent_steps=args.max_latent_steps)
    if cli_provided(raw_argv, "--fixed-target-latent-steps"):
        train_cfg = replace(train_cfg, fixed_target_latent_steps=args.fixed_target_latent_steps)
    if cli_provided(raw_argv, "--fixed-target-full-mask"):
        train_cfg = replace(train_cfg, fixed_target_full_mask=True)
    if cli_provided(raw_argv, "--rf-loss-mode"):
        train_cfg = replace(train_cfg, rf_loss_mode=args.rf_loss_mode)
    if cli_provided(raw_argv, "--duration-loss-weight"):
        train_cfg = replace(train_cfg, duration_loss_weight=args.duration_loss_weight)
    if cli_provided(raw_argv, "--duration-speaker-dropout"):
        train_cfg = replace(train_cfg, duration_speaker_dropout=args.duration_speaker_dropout)
    if cli_provided(raw_argv, "--duration-huber-delta"):
        train_cfg = replace(train_cfg, duration_huber_delta=args.duration_huber_delta)
    if cli_provided(raw_argv, "--log-every"):
        train_cfg = replace(train_cfg, log_every=args.log_every)
    if cli_provided(raw_argv, "--save-every"):
        train_cfg = replace(train_cfg, save_every=args.save_every)
    if cli_provided(raw_argv, "--checkpoint-best-n"):
        train_cfg = replace(train_cfg, checkpoint_best_n=args.checkpoint_best_n)
    if cli_provided(raw_argv, "--valid-ratio"):
        train_cfg = replace(train_cfg, valid_ratio=args.valid_ratio)
    if cli_provided(raw_argv, "--valid-every"):
        train_cfg = replace(train_cfg, valid_every=args.valid_every)
    if args.progress is not None:
        train_cfg = replace(train_cfg, progress=args.progress)
    if args.progress_all is not None:
        train_cfg = replace(train_cfg, progress_all_ranks=args.progress_all)
    if args.wandb_enabled is not None:
        train_cfg = replace(train_cfg, wandb_enabled=args.wandb_enabled)
    if cli_provided(raw_argv, "--wandb-project"):
        train_cfg = replace(train_cfg, wandb_project=args.wandb_project)
    if cli_provided(raw_argv, "--wandb-entity"):
        train_cfg = replace(train_cfg, wandb_entity=args.wandb_entity)
    if cli_provided(raw_argv, "--wandb-run-name"):
        train_cfg = replace(train_cfg, wandb_run_name=args.wandb_run_name)
    if cli_provided(raw_argv, "--wandb-mode"):
        train_cfg = replace(train_cfg, wandb_mode=args.wandb_mode)
    if args.ddp_find_unused_parameters is not None:
        train_cfg = replace(train_cfg, ddp_find_unused_parameters=args.ddp_find_unused_parameters)
    if cli_provided(raw_argv, "--seed"):
        train_cfg = replace(train_cfg, seed=args.seed)

    # 拡張引数はtrain_cfgに存在しないためargs直接参照
    attention_backend: str = args.attention_backend
    use_grad_checkpoint: bool = bool(
        args.grad_checkpoint or train_cfg.gradient_checkpointing
    )
    if cli_provided(raw_argv, "--grad-checkpoint-store-every"):
        grad_checkpoint_store_every: int = int(args.grad_checkpoint_store_every)
    else:
        grad_checkpoint_store_every = int(train_cfg.gradient_checkpointing_store_every)
    ema_decay: float | None = args.ema_decay
    clip_grad_norm: float = args.clip_grad_norm
    lion_betas: tuple[float, float] = (args.lion_beta1, args.lion_beta2)
    ademamix_alpha: float = args.ademamix_alpha
    ademamix_beta3: float = args.ademamix_beta3
    use_early_stopping: bool = args.early_stopping
    es_patience: int = args.early_stopping_patience
    es_min_delta: float = args.early_stopping_min_delta
    save_full: bool = args.save_full

    if cli_provided(raw_argv, "--latent-dim"):
        model_cfg = replace(model_cfg, latent_dim=args.latent_dim)
    if cli_provided(raw_argv, "--latent-patch-size"):
        model_cfg = replace(model_cfg, latent_patch_size=args.latent_patch_size)

    # ── バリデーション ──────────────────────────────────────────────────────
    train_cfg = replace(train_cfg, train_mode=str(train_cfg.train_mode).strip().lower())
    if train_cfg.train_mode not in TRAIN_MODES:
        raise ValueError(f"train_mode must be one of {sorted(TRAIN_MODES)}, got {train_cfg.train_mode!r}")
    # ── MeanFlow 蒸留の設定検証（model_cfg 同期前） ────────────────────────
    _flow_parameterization = str(model_cfg.flow_parameterization).strip().lower()
    if _flow_parameterization not in {"rf_velocity", "meanflow"}:
        raise ValueError(
            "model.flow_parameterization must be 'rf_velocity' or 'meanflow', "
            f"got {model_cfg.flow_parameterization!r}"
        )
    if train_cfg.train_mode == "meanflow_distill":
        if not train_cfg.teacher_checkpoint:
            raise ValueError("train_mode='meanflow_distill' requires train.teacher_checkpoint.")
        if args.resume is None:
            _mf_teacher_path = str(Path(train_cfg.teacher_checkpoint).expanduser())
            if args.init_checkpoint is not None:
                _mf_init_path = str(_normalize_checkpoint_path(args.init_checkpoint))
                if _mf_init_path != str(_normalize_checkpoint_path(_mf_teacher_path)):
                    raise ValueError(
                        "MeanFlow student and teacher must initialize from the same RF checkpoint; "
                        "omit --init-checkpoint or set it equal to train.teacher_checkpoint."
                    )
            args.init_checkpoint = _mf_teacher_path
        model_cfg = replace(model_cfg, flow_parameterization="meanflow")
        if train_cfg.speaker_inversion_enabled:
            raise ValueError("meanflow_distill does not support Speaker Inversion training.")
        if train_cfg.caption_warmup or train_cfg.pretrained_projector_warmup_steps > 0:
            raise ValueError("meanflow_distill does not train condition encoders or their warmups.")
        if train_cfg.duration_backprop_to_condition:
            raise ValueError("meanflow_distill freezes the duration and condition modules.")
        if train_cfg.teacher_steps <= 0:
            raise ValueError("meanflow_distill requires teacher_steps > 0.")
        if not 0.0 <= train_cfg.meanflow_anchor_prob <= 1.0:
            raise ValueError("meanflow_anchor_prob must be in [0, 1].")
        if train_cfg.meanflow_time_logit_std <= 0:
            raise ValueError("meanflow_time_logit_std must be positive.")
        if train_cfg.meanflow_adaptive_weight_power < 0:
            raise ValueError("meanflow_adaptive_weight_power must be non-negative.")
        if train_cfg.meanflow_adaptive_weight_eps <= 0:
            raise ValueError("meanflow_adaptive_weight_eps must be positive.")
        if train_cfg.meanflow_cfg_min_t > train_cfg.meanflow_cfg_max_t:
            raise ValueError("meanflow_cfg_min_t must be <= meanflow_cfg_max_t.")
        if train_cfg.meanflow_teacher_chunk_size <= 0:
            raise ValueError("meanflow_teacher_chunk_size must be positive.")
        if not 1 <= train_cfg.meanflow_teacher_branch_batch_size <= 3:
            raise ValueError("meanflow_teacher_branch_batch_size must be in [1, 3].")
        if train_cfg.meanflow_teacher_device_offset < 0:
            raise ValueError("meanflow_teacher_device_offset must be non-negative.")
        if train_cfg.meanflow_compile_teacher and not hasattr(torch, "compile"):
            raise RuntimeError("meanflow_compile_teacher requires torch.compile (PyTorch 2+).")
    elif _flow_parameterization != "rf_velocity":
        raise ValueError(
            "model.flow_parameterization='meanflow' is only valid with "
            "train_mode='meanflow_distill'."
        )
    if str(train_cfg.rf_loss_mode).strip().lower() not in {"echo", "utterance_mean"}:
        raise ValueError(f"rf_loss_mode must be 'echo' or 'utterance_mean', got {train_cfg.rf_loss_mode!r}")
    if not (0.0 <= train_cfg.text_condition_dropout <= 1.0):
        raise ValueError(f"text_condition_dropout must be in [0, 1], got {train_cfg.text_condition_dropout}")
    if not (0.0 <= train_cfg.caption_condition_dropout <= 1.0):
        raise ValueError(f"caption_condition_dropout must be in [0, 1], got {train_cfg.caption_condition_dropout}")
    if not (0.0 <= train_cfg.speaker_condition_dropout <= 1.0):
        raise ValueError(f"speaker_condition_dropout must be in [0, 1], got {train_cfg.speaker_condition_dropout}")
    if train_cfg.max_text_len <= 0:
        raise ValueError(f"max_text_len must be > 0, got {train_cfg.max_text_len}")
    if train_cfg.gradient_accumulation_steps <= 0:
        raise ValueError(f"gradient_accumulation_steps must be > 0, got {train_cfg.gradient_accumulation_steps}")
    if train_cfg.fixed_target_latent_steps is not None and train_cfg.fixed_target_latent_steps <= 0:
        raise ValueError(f"fixed_target_latent_steps must be > 0 when provided, got {train_cfg.fixed_target_latent_steps}")
    if train_cfg.fixed_target_full_mask and train_cfg.fixed_target_latent_steps is None:
        raise ValueError("fixed_target_full_mask=True requires fixed_target_latent_steps to be set.")
    if train_cfg.train_mode == "duration_only" and not model_cfg.use_duration_predictor:
        raise ValueError("train_mode='duration_only' requires model.use_duration_predictor=True.")
    if not (0.0 <= train_cfg.valid_ratio < 1.0):
        raise ValueError(f"valid_ratio must be in [0, 1), got {train_cfg.valid_ratio}")
    if train_cfg.valid_ratio > 0.0 and train_cfg.valid_every <= 0:
        raise ValueError("valid_every must be > 0 when valid_ratio > 0.")
    if train_cfg.valid_ratio == 0.0 and train_cfg.valid_every > 0 and is_main_process:
        print("warning: valid_every is set but valid_ratio=0. Validation is disabled.")
    if train_cfg.checkpoint_best_n < 0:
        raise ValueError(f"checkpoint_best_n must be >= 0, got {train_cfg.checkpoint_best_n}")
    if train_cfg.wandb_mode not in WANDB_MODES:
        raise ValueError(f"wandb_mode must be one of {sorted(WANDB_MODES)}, got {train_cfg.wandb_mode!r}")
    if train_cfg.dataloader_prefetch_factor <= 0:
        raise ValueError(f"dataloader_prefetch_factor must be > 0, got {train_cfg.dataloader_prefetch_factor}")
    precision = str(train_cfg.precision).lower()
    if precision not in {"fp32", "bf16"}:
        raise ValueError(f"precision must be one of ['fp32', 'bf16'], got {train_cfg.precision!r}")
    if precision == "bf16":
        if device.type != "cuda":
            if is_main_process:
                print("warning: precision=bf16 requested on non-CUDA device. Falling back to fp32.")
            train_cfg = replace(train_cfg, precision="fp32")
        elif not torch.cuda.is_bf16_supported():
            if is_main_process:
                print("warning: CUDA bf16 is not supported on this GPU. Falling back to fp32.")
            train_cfg = replace(train_cfg, precision="fp32")
    use_bf16 = train_cfg.precision == "bf16"
    if device.type == "cuda":
        tf32_enabled = bool(train_cfg.allow_tf32)
        torch.backends.cuda.matmul.allow_tf32 = tf32_enabled
        torch.backends.cudnn.allow_tf32 = tf32_enabled
        torch.set_float32_matmul_precision("high" if tf32_enabled else "highest")
        if is_main_process:
            print(f"TF32 matmul/cuDNN: {'enabled' if tf32_enabled else 'disabled'}")
    elif train_cfg.allow_tf32 and is_main_process:
        print("warning: allow_tf32=True requested on non-CUDA device; ignoring.")

    set_seed(train_cfg.seed + rank)

    output_dir = Path(train_cfg.output_dir)
    if is_main_process:
        output_dir.mkdir(parents=True, exist_ok=True)
        dump_configs(output_dir / "config.json", model_cfg, train_cfg)
        print(f"Compute precision={train_cfg.precision} (weights/optimizer states kept in fp32).")
    if distributed:
        dist.barrier()
    if is_main_process and distributed:
        print(f"DDP enabled: world_size={world_size} (local_rank={local_rank})")

    wandb_run = None
    _metrics_log_file = None
    if train_cfg.wandb_enabled and is_main_process:
        if args.metrics_log_dir is not None:
            _metrics_log_dir = Path(args.metrics_log_dir)
        else:
            _metrics_log_dir = output_dir / "logs"
        _metrics_log_dir.mkdir(parents=True, exist_ok=True)
        _metrics_log_path = _metrics_log_dir / "metrics_log.jsonl"
        _metrics_log_file = open(_metrics_log_path, "a", encoding="utf-8", buffering=1)
        print(f"Local metrics log enabled: {_metrics_log_path}")

    # ── --init-checkpoint からモデル設定を自動判別して model_cfg を更新 ──────
    # --resume (.pt) も同様に model_cfg を自動同期する
    _sync_checkpoint_path: Path | None = None
    _sync_model_cfg_dict: dict | None = None
    _sync_train_cfg_dict: dict | None = None
    _text_encoder_config: dict | None = None  # v4系 事前学習テキストエンコーダ設定
    if args.init_checkpoint is not None:
        _sync_checkpoint_path = Path(args.init_checkpoint).expanduser()
    elif args.resume is not None:
        _sync_checkpoint_path = Path(args.resume).expanduser()
    else:
        # デフォルト候補: v4.1 → v4 → v3 → v2 の順で自動参照
        _DEFAULT_PRETRAINED_SAFETENSORS_CANDIDATES = [
            _PROJECT_CHECKPOINTS_DIR / "Aratako_Irodori-TTS-v4.1-Small" / "model.safetensors",
            _PROJECT_CHECKPOINTS_DIR / "Aratako_Irodori-TTS-v4-Small" / "model.safetensors",
            _PROJECT_CHECKPOINTS_DIR / "Aratako_Irodori-TTS-500M-v3" / "model.safetensors",
            _PROJECT_CHECKPOINTS_DIR / "Aratako_Irodori-TTS-500M-v2" / "model.safetensors",
            _PROJECT_CHECKPOINTS_DIR / "Aratako_Irodori-TTS-500M" / "model.safetensors",
        ]
        _auto = next((p for p in _DEFAULT_PRETRAINED_SAFETENSORS_CANDIDATES if p.exists()), None)
        if _auto is not None:
            _sync_checkpoint_path = _auto
            if is_main_process:
                print(f"--init-checkpoint not specified. Auto-loading: {_auto}")
            args.init_checkpoint = str(_auto)

    if _sync_checkpoint_path is not None and _sync_checkpoint_path.exists():
        _sync_state, _sync_model_cfg_dict, _sync_train_cfg_dict, _sync_te_cfg = _load_model_state_from_checkpoint(_sync_checkpoint_path)
        if _sync_te_cfg is not None:
            _text_encoder_config = _sync_te_cfg
        if _sync_model_cfg_dict is not None:
            _has_caption = checkpoint_uses_caption_condition(_sync_model_cfg_dict, _sync_state)
            _has_duration = checkpoint_uses_duration_predictor(_sync_model_cfg_dict, _sync_state)
            _sync_overrides: dict = {}
            for _k in ("latent_dim", "latent_patch_size", "text_vocab_size", "text_dim", "model_dim",
                        "num_layers", "num_heads", "adaln_rank", "text_tokenizer_repo", "text_add_bos",
                        "speaker_dim", "speaker_layers", "speaker_heads", "speaker_patch_size",
                        "text_encoder_type", "pretrained_projector_type",
                        "pretrained_projector_hidden_ratio", "pretrained_projector_dropout",
                        "use_speaker_condition", "text_layers", "text_heads",
                        "text_mlp_ratio"):
                if _k == "latent_dim" and cli_provided(raw_argv, "--latent-dim"):
                    continue
                if _k == "latent_patch_size" and cli_provided(raw_argv, "--latent-patch-size"):
                    continue
                _v = _sync_model_cfg_dict.get(_k)
                if _v is not None:
                    _sync_overrides[_k] = _v
            if _has_caption:
                _sync_overrides["use_caption_condition"] = True
                for _k in ("caption_vocab_size", "caption_tokenizer_repo", "caption_add_bos",
                            "caption_dim", "caption_layers", "caption_heads", "caption_mlp_ratio"):
                    _v = _sync_model_cfg_dict.get(_k)
                    if _v is not None:
                        _sync_overrides[_k] = _v
            if _has_duration:
                _sync_overrides["use_duration_predictor"] = True
                for _k in ("duration_aux_dim", "duration_hidden_dim", "duration_layers",
                            "duration_dropout", "duration_attention_heads",
                            "duration_architecture", "duration_token_init_frames",
                            "duration_speaker_fusion"):
                    _v = _sync_model_cfg_dict.get(_k)
                    if _v is not None:
                        _sync_overrides[_k] = _v
            if _sync_overrides:
                model_cfg = replace(model_cfg, **_sync_overrides)
                if is_main_process:
                    from irodori_tts import model_version as _mv
                    _ver_info = _mv.classify_model_version(_sync_model_cfg_dict, _sync_state.keys())
                    if _ver_info is not None:
                        print(f"Model version detected: {_mv.describe_model_version(_ver_info)}")
                    else:
                        _ver = "v3" if (_has_caption or _has_duration) else "v2/v1"
                        print(f"Model version detected: {_ver} (caption={_has_caption}, duration={_has_duration})")
                    if _text_encoder_config is not None:
                        print("Embedded text_encoder_config found; building pretrained backbone without HF download.")
                    print(f"Synchronized model_cfg from checkpoint: {_sync_checkpoint_path}")
        del _sync_state

    # ── MeanFlow resume: 学習目的（teacher・時間サンプリング等）の一致検証 ──
    if train_cfg.train_mode == "meanflow_distill" and args.resume is not None:
        _validate_meanflow_resume_objective(
            train_cfg,
            resume_train_cfg=_sync_train_cfg_dict,
            resume_model_cfg=_sync_model_cfg_dict,
        )

    # ── v4 / Speaker Inversion 関連のバリデーション（model_cfg 同期後） ────
    _pre_opt_name = str(train_cfg.pretrained_text_encoder_optimizer).strip().lower()
    if _pre_opt_name not in {"adamw", "muon"}:
        raise ValueError(
            "pretrained_text_encoder_optimizer must be one of ['adamw', 'muon'], "
            f"got {train_cfg.pretrained_text_encoder_optimizer!r}."
        )
    if _pre_opt_name == "muon" and str(train_cfg.optimizer).strip().lower() != "muon":
        raise ValueError(
            "pretrained_text_encoder_optimizer='muon' requires optimizer='muon'."
        )
    if train_cfg.pretrained_projector_warmup_steps < 0:
        raise ValueError(
            "pretrained_projector_warmup_steps must be >= 0, "
            f"got {train_cfg.pretrained_projector_warmup_steps}"
        )
    if train_cfg.pretrained_projector_warmup_steps > 0 and not model_cfg.use_pretrained_text_encoder:
        raise ValueError(
            "pretrained_projector_warmup_steps requires model.text_encoder_type='pretrained'."
        )
    if train_cfg.pretrained_projector_warmup_steps > 0 and train_cfg.caption_warmup:
        raise ValueError("pretrained_projector_warmup_steps and caption_warmup cannot be combined.")
    if train_cfg.speaker_inversion_enabled:
        if not model_cfg.use_speaker_condition_resolved:
            raise ValueError(
                "speaker_inversion_enabled=True requires a speaker-conditioned model config."
            )
        if args.init_checkpoint is None and args.resume is None:
            raise ValueError(
                "speaker_inversion_enabled=True requires --init-checkpoint so the frozen "
                "base model is loaded first."
            )
        if args.resume is not None:
            raise ValueError(
                "speaker_inversion_enabled=True saves embedding-only checkpoints; "
                "--resume is not supported. Use --speaker-inversion-init-embedding "
                "to continue from a saved embedding."
            )
        if train_cfg.train_mode != "rf":
            raise ValueError("speaker_inversion_enabled=True supports train_mode='rf' only.")
        if train_cfg.caption_warmup or train_cfg.pretrained_projector_warmup_steps > 0:
            raise ValueError(
                "speaker_inversion_enabled=True does not support caption_warmup or "
                "pretrained_projector_warmup_steps."
            )
        if train_cfg.optimizer != "adamw":
            raise ValueError(
                "speaker_inversion_enabled=True supports optimizer='adamw' "
                f"(got {train_cfg.optimizer!r}) because only the embedding tokens are trained."
            )
        if train_cfg.speaker_inversion_tokens <= 0:
            raise ValueError(
                f"speaker_inversion_tokens must be > 0, got {train_cfg.speaker_inversion_tokens}"
            )
        if train_cfg.speaker_inversion_init_std < 0:
            raise ValueError(
                "speaker_inversion_init_std must be >= 0, "
                f"got {train_cfg.speaker_inversion_init_std}"
            )

    # ── tokenizer / caption tokenizer ────────────────────────────────────
    if distributed:
        if is_main_process:
            tokenizer = build_text_tokenizer(model_cfg, local_files_only=False)
            text_hidden_size = validate_text_backbone_dim(model_cfg, local_files_only=False)
            caption_tokenizer = None
            if model_cfg.use_caption_condition:
                from irodori_tts.tokenizer import PretrainedTextTokenizer as _PT
                caption_tokenizer = _PT.from_pretrained(
                    repo_id=model_cfg.caption_tokenizer_repo_resolved,
                    add_bos=model_cfg.caption_add_bos_resolved,
                    local_files_only=False,
                )
        dist.barrier()
        if not is_main_process:
            tokenizer = build_text_tokenizer(model_cfg, local_files_only=True)
            text_hidden_size = validate_text_backbone_dim(model_cfg, local_files_only=True)
            caption_tokenizer = None
            if model_cfg.use_caption_condition:
                from irodori_tts.tokenizer import PretrainedTextTokenizer as _PT
                caption_tokenizer = _PT.from_pretrained(
                    repo_id=model_cfg.caption_tokenizer_repo_resolved,
                    add_bos=model_cfg.caption_add_bos_resolved,
                    local_files_only=True,
                )
        dist.barrier()
    else:
        tokenizer = build_text_tokenizer(model_cfg, local_files_only=False)
        text_hidden_size = validate_text_backbone_dim(model_cfg, local_files_only=False)
        caption_tokenizer = None
        if model_cfg.use_caption_condition:
            from irodori_tts.tokenizer import PretrainedTextTokenizer as _PT
            caption_tokenizer = _PT.from_pretrained(
                repo_id=model_cfg.caption_tokenizer_repo_resolved,
                add_bos=model_cfg.caption_add_bos_resolved,
                local_files_only=False,
            )

    if is_main_process:
        print(
            f"Text tokenizer={model_cfg.text_tokenizer_repo} vocab={tokenizer.vocab_size} "
            f"add_bos={model_cfg.text_add_bos} (pretrained hidden_size={text_hidden_size})."
        )
        if model_cfg.use_caption_condition and caption_tokenizer is not None:
            print(f"Caption tokenizer={model_cfg.caption_tokenizer_repo_resolved} vocab={caption_tokenizer.vocab_size}.")

    # ── Dataset / DataLoader ──────────────────────────────────────────────
    full_dataset = LatentTextDataset(
        manifest_path=train_cfg.manifest_path,
        latent_dim=model_cfg.latent_dim,
        max_latent_steps=train_cfg.max_latent_steps,
        enable_caption_condition=model_cfg.use_caption_condition,
        enable_speaker_condition=model_cfg.use_speaker_condition_resolved,
    )
    train_dataset = full_dataset
    valid_dataset = None
    if train_cfg.valid_ratio > 0.0:
        train_indices, valid_indices = split_train_valid_indices(
            num_samples=len(full_dataset), valid_ratio=train_cfg.valid_ratio, seed=train_cfg.seed,
        )
        train_dataset = LatentTextDataset(
            manifest_path=train_cfg.manifest_path,
            latent_dim=model_cfg.latent_dim,
            max_latent_steps=train_cfg.max_latent_steps,
            subset_indices=train_indices,
            enable_caption_condition=model_cfg.use_caption_condition,
            enable_speaker_condition=model_cfg.use_speaker_condition_resolved,
        )
        valid_dataset = LatentTextDataset(
            manifest_path=train_cfg.manifest_path,
            latent_dim=model_cfg.latent_dim,
            max_latent_steps=train_cfg.max_latent_steps,
            subset_indices=valid_indices,
            enable_caption_condition=model_cfg.use_caption_condition,
            enable_speaker_condition=model_cfg.use_speaker_condition_resolved,
        )
        if is_main_process:
            print(
                f"Validation split enabled: train={len(train_dataset)} valid={len(valid_dataset)} "
                f"(ratio={train_cfg.valid_ratio:.4f}, valid_every={train_cfg.valid_every} steps)."
            )
    drop_last = len(train_dataset) >= train_cfg.batch_size
    if not drop_last and is_main_process:
        print(f"warning: dataset size ({len(train_dataset)}) is smaller than batch_size ({train_cfg.batch_size}). Using drop_last=False.")
    collator = TTSCollator(
        tokenizer=tokenizer,
        caption_tokenizer=caption_tokenizer,
        latent_dim=model_cfg.latent_dim,
        latent_patch_size=model_cfg.latent_patch_size,
        fixed_target_latent_steps=train_cfg.fixed_target_latent_steps,
        fixed_target_full_mask=train_cfg.fixed_target_full_mask,
        max_text_len=train_cfg.max_text_len,
        max_caption_len=(
            train_cfg.max_text_len if train_cfg.max_caption_len is None else train_cfg.max_caption_len
        ),
    )
    if train_cfg.fixed_target_latent_steps is not None and is_main_process:
        print(f"Fixed target latent length enabled: steps={train_cfg.fixed_target_latent_steps} full_mask={train_cfg.fixed_target_full_mask}")
    if train_cfg.timestep_stratified and is_main_process:
        print("Using stratified logit-normal timestep sampling.")

    train_sampler = None
    if distributed:
        from torch.utils.data import DistributedSampler
        train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True, drop_last=drop_last)
    dataloader_common_kwargs = {
        "batch_size": train_cfg.batch_size,
        "num_workers": train_cfg.num_workers,
        "pin_memory": (device.type == "cuda"),
        "collate_fn": collator,
    }
    if train_cfg.num_workers > 0:
        dataloader_common_kwargs["persistent_workers"] = bool(train_cfg.dataloader_persistent_workers)
        dataloader_common_kwargs["prefetch_factor"] = int(train_cfg.dataloader_prefetch_factor)
    elif train_cfg.dataloader_persistent_workers and is_main_process:
        print("warning: dataloader_persistent_workers=True is ignored because num_workers=0.")
    loader = DataLoader(dataset=train_dataset, shuffle=(train_sampler is None), sampler=train_sampler, drop_last=drop_last, **dataloader_common_kwargs)
    if len(loader) == 0:
        raise ValueError("Dataloader yielded zero batches. Check manifest and batch_size settings.")
    valid_loader = None
    valid_sampler = None
    if valid_dataset is not None:
        if distributed:
            valid_sampler = DistributedSampler(valid_dataset, num_replicas=world_size, rank=rank, shuffle=False, drop_last=False)
        valid_loader = DataLoader(dataset=valid_dataset, shuffle=False, sampler=valid_sampler, drop_last=False, **dataloader_common_kwargs)
        if len(valid_loader) == 0:
            raise ValueError("Validation dataloader yielded zero batches. Decrease batch_size or valid_ratio.")

    has_validation = valid_loader is not None and train_cfg.valid_every > 0
    checkpoint_retention_enabled = train_cfg.checkpoint_best_n > 0
    periodic_checkpoint_keep = 0
    if checkpoint_retention_enabled:
        periodic_checkpoint_keep = 1 if has_validation else int(train_cfg.checkpoint_best_n) + 1
    best_val_checkpoints: list[tuple[float, int, Path]] = []
    if is_main_process:
        if checkpoint_retention_enabled and has_validation:
            best_val_checkpoints = list_best_val_loss_checkpoints(output_dir)
            best_val_checkpoints = prune_best_val_loss_checkpoints(best_val_checkpoints, train_cfg.checkpoint_best_n)
        if checkpoint_retention_enabled and has_validation:
            print(f"Checkpoint retention: latest=1 + best_val_loss={train_cfg.checkpoint_best_n}.")
        elif checkpoint_retention_enabled:
            print(f"Checkpoint retention: validation disabled, keep latest {periodic_checkpoint_keep} periodic checkpoints.")

    # ── モデル構築 ───────────────────────────────────────────────────────
    # v4系でチェックポイントに text_encoder_config が埋め込まれている場合は、
    # HFから事前学習重みをダウンロードせず空構築して重みを流し込む。
    meanflow_mode = train_cfg.train_mode == "meanflow_distill"
    local_world_size = int(os.environ.get("LOCAL_WORLD_SIZE", str(world_size)))
    meanflow_teacher_device = resolve_meanflow_teacher_device(
        student_device=device,
        local_rank=local_rank,
        local_world_size=local_world_size,
        device_offset=(train_cfg.meanflow_teacher_device_offset if meanflow_mode else 0),
    )
    # MeanFlow student: fresh run は RF velocity で構築して teacher を複製し、
    # 之後 delta 条件モジュールを追加する。resume は meanflow cfg で直接構築する。
    construction_model_cfg = (
        replace(model_cfg, flow_parameterization="rf_velocity")
        if meanflow_mode and args.resume is None
        else model_cfg
    )
    raw_model = TextToLatentRFDiT(
        construction_model_cfg,
        pretrained_backbone_config=_text_encoder_config,
        load_pretrained_backbone_weights=not (
            model_cfg.use_pretrained_text_encoder and _text_encoder_config is not None
        ),
    ).to(device)

    # --init-checkpoint または自動参照: モデル重みのみロード
    if args.init_checkpoint is not None:
        init_path = Path(args.init_checkpoint).expanduser()
        init_state, init_model_cfg_dict, _, _init_te_cfg = _load_model_state_from_checkpoint(init_path)
        if _text_encoder_config is None and _init_te_cfg is not None:
            _text_encoder_config = _init_te_cfg
        _has_caption_ckpt = checkpoint_uses_caption_condition(init_model_cfg_dict, init_state)
        _has_duration_ckpt = checkpoint_uses_duration_predictor(init_model_cfg_dict, init_state)
        _current_has_caption = bool(model_cfg.use_caption_condition)
        _current_has_duration = bool(model_cfg.use_duration_predictor)
        _upgrade_caption = _current_has_caption and not _has_caption_ckpt
        _upgrade_duration = _current_has_duration and not _has_duration_ckpt
        if _upgrade_caption or _upgrade_duration:
            missing_keys, skipped_shape, skipped_extra = load_model_state_partially(raw_model, init_state)
            if skipped_shape and is_main_process:
                print(f"Partial load skipped (shape mismatch): {len(skipped_shape)} keys")
            if is_main_process:
                print(f"Partial init from: {init_path} (missing={len(missing_keys)})")
        else:
            # shape mismatch があると strict=False でも RuntimeError になるため、
            # 事前に shape を確認して不一致があれば部分ロードにフォールバックする
            _model_state = raw_model.state_dict()
            _has_shape_mismatch = any(
                k in _model_state and tuple(_model_state[k].shape) != tuple(v.shape)
                for k, v in init_state.items()
            )
            if _has_shape_mismatch:
                missing, skipped_shape, skipped_extra = load_model_state_partially(raw_model, init_state)
                if is_main_process:
                    print(f"Loaded weights from: {init_path} (partial: shape mismatch fallback)")
                    print(f"  Shape mismatch skipped: {len(skipped_shape)} keys")
                    if missing:
                        print(f"  Missing keys: {len(missing)}")
                        _summarize_missing_keys(missing)
            else:
                missing, unexpected = raw_model.load_state_dict(init_state, strict=False)
                if is_main_process:
                    print(f"Loaded weights from: {init_path}")
                    if missing:
                        print(f"  Missing keys: {len(missing)}")
                        _summarize_missing_keys(missing)
                    if unexpected:
                        print(f"  Unexpected keys: {len(unexpected)}")
        # caption embeddingをpretrained initializeする（v3アップグレード時）
        # pretrainedテキストエンコーダ(v4)はcaption_encoderがprojectorのため対象外。
        if _upgrade_caption and is_main_process and not model_cfg.use_pretrained_text_encoder:
            print(f"Initializing caption embedding from pretrained: {model_cfg.caption_tokenizer_repo_resolved}")
            try:
                from transformers import AutoModel as _AM
                _backbone = _AM.from_pretrained(model_cfg.caption_tokenizer_repo_resolved, trust_remote_code=False, dtype=torch.float32, low_cpu_mem_usage=True)
                _src = _backbone.get_input_embeddings().weight.detach().float()
                _tgt = raw_model.caption_encoder.text_embedding.weight
                _rows = min(_src.shape[0], _tgt.shape[0])
                with torch.no_grad():
                    _tgt[:_rows].copy_(_src[:_rows].to(device=_tgt.device, dtype=_tgt.dtype))
                del _backbone
            except Exception as _e:
                print(f"  warning: caption embedding init failed: {_e}")
    elif args.resume is None:
        # スクラッチ: text embeddingをpretrained初期化（pretrained TE(v4)は対象外）
        if model_cfg.use_pretrained_text_encoder:
            if is_main_process:
                print("Scratch training with text_encoder_type='pretrained'; "
                      "backbone weights loaded from HF at construction.")
        else:
            if is_main_process:
                print(f"Initializing text embedding from pretrained model: {model_cfg.text_tokenizer_repo}")
            initialize_text_embedding_from_pretrained(raw_model, model_cfg, local_files_only=False)

    # ── MeanFlow 蒸留: teacher 構築と student の parameterization 切替 ──────
    # 勾配チェックポイント適用前（closure が元モジュールを捕捉しない時点）に構築する。
    meanflow_teacher: TextToLatentRFDiT | None = None
    if meanflow_mode:
        if args.resume is None:
            # fresh: RF 重み込みの student を複製して teacher にする
            if meanflow_teacher_device == device:
                meanflow_teacher = copy.deepcopy(raw_model)
            else:
                _mf_t_state, _, _, _ = _load_model_state_from_checkpoint(
                    _normalize_checkpoint_path(train_cfg.teacher_checkpoint)
                )
                meanflow_teacher = TextToLatentRFDiT(
                    construction_model_cfg,
                    pretrained_backbone_config=_text_encoder_config,
                    load_pretrained_backbone_weights=False,
                ).to(meanflow_teacher_device)
                meanflow_teacher.load_state_dict(_mf_t_state, strict=True)
                del _mf_t_state
            raw_model.enable_meanflow_parameterization()
            raw_model.cfg = model_cfg
        else:
            # resume: student は meanflow cfg で構築済み。teacher は別経路で再構築。
            _mf_t_path = _normalize_checkpoint_path(train_cfg.teacher_checkpoint)
            _mf_t_state, _mf_t_cfg_dict, _, _mf_t_te_cfg = _load_model_state_from_checkpoint(
                _mf_t_path
            )
            _mf_teacher_cfg = replace(model_cfg, flow_parameterization="rf_velocity")
            if _mf_t_cfg_dict is not None:
                _mf_teacher_cfg = merge_dataclass_overrides(
                    ModelConfig(), _mf_t_cfg_dict, section="teacher model_config"
                )
            if str(_mf_teacher_cfg.flow_parameterization).strip().lower() != "rf_velocity":
                raise ValueError("train.teacher_checkpoint must be an RF velocity checkpoint.")
            meanflow_teacher = TextToLatentRFDiT(
                _mf_teacher_cfg,
                pretrained_backbone_config=_mf_t_te_cfg,
                load_pretrained_backbone_weights=not _mf_teacher_cfg.use_pretrained_text_encoder,
            ).to(meanflow_teacher_device)
            meanflow_teacher.load_state_dict(_mf_t_state, strict=True)
            del _mf_t_state, _mf_t_cfg_dict, _mf_t_te_cfg, _mf_teacher_cfg
        for parameter in meanflow_teacher.parameters():
            parameter.requires_grad_(False)
        meanflow_teacher.float().eval()
        if train_cfg.meanflow_compile_teacher:
            object.__setattr__(
                meanflow_teacher,
                "_meanflow_compiled_forward",
                torch.compile(
                    meanflow_teacher.forward_with_encoded_conditions,
                    dynamic=True,
                    fullgraph=False,
                ),
            )
        if is_main_process:
            print(f"MeanFlow teacher initialized on {meanflow_teacher_device}.")

    # ── Speaker Inversion 学習: embedding トークン追加 + 他パラメータ凍結 ──
    if train_cfg.speaker_inversion_enabled:
        from irodori_tts.speaker_inversion import (
            SPEAKER_EMBEDDING_KEY,
            load_speaker_inversion_payload,
        )
        _si_init_embedding = None
        if train_cfg.speaker_inversion_init_embedding is not None:
            _si_payload = load_speaker_inversion_payload(train_cfg.speaker_inversion_init_embedding)
            _si_init_embedding = _si_payload[SPEAKER_EMBEDDING_KEY]
            if is_main_process:
                print(f"Loaded Speaker Inversion init embedding: {train_cfg.speaker_inversion_init_embedding}")
        _speaker_inversion = raw_model.enable_speaker_inversion(
            num_tokens=train_cfg.speaker_inversion_tokens,
            init_std=train_cfg.speaker_inversion_init_std,
            init_embedding=_si_init_embedding,
        )
        _speaker_inversion.to(device)
        if is_main_process:
            print(
                f"Speaker Inversion parameters initialized: "
                f"embedding={tuple(_speaker_inversion.embedding.shape)}."
            )
        _trainable_si, _frozen_si = freeze_for_speaker_inversion(raw_model)
        if _trainable_si == 0:
            raise RuntimeError("No Speaker Inversion parameters were found.")
        if is_main_process:
            print(
                f"Speaker Inversion freeze applied: trainable={_trainable_si:,} frozen={_frozen_si:,}."
            )

    # ── MeanFlow 蒸留: DiT 以外を凍結（student DiT 全体のみ学習） ──────────
    if meanflow_mode:
        _trainable_dit, _frozen_mf = freeze_for_meanflow(raw_model)
        if _trainable_dit == 0:
            raise RuntimeError("No student DiT parameters were found for MeanFlow training.")
        set_meanflow_frozen_modules_eval(raw_model)
        if is_main_process:
            print(
                "MeanFlow distillation enabled: "
                f"trainable_dit={_trainable_dit:,} frozen={_frozen_mf:,} "
                f"teacher_steps={train_cfg.teacher_steps} "
                f"teacher_device={meanflow_teacher_device} "
                f"teacher_chunk={train_cfg.meanflow_teacher_chunk_size} "
                f"branch_batch={train_cfg.meanflow_teacher_branch_batch_size} "
                f"fuse_base_branch={train_cfg.meanflow_teacher_fuse_base_branch} "
                f"compile_teacher={train_cfg.meanflow_compile_teacher}."
            )

    train_model = raw_model
    if train_cfg.compile_model:
        if not hasattr(torch, "compile"):
            raise RuntimeError("compile_model=True requires torch.compile (PyTorch 2+).")
        if is_main_process:
            print("torch.compile enabled (dynamic=True).")
        train_model = torch.compile(raw_model, dynamic=True)

    # ── Attentionバックエンド設定 ────────────────────────────────────────
    if device.type == "cuda":
        apply_attention_backend(raw_model, attention_backend)

    # ── 勾配チェックポイント ─────────────────────────────────────────────
    if use_grad_checkpoint:
        if grad_checkpoint_store_every < 0:
            raise ValueError(
                "gradient_checkpointing_store_every must be >= 0, "
                f"got {grad_checkpoint_store_every}"
            )
        # DiffusionBlock と事前学習TEは model.py 内蔵機構で処理（store_every 対応）
        raw_model.set_gradient_checkpointing(True, store_every=grad_checkpoint_store_every)
        # TextBlock（テキスト/キャプションエンコーダ）はクロージャ方式で適用
        apply_gradient_checkpointing(raw_model)
        if meanflow_mode and raw_model.pretrained_text_backbone is not None:
            # MeanFlow は事前学習TEを学習しないため checkpointing しない
            raw_model.pretrained_text_backbone.set_gradient_checkpointing(False)
        if is_main_process:
            scope = "diffusion blocks"
            if model_cfg.use_pretrained_text_encoder and not meanflow_mode:
                scope += " and the pretrained text encoder (when supported)"
            store_note = (
                ""
                if grad_checkpoint_store_every == 0
                else f" (activations kept for 1 of every {grad_checkpoint_store_every} blocks)"
            )
            print(f"Gradient checkpointing enabled on {scope}{store_note}.")

    # ── EMA初期化 ────────────────────────────────────────────────────────
    ema_model: EMAModel | None = None
    if ema_decay is not None:
        if not (0.0 < ema_decay < 1.0):
            raise ValueError(f"--ema-decay must be in (0, 1), got {ema_decay}")
        ema_model = EMAModel(raw_model, decay=ema_decay)
        if is_main_process:
            print(f"EMA enabled (decay={ema_decay})")

    # ── Early Stopping初期化 ─────────────────────────────────────────────
    early_stopper: EarlyStopping | None = None
    if use_early_stopping:
        if train_cfg.valid_ratio <= 0.0:
            raise ValueError("--early-stopping requires --valid-ratio > 0.")
        early_stopper = EarlyStopping(patience=es_patience, min_delta=es_min_delta, mode="min")
        if is_main_process:
            print(f"Early Stopping: enabled (patience={es_patience}, min_delta={es_min_delta})")

    ddp_find_unused_parameters = bool(train_cfg.ddp_find_unused_parameters)
    ddp_find_unused_parameters_explicit = args.ddp_find_unused_parameters is not None or (
        isinstance(exp_cfg.get("train"), dict) and "ddp_find_unused_parameters" in exp_cfg.get("train", {})
    )
    if distributed:
        if not ddp_find_unused_parameters and not ddp_find_unused_parameters_explicit:
            try:
                speaker_labeled_count = train_dataset.speaker_labeled_count
            except AttributeError:
                speaker_labeled_count = sum(1 for x in train_dataset.samples if x.get("speaker_id") is not None)
            has_partial_or_no_speaker_labels = speaker_labeled_count < len(train_dataset)
            has_stochastic_cond_drop = (
                train_cfg.text_condition_dropout > 0.0
                or train_cfg.speaker_condition_dropout > 0.0
                or (model_cfg.use_caption_condition and train_cfg.caption_condition_dropout > 0.0)
            )
            if has_partial_or_no_speaker_labels or has_stochastic_cond_drop:
                ddp_find_unused_parameters = True
                if is_main_process:
                    print("DDP find_unused_parameters auto-enabled (conditional branches may be fully masked in some steps).")
        model = DDP(train_model, device_ids=[local_rank], output_device=local_rank,
                    find_unused_parameters=ddp_find_unused_parameters, broadcast_buffers=False)
    else:
        model = train_model

    optimizer = build_optimizer_extended(
        raw_model, train_cfg,
        optimizer_name=train_cfg.optimizer,
        lion_betas=lion_betas,
        ademamix_alpha=ademamix_alpha,
        ademamix_beta3=ademamix_beta3,
    )
    scheduler = build_scheduler(optimizer, train_cfg)
    if is_main_process:
        print(f"Optimizer={train_cfg.optimizer} Scheduler={train_cfg.lr_scheduler} lr={current_lr(optimizer):.3e}")
        if train_cfg.gradient_accumulation_steps > 1:
            print(f"Gradient accumulation enabled: steps={train_cfg.gradient_accumulation_steps} (effective global batch={train_cfg.batch_size * world_size * train_cfg.gradient_accumulation_steps}).")

    # ── --resume: optimizer/scheduler/step を復元 ────────────────────────
    step = 0
    if args.resume is not None:
        resume_path = Path(args.resume).expanduser()
        ckpt = torch.load(resume_path, map_location=device, weights_only=True)
        raw_model.load_state_dict(ckpt["model"])
        try:
            optimizer.load_state_dict(ckpt["optimizer"])
        except (ValueError, KeyError, TypeError) as exc:
            # pretrained text encoder optimizer 分割など、optimizer 構成が変わった場合
            if is_main_process:
                print(
                    "Warning: optimizer state does not match the current optimizer configuration; "
                    f"continuing without restoring optimizer state ({exc})."
                )
        step = int(ckpt["step"])
        if scheduler is not None:
            scheduler_state = ckpt.get("scheduler")
            if scheduler_state is not None:
                scheduler.load_state_dict(scheduler_state)
            elif step > 0:
                scheduler.last_step = step
        if ema_model is not None and "ema" in ckpt:
            ema_model.load_state_dict(ckpt["ema"])
            if is_main_process:
                print("EMA state restored from checkpoint.")
        if is_main_process:
            print(f"Resumed from step={step}")

    progress = TrainProgress(
        max_steps=train_cfg.max_steps,
        start_step=step,
        rank=rank,
        world_size=world_size,
        enabled=train_cfg.progress,
        show_all_ranks=train_cfg.progress_all_ranks,
        description=(
            "Train Duration" if train_cfg.train_mode == "duration_only"
            else ("Train MeanFlow" if train_cfg.train_mode == "meanflow_distill" else "Train RF")
        ),
    )
    accum_steps = int(train_cfg.gradient_accumulation_steps)
    global_batch_size = train_cfg.batch_size * world_size * accum_steps
    duration_only = train_cfg.train_mode == "duration_only"

    # caption warmup: caption-only パラメータのみを最初のN stepだけ更新
    caption_warmup_active = bool(
        train_cfg.caption_warmup
        and model_cfg.use_caption_condition
        and train_cfg.caption_warmup_steps > 0
        and step < train_cfg.caption_warmup_steps
    )
    if caption_warmup_active and is_main_process:
        print(f"Caption warmup active: caption-only parameters only for first {train_cfg.caption_warmup_steps} optimizer steps.")

    # pretrained projector warmup: text/caption projector のみ最初のN stepだけ更新
    pretrained_projector_warmup_active = bool(
        model_cfg.use_pretrained_text_encoder
        and train_cfg.pretrained_projector_warmup_steps > 0
        and step < train_cfg.pretrained_projector_warmup_steps
    )
    if pretrained_projector_warmup_active and is_main_process:
        print(
            f"Pretrained projector warmup active: projectors only for first "
            f"{train_cfg.pretrained_projector_warmup_steps} optimizer steps."
        )

    # ETA推定用
    _step_times: list[float] = []
    _ETA_WINDOW = 100

    def _fmt_eta(s: float) -> str:
        s = int(s)
        h, rem = divmod(s, 3600)
        m, sec = divmod(rem, 60)
        if h > 0:
            return f"{h}時間{m}分"
        elif m > 0:
            return f"{m}分{sec}秒"
        else:
            return f"{sec}秒"

    try:
        model.train()
        if meanflow_mode:
            set_meanflow_frozen_modules_eval(raw_model)
            if meanflow_teacher is None:
                raise RuntimeError("MeanFlow teacher was not initialized.")
            meanflow_teacher.eval()
        if scheduler is not None and step == 0:
            scheduler.step()
        optimizer.zero_grad(set_to_none=True)
        accum_micro_steps = 0
        accum_loss = torch.zeros((), device=device, dtype=torch.float32)
        accum_rf_loss = torch.zeros((), device=device, dtype=torch.float32)
        accum_duration_loss = torch.zeros((), device=device, dtype=torch.float32)
        epoch = 0
        while step < train_cfg.max_steps and not (early_stopper is not None and early_stopper.should_stop):
            if train_sampler is not None:
                train_sampler.set_epoch(epoch)
            epoch += 1
            for epoch_step, batch in enumerate(loader, start=1):
                accum_micro_steps += 1
                text_ids = batch["text_ids"].to(device, non_blocking=True)
                text_mask = batch["text_mask"].to(device, non_blocking=True)
                caption_ids = None
                caption_mask = None
                has_caption = None
                if raw_model.cfg.use_caption_condition:
                    caption_ids = batch["caption_ids"].to(device, non_blocking=True)
                    caption_mask = batch["caption_mask"].to(device, non_blocking=True)
                    has_caption = batch["has_caption"].to(device, non_blocking=True)
                num_frames = batch["num_frames"].to(device, non_blocking=True)
                duration_features = batch["duration_features"].to(device, non_blocking=True)
                ref_latent = None
                ref_mask = None
                if raw_model.cfg.use_speaker_condition_resolved:
                    ref_latent = batch["ref_latent_patched"].to(device, non_blocking=True)
                    ref_mask = batch["ref_latent_mask_patched"].to(device, non_blocking=True)
                    has_speaker = batch["has_speaker"].to(device, non_blocking=True)
                else:
                    has_speaker = None

                bsz = text_ids.shape[0]
                x_mask = x_mask_valid = x_t = t = v_target = None
                if not duration_only:
                    x0 = batch["latent_patched"].to(device, non_blocking=True)
                    x_mask = batch["latent_mask_patched"].to(device, non_blocking=True)
                    x_mask_valid = batch["latent_mask_valid_patched"].to(device, non_blocking=True)
                    if train_cfg.timestep_stratified and not meanflow_mode:
                        t = sample_stratified_logit_normal_t(
                            batch_size=bsz, device=device,
                            mean=train_cfg.timestep_logit_mean, std=train_cfg.timestep_logit_std,
                            t_min=train_cfg.timestep_min, t_max=train_cfg.timestep_max,
                        )
                    elif not meanflow_mode:
                        t = sample_logit_normal_t(
                            batch_size=bsz, device=device,
                            mean=train_cfg.timestep_logit_mean, std=train_cfg.timestep_logit_std,
                            t_min=train_cfg.timestep_min, t_max=train_cfg.timestep_max,
                        )
                    # MeanFlow の t/r/delta_t と x_t は loss 計算側でサンプリングする
                    if not meanflow_mode:
                        noise = torch.randn_like(x0)
                        x_t = rf_interpolate(x0, noise, t)
                        v_target = rf_velocity_target(x0, noise)

                # ── Conditioning dropout ────────────────────────────────
                text_cond_drop = torch.rand(bsz, device=device) < train_cfg.text_condition_dropout
                caption_drop_for_model = None
                duration_has_caption = None
                if raw_model.cfg.use_caption_condition:
                    caption_cond_drop = torch.rand(bsz, device=device) < train_cfg.caption_condition_dropout
                    use_caption = has_caption & (~caption_cond_drop)
                    caption_drop_for_model = ~use_caption
                    if not meanflow_mode:
                        duration_caption_drop = torch.rand(bsz, device=device) < train_cfg.duration_caption_dropout
                        duration_has_caption = has_caption & (~duration_caption_drop)
                    if not raw_model.cfg.use_duration_predictor and not meanflow_mode:
                        caption_mask = caption_mask & use_caption[:, None]

                speaker_drop_for_model = None
                duration_has_speaker = None
                if raw_model.cfg.use_speaker_condition_resolved:
                    speaker_cond_drop = torch.rand(bsz, device=device) < train_cfg.speaker_condition_dropout
                    if train_cfg.speaker_inversion_enabled:
                        # Speaker Inversion は1つの学習済みembeddingを全サンプルで共有するため、
                        # マニフェストに speaker_id が無くても全サンプル話者条件付きとして扱う。
                        use_speaker = ~speaker_cond_drop
                        speaker_drop_for_model = speaker_cond_drop
                    else:
                        use_speaker = has_speaker & (~speaker_cond_drop)
                        speaker_drop_for_model = ~use_speaker
                    if not meanflow_mode:
                        duration_speaker_drop = torch.rand(bsz, device=device) < train_cfg.duration_speaker_dropout
                        if train_cfg.speaker_inversion_enabled:
                            duration_has_speaker = ~duration_speaker_drop
                        else:
                            duration_has_speaker = has_speaker & (~duration_speaker_drop)
                        duration_features = set_duration_has_speaker_feature(duration_features, duration_has_speaker)
                    if (
                        not raw_model.cfg.use_duration_predictor
                        and not train_cfg.speaker_inversion_enabled
                        and not meanflow_mode
                    ):
                        # v2以前: 手動マスク
                        ref_mask = ref_mask & use_speaker[:, None]
                        ref_latent = ref_latent * use_speaker[:, None, None].to(ref_latent.dtype)
                elif not raw_model.cfg.use_caption_condition:
                    # v2以前 speakerあり: 手動マスク（use_speaker_condition=Trueが本来のケース）
                    if has_speaker is not None:
                        speaker_cond_drop = torch.rand(bsz, device=device) < train_cfg.speaker_condition_dropout
                        use_speaker = has_speaker & (~speaker_cond_drop)
                        ref_mask = ref_mask & use_speaker[:, None]
                        ref_latent = ref_latent * use_speaker[:, None, None].to(ref_latent.dtype)

                should_step = (accum_micro_steps % accum_steps) == 0
                sync_context = model.no_sync() if distributed and not should_step else nullcontext()
                with sync_context:
                    meanflow_loss = None
                    with (
                        torch.autocast(device_type="cuda", dtype=torch.bfloat16)
                        if use_bf16
                        else nullcontext()
                    ):
                        if meanflow_mode:
                            if (
                                meanflow_teacher is None
                                or x0 is None
                                or x_mask is None
                                or x_mask_valid is None
                            ):
                                raise RuntimeError("MeanFlow training tensors are missing.")
                            meanflow_loss = compute_meanflow_distillation_loss(
                                model=model,
                                teacher=meanflow_teacher,
                                train_cfg=train_cfg,
                                x0=x0,
                                latent_mask=x_mask,
                                valid_mask=x_mask_valid,
                                text_ids=text_ids,
                                text_mask=text_mask,
                                ref_latent=ref_latent,
                                ref_mask=ref_mask,
                                caption_ids=caption_ids,
                                caption_mask=caption_mask,
                                text_drop=text_cond_drop,
                                speaker_drop=speaker_drop_for_model,
                                caption_drop=caption_drop_for_model,
                                use_bf16=use_bf16,
                            )
                            v_pred = None
                            duration_pred = None
                        elif duration_only:
                            duration_pred = model(
                                x_t=None, t=None,
                                text_input_ids=text_ids, text_mask=text_mask,
                                ref_latent=ref_latent, ref_mask=ref_mask,
                                caption_input_ids=caption_ids, caption_mask=caption_mask,
                                latent_mask=None,
                                duration_features=duration_features,
                                duration_has_speaker=duration_has_speaker,
                                duration_has_caption=duration_has_caption,
                                duration_only=True,
                            )
                            v_pred = None
                        elif raw_model.cfg.use_duration_predictor:
                            v_pred, duration_pred = model(
                                x_t=x_t, t=t,
                                text_input_ids=text_ids, text_mask=text_mask,
                                ref_latent=ref_latent, ref_mask=ref_mask,
                                caption_input_ids=caption_ids, caption_mask=caption_mask,
                                latent_mask=x_mask,
                                text_condition_dropout=text_cond_drop,
                                speaker_condition_dropout=speaker_drop_for_model,
                                caption_condition_dropout=caption_drop_for_model,
                                duration_features=duration_features,
                                duration_has_speaker=duration_has_speaker,
                                duration_has_caption=duration_has_caption,
                            )
                        else:
                            # v2以前
                            v_pred = model(
                                x_t=x_t, t=t,
                                text_input_ids=text_ids, text_mask=text_mask,
                                ref_latent=ref_latent, ref_mask=ref_mask,
                                caption_input_ids=caption_ids, caption_mask=caption_mask,
                                latent_mask=x_mask,
                                speaker_condition_dropout=(
                                    speaker_drop_for_model if train_cfg.speaker_inversion_enabled else None
                                ),
                            )
                            duration_pred = None

                    rf_loss = torch.zeros((), device=device, dtype=torch.float32)
                    if meanflow_mode:
                        if meanflow_loss is None:
                            raise RuntimeError("MeanFlow loss is missing.")
                        rf_loss = meanflow_loss
                    elif not duration_only:
                        v_pred = v_pred.float()
                        rf_loss = compute_rf_loss(
                            pred=v_pred, target=v_target.float(),
                            loss_mask=x_mask, valid_mask=x_mask_valid,
                            mode=train_cfg.rf_loss_mode,
                        )
                    duration_loss = torch.zeros((), device=device, dtype=torch.float32)
                    if raw_model.cfg.use_duration_predictor and duration_pred is not None:
                        duration_target = torch.log1p(num_frames.float())
                        duration_loss = F.huber_loss(
                            duration_pred.float(), duration_target,
                            delta=float(train_cfg.duration_huber_delta), reduction="mean",
                        )
                    if duration_only:
                        loss = duration_loss
                    else:
                        loss = rf_loss + float(train_cfg.duration_loss_weight) * duration_loss
                    (loss / float(accum_steps)).backward()

                    # pretrained projector warmup: text/caption projector のgradだけ残す
                    if pretrained_projector_warmup_active:
                        _clear_non_pretrained_projector_grads(raw_model)

                    # caption warmup: caption以外のgradをゼロ化
                    if caption_warmup_active:
                        for key, param in raw_model.named_parameters():
                            _is_caption = (
                                key.startswith("caption_encoder.")
                                or key.startswith("caption_norm.")
                                or ".wk_caption." in key
                                or ".wv_caption." in key
                            )
                            if not _is_caption and param.grad is not None:
                                param.grad = None

                accum_loss += loss.detach()
                accum_rf_loss += rf_loss.detach()
                accum_duration_loss += duration_loss.detach()
                if not should_step:
                    continue

                step_loss = accum_loss / float(accum_steps)
                step_rf_loss = accum_rf_loss / float(accum_steps)
                step_duration_loss = accum_duration_loss / float(accum_steps)
                accum_loss.zero_()
                accum_rf_loss.zero_()
                accum_duration_loss.zero_()

                torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad_norm if clip_grad_norm > 0 else float("inf"))
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                if scheduler is not None:
                    scheduler.step()
                if ema_model is not None:
                    ema_model.update(raw_model)
                step += 1
                _step_times.append(_time.monotonic())
                if len(_step_times) > _ETA_WINDOW + 1:
                    _step_times.pop(0)
                progress.update(step)

                if caption_warmup_active and step >= train_cfg.caption_warmup_steps:
                    caption_warmup_active = False
                    if is_main_process:
                        progress.write("Caption warmup complete; all parameters are now updating.")

                if (
                    pretrained_projector_warmup_active
                    and step >= train_cfg.pretrained_projector_warmup_steps
                ):
                    pretrained_projector_warmup_active = False
                    if is_main_process:
                        progress.write("Pretrained projector warmup complete; all parameters are now updating.")

                # ── ログ出力 ──────────────────────────────────────────────
                if step <= 5 or step % train_cfg.log_every == 0:
                    loss_value = reduce_mean(step_loss, world_size, distributed).item()
                    rf_loss_value = reduce_mean(step_rf_loss, world_size, distributed).item()
                    dur_loss_value = reduce_mean(step_duration_loss, world_size, distributed).item()
                    lr_value = current_lr(optimizer)
                    progress_metrics: dict[str, float] = {"loss": loss_value, "rf": rf_loss_value, "lr": lr_value}
                    if raw_model.cfg.use_duration_predictor:
                        progress_metrics["dur"] = dur_loss_value
                    progress.log(
                        step=step, epoch=epoch, epoch_step=epoch_step, epoch_total=len(loader),
                        metrics=progress_metrics, global_batch_size=global_batch_size,
                    )
                    if is_main_process:
                        if len(_step_times) >= 2:
                            _elapsed = _step_times[-1] - _step_times[0]
                            _sps = _elapsed / (len(_step_times) - 1)
                            _remaining = max(0, train_cfg.max_steps - step)
                            _eta_str = _fmt_eta(_sps * _remaining)
                            _speed_str = f"{1.0/_sps:.3f}steps/s" if _sps > 0 else "?"
                            _log_msg = f"step={step} loss={loss_value:.6f} rf={rf_loss_value:.6f}"
                            if raw_model.cfg.use_duration_predictor:
                                _log_msg += f" dur={dur_loss_value:.6f}"
                            _log_msg += f" lr={lr_value:.3e} speed={_speed_str} eta={_eta_str}"
                        else:
                            _log_msg = f"step={step} loss={loss_value:.6f} rf={rf_loss_value:.6f} lr={lr_value:.3e}"
                        progress.write(_log_msg)
                        if _metrics_log_file is not None:
                            _wmetrics = {"type": "train", "step": step, "loss": loss_value, "rf_loss": rf_loss_value, "lr": lr_value}
                            if raw_model.cfg.use_duration_predictor:
                                _wmetrics["duration_loss"] = dur_loss_value
                            _metrics_log_file.write(json.dumps(_wmetrics) + "\n")

                # ── チェックポイント保存 ──────────────────────────────────
                if step % train_cfg.save_every == 0 and is_main_process:
                    save_checkpoint(
                        output_dir / f"checkpoint_{step:07d}.pt",
                        raw_model, optimizer, scheduler, step, model_cfg, train_cfg,
                        ema_model=ema_model, save_full=save_full,
                        text_encoder_config=_text_encoder_config,
                    )
                    enforce_periodic_checkpoint_limit(output_dir=output_dir, keep_count=periodic_checkpoint_keep)

                # ── バリデーション ────────────────────────────────────────
                if valid_loader is not None and train_cfg.valid_every > 0 and step % train_cfg.valid_every == 0:
                    valid_metrics = run_validation(
                        model=model, teacher=meanflow_teacher, loader=valid_loader, train_cfg=train_cfg,
                        device=device, use_bf16=use_bf16, distributed=distributed,
                    )
                    if is_main_process:
                        _vmsg = f"valid step={step} loss={valid_metrics['loss']:.6f} rf={valid_metrics['rf_loss']:.6f}"
                        if raw_model.cfg.use_duration_predictor:
                            _vmsg += f" dur={valid_metrics['duration_loss']:.6f}"
                        _vmsg += f" (samples={valid_metrics['num_samples']:.0f})"
                        progress.write(_vmsg)
                        if _metrics_log_file is not None:
                            _vmetrics = {"type": "valid", "step": step, "loss": valid_metrics["loss"], "rf_loss": valid_metrics["rf_loss"]}
                            if raw_model.cfg.use_duration_predictor:
                                _vmetrics["duration_loss"] = valid_metrics["duration_loss"]
                            _metrics_log_file.write(json.dumps(_vmetrics) + "\n")
                        best_val_checkpoints, best_path = maybe_save_best_val_loss_checkpoint(
                            output_dir=output_dir, checkpoints=best_val_checkpoints,
                            keep_best_n=train_cfg.checkpoint_best_n,
                            val_loss=float(valid_metrics["loss"]), step=step,
                            model=raw_model, optimizer=optimizer, scheduler=scheduler,
                            model_cfg=model_cfg, train_cfg=train_cfg,
                            ema_model=ema_model, save_full=save_full,
                            text_encoder_config=_text_encoder_config,
                        )
                        if best_path is not None:
                            progress.write(f"saved best val checkpoint: {best_path.name} (loss={valid_metrics['loss']:.6f})")
                        if early_stopper is not None:
                            should_stop_early = early_stopper.step(score=float(valid_metrics["loss"]), current_step=step)
                            progress.write(early_stopper.status())
                            if should_stop_early:
                                progress.write(
                                    f"Early Stopping: valid lossが{es_patience}回連続で改善しませんでした。"
                                    f"最良はstep={early_stopper.best_step} (valid loss={early_stopper.best_score:.6f})。学習を停止します。"
                                )
                                save_checkpoint(
                                    output_dir / f"checkpoint_early_stop_{step:07d}.pt",
                                    raw_model, optimizer, scheduler, step, model_cfg, train_cfg,
                                    ema_model=ema_model, save_full=save_full,
                                    text_encoder_config=_text_encoder_config,
                                )

                if step >= train_cfg.max_steps:
                    break
                if early_stopper is not None and early_stopper.should_stop:
                    break

        # ── 最終バリデーション ────────────────────────────────────────────
        if valid_loader is not None and train_cfg.valid_every > 0 and step % train_cfg.valid_every != 0:
            valid_metrics = run_validation(
                model=model, teacher=meanflow_teacher, loader=valid_loader, train_cfg=train_cfg,
                device=device, use_bf16=use_bf16, distributed=distributed,
            )
            if is_main_process:
                _vmsg = f"valid final step={step} loss={valid_metrics['loss']:.6f} rf={valid_metrics['rf_loss']:.6f}"
                if raw_model.cfg.use_duration_predictor:
                    _vmsg += f" dur={valid_metrics['duration_loss']:.6f}"
                _vmsg += f" (samples={valid_metrics['num_samples']:.0f})"
                progress.write(_vmsg)
                if _metrics_log_file is not None:
                    _vmetrics = {"type": "valid", "step": step, "loss": valid_metrics["loss"], "rf_loss": valid_metrics["rf_loss"]}
                    if raw_model.cfg.use_duration_predictor:
                        _vmetrics["duration_loss"] = valid_metrics["duration_loss"]
                    _metrics_log_file.write(json.dumps(_vmetrics) + "\n")
                best_val_checkpoints, best_path = maybe_save_best_val_loss_checkpoint(
                    output_dir=output_dir, checkpoints=best_val_checkpoints,
                    keep_best_n=train_cfg.checkpoint_best_n,
                    val_loss=float(valid_metrics["loss"]), step=step,
                    model=raw_model, optimizer=optimizer, scheduler=scheduler,
                    model_cfg=model_cfg, train_cfg=train_cfg,
                    ema_model=ema_model, save_full=save_full,
                    text_encoder_config=_text_encoder_config,
                )
                if best_path is not None:
                    progress.write(f"saved best val checkpoint: {best_path.name} (loss={valid_metrics['loss']:.6f})")

        if is_main_process:
            save_checkpoint(
                output_dir / "checkpoint_final.pt",
                raw_model, optimizer, scheduler, step, model_cfg, train_cfg,
                ema_model=ema_model, save_full=save_full,
                text_encoder_config=_text_encoder_config,
            )
            if _metrics_log_file is not None:
                pass  # final_step はログ行から取得可能なため summary 不要
            progress.write(f"Training finished at step={step}.")
    finally:
        if progress is not None:
            progress.close()
        if _metrics_log_file is not None:
            _metrics_log_file.close()
            _metrics_log_file = None

        for _dl in [loader, valid_loader]:
            if _dl is None:
                continue
            try:
                _dl._iterator = None
            except Exception:
                pass
            try:
                if hasattr(_dl, "_workers") and _dl._workers:
                    for _w in _dl._workers:
                        try:
                            _w.terminate()
                        except Exception:
                            pass
            except Exception:
                pass

        if is_main_process:
            print("モデルをVRAMからアンロード中...")
            try:
                raw_model.cpu()
            except Exception:
                pass
            if ema_model is not None:
                try:
                    ema_model.shadow.clear()
                    ema_model.backup.clear()
                except Exception:
                    pass
            try:
                del raw_model
            except Exception:
                pass
            if ema_model is not None:
                try:
                    del ema_model
                except Exception:
                    pass
            if device.type == "cuda":
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
            print("VRAMアンロード完了")

        if distributed and dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
