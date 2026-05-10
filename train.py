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
CHECKPOINT_STEP_RE = re.compile(r"^checkpoint_(\d+)\.pt$")
CHECKPOINT_BEST_VAL_LOSS_RE = re.compile(r"^checkpoint_best_val_loss_(\d+)_(-?\d+(?:\.\d+)?)\.pt$")
SAFETENSORS_CONFIG_META_KEY = "config_json"

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
        self.shadow: dict[str, torch.Tensor] = {}
        self.backup: dict[str, torch.Tensor] = {}
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone().float()

    @torch.no_grad()
    def update(self, model: torch.nn.Module) -> None:
        for name, param in model.named_parameters():
            if param.requires_grad and name in self.shadow:
                self.shadow[name].mul_(self.decay).add_(
                    param.data.float(), alpha=1.0 - self.decay
                )

    def apply_shadow(self, model: torch.nn.Module) -> None:
        """EMAの重みをモデルに適用（チェックポイント保存用）"""
        for name, param in model.named_parameters():
            if param.requires_grad and name in self.shadow:
                self.backup[name] = param.data.clone()
                param.data.copy_(self.shadow[name].to(param.dtype))

    def restore(self, model: torch.nn.Module) -> None:
        """学習用の元の重みに戻す"""
        for name, param in model.named_parameters():
            if param.requires_grad and name in self.backup:
                param.data.copy_(self.backup[name])
        self.backup.clear()

    def state_dict(self) -> dict:
        return {"decay": self.decay, "shadow": self.shadow}

    def load_state_dict(self, state: dict) -> None:
        self.decay = state["decay"]
        self.shadow = state["shadow"]


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
    """モデルのDiffusionBlock/TextBlockに勾配チェックポイントを適用する。
    VRAMを最大40%削減できるが、再計算のため学習速度が約20%低下する。

    Irodori-TTSのモデル構造（model.py）に対応:
      - DiffusionBlock : forward(x, cond_embed, text_state, text_mask,
                                 speaker_state, speaker_mask, freqs_cis,
                                 self_mask=None, context_kv=None)
      - TextBlock       : forward(x, mask, freqs_cis)
    """
    from torch.utils.checkpoint import checkpoint as ckpt_fn

    TARGET_CLASSES = ("DiffusionBlock", "TextBlock")
    count = 0

    for module in model.modules():
        cls_name = type(module).__name__
        if cls_name not in TARGET_CLASSES:
            continue

        # クロージャでmoduleをキャプチャし元のforwardを保持
        orig_forward = module.forward

        if cls_name == "DiffusionBlock":
            def make_diffusion_ckpt(orig):
                def checkpointed(
                    x, cond_embed, text_state, text_mask,
                    speaker_state, speaker_mask, freqs_cis,
                    self_mask=None, context_kv=None,
                ):
                    # context_kvはTensorでないためuse_reentrant=Falseで渡す
                    def _fn(x_, ce_, ts_, tm_, ss_, sm_, fc_):
                        return orig(
                            x_, ce_, ts_, tm_, ss_, sm_, fc_,
                            self_mask=self_mask, context_kv=context_kv,
                        )
                    return ckpt_fn(
                        _fn, x, cond_embed, text_state, text_mask,
                        speaker_state, speaker_mask, freqs_cis,
                        use_reentrant=False,
                    )
                return checkpointed
            module.forward = make_diffusion_ckpt(orig_forward)

        elif cls_name == "TextBlock":
            def make_text_ckpt(orig):
                def checkpointed(x, mask, freqs_cis):
                    def _fn(x_, fc_):
                        return orig(x_, mask, fc_)
                    return ckpt_fn(_fn, x, freqs_cis, use_reentrant=False)
                return checkpointed
            module.forward = make_text_ckpt(orig_forward)

        count += 1

    if count > 0:
        print(f"Gradient checkpointing: enabled ({count}ブロックに適用, VRAM削減, 速度約20%低下)")
    else:
        print("warning: gradient checkpointing: DiffusionBlock/TextBlockが見つかりませんでした。")


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
) -> None:
    """チェックポイントを保存する。

    保存方針:
    - EMAが有効な場合:
        checkpoint_XXXXXXX_ema.pt  ... EMA版（推論用・軽量）常に保存
        checkpoint_XXXXXXX_full.pt ... フルサイズ版（追加学習用）--save-full指定時のみ
    - EMAが無効な場合:
        checkpoint_XXXXXXX.pt      ... 従来通りフルサイズのみ
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    if ema_model is not None:
        # EMA版を _ema.pt で保存（推論用・軽量）
        ema_path = path.with_name(path.stem + "_ema" + path.suffix)
        ema_model.apply_shadow(model)
        ema_payload = {
            "step": step,
            "model": model.state_dict(),
            "model_config": asdict(model_cfg),
        }
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
        torch.save(payload, path)


def _safe_unlink(path: Path) -> None:
    try:
        path.unlink()
    except FileNotFoundError:
        return


def list_periodic_checkpoints(output_dir: Path) -> list[tuple[int, Path]]:
    checkpoints: list[tuple[int, Path]] = []
    for path in output_dir.glob("checkpoint_*.pt"):
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
    for path in output_dir.glob("checkpoint_best_val_loss_*.pt"):
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
    if hidden_size != model_cfg.text_dim:
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


def _load_model_state_from_checkpoint(path: Path) -> tuple[dict[str, torch.Tensor], dict | None]:
    if path.suffix.lower() == ".safetensors":
        from safetensors import safe_open
        from safetensors.torch import load_file as load_safetensors_file

        checkpoint_model_cfg = None
        with safe_open(str(path), framework="pt", device="cpu") as handle:
            metadata = dict(handle.metadata() or {})
        config_json = metadata.get(SAFETENSORS_CONFIG_META_KEY)
        if config_json:
            parsed = json.loads(config_json)
            if isinstance(parsed, dict):
                checkpoint_model_cfg = parsed
        return load_safetensors_file(str(path), device="cpu"), checkpoint_model_cfg

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
    return raw_model, checkpoint_model_cfg


def _check_model_config_compatibility(
    checkpoint_path: Path,
    checkpoint_model_cfg: dict | None,
    current_model_cfg: ModelConfig,
) -> None:
    if checkpoint_model_cfg is None:
        return

    for key in ("latent_dim", "latent_patch_size", "text_vocab_size", "text_dim", "model_dim"):
        checkpoint_value = checkpoint_model_cfg.get(key)
        if checkpoint_value is None:
            continue
        current_value = getattr(current_model_cfg, key)
        if int(checkpoint_value) != int(current_value):
            raise ValueError(
                f"Checkpoint/config mismatch for '{key}': checkpoint={checkpoint_value} "
                f"current={current_value} ({checkpoint_path})"
            )


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


def run_validation(
    *,
    model,
    loader: DataLoader,
    train_cfg: TrainConfig,
    device: torch.device,
    use_bf16: bool,
    distributed: bool,
) -> dict[str, float]:
    was_training = model.training
    model.eval()
    totals = torch.zeros(3, device=device, dtype=torch.float64)

    with torch.no_grad():
        for batch in loader:
            text_ids = batch["text_ids"].to(device, non_blocking=True)
            text_mask = batch["text_mask"].to(device, non_blocking=True)
            x0 = batch["latent_patched"].to(device, non_blocking=True)
            x_mask = batch["latent_mask_patched"].to(device, non_blocking=True)
            x_mask_valid = batch["latent_mask_valid_patched"].to(device, non_blocking=True)
            ref_latent = batch["ref_latent_patched"].to(device, non_blocking=True)
            ref_mask = batch["ref_latent_mask_patched"].to(device, non_blocking=True)
            has_speaker = batch["has_speaker"].to(device, non_blocking=True)

            bsz = x0.shape[0]
            if train_cfg.timestep_stratified:
                t = sample_stratified_logit_normal_t(
                    batch_size=bsz,
                    device=device,
                    mean=train_cfg.timestep_logit_mean,
                    std=train_cfg.timestep_logit_std,
                    t_min=train_cfg.timestep_min,
                    t_max=train_cfg.timestep_max,
                )
            else:
                t = sample_logit_normal_t(
                    batch_size=bsz,
                    device=device,
                    mean=train_cfg.timestep_logit_mean,
                    std=train_cfg.timestep_logit_std,
                    t_min=train_cfg.timestep_min,
                    t_max=train_cfg.timestep_max,
                )
            noise = torch.randn_like(x0)
            x_t = rf_interpolate(x0, noise, t)
            v_target = rf_velocity_target(x0, noise)

            use_speaker = has_speaker
            ref_mask = ref_mask & use_speaker[:, None]
            ref_latent = ref_latent * use_speaker[:, None, None].to(ref_latent.dtype)

            with (
                torch.autocast(device_type="cuda", dtype=torch.bfloat16)
                if use_bf16
                else nullcontext()
            ):
                v_pred = model(
                    x_t=x_t,
                    t=t,
                    text_input_ids=text_ids,
                    text_mask=text_mask,
                    ref_latent=ref_latent,
                    ref_mask=ref_mask,
                    latent_mask=x_mask,
                )

            v_pred = v_pred.float()
            rf_loss = echo_style_masked_mse(
                v_pred,
                v_target.float(),
                loss_mask=x_mask,
                valid_mask=x_mask_valid,
            )
            loss = rf_loss

            weight = float(bsz)
            totals[0] += loss.detach().double() * weight
            totals[1] += rf_loss.detach().double() * weight
            totals[2] += weight

    if distributed:
        dist.all_reduce(totals, op=dist.ReduceOp.SUM)
    denom = max(float(totals[2].item()), 1.0)
    metrics = {
        "loss": float(totals[0].item() / denom),
        "rf_loss": float(totals[1].item() / denom),
        "num_samples": float(totals[2].item()),
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
        "--precision",
        choices=["fp32", "bf16"],
        default="bf16",
        help=(
            "Compute precision for model forward pass. "
            "Model weights and optimizer states remain FP32."
        ),
    )
    parser.add_argument(
        "--tf32",
        dest="allow_tf32",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Enable TF32 matmul/cuDNN kernels on CUDA for speed.",
    )
    parser.add_argument(
        "--compile-model",
        dest="compile_model",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Enable torch.compile for the training model.",
    )
    parser.add_argument(
        "--resume",
        default=None,
        help="Resume full training state from a training checkpoint (.pt).",
    )
    parser.add_argument(
        "--init-checkpoint",
        default=None,
        help=(
            "Initialize model weights from a checkpoint (.pt or .safetensors) and start a new run "
            "with fresh optimizer / scheduler state."
        ),
    )
    parser.add_argument("--max-steps", type=int, default=200000)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument(
        "--gradient-accumulation-steps",
        type=int,
        default=1,
        help=(
            "Number of micro-batches to accumulate before optimizer.step(). "
            "1 disables accumulation."
        ),
    )
    parser.add_argument(
        "--max-text-len",
        type=int,
        default=256,
        help="Maximum token length for text conditioning (right-truncated).",
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
    parser.add_argument("--stable-steps", type=int, default=0)
    parser.add_argument("--min-lr-scale", type=float, default=0.1)
    parser.add_argument("--latent-dim", type=int, default=128)
    parser.add_argument("--latent-patch-size", type=int, default=1)
    parser.add_argument("--max-latent-steps", type=int, default=750)
    parser.add_argument(
        "--fixed-target-latent-steps",
        type=int,
        default=None,
        help=(
            "If set, always train on this fixed target latent length "
            "(short samples are right-padded with zeros, long samples are truncated)."
        ),
    )
    parser.add_argument(
        "--fixed-target-full-mask",
        action="store_true",
        help="Use full target mask for fixed-length training (Echo-style includes padded tail in loss).",
    )
    parser.add_argument(
        "--text-condition-dropout",
        type=float,
        default=0.1,
        help="Probability of dropping text conditioning during training.",
    )
    parser.add_argument(
        "--speaker-condition-dropout",
        type=float,
        default=0.1,
        help="Probability of dropping speaker/reference conditioning during training.",
    )
    parser.add_argument(
        "--timestep-stratified",
        action="store_true",
        help="Use stratified logit-normal timestep sampling (Echo-style).",
    )
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--save-every", type=int, default=1000)
    parser.add_argument(
        "--checkpoint-best-n",
        type=int,
        default=0,
        help=(
            "Keep up to N best validation-loss checkpoints in addition to latest. "
            "When validation is disabled, keeps latest N+1 periodic checkpoints. "
            "Set 0 to disable checkpoint-count limiting."
        ),
    )
    parser.add_argument(
        "--valid-ratio",
        type=float,
        default=0.0,
        help=("Split ratio for validation set from the single manifest. 0 disables validation."),
    )
    parser.add_argument(
        "--valid-every",
        type=int,
        default=0,
        help=("Run validation every N training steps. Set <=0 to disable validation."),
    )
    parser.add_argument(
        "--progress",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Enable tqdm progress bar.",
    )
    parser.add_argument(
        "--progress-all",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Show tqdm progress bars for all ranks in DDP mode (default: rank0 only).",
    )
    wandb_group = parser.add_mutually_exclusive_group()
    wandb_group.add_argument(
        "--wandb",
        dest="wandb_enabled",
        action="store_true",
        help="Enable Weights & Biases logging.",
    )
    wandb_group.add_argument(
        "--no-wandb",
        dest="wandb_enabled",
        action="store_false",
        help="Disable Weights & Biases logging.",
    )
    parser.set_defaults(wandb_enabled=None)
    parser.add_argument(
        "--wandb-project",
        default=None,
        help="Weights & Biases project name.",
    )
    parser.add_argument(
        "--wandb-entity",
        default=None,
        help="Weights & Biases entity/team name.",
    )
    parser.add_argument(
        "--wandb-run-name",
        default=None,
        help="Weights & Biases run name.",
    )
    parser.add_argument(
        "--wandb-mode",
        choices=sorted(WANDB_MODES),
        default=None,
        help="Weights & Biases mode.",
    )
    parser.add_argument("--seed", type=int, default=0)
    # ── 拡張オプティマイザ引数 ──────────────────────────────────
    parser.add_argument(
        "--lion-beta1", type=float, default=0.9,
        help="Lion optimizer beta1 (default: 0.9).",
    )
    parser.add_argument(
        "--lion-beta2", type=float, default=0.99,
        help="Lion optimizer beta2 (default: 0.99).",
    )
    parser.add_argument(
        "--ademamix-alpha", type=float, default=5.0,
        help="AdEMAMix alpha (slow EMA weight, default: 5.0).",
    )
    parser.add_argument(
        "--ademamix-beta3", type=float, default=0.9999,
        help="AdEMAMix beta3 (slow EMA decay, default: 0.9999).",
    )
    # ── Attentionバックエンド ────────────────────────────────────
    parser.add_argument(
        "--attention-backend",
        choices=["sdpa", "flash2", "eager"],
        default="sdpa",
        help=(
            "Attention実装を選択。"
            "'sdpa': PyTorch標準(デフォルト), "
            "'flash2': FlashAttention2(要pip install flash-attn), "
            "'eager': フォールバック用mathカーネル。"
        ),
    )
    # ── 勾配チェックポイント ─────────────────────────────────────
    parser.add_argument(
        "--grad-checkpoint",
        action="store_true",
        default=False,
        help="勾配チェックポイントを有効化。VRAMを最大40%削減(速度約20%低下)。",
    )
    # ── EMA ─────────────────────────────────────────────────────
    parser.add_argument(
        "--ema-decay",
        type=float,
        default=None,
        help=(
            "EMA(指数移動平均)のdecay値を指定すると有効化(例: 0.9999)。"
            "チェックポイント保存時にEMAモデルも別途保存される。"
        ),
    )
    # ── 勾配クリッピング ─────────────────────────────────────────
    parser.add_argument(
        "--clip-grad-norm",
        type=float,
        default=1.0,
        help="勾配クリッピングのmax norm値(default: 1.0)。0で無効化。",
    )
    # ── Early Stopping ───────────────────────────────────────────
    parser.add_argument(
        "--early-stopping",
        action="store_true",
        default=False,
        help="Early Stoppingを有効化。valid lossが改善しなくなったら自動停止。",
    )
    parser.add_argument(
        "--early-stopping-patience",
        type=int,
        default=3,
        help=(
            "valid lossの悪化を何回連続で許容するか(default: 3)。"
            "1回 = valid_every step分の猶予。"
            "例: patience=3, valid_every=100 なら300step悪化継続で停止。"
        ),
    )
    parser.add_argument(
        "--early-stopping-min-delta",
        type=float,
        default=0.01,
        help=(
            "この値以上悪化した場合のみカウント(default: 0.01)。"
            "微小な揺れによる誤検知を防ぐ。"
        ),
    )
    # ── チェックポイント保存方式 ─────────────────────────────────
    parser.add_argument(
        "--save-full",
        dest="save_full",
        action="store_true",
        default=False,
        help=(
            "EMA版に加えてフルサイズ版(_full.pt)も保存する。"
            "追加学習を行う場合に指定。デフォルトはEMA版のみ保存。"
        ),
    )
    ddp_group = parser.add_mutually_exclusive_group()
    ddp_group.add_argument(
        "--ddp-find-unused-parameters",
        dest="ddp_find_unused_parameters",
        action="store_true",
        help=(
            "Enable DDP find_unused_parameters. Useful when conditional branches "
            "(e.g., speaker/text conditioning) may be fully masked in some steps."
        ),
    )
    ddp_group.add_argument(
        "--no-ddp-find-unused-parameters",
        dest="ddp_find_unused_parameters",
        action="store_false",
        help="Disable DDP find_unused_parameters.",
    )
    parser.set_defaults(ddp_find_unused_parameters=None)
    args = parser.parse_args()
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
    if train_cfg.output_dir == default_train_cfg.output_dir and not cli_provided(
        raw_argv, "--output-dir"
    ):
        train_cfg = replace(train_cfg, output_dir=args.output_dir)

    if cli_provided(raw_argv, "--output-dir"):
        train_cfg = replace(train_cfg, output_dir=args.output_dir)
    if cli_provided(raw_argv, "--precision"):
        train_cfg = replace(train_cfg, precision=args.precision)
    if args.allow_tf32 is not None:
        train_cfg = replace(train_cfg, allow_tf32=args.allow_tf32)
    if args.compile_model is not None:
        train_cfg = replace(train_cfg, compile_model=args.compile_model)
    if cli_provided(raw_argv, "--batch-size"):
        train_cfg = replace(train_cfg, batch_size=args.batch_size)
    if cli_provided(raw_argv, "--gradient-accumulation-steps"):
        train_cfg = replace(
            train_cfg,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
        )
    if cli_provided(raw_argv, "--max-text-len"):
        train_cfg = replace(train_cfg, max_text_len=args.max_text_len)
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
    if cli_provided(raw_argv, "--stable-steps"):
        train_cfg = replace(train_cfg, stable_steps=args.stable_steps)
    if cli_provided(raw_argv, "--min-lr-scale"):
        train_cfg = replace(train_cfg, min_lr_scale=args.min_lr_scale)
    if cli_provided(raw_argv, "--max-steps"):
        train_cfg = replace(train_cfg, max_steps=args.max_steps)
    if cli_provided(raw_argv, "--text-condition-dropout"):
        train_cfg = replace(train_cfg, text_condition_dropout=args.text_condition_dropout)
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
        train_cfg = replace(
            train_cfg,
            ddp_find_unused_parameters=args.ddp_find_unused_parameters,
        )
    if cli_provided(raw_argv, "--seed"):
        train_cfg = replace(train_cfg, seed=args.seed)

    # 拡張引数はtrain_cfgに存在しないためargs直接参照
    attention_backend: str = args.attention_backend
    use_grad_checkpoint: bool = args.grad_checkpoint
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

    # --resume 未指定時はデフォルトの safetensors を自動参照（v2優先）
    _DEFAULT_PRETRAINED_SAFETENSORS_CANDIDATES = [
        _PROJECT_CHECKPOINTS_DIR / "Aratako_Irodori-TTS-500M-v2" / "model.safetensors",
        _PROJECT_CHECKPOINTS_DIR / "Aratako_Irodori-TTS-500M" / "model.safetensors",
    ]
    if args.resume is None:
        _auto_resume = next((p for p in _DEFAULT_PRETRAINED_SAFETENSORS_CANDIDATES if p.exists()), None)
        if _auto_resume is not None:
            args.resume = str(_auto_resume)
            if is_main_process:
                print(f"--resume not specified. Auto-loading: {args.resume}")

    # safetensors を resume する場合、必要に応じてモデル設定をチェックポイントに同期
    if args.resume is not None and Path(args.resume).suffix.lower() == ".safetensors":
        from safetensors import safe_open

        resume_path = Path(args.resume).expanduser()
        checkpoint_model_cfg: dict | None = None
        with safe_open(str(resume_path), framework="pt", device="cpu") as handle:
            metadata = dict(handle.metadata() or {})
        config_json = metadata.get(SAFETENSORS_CONFIG_META_KEY)
        if config_json:
            parsed = json.loads(config_json)
            if isinstance(parsed, dict):
                checkpoint_model_cfg = parsed

        if checkpoint_model_cfg is not None:
            latent_dim_overridden = cli_provided(raw_argv, "--latent-dim")
            latent_patch_overridden = cli_provided(raw_argv, "--latent-patch-size")
            sync_kwargs: dict[str, int] = {}
            for key in ("latent_dim", "latent_patch_size", "text_vocab_size", "text_dim", "model_dim"):
                if key == "latent_dim" and latent_dim_overridden:
                    continue
                if key == "latent_patch_size" and latent_patch_overridden:
                    continue
                value = checkpoint_model_cfg.get(key)
                if value is not None:
                    sync_kwargs[key] = int(value)
            if sync_kwargs:
                model_cfg = replace(model_cfg, **sync_kwargs)
                if is_main_process:
                    print(
                        f"Synchronized model config from resume safetensors metadata: {resume_path}"
                    )

    set_seed(train_cfg.seed + rank)
    if not (0.0 <= train_cfg.text_condition_dropout <= 1.0):
        raise ValueError(
            f"text_condition_dropout must be in [0, 1], got {train_cfg.text_condition_dropout}"
        )
    if train_cfg.max_text_len <= 0:
        raise ValueError(f"max_text_len must be > 0, got {train_cfg.max_text_len}")
    if train_cfg.gradient_accumulation_steps <= 0:
        raise ValueError(
            f"gradient_accumulation_steps must be > 0, got {train_cfg.gradient_accumulation_steps}"
        )
    if not (0.0 <= train_cfg.speaker_condition_dropout <= 1.0):
        raise ValueError(
            "speaker_condition_dropout must be in [0, 1], "
            f"got {train_cfg.speaker_condition_dropout}"
        )
    if train_cfg.fixed_target_latent_steps is not None and train_cfg.fixed_target_latent_steps <= 0:
        raise ValueError(
            "fixed_target_latent_steps must be > 0 when provided, "
            f"got {train_cfg.fixed_target_latent_steps}"
        )
    if train_cfg.fixed_target_full_mask and train_cfg.fixed_target_latent_steps is None:
        raise ValueError(
            "fixed_target_full_mask=True requires fixed_target_latent_steps to be set."
        )
    if train_cfg.dataloader_prefetch_factor <= 0:
        raise ValueError(
            f"dataloader_prefetch_factor must be > 0, got {train_cfg.dataloader_prefetch_factor}"
        )
    if not (0.0 <= train_cfg.valid_ratio < 1.0):
        raise ValueError(f"valid_ratio must be in [0, 1), got {train_cfg.valid_ratio}")
    if train_cfg.valid_every < 0:
        raise ValueError(f"valid_every must be >= 0, got {train_cfg.valid_every}")
    if train_cfg.valid_ratio > 0.0 and train_cfg.valid_every <= 0:
        raise ValueError("valid_every must be > 0 when valid_ratio > 0.")
    if train_cfg.valid_ratio == 0.0 and train_cfg.valid_every > 0 and is_main_process:
        print("warning: valid_every is set but valid_ratio=0. Validation is disabled.")
    if train_cfg.checkpoint_best_n < 0:
        raise ValueError(f"checkpoint_best_n must be >= 0, got {train_cfg.checkpoint_best_n}")
    if train_cfg.wandb_mode not in WANDB_MODES:
        raise ValueError(
            f"wandb_mode must be one of {sorted(WANDB_MODES)}, got {train_cfg.wandb_mode!r}"
        )
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
    if train_cfg.wandb_enabled and is_main_process:
        try:
            import wandb
        except ImportError as exc:
            raise RuntimeError(
                "W&B logging is enabled, but `wandb` is not installed. "
                "Install it with `pip install wandb`."
            ) from exc
        wandb_run = wandb.init(
            project=train_cfg.wandb_project,
            entity=train_cfg.wandb_entity,
            name=train_cfg.wandb_run_name,
            mode=train_cfg.wandb_mode,
            dir=str(output_dir),
            config={
                "model": asdict(model_cfg),
                "train": asdict(train_cfg),
                "script": "train.py",
            },
        )
        print(
            f"W&B enabled: project={train_cfg.wandb_project} mode={train_cfg.wandb_mode} run={wandb_run.name if wandb_run is not None else train_cfg.wandb_run_name}"
        )

    if distributed:
        local_files_only = not is_main_process
        if is_main_process:
            tokenizer = build_text_tokenizer(model_cfg, local_files_only=False)
            text_hidden_size = validate_text_backbone_dim(model_cfg, local_files_only=False)
        dist.barrier()
        if not is_main_process:
            tokenizer = build_text_tokenizer(model_cfg, local_files_only=local_files_only)
            text_hidden_size = validate_text_backbone_dim(
                model_cfg,
                local_files_only=local_files_only,
            )
        dist.barrier()
    else:
        tokenizer = build_text_tokenizer(model_cfg, local_files_only=False)
        text_hidden_size = validate_text_backbone_dim(model_cfg, local_files_only=False)
    if is_main_process:
        print(
            f"Text tokenizer={model_cfg.text_tokenizer_repo} vocab={tokenizer.vocab_size} add_bos={model_cfg.text_add_bos} padding_side=right "
            f"(pretrained hidden_size={text_hidden_size})."
        )
    full_dataset = LatentTextDataset(
        manifest_path=train_cfg.manifest_path,
        latent_dim=model_cfg.latent_dim,
        max_latent_steps=train_cfg.max_latent_steps,
    )
    train_dataset = full_dataset
    valid_dataset = None
    if train_cfg.valid_ratio > 0.0:
        train_indices, valid_indices = split_train_valid_indices(
            num_samples=len(full_dataset),
            valid_ratio=train_cfg.valid_ratio,
            seed=train_cfg.seed,
        )
        train_dataset = LatentTextDataset(
            manifest_path=train_cfg.manifest_path,
            latent_dim=model_cfg.latent_dim,
            max_latent_steps=train_cfg.max_latent_steps,
            subset_indices=train_indices,
        )
        valid_dataset = LatentTextDataset(
            manifest_path=train_cfg.manifest_path,
            latent_dim=model_cfg.latent_dim,
            max_latent_steps=train_cfg.max_latent_steps,
            subset_indices=valid_indices,
        )
        if is_main_process:
            print(
                f"Validation split enabled: train={len(train_dataset)} valid={len(valid_dataset)} (ratio={train_cfg.valid_ratio:.4f}, valid_every={train_cfg.valid_every} steps)."
            )
    drop_last = len(train_dataset) >= train_cfg.batch_size
    if not drop_last and is_main_process:
        print(
            f"warning: dataset size ({len(train_dataset)}) is smaller than batch_size ({train_cfg.batch_size}). "
            "Using drop_last=False to avoid empty dataloader."
        )
    collator = TTSCollator(
        tokenizer=tokenizer,
        caption_tokenizer=None,
        latent_dim=model_cfg.latent_dim,
        latent_patch_size=model_cfg.latent_patch_size,
        fixed_target_latent_steps=train_cfg.fixed_target_latent_steps,
        fixed_target_full_mask=train_cfg.fixed_target_full_mask,
        max_text_len=train_cfg.max_text_len,
    )
    if train_cfg.fixed_target_latent_steps is not None and is_main_process:
        print(
            f"Fixed target latent length enabled: steps={train_cfg.fixed_target_latent_steps} full_mask={train_cfg.fixed_target_full_mask}"
        )
    if train_cfg.timestep_stratified and is_main_process:
        print("Using stratified logit-normal timestep sampling.")
    train_sampler = None
    if distributed:
        train_sampler = DistributedSampler(
            train_dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=True,
            drop_last=drop_last,
        )
    dataloader_common_kwargs = {
        "batch_size": train_cfg.batch_size,
        "num_workers": train_cfg.num_workers,
        "pin_memory": (device.type == "cuda"),
        "collate_fn": collator,
    }
    if train_cfg.num_workers > 0:
        dataloader_common_kwargs["persistent_workers"] = bool(
            train_cfg.dataloader_persistent_workers
        )
        dataloader_common_kwargs["prefetch_factor"] = int(train_cfg.dataloader_prefetch_factor)
    elif train_cfg.dataloader_persistent_workers and is_main_process:
        print("warning: dataloader_persistent_workers=True is ignored because num_workers=0.")
    loader = DataLoader(
        dataset=train_dataset,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        drop_last=drop_last,
        **dataloader_common_kwargs,
    )
    if len(loader) == 0:
        raise ValueError("Dataloader yielded zero batches. Check manifest and batch_size settings.")
    valid_loader = None
    valid_sampler = None
    if valid_dataset is not None:
        if distributed:
            valid_sampler = DistributedSampler(
                valid_dataset,
                num_replicas=world_size,
                rank=rank,
                shuffle=False,
                drop_last=False,
            )
        valid_loader = DataLoader(
            dataset=valid_dataset,
            shuffle=False,
            sampler=valid_sampler,
            drop_last=False,
            **dataloader_common_kwargs,
        )
        if len(valid_loader) == 0:
            raise ValueError(
                "Validation dataloader yielded zero batches. Decrease batch_size or valid_ratio."
            )

    has_validation = valid_loader is not None and train_cfg.valid_every > 0
    checkpoint_retention_enabled = train_cfg.checkpoint_best_n > 0
    periodic_checkpoint_keep = 0
    if checkpoint_retention_enabled:
        periodic_checkpoint_keep = 1 if has_validation else int(train_cfg.checkpoint_best_n) + 1
    best_val_checkpoints: list[tuple[float, int, Path]] = []
    if is_main_process:
        if checkpoint_retention_enabled and has_validation:
            best_val_checkpoints = list_best_val_loss_checkpoints(output_dir)
            best_val_checkpoints = prune_best_val_loss_checkpoints(
                best_val_checkpoints,
                train_cfg.checkpoint_best_n,
            )
        if checkpoint_retention_enabled and has_validation:
            print(f"Checkpoint retention: latest=1 + best_val_loss={train_cfg.checkpoint_best_n}.")
        elif checkpoint_retention_enabled:
            print(
                f"Checkpoint retention: validation disabled, keep latest {periodic_checkpoint_keep} periodic checkpoints."
            )

    raw_model = TextToLatentRFDiT(model_cfg).to(device)
    # デフォルト safetensors が存在する場合も resume 扱いとして扱い、
    # テキスト埋め込みの個別初期化はスキップする
    _DEFAULT_PRETRAINED_SAFETENSORS_CANDIDATES = [
        _PROJECT_CHECKPOINTS_DIR / "Aratako_Irodori-TTS-500M-v2" / "model.safetensors",
        _PROJECT_CHECKPOINTS_DIR / "Aratako_Irodori-TTS-500M" / "model.safetensors",
    ]
    _default_safetensors_exists = any(p.exists() for p in _DEFAULT_PRETRAINED_SAFETENSORS_CANDIDATES)
    if args.resume is None and args.init_checkpoint is None and not _default_safetensors_exists:
        if distributed:
            if is_main_process:
                print(
                    f"Initializing text embedding from pretrained model: {model_cfg.text_tokenizer_repo}"
                )
                initialize_text_embedding_from_pretrained(
                    raw_model,
                    model_cfg,
                    local_files_only=False,
                )
            dist.barrier()
            if not is_main_process:
                initialize_text_embedding_from_pretrained(
                    raw_model,
                    model_cfg,
                    local_files_only=True,
                )
            dist.barrier()
        else:
            if is_main_process:
                print(
                    f"Initializing text embedding from pretrained model: {model_cfg.text_tokenizer_repo}"
                )
            initialize_text_embedding_from_pretrained(
                raw_model,
                model_cfg,
                local_files_only=False,
            )
    train_model = raw_model
    if train_cfg.compile_model:
        if not hasattr(torch, "compile"):
            raise RuntimeError("compile_model=True requires torch.compile (PyTorch 2+).")
        if is_main_process:
            print("torch.compile enabled (dynamic=True).")
        train_model = torch.compile(raw_model, dynamic=True)

    # ── Attentionバックエンド設定 ────────────────────────────────
    if device.type == "cuda":
        apply_attention_backend(raw_model, attention_backend)

    # ── 勾配チェックポイント ─────────────────────────────────────
    if use_grad_checkpoint:
        apply_gradient_checkpointing(raw_model)

    # ── EMA初期化 ────────────────────────────────────────────────
    ema_model: EMAModel | None = None
    if ema_decay is not None:
        if not (0.0 < ema_decay < 1.0):
            raise ValueError(f"--ema-decay must be in (0, 1), got {ema_decay}")
        ema_model = EMAModel(raw_model, decay=ema_decay)
        if is_main_process:
            print(f"EMA enabled (decay={ema_decay})")

    # ── Early Stopping初期化 ─────────────────────────────────────
    early_stopper: EarlyStopping | None = None
    if use_early_stopping:
        if train_cfg.valid_ratio <= 0.0:
            raise ValueError(
                "--early-stopping requires --valid-ratio > 0. "
                "valid lossを監視するためバリデーションデータが必要です。"
            )
        early_stopper = EarlyStopping(
            patience=es_patience,
            min_delta=es_min_delta,
            mode="min",
        )
        if is_main_process:
            print(
                f"Early Stopping: enabled "
                f"(patience={es_patience}, min_delta={es_min_delta})"
            )
    ddp_find_unused_parameters = bool(train_cfg.ddp_find_unused_parameters)
    ddp_find_unused_parameters_explicit = args.ddp_find_unused_parameters is not None or (
        isinstance(exp_cfg.get("train"), dict)
        and "ddp_find_unused_parameters" in exp_cfg.get("train", {})
    )
    if distributed:
        # Auto-enable for common configs where conditional branches can be fully
        # masked in a step. Without this, DDP can hang after step 1 due to
        # unreduced gradients in ranks where a branch is entirely unused.
        if not ddp_find_unused_parameters and not ddp_find_unused_parameters_explicit:
            speaker_labeled_count = sum(
                1 for x in train_dataset.samples if x.get("speaker_id") is not None
            )
            has_partial_or_no_speaker_labels = speaker_labeled_count < len(train_dataset)
            has_stochastic_cond_drop = (
                train_cfg.text_condition_dropout > 0.0 or train_cfg.speaker_condition_dropout > 0.0
            )
            if has_partial_or_no_speaker_labels or has_stochastic_cond_drop:
                ddp_find_unused_parameters = True
                if is_main_process:
                    print(
                        "DDP find_unused_parameters auto-enabled "
                        "(conditional branches may be fully masked in some steps)."
                    )
        model = DDP(
            train_model,
            device_ids=[local_rank],
            output_device=local_rank,
            find_unused_parameters=ddp_find_unused_parameters,
            broadcast_buffers=False,
        )
    else:
        model = train_model
    optimizer = build_optimizer_extended(
        raw_model,
        train_cfg,
        optimizer_name=train_cfg.optimizer,
        lion_betas=lion_betas,
        ademamix_alpha=ademamix_alpha,
        ademamix_beta3=ademamix_beta3,
    )
    scheduler = build_scheduler(optimizer, train_cfg)
    if is_main_process:
        print(
            f"Optimizer={train_cfg.optimizer} Scheduler={train_cfg.lr_scheduler} lr={current_lr(optimizer):.3e}"
        )
        if train_cfg.gradient_accumulation_steps > 1:
            print(
                f"Gradient accumulation enabled: steps={train_cfg.gradient_accumulation_steps} (effective global batch={train_cfg.batch_size * world_size * train_cfg.gradient_accumulation_steps})."
            )

    # --init-checkpoint: モデル重みのみ読み込み、optimizer/schedulerは新規
    if args.init_checkpoint is not None:
        init_checkpoint_path = Path(args.init_checkpoint).expanduser()
        init_state, init_model_cfg = _load_model_state_from_checkpoint(init_checkpoint_path)
        _check_model_config_compatibility(init_checkpoint_path, init_model_cfg, model_cfg)
        raw_model.load_state_dict(init_state)
        if is_main_process:
            print(f"Initialized model weights from: {init_checkpoint_path}")

    # --resume 未指定時はデフォルトの safetensors を自動参照
    _DEFAULT_PRETRAINED_SAFETENSORS = next(
        (p for p in _DEFAULT_PRETRAINED_SAFETENSORS_CANDIDATES if p.exists()),
        None,
    )
    if args.resume is None and _DEFAULT_PRETRAINED_SAFETENSORS is not None:
        args.resume = str(_DEFAULT_PRETRAINED_SAFETENSORS)
        if is_main_process:
            print(f"--resume not specified. Auto-loading: {args.resume}")

    step = 0
    progress: TrainProgress | None = None
    if args.resume is not None:
        if args.resume.endswith(".safetensors"):
            # Aratako_Irodori-TTS-500M 等の HuggingFace safetensors から追加学習
            from safetensors.torch import load_file
            weights = load_file(args.resume, device=str(device))
            missing, unexpected = raw_model.load_state_dict(weights, strict=False)
            if is_main_process:
                print(f"Loaded pretrained weights from {args.resume}")
                if missing:
                    print(f"  Missing keys: {len(missing)}")
                if unexpected:
                    print(f"  Unexpected keys: {len(unexpected)}")
            step = 0  # 追加学習はstep=0からカウント
        else:
            ckpt = torch.load(args.resume, map_location=device, weights_only=True)
            raw_model.load_state_dict(ckpt["model"])
            optimizer.load_state_dict(ckpt["optimizer"])
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
                    print(f"EMA state restored from checkpoint.")
            if is_main_process:
                print(f"Resumed from step={step}")

    progress = TrainProgress(
        max_steps=train_cfg.max_steps,
        start_step=step,
        rank=rank,
        world_size=world_size,
        enabled=train_cfg.progress,
        show_all_ranks=train_cfg.progress_all_ranks,
        description="Train RF",
    )
    accum_steps = int(train_cfg.gradient_accumulation_steps)
    global_batch_size = train_cfg.batch_size * world_size * accum_steps

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
        if scheduler is not None and step == 0:
            # Ensure the very first optimizer step uses warmup-scaled LR.
            scheduler.step()
        optimizer.zero_grad(set_to_none=True)
        accum_micro_steps = 0
        accum_loss = torch.zeros((), device=device, dtype=torch.float32)
        accum_rf_loss = torch.zeros((), device=device, dtype=torch.float32)
        epoch = 0
        while step < train_cfg.max_steps and not (
            early_stopper is not None and early_stopper.should_stop
        ):
            if train_sampler is not None:
                train_sampler.set_epoch(epoch)
            epoch += 1
            for epoch_step, batch in enumerate(loader, start=1):
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
                if train_cfg.timestep_stratified:
                    t = sample_stratified_logit_normal_t(
                        batch_size=bsz,
                        device=device,
                        mean=train_cfg.timestep_logit_mean,
                        std=train_cfg.timestep_logit_std,
                        t_min=train_cfg.timestep_min,
                        t_max=train_cfg.timestep_max,
                    )
                else:
                    t = sample_logit_normal_t(
                        batch_size=bsz,
                        device=device,
                        mean=train_cfg.timestep_logit_mean,
                        std=train_cfg.timestep_logit_std,
                        t_min=train_cfg.timestep_min,
                        t_max=train_cfg.timestep_max,
                    )
                noise = torch.randn_like(x0)
                x_t = rf_interpolate(x0, noise, t)
                v_target = rf_velocity_target(x0, noise)

                text_cond_drop = torch.rand(bsz, device=device) < train_cfg.text_condition_dropout
                if text_cond_drop.any():
                    text_mask = text_mask.clone()
                    text_mask[text_cond_drop] = False

                speaker_cond_drop = (
                    torch.rand(bsz, device=device) < train_cfg.speaker_condition_dropout
                )
                use_speaker = has_speaker & (~speaker_cond_drop)
                ref_mask = ref_mask & use_speaker[:, None]
                ref_latent = ref_latent * use_speaker[:, None, None].to(ref_latent.dtype)

                should_step = (accum_micro_steps % accum_steps) == 0
                sync_context = model.no_sync() if distributed and not should_step else nullcontext()
                with sync_context:
                    with (
                        torch.autocast(device_type="cuda", dtype=torch.bfloat16)
                        if use_bf16
                        else nullcontext()
                    ):
                        v_pred = model(
                            x_t=x_t,
                            t=t,
                            text_input_ids=text_ids,
                            text_mask=text_mask,
                            ref_latent=ref_latent,
                            ref_mask=ref_mask,
                            latent_mask=x_mask,
                        )

                    v_pred = v_pred.float()
                    rf_loss = echo_style_masked_mse(
                        v_pred,
                        v_target.float(),
                        loss_mask=x_mask,
                        valid_mask=x_mask_valid,
                    )
                    loss = rf_loss
                    (loss / float(accum_steps)).backward()

                accum_loss += loss.detach()
                accum_rf_loss += rf_loss.detach()
                if not should_step:
                    continue

                step_loss = accum_loss / float(accum_steps)
                step_rf_loss = accum_rf_loss / float(accum_steps)
                accum_loss.zero_()
                accum_rf_loss.zero_()

                torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad_norm if clip_grad_norm > 0 else float("inf"))
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                if scheduler is not None:
                    scheduler.step()
                # EMA更新
                if ema_model is not None:
                    ema_model.update(raw_model)
                step += 1
                _step_times.append(_time.monotonic())
                if len(_step_times) > _ETA_WINDOW + 1:
                    _step_times.pop(0)
                progress.update(step)

                # 最初の5ステップは強制ログ出力（早期ETA概算のため）
                if step <= 5 or step % train_cfg.log_every == 0:
                    loss_value = reduce_mean(step_loss, world_size, distributed).item()
                    rf_loss_value = reduce_mean(step_rf_loss, world_size, distributed).item()
                    lr_value = current_lr(optimizer)
                    progress_metrics: dict[str, float] = {
                        "loss": loss_value,
                        "rf": rf_loss_value,
                        "lr": lr_value,
                    }
                    progress.log(
                        step=step,
                        epoch=epoch,
                        epoch_step=epoch_step,
                        epoch_total=len(loader),
                        metrics=progress_metrics,
                        global_batch_size=global_batch_size,
                    )
                    if is_main_process:
                        # ETA計算
                        if len(_step_times) >= 2:
                            _elapsed = _step_times[-1] - _step_times[0]
                            _sps = _elapsed / (len(_step_times) - 1)
                            _remaining = max(0, train_cfg.max_steps - step)
                            _eta_str = _fmt_eta(_sps * _remaining)
                            _speed_str = f"{1.0/_sps:.3f}steps/s" if _sps > 0 else "?"
                            progress.write(
                                f"step={step} loss={loss_value:.6f} rf={rf_loss_value:.6f} lr={lr_value:.3e}"
                                f" speed={_speed_str} eta={_eta_str}"
                            )
                        else:
                            progress.write(
                                f"step={step} loss={loss_value:.6f} rf={rf_loss_value:.6f} lr={lr_value:.3e}"
                            )
                        if wandb_run is not None:
                            metrics = {
                                "train/loss": loss_value,
                                "train/rf_loss": rf_loss_value,
                                "train/lr": lr_value,
                            }
                            wandb_run.log(metrics, step=step)

                if step % train_cfg.save_every == 0 and is_main_process:
                    save_checkpoint(
                        output_dir / f"checkpoint_{step:07d}.pt",
                        raw_model,
                        optimizer,
                        scheduler,
                        step,
                        model_cfg,
                        train_cfg,
                        ema_model=ema_model,
                        save_full=save_full,
                    )
                    enforce_periodic_checkpoint_limit(
                        output_dir=output_dir,
                        keep_count=periodic_checkpoint_keep,
                    )

                if (
                    valid_loader is not None
                    and train_cfg.valid_every > 0
                    and step % train_cfg.valid_every == 0
                ):
                    valid_metrics = run_validation(
                        model=model,
                        loader=valid_loader,
                        train_cfg=train_cfg,
                        device=device,
                        use_bf16=use_bf16,
                        distributed=distributed,
                    )
                    if is_main_process:
                        progress.write(
                            ("valid step={} loss={:.6f} rf={:.6f} (samples={:.0f})").format(
                                step,
                                valid_metrics["loss"],
                                valid_metrics["rf_loss"],
                                valid_metrics["num_samples"],
                            )
                        )
                        if wandb_run is not None:
                            wandb_run.log(
                                {
                                    "valid/loss": valid_metrics["loss"],
                                    "valid/rf_loss": valid_metrics["rf_loss"],
                                },
                                step=step,
                            )
                        best_val_checkpoints, best_path = maybe_save_best_val_loss_checkpoint(
                            output_dir=output_dir,
                            checkpoints=best_val_checkpoints,
                            keep_best_n=train_cfg.checkpoint_best_n,
                            val_loss=float(valid_metrics["loss"]),
                            step=step,
                            model=raw_model,
                            optimizer=optimizer,
                            scheduler=scheduler,
                            model_cfg=model_cfg,
                            train_cfg=train_cfg,
                            ema_model=ema_model,
                            save_full=save_full,
                        )
                        if best_path is not None:
                            progress.write(
                                "saved best val checkpoint: {} (loss={:.6f})".format(
                                    best_path.name,
                                    float(valid_metrics["loss"]),
                                )
                            )

                        # ── Early Stopping判定 ──────────────────
                        if early_stopper is not None:
                            should_stop_early = early_stopper.step(
                                score=float(valid_metrics["loss"]),
                                current_step=step,
                            )
                            progress.write(early_stopper.status())
                            if should_stop_early:
                                progress.write(
                                    f"Early Stopping: valid lossが{es_patience}回連続で"
                                    f"改善しませんでした。"
                                    f"最良はstep={early_stopper.best_step} "
                                    f"(valid loss={early_stopper.best_score:.6f})。"
                                    f"学習を停止します。"
                                )
                                # 停止直前に現在のチェックポイントを保存
                                save_checkpoint(
                                    output_dir / f"checkpoint_early_stop_{step:07d}.pt",
                                    raw_model,
                                    optimizer,
                                    scheduler,
                                    step,
                                    model_cfg,
                                    train_cfg,
                                    ema_model=ema_model,
                                    save_full=save_full,
                                )

                if step >= train_cfg.max_steps:
                    break
                if early_stopper is not None and early_stopper.should_stop:
                    break

        if (
            valid_loader is not None
            and train_cfg.valid_every > 0
            and step % train_cfg.valid_every != 0
        ):
            valid_metrics = run_validation(
                model=model,
                loader=valid_loader,
                train_cfg=train_cfg,
                device=device,
                use_bf16=use_bf16,
                distributed=distributed,
            )
            if is_main_process:
                progress.write(
                    ("valid final step={} loss={:.6f} rf={:.6f} (samples={:.0f})").format(
                        step,
                        valid_metrics["loss"],
                        valid_metrics["rf_loss"],
                        valid_metrics["num_samples"],
                    )
                )
                if wandb_run is not None:
                    wandb_run.log(
                        {
                            "valid/loss": valid_metrics["loss"],
                            "valid/rf_loss": valid_metrics["rf_loss"],
                        },
                        step=step,
                    )
                best_val_checkpoints, best_path = maybe_save_best_val_loss_checkpoint(
                    output_dir=output_dir,
                    checkpoints=best_val_checkpoints,
                    keep_best_n=train_cfg.checkpoint_best_n,
                    val_loss=float(valid_metrics["loss"]),
                    step=step,
                    model=raw_model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    model_cfg=model_cfg,
                    train_cfg=train_cfg,
                )
                if best_path is not None:
                    progress.write(
                        "saved best val checkpoint: {} (loss={:.6f})".format(
                            best_path.name,
                            float(valid_metrics["loss"]),
                        )
                    )

        if is_main_process:
            save_checkpoint(
                output_dir / "checkpoint_final.pt",
                raw_model,
                optimizer,
                scheduler,
                step,
                model_cfg,
                train_cfg,
                ema_model=ema_model,
                save_full=save_full,
            )
            if wandb_run is not None:
                wandb_run.summary["train/final_step"] = step
            progress.write(f"Training finished at step={step}.")
    finally:
        if progress is not None:
            progress.close()
        if wandb_run is not None:
            wandb_run.finish()

        # DataLoaderワーカーを先にシャットダウン（SIGINT中断時のpickleエラー抑制）
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

        # VRAMアンロード（main processのみ）
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
