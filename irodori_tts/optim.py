from __future__ import annotations

import math

import torch

from .config import TrainConfig


class OptimizerChain:
    """複数のオプティマイザを1つにまとめて学習ループから扱えるようにする。

    param_groups は main のグループを先に並べる（current_lr が param_groups[0] を参照するため）。
    """

    def __init__(self, optimizers: list):
        if len(optimizers) < 2:
            raise ValueError("OptimizerChain requires at least two optimizers.")
        self.optimizers = list(optimizers)
        self.param_groups = []
        for opt in self.optimizers:
            self.param_groups.extend(opt.param_groups)

    def zero_grad(self, set_to_none: bool = True) -> None:
        for opt in self.optimizers:
            opt.zero_grad(set_to_none=set_to_none)

    def step(self) -> None:
        for opt in self.optimizers:
            opt.step()

    def state_dict(self) -> dict:
        return {"parts": [opt.state_dict() for opt in self.optimizers]}

    def load_state_dict(self, state_dict: dict) -> None:
        parts = state_dict.get("parts")
        if not isinstance(parts, list) or len(parts) != len(self.optimizers):
            raise ValueError(
                "OptimizerChain state_dict must contain 'parts' matching the chained optimizers."
            )
        for opt, part in zip(self.optimizers, parts, strict=True):
            opt.load_state_dict(part)


class MuonWithAuxAdamW:
    """
    Simple single-process wrapper:
    - torch.optim.Muon for Muon-compatible parameters
    - AdamW for auxiliary parameters (embeddings, biases, output heads, etc.)
    """

    def __init__(self, muon_opt: torch.optim.Optimizer, aux_opt: torch.optim.Optimizer | None):
        self.muon_opt = muon_opt
        self.aux_opt = aux_opt
        self.param_groups = list(muon_opt.param_groups)
        if aux_opt is not None:
            self.param_groups.extend(aux_opt.param_groups)

    def zero_grad(self, set_to_none: bool = True) -> None:
        self.muon_opt.zero_grad(set_to_none=set_to_none)
        if self.aux_opt is not None:
            self.aux_opt.zero_grad(set_to_none=set_to_none)

    def step(self) -> None:
        self.muon_opt.step()
        if self.aux_opt is not None:
            self.aux_opt.step()

    def state_dict(self) -> dict:
        return {
            "muon": self.muon_opt.state_dict(),
            "aux": None if self.aux_opt is None else self.aux_opt.state_dict(),
        }

    def load_state_dict(self, state_dict: dict) -> None:
        if "muon" not in state_dict:
            raise ValueError(
                "MuonWithAuxAdamW state_dict must contain 'muon' key (and optional 'aux' key)."
            )

        self.muon_opt.load_state_dict(state_dict["muon"])
        if self.aux_opt is not None and state_dict.get("aux") is not None:
            self.aux_opt.load_state_dict(state_dict["aux"])


class ScalarLRScheduler:
    """
    Scheduler that applies a scalar multiplier to all optimizer param-group LRs.
    Works with both torch optimizers and MuonWithAuxAdamW wrapper.
    """

    def __init__(self, optimizer, lr_lambda):
        self.optimizer = optimizer
        self.lr_lambda = lr_lambda
        self.base_lrs = [float(group["lr"]) for group in optimizer.param_groups]
        self.last_step = -1

    def step(self) -> None:
        self.last_step += 1
        scale = float(self.lr_lambda(self.last_step))
        for base_lr, group in zip(self.base_lrs, self.optimizer.param_groups, strict=False):
            group["lr"] = base_lr * scale

    def state_dict(self) -> dict:
        return {
            "base_lrs": list(self.base_lrs),
            "last_step": int(self.last_step),
        }

    def load_state_dict(self, state_dict: dict) -> None:
        if "base_lrs" in state_dict:
            loaded_base_lrs = [float(x) for x in state_dict["base_lrs"]]
            if len(loaded_base_lrs) == len(self.optimizer.param_groups):
                self.base_lrs = loaded_base_lrs
        if "last_step" in state_dict:
            self.last_step = int(state_dict["last_step"])


def _use_weight_decay(name: str, p: torch.nn.Parameter) -> bool:
    """
    Approximate Echo/JAX no-decay mask semantics for PyTorch modules.

    Flax side no-decay keys include:
      bias, scale, weight, gate, shift, freqs, phases, out_proj,
      adaln_rank_shift, adaln_rank_scale, adaln_rank_gate.

    In PyTorch:
    - keep all `.bias` no-decay,
    - map Flax `weight` mostly to normalization weights (`*norm*.weight`),
    - apply token-based exclusions for AdaLN / out-proj / freq-phase terms.
    """
    lname = name.lower()

    if lname.endswith(".bias"):
        return False

    # Flax `weight` token mostly corresponds to norm weights in this model family.
    if lname.endswith(".weight") and "norm" in lname:
        return False

    # Echo Flax wd_mask matches exact keys. In this PyTorch model the closest
    # equivalent is to restrict shift/scale/gate exclusions to AdaLN modules.
    if (".attention_adaln." in lname or ".mlp_adaln." in lname) and any(
        token in lname for token in ("shift", "scale", "gate", "adaln_rank_")
    ):
        return False

    if "out_proj" in lname:
        return False

    if "freqs" in lname or "phases" in lname:
        return False

    return True


def _partition_adamw_params(
    model: torch.nn.Module,
) -> tuple[list[torch.nn.Parameter], list[torch.nn.Parameter]]:
    decay: list[torch.nn.Parameter] = []
    no_decay: list[torch.nn.Parameter] = []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if _use_weight_decay(name, p):
            decay.append(p)
        else:
            no_decay.append(p)
    return decay, no_decay


def _partition_muon_params(
    model: torch.nn.Module,
    exclude_ids: set[int] | None = None,
) -> tuple[
    list[torch.nn.Parameter],
    list[torch.nn.Parameter],
    list[torch.nn.Parameter],
    list[torch.nn.Parameter],
]:
    """
    Split parameters into Muon-compatible tensors and aux-Adam tensors,
    each with decay/no-decay partitions.
    """
    muon_decay: list[torch.nn.Parameter] = []
    muon_no_decay: list[torch.nn.Parameter] = []
    aux_decay: list[torch.nn.Parameter] = []
    aux_no_decay: list[torch.nn.Parameter] = []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if exclude_ids is not None and id(p) in exclude_ids:
            continue
        # Muon is intended for hidden matrix-like weights.
        # Keep embeddings/output heads/bias-like params on Adam.
        is_muon_candidate = (
            p.ndim >= 2 and "embedding" not in name and not name.endswith("out_proj.weight")
        )
        has_decay = _use_weight_decay(name, p)
        if is_muon_candidate:
            if has_decay:
                muon_decay.append(p)
            else:
                muon_no_decay.append(p)
        else:
            if has_decay:
                aux_decay.append(p)
            else:
                aux_no_decay.append(p)
    return muon_decay, muon_no_decay, aux_decay, aux_no_decay


def _is_pretrained_text_backbone_name(name: str) -> bool:
    return name.startswith("pretrained_text_backbone.") or ".pretrained_text_backbone." in name


def _trainable_pretrained_backbone_ids(model: torch.nn.Module) -> set[int]:
    """学習対象になっている事前学習テキストバックボーンパラメータの id 集合。"""
    return {
        id(p)
        for name, p in model.named_parameters()
        if p.requires_grad and _is_pretrained_text_backbone_name(name)
    }


def _pretrained_text_encoder_lr(cfg) -> float:
    lr = getattr(cfg, "pretrained_text_encoder_learning_rate", None)
    return float(lr) if lr is not None else float(cfg.learning_rate)


def _build_pretrained_text_encoder_optimizer(
    model: torch.nn.Module,
    named_params: list[tuple[str, torch.nn.Parameter]],
    cfg,
):
    """事前学習テキストバックボーン専用のオプティマイザを作る（adamw / muon）。

    muon の場合は本体と同基準（nn.Embedding 由来・out_proj を除く 2 次元行列）で
    Muon 対象と補助 AdamW 対象に分割し、MuonWithAuxAdamW にまとめて返す。
    """
    pre_lr = _pretrained_text_encoder_lr(cfg)
    opt_name = str(getattr(cfg, "pretrained_text_encoder_optimizer", "adamw") or "adamw").strip().lower()
    if opt_name == "adamw":
        wd_params = [p for _, p in named_params if p.dim() >= 2]
        no_wd_params = [p for _, p in named_params if p.dim() < 2]
        param_groups = []
        if wd_params:
            param_groups.append({"params": wd_params, "weight_decay": cfg.weight_decay})
        if no_wd_params:
            param_groups.append({"params": no_wd_params, "weight_decay": 0.0})
        return torch.optim.AdamW(
            param_groups,
            lr=pre_lr,
            betas=(cfg.adam_beta1, cfg.adam_beta2),
            eps=cfg.adam_eps,
        )
    if opt_name == "muon":
        if str(cfg.optimizer).strip().lower() != "muon":
            raise ValueError(
                "pretrained_text_encoder_optimizer='muon' requires optimizer='muon' "
                f"(got {cfg.optimizer!r}) so Muon hyperparameters apply consistently."
            )
        if not hasattr(torch.optim, "Muon"):
            raise RuntimeError(
                "pretrained_text_encoder_optimizer='muon' requires torch.optim.Muon "
                "(available in newer PyTorch releases)."
            )
        adjust_lr_fn = cfg.muon_adjust_lr_fn
        if adjust_lr_fn not in {"original", "match_rms_adamw"}:
            raise ValueError(
                "muon_adjust_lr_fn must be one of ['original', 'match_rms_adamw'], "
                f"got {adjust_lr_fn!r}"
            )
        muon_decay: list[torch.nn.Parameter] = []
        muon_no_decay: list[torch.nn.Parameter] = []
        aux_decay: list[torch.nn.Parameter] = []
        aux_no_decay: list[torch.nn.Parameter] = []
        embedding_param_ids = {
            id(parameter)
            for module in model.modules()
            if isinstance(module, torch.nn.Embedding)
            for parameter in module.parameters(recurse=False)
        }
        for name, p in named_params:
            # 本体（_partition_muon_params 相当）と同じ基準:
            # 2 次元行列のみ。埋め込み・out_proj は補助 AdamW 側へ回す。
            is_muon_candidate = (
                p.ndim == 2
                and id(p) not in embedding_param_ids
                and not name.endswith("out_proj.weight")
            )
            has_decay = _use_weight_decay(name, p)
            if is_muon_candidate:
                (muon_decay if has_decay else muon_no_decay).append(p)
            else:
                (aux_decay if has_decay else aux_no_decay).append(p)
        if not muon_decay and not muon_no_decay:
            raise ValueError(
                "No Muon-compatible parameters found for "
                "pretrained_text_encoder_optimizer='muon'."
            )
        muon_param_groups = []
        if muon_decay:
            muon_param_groups.append({"params": muon_decay, "weight_decay": cfg.weight_decay})
        if muon_no_decay:
            muon_param_groups.append({"params": muon_no_decay, "weight_decay": 0.0})
        muon_opt = torch.optim.Muon(
            muon_param_groups,
            lr=pre_lr,
            weight_decay=0.0,
            momentum=cfg.muon_momentum,
            adjust_lr_fn=adjust_lr_fn,
        )
        aux_param_groups = []
        if aux_decay:
            aux_param_groups.append({"params": aux_decay, "weight_decay": cfg.weight_decay})
        if aux_no_decay:
            aux_param_groups.append({"params": aux_no_decay, "weight_decay": 0.0})
        if not aux_param_groups:
            return muon_opt
        aux_opt = torch.optim.AdamW(
            aux_param_groups,
            lr=pre_lr,
            weight_decay=0.0,
            betas=(cfg.adam_beta1, cfg.adam_beta2),
            eps=cfg.adam_eps,
        )
        return MuonWithAuxAdamW(muon_opt=muon_opt, aux_opt=aux_opt)
    raise ValueError(
        "pretrained_text_encoder_optimizer must be one of ['adamw', 'muon'], "
        f"got {cfg.pretrained_text_encoder_optimizer!r}"
    )


def _pretrained_optimizer_label(cfg) -> str:
    opt_name = str(getattr(cfg, "pretrained_text_encoder_optimizer", "adamw") or "adamw").strip().lower()
    return "Muon+aux AdamW" if opt_name == "muon" else "AdamW"


def _split_params_by_ids(
    params: list[torch.nn.Parameter],
    pre_ids: set[int],
) -> tuple[list[torch.nn.Parameter], list[torch.nn.Parameter]]:
    base: list[torch.nn.Parameter] = []
    pretrained: list[torch.nn.Parameter] = []
    for p in params:
        if id(p) in pre_ids:
            pretrained.append(p)
        else:
            base.append(p)
    return base, pretrained


def build_optimizer(model: torch.nn.Module, cfg: TrainConfig):
    opt_name = cfg.optimizer.lower()
    pre_ids = _trainable_pretrained_backbone_ids(model)
    if opt_name == "adamw":
        decay, no_decay = _partition_adamw_params(model)
        pre_named = [(name, p) for name, p in model.named_parameters() if id(p) in pre_ids]
        if pre_named:
            pre_set = {id(p) for _, p in pre_named}
            decay = [p for p in decay if id(p) not in pre_set]
            no_decay = [p for p in no_decay if id(p) not in pre_set]
        param_groups = []
        if decay:
            param_groups.append({"params": decay, "weight_decay": cfg.weight_decay})
        if no_decay:
            param_groups.append({"params": no_decay, "weight_decay": 0.0})
        if pre_named:
            pre_lr = _pretrained_text_encoder_lr(cfg)
            pre_opt = _build_pretrained_text_encoder_optimizer(model, pre_named, cfg)
            print(
                f"Pretrained text encoder optimizer: {_pretrained_optimizer_label(cfg)} "
                f"(lr={pre_lr:.3e}, params={len(pre_named)})"
            )
            if not param_groups:
                return pre_opt
            main_opt = torch.optim.AdamW(
                param_groups,
                lr=cfg.learning_rate,
                weight_decay=0.0,
                betas=(cfg.adam_beta1, cfg.adam_beta2),
                eps=cfg.adam_eps,
            )
            return OptimizerChain([main_opt, pre_opt])
        return torch.optim.AdamW(
            param_groups if param_groups else model.parameters(),
            lr=cfg.learning_rate,
            weight_decay=0.0,
            betas=(cfg.adam_beta1, cfg.adam_beta2),
            eps=cfg.adam_eps,
        )
    if opt_name == "muon":
        if not hasattr(torch.optim, "Muon"):
            raise RuntimeError(
                "optimizer=muon requires torch.optim.Muon (available in newer PyTorch releases)."
            )
        adjust_lr_fn = cfg.muon_adjust_lr_fn
        if adjust_lr_fn not in {"original", "match_rms_adamw"}:
            raise ValueError(
                "muon_adjust_lr_fn must be one of ['original', 'match_rms_adamw'], "
                f"got {adjust_lr_fn!r}"
            )

        muon_decay, muon_no_decay, aux_decay, aux_no_decay = _partition_muon_params(model, exclude_ids=pre_ids)
        muon_param_groups = []
        if muon_decay:
            muon_param_groups.append({"params": muon_decay, "weight_decay": cfg.weight_decay})
        if muon_no_decay:
            muon_param_groups.append({"params": muon_no_decay, "weight_decay": 0.0})
        if not muon_param_groups:
            raise ValueError("No Muon-compatible parameters found for optimizer=muon.")

        muon_opt = torch.optim.Muon(
            muon_param_groups,
            lr=cfg.learning_rate,
            weight_decay=0.0,
            momentum=cfg.muon_momentum,
            adjust_lr_fn=adjust_lr_fn,
        )
        aux_opt = None
        aux_param_groups = []
        if aux_decay:
            aux_param_groups.append({"params": aux_decay, "weight_decay": cfg.weight_decay})
        if aux_no_decay:
            aux_param_groups.append({"params": aux_no_decay, "weight_decay": 0.0})
        if aux_param_groups:
            aux_opt = torch.optim.AdamW(
                aux_param_groups,
                lr=cfg.learning_rate,
                weight_decay=0.0,
                betas=(cfg.adam_beta1, cfg.adam_beta2),
                eps=cfg.adam_eps,
            )
        wrapped = MuonWithAuxAdamW(muon_opt=muon_opt, aux_opt=aux_opt)
        if pre_ids:
            pre_named = [(name, p) for name, p in model.named_parameters() if id(p) in pre_ids]
            pre_lr = _pretrained_text_encoder_lr(cfg)
            pre_opt = _build_pretrained_text_encoder_optimizer(model, pre_named, cfg)
            print(
                f"Pretrained text encoder optimizer: {_pretrained_optimizer_label(cfg)} "
                f"(lr={pre_lr:.3e}, params={len(pre_named)})"
            )
            return OptimizerChain([wrapped, pre_opt])
        return wrapped

    raise ValueError(f"Unsupported optimizer: {cfg.optimizer}")


def build_scheduler(
    optimizer,
    cfg: TrainConfig,
):
    sched_name = cfg.lr_scheduler.lower()
    if sched_name == "none":
        return None
    if sched_name not in {"cosine", "wsd"}:
        raise ValueError(f"Unsupported lr_scheduler: {cfg.lr_scheduler}")

    max_steps = max(1, int(cfg.max_steps))
    warmup_steps = max(0, int(cfg.warmup_steps))
    stable_steps = max(0, int(cfg.stable_steps))
    min_lr_scale = float(max(0.0, min(1.0, cfg.min_lr_scale)))

    def lr_lambda(step: int) -> float:
        s = int(step)
        if warmup_steps > 0 and s < warmup_steps:
            return float(s + 1) / float(warmup_steps)

        if sched_name == "cosine":
            denom = max(1, max_steps - warmup_steps)
            progress = min(1.0, max(0.0, float(s - warmup_steps) / float(denom)))
        else:
            if s < warmup_steps + stable_steps:
                return 1.0
            denom = max(1, max_steps - warmup_steps - stable_steps)
            progress = min(1.0, max(0.0, float(s - warmup_steps - stable_steps) / float(denom)))

        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_lr_scale + (1.0 - min_lr_scale) * cosine

    return ScalarLRScheduler(optimizer, lr_lambda=lr_lambda)


def current_lr(optimizer) -> float:
    return float(optimizer.param_groups[0]["lr"])


def build_optimizer_extended(
    model: torch.nn.Module,
    train_cfg,
    optimizer_name: str | None = None,
    *,
    lion_betas: tuple[float, float] = (0.9, 0.99),
    ademamix_alpha: float = 5.0,
    ademamix_beta3: float = 0.9999,
    trainable_params_override: list[torch.nn.Parameter] | None = None,
) -> torch.optim.Optimizer:
    """build_optimizerを拡張し、Lion / AdEMAMix / SGD を追加サポート。

    muon / adamw / その他は既存の build_optimizer に委譲する。
    trainable_params_override: 指定された場合、model.parameters()の代わりにこのリストを使用する。
                               LoRA差分学習等でアダプタパラメータのみを渡す際に使用。
    """
    name = (optimizer_name or train_cfg.optimizer).lower()
    if trainable_params_override is not None:
        params = [p for p in trainable_params_override if p.requires_grad]
    else:
        params = [p for p in model.parameters() if p.requires_grad]
    # 事前学習テキストバックボーン（v4系）は別学習率の専用オプティマイザへ分離する。
    pre_ids = _trainable_pretrained_backbone_ids(model)
    params, pre_params = _split_params_by_ids(params, pre_ids)

    def _maybe_chain(main_opt):
        if not pre_params:
            return main_opt
        pre_id_set = {id(p) for p in pre_params}
        pre_named = [(name, p) for name, p in model.named_parameters() if id(p) in pre_id_set]
        pre_lr = _pretrained_text_encoder_lr(train_cfg)
        pre_opt = _build_pretrained_text_encoder_optimizer(model, pre_named, train_cfg)
        print(
            f"Pretrained text encoder optimizer: {_pretrained_optimizer_label(train_cfg)} "
            f"(lr={pre_lr:.3e}, params={sum(p.numel() for p in pre_params):,})"
        )
        return OptimizerChain([main_opt, pre_opt])

    wd_params = [p for p in params if p.dim() >= 2]
    no_wd_params = [p for p in params if p.dim() < 2]
    param_groups = [
        {"params": wd_params, "weight_decay": train_cfg.weight_decay},
        {"params": no_wd_params, "weight_decay": 0.0},
    ]

    if name == "lion":
        try:
            from lion_pytorch import Lion
            opt = Lion(param_groups, lr=train_cfg.learning_rate, betas=lion_betas)
            print(f"Optimizer: Lion (betas={lion_betas})")
            return _maybe_chain(opt)
        except ImportError:
            print(
                "warning: lion_pytorch未インストール。"
                "pip install lion-pytorch でインストールしてください。AdamWにフォールバック。"
            )
            name = "adamw"

    if name == "ademamix":
        try:
            from optimi import AdEMAMix
            opt = AdEMAMix(
                param_groups,
                lr=train_cfg.learning_rate,
                betas=(train_cfg.adam_beta1, train_cfg.adam_beta2, ademamix_beta3),
                alpha=ademamix_alpha,
                eps=train_cfg.adam_eps,
                weight_decay=train_cfg.weight_decay,
            )
            print(f"Optimizer: AdEMAMix (alpha={ademamix_alpha}, beta3={ademamix_beta3})")
            return _maybe_chain(opt)
        except ImportError:
            print(
                "warning: optimi未インストール。"
                "pip install optimi でインストールしてください。AdamWにフォールバック。"
            )
            name = "adamw"

    if name == "sgd":
        opt = torch.optim.SGD(
            param_groups,
            lr=train_cfg.learning_rate,
            momentum=0.9,
            nesterov=True,
        )
        print("Optimizer: SGD (momentum=0.9, nesterov=True)")
        return _maybe_chain(opt)

    if name == "adamw":
        opt = torch.optim.AdamW(
            param_groups,
            lr=train_cfg.learning_rate,
            betas=(train_cfg.adam_beta1, train_cfg.adam_beta2),
            eps=train_cfg.adam_eps,
            weight_decay=train_cfg.weight_decay,
            fused=torch.cuda.is_available(),
        )
        print(f"Optimizer: AdamW (fused={torch.cuda.is_available()})")
        return _maybe_chain(opt)

    # muon / その他はオリジナルの build_optimizer に委譲
    return build_optimizer(model, train_cfg)
