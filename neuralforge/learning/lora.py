"""
LoRA (Low-Rank Adaptation) for NeuralForge.

A LoRA layer wraps a frozen nn.Linear and adds a trainable low-rank update
`B @ A * (alpha / rank)`. B starts at zero, so the wrapped model is initially
identical to the base model and training can only move it deliberately.

This lives in the package rather than in scripts/ because two callers need it:
the offline fine-tuner (scripts/lora_finetune_v2.py) and the live teacher
(OnlineLearner in lora mode), which must agree on layer naming or adapters
saved by one cannot be loaded by the other.
"""

from typing import Sequence

import torch
import torch.nn as nn

# Attention + MLP projections. Adapting both is what makes LoRA able to shift
# instruction-following behaviour, not just attention routing.
DEFAULT_TARGETS = ("q_proj", "k_proj", "v_proj", "o_proj",
                   "gate_proj", "up_proj", "down_proj")


class LoRALinear(nn.Module):
    """Low-Rank Adaptation wrapper for a Linear layer."""

    def __init__(self, base_layer: nn.Linear, rank: int = 16,
                 alpha: float = 32.0, dropout: float = 0.0):
        super().__init__()
        self.base_layer = base_layer
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank

        dev = base_layer.weight.device
        self.lora_A = nn.Linear(base_layer.in_features, rank, bias=False, device=dev)
        self.lora_B = nn.Linear(rank, base_layer.out_features, bias=False, device=dev)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        nn.init.normal_(self.lora_A.weight, std=0.02)
        nn.init.zeros_(self.lora_B.weight)          # zero update at init

        for p in base_layer.parameters():
            p.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.base_layer(x) + self.lora_B(self.dropout(self.lora_A(x))) * self.scaling

    @torch.no_grad()
    def merged_weight(self) -> torch.Tensor:
        """Base weight with the low-rank update folded in."""
        delta = (self.lora_B.weight @ self.lora_A.weight) * self.scaling
        return self.base_layer.weight.data + delta


def inject_lora(model, rank: int = 16, alpha: float = 32.0,
                target_modules: Sequence[str] = DEFAULT_TARGETS,
                dropout: float = 0.0, verbose: bool = False):
    """Replace target Linear layers with LoRA versions. Returns the new layers.

    Collects targets before mutating: named_modules() is a live traversal, and
    swapping children while iterating it can skip layers.
    """
    targets = []
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear) and any(name.endswith(t) for t in target_modules):
            targets.append((name, module))

    layers = []
    for name, module in targets:
        parent_name, _, child_name = name.rpartition(".")
        parent = model.get_submodule(parent_name) if parent_name else model
        layer = LoRALinear(module, rank=rank, alpha=alpha, dropout=dropout)
        setattr(parent, child_name, layer)
        layers.append((name, layer))
        if verbose:
            print(f"  Injected LoRA into {name} (rank={rank}, alpha={alpha})")
    return layers


def freeze_base(model):
    """Freeze everything except LoRA parameters. Returns the trainable list."""
    trainable = []
    for name, param in model.named_parameters():
        if "lora_" in name:
            param.requires_grad = True
            trainable.append(param)
        else:
            param.requires_grad = False
    return trainable


def lora_state_dict(model):
    """Just the adapter tensors - a few MB rather than the whole model."""
    return {n: p.detach().cpu() for n, p in model.named_parameters() if "lora_" in n}


def count_lora_params(model):
    lora = sum(p.numel() for n, p in model.named_parameters() if "lora_" in n)
    total = sum(p.numel() for p in model.parameters())
    return lora, total


@torch.no_grad()
def merged_state_dict(model):
    """A plain-NeuralForge state_dict with every adapter folded in.

    Unlike merge_lora this leaves the model wrapped, so a live learner can
    save a checkpoint and keep teaching. Unwrapping in place left the
    optimizer holding adapter tensors that were no longer in the model, and
    every lesson after a save silently changed nothing.
    Returns (state_dict, merged_layer_count).
    """
    lora_names = {name for name, m in model.named_modules() if isinstance(m, LoRALinear)}
    out = {}
    for key, tensor in model.state_dict().items():
        owner, _, leaf = key.rpartition(".")
        if owner.endswith(".lora_A") or owner.endswith(".lora_B"):
            continue
        if owner.endswith(".base_layer") and owner[:-len(".base_layer")] in lora_names:
            layer_name = owner[:-len(".base_layer")]
            if leaf == "weight":
                tensor = model.get_submodule(layer_name).merged_weight()
            key = f"{layer_name}.{leaf}"
        out[key] = tensor.detach().to("cpu", copy=True)
    return out, len(lora_names)


@torch.no_grad()
def merge_lora(model):
    """Fold every adapter back into its base Linear and unwrap.

    Produces a plain NeuralForge whose state_dict loads into an unmodified
    model, so a LoRA-taught model can be published as a normal checkpoint.
    """
    merged = 0
    while True:
        target = None
        for name, module in model.named_modules():
            if isinstance(module, LoRALinear):
                target = (name, module)
                break
        if target is None:
            break
        name, module = target
        base = module.base_layer
        base.weight.data.copy_(module.merged_weight())
        for p in base.parameters():
            p.requires_grad = True
        parent_name, _, child_name = name.rpartition(".")
        parent = model.get_submodule(parent_name) if parent_name else model
        setattr(parent, child_name, base)
        merged += 1
    return merged
