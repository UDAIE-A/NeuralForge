"""NeuralForge learning package: live online fine-tuning from human feedback."""

from .online import OnlineLearner, FeedbackStore
from .lora import (
    LoRALinear, inject_lora, freeze_base, lora_state_dict,
    count_lora_params, merge_lora, DEFAULT_TARGETS,
)
from .replay import ReplayBuffer, RegressionProbe

__all__ = [
    "OnlineLearner", "FeedbackStore",
    "LoRALinear", "inject_lora", "freeze_base", "lora_state_dict",
    "count_lora_params", "merge_lora", "DEFAULT_TARGETS",
    "ReplayBuffer", "RegressionProbe",
]
