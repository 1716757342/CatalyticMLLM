import transformers
from dataclasses import dataclass, field
from typing import Dict, Optional, Sequence, List


@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="Qwen/Qwen2.5-VL-3B-Instruct")
    tune_mm_llm: bool = field(default=False)
    tune_mm_mlp: bool = field(default=False)
    tune_mm_vision: bool = field(default=False)
    # [[[Added]]] LoRA related parameters
    use_lora: bool = field(
        default=False,
        metadata={"help": "Whether to use LoRA for fine-tuning."}
    )
    lora_r: int = field(
        default=64,
        metadata={"help": "LoRA attention dimension (rank)."}
    )
    lora_alpha: int = field(
        default=16,
        metadata={"help": "The alpha parameter for LoRA scaling."}
    )
    lora_dropout: float = field(
        default=0.05,
        metadata={"help": "The dropout probability for LoRA layers."}
    )
    lora_target_modules: List[str] = field(
        default_factory=lambda: ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        metadata={"help": "The list of modules to apply LoRA to."}
    )

@dataclass
class DataArguments:
    dataset_use: str = field(default="")
    video_max_frames: Optional[int] = field(default=8)
    video_min_frames: Optional[int] = field(default=4)
    data_flatten: bool = field(default=False)
    data_packing: bool = field(default=False)
    base_interval: int = field(default=2)
    max_pixels: int = field(default=28 * 28 * 576)
    min_pixels: int = field(default=28 * 28 * 16)
    video_max_frame_pixels: int = field(default=32 * 28 * 28)
    video_min_frame_pixels: int = field(default=4 * 28 * 28)
    # Common data path (for supervised learning and PPO)
    data_path: Optional[str] = field(
        default=None,
        metadata={"help": "Path to the training data"}
    )
    eval_data_path: Optional[str] = field(
        default=None,
        metadata={"help": "Path to the evaluation data"}
    )
    # GRPO-specific parameters
    preference_data_path: Optional[str] = field(
        default=None,
        metadata={"help": "Path to the preference data for GRPO training"}
    )
    eval_preference_data_path: Optional[str] = field(
        default=None,
        metadata={"help": "Path to the evaluation preference data for GRPO training"}
    )


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    model_max_length: int = field(
        default=512,
        metadata={
            "help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."
        },
    )
    mm_projector_lr: Optional[float] = None
    vision_tower_lr: Optional[float] = None
    # [[[↓↓↓ Add new fields here ↓↓↓]]]
    remove_unused_columns: bool = field(
        default=False,
        metadata={"help": "Whether to remove unused columns from the dataset. Set to False for custom data processing."},
    )
    # GRPO-specific parameters
    grpo_beta: float = field(
        default=0.1,
        metadata={"help": "Temperature parameter for GRPO loss, controls deviation from reference model"}
    )
    grpo_label_smoothing: float = field(
        default=0.0,
        metadata={"help": "Label smoothing parameter for GRPO loss"}
    )
    grpo_reference_free: bool = field(
        default=False,
        metadata={"help": "Whether to use reference-free GRPO variant"}
    )
    # PPO-specific parameters
    ppo_beta: float = field(
        default=0.1,
        metadata={"help": "KL penalty coefficient for PPO, controls deviation from reference model"}
    )
    ppo_entropy_coef: float = field(
        default=0.01,
        metadata={"help": "Entropy coefficient for PPO, encourages exploration"}
    )
    ppo_use_reference: bool = field(
        default=True,
        metadata={"help": "Whether to use reference model for PPO (if False, pure policy gradient)"}
    )
    # ============================================================
    # Stage 3 IRFT (Iterative Reward Fine-Tuning) specific parameters
    # ============================================================
    energy_reward_weight: float = field(
        default=0.3,
        metadata={"help": "Stage 3: weight ω₂ for energy reward in R_step3 = ω₁·R_step2 + ω₂·R_energy"}
    )
    structure_reward_weight: float = field(
        default=0.7,
        metadata={"help": "Stage 3: weight ω₁ for structure reward in R_step3 = ω₁·R_step2 + ω₂·R_energy"}
    )
    energy_lambda: float = field(
        default=1.0,
        metadata={"help": "Stage 3: λ in R_energy = exp(-λ·|E_pred - E_target|)"}
    )
    exemplar_pool_size: int = field(
        default=50,
        metadata={"help": "Stage 3: maximum number of exemplars in the exemplar pool"}
    )
    exemplar_pool_path: Optional[str] = field(
        default=None,
        metadata={"help": "Stage 3: path to load/save the exemplar pool JSON file"}
    )
    max_energy_pred_tokens: int = field(
        default=64,
        metadata={"help": "Stage 3: max new tokens for energy prediction inference"}
    )
