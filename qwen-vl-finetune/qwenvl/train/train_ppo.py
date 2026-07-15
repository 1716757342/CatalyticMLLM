# qwen-vl-finetune/qwenvl/train/train_ppo.py

import os
import logging
import pathlib
import torch
import torch.nn as nn
import transformers
import json
import copy
from typing import Dict, Optional, List, Union, Tuple
import sys
from pathlib import Path
import warnings

# Suppress transformers tokenizer deprecation warning
warnings.filterwarnings("ignore", message=".*Trainer.tokenizer is now deprecated.*")
warnings.filterwarnings("ignore", message=".*processing_class.*")

# Import existing modules
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from transformers import (
    Qwen2VLForConditionalGeneration,
    Qwen2_5_VLForConditionalGeneration,
    AutoTokenizer,
    AutoProcessor,
    Trainer
)

# Import custom modules
try:
    from .train_qwen import Qwen2_5_VLForMolecule, set_model, safe_save_model_for_hf_trainer
    from ..data.data_ppo import make_ppo_data_module
    from ..model.equiformer_v2_wrapper import EquiformerV2Wrapper
    from .argument import ModelArguments, DataArguments, TrainingArguments
    from .ppo_loss import PPOLoss
    from .reward_model import EnergyRewardModel
    from .reward_model_continuous import ContinuousEnergyRewardModel
    from .reward_model_pure_continuous import PureContinuousEnergyRewardModel
except ImportError:
    try:
        from qwenvl.train.train_qwen import Qwen2_5_VLForMolecule, set_model, safe_save_model_for_hf_trainer
        from qwenvl.data.data_ppo import make_ppo_data_module
        from qwenvl.model.equiformer_v2_wrapper import EquiformerV2Wrapper
        from qwenvl.train.argument import ModelArguments, DataArguments, TrainingArguments
        from qwenvl.train.ppo_loss import PPOLoss
        from qwenvl.train.reward_model import EnergyRewardModel
        from qwenvl.train.reward_model_continuous import ContinuousEnergyRewardModel
        from qwenvl.train.reward_model_pure_continuous import PureContinuousEnergyRewardModel
    except ImportError:
        current_dir = Path(__file__).parent
        sys.path.insert(0, str(current_dir))
        sys.path.insert(0, str(current_dir.parent / "data"))
        sys.path.insert(0, str(current_dir.parent / "model"))
        
        from train_qwen import Qwen2_5_VLForMolecule, set_model, safe_save_model_for_hf_trainer
        from data_ppo import make_ppo_data_module
        from equiformer_v2_wrapper import EquiformerV2Wrapper
        from argument import ModelArguments, DataArguments, TrainingArguments
        from ppo_loss import PPOLoss
        from reward_model import EnergyRewardModel
        from reward_model_continuous import ContinuousEnergyRewardModel
        from reward_model_pure_continuous import PureContinuousEnergyRewardModel


class PPOTrainer(Trainer):
    """
    PPO-style Policy Gradient trainer
    Optimizes directly from task rewards without preference pairs
    """
    
    def __init__(
        self,
        model=None,
        reference_model=None,
        reward_model=None,
        ppo_config: Dict = None,
        **kwargs
    ):
        super().__init__(model=model, **kwargs)
        
        # PPOconfiguration
        self.ppo_config = ppo_config or {}
        self.beta = self.ppo_config.get("beta", 0.1)
        self.entropy_coef = self.ppo_config.get("entropy_coef", 0.01)
        self.use_reference = self.ppo_config.get("use_reference", True)
        
        # reference model (frozen initial model)
        self.reference_model = reference_model
        if self.use_reference and self.reference_model is not None:
            self.reference_model.eval()
            for param in self.reference_model.parameters():
                param.requires_grad = False
        
        # reward model (using pure continuous version)
        self.reward_model = reward_model
        if self.reward_model is None:
            # Use pure continuous reward model by default (no segmentation)
            self.reward_model = PureContinuousEnergyRewardModel(
                reward_scale=15.0,     # maximum reward
                sensitivity=5.0,       # error sensitivity
                reward_offset=-5.0,    # offset
                min_reward=-15.0       # minimum reward
            )
            print(f"   - Using pure continuous reward model (no segmentation) ✅")
        
        # PPO loss function
        self.ppo_loss_fn = PPOLoss(
            beta=self.beta,
            entropy_coef=self.entropy_coef,
            baseline_decay=0.99
        )
        
        print(f"✅ PPO Trainer initialized:")
        print(f"   - Beta (KL penalty): {self.beta}")
        print(f"   - Entropy coef: {self.entropy_coef}")
        print(f"   - Use reference model: {self.use_reference}")
    
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        """
        Compute PPO loss
        
        Core logic:
        1. Model forward pass generates logits
        2. Compute log probabilities
        3. Extract predicted energy value
        4. Compute rewards
        5. Compute PPO loss
        """
        # Ensure all tensors are on the correct device
        device = next(model.parameters()).device
        
        # Move inputs to device
        device_inputs = {}
        for key, value in inputs.items():
            if isinstance(value, torch.Tensor):
                device_inputs[key] = value.to(device)
            else:
                device_inputs[key] = value
        
        # Extract true energy value
        true_energies = device_inputs.pop("true_energies")
        
        try:
            # policy model forward pass
            outputs = model(**device_inputs)
            logits = outputs.logits
            labels = device_inputs["labels"]
            
            # Compute log probabilities
            policy_log_probs = self.ppo_loss_fn.get_log_probs(logits, labels)
            
            # Generate predicted text and extract energy value
            with torch.no_grad():
                # Method:extract generated part from labels (tokens that are not IGNORE_INDEX)
                # then replace these positions with argmax of logits
                IGNORE_INDEX = -100
                
                # Process each sample separately
                predicted_texts = []
                for i in range(labels.shape[0]):
                    label_seq = labels[i]
                    logit_seq = logits[i]
                    
                    # Find generated part (positions where labels are not IGNORE_INDEX)
                    valid_mask = (label_seq != IGNORE_INDEX)
                    
                    if valid_mask.sum() > 0:
                        # Get predicted tokens for generated part
                        valid_positions = torch.where(valid_mask)[0]
                        predicted_tokens = torch.argmax(logit_seq[valid_positions], dim=-1)
                        
                        # Decode (only generated part)
                        pred_text = self.processing_class.decode(predicted_tokens, skip_special_tokens=True)
                    else:
                        pred_text = ""
                    
                    predicted_texts.append(pred_text)
                
                # Extract energy value and compute reward
                rewards = []
                reward_details = []  # for debugging
                for pred_text, true_energy in zip(predicted_texts, true_energies):
                    result = self.reward_model.compute_single_reward(pred_text, true_energy.item())
                    rewards.append(result["reward"])
                    reward_details.append({
                        "text": pred_text[:100],  # record only first 100 characters
                        "reward": result["reward"],
                        "predicted_energy": result.get("predicted_energy"),
                        "true_energy": true_energy.item(),
                        "abs_error": result.get("abs_error")
                    })
                
                # Print samples every 100 steps for debugging
                if self.state.global_step % 100 == 0 and self.state.global_step > 0:
                    print(f"\n🔍 [Step {self.state.global_step}] Prediction sample example:")
                    for idx, detail in enumerate(reward_details[:2]):  # print only the first 2
                        print(f"  sample{idx+1}:")
                        print(f"    Predicted text: {detail['text']}")
                        print(f"    True energy: {detail['true_energy']:.6f}")
                        print(f"    Predicted energy: {detail['predicted_energy']}")
                        print(f"    Absolute error: {detail['abs_error']:.6f}")
                        print(f"    reward: {detail['reward']:.4f}")
                
                rewards = torch.tensor(rewards, dtype=torch.float32, device=device)
            
            # Reference-model forward pass (if used)
            if self.use_reference and self.reference_model is not None:
                # Ensure reference model is on the correct device
                if next(self.reference_model.parameters()).device != device:
                    self.reference_model = self.reference_model.to(device)
                
                with torch.no_grad():
                    ref_outputs = self.reference_model(**device_inputs)
                    ref_logits = ref_outputs.logits
                    ref_log_probs = self.ppo_loss_fn.get_log_probs(ref_logits, labels)
            else:
                ref_log_probs = None
            
            # Compute PPO loss
            loss_dict = self.ppo_loss_fn(
                policy_log_probs=policy_log_probs,
                reference_log_probs=ref_log_probs,
                rewards=rewards,
                logits=logits
            )
            
            # Log statistics
            if self.state.global_step % self.args.logging_steps == 0:
                self.log({
                    "train/ppo_loss": loss_dict["loss"].item(),
                    "train/pg_loss": loss_dict["pg_loss"].item(),
                    "train/entropy": loss_dict["entropy"].item(),
                    "train/kl_divergence": loss_dict["kl_divergence"].item(),
                    "train/reward_mean": loss_dict["reward_mean"].item(),
                    "train/reward_std": loss_dict["reward_std"].item(),
                    "train/advantage_mean": loss_dict["advantage_mean"].item(),
                    "train/baseline": loss_dict["baseline"].item(),
                    "train/positive_reward_rate": loss_dict["positive_reward_rate"].item(),
                    "train/high_quality_rate": loss_dict["high_quality_rate"].item(),
                    "train/policy_log_probs": loss_dict["policy_log_probs"].item(),
                })
            
        except Exception as e:
            print(f"❌ PPO forward pass failed: {e}")
            print(f"Input tensor information:")
            for key, value in device_inputs.items():
                if isinstance(value, torch.Tensor):
                    print(f"  {key}: shape={value.shape}, device={value.device}, dtype={value.dtype}")
            raise e
        
        if return_outputs:
            class DummyOutputs:
                def __init__(self, logits):
                    self.logits = logits
                    self.loss = loss_dict["loss"]
            
            return loss_dict["loss"], DummyOutputs(logits)
        else:
            return loss_dict["loss"]
    
    def save_model(self, output_dir: Optional[str] = None, _internal_call: bool = False):
        """savemodel"""
        if output_dir is None:
            output_dir = self.args.output_dir
        
        # Call parent save_model
        super().save_model(output_dir, _internal_call)
        
        # Save PPO configuration
        ppo_config_path = os.path.join(output_dir, "ppo_config.json")
        with open(ppo_config_path, 'w') as f:
            json.dump(self.ppo_config, f, indent=2)
        
        # Save baseline
        baseline_path = os.path.join(output_dir, "ppo_baseline.pt")
        torch.save({
            'baseline': self.ppo_loss_fn.baseline,
            'baseline_count': self.ppo_loss_fn.baseline_count
        }, baseline_path)
        
        print(f"✅ PPO model and configuration saved to: {output_dir}")


def create_reference_model(model_args, training_args, attn_implementation="flash_attention_2"):
    """Create reference model (frozen copy of the original model)"""
    print("Creating reference model...")
    
    reference_model = Qwen2_5_VLForMolecule.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        attn_implementation=attn_implementation,
        torch_dtype=(torch.bfloat16 if training_args.bf16 else None),
    )
    
    # Perform the same surgery as the policy model (replace visual module)
    equiformer_v2_config_path = 'pretrained_equiformer_v2/equiformer_v2_N@8_L@4_M@2_31M.yml'
    equiformer_v2_weights_path = 'pretrained_equiformer_v2/eq2_31M_ec4_allmd.pt'
    original_vision_config = reference_model.config.vision_config
    equiformer_v2_wrapper = EquiformerV2Wrapper(
        config_file=equiformer_v2_config_path,
        pretrained_path=equiformer_v2_weights_path,
        qwen_vision_config=original_vision_config
    )
    reference_model.visual = equiformer_v2_wrapper
    
    # Device assignment
    target_dtype = torch.bfloat16 if training_args.bf16 else (torch.float16 if training_args.fp16 else reference_model.dtype)
    
    if torch.distributed.is_initialized():
        device = torch.device(f"cuda:{training_args.local_rank}")
    else:
        device = training_args.device if hasattr(training_args, 'device') else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    reference_model = reference_model.to(device=device, dtype=target_dtype)
    
    # Freeze
    reference_model.config.use_cache = False
    if hasattr(reference_model.config, 'tokenizer_padding_side'):
        reference_model.config.tokenizer_padding_side = "left"
    
    reference_model.eval()
    for param in reference_model.parameters():
        param.requires_grad = False
    
    print(f"✅ Reference model created, device: {device}")
    return reference_model


def train_ppo(attn_implementation="flash_attention_2"):
    """PPO training main function"""
    global local_rank
    
    # Parse arguments
    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments)
    )
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    
    local_rank = training_args.local_rank
    os.makedirs(training_args.output_dir, exist_ok=True)
    
    # Create policy model
    print("Creating policy model...")
    policy_model = Qwen2_5_VLForMolecule.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        attn_implementation=attn_implementation,
        torch_dtype=(torch.bfloat16 if training_args.bf16 else None),
    )
    data_args.image_processor = AutoProcessor.from_pretrained(
        model_args.model_name_or_path,
    ).image_processor
    data_args.model_type = "qwen2.5vl"
    
    # Perform model surgery
    if (not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0):
        print("="*50)
        print("Perform model surgery: replacing ViT with pretrained EquiformerV2...")
    
    equiformer_v2_config_path = 'pretrained_equiformer_v2/equiformer_v2_N@8_L@4_M@2_31M.yml'
    equiformer_v2_weights_path = 'pretrained_equiformer_v2/eq2_31M_ec4_allmd.pt'
    original_vision_config = policy_model.config.vision_config
    equiformer_v2_wrapper = EquiformerV2Wrapper(
        config_file=equiformer_v2_config_path,
        pretrained_path=equiformer_v2_weights_path,
        qwen_vision_config=original_vision_config
    )
    policy_model.visual = equiformer_v2_wrapper
    
    # Device assignment
    target_dtype = torch.bfloat16 if training_args.bf16 else (torch.float16 if training_args.fp16 else policy_model.dtype)
    
    if torch.distributed.is_initialized():
        device = torch.device(f"cuda:{training_args.local_rank}")
    else:
        device = training_args.device if hasattr(training_args, 'device') else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    policy_model = policy_model.to(device=device, dtype=target_dtype)
    
    if (not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0):
        print("Policy model surgery complete.")
        print("="*50)
    
    policy_model.config.use_cache = False
    
    if hasattr(policy_model.config, 'tokenizer_padding_side'):
        policy_model.config.tokenizer_padding_side = "left"
    
    set_model(model_args, policy_model)
    
    if training_args.gradient_checkpointing:
        policy_model.enable_input_require_grads()
    
    # createReference model (optional)
    use_reference = getattr(training_args, 'ppo_use_reference', True)
    if use_reference:
        reference_model = create_reference_model(model_args, training_args, attn_implementation)
    else:
        reference_model = None
        print("⚠️  Reference model not used (pure Policy Gradient mode)")
    
    # Create tokenizer
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        model_max_length=training_args.model_max_length,
        padding_side="left",  # Flash Attention 2 requirement
        use_fast=False,
    )
    tokenizer.padding_side = "left"  # Explicitly ensure
    
    if (not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0):
        print(f"✅ Tokenizer padding_side set to: {tokenizer.padding_side}")
    
    # Print trainable parameters
    if (not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0):
        print("="*50)
        print("--- Policy model trainable parameter status ---")
        policy_model.visual.print_trainable_parameters()
        
        trainable_params = 0
        total_params = 0
        for name, param in policy_model.model.named_parameters():
            total_params += param.numel()
            if param.requires_grad:
                trainable_params += param.numel()
        
        print("--- Language Model (LLM) ---")
        print(f" - Trainable: {'Yes' if trainable_params > 0 else 'No'}")
        print(f" - Trainable Parameters: {trainable_params} / {total_params} ({100.0 * trainable_params / total_params:.2f}%)")
        print("="*50)
    
    # Load PPO training data
    if not hasattr(data_args, 'data_path') or data_args.data_path is None:
        raise ValueError("data_path must be specified in data_args")
    
    if not os.path.exists(data_args.data_path):
        raise ValueError(f"Training data file does not exist: {data_args.data_path}")
    
    data_module = make_ppo_data_module(
        tokenizer=tokenizer,
        data_path=data_args.data_path,
        eval_data_path=getattr(data_args, 'eval_data_path', None)
    )
    
    # PPOconfiguration
    ppo_config = {
        "beta": getattr(training_args, 'ppo_beta', 0.1),
        "entropy_coef": getattr(training_args, 'ppo_entropy_coef', 0.01),
        "use_reference": use_reference,
        "use_continuous_reward": True,  # use continuous reward by default
    }
    
    # Create PPO trainer
    trainer = PPOTrainer(
        model=policy_model,
        reference_model=reference_model,
        ppo_config=ppo_config,
        processing_class=tokenizer,
        args=training_args,
        **data_module
    )
    
    # Start training
    if list(pathlib.Path(training_args.output_dir).glob("checkpoint-*")):
        logging.info("Checkpoint found; resuming training")
        trainer.train(resume_from_checkpoint=True)
    else:
        trainer.train()
    
    # savemodel
    trainer.save_state()
    if hasattr(data_args, 'image_processor'):
        data_args.image_processor.save_pretrained(training_args.output_dir)
    
    policy_model.config.use_cache = True
    trainer.save_model()
    
    # Close distributed process group
    if torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()


if __name__ == "__main__":
    train_ppo(attn_implementation="flash_attention_2")

