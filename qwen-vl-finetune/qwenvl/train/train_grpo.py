# qwen-vl-finetune/qwenvl/train/train_grpo.py

import os
import logging
import pathlib
import torch
import torch.nn as nn
import transformers
import json
import copy
from typing import Dict, Optional, List, Union, Tuple
import shutil
import sys
from pathlib import Path

# Import existing modules
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from transformers import (
    Qwen2VLForConditionalGeneration,
    Qwen2_5_VLForConditionalGeneration,
    AutoTokenizer,
    AutoProcessor,
    Qwen2VLImageProcessor,
    Trainer
)

# Import custom modules - use try-except to handle relative import issues
try:
    # Try relative imports (when running as a module)
    from .train_qwen import Qwen2_5_VLForMolecule, set_model, safe_save_model_for_hf_trainer
    from ..data.data_grpo import make_grpo_data_module
    from ..model.equiformer_v2_wrapper import EquiformerV2Wrapper
    from .argument import ModelArguments, DataArguments, TrainingArguments
    from .grpo_loss import GRPOLoss
    from .reward_model import EnergyRewardModel
except ImportError:
    # If relative imports fail, try absolute imports (when running directly as a script)
    try:
        from qwenvl.train.train_qwen import Qwen2_5_VLForMolecule, set_model, safe_save_model_for_hf_trainer
        from qwenvl.data.data_grpo import make_grpo_data_module
        from qwenvl.model.equiformer_v2_wrapper import EquiformerV2Wrapper
        from qwenvl.train.argument import ModelArguments, DataArguments, TrainingArguments
        from qwenvl.train.grpo_loss import GRPOLoss
        from qwenvl.train.reward_model import EnergyRewardModel
    except ImportError:
        # Finally try direct imports (via paths added to sys.path)
        current_dir = Path(__file__).parent
        sys.path.insert(0, str(current_dir))
        sys.path.insert(0, str(current_dir.parent / "data"))
        sys.path.insert(0, str(current_dir.parent / "model")) 
        
        from train_qwen import Qwen2_5_VLForMolecule, set_model, safe_save_model_for_hf_trainer
        from data_grpo import make_grpo_data_module
        from equiformer_v2_wrapper import EquiformerV2Wrapper
        from argument import ModelArguments, DataArguments, TrainingArguments
        from grpo_loss import GRPOLoss
        from reward_model import EnergyRewardModel

class GRPOTrainer(Trainer):
    """
    GRPOtrainer
    Inherits from HuggingFace Trainer and implements GRPO-specific training logic
    """
    
    def __init__(
        self,
        model=None,
        reference_model=None,
        grpo_config: Dict = None,
        **kwargs
    ):
        super().__init__(model=model, **kwargs)
        
        # GRPOconfiguration
        self.grpo_config = grpo_config or {}
        self.beta = self.grpo_config.get("beta", 0.1)
        self.label_smoothing = self.grpo_config.get("label_smoothing", 0.0)
        self.reference_free = self.grpo_config.get("reference_free", False)
        
        # Reference model
        self.reference_model = reference_model
        if not self.reference_free and self.reference_model is None:
            raise ValueError("Reference model is required when reference_free=False")
        
        # initializedGRPOloss function
        self.grpo_loss_fn = GRPOLoss(
            beta=self.beta,
            label_smoothing=self.label_smoothing,
            reference_free=self.reference_free
        )
        
        # Set reference model to eval mode and freeze parameters
        if self.reference_model is not None:
            self.reference_model.eval()
            for param in self.reference_model.parameters():
                param.requires_grad = False
    
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        """
        Compute GRPO loss
        
        Args:
            model: policy model
            inputs: input data
            return_outputs: whether to return model outputs
            num_items_in_batch: number of items in batch (compatible with newer transformers)
        """
        # Ensure all tensors are on the correct device
        device = next(model.parameters()).device
        
        # Move all input tensors to the model device
        device_inputs = {}
        for key, value in inputs.items():
            if isinstance(value, torch.Tensor):
                device_inputs[key] = value.to(device)
            else:
                device_inputs[key] = value
        
        try:
            # policy model forward pass
            policy_chosen_logps, policy_rejected_logps, chosen_logits, rejected_logits = \
                self.grpo_loss_fn.concatenated_forward(model, device_inputs)
                
        except Exception as e:
            print(f"❌ Policy model forward pass failed: {e}")
            print(f"Input tensor information:")
            for key, value in device_inputs.items():
                if isinstance(value, torch.Tensor):
                    print(f"  {key}: shape={value.shape}, device={value.device}, dtype={value.dtype}")
                else:
                    print(f"  {key}: {type(value)}")
            raise e
        
        # Reference model forward pass (if needed)
        if not self.reference_free and self.reference_model is not None:
            # Ensure reference model is on the correct device
            if next(self.reference_model.parameters()).device != device:
                self.reference_model = self.reference_model.to(device)
            
            with torch.no_grad():
                ref_chosen_logps, ref_rejected_logps, _, _ = \
                    self.grpo_loss_fn.concatenated_forward(self.reference_model, device_inputs)
        else:
            ref_chosen_logps, ref_rejected_logps = None, None
        
        # Compute GRPO loss
        loss_dict = self.grpo_loss_fn(
            policy_chosen_logps=policy_chosen_logps,
            policy_rejected_logps=policy_rejected_logps,
            reference_chosen_logps=ref_chosen_logps,
            reference_rejected_logps=ref_rejected_logps,
            rewards=device_inputs.get("rewards")
        )
        
        # Log statistics
        if self.state.global_step % self.args.logging_steps == 0:
            self.log({
                "train/grpo_loss": loss_dict["loss"].item(),
                "train/accuracy": loss_dict["accuracy"].item(),
                "train/kl_divergence": loss_dict["kl_divergence"].item(),
                "train/reward_mean": loss_dict["reward_mean"].item(),
                "train/reward_std": loss_dict["reward_std"].item(),
                "train/chosen_logps": loss_dict["chosen_logps"].item(),
                "train/rejected_logps": loss_dict["rejected_logps"].item(),
                "train/pi_logratios": loss_dict["pi_logratios"].item(),
            })
        
        if return_outputs:
            # Construct dummy outputs for compatibility
            class DummyOutputs:
                def __init__(self, logits):
                    self.logits = logits
                    self.loss = loss_dict["loss"]
            
            outputs = DummyOutputs(chosen_logits)
            return loss_dict["loss"], outputs
        else:
            return loss_dict["loss"]
    
    def evaluation_loop(self, dataloader, description, prediction_loss_only=None, ignore_keys=None, metric_key_prefix="eval"):
        """
        Override evaluation loop for GRPO
        """
        # Call parent evaluation loop
        output = super().evaluation_loop(
            dataloader, description, prediction_loss_only, ignore_keys, metric_key_prefix
        )
        
        return output
    
    def save_model(self, output_dir: Optional[str] = None, _internal_call: bool = False):
        """
        Save model; only save policy model
        """
        if output_dir is None:
            output_dir = self.args.output_dir
        
        # Call parent save_model method to avoid recursion
        super().save_model(output_dir, _internal_call)
        
        # Save GRPO configuration
        grpo_config_path = os.path.join(output_dir, "grpo_config.json")
        with open(grpo_config_path, 'w') as f:
            json.dump(self.grpo_config, f, indent=2)
        
        print(f"GRPO model and configuration saved to: {output_dir}")

def create_reference_model(model_args, training_args, attn_implementation="flash_attention_2"):
    """
    Create reference model (frozen copy of the original model)
    """
    print("Creating reference model...")
    
    # Loading model - force use of molecular model class
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
    
    # Correctly handle device assignment in distributed training
    target_dtype = torch.bfloat16 if training_args.bf16 else (torch.float16 if training_args.fp16 else reference_model.dtype)
    
    # In distributed training, use local_rank to determine device
    if torch.distributed.is_initialized():
        device = torch.device(f"cuda:{training_args.local_rank}")
    else:
        device = training_args.device if hasattr(training_args, 'device') else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Move the whole reference model to the correct device
    reference_model = reference_model.to(device=device, dtype=target_dtype)
    
    # Set to eval mode and freeze parameters
    reference_model.eval()
    for param in reference_model.parameters():
        param.requires_grad = False
    
    print(f"Reference model created, device: {device}")
    return reference_model

def train_grpo(attn_implementation="flash_attention_2"):
    """
    GRPOtraining main function
    """
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
    # Force use of our molecular model class, because GRPO needs to handle molecular data
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
    
    # Correctly handle device assignment in distributed training
    target_dtype = torch.bfloat16 if training_args.bf16 else (torch.float16 if training_args.fp16 else policy_model.dtype)
    
    # In distributed training, use local_rank to determine device
    if torch.distributed.is_initialized():
        device = torch.device(f"cuda:{training_args.local_rank}")
    else:
        device = training_args.device if hasattr(training_args, 'device') else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Move the whole policy model to the correct device
    policy_model = policy_model.to(device=device, dtype=target_dtype)
    
    if (not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0):
        print("Policy model surgery complete.")
        print("="*50)
    
    policy_model.config.use_cache = False
    
    # Set policy model parameters
    set_model(model_args, policy_model)
    
    if training_args.gradient_checkpointing:
        policy_model.enable_input_require_grads()
    
    # createReference model
    reference_model = create_reference_model(model_args, training_args, attn_implementation)
    
    # Create tokenizer
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        model_max_length=training_args.model_max_length,
        padding_side="left",  # Flash Attention 2 requires left padding
        use_fast=False,
    )
    
    # Print trainable parameter status
    if (not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0):
        print("="*50)
        print("--- Policy model trainable parameter status ---")
        policy_model.visual.print_trainable_parameters()
        
        # Manually compute and print trainable parameters for the LLM part
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
    
    # Load GRPO preference data
    if not hasattr(data_args, 'preference_data_path') or data_args.preference_data_path is None:
        raise ValueError("preference_data_path must be specified in data_args")
    
    if not os.path.exists(data_args.preference_data_path):
        raise ValueError(f"Preference data file does not exist: {data_args.preference_data_path}")
    
    data_module = make_grpo_data_module(
        tokenizer=tokenizer,
        preference_data_path=data_args.preference_data_path,
        eval_preference_data_path=getattr(data_args, 'eval_preference_data_path', None)
    )
    
    # GRPOconfiguration
    grpo_config = {
        "beta": getattr(training_args, 'grpo_beta', 0.1),
        "label_smoothing": getattr(training_args, 'grpo_label_smoothing', 0.0),
        "reference_free": getattr(training_args, 'grpo_reference_free', False),
    }
    
    # Create GRPO trainer
    trainer = GRPOTrainer(
        model=policy_model,
        reference_model=reference_model,
        grpo_config=grpo_config,
        processing_class=tokenizer,  # use processing_class instead of tokenizer
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
    
    # Correctly close distributed process group to avoid resource-leak warnings
    if torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()

if __name__ == "__main__":
    train_grpo(attn_implementation="flash_attention_2")
