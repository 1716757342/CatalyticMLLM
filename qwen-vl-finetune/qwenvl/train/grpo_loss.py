# qwen-vl-finetune/qwenvl/train/grpo_loss.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple, Any
import math

class GRPOLoss(nn.Module):
    """
    GRPO (Generalized Relative Preference Optimization) loss function
    Based on DPO principles and optimized for text generation tasks
    """
    
    def __init__(
        self, 
        beta: float = 0.1,
        label_smoothing: float = 0.0,
        reference_free: bool = False
    ):
        """
        Args:
            beta: temperature parameter controlling deviation between policy and reference model
            label_smoothing: label-smoothing parameter
            reference_free: whether to use the reference-free variant
        """
        super().__init__()
        self.beta = beta
        self.label_smoothing = label_smoothing
        self.reference_free = reference_free
    
    def get_batch_logps(
        self, 
        logits: torch.FloatTensor, 
        labels: torch.LongTensor,
        average_log_prob: bool = False
    ) -> torch.FloatTensor:
        """
        Compute batch log probabilities
        
        Args:
            logits: model output logits [batch_size, seq_len, vocab_size]
            labels: target tokens [batch_size, seq_len]
            average_log_prob: whether to compute average log probability
            
        Returns:
            log probability for each sequence [batch_size]
        """
        if logits.shape[:-1] != labels.shape:
            raise ValueError(
                f"Logits shape {logits.shape} and labels shape {labels.shape} do not match"
            )
        
        # Shift logits and labels for next token prediction
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        
        # Flatten for loss computation
        shift_logits = shift_logits.view(-1, shift_logits.size(-1))
        shift_labels = shift_labels.view(-1)
        
        # Compute log probabilities
        log_probs = F.log_softmax(shift_logits, dim=-1)
        
        # Ensure device consistency and avoid index out of bounds
        shift_labels = shift_labels.to(log_probs.device)
        
        # Filter invalid indices (-100oroutside vocab range)
        vocab_size = log_probs.size(-1)
        valid_mask = (shift_labels >= 0) & (shift_labels < vocab_size)
        
        # Set invalid indices to 0 and handle them later with a mask
        safe_labels = torch.where(valid_mask, shift_labels, torch.zeros_like(shift_labels))
        
        # Select the log probabilities of the target tokens
        selected_log_probs = log_probs.gather(dim=-1, index=safe_labels.unsqueeze(-1)).squeeze(-1)
        
        # Reshape back to [batch_size, seq_len-1]
        selected_log_probs = selected_log_probs.view(labels.shape[0], -1)
        
        # Combined mask: exclude padding tokens and invalid indices
        mask = valid_mask.view(labels.shape[0], -1).float()
        selected_log_probs = selected_log_probs * mask
        
        # Sum log probabilities for each sequence
        sequence_log_probs = selected_log_probs.sum(dim=-1)
        
        if average_log_prob:
            # Average over valid tokens
            valid_tokens = mask.sum(dim=-1).float()
            sequence_log_probs = sequence_log_probs / (valid_tokens + 1e-8)
        
        return sequence_log_probs
    
    def concatenated_forward(
        self, 
        model: nn.Module, 
        batch: Dict[str, Any]
    ) -> Tuple[torch.FloatTensor, torch.FloatTensor, torch.FloatTensor, torch.FloatTensor]:
        """
        Handle forward pass for both chosen and rejected data
        
        Args:
            model: policy model
            batch: batch containing chosen and rejected data
            
        Returns:
            (chosen_logps, rejected_logps, chosen_logits, rejected_logits)
        """
        # Ensure all tensors are on the correct device
        device = next(model.parameters()).device
        
        # Get chosen data and move it to the correct device
        chosen_inputs = {}
        for key in ["input_ids", "attention_mask", "atomic_numbers", "coordinates", "molecule_mask", "cell", "labels"]:
            batch_key = f"chosen_{key}" if key != "labels" else "chosen_labels"
            if batch_key in batch and batch[batch_key] is not None:
                if isinstance(batch[batch_key], torch.Tensor):
                    chosen_inputs[key] = batch[batch_key].to(device)
                else:
                    chosen_inputs[key] = batch[batch_key]
        
        # Get rejected data and move it to the correct device  
        rejected_inputs = {}
        for key in ["input_ids", "attention_mask", "atomic_numbers", "coordinates", "molecule_mask", "cell", "labels"]:
            batch_key = f"rejected_{key}" if key != "labels" else "rejected_labels"
            if batch_key in batch and batch[batch_key] is not None:
                if isinstance(batch[batch_key], torch.Tensor):
                    rejected_inputs[key] = batch[batch_key].to(device)
                else:
                    rejected_inputs[key] = batch[batch_key]
        
        # Forward pass
        chosen_outputs = model(**chosen_inputs)
        rejected_outputs = model(**rejected_inputs)
        
        # Compute log probabilities - ensure labels are also on the correct device
        chosen_labels = batch["chosen_labels"].to(device)
        rejected_labels = batch["rejected_labels"].to(device)
        
        chosen_logps = self.get_batch_logps(chosen_outputs.logits, chosen_labels)
        rejected_logps = self.get_batch_logps(rejected_outputs.logits, rejected_labels)
        
        return chosen_logps, rejected_logps, chosen_outputs.logits, rejected_outputs.logits
    
    def forward(
        self,
        policy_chosen_logps: torch.FloatTensor,
        policy_rejected_logps: torch.FloatTensor,
        reference_chosen_logps: Optional[torch.FloatTensor] = None,
        reference_rejected_logps: Optional[torch.FloatTensor] = None,
        rewards: Optional[torch.FloatTensor] = None
    ) -> Dict[str, torch.FloatTensor]:
        """
        Compute GRPO loss
        
        Args:
            policy_chosen_logps: policy-model log probability for chosen sequences
            policy_rejected_logps: policy-model log probability for rejected sequences
            reference_chosen_logps: reference-model log probability for chosen sequences (optional)
            reference_rejected_logps: reference-model log probability for rejected sequences (optional)
            rewards: reward differences (optional)
            
        Returns:
            dictionary containing loss and statistics
        """
        if self.reference_free:
            # reference-free version
            pi_logratios = policy_chosen_logps - policy_rejected_logps
            ref_logratios = 0.0
        else:
            # standard version requiring a reference model
            if reference_chosen_logps is None or reference_rejected_logps is None:
                raise ValueError("Reference logps are required when reference_free=False")
            
            # Compute log ratio of policy model relative to reference model
            pi_logratios = (policy_chosen_logps - policy_rejected_logps) - (reference_chosen_logps - reference_rejected_logps)
            ref_logratios = reference_chosen_logps - reference_rejected_logps
        
        # GRPOlosscompute
        if self.label_smoothing == 0.0:
            # standard DPO loss
            losses = -F.logsigmoid(self.beta * pi_logratios)
        else:
            # loss with label smoothing
            losses = (
                -F.logsigmoid(self.beta * pi_logratios) * (1 - self.label_smoothing)
                - F.logsigmoid(-self.beta * pi_logratios) * self.label_smoothing
            )
        
        # Average loss
        loss = losses.mean()
        
        # Compute statistics
        with torch.no_grad():
            # accuracy:proportion where chosen sequence probability is higher than rejected sequence probability
            accuracy = (pi_logratios > 0).float().mean()
            
            # KL-divergence estimate
            if not self.reference_free:
                chosen_kl = policy_chosen_logps - reference_chosen_logps
                rejected_kl = policy_rejected_logps - reference_rejected_logps
                kl_mean = (chosen_kl + rejected_kl).mean() / 2.0
            else:
                kl_mean = torch.tensor(0.0, device=loss.device)
            
            # Reward statistics
            if rewards is not None:
                reward_mean = rewards.mean()
                # Fix standard-deviation computation warning:when sample count <= 1, use 0 instead of std()
                if rewards.numel() > 1:
                    reward_std = rewards.std()
                else:
                    reward_std = torch.tensor(0.0, device=loss.device, dtype=rewards.dtype)
            else:
                reward_mean = torch.tensor(0.0, device=loss.device)
                reward_std = torch.tensor(0.0, device=loss.device)
        
        return {
            "loss": loss,
            "accuracy": accuracy,
            "kl_divergence": kl_mean,
            "reward_mean": reward_mean,
            "reward_std": reward_std,
            "chosen_logps": policy_chosen_logps.mean(),
            "rejected_logps": policy_rejected_logps.mean(),
            "pi_logratios": pi_logratios.mean()
        }

class GRPOTrainerMixin:
    """
    GRPOtrainer mixin class providing GRPO-related training functionality
    """
    
    def __init__(self, *args, grpo_config: Dict[str, Any] = None, **kwargs):
        super().__init__(*args, **kwargs)
        
        # GRPOconfiguration
        self.grpo_config = grpo_config or {}
        self.beta = self.grpo_config.get("beta", 0.1)
        self.label_smoothing = self.grpo_config.get("label_smoothing", 0.0)
        self.reference_free = self.grpo_config.get("reference_free", False)
        
        # initializedGRPOloss function
        self.grpo_loss_fn = GRPOLoss(
            beta=self.beta,
            label_smoothing=self.label_smoothing,
            reference_free=self.reference_free
        )
        
        # reference model (if needed)
        self.reference_model = None
        if not self.reference_free:
            self.reference_model = self.grpo_config.get("reference_model")
    
    def compute_grpo_loss(self, model: nn.Module, inputs: Dict[str, Any]) -> Dict[str, torch.FloatTensor]:
        """
        Compute GRPO loss
        
        Args:
            model: policy model
            inputs: inputbatchdata
            
        Returns:
            loss and statistics information
        """
        # policy model forward pass
        policy_chosen_logps, policy_rejected_logps, _, _ = self.grpo_loss_fn.concatenated_forward(model, inputs)
        
        # Reference model forward pass (if needed)
        if not self.reference_free and self.reference_model is not None:
            with torch.no_grad():
                ref_chosen_logps, ref_rejected_logps, _, _ = self.grpo_loss_fn.concatenated_forward(
                    self.reference_model, inputs
                )
        else:
            ref_chosen_logps, ref_rejected_logps = None, None
        
        # computeloss
        loss_dict = self.grpo_loss_fn(
            policy_chosen_logps=policy_chosen_logps,
            policy_rejected_logps=policy_rejected_logps,
            reference_chosen_logps=ref_chosen_logps,
            reference_rejected_logps=ref_rejected_logps,
            rewards=inputs.get("rewards")
        )
        
        return loss_dict
    
    def compute_loss(self, model: nn.Module, inputs: Dict[str, Any], return_outputs: bool = False):
        """
        Override compute_loss to use GRPO loss
        """
        loss_dict = self.compute_grpo_loss(model, inputs)
        
        # Log statistics
        if hasattr(self, 'log'):
            self.log({
                "grpo/loss": loss_dict["loss"].item(),
                "grpo/accuracy": loss_dict["accuracy"].item(),
                "grpo/kl_divergence": loss_dict["kl_divergence"].item(),
                "grpo/reward_mean": loss_dict["reward_mean"].item(),
                "grpo/reward_std": loss_dict["reward_std"].item(),
                "grpo/chosen_logps": loss_dict["chosen_logps"].item(),
                "grpo/rejected_logps": loss_dict["rejected_logps"].item(),
            })
        
        if return_outputs:
            return loss_dict["loss"], loss_dict
        else:
            return loss_dict["loss"]
