# qwen-vl-finetune/qwenvl/train/ppo_loss.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple
import math

class PPOLoss(nn.Module):
    """
    PPO-style Policy Gradient loss function
    Optimizes directly from task rewards without preference pairs
    """
    
    def __init__(
        self, 
        beta: float = 0.1,
        clip_ratio: float = 0.2,
        value_loss_coef: float = 0.1,
        entropy_coef: float = 0.01,
        baseline_decay: float = 0.99
    ):
        """
        Args:
            beta: KL penalty coefficient (prevents drifting too far from the reference model)
            clip_ratio: PPO clip parameter (if using the clip version)
            value_loss_coef: value-function loss coefficient (if using a critic)
            entropy_coef: entropy regularization coefficient (encourages exploration)
            baseline_decay: baseline moving-average decay rate
        """
        super().__init__()
        self.beta = beta
        self.clip_ratio = clip_ratio
        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef
        self.baseline_decay = baseline_decay
        
        # Baseline (moving average)
        self.register_buffer('baseline', torch.tensor(0.0))
        self.register_buffer('baseline_count', torch.tensor(0))
    
    def get_log_probs(
        self, 
        logits: torch.FloatTensor, 
        labels: torch.LongTensor
    ) -> torch.FloatTensor:
        """
        Compute log probabilities of generated sequences
        
        Args:
            logits: [batch_size, seq_len, vocab_size]
            labels: [batch_size, seq_len]
            
        Returns:
            log_probs: [batch_size] eachsequence total log probability
        """
        # Shift for next token prediction
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        
        # Flatten
        shift_logits = shift_logits.view(-1, shift_logits.size(-1))
        shift_labels = shift_labels.view(-1)
        
        # Compute log probabilities
        log_probs = F.log_softmax(shift_logits, dim=-1)
        
        # Handle device consistency and index range
        shift_labels = shift_labels.to(log_probs.device)
        vocab_size = log_probs.size(-1)
        valid_mask = (shift_labels >= 0) & (shift_labels < vocab_size)
        
        # Safe indexing
        safe_labels = torch.where(valid_mask, shift_labels, torch.zeros_like(shift_labels))
        selected_log_probs = log_probs.gather(dim=-1, index=safe_labels.unsqueeze(-1)).squeeze(-1)
        
        # Reshape and apply mask
        selected_log_probs = selected_log_probs.view(labels.shape[0], -1)
        mask = valid_mask.view(labels.shape[0], -1).float()
        selected_log_probs = selected_log_probs * mask
        
        # Sum log probabilities for each sequence
        sequence_log_probs = selected_log_probs.sum(dim=-1)
        
        return sequence_log_probs
    
    def compute_entropy(self, logits: torch.FloatTensor) -> torch.FloatTensor:
        """Compute entropy (to encourage exploration)"""
        probs = F.softmax(logits, dim=-1)
        log_probs = F.log_softmax(logits, dim=-1)
        entropy = -(probs * log_probs).sum(dim=-1).mean()
        return entropy
    
    def update_baseline(self, rewards: torch.FloatTensor):
        """Update baseline (moving average)"""
        with torch.no_grad():
            current_mean = rewards.mean()
            self.baseline_count += 1
            
            # moving average
            self.baseline = (
                self.baseline_decay * self.baseline + 
                (1 - self.baseline_decay) * current_mean
            )
    
    def forward(
        self,
        policy_log_probs: torch.FloatTensor,
        reference_log_probs: Optional[torch.FloatTensor],
        rewards: torch.FloatTensor,
        logits: Optional[torch.FloatTensor] = None
    ) -> Dict[str, torch.FloatTensor]:
        """
        Compute PPO-style Policy Gradient loss
        
        Args:
            policy_log_probs: policy-model log probabilities [batch_size]
            reference_log_probs: reference-model log probabilities [batch_size] (optional)
            rewards: task rewards [batch_size]
            logits: model logits (for computing entropy) (optional)
            
        Returns:
            dictionary containing loss and statistics
        """
        batch_size = policy_log_probs.shape[0]
        
        # 1. Update baseline
        self.update_baseline(rewards)
        
        # 2. Compute advantage
        advantages = rewards - self.baseline
        
        # 3. Compute KL divergence (if there is a reference model)
        if reference_log_probs is not None:
            kl_divergence = policy_log_probs - reference_log_probs
            
            # Reward after KL penalty
            penalized_rewards = rewards - self.beta * kl_divergence
            penalized_advantages = penalized_rewards - self.baseline
        else:
            kl_divergence = torch.zeros_like(rewards)
            penalized_advantages = advantages
        
        # 4. Policy Gradientloss
        # Use the sign of advantage to guide optimization direction
        # advantage > 0: this action is good; increase probability (reduce negative loss)
        # advantage < 0: this action is bad; reduce probability (increase negative loss)
        pg_loss = -(penalized_advantages.detach() * policy_log_probs).mean()
        
        # 5. Entropy regularization (encourage exploration)
        if logits is not None:
            entropy = self.compute_entropy(logits)
            entropy_loss = -self.entropy_coef * entropy
        else:
            entropy = torch.tensor(0.0, device=pg_loss.device)
            entropy_loss = torch.tensor(0.0, device=pg_loss.device)
        
        # 6. Total loss
        total_loss = pg_loss + entropy_loss
        
        # 7. statisticsinformation
        with torch.no_grad():
            # proportion of positive rewards
            positive_reward_rate = (rewards > 0).float().mean()
            
            # proportion of high-quality predictions (reward>=5)
            high_quality_rate = (rewards >= 5.0).float().mean()
            
            # advantage statistics
            positive_advantage_rate = (advantages > 0).float().mean()
        
        return {
            "loss": total_loss,
            "pg_loss": pg_loss,
            "entropy_loss": entropy_loss,
            "entropy": entropy,
            "kl_divergence": kl_divergence.mean(),
            "reward_mean": rewards.mean(),
            "reward_std": rewards.std() if batch_size > 1 else torch.tensor(0.0, device=rewards.device),
            "advantage_mean": advantages.mean(),
            "advantage_std": advantages.std() if batch_size > 1 else torch.tensor(0.0, device=advantages.device),
            "baseline": self.baseline,
            "positive_reward_rate": positive_reward_rate,
            "high_quality_rate": high_quality_rate,
            "positive_advantage_rate": positive_advantage_rate,
            "policy_log_probs": policy_log_probs.mean(),
        }


class PPOClipLoss(PPOLoss):
    """
    PPO loss with clipping (more conservative version)
    """
    
    def forward(
        self,
        policy_log_probs: torch.FloatTensor,
        old_log_probs: torch.FloatTensor,
        reference_log_probs: Optional[torch.FloatTensor],
        rewards: torch.FloatTensor,
        logits: Optional[torch.FloatTensor] = None
    ) -> Dict[str, torch.FloatTensor]:
        """
        Use PPO clipping
        
        Args:
            policy_log_probs: current policy log probabilities
            old_log_probs: old policy log probabilities (for computing ratio)
            reference_log_probs: reference model log probabilities (for KL penalty)
            rewards: task rewards
            logits: modellogits
        """
        batch_size = policy_log_probs.shape[0]
        
        # Update baseline
        self.update_baseline(rewards)
        
        # Compute advantage
        advantages = rewards - self.baseline
        
        # KL penalty
        if reference_log_probs is not None:
            kl_divergence = policy_log_probs - reference_log_probs
            penalized_advantages = advantages - self.beta * kl_divergence
        else:
            kl_divergence = torch.zeros_like(rewards)
            penalized_advantages = advantages
        
        # Compute probability ratio
        ratio = torch.exp(policy_log_probs - old_log_probs)
        
        # PPO clip
        clipped_ratio = torch.clamp(ratio, 1 - self.clip_ratio, 1 + self.clip_ratio)
        
        # Take the minimum of the two (conservative update)
        pg_loss1 = -penalized_advantages.detach() * ratio
        pg_loss2 = -penalized_advantages.detach() * clipped_ratio
        pg_loss = torch.max(pg_loss1, pg_loss2).mean()
        
        # Entropy loss
        if logits is not None:
            entropy = self.compute_entropy(logits)
            entropy_loss = -self.entropy_coef * entropy
        else:
            entropy = torch.tensor(0.0, device=pg_loss.device)
            entropy_loss = torch.tensor(0.0, device=pg_loss.device)
        
        total_loss = pg_loss + entropy_loss
        
        # statisticsinformation
        with torch.no_grad():
            # Clip rate (proportion where ratio is clipped)
            clip_rate = ((ratio < 1 - self.clip_ratio) | (ratio > 1 + self.clip_ratio)).float().mean()
            positive_reward_rate = (rewards > 0).float().mean()
            high_quality_rate = (rewards >= 5.0).float().mean()
        
        return {
            "loss": total_loss,
            "pg_loss": pg_loss,
            "entropy_loss": entropy_loss,
            "entropy": entropy,
            "kl_divergence": kl_divergence.mean(),
            "reward_mean": rewards.mean(),
            "reward_std": rewards.std() if batch_size > 1 else torch.tensor(0.0, device=rewards.device),
            "advantage_mean": penalized_advantages.mean(),
            "baseline": self.baseline,
            "ratio_mean": ratio.mean(),
            "clip_rate": clip_rate,
            "positive_reward_rate": positive_reward_rate,
            "high_quality_rate": high_quality_rate,
            "policy_log_probs": policy_log_probs.mean(),
        }




