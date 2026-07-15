#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Real GRPO (Group Relative Policy Optimization) trainer
Online sampling + group-relative reward optimization
"""

import os
import sys
import json
import gc
import torch
import torch.nn as nn
import transformers
from typing import Dict, List, Optional
from dataclasses import dataclass, field
from pathlib import Path
import logging
from datetime import datetime

# Add path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from transformers import (
    Trainer,
    TrainingArguments,
    AutoTokenizer,
    HfArgumentParser,
)

try:
    from qwenvl.train.train_qwen import Qwen2_5_VLForMolecule
    from qwenvl.train.reward_model_cif import CIFRewardModel
    from qwenvl.train.argument import ModelArguments, DataArguments
    from qwenvl.data.grpo_dataset import GRPODataset, GRPODataCollator
except ImportError as e:
    print(f"Failed to import modules: {e}")
    print("Please check paths")
    sys.exit(1)

logger = logging.getLogger(__name__)


@dataclass
class GRPOArguments(TrainingArguments):
    """GRPO training arguments"""
    grpo_beta: float = field(default=0.1, metadata={"help": "KL penalty coefficient"})
    num_samples_per_prompt: int = field(default=4, metadata={"help": "number of responses sampled per prompt"})
    temperature: float = field(default=0.7, metadata={"help": "sampling temperature"})
    top_p: float = field(default=0.9, metadata={"help": "nucleus sampling"})
    max_new_tokens: int = field(default=4096, metadata={"help": "maximum generation length"})
    model_max_length: int = field(default=4096, metadata={"help": "maximum sequence length"})


class OnlineGRPOTrainer(Trainer):
    """
    Online GRPO trainer
    
    Workflow:
    1. Sample K responses for each prompt
    2. Score with the reward model
    3. Compute within-group relative advantages
    4. Optimize policy
    """
    
    def __init__(
        self,
        model=None,
        grpo_args: GRPOArguments = None,
        reward_model: CIFRewardModel = None,
        **kwargs
    ):
        super().__init__(model=model, **kwargs)
        
        self.grpo_args = grpo_args
        self.reward_model = reward_model or CIFRewardModel()
        
        # Sampling parameters
        self.num_samples = grpo_args.num_samples_per_prompt
        self.temperature = grpo_args.temperature
        self.top_p = grpo_args.top_p
        self.max_new_tokens = grpo_args.max_new_tokens
        self.beta = grpo_args.grpo_beta
        
        # Create directory for sampling results
        self.samples_dir = Path(grpo_args.output_dir) / "grpo_samples"
        self.samples_dir.mkdir(parents=True, exist_ok=True)
        self.global_step_counter = 0
        
        logger.info(f"GRPO configuration: samples={self.num_samples}, temp={self.temperature}, beta={self.beta}")
        logger.info(f"Sampling results directory: {self.samples_dir}")
    
    def _save_sample_result(
        self,
        step: int,
        prompt_idx: int,
        sample_idx: int,
        prompt: str,
        expected_atoms: Dict,
        actual_atoms: Dict,
        generated_cif: str,
        reward: float,
        sub_scores: Dict,
        reward_details: Dict,
        gen_time: float,
        gen_length: int
    ):
        """Save one sampling result to file"""
        
        # Create filename: step_prompt_sample.txt
        filename = f"step_{step:04d}_prompt_{prompt_idx}_sample_{sample_idx}.txt"
        filepath = self.samples_dir / filename
        
        # Prepare content to save
        content = []
        content.append("=" * 80)
        content.append(f"GRPO sampling result - Step {step} | Prompt {prompt_idx} | Sample {sample_idx}")
        content.append(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        content.append("=" * 80)
        content.append("")
        
        # Prompt information
        content.append("[Prompt]")
        content.append(prompt)
        content.append("")
        
        # expected atomic composition
        content.append("[expected atomic composition]")
        if expected_atoms:
            for element, count in sorted(expected_atoms.items()):
                content.append(f"  {element}: {count}")
            content.append(f"  Total: {sum(expected_atoms.values())} atoms")
        else:
            content.append("  (none)")
        content.append("")
        
        # actualgenerateofatomic composition
        content.append("[generateofatomic composition]")
        if actual_atoms:
            for element, count in sorted(actual_atoms.items()):
                content.append(f"  {element}: {count}")
            content.append(f"  Total: {sum(actual_atoms.values())} atoms")
        else:
            content.append("  (unable to parse)")
        content.append("")
        
        # Comparison
        content.append("[Atomic composition comparison]")
        if expected_atoms and actual_atoms:
            all_elements = set(expected_atoms.keys()) | set(actual_atoms.keys())
            for element in sorted(all_elements):
                exp = expected_atoms.get(element, 0)
                act = actual_atoms.get(element, 0)
                diff = act - exp
                status = "✓" if diff == 0 else "✗"
                content.append(f"  {status} {element}: expected={exp}, actual={act}, diff={diff:+d}")
        content.append("")
        
        # Reward scoring
        content.append("[Reward scoring]")
        content.append(f"  Total score: {reward:+.2f}")
        content.append(f"  atomic composition: {sub_scores.get('atom_composition', 0):+.2f} (weight60%)")
        content.append(f"  parseability: {sub_scores.get('parseability', 0):+.2f} (weight20%)")
        content.append(f"  structure validity: {sub_scores.get('structure_validity', 0):+.2f} (weight10%)")
        content.append(f"  physical plausibility: {sub_scores.get('physical', 0):+.2f} (weight10%)")
        content.append("")
        
        # Reward details
        if reward_details:
            composition_info = reward_details.get("composition", {})
            if composition_info:
                content.append("[Atomic composition details]")
                content.append(f"  Match type: {composition_info.get('match_type', 'unknown')}")
                if composition_info.get("missing"):
                    content.append(f"  Missing elements: {', '.join(composition_info['missing'])}")
                if composition_info.get("extra"):
                    content.append(f"  Extra elements: {', '.join(composition_info['extra'])}")
                if composition_info.get("avg_error") is not None:
                    content.append(f"  Average relative error: {composition_info['avg_error']*100:.1f}%")
                content.append("")
        
        # Generation statistics
        content.append("[Generation statistics]")
        content.append(f"  Generation time: {gen_time:.1f} seconds")
        content.append(f"  Generated length: {gen_length} tokens")
        content.append(f"  Character count: {len(generated_cif)}")
        content.append("")
        
        # Generated CIF content
        content.append("[Generated CIF content]")
        content.append("-" * 80)
        content.append(generated_cif)
        content.append("-" * 80)
        content.append("")
        
        # Write file
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write('\n'.join(content))
    
    def sample_responses(
        self, 
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        prompts: List[str], 
        expected_atoms_list: List[Dict]
    ) -> tuple:
        """
        Sample multiple responses for each prompt
        
        Args:
            input_ids: tokenized prompts [batch_size, seq_len]
            attention_mask: attention mask
            prompts: original prompt text
            expected_atoms_list: expected atomic composition
        
        Returns:
            responses: List[List[str]] - multiple responses for each prompt
            rewards: List[List[float]] - corresponding rewards
            log_probs: List[List[float]] - corresponding log probabilities
        """
        all_responses = []
        all_rewards = []
        all_log_probs = []
        
        # Get original model (remove DDP wrapper)
        model_for_generation = self.model.module if hasattr(self.model, 'module') else self.model
        model_for_generation.eval()
        
        # Get current process rank (for controlling log output)
        import torch.distributed as dist
        rank = dist.get_rank() if dist.is_initialized() else 0
        
        batch_size = input_ids.shape[0]
        
        # Sampling phase does not need gradients, only no_grad
        with torch.no_grad():
            for idx in range(batch_size):
                prompt = prompts[idx]
                expected_atoms = expected_atoms_list[idx]
                
                # input_ids for current prompt
                prompt_input_ids = input_ids[idx:idx+1]  # [1, seq_len]
                prompt_attention_mask = attention_mask[idx:idx+1]
                
                # Sample K responses for each prompt (batch-parallel generation)
                responses = []
                rewards = []
                log_probs = []
                
                try:
                    import time
                    batch_start_time = time.time()
                    
                    # 🚀 Batch-parallel sampling optimization:Copy prompt num_samples times and generate all responses at once
                    # [1, seq_len] -> [num_samples, seq_len]
                    batch_input_ids = prompt_input_ids.repeat(self.num_samples, 1)
                    batch_attention_mask = prompt_attention_mask.repeat(self.num_samples, 1)
                    
                    if rank == 0:
                        logger.info(f"  🚀 Batch-generate {self.num_samples} responses...")
                    
                    # Generate all responses at once (batch parallel)
                    batch_output_ids = model_for_generation.generate(
                        input_ids=batch_input_ids,
                        attention_mask=batch_attention_mask,
                        max_new_tokens=self.max_new_tokens,
                        temperature=self.temperature,
                        top_p=self.top_p,
                        do_sample=True,
                        pad_token_id=self.processing_class.pad_token_id,
                        eos_token_id=self.processing_class.eos_token_id,
                        max_time=1800.0,  # 30minute timeout
                    )
                    
                    batch_gen_time = time.time() - batch_start_time
                    
                    if rank == 0:
                        avg_time_per_sample = batch_gen_time / self.num_samples
                        logger.info(f"  ⚡ Batch generation complete: total time={batch_gen_time:.1f}s | average={avg_time_per_sample:.1f}s/sample")
                    
                    # Process each generated response
                    for sample_idx in range(self.num_samples):
                        try:
                            # Decode current response (only generated part)
                            generated_ids = batch_output_ids[sample_idx, prompt_input_ids.shape[1]:]
                            response = self.processing_class.decode(generated_ids, skip_special_tokens=True)
                            
                            gen_length = len(generated_ids)
                            
                            # Compute reward (get details)
                            reward_result = self.reward_model.compute_single_reward(
                                response, expected_atoms, return_details=True
                            )
                            reward = reward_result["total_reward"]
                            sub_scores = reward_result["sub_scores"]
                            
                            # Extract actual generated atom information
                            actual_atoms = reward_result.get("details", {}).get("composition", {}).get("actual", {})
                            
                            # Temporarily use placeholder log probability
                            log_prob = -1.0
                            
                            responses.append(response)
                            rewards.append(reward)
                            log_probs.append(log_prob)
                            
                            # Save sampling result to file (only on main process)
                            if rank == 0:
                                self._save_sample_result(
                                    step=self.global_step_counter,
                                    prompt_idx=idx,
                                    sample_idx=sample_idx,
                                    prompt=prompt,
                                    expected_atoms=expected_atoms,
                                    actual_atoms=actual_atoms,
                                    generated_cif=response,
                                    reward=reward,
                                    sub_scores=sub_scores,
                                    reward_details=reward_result.get("details", {}),
                                    gen_time=batch_gen_time / self.num_samples,  # average time
                                    gen_length=gen_length
                                )
                            
                            # Log detailed reward information (only on main process)
                            if rank == 0:
                                logger.info(
                                    f"  └─ Sample {sample_idx+1}: Total score={reward:+.2f} | "
                                    f"atom={sub_scores.get('atom_composition', 0):+.1f} "
                                    f"parse={sub_scores.get('parseability', 0):+.1f} "
                                    f"structure={sub_scores.get('structure_validity', 0):+.1f} "
                                    f"physical={sub_scores.get('physical', 0):+.1f} "
                                    f"({gen_length} tokens)"
                                )
                            
                            # Immediately release the current response tensor
                            del generated_ids
                            
                        except Exception as e:
                            if rank == 0:
                                logger.error(f"  └─ Sample {sample_idx+1}: processing failed - {e}")
                            # Use default values
                            responses.append("")
                            rewards.append(-10.0)
                            log_probs.append(-1.0)
                    
                    # Release all tensors from batch generation
                    del batch_output_ids, batch_input_ids, batch_attention_mask
                    
                except Exception as e:
                    if rank == 0:
                        logger.error(f"  ❌ Batch generation failed: {e}")
                        logger.error(f"  Falling back to default values...")
                    # Batch generation failed; use default values
                    for sample_idx in range(self.num_samples):
                        responses.append("")
                        rewards.append(-10.0)
                        log_probs.append(-1.0)
                
                # Clear GPU memory once after batch generation (instead of after each sample)
                torch.cuda.empty_cache()
                gc.collect()
                
                all_responses.append(responses)
                all_rewards.append(rewards)
                all_log_probs.append(log_probs)
                
                # Clean once after each prompt
                torch.cuda.empty_cache()
                gc.collect()
        
        model_for_generation.train()
        self.model.train()
        
        # Clear sampling-phase cache to release GPU memory
        torch.cuda.empty_cache()
        gc.collect()
        
        return all_responses, all_rewards, all_log_probs
    
    def compute_grpo_loss(
        self,
        prompts: List[str],
        responses: List[List[str]],
        rewards: List[List[float]]
    ) -> torch.Tensor:
        """
        compute GRPO loss
        
        Use within-group relative advantages:
        advantage_i = reward_i - mean(rewards_in_group)
        loss = -sum(advantage_i * log_prob_i)
        """
        total_loss = 0.0
        num_examples = 0
        
        for prompt, response_group, reward_group in zip(prompts, responses, rewards):
            # Compute group mean reward
            mean_reward = sum(reward_group) / len(reward_group)
            
            # Compute advantage for each response
            advantages = [r - mean_reward for r in reward_group]
            
            # Compute log probability for each response
            for response, advantage in zip(response_group, advantages):
                # TODO: implement actual log-probability computation
                # log_prob = self.model.compute_log_prob(prompt, response)
                log_prob = torch.tensor(0.0)  # placeholder
                
                # GRPO loss: -advantage * log_prob
                loss = -advantage * log_prob
                total_loss += loss
                num_examples += 1
        
        return total_loss / num_examples if num_examples > 0 else torch.tensor(0.0)
    
    def training_step(self, model, inputs, num_items_in_batch=None):
        """
        GRPO training step - real online training
        
        Step:
        1. Sample K responses for each prompt
        2. Score with the reward model
        3. Compute within-group relative advantages
        4. computepolicy-gradient loss and update
        """
        # Extract data
        input_ids = inputs.get("input_ids")
        attention_mask = inputs.get("attention_mask")
        prompts = inputs.get("prompts", [])
        expected_atoms_list = inputs.get("expected_atoms_list", [])
        
        # Step 1: Online sampling of multiple responses (each GPU samples independently, unsynchronized)
        import torch.distributed as dist
        rank = dist.get_rank() if dist.is_initialized() else 0
        
        if rank == 0:  # Print only on main process
            logger.info(f"\n{'='*70}")
            logger.info(f"📊 Batch {len(prompts)} | sampling {self.num_samples}x{len(prompts)}={self.num_samples*len(prompts)} responses...")
            logger.info(f"{'='*70}")
        
        all_responses, all_rewards, all_log_probs = self.sample_responses(
            input_ids=input_ids,
            attention_mask=attention_mask,
            prompts=prompts,
            expected_atoms_list=expected_atoms_list
        )
        
        # Step 2: Compute the real GRPO loss (policy gradient)
        
        device = input_ids.device
        
        # Get the actual model (handle DataParallel wrapper)
        if hasattr(model, 'module'):
            actual_model = model.module
        else:
            actual_model = model
        
        total_loss = torch.tensor(0.0, device=device, requires_grad=True)
        num_valid_samples = 0
        
        for batch_idx in range(len(prompts)):
            prompt = prompts[batch_idx]
            responses = all_responses[batch_idx]
            rewards = all_rewards[batch_idx]
            
            # Compute group mean reward (baseline)
            mean_reward = sum(rewards) / len(rewards) if rewards else 0.0
            
            # Compute reward standard deviation (for normalization)
            if len(rewards) > 1:
                reward_std = (sum((r - mean_reward)**2 for r in rewards) / len(rewards)) ** 0.5
                reward_std = max(reward_std, 1e-8)  # prevent division by zero
            else:
                reward_std = 1.0
            
            # Compute policy gradient for each response
            for response, reward in zip(responses, rewards):
                if not response:  # Skip empty responses
                    continue
                
                try:
                    # Compute advantage (normalized)
                    advantage = (reward - mean_reward) / reward_std
                    
                    # Build full input: prompt + response
                    full_text = prompt + " " + response
                    
                    # Tokenize (limit maximum length to avoid OOM)
                    # Sequence length = prompt + response, so max_new_tokens plus prompt reserve is needed
                    # Policy:
                    # 1. Prefer full length max_new_tokens + 1024
                    # 2. if too long (>10K), limit to a reasonable range
                    # 3. Use chunked log_softmax to save GPU memory (implemented)
                    # 4. [Important]Training must save gradients; memory use is much larger than inference, so length must be strictly limited
                    desired_max_length = self.max_new_tokens + 1024
                    # To avoid OOM, set a practical upper bound (based on available GPU memory)
                    # Training memory (gradients + activations) >> inference memory; limit to within 4096
                    # If max_new_tokens is large (e.g. 9120), limit to 4096 (training-safe upper bound)
                    # otherwise use full length
                    safe_max_length = min(desired_max_length, 4096) if desired_max_length > 5000 else desired_max_length
                    
                    inputs = self.processing_class(
                        full_text,
                        return_tensors="pt",
                        truncation=True,
                        max_length=safe_max_length
                    )
                    
                    input_ids_full = inputs["input_ids"].to(device)
                    attention_mask_full = inputs["attention_mask"].to(device)
                    
                    # Forward pass to compute logits
                    with torch.set_grad_enabled(True):
                        outputs = actual_model(
                            input_ids=input_ids_full,
                            attention_mask=attention_mask_full,
                        )
                        logits = outputs.logits  # [1, seq_len, vocab_size]
                        
                        # Immediately release other contents in outputs to save GPU memory
                        del outputs
                    
                    # Compute prompt length
                    prompt_inputs = self.processing_class(
                        prompt,
                        return_tensors="pt",
                        add_special_tokens=False
                    )
                    prompt_len = prompt_inputs["input_ids"].shape[1]
                    
                    # Only compute log probability for the response part
                    # logits: [1, seq_len, vocab_size]
                    # We need logits[prompt_len-1 : -1] corresponding log probs
                    response_logits = logits[0, prompt_len-1:-1, :]  # [response_len, vocab_size]
                    response_token_ids = input_ids_full[0, prompt_len:]  # [response_len]
                    
                    # Compute log probabilities in chunks to save GPU memory
                    # vocab_size is very large (151936), computing all at once will exhaust GPU memory
                    chunk_size = 512  # process per chunk 512  tokens
                    token_log_probs_list = []
                    
                    for i in range(0, response_logits.shape[0], chunk_size):
                        end_idx = min(i + chunk_size, response_logits.shape[0])
                        chunk_logits = response_logits[i:end_idx]  # [chunk_size, vocab_size]
                        chunk_token_ids = response_token_ids[i:end_idx]  # [chunk_size]
                        
                        # Compute this chunk of log probabilities
                        chunk_log_probs = torch.nn.functional.log_softmax(chunk_logits, dim=-1)
                        
                        # Get log probability for each token
                        chunk_token_log_probs = chunk_log_probs.gather(
                            dim=-1,
                            index=chunk_token_ids.unsqueeze(-1)
                        ).squeeze(-1)  # [chunk_size]
                        
                        token_log_probs_list.append(chunk_token_log_probs)
                        
                        # Immediately release GPU memory
                        del chunk_logits, chunk_log_probs, chunk_token_log_probs
                    
                    # Merge all chunks
                    token_log_probs = torch.cat(token_log_probs_list, dim=0)  # [response_len]
                    
                    # Total log probability of sequence
                    seq_log_prob = token_log_probs.sum()
                    
                    # Save sequence length (before deleting tensor)
                    seq_length = len(response_token_ids)
                    
                    # Immediately release large tensors
                    del logits, response_logits, response_token_ids, token_log_probs
                    del input_ids_full, attention_mask_full
                    
                    # Aggressive GPU-memory cleanup
                    torch.cuda.empty_cache()
                    gc.collect()
                    
                    # GRPO loss:-advantage * log_prob
                    # High advantage (good response) -> negative loss -> increase log_prob (increase generation probability)
                    # Low advantage (bad response) -> positive loss -> decrease log_prob (decrease generation probability)
                    
                    # Normalize log_prob (by sequence length)
                    normalized_log_prob = seq_log_prob / max(seq_length, 1)
                    
                    sample_loss = -advantage * normalized_log_prob
                    
                    total_loss = total_loss + sample_loss
                    num_valid_samples += 1
                    
                    # note:large tensors have already been released above (No. 501, 551-553 line)
                    # only release remaining small objects here
                    del inputs, prompt_inputs
                    
                except Exception as e:
                    logger.warning(f"Error while processing response: {e}")
                    import traceback
                    logger.warning(traceback.format_exc())
                    continue
                finally:
                    # Clear GPU memory regardless of success or failure
                    torch.cuda.empty_cache()
                    gc.collect()
        
        # Average loss
        if num_valid_samples > 0:
            loss = total_loss / num_valid_samples
        else:
            # If there are no valid samples, return a small loss
            logger.warning("⚠️ No valid samples; using placeholder loss")
            loss = torch.tensor(0.01, device=device, requires_grad=True)
            num_valid_samples = 0
        
        # Log statistics (detailed format)
        all_rewards_flat = [r for rewards in all_rewards for r in rewards]
        avg_reward = sum(all_rewards_flat) / len(all_rewards_flat) if all_rewards_flat else 0.0
        max_reward = max(all_rewards_flat) if all_rewards_flat else 0.0
        min_reward = min(all_rewards_flat) if all_rewards_flat else 0.0
        
        # Compute reward distribution
        excellent = sum(1 for r in all_rewards_flat if r >= 9.0)
        good = sum(1 for r in all_rewards_flat if 7.0 <= r < 9.0)
        acceptable = sum(1 for r in all_rewards_flat if 4.0 <= r < 7.0)
        poor = sum(1 for r in all_rewards_flat if 0.0 <= r < 4.0)
        very_poor = sum(1 for r in all_rewards_flat if r < 0.0)
        
        if rank == 0:  # Print only on main process
            logger.info(f"\n{'─'*70}")
            logger.info(f"💰 Reward statistics: avg={avg_reward:+.2f} | max={max_reward:+.2f} | min={min_reward:+.2f}")
            logger.info(f"   Distribution: excellent>=9({excellent}) good7-9({good}) acceptable4-7({acceptable}) poor0-4({poor}) very poor<0({very_poor})")
            logger.info(f"📉 Loss: {loss.item():.4f} | samples={num_valid_samples}/{self.num_samples*len(prompts)}")
            logger.info(f"{'='*70}\n")
        
        # Clean once more to ensure GPU memory is released
        del all_responses, all_rewards, all_log_probs
        torch.cuda.empty_cache()
        gc.collect()
        
        # Increment global step counter
        self.global_step_counter += 1
        
        return loss


def train_grpo_online():
    """Main training function"""
    
    # Parse arguments
    parser = HfArgumentParser((ModelArguments, DataArguments, GRPOArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    
    # Loading model
    logger.info(f"Loading model: {model_args.model_name_or_path}")
    model = Qwen2_5_VLForMolecule.from_pretrained(
        model_args.model_name_or_path,
        torch_dtype=torch.bfloat16 if training_args.bf16 else torch.float32,
    )
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        padding_side="left",
        use_fast=False,
    )
    
    # Load training data
    logger.info(f"load GRPO trainingdata: {data_args.data_path}")
    train_dataset = GRPODataset(
        data_path=data_args.data_path,
        tokenizer=tokenizer
    )
    
    # Create data collator (requires processor)
    from transformers import AutoProcessor
    processor = AutoProcessor.from_pretrained(
        model_args.model_name_or_path,
        trust_remote_code=True
    )
    
    data_collator = GRPODataCollator(
        tokenizer=tokenizer,
        processor=processor,
        max_length=training_args.model_max_length
    )
    
    logger.info(f"✓ Dataset preparation complete")
    
    # Initialize reward model
    reward_model = CIFRewardModel()
    
    # create GRPO trainer
    trainer = OnlineGRPOTrainer(
        model=model,
        grpo_args=training_args,
        reward_model=reward_model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=data_collator,
        processing_class=tokenizer,
    )
    
    # Start training
    logger.info("Start GRPO online training...")
    trainer.train()
    
    # Force-save final model after training ends
    logger.info("Training complete; saving final model...")
    trainer.save_model()
    
    # Also save state
    trainer.save_state()
    
    logger.info(f"✓ Final model saved to: {training_args.output_dir}")
    logger.info(f"✓ Training state saved to: {training_args.output_dir}/trainer_state.json")


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    print("="*80)
    print("Real GRPO online training")
    print("="*80)
    print("\nNote: this is a framework implementation; the following parts need completion:")
    print("1. actual sampling logic (model.generate)")
    print("2. log-probability computation (compute_log_prob)")
    print("3. data loading and batching")
    print("4. integration with visual inputs")
    print("="*80)
    print()
    
    train_grpo_online()

