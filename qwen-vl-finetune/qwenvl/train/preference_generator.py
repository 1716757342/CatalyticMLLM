# qwen-vl-finetune/qwenvl/train/preference_generator.py

import torch
import torch.nn.functional as F
from typing import List, Dict, Tuple, Optional, Any
import random
import numpy as np
from dataclasses import dataclass
from .reward_model import EnergyRewardModel
import re

@dataclass
class GenerationConfig:
    """Generation configuration"""
    max_new_tokens: int = 100
    temperature: float = 1.0
    top_p: float = 1.0
    top_k: Optional[int] = None
    do_sample: bool = True
    num_beams: int = 1
    repetition_penalty: float = 1.0

@dataclass
class PreferencePair:
    """Preference-pair data structure"""
    chosen_text: str
    rejected_text: str
    chosen_reward: float
    rejected_reward: float
    reward_diff: float
    molecule_data: Dict[str, Any]
    prompt: str

class PreferenceDataGenerator:
    """
    Preference data generator
    Generate multiple candidate answers with the existing model, then build preference pairs based on reward scores
    """
    
    def __init__(
        self, 
        model, 
        tokenizer, 
        reward_model: EnergyRewardModel,
        num_candidates: int = 4,
        min_reward_diff: float = 1.0
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.reward_model = reward_model
        self.num_candidates = num_candidates
        self.min_reward_diff = min_reward_diff
        
        # Different generation configurations used to produce diverse candidates
        self.generation_configs = [
            GenerationConfig(temperature=0.7, top_p=0.9, do_sample=True),
            GenerationConfig(temperature=1.0, top_p=0.8, do_sample=True),
            GenerationConfig(temperature=0.5, top_k=50, do_sample=True),
            GenerationConfig(temperature=0.3, top_p=0.95, do_sample=True),
            GenerationConfig(do_sample=False),  # greedy decoding
        ]
    
    def prepare_generation_inputs(self, sample: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        """
        Prepare inputs required for generation
        
        Args:
            sample: sample containing molecule data and conversation
            
        Returns:
            prepared model inputs
        """
        conversations = sample["conversations"]
        human_message = conversations[0]["value"]  # get the human question
        
        # Extract molecule data
        molecule_data = sample.get("molecule", {})
        
        # Build input
        inputs = {
            "input_ids": None,  # will be handled during generate
            "prompt_text": human_message,
        }
        
        # Add molecule-related data
        if molecule_data:
            inputs.update({
                "atomic_numbers": torch.tensor(molecule_data["z"], dtype=torch.long),
                "coordinates": torch.tensor(molecule_data["pos"], dtype=torch.float32),
                "molecule_mask": torch.ones(len(molecule_data["z"]), dtype=torch.bool)
            })
            
            # Add cell parameters (if present)
            if "cell" in molecule_data:
                inputs["cell"] = torch.tensor(molecule_data["cell"], dtype=torch.float32)
        
        return inputs
    
    def generate_candidates(self, sample: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Generate multiple candidate answers for one sample
        
        Args:
            sample: inputsample
            
        Returns:
            candidate answer list
        """
        candidates = []
        inputs = self.prepare_generation_inputs(sample)
        
        # Prepare tokenizer input
        prompt_text = inputs["prompt_text"]
        
        # Generate candidates with different configurations
        used_configs = self.generation_configs[:self.num_candidates]
        
        for i, config in enumerate(used_configs):
            try:
                # Set random seed to ensure diversity
                if config.do_sample:
                    torch.manual_seed(random.randint(0, 10000))
                
                # [Fix]Use chat template to build the correct conversation format
                # This lets the model generate a complete answer rather than only a number
                chat_history = [{"role": "user", "content": prompt_text}]
                
                templated_text = self.tokenizer.apply_chat_template(
                    chat_history,
                    tokenize=False,
                    add_generation_prompt=True  # Add assistant start marker
                )
                
                # Tokenizetemplated input
                tokenized = self.tokenizer(
                    [templated_text],  # Note: pass a list
                    return_tensors="pt",
                    padding=True,
                    truncation=True
                )
                
                # Move to model device
                device = next(self.model.parameters()).device
                for key in tokenized:
                    tokenized[key] = tokenized[key].to(device)
                
                # Add molecule data to inputs
                if "atomic_numbers" in inputs:
                    tokenized["atomic_numbers"] = inputs["atomic_numbers"].unsqueeze(0).to(device)
                    tokenized["coordinates"] = inputs["coordinates"].unsqueeze(0).to(device)
                    tokenized["molecule_mask"] = inputs["molecule_mask"].unsqueeze(0).to(device)
                    
                    if "cell" in inputs:
                        tokenized["cell"] = inputs["cell"].unsqueeze(0).to(device)
                
                # Generate answer
                with torch.no_grad():
                    generated_ids = self.model.generate(
                        **tokenized,
                        max_new_tokens=config.max_new_tokens,
                        temperature=config.temperature if config.do_sample else 1.0,
                        top_p=config.top_p if config.do_sample else 1.0,
                        top_k=config.top_k,
                        do_sample=config.do_sample,
                        num_beams=config.num_beams,
                        repetition_penalty=config.repetition_penalty,
                        pad_token_id=self.tokenizer.pad_token_id,
                        eos_token_id=self.tokenizer.eos_token_id
                    )
                
                # Decode generated text
                # Take only the newly generated part
                input_length = tokenized["input_ids"].shape[1]
                generated_text = self.tokenizer.decode(
                    generated_ids[0][input_length:], 
                    skip_special_tokens=True
                ).strip()
                
                candidates.append({
                    "text": generated_text,
                    "config_index": i,
                    "config": config
                })
                
            except Exception as e:
                print(f"Error generating candidate {i} : {e}")
                # Generate a default error answer
                candidates.append({
                    "text": "The relaxed energy of this molecule is 0.0.",
                    "config_index": i,
                    "config": config,
                    "error": str(e)
                })
        
        return candidates
    
    def evaluate_candidates(self, candidates: List[Dict[str, Any]], true_energy: float) -> List[Dict[str, Any]]:
        """
        Evaluate candidate answers with the reward model
        
        Args:
            candidates: candidate answer list
            true_energy: true energy value
            
        Returns:
            candidate list with reward scores
        """
        texts = [candidate["text"] for candidate in candidates]
        true_energies = [true_energy] * len(texts)
        
        # Compute rewards in batch
        batch_results = self.reward_model.compute_batch_rewards(texts, true_energies)
        
        # Add reward scores to candidates
        for i, candidate in enumerate(candidates):
            candidate.update({
                "reward": batch_results["rewards"][i].item(),
                "predicted_energy": batch_results["predicted_energies"][i].item(),
                "abs_error": batch_results["abs_errors"][i].item(),
                "category": batch_results["categories"][i]
            })
        
        return candidates
    
    def create_preference_pairs(self, evaluated_candidates: List[Dict[str, Any]]) -> List[Tuple[Dict[str, Any], Dict[str, Any]]]:
        """
        Create preference pairs from evaluated candidates
        
        Args:
            evaluated_candidates: evaluated candidate list
            
        Returns:
            preference-pair list [(chosen, rejected), ...]
        """
        # Sort by reward score
        sorted_candidates = sorted(evaluated_candidates, key=lambda x: x["reward"], reverse=True)
        
        pairs = []
        
        # Create all possible preference pairs
        for i in range(len(sorted_candidates)):
            for j in range(i + 1, len(sorted_candidates)):
                chosen = sorted_candidates[i]
                rejected = sorted_candidates[j]
                
                reward_diff = chosen["reward"] - rejected["reward"]
                
                # Keep only preference pairs with sufficiently large reward differences
                if reward_diff >= self.min_reward_diff:
                    pairs.append((chosen, rejected))
        
        return pairs
    
    def generate_preference_data(self, samples: List[Dict[str, Any]]) -> List[PreferencePair]:
        """
        Generate preference data for a sample list
        
        Args:
            samples: original sample list
            
        Returns:
            preference-pair list
        """
        preference_pairs = []
        
        for i, sample in enumerate(samples):
            print(f"Processing sample {i+1}/{len(samples)}...")
            
            try:
                # Get true energy value
                true_energy = None
                if "conversations" in sample and len(sample["conversations"]) > 1:
                    gpt_response = sample["conversations"][1]["value"]
                    true_energy = self.reward_model.extract_energy_from_text(gpt_response)
                
                if true_energy is None:
                    print(f"sample {i+1} unable to extract true energy; skipping")
                    continue
                
                # Error generating candidate
                candidates = self.generate_candidates(sample)
                
                # evaluationCandidate
                evaluated_candidates = self.evaluate_candidates(candidates, true_energy)
                
                # Create preference pairs
                pairs = self.create_preference_pairs(evaluated_candidates)
                
                # Convert to PreferencePair objects
                for chosen, rejected in pairs:
                    preference_pair = PreferencePair(
                        chosen_text=chosen["text"],
                        rejected_text=rejected["text"],
                        chosen_reward=chosen["reward"],
                        rejected_reward=rejected["reward"],
                        reward_diff=chosen["reward"] - rejected["reward"],
                        molecule_data=sample.get("molecule", {}),
                        prompt=sample["conversations"][0]["value"]
                    )
                    preference_pairs.append(preference_pair)
                
                print(f"sample {i+1} generated {len(pairs)} preference pairs")
                
            except Exception as e:
                print(f"Processing sample {i+1} : {e}")
                continue
        
        print(f"Generated total {len(preference_pairs)} preference pairs")
        return preference_pairs
    
    def save_preference_data(self, preference_pairs: List[PreferencePair], save_path: str):
        """
        Save preference data to file
        
        Args:
            preference_pairs: preference-pair list
            save_path: save path
        """
        import json
        
        data = []
        for pair in preference_pairs:
            data.append({
                "chosen_text": pair.chosen_text,
                "rejected_text": pair.rejected_text,
                "chosen_reward": pair.chosen_reward,
                "rejected_reward": pair.rejected_reward,
                "reward_diff": pair.reward_diff,
                "molecule_data": pair.molecule_data,
                "prompt": pair.prompt
            })
        
        with open(save_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        
        print(f"Preference data saved to: {save_path}")

def load_preference_data(file_path: str) -> List[PreferencePair]:
    """
    Load preference data from file
    
    Args:
        file_path: file path
        
    Returns:
        preference-pair list
    """
    import json
    
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    preference_pairs = []
    for item in data:
        pair = PreferencePair(
            chosen_text=item["chosen_text"],
            rejected_text=item["rejected_text"],
            chosen_reward=item["chosen_reward"],
            rejected_reward=item["rejected_reward"],
            reward_diff=item["reward_diff"],
            molecule_data=item["molecule_data"],
            prompt=item["prompt"]
        )
        preference_pairs.append(pair)
    
    return preference_pairs
