# qwen-vl-finetune/qwenvl/data/data_ppo.py

import json
import torch
from torch.utils.data import Dataset
from dataclasses import dataclass
import transformers
from typing import Dict, Sequence, List, Any, Optional
import copy

# Try importing; if it fails, use local definitions
try:
    from .data_qwen import preprocess_qwen_2_visual, IGNORE_INDEX, IMAGE_TOKEN_INDEX
except ImportError:
    print(f"Warning: Could not import from data_qwen")
    IGNORE_INDEX = -100
    IMAGE_TOKEN_INDEX = 151655
    
    def preprocess_qwen_2_visual(conversations, tokenizer, grid_thw_image=None):
        """Simplified preprocessing function"""
        if grid_thw_image is None:
            grid_thw_image = []
        
        conversation_text = ""
        for conv in conversations[0]:
            if conv["from"] == "human":
                text = conv["value"]
                if grid_thw_image and len(grid_thw_image) > 0:
                    image_tokens = "".join([tokenizer.decode([IMAGE_TOKEN_INDEX])] * grid_thw_image[0])
                    text = text.replace("<image>", image_tokens)
                conversation_text += text
            elif conv["from"] == "gpt":
                conversation_text += conv["value"]
        
        tokenized = tokenizer(
            conversation_text,
            return_tensors="pt",
            padding=False,
            truncation=True,
            max_length=4096
        )
        
        labels = tokenized["input_ids"].clone()
        
        return {
            "input_ids": tokenized["input_ids"],
            "labels": labels
        }


class PPODataset(Dataset):
    """
    PPO training dataset
    Directly use the supervised learning data format, including molecules and true energy values
    """
    
    def __init__(self, data_path: str, tokenizer: transformers.PreTrainedTokenizer):
        super(PPODataset, self).__init__()
        self.tokenizer = tokenizer
        
        # Load data
        with open(data_path, 'r', encoding='utf-8') as f:
            self.data = json.load(f)
        
        print(f"✅ Successfully loaded {len(self.data)} training samples (PPO mode)")
        
        # Count valid samples
        valid_samples = 0
        for sample in self.data:
            if 'molecule' in sample and 'conversations' in sample:
                valid_samples += 1
        
        print(f"   Valid sample count: {valid_samples}")
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        sample = self.data[i]
        
        # Get molecule data
        molecule_data = sample.get("molecule", {})
        conversations = sample.get("conversations", [])
        
        # Extract the prompt (the human question)
        prompt = ""
        true_answer = ""
        for conv in conversations:
            if conv.get("from") == "human":
                prompt = conv.get("value", "")
            elif conv.get("from") == "gpt":
                true_answer = conv.get("value", "")
        
        # Extract the true energy value
        true_energy = molecule_data.get("energy", None)
        if true_energy is None:
            # Try extracting from the answer
            true_energy = self._extract_energy_from_text(true_answer)
        
        # Process conversation data (used to compute log probability)
        item = self._process_conversation(conversations, molecule_data)
        
        # Add extra information
        item["true_energy"] = torch.tensor(true_energy if true_energy is not None else 0.0, dtype=torch.float32)
        item["prompt"] = prompt
        item["true_answer"] = true_answer
        
        return item
    
    def _extract_energy_from_text(self, text: str) -> Optional[float]:
        """Extract the energy value from text"""
        import re
        patterns = [
            r"energy.*?is\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)",
            r"is\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)。",
            r"([-+]?\d+\.?\d*(?:[eE][-+]?\d+)?)"
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            if matches:
                try:
                    return float(matches[-1])
                except (ValueError, IndexError):
                    continue
        return None
    
    def _process_conversation(self, conversation: List[Dict[str, str]], molecule_data: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        """Process a single conversation and generate tokenized data"""
        has_molecule = bool(molecule_data)
        
        if has_molecule:
            num_atoms = len(molecule_data["z"])
            
            data_dict = preprocess_qwen_2_visual(
                [conversation],
                self.tokenizer,
                grid_thw_image=[num_atoms],
            )
            
            item = {
                "input_ids": data_dict["input_ids"].squeeze(0),
                "labels": data_dict["labels"].squeeze(0),
                "atomic_numbers": torch.tensor(molecule_data["z"], dtype=torch.long),
                "coordinates": torch.tensor(molecule_data["pos"], dtype=torch.float32),
                "molecule_mask": torch.ones(num_atoms, dtype=torch.bool)
            }
            
            if "cell" in molecule_data:
                item["cell"] = torch.tensor(molecule_data["cell"], dtype=torch.float32)
        else:
            data_dict = preprocess_qwen_2_visual(
                [conversation],
                self.tokenizer,
                grid_thw_image=[],
            )
            
            item = {
                "input_ids": data_dict["input_ids"].squeeze(0),
                "labels": data_dict["labels"].squeeze(0),
            }
        
        return item


@dataclass
class PPODataCollator:
    """
    Data collator dedicated to PPO
    """
    tokenizer: transformers.PreTrainedTokenizer
    
    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids = [instance["input_ids"] for instance in instances]
        labels = [instance["labels"] for instance in instances]
        true_energies = [instance["true_energy"] for instance in instances]
        
        # Ensure pad_token_id is valid
        pad_token_id = self.tokenizer.pad_token_id
        if pad_token_id is None:
            pad_token_id = self.tokenizer.eos_token_id
        
        # Limit sequence length
        max_length = 2048
        
        # Ensure types are correct
        input_ids = [ids.long() for ids in input_ids]
        labels = [l.long() for l in labels]
        
        # Padding
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=pad_token_id
        )
        labels = torch.nn.utils.rnn.pad_sequence(
            labels, batch_first=True, padding_value=IGNORE_INDEX
        )
        
        # Limit length
        if input_ids.size(1) > max_length:
            input_ids = input_ids[:, :max_length]
            labels = labels[:, :max_length]
        
        batch = {
            "input_ids": input_ids,
            "labels": labels,
            "attention_mask": input_ids.ne(pad_token_id).long(),
            "true_energies": torch.stack(true_energies),
        }
        
        # Process molecule data if present
        if "atomic_numbers" in instances[0] and instances[0]["atomic_numbers"] is not None:
            max_atoms = max(
                len(inst["atomic_numbers"]) if inst.get("atomic_numbers") is not None else 0
                for inst in instances
            )
            
            if max_atoms > 0:
                atomic_numbers = []
                coordinates = []
                molecule_mask = []
                cells = []
                
                for instance in instances:
                    data = self._pad_molecule_data(
                        instance.get("atomic_numbers"),
                        instance.get("coordinates"),
                        instance.get("molecule_mask"),
                        instance.get("cell"),
                        max_atoms
                    )
                    atomic_numbers.append(data["atomic_numbers"])
                    coordinates.append(data["coordinates"])
                    molecule_mask.append(data["molecule_mask"])
                    cells.append(data["cell"])
                
                batch.update({
                    "atomic_numbers": torch.stack(atomic_numbers),
                    "coordinates": torch.stack(coordinates),
                    "molecule_mask": torch.stack(molecule_mask),
                    "cell": torch.stack(cells),
                })
        
        return batch
    
    def _pad_molecule_data(self, atomic_numbers, coordinates, molecule_mask, cell, max_atoms):
        """Pad molecule data to the specified maximum number of atoms"""
        if atomic_numbers is not None:
            num_atoms = len(atomic_numbers)
            pad_len = max_atoms - num_atoms
            
            padded_z = torch.cat([atomic_numbers, torch.zeros(pad_len, dtype=torch.long)])
            padded_pos = torch.cat([coordinates, torch.zeros(pad_len, 3, dtype=torch.float32)])
            padded_mask = torch.cat([molecule_mask, torch.zeros(pad_len, dtype=torch.bool)])
            
            if cell is not None:
                padded_cell = cell
            else:
                padded_cell = torch.zeros(6, dtype=torch.float32)
        else:
            padded_z = torch.zeros(max_atoms, dtype=torch.long)
            padded_pos = torch.zeros(max_atoms, 3, dtype=torch.float32)
            padded_mask = torch.zeros(max_atoms, dtype=torch.bool)
            padded_cell = torch.zeros(6, dtype=torch.float32)
        
        return {
            "atomic_numbers": padded_z,
            "coordinates": padded_pos,
            "molecule_mask": padded_mask,
            "cell": padded_cell
        }


def make_ppo_data_module(
    tokenizer: transformers.PreTrainedTokenizer, 
    data_path: str,
    eval_data_path: Optional[str] = None
) -> Dict:
    """
    Factory function for creating a PPO data module
    
    Args:
        tokenizer: Tokenizer
        data_path: Training data path
        eval_data_path: Validation data path (optional)
        
    Returns:
        Dictionary containing the dataset and data collator
    """
    train_dataset = PPODataset(
        data_path=data_path,
        tokenizer=tokenizer
    )
    
    eval_dataset = None
    if eval_data_path:
        eval_dataset = PPODataset(
            data_path=eval_data_path,
            tokenizer=tokenizer
        )
    
    data_collator = PPODataCollator(tokenizer=tokenizer)
    
    return dict(
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator
    )




