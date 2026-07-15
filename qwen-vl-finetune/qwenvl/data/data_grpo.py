# qwen-vl-finetune/qwenvl/data/data_grpo.py

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
except ImportError as e:
    print(f"Warning: Could not import from data_qwen: {e}")
    # Use local constant definitions
    IGNORE_INDEX = -100
    IMAGE_TOKEN_INDEX = 151655
    
    # Simplified preprocessing function
    def preprocess_qwen_2_visual(conversations, tokenizer, grid_thw_image=None):
        """Simplified preprocessing function"""
        if grid_thw_image is None:
            grid_thw_image = []
        
        # Build conversation text
        conversation_text = ""
        for conv in conversations[0]:  # conversations is a nested list
            if conv["from"] == "human":
                # If image placeholders exist, insert the corresponding number of IMAGE_TOKENs
                text = conv["value"]
                if grid_thw_image and len(grid_thw_image) > 0:
                    # Replace <image> with the actual IMAGE_TOKEN
                    image_tokens = "".join([tokenizer.decode([IMAGE_TOKEN_INDEX])] * grid_thw_image[0])
                    text = text.replace("<image>", image_tokens)
                conversation_text += text
            elif conv["from"] == "gpt":
                conversation_text += conv["value"]
        
        # Simple tokenization
        tokenized = tokenizer(
            conversation_text,
            return_tensors="pt",
            padding=False,
            truncation=True,
            max_length=4096
        )
        
        # Create labels (same as input_ids, with the human part set to IGNORE_INDEX)
        labels = tokenized["input_ids"].clone()
        
        return {
            "input_ids": tokenized["input_ids"],
            "labels": labels
        }

class GRPOPreferenceDataset(Dataset):
    """
    GRPO preference dataset
    Process preference data containing chosen and rejected pairs
    """
    
    def __init__(self, preference_data_path: str, tokenizer: transformers.PreTrainedTokenizer):
        super(GRPOPreferenceDataset, self).__init__()
        self.tokenizer = tokenizer
        
        # Load preference data
        with open(preference_data_path, 'r', encoding='utf-8') as f:
            self.preference_data = json.load(f)
        
        print(f"Successfully loaded {len(self.preference_data)} preference-pair data items.")
    
    def __len__(self):
        return len(self.preference_data)
    
    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        preference_pair = self.preference_data[i]
        
        # Get basic information
        prompt = preference_pair["prompt"]
        chosen_text = preference_pair["chosen_text"] 
        rejected_text = preference_pair["rejected_text"]
        molecule_data = preference_pair["molecule_data"]
        
        # Build the complete conversation format
        chosen_conversation = [
            {"from": "human", "value": prompt},
            {"from": "gpt", "value": chosen_text}
        ]
        
        rejected_conversation = [
            {"from": "human", "value": prompt},
            {"from": "gpt", "value": rejected_text}
        ]
        
        # Process chosen data
        chosen_item = self._process_conversation(chosen_conversation, molecule_data)
        
        # Process rejected data
        rejected_item = self._process_conversation(rejected_conversation, molecule_data)
        
        # Combine data
        result = {
            # Chosen data
            "chosen_input_ids": chosen_item["input_ids"],
            "chosen_labels": chosen_item["labels"],
            "chosen_atomic_numbers": chosen_item.get("atomic_numbers"),
            "chosen_coordinates": chosen_item.get("coordinates"),
            "chosen_molecule_mask": chosen_item.get("molecule_mask"),
            "chosen_cell": chosen_item.get("cell"),
            
            # Rejected data
            "rejected_input_ids": rejected_item["input_ids"],
            "rejected_labels": rejected_item["labels"],
            "rejected_atomic_numbers": rejected_item.get("atomic_numbers"),
            "rejected_coordinates": rejected_item.get("coordinates"),
            "rejected_molecule_mask": rejected_item.get("molecule_mask"),
            "rejected_cell": rejected_item.get("cell"),
            
            # Reward information
            "reward_diff": torch.tensor(preference_pair.get("reward_diff", 0.0), dtype=torch.float32),
            "chosen_reward": torch.tensor(preference_pair.get("chosen_reward", 0.0), dtype=torch.float32),
            "rejected_reward": torch.tensor(preference_pair.get("rejected_reward", 0.0), dtype=torch.float32),
        }
        
        return result
    
    def _process_conversation(self, conversation: List[Dict[str, str]], molecule_data: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        """
        Process a single conversation, similar to the original molecule data processing
        """
        # Check whether molecule data is present
        has_molecule = bool(molecule_data)
        
        if has_molecule:
            num_atoms = len(molecule_data["z"])
            
            # Insert placeholders for molecule samples
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
            
            # Add cell parameters if present
            if "cell" in molecule_data:
                item["cell"] = torch.tensor(molecule_data["cell"], dtype=torch.float32)
        else:
            # Plain-text data
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
class GRPODataCollator:
    """
    Data collator dedicated to GRPO
    Batch preference-pair data
    """
    tokenizer: transformers.PreTrainedTokenizer
    
    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        # Extract chosen and rejected data separately
        chosen_input_ids = [instance["chosen_input_ids"] for instance in instances]
        chosen_labels = [instance["chosen_labels"] for instance in instances]
        
        rejected_input_ids = [instance["rejected_input_ids"] for instance in instances]
        rejected_labels = [instance["rejected_labels"] for instance in instances]
        
        # Ensure pad_token_id is valid
        pad_token_id = self.tokenizer.pad_token_id
        if pad_token_id is None:
            pad_token_id = self.tokenizer.eos_token_id
        
        # Pad the text portion and limit sequence length to avoid index issues
        max_length = 2048  # Set a reasonable maximum length
        
        # Ensure all input_ids and labels are LongTensor type to avoid device compatibility issues
        chosen_input_ids = [ids.long() for ids in chosen_input_ids]
        chosen_labels = [labels.long() for labels in chosen_labels]
        rejected_input_ids = [ids.long() for ids in rejected_input_ids]
        rejected_labels = [labels.long() for labels in rejected_labels]
        
        chosen_input_ids = torch.nn.utils.rnn.pad_sequence(
            chosen_input_ids, batch_first=True, padding_value=pad_token_id
        )
        chosen_labels = torch.nn.utils.rnn.pad_sequence(
            chosen_labels, batch_first=True, padding_value=IGNORE_INDEX
        )
        
        rejected_input_ids = torch.nn.utils.rnn.pad_sequence(
            rejected_input_ids, batch_first=True, padding_value=pad_token_id
        )
        rejected_labels = torch.nn.utils.rnn.pad_sequence(
            rejected_labels, batch_first=True, padding_value=IGNORE_INDEX
        )
        
        # Limit sequence length to avoid index out of bounds
        if chosen_input_ids.size(1) > max_length:
            chosen_input_ids = chosen_input_ids[:, :max_length]
            chosen_labels = chosen_labels[:, :max_length]
        
        if rejected_input_ids.size(1) > max_length:
            rejected_input_ids = rejected_input_ids[:, :max_length]
            rejected_labels = rejected_labels[:, :max_length]
        
        # Ensure pad_token_id has the correct type and value
        if self.tokenizer.pad_token_id is None:
            pad_token_id = self.tokenizer.eos_token_id
        else:
            pad_token_id = self.tokenizer.pad_token_id
        
        # Process molecule data
        batch = {
            "chosen_input_ids": chosen_input_ids,
            "chosen_labels": chosen_labels,
            "chosen_attention_mask": chosen_input_ids.ne(pad_token_id).long(),
            
            "rejected_input_ids": rejected_input_ids,
            "rejected_labels": rejected_labels,
            "rejected_attention_mask": rejected_input_ids.ne(pad_token_id).long(),
        }
        
        # Process molecule data if present
        if "chosen_atomic_numbers" in instances[0] and instances[0]["chosen_atomic_numbers"] is not None:
            # Determine the maximum number of atoms
            max_atoms = max(
                len(inst["chosen_atomic_numbers"]) if inst["chosen_atomic_numbers"] is not None else 0
                for inst in instances
            )
            max_atoms = max(max_atoms, max(
                len(inst["rejected_atomic_numbers"]) if inst["rejected_atomic_numbers"] is not None else 0
                for inst in instances
            ))
            
            if max_atoms > 0:
                # Process chosen molecule data
                chosen_atomic_numbers = []
                chosen_coordinates = []
                chosen_molecule_mask = []
                chosen_cells = []
                
                # Process rejected molecule data
                rejected_atomic_numbers = []
                rejected_coordinates = []
                rejected_molecule_mask = []
                rejected_cells = []
                
                for instance in instances:
                    # Process chosen data
                    chosen_data = self._pad_molecule_data(
                        instance.get("chosen_atomic_numbers"),
                        instance.get("chosen_coordinates"),
                        instance.get("chosen_molecule_mask"),
                        instance.get("chosen_cell"),
                        max_atoms
                    )
                    chosen_atomic_numbers.append(chosen_data["atomic_numbers"])
                    chosen_coordinates.append(chosen_data["coordinates"])
                    chosen_molecule_mask.append(chosen_data["molecule_mask"])
                    chosen_cells.append(chosen_data["cell"])
                    
                    # Process rejected data
                    rejected_data = self._pad_molecule_data(
                        instance.get("rejected_atomic_numbers"),
                        instance.get("rejected_coordinates"),
                        instance.get("rejected_molecule_mask"),
                        instance.get("rejected_cell"),
                        max_atoms
                    )
                    rejected_atomic_numbers.append(rejected_data["atomic_numbers"])
                    rejected_coordinates.append(rejected_data["coordinates"])
                    rejected_molecule_mask.append(rejected_data["molecule_mask"])
                    rejected_cells.append(rejected_data["cell"])
                
                # Add to the batch
                batch.update({
                    "chosen_atomic_numbers": torch.stack(chosen_atomic_numbers),
                    "chosen_coordinates": torch.stack(chosen_coordinates),
                    "chosen_molecule_mask": torch.stack(chosen_molecule_mask),
                    "chosen_cell": torch.stack(chosen_cells),
                    
                    "rejected_atomic_numbers": torch.stack(rejected_atomic_numbers),
                    "rejected_coordinates": torch.stack(rejected_coordinates),
                    "rejected_molecule_mask": torch.stack(rejected_molecule_mask),
                    "rejected_cell": torch.stack(rejected_cells),
                })
        
        # Add reward information
        if "reward_diff" in instances[0]:
            batch["rewards"] = torch.stack([instance["reward_diff"] for instance in instances])
            batch["chosen_rewards"] = torch.stack([instance["chosen_reward"] for instance in instances])
            batch["rejected_rewards"] = torch.stack([instance["rejected_reward"] for instance in instances])
        
        return batch
    
    def _pad_molecule_data(self, atomic_numbers, coordinates, molecule_mask, cell, max_atoms):
        """
        Pad molecule data to the specified maximum number of atoms
        """
        if atomic_numbers is not None:
            num_atoms = len(atomic_numbers)
            pad_len = max_atoms - num_atoms
            
            padded_z = torch.cat([atomic_numbers, torch.zeros(pad_len, dtype=torch.long)])
            padded_pos = torch.cat([coordinates, torch.zeros(pad_len, 3, dtype=torch.float32)])
            padded_mask = torch.cat([molecule_mask, torch.zeros(pad_len, dtype=torch.bool)])
            
            # Process cell
            if cell is not None:
                padded_cell = cell
            else:
                padded_cell = torch.zeros(6, dtype=torch.float32)
        else:
            # Create dummy data
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

def make_grpo_data_module(
    tokenizer: transformers.PreTrainedTokenizer, 
    preference_data_path: str,
    eval_preference_data_path: Optional[str] = None
) -> Dict:
    """
    Factory function for creating a GRPO data module
    
    Args:
        tokenizer: Tokenizer
        preference_data_path: Training preference data path
        eval_preference_data_path: Validation preference data path (optional)
        
    Returns:
        Dictionary containing the dataset and data collator
    """
    train_dataset = GRPOPreferenceDataset(
        preference_data_path=preference_data_path,
        tokenizer=tokenizer
    )
    
    eval_dataset = None
    if eval_preference_data_path:
        eval_dataset = GRPOPreferenceDataset(
            preference_data_path=eval_preference_data_path,
            tokenizer=tokenizer
        )
    
    data_collator = GRPODataCollator(tokenizer=tokenizer)
    
    return dict(
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator
    )
