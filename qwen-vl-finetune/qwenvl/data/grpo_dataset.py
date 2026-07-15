#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GRPO training dataset - supports both multimodal and plain-text modes
"""

import json
import torch
from torch.utils.data import Dataset
from typing import Dict, List, Any


class GRPODataset(Dataset):
    """
    GRPO training dataset
    
    Data format:
    {
        "prompt": "...",
        "expected_atoms": {...},
        "molecule_data": {...}  # Optional, needed only in multimodal mode
    }
    """
    
    def __init__(self, data_path: str, tokenizer=None):
        super().__init__()
        
        # Load data
        with open(data_path, 'r', encoding='utf-8') as f:
            self.data = json.load(f)
        
        self.tokenizer = tokenizer
        
        print(f"✓ Loaded {len(self.data)} GRPO training records")
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx) -> Dict[str, Any]:
        """
        Return raw data
        """
        item = self.data[idx]
        
        return {
            "prompt": item["prompt"],
            "expected_atoms": item["expected_atoms"],
            "molecule_data": item.get("molecule_data", {}),
        }


class GRPODataCollator:
    """
    GRPO data collator - automatically detects multimodal/plain-text mode
    """
    
    IMAGE_TOKEN = "<image>"
    
    def __init__(self, tokenizer, processor, max_length: int = 4096):
        self.tokenizer = tokenizer
        self.processor = processor
        self.max_length = max_length
    
    def detect_multimodal_mode(self, prompt: str) -> bool:
        """
        Detect whether the prompt uses multimodal mode (contains the <image> token)
        """
        return self.IMAGE_TOKEN in prompt
    
    def build_3d_inputs(self, molecule_data: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        """
        Build 3D input tensors from molecule_data (needed only in multimodal mode)
        """
        if not molecule_data or "z" not in molecule_data or "pos" not in molecule_data:
            # Return empty 3D inputs
            return {
                "atomic_numbers": torch.zeros(1, 1, dtype=torch.long),
                "coordinates": torch.zeros(1, 1, 3, dtype=torch.float32),
                "molecule_mask": torch.zeros(1, 1, dtype=torch.bool),
            }
        
        # Atomic numbers
        z = torch.tensor(molecule_data["z"], dtype=torch.long).unsqueeze(0)  # (1, N)
        
        # Atomic coordinates
        pos = torch.tensor(molecule_data["pos"], dtype=torch.float32).unsqueeze(0)  # (1, N, 3)
        
        # Mask
        n = z.shape[1]
        mask = torch.ones(n, dtype=torch.bool).unsqueeze(0)  # (1, N)
        
        model_inputs = {
            "atomic_numbers": z,
            "coordinates": pos,
            "molecule_mask": mask,
        }
        
        # Cell parameters (optional)
        if "cell" in molecule_data and molecule_data["cell"]:
            cell = torch.tensor(molecule_data["cell"], dtype=torch.float32).unsqueeze(0)  # (1, 6)
            model_inputs["cell"] = cell
        
        return model_inputs
    
    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Process one batch of data - automatically detect and handle multimodal/plain-text mode
        
        According to the logic in CLI_many_turn_cell_single.py:
        - If the prompt contains <image> -> multimodal mode -> 3D inputs are required
        - If the prompt does not contain <image> -> plain-text mode -> 3D inputs are not required
        """
        # Extract data
        prompts = [f["prompt"] for f in features]
        expected_atoms_list = [f["expected_atoms"] for f in features]
        molecule_data_list = [f["molecule_data"] for f in features]
        
        # Detect whether this is multimodal mode
        is_multimodal = any(self.detect_multimodal_mode(p) for p in prompts)
        
        # Build text inputs (using chat template)
        templated_texts = []
        
        for prompt in prompts:
            # Use chat template
            chat_history = [{"role": "user", "content": prompt}]
            templated_text = self.tokenizer.apply_chat_template(
                chat_history,
                tokenize=False,
                add_generation_prompt=True
            )
            templated_texts.append(templated_text)
        
        # Tokenize text
        text_inputs = self.tokenizer(
            templated_texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        )
        
        # Build the return dictionary
        batch_dict = {
            "input_ids": text_inputs["input_ids"],
            "attention_mask": text_inputs["attention_mask"],
            "prompts": prompts,  # Original prompts
            "expected_atoms_list": expected_atoms_list,
            "molecule_data_list": molecule_data_list,
            "is_multimodal": is_multimodal,
        }
        
        # If in multimodal mode, add 3D inputs
        if is_multimodal:
            batch_atomic_numbers = []
            batch_coordinates = []
            batch_molecule_mask = []
            batch_cell = []
            has_cell = False
            
            for molecule_data in molecule_data_list:
                try:
                    inputs_3d = self.build_3d_inputs(molecule_data)
                    batch_atomic_numbers.append(inputs_3d["atomic_numbers"])
                    batch_coordinates.append(inputs_3d["coordinates"])
                    batch_molecule_mask.append(inputs_3d["molecule_mask"])
                    
                    if "cell" in inputs_3d:
                        batch_cell.append(inputs_3d["cell"])
                        has_cell = True
                    else:
                        batch_cell.append(torch.zeros(1, 6, dtype=torch.float32))
                except Exception as e:
                    print(f"⚠️ Failed to build 3D inputs: {e}，using empty inputs")
                    batch_atomic_numbers.append(torch.zeros(1, 1, dtype=torch.long))
                    batch_coordinates.append(torch.zeros(1, 1, 3, dtype=torch.float32))
                    batch_molecule_mask.append(torch.zeros(1, 1, dtype=torch.bool))
                    batch_cell.append(torch.zeros(1, 6, dtype=torch.float32))
            
            # Find the maximum number of atoms (for padding)
            max_atoms = max(t.shape[1] for t in batch_atomic_numbers)
            
            # Pad 3D inputs to the same length
            padded_atomic_numbers = []
            padded_coordinates = []
            padded_molecule_mask = []
            
            for z, pos, mask in zip(batch_atomic_numbers, batch_coordinates, batch_molecule_mask):
                n_atoms = z.shape[1]
                if n_atoms < max_atoms:
                    pad_size = max_atoms - n_atoms
                    z_padded = torch.cat([z, torch.zeros(1, pad_size, dtype=torch.long)], dim=1)
                    pos_padded = torch.cat([pos, torch.zeros(1, pad_size, 3, dtype=torch.float32)], dim=1)
                    mask_padded = torch.cat([mask, torch.zeros(1, pad_size, dtype=torch.bool)], dim=1)
                else:
                    z_padded = z
                    pos_padded = pos
                    mask_padded = mask
                
                padded_atomic_numbers.append(z_padded)
                padded_coordinates.append(pos_padded)
                padded_molecule_mask.append(mask_padded)
            
            # Add 3D inputs to the batch
            batch_dict["atomic_numbers"] = torch.cat(padded_atomic_numbers, dim=0)
            batch_dict["coordinates"] = torch.cat(padded_coordinates, dim=0)
            batch_dict["molecule_mask"] = torch.cat(padded_molecule_mask, dim=0)
            
            if has_cell:
                batch_dict["cell"] = torch.cat(batch_cell, dim=0)
        
        return batch_dict
