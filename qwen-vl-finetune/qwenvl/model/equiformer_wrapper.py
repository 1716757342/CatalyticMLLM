# qwen-vl-finetune/qwenvl/model/equiformer_wrapper.py

import torch
import torch.nn as nn
from typing import Optional, Tuple
from equiformer_pytorch import Equiformer

class EquiformerWrapper(nn.Module):
    """
    A wrapper class that adapts Equiformer as the visual module for Qwen-VL.
    """
    def __init__(self, equiformer_config: dict, qwen_vision_config):
        super().__init__()
        equiformer_config['dim'] = qwen_vision_config.hidden_size 
        print(f"Equiformer output dimension set to match Qwen's vision hidden size: {equiformer_config['dim']}")
        self.equiformer = Equiformer(**equiformer_config)
        self.config = qwen_vision_config
        self.merger = nn.Identity()

    def forward(
        self,
        atomic_numbers: torch.Tensor,
        coordinates: torch.Tensor,
        molecule_mask: torch.Tensor,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Tuple[torch.FloatTensor]:
        """
        Forward function of the wrapper.
        [Key fix] Normalize the dtype and device of all input tensors here.
        """
        # 1. Get the dtype (should be bfloat16) and device currently expected by the model
        #    This is the most robust method and adapts to whatever state the model is in.
        target_dtype = next(self.parameters()).dtype
        target_device = next(self.parameters()).device

        # 2. Before feeding data into Equiformer, force their types and devices
        #    atomic_numbers is long type and only needs to be moved to the device
        atomic_numbers = atomic_numbers.to(device=target_device)
        #    coordinates must be converted to the target dtype (bfloat16)
        coordinates = coordinates.to(device=target_device, dtype=target_dtype)
        #    molecule_mask is bool type and only needs to be moved to the device
        molecule_mask = molecule_mask.to(device=target_device)
        
        # 3. Now all inputs have the correct dtype and device and can be called safely
        molecule_features, _ = self.equiformer(
            inputs=atomic_numbers, 
            coors=coordinates, 
            mask=molecule_mask
        )
        
        return (molecule_features,)

    # --- The following are newly added methods ---
    def print_trainable_parameters(self):
        """
        Parameter printing function implemented for EquiformerWrapper to be compatible with the original training script.
        """
        print("--- Vision Module (Equiformer) ---")
        try:
            # Check whether the overall Equiformer module is trainable
            is_trainable = any(p.requires_grad for p in self.equiformer.parameters())
            
            if is_trainable:
                # Compute and print details of trainable parameters
                trainable_params = sum(p.numel() for p in self.equiformer.parameters() if p.requires_grad)
                total_params = sum(p.numel() for p in self.equiformer.parameters())
                print(f"  - Trainable: Yes")
                print(f"  - Trainable Parameters: {trainable_params} / {total_params} ({100 * trainable_params / total_params:.2f}%)")
            else:
                print(f"  - Trainable: No")

        except Exception as e:
            print(f"  - Could not determine trainable status: {e}")
        
        # Compatibility print: the original script checks merger, so we also print a status
        merger_trainable = any(p.requires_grad for p in self.merger.parameters())
        print(f"  - Merger Module Trainable: {merger_trainable}")
        print("------------------------------------")