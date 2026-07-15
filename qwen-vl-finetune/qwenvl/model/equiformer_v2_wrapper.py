# File path: Qwen2.5_vl_EQU2/Qwen2.5-VL-main/qwen-vl-finetune/qwenvl/model/equiformer_v2_wrapper.py

import torch
import torch.nn as nn
import yaml
from typing import Optional, Tuple
import sys
from pathlib import Path
from types import SimpleNamespace
import numpy as np # 【【【 1. Import numpy 】】】


# --- Dynamic path addition section (verified, keep unchanged) ---
try:
    project_root = Path(__file__).resolve().parents[3] 
    equiformer_v2_root = project_root / 'equiformer_v2_all'
    if not equiformer_v2_root.exists():
        raise FileNotFoundError(f"Directory not found: {equiformer_v2_root}")
    if str(equiformer_v2_root) not in sys.path:
        sys.path.insert(0, str(equiformer_v2_root))
        print(f"Successfully added path: {equiformer_v2_root}")
    from nets.equiformer_v2.equiformer_v2_oc20 import EquiformerV2_OC20
    print("Successfully imported the EquiformerV2_OC20 module.")
except (ImportError, FileNotFoundError) as e:
    raise

class EquiformerV2Wrapper(nn.Module):
    def __init__(self, config_file: str, pretrained_path: str, qwen_vision_config):
        super().__init__()
        self.device = torch.device("cpu")

        # 1. Load config
        with open(config_file, 'r') as f:
            equiformer_v2_config = yaml.safe_load(f)
        model_args = equiformer_v2_config['model']
        model_args.pop('name', None)
        
        # 2. Instantiate the model
        self.equiformer = EquiformerV2_OC20(
            num_atoms=1, bond_feat_dim=1, num_targets=1, **model_args
        )

        # 3. Load pretrained weights
        try:
            state_dict = torch.load(pretrained_path, map_location="cpu")
            model_state_dict = state_dict.get('state_dict', state_dict.get('model', state_dict))
            new_state_dict = {k.replace('module.', '', 1): v for k, v in model_state_dict.items()}
            self.equiformer.load_state_dict(new_state_dict, strict=False)
            print("EquiformerV2 pretrained weights loaded successfully.")
        except Exception as e:
            print(f"Error loading pretrained weights: {e}. The model will be initialized with random weights.")

        # 4. Create the final projection layer
        self.final_projector = nn.Linear(self.equiformer.sphere_channels, qwen_vision_config.hidden_size)
        self.config = qwen_vision_config
        self.merger = nn.Identity()

    # 【【【 2. Add a helper function to convert cell parameters 】】】
    @staticmethod
    def _get_cell_matrix(cell_params: torch.Tensor) -> torch.Tensor:
        """
        Convert [N, 6] cell parameters (a, b, c, alpha, beta, gamma) to [N, 3, 3] cell matrices.
        OC20/ASE convention.
        """
        # angles are in degrees, convert to radians
        alpha, beta, gamma = torch.deg2rad(cell_params[:, 3:]).T

        cos_alpha, cos_beta, cos_gamma = torch.cos(alpha), torch.cos(beta), torch.cos(gamma)
        sin_gamma = torch.sin(gamma)

        # Volume calculation
        volume = (
            cell_params[:, 0] * cell_params[:, 1] * cell_params[:, 2] *
            torch.sqrt(
                1 - cos_alpha**2 - cos_beta**2 - cos_gamma**2 + 2 * cos_alpha * cos_beta * cos_gamma
            )
        )

        cell = torch.zeros((cell_params.shape[0], 3, 3), device=cell_params.device, dtype=torch.float32)
        
        cell[:, 0, 0] = cell_params[:, 0]
        cell[:, 0, 1] = cell_params[:, 1] * cos_gamma
        cell[:, 0, 2] = cell_params[:, 2] * cos_beta

        cell[:, 1, 1] = cell_params[:, 1] * sin_gamma
        cell[:, 1, 2] = cell_params[:, 2] * (cos_alpha - cos_beta * cos_gamma) / sin_gamma

        cell[:, 2, 2] = volume / (cell_params[:, 0] * cell_params[:, 1] * sin_gamma)

        return cell.transpose(1, 2) # Transpose to get vectors as columns

    def to(self, *args, **kwargs):
        result = super().to(*args, **kwargs)
        # Update internal device tracking
        new_device = None
        if 'device' in kwargs:
            new_device = kwargs['device']
        elif len(args) > 0 and isinstance(args[0], (torch.device, str)):
            new_device = torch.device(args[0])
        if new_device: 
            self.device = new_device
        return result

    # 【【【 3. Modify the forward method to receive and use cell 】】】
    def forward(
        self,
        atomic_numbers: torch.LongTensor,
        coordinates: torch.Tensor,
        molecule_mask: torch.BoolTensor,
        cell: Optional[torch.Tensor] = None, # <-- Receive cell
        **kwargs
    ) -> Tuple[torch.FloatTensor]:
        
        # Ensure all inputs are on the correct device
        device = coordinates.device  # Use the input tensor device instead of self.device
        
        batch_size = coordinates.shape[0]
        num_atoms_per_molecule = torch.sum(molecule_mask, dim=1)
        
        batch_tensor = torch.repeat_interleave(
            torch.arange(batch_size, device=device),  # Use the correct device
            num_atoms_per_molecule
        )
        
        # Ensure all tensors are on the correct device
        mock_data_dict = {
            'pos': coordinates[molecule_mask].to(device),
            'atomic_numbers': atomic_numbers[molecule_mask].to(device),
            'batch': batch_tensor,
            'natoms': num_atoms_per_molecule.to(device)
        }

        # [[[ New code block starts ]]]
        # If cell parameters are provided, compute the cell matrix and add it to mock_data
        if cell is not None:
            # Check whether any nonzero cell parameters exist to determine whether to apply PBC
            # (A simple heuristic; a more robust method is to check whether alpha/beta/gamma are 90)
            is_periodic = torch.any(cell > 0, dim=1)
            if torch.any(is_periodic):
                # Convert [B, 6] -> [B, 3, 3]
                cell_matrix = self._get_cell_matrix(cell) 
                
                # EquiformerV2_OC20 requires cell and pbc flags
                mock_data_dict['cell'] = cell_matrix
                # Assume all directions are periodic
                mock_data_dict['pbc'] = torch.tensor([True, True, True], device=device).expand(batch_size, 3)
        # [[[ New code block ends ]]]

        mock_data = SimpleNamespace(**mock_data_dict)

        # 1. Run the forward pass. Because we removed the no_grad decorator,
        #    the computation graph for backpropagation will be built automatically here.
        outputs_dict = self.equiformer(mock_data)

        # 2. Get embeddings directly from the returned dictionary
        node_embedding = outputs_dict['embedding']

        # 3. Extract scalar features [N, D] from [N, 25, D]
        scalar_features = node_embedding[:, 0, :]

        # 4. Project features
        projected_features = self.final_projector(scalar_features)
        
        # 5. Prepare output
        output_tensor = torch.zeros(
            batch_size, 
            coordinates.shape[1],
            projected_features.shape[-1],
            device=device, 
            dtype=projected_features.dtype
        )
        output_tensor[molecule_mask] = projected_features
        
        return (output_tensor,)
        
    def print_trainable_parameters(self):
        print("--- Vision Module (EquiformerV2) ---")
        try:
            is_trainable = any(p.requires_grad for p in self.equiformer.parameters())
            if is_trainable:
                trainable_params = sum(p.numel() for p in self.equiformer.parameters() if p.requires_grad)
                total_params = sum(p.numel() for p in self.equiformer.parameters())
                print(f" - Trainable: Yes")
                print(f" - Trainable Parameters: {trainable_params} / {total_params} ({100 * trainable_params / total_params:.2f}%)")
            else:
                print(f" - Trainable: No")
        except Exception as e:
            print(f" - Could not determine trainable status: {e}")