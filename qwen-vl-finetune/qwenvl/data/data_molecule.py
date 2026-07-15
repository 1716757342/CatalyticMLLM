# qwen-vl-finetune/qwenvl/data/data_molecule.py

import json
import torch
from torch.utils.data import Dataset
from dataclasses import dataclass
import transformers
from typing import Dict, Sequence

# Ensure required functions and constants can be imported from data_qwen.py in the same directory
from .data_qwen import preprocess_qwen_2_visual, IGNORE_INDEX, IMAGE_TOKEN_INDEX

# qwen-vl-finetune/qwenvl/data/data_molecule.py

# Ensure required functions and constants can be imported from data_qwen.py in the same directory
class LazyMoleculeDataset(Dataset):
    """
    A Dataset that can load both 3D molecule data and plain-text data.
    It checks whether each sample contains a "molecule" field.
    """
    def __init__(self, json_path: str, tokenizer: transformers.PreTrainedTokenizer):
        super(LazyMoleculeDataset, self).__init__()
        self.tokenizer = tokenizer
      
        with open(json_path, 'r') as f:
            self.annotations = json.load(f)
      
        # Count samples of different types
        molecule_samples = sum(1 for ann in self.annotations if "molecule" in ann and ann["molecule"])
        text_samples = len(self.annotations) - molecule_samples
        print(f"Successfully loaded {len(self.annotations)} data samples, including {molecule_samples} molecule samples and {text_samples} plain-text samples.")

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        source_data = self.annotations[i]
        conversations = source_data["conversations"]

        # Check whether this is a valid molecule sample
        is_molecule_sample = "molecule" in source_data and source_data["molecule"]

        if is_molecule_sample:
            molecule_data = source_data["molecule"]
            num_atoms = len(molecule_data["z"])

            # Insert placeholders for molecule samples
            data_dict = preprocess_qwen_2_visual(
                [conversations],
                self.tokenizer,
                grid_thw_image=[num_atoms],
            )

            item = {
                "input_ids": data_dict["input_ids"].squeeze(0),
                "labels": data_dict["labels"].squeeze(0),
                "atomic_numbers": torch.tensor(molecule_data["z"], dtype=torch.long),
                "coordinates": torch.tensor(molecule_data["pos"], dtype=torch.float32),
                "is_molecule": torch.tensor(True) # Add a flag
            }
            
            # [[[ New code block starts ]]]
            # Check and add cell parameters
            if "cell" in molecule_data:
                item["cell"] = torch.tensor(molecule_data["cell"], dtype=torch.float32)
            # [[[ New code block ends ]]]

        else:
            # For plain-text samples, do not insert any placeholders
            # Note: Ensure plain-text sample prompts (value) do not contain the <image> string
            data_dict = preprocess_qwen_2_visual(
                [conversations],
                self.tokenizer,
                grid_thw_image=[], # Pass an empty list
            )
            item = {
                "input_ids": data_dict["input_ids"].squeeze(0),
                "labels": data_dict["labels"].squeeze(0),
                "is_molecule": torch.tensor(False) # Add a flag
            }
        return item


@dataclass
class DataCollatorForMoleculeDataset(object):
    """
    DataCollator designed for mixed molecule/text datasets.
    It intelligently handles batches containing plain-text, molecule-only, or mixed samples.
    """
    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        # 1. Pad the text portion (common to all samples)
        input_ids, labels = tuple([instance[key] for instance in instances]
                                  for key in ("input_ids", "labels"))
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
        )
        labels = torch.nn.utils.rnn.pad_sequence(
            labels, batch_first=True, padding_value=IGNORE_INDEX
        )
        
        # 2. Determine the maximum number of atoms from molecule samples in the batch
        max_atoms = max(
            (len(inst["atomic_numbers"]) for inst in instances if inst.get("is_molecule", False)),
            default=0
        )

        padded_atomic_numbers = []
        padded_coordinates = []
        molecule_mask = []
        
        # [[[ New cell list ]]]
        padded_cells = []

        # 3. Iterate over all samples, padding molecule data or generating dummy data
        for instance in instances:
            if instance.get("is_molecule", False):
                # For molecule samples, pad normally
                num_atoms = len(instance["atomic_numbers"])
                pad_len = max_atoms - num_atoms
                
                padded_z = torch.cat([instance["atomic_numbers"], torch.zeros(pad_len, dtype=torch.long)])
                padded_pos = torch.cat([instance["coordinates"], torch.zeros(pad_len, 3, dtype=torch.float32)])
                mask = torch.cat([torch.ones(num_atoms, dtype=torch.bool), torch.zeros(pad_len, dtype=torch.bool)])
                
                # [[[ New code block starts ]]]
                # Use cell if it exists; otherwise create an all-zero dummy tensor
                if "cell" in instance:
                    cell_tensor = instance["cell"]
                else:
                    cell_tensor = torch.zeros(6, dtype=torch.float32)
                padded_cells.append(cell_tensor)
                # [[[ New code block ends ]]]

            else:
                # For plain-text samples, generate dummy tensors matching max_atoms
                padded_z = torch.zeros(max_atoms, dtype=torch.long)
                padded_pos = torch.zeros(max_atoms, 3, dtype=torch.float32)
                mask = torch.zeros(max_atoms, dtype=torch.bool)
                
                # [[[ New code block starts ]]]
                # Plain-text samples also need a dummy cell tensor to keep batch dimensions consistent
                padded_cells.append(torch.zeros(6, dtype=torch.float32))
                # [[[ New code block ends ]]]

            padded_atomic_numbers.append(padded_z)
            padded_coordinates.append(padded_pos)
            molecule_mask.append(mask)

        # 4. Final consistency check
        batch_image_tokens = (input_ids == IMAGE_TOKEN_INDEX).sum().item()
        batch_num_atoms = torch.stack(molecule_mask).sum().item()

        assert batch_image_tokens == batch_num_atoms, (
             f"The number of placeholders in the batch ({batch_image_tokens}) does not match the total number of atoms ({batch_num_atoms}) Please check the data and preprocessing logic."
        )

        # 5. Assemble the final batch
        batch = dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
            atomic_numbers=torch.stack(padded_atomic_numbers),
            coordinates=torch.stack(padded_coordinates),
            molecule_mask=torch.stack(molecule_mask),
            # [[[ Add cell to the batch ]]]
            cell=torch.stack(padded_cells)
        )
        return batch

def make_supervised_molecule_data_module(tokenizer: transformers.PreTrainedTokenizer, data_args) -> Dict:
    """Factory function for creating a molecule dataset (now compatible with mixed data)"""
    train_dataset = LazyMoleculeDataset(json_path=data_args.dataset_use, tokenizer=tokenizer)
    data_collator = DataCollatorForMoleculeDataset(tokenizer=tokenizer)
    return dict(train_dataset=train_dataset, eval_dataset=None, data_collator=data_collator)