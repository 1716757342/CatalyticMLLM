# Adopted from https://github.com/lm-sys/FastChat. Below is the original copyright:
# Adopted from tatsu-lab@stanford_alpaca. Below is the original copyright:
#    Copyright 2023 Rohan Taori, Ishaan Gulrajani, Tianyi Zhang, Yann Dubois, Xuechen Li
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.

import os
import logging
import pathlib
import torch
import torch.nn as nn

# ===============================
# [Fix]Linear layer type-conversion issue
# ===============================
# Save original Linear forward method
_original_linear_forward = nn.Linear.forward

def patched_linear_forward(self, input):
    """
    Fixed Linear forward method that avoids dynamically modifying weight types
    """
    # Ensure inputs and weights are on the same device and dtype
    if input.device != self.weight.device:
        input = input.to(self.weight.device)
    
    if input.dtype != self.weight.dtype:
        input = input.to(self.weight.dtype)
    
    # Use standard linear computation without modifying weight parameters
    return nn.functional.linear(input, self.weight, self.bias)

# Apply fix
print("🔧 Applying Linear layer dynamic type-conversion fix...")
nn.Linear.forward = patched_linear_forward
print("✅ Linear layer fix applied")
# ===============================

import transformers
import json
from typing import Dict
from typing import Optional, List, Union, Tuple
import shutil
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

# import qwenvl.train.trainer  # temporarily commented out to avoid flash_attn import issues
# from trainer import replace_qwen2_vl_attention_class

from transformers import (
    Qwen2VLForConditionalGeneration,
    Qwen2_5_VLForConditionalGeneration,
)
from qwenvl.data.data_qwen import make_supervised_data_module
from qwenvl.data.data_qwen_packed import make_supervised_data_module_packed
from qwenvl.train.argument import (
    ModelArguments,
    DataArguments,
    TrainingArguments,
)
from transformers import AutoTokenizer, AutoProcessor, Qwen2VLImageProcessor, Trainer
from qwenvl.model.equiformer_v2_wrapper import EquiformerV2Wrapper

from qwenvl.data.data_molecule import make_supervised_molecule_data_module

# Import modules created for molecular data
from qwenvl.data.data_molecule import make_supervised_molecule_data_module
# Import data_list to parse paths
from qwenvl.data import data_list 
from qwenvl.data.data_qwen import IMAGE_TOKEN_INDEX


from transformers import Qwen2_5_VLForConditionalGeneration
from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import Qwen2_5_VLCausalLMOutputWithPast
from peft import LoraConfig, get_peft_model, PeftModel

class Qwen2_5_VLForMolecule(Qwen2_5_VLForConditionalGeneration):
    # [[[Added __init__ Method]]]
    def __init__(self, config):
        # 1. First, call the parent __init__ method to build the base Qwen2.5-VL model
        super().__init__(config)
        
        # 2. Then define and create our new MLP Projector here
        #    This way, each Qwen2_5_VLForMolecule instance includes this projector
        input_dim = self.config.vision_config.hidden_size # e.g., 1280
        output_dim = self.config.hidden_size            # e.g., 2048
        
        self.molecule_projector = nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.GELU(),
            nn.Linear(output_dim, output_dim)
        )


    def forward(
        self,
        input_ids: torch.LongTensor,
        # Addedofmolecule-related inputs
        atomic_numbers: Optional[torch.LongTensor] = None,
        coordinates: Optional[torch.Tensor] = None,
        molecule_mask: Optional[torch.BoolTensor] = None,
        # [[[ Added cell argument ]]]
        cell: Optional[torch.Tensor] = None,
        # other original parameters
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        # For compatibility, accept but do not use original visual inputs
        pixel_values: Optional[torch.Tensor] = None,
        **kwargs, # accept any other possible parameters
    ) -> Union[Tuple, Qwen2_5_VLCausalLMOutputWithPast]:

        # ---- START: exactly mimic original Qwen-VL fusion logic ----
        
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if inputs_embeds is None:
            # Ensure input_ids is on the correct device
            device = next(self.parameters()).device
            input_ids = input_ids.to(device)
            
            # 1. First get text embeddings
            inputs_embeds = self.model.embed_tokens(input_ids)
    
            # 2. If molecule data exists, process it
            if atomic_numbers is not None and torch.sum(input_ids == IMAGE_TOKEN_INDEX) > 0:
                # Ensure all molecule data tensors are on the correct device
                atomic_numbers = atomic_numbers.to(device)
                coordinates = coordinates.to(device)
                molecule_mask = molecule_mask.to(device)
                if cell is not None:
                    cell = cell.to(device)
                
                # 2.1 Get molecule features through Equiformer and Projector
                # [[[ Pass cell to visual module ]]]
                vision_outputs = self.visual(
                    atomic_numbers=atomic_numbers,
                    coordinates=coordinates,
                    molecule_mask=molecule_mask,
                    cell=cell, # <-- pass cell through
                )
                molecule_embeds_raw = vision_outputs[0]
                molecule_embeds_projected = self.molecule_projector(molecule_embeds_raw)
                # print("Total valid atoms (molecule_mask True count):", molecule_mask.sum().item())
                # 2.2 Prepare source tensor needed for scatter
                # molecule_embeds_projected shape is (batch_size, max_atoms, embed_dim)
                # We need to reshape it to (total_atoms_in_batch, embed_dim)
                # Select only real atom features (according to molecule_mask)
                source_features = molecule_embeds_projected[molecule_mask]

                # 2.3 Prepare mask tensor needed for scatter
                # image_mask marks all placeholder positions in input_ids
                image_mask = (input_ids == IMAGE_TOKEN_INDEX)
                
                # 2.4 [key]Perform replacement operation (masked_scatter)
                # Expand mask to match embedding dimensions
                image_mask_expanded = image_mask.unsqueeze(-1).expand_as(inputs_embeds)
                
                # Now, image_mask_expanded the number of True values should equal total length of source_features
                # because preprocessing guarantees the number of placeholders equals the number of real atoms
                # image_mask_expanded = image_mask_expanded.to(torch.float32)  # ensure both dtypes match
                # source_features = source_features.to(torch.float32)  # ensure both dtypes match
                # inputs_embeds = inputs_embeds.to(torch.float32)  # ensure both dtypes match
                inputs_embeds = inputs_embeds.masked_scatter(image_mask_expanded, source_features)

        # ---- END: fusion logic ends ----

        # 3. Feed final inputs_embeds into main LLM
        outputs = self.model(
            input_ids=None, # because inputs_embeds has already been created
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )


        hidden_states = outputs[0]
        logits = self.lm_head(hidden_states)
        
        # [third modification]:complete loss computation logic
        loss = None
        if labels is not None:
            labels = labels.to(logits.device)
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = torch.nn.CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, self.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            loss = loss_fct(shift_logits, shift_labels)

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return Qwen2_5_VLCausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


local_rank = None


def rank0_print(*args):
    if local_rank == 0:
        print(*args)


def safe_save_model_for_hf_trainer(trainer: transformers.Trainer, output_dir: str):
    """Collects the state dict and dump to disk."""

    if trainer.deepspeed:
        torch.cuda.synchronize()
        trainer.save_model(output_dir)
        return

    state_dict = trainer.model.state_dict()
    if trainer.args.should_save:
        cpu_state_dict = {key: value.cpu() for key, value in state_dict.items()}
        del state_dict
        trainer._save(output_dir, state_dict=cpu_state_dict)  # noqa


def set_model(model_args, model):
    if model_args.tune_mm_vision:
        for n, p in model.visual.named_parameters():
            p.requires_grad = True
    else:
        for n, p in model.visual.named_parameters():
            p.requires_grad = False
    # [[[modify this logic]]]
    # Change operation target from old merger to our new projector
    if hasattr(model, 'molecule_projector'):
        if model_args.tune_mm_mlp:
            for n, p in model.molecule_projector.named_parameters():
                p.requires_grad = True
        else:
            for n, p in model.molecule_projector.named_parameters():
                p.requires_grad = False
    # [[[end modification]]]

    # if model_args.tune_mm_mlp:
    #     for n, p in model.visual.merger.named_parameters():
    #         p.requires_grad = True
    # else:
    #     for n, p in model.visual.merger.named_parameters():
    #         p.requires_grad = False

    if model_args.tune_mm_llm:
        for n, p in model.model.named_parameters():
            p.requires_grad = True
        model.lm_head.requires_grad = True
    else:
        for n, p in model.model.named_parameters():
            p.requires_grad = False
        model.lm_head.requires_grad = False


# [[[Final version: replace the train function in your file completely with this function]]]

def train(attn_implementation="flash_attention_2"):
    global local_rank

    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments)
    )
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    local_rank = training_args.local_rank
    os.makedirs(training_args.output_dir, exist_ok=True)

    # --- 1. Model loading ---
    if "qwen2.5" in model_args.model_name_or_path.lower():
        model = Qwen2_5_VLForMolecule.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=training_args.cache_dir,
            attn_implementation=attn_implementation,
            torch_dtype=(torch.bfloat16 if training_args.bf16 else None),
        )
        data_args.image_processor = AutoProcessor.from_pretrained(
            model_args.model_name_or_path,
        ).image_processor
        data_args.model_type = "qwen2.5vl"
    else:
        model = Qwen2VLForConditionalGeneration.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=training_args.cache_dir,
            attn_implementation=attn_implementation,
            torch_dtype=(torch.bfloat16 if training_args.bf16 else None),
        )
        data_args.image_processor = Qwen2VLImageProcessor.from_pretrained(
            model_args.model_name_or_path,
        )
        data_args.model_type = "qwen2vl"
    
    # --- 2. Model surgery: replace Vision Transformer ---
    if torch.distributed.get_rank() == 0:
        print("="*50)
        print("Perform model surgery: replacing ViT with pretrained EquiformerV2...")
    equiformer_v2_config_path = 'pretrained_equiformer_v2/equiformer_v2_N@8_L@4_M@2_31M.yml'
    equiformer_v2_weights_path = 'pretrained_equiformer_v2/eq2_31M_ec4_allmd.pt'
    original_vision_config = model.config.vision_config
    equiformer_v2_wrapper = EquiformerV2Wrapper(
        config_file=equiformer_v2_config_path,
        pretrained_path=equiformer_v2_weights_path,
        qwen_vision_config=original_vision_config
    )
    model.visual = equiformer_v2_wrapper
    target_dtype = torch.bfloat16 if training_args.bf16 else (torch.float16 if training_args.fp16 else model.dtype)
    target_device = training_args.device
    model.visual.to(device=target_device, dtype=target_dtype)
    if torch.distributed.get_rank() == 0:
        print("Model surgery complete.")
        print("="*50)
        
    if data_args.data_flatten:
        print('4'*100)
        # replace_qwen2_vl_attention_class() # Assuming this is defined somewhere
    model.config.use_cache = False

    # --- 3. Set model parameters (LoRA or full-tuning) and print status ---
    if model_args.use_lora:
        rank0_print("INFO: LoRA is enabled. Preparing model for PEFT...")
        lora_config = LoraConfig(
            r=model_args.lora_r,
            lora_alpha=model_args.lora_alpha,
            target_modules=model_args.lora_target_modules,
            lora_dropout=model_args.lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
        )
        set_model(model_args, model)
        model = get_peft_model(model, lora_config)
    else:
        rank0_print("INFO: LoRA is disabled. Using standard fine-tuning.")
        set_model(model_args, model)

    if training_args.gradient_checkpointing:
        model.enable_input_require_grads()

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        model_max_length=training_args.model_max_length,
        padding_side="left",  # Flash Attention 2 requires left padding
        use_fast=False,
    )
    
    # Explicitly ensure padding_side is left (to prevent model config from overriding it)
    tokenizer.padding_side = "left"

    # Unified, correct parameter-printing logic
    if torch.distributed.get_rank() == 0:
        print("="*50)
        print("--- Final Trainable Parameters Status ---")
        if model_args.use_lora:
            model.print_trainable_parameters()
        else:
            # Print Vision module parameters
            model.visual.print_trainable_parameters()
            
            # Manually compute and print trainable parameters for the LLM part
            trainable_params = 0
            total_params = 0
            for name, param in model.model.named_parameters():
                total_params += param.numel()
                if param.requires_grad:
                    trainable_params += param.numel()
            
            print("--- Language Model (LLM) ---")
            print(f" - Trainable: {'Yes' if trainable_params > 0 else 'No'}")
            print(f" - Trainable Parameters: {trainable_params} / {total_params} ({trainable_params/total_params*100:.2f}%)")
        print("="*50)

    # --- 4. dataload ---
    if "MOLECULE_RELAXED_ENERGY" in data_args.dataset_use:
        if torch.distributed.get_rank() == 0:
            print("INFO: Detected molecule dataset. Using `make_supervised_molecule_data_module`.")
        dataset_name_key = data_args.dataset_use.split('%')[0]
        dataset_config = data_list([dataset_name_key])[0]
        data_args.dataset_use = dataset_config['annotation_path'] 
        data_module = make_supervised_molecule_data_module(tokenizer=tokenizer, data_args=data_args)
    elif data_args.data_packing:
        data_module = make_supervised_data_module_packed(tokenizer=tokenizer, data_args=data_args)
    else:
        data_module = make_supervised_data_module(tokenizer=tokenizer, data_args=data_args)
    
    # --- 5. Trainer initialization and training ---
    trainer = Trainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        **data_module
    )

    if list(pathlib.Path(training_args.output_dir).glob("checkpoint-*")):
        logging.info("checkpoint found, resume training")
        trainer.train(resume_from_checkpoint=True)
    else:
        trainer.train()
        
    trainer.save_state()
    # Assuming data_args.image_processor is correctly defined
    if hasattr(data_args, 'image_processor'):
        data_args.image_processor.save_pretrained(training_args.output_dir)

    model.config.use_cache = True
    safe_save_model_for_hf_trainer(trainer=trainer, output_dir=training_args.output_dir)

if __name__ == "__main__":
    train(attn_implementation="flash_attention_2")
