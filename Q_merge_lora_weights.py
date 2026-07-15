# import torch
# import os
# import sys
# from pathlib import Path
# from transformers import AutoTokenizer

# # [[[Added]]]: dynamicadd projectpath
# # ------------------------------------------------------------------
# # Get the directory where this script is located (Qwen2.5-VL-main)
# script_dir = Path(__file__).resolve().parent

# # The 'qwenvl' module is inside the 'qwen-vl-finetune' subdirectory
# finetune_dir = script_dir / 'qwen-vl-finetune'

# # Add the 'qwen-vl-finetune' directory to Python's search path
# if str(finetune_dir) not in sys.path:
#     sys.path.insert(0, str(finetune_dir))
#     print(f"✅ Temporarily added to sys.path: {finetune_dir}")
# # ------------------------------------------------------------------

# from qwenvl.train.train_qwen import Qwen2_5_VLForMolecule
# from qwenvl.model.equiformer_v2_wrapper import EquiformerV2Wrapper
# from peft import PeftModel

# def merge_lora_model(base_model_path, lora_adapter_path, output_path):
#     """
#     Merge the LoRA adapter into the base model and save it.
#     """
#     print("="*50)
#     print(f"Loading base model from: {base_model_path}")
    
#     # 1. Load base model (same as during training)
#     base_model = Qwen2_5_VLForMolecule.from_pretrained(
#         base_model_path,
#         torch_dtype=torch.bfloat16,
#         device_map="cpu", # merge on CPU to save GPU memory
#     )

#     print(f"Loading LoRA adapter from: {lora_adapter_path}")

#     # 2. Load adapter from LoRA directory
#     # PeftModel will automatically apply the adapter to the base model
#     lora_model = PeftModel.from_pretrained(base_model, lora_adapter_path)

#     print("Merging LoRA adapters into the base model...")
#     # 3. Execute merge
#     merged_model = lora_model.merge_and_unload()
#     print("Merge complete.")

#     # 4. Save the fully merged model
#     os.makedirs(output_path, exist_ok=True)
#     print(f"Saving merged model to: {output_path}")
#     merged_model.save_pretrained(output_path)

#     # 5. Save tokenizer
#     tokenizer = AutoTokenizer.from_pretrained(base_model_path)
#     tokenizer.save_pretrained(output_path)
    
#     print("="*50)
#     print("✅ Merged model and tokenizer have been saved successfully!")
#     print(f"You can now load the full model from: {output_path}")


# if __name__ == "__main__":
#     # --- [Configure your paths here] ---
#     # Base model path (the model used when starting training, not the LoRA training output directory)
#     BASE_MODEL_PATH = "/path/to/base_model"
    
#     # LoRA adapter path (your LoRA training output directory)
#     LORA_ADAPTER_PATH = "/path/to/lora_adapter"
    
#     # Save path for the fully merged model
#     MERGED_MODEL_OUTPUT_PATH = "/path/to/merged_model"

#     # Note: before running this script, you need to manually execute your "model surgery" logic, 
#     # or rebuild the model structure in the script like this.
#     # Because your `Qwen2_5_VLForMolecule` and `EquiformerV2Wrapper` are custom, 
#     # we need to ensure these classes are registered correctly before loading the model. The imports above handle this.
    
#     merge_lora_model(BASE_MODEL_PATH, LORA_ADAPTER_PATH, MERGED_MODEL_OUTPUT_PATH)

import torch
import os
import sys
from pathlib import Path
# [[[Added]]] Import AutoProcessor
from transformers import AutoTokenizer, AutoProcessor
from peft import PeftModel

# --- dynamicadd projectpath ---
script_dir = Path(__file__).resolve().parent
finetune_dir = script_dir / 'qwen-vl-finetune'
if str(finetune_dir) not in sys.path:
    sys.path.insert(0, str(finetune_dir))
    print(f"✅ Temporarily added to sys.path: {finetune_dir}")

# --- Import custom modules ---
from qwenvl.train.train_qwen import Qwen2_5_VLForMolecule
from qwenvl.model.equiformer_v2_wrapper import EquiformerV2Wrapper


def merge_lora_model(base_model_path, lora_adapter_path, output_path):
    """
    Merge the LoRA adapter into the base model and save it with the full processor configuration.
    """
    print("="*50)
    print(f"🚀 Loading base model from: {base_model_path}")
    
    # 1. Load base model
    base_model = Qwen2_5_VLForMolecule.from_pretrained(
        base_model_path,
        torch_dtype=torch.bfloat16,
        device_map="cpu",
    )

    print(f"📂 Loading LoRA adapter from: {lora_adapter_path}")
    # 2. Load adapter from LoRA directory
    lora_model = PeftModel.from_pretrained(base_model, lora_adapter_path)

    print("🔄 Merging LoRA adapters into the base model...")
    # 3. Execute merge
    merged_model = lora_model.merge_and_unload()
    print("Merge complete.")

    # 4. Save the fully merged model
    os.makedirs(output_path, exist_ok=True)
    print(f"💾 Saving merged model to: {output_path}")
    merged_model.save_pretrained(output_path)

    # 5. [[[Fixed section]]]Save the full processor (Tokenizer + Image Processor)
    print(f"📝 Loading and saving the full processor...")
    # Load the full Processor from the base model path
    processor = AutoProcessor.from_pretrained(base_model_path)
    # Save it to the new output path; this automatically creates preprocessor_config.json and all required files
    processor.save_pretrained(output_path)
    
    print("="*50)
    print(f"🎉 Success! Merged model and full processor are ready at: {output_path}")


if __name__ == "__main__":
    # --- [Configure your paths here] ---
    # Base model path
    BASE_MODEL_PATH = "/path/to/base_model"
    
    # LoRA adapter path (training output directory)
    LORA_ADAPTER_PATH = "/path/to/lora_adapter"
    
    
    # Save path for the fully merged model
    # This script will overwrite or create this directory: /path/to/merged_model
    MERGED_MODEL_OUTPUT_PATH = "/path/to/merged_model"

    merge_lora_model(BASE_MODEL_PATH, LORA_ADAPTER_PATH, MERGED_MODEL_OUTPUT_PATH)