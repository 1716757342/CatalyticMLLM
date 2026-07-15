#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Merge LoRA weights into the base model
Supports command-line arguments
"""

import torch
import os
import sys
import argparse
from pathlib import Path
from transformers import AutoTokenizer, AutoProcessor
from peft import PeftModel

# --- dynamicadd projectpath ---
script_dir = Path(__file__).resolve().parent
finetune_dir = script_dir / 'qwen-vl-finetune'
if str(finetune_dir) not in sys.path:
    sys.path.insert(0, str(finetune_dir))

# --- Import custom modules ---
from qwenvl.train.train_qwen import Qwen2_5_VLForMolecule


def merge_lora_model(base_model_path, lora_adapter_path, output_path, device_map="cpu"):
    """
    Merge the LoRA adapter into the base model and save it with the full processor configuration.
    
    Args:
        base_model_path: Base model path
        lora_adapter_path: LoRA adapter path
        output_path: Output path
        device_map: Device map ("cpu", "cuda", "auto")
    """
    print("="*80)
    print("LoRA weight merge tool")
    print("="*80)
    print(f"\n📂 Base model path: {base_model_path}")
    print(f"📂 LoRA adapter path: {lora_adapter_path}")
    print(f"📂 Output path: {output_path}")
    print(f"🖥️  device: {device_map}")
    print()
    
    # Check whether paths exist
    if not os.path.exists(base_model_path):
        raise ValueError(f"❌ Base model path does not exist: {base_model_path}")
    if not os.path.exists(lora_adapter_path):
        raise ValueError(f"❌ LoRA adapter path does not exist: {lora_adapter_path}")
    
    # 1. Load base model
    print("🚀 Step 1/5: Load base model...")
    try:
        base_model = Qwen2_5_VLForMolecule.from_pretrained(
            base_model_path,
            torch_dtype=torch.bfloat16,
            device_map=device_map,
            trust_remote_code=True,
            local_files_only=True,
        )
        print("✓ Base model loaded successfully")
    except Exception as e:
        print(f"❌ Failed to load base model: {e}")
        raise

    # 2. Load LoRA adapter
    print("\n🔌 Step 2/5: Load LoRA adapter...")
    try:
        lora_model = PeftModel.from_pretrained(base_model, lora_adapter_path)
        print("✓ LoRA adapter loaded successfully")
        
        # Print LoRA configuration information
        print("\nLoRA configuration:")
        if hasattr(lora_model, 'peft_config'):
            peft_config = lora_model.peft_config.get('default', None)
            if peft_config:
                print(f"  - Rank (r): {peft_config.r}")
                print(f"  - Alpha: {peft_config.lora_alpha}")
                print(f"  - Dropout: {peft_config.lora_dropout}")
                print(f"  - Target modules: {peft_config.target_modules}")
    except Exception as e:
        print(f"❌ Failed to load LoRA adapter: {e}")
        raise

    # 3. Merge weights
    print("\n🔄 Step 3/5: Merge LoRA weights into the base model...")
    try:
        merged_model = lora_model.merge_and_unload()
        print("✓ Weights merged successfully")
    except Exception as e:
        print(f"❌ Failed to merge weights: {e}")
        raise

    # 4. Save merged model
    print(f"\n💾 Step 4/5: Saving merged model to {output_path}...")
    try:
        os.makedirs(output_path, exist_ok=True)
        merged_model.save_pretrained(output_path)
        print("✓ Model saved successfully")
    except Exception as e:
        print(f"❌ Failed to save model: {e}")
        raise

    # 5. saveprocessor (Tokenizer + Image Processor)
    print("\n📝 Step 5/5: Save processor configuration...")
    try:
        processor = AutoProcessor.from_pretrained(
            base_model_path,
            trust_remote_code=True,
            local_files_only=True
        )
        processor.save_pretrained(output_path)
        print("✓ Processor saved successfully")
    except Exception as e:
        print(f"⚠️  Failed to save processor: {e}")
        print("   The model was saved, but processor configuration files may need to be copied manually")
    
    print("\n" + "="*80)
    print("🎉 Success! The merged model has been saved")
    print("="*80)
    print(f"\nOutput path: {output_path}")
    print("\nYou can load the merged model with the following code:")
    print(f"""
from transformers import AutoModel, AutoProcessor

model = AutoModel.from_pretrained(
    "{output_path}",
    torch_dtype=torch.bfloat16,
    device_map="auto",
    trust_remote_code=True
)
processor = AutoProcessor.from_pretrained("{output_path}", trust_remote_code=True)
""")
    print("="*80)


def main():
    parser = argparse.ArgumentParser(
        description="Merge LoRA adapter weights into the base model",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
example:
  python merge_lora_weights.py \\
      --base_model /path/to/base_model \\
      --lora_adapter /path/to/lora_adapter \\
      --output_path /path/to/output

  # Merge with GPU (faster)
  python merge_lora_weights.py \\
      --base_model /path/to/base_model \\
      --lora_adapter /path/to/lora_adapter \\
      --output_path /path/to/output \\
      --device_map auto
        """
    )
    
    parser.add_argument(
        "--base_model",
        type=str,
        required=True,
        help="Base model path (original model before training)"
    )
    parser.add_argument(
        "--lora_adapter",
        type=str,
        required=True,
        help="LoRA adapter path (checkpoint from training output)"
    )
    parser.add_argument(
        "--output_path",
        type=str,
        required=True,
        help="Save path for the merged model"
    )
    parser.add_argument(
        "--device_map",
        type=str,
        default="cpu",
        choices=["cpu", "cuda", "auto"],
        help="Device map: cpu (saves memory), cuda (single GPU), auto (automatic multi-GPU allocation)"
    )
    
    args = parser.parse_args()
    
    # Execute merge
    merge_lora_model(
        base_model_path=args.base_model,
        lora_adapter_path=args.lora_adapter,
        output_path=args.output_path,
        device_map=args.device_map
    )


if __name__ == "__main__":
    main()
