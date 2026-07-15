

import torch
from transformers import AutoProcessor
import os
import json
import re
import csv
import yaml
from tqdm import tqdm
from typing import Optional, List, Union, Tuple
import numpy as np
import sys
from pathlib import Path
import random
from types import SimpleNamespace

# ==============================================================================
# --- Step 1: Environment setup and custom module imports ---
# ==============================================================================
print("="*60)
print("📦 Setting up the environment and importing custom modules...")

# --- Dynamic path setup (for finding custom class definitions) ---
# Ensure this script can find your `qwen-vl-finetune` directory
try:
    # Assume the script and 'qwen-vl-finetune' are under the same parent directory
    qwen_finetune_root = Path(__file__).resolve().parent / 'qwen-vl-finetune'
    if not qwen_finetune_root.is_dir():
        # If the structure differs, hard-code the correct path here
        # qwen_finetune_root = Path("/path/to/your/Qwen2.5-VL-main/qwen-vl-finetune")
        raise FileNotFoundError
        
    if str(qwen_finetune_root) not in sys.path:
        sys.path.append(str(qwen_finetune_root))
    
    # Import the custom model class from your project
    from qwenvl.train.train_qwen import Qwen2_5_VLForMolecule
    print("✅ Successfully imported custom model class Qwen2_5_VLForMolecule.")

except (ImportError, FileNotFoundError) as e:
    print(f"❌ Error: unable to import custom modules. Ensure this script can locate 'qwen-vl-finetune' directory.")
    print(f"   Current attempted path: {qwen_finetune_root}")
    print(f"   Detailed error: {e}")
    sys.exit(1)

try:
    import ase.data
    import ase.io
    print("✅ ASE library found.")
except ImportError:
    print("❌ Error: 'ase' library not found. Please run 'pip install ase'.")
    sys.exit(1)


# ==============================================================================
# --- Step 2: Configuration items ---
# ==============================================================================
print("\n" + "="*60)
print("⚙️  Loading configuration...")


# Evaluate using the model trained with GRPO
FINETUNED_MODEL_PATH = "/path/to/finetuned_model"
JSON_DATA_PATH = "/path/to/training_data.json"
# Create a timestamped output directory
from datetime import datetime
TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
OUTPUT_DIR = f"/path/to/evaluation_results_{TIMESTAMP}"
os.makedirs(OUTPUT_DIR, exist_ok=True)
CSV_OUTPUT_PATH = os.path.join(OUTPUT_DIR, "grpo_evaluation_report.csv")
NUM_SAMPLES_TO_TEST = 50 # Test 50 samples first
OUTLIER_THRESHOLD = 38 # Samples with absolute error above this value are treated as outliers

print(f"   - Model path: {FINETUNED_MODEL_PATH}")
print(f"   - Data path: {JSON_DATA_PATH}")
print(f"   - Output directory: {OUTPUT_DIR}")
print(f"   - Number of test samples: {NUM_SAMPLES_TO_TEST}")
print(f"   - Outlier threshold: {OUTLIER_THRESHOLD}")

# ==============================================================================
# --- Step 3: Model and processor initialization ---
# ==============================================================================
print("\n" + "="*60)
print("🚀 Initializing model...")
# Use GPU 6 for evaluation to avoid conflicts with training
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")  # The sixth GPU is visible to the OS and mapped to cuda:0
print(f"   - Using device: {device}")

try:
    # Key change:`from_pretrained` automatically handles everything, including loading the correct fine-tuned visual-module weights
    model = Qwen2_5_VLForMolecule.from_pretrained(
        FINETUNED_MODEL_PATH, 
        torch_dtype=torch.bfloat16, 
        trust_remote_code=True # trust_remote_code allows loading custom Python code from your project
    ).to(device).eval()
    
    processor = AutoProcessor.from_pretrained(FINETUNED_MODEL_PATH, trust_remote_code=True)
    IMAGE_TOKEN = "<image>"
    print("✅ Model and processor initialized successfully. Full fine-tuned weights are now being used.")
except Exception as e:
    print(f"❌ Initialization failed: {e}")
    sys.exit(1)

# ==============================================================================
# --- Step 4, 5, 6: data loading, evaluation loop, and report generation (these parts do not need changes) ---
# ==============================================================================
# ... (In the script you provided, everything from "Data loading and helper functions" to the end does not need changes, 
#      because the core logic is correct. Just ensure the model loading above is correct.)

# (The remaining unchanged parts of your script are omitted here for brevity.
#  You only need to replace the corresponding first half of your original script with this file.)

# Paste the remaining part as an example to keep the script complete

# --- Data loading and helper functions ---
print("\n" + "="*60)
print(f"📚 Loading data...")
# ... (This part of the code is unchanged)
try:
    with open(JSON_DATA_PATH, 'r', encoding='utf-8') as f:
        all_data = json.load(f)
    if NUM_SAMPLES_TO_TEST is not None and len(all_data) > NUM_SAMPLES_TO_TEST:
        print(f"   - Total number of samples in dataset: {len(all_data)}.Randomly sample {NUM_SAMPLES_TO_TEST} samples for testing.")
        all_data = random.sample(all_data, NUM_SAMPLES_TO_TEST)
    else:
        print(f"   - Will use all {len(all_data)} samples for evaluation.")
except Exception as e:
    print(f"❌ Failed to load data: {e}")
    sys.exit(1)

def parse_numerical_value(text: str) -> Optional[float]:
    """Extract the last floating-point number from a string."""
    matches = re.findall(r"[-+]?\d*\.\d+|\d+", text)
    return float(matches[-1]) if matches else None

# --- Unified batch evaluation loop ---
def run_unified_batch_evaluation():
    # ... (the full contents of this function do not need changes)
    results_by_type = {
        '3D_Molecule_Input': {'errors': [], 'success': 0, 'fail': 0, 'outliers': 0},
        'Text_Only_Input': {'errors': [], 'success': 0, 'fail': 0, 'outliers': 0}
    }
    
    print("\n" + "="*60)
    print("📊 Starting unified batch evaluation...")
    
    with open(CSV_OUTPUT_PATH, 'w', newline='', encoding='utf-8') as csv_file:
        csv_writer = csv.writer(csv_file)
        header = ['ID', 'Input type', 'question', 'Model predicted answer', 'Ground-truth answer', 'Absolute error', 'status']
        csv_writer.writerow(header)

        progress_bar = tqdm(all_data, desc="Evaluation progress", unit="items")

        for item in progress_bar:
            item_id = item.get('id', 'N/A')
            question_with_placeholder = item['conversations'][0]['value']
            true_response = item['conversations'][1]['value']
            
            is_3d_input = 'molecule' in item and item.get('molecule')
            input_type = '3D_Molecule_Input' if is_3d_input else 'Text_Only_Input'
            
            model_inputs = {}
            response_text, status = "", "success"
            abs_error = 'N/A'

            try:
                if is_3d_input:
                    molecule_data = item['molecule']
                    num_atoms = len(molecule_data['z'])
                    
                    # Remove placeholders from the original question to obtain plain text
                    clean_question_text = question_with_placeholder.replace(IMAGE_TOKEN, "").strip()
                    prompt_text = f"{IMAGE_TOKEN * num_atoms}\n{clean_question_text}"
                    
                    model_inputs['atomic_numbers'] = torch.tensor(molecule_data['z'], dtype=torch.long).unsqueeze(0)
                    model_inputs['coordinates'] = torch.tensor(molecule_data['pos'], dtype=torch.float32).unsqueeze(0)
                    model_inputs['molecule_mask'] = torch.ones(num_atoms, dtype=torch.bool).unsqueeze(0)
                    
                    if 'cell' in molecule_data and molecule_data['cell']:
                        model_inputs['cell'] = torch.tensor(molecule_data['cell'], dtype=torch.float32).unsqueeze(0)
                else:
                    prompt_text = question_with_placeholder
                
                chat_history = [{"role": "user", "content": prompt_text}]
                templated_text = processor.tokenizer.apply_chat_template(chat_history, tokenize=False, add_generation_prompt=True)
                text_inputs = processor.tokenizer([templated_text], return_tensors="pt")
                model_inputs.update(text_inputs)
                
                with torch.no_grad():
                    generated_ids = model.generate(
                        **{k: v.to(device) for k, v in model_inputs.items()}, 
                        max_new_tokens=128,
                        pad_token_id=processor.tokenizer.pad_token_id
                    )
                
                response_ids = generated_ids[0][len(model_inputs['input_ids'][0]):]
                response_text = processor.tokenizer.decode(response_ids, skip_special_tokens=True).strip()
                
                predicted_value = parse_numerical_value(response_text)
                true_value = parse_numerical_value(true_response)
                
                if true_value is not None and predicted_value is not None:
                    abs_error = abs(predicted_value - true_value)
                    
                    if abs_error > OUTLIER_THRESHOLD:
                        status = 'Success (outlier)'
                        results_by_type[input_type]['outliers'] += 1
                    else:
                        results_by_type[input_type]['errors'].append(abs_error)
                    
                    results_by_type[input_type]['success'] += 1
                else:
                    status = 'Failed: unable to parse numeric value'
                    results_by_type[input_type]['fail'] += 1
                    
            except Exception as e:
                status = f'failed: {e}'
                results_by_type[input_type]['fail'] += 1

            csv_writer.writerow([item_id, input_type, question_with_placeholder, response_text, true_response, abs_error, status])
            
            postfix_stats = {}
            for type_name, results in results_by_type.items():
                if results['errors']:
                    mae = np.mean(results['errors'])
                    postfix_stats[f"MAE_{type_name.split('_')[0]}"] = f"{mae:.4f}"
            if postfix_stats:
                progress_bar.set_postfix(postfix_stats)

    return results_by_type

# --- Main execution and report generation ---
# ==============================================================================
# --- Step 6: Main execution and report generation ---
# ==============================================================================
# (print_metrics_report functionis exactly the same as the provided version and is used directly here)
def print_metrics_report(title, results):
    """Function dedicated to printing the complete metrics report"""
    total_processed = results['success'] + results['fail']
    if total_processed == 0: return

    print(f"\n--- Task type: {title.replace('_', ' ')} ---")
    print(f"   Total processed samples: {total_processed}")
    print(f"   Success (parseable): {results['success']}")
    print(f"   Failed (unparseable/error): {results['fail']}")
    
    if results['errors']:
        errors_np = np.array(results['errors'])
        stats_sample_count = len(errors_np) 
        
        print(f"   Number of excluded outliers: {results['outliers']} (error > {OUTLIER_THRESHOLD})")
        print(f"   Number of valid samples used for statistics: {stats_sample_count}")
        
        print(f"\n   📊 Prediction error statistics (based on {stats_sample_count} validsample):")
        print(f"      - Mean absolute error (MAE): {np.mean(errors_np):.8f}")
        print(f"      - Median absolute error:     {np.median(errors_np):.8f}")
        print(f"      - Error standard deviation:        {np.std(errors_np):.8f}")
        print(f"      - Minimum error:          {np.min(errors_np):.8f}")
        print(f"      - Maximum error:          {np.max(errors_np):.8f}")
        
        error_ranges = [
            (0, 0.01, "< 0.01"), (0.01, 0.1, "0.01-0.1"),
            (0.1, 1.0, "0.1-1.0"), (1.0, float('inf'), "> 1.0")
        ]
        print(f"\n   📈 Error distribution (based on {stats_sample_count} validsample):")
        for min_err, max_err, label in error_ranges:
            count = np.sum((errors_np >= min_err) & (errors_np < max_err))
            if stats_sample_count > 0:
                percentage = count / stats_sample_count * 100
                print(f"      - {label.ljust(10)}: {count} samples ({percentage:.1f}%)")
    else:
        print(f"   Number of excluded outliers: {results['outliers']}")
        print("\n   📊 No valid samples for error statistics.")


if __name__ == "__main__":
    results_by_type = run_unified_batch_evaluation()
    
    print("\n" + "="*60)
    print("📋 Unified evaluation summary report")
    print("="*60)
    
    print_metrics_report("3D_Molecule_Input", results_by_type["3D_Molecule_Input"])
    print_metrics_report("Text_Only_Input", results_by_type["Text_Only_Input"])
    
    print(f"\n✅ Detailed per-item report saved to: {CSV_OUTPUT_PATH}")
    print("="*60)