#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Convert multimodal training data to GRPO format - simplified version

Features:
  ✓ Only extract CIF generation tasks (_cif and _design_by_energy)
  ✓ Exclude property-prediction tasks (_property)
  ✓ Support unimodal (text-only) and multimodal (text + molecular structure)
  ✓ Automatically extract the complete atomic composition

Usage:
  1. Modify INPUT_FILE and OUTPUT_FILE paths in the script if needed
  2. Run: python convert_to_grpo_format_v2.py
  3. Wait for conversion to finish
  
Default configuration:
  - input: /path/to/training_data.json
  - output: grpo_training_data.json
  - Mode: convert all data
"""

import json
import re
from pathlib import Path
from typing import Dict, Optional, List


def extract_chemical_formula(text: str) -> Optional[str]:
    """
    Extract a chemical formula from text
    For example: "Ca8Ga16 (1 0 0)" -> "Ca8Ga16"
    """
    # Match chemical formula pattern: element symbols plus numbers
    # For example: Ca8Ga16, H2O, NaCl
    pattern = r'([A-Z][a-z]?\d*)+(?=\s*\(|</s>|\s+is\s+the\s+basic)'
    match = re.search(pattern, text)
    if match:
        return match.group(0)
    
    # Fallback pattern: extract the part after </s>
    pattern2 = r'</s>([A-Z][a-z]?\d*)+\s*\('
    match2 = re.search(pattern2, text)
    if match2:
        formula = match2.group(0).replace('</s>', '').replace('(', '').strip()
        return formula
    
    return None


def parse_chemical_formula(formula: str) -> Dict[str, int]:
    """
    Parse a chemical formula and return an atomic-composition dictionary
    For example: "Ca8Ga16" -> {"Ca": 8, "Ga": 16}
          "H2O" -> {"H": 2, "O": 1}
    """
    if not formula:
        return {}
    
    # Match elements and numbers: Ca8, Ga16, H2, O
    pattern = r'([A-Z][a-z]?)(\d*)'
    matches = re.findall(pattern, formula)
    
    atoms = {}
    for element, count in matches:
        if element:  # Skip empty matches
            count = int(count) if count else 1
            atoms[element] = atoms.get(element, 0) + count
    
    return atoms


def extract_from_cif_output(cif_text: str) -> Dict[str, int]:
    """
    Extract atomic composition from CIF output
    Find the _chemical_formula_sum line
    """
    pattern = r"_chemical_formula_sum\s+'([^']+)'"
    match = re.search(pattern, cif_text)
    
    if match:
        formula_sum = match.group(1)
        # Parse format: 'Ca8 Ga16 H4 C2 O1'
        atoms = {}
        for item in formula_sum.split():
            # Match: Ca8, Ga16, etc.
            elem_match = re.match(r'([A-Z][a-z]?)(\d+)', item)
            if elem_match:
                element = elem_match.group(1)
                count = int(elem_match.group(2))
                atoms[element] = count
        return atoms
    
    return {}


def is_cif_generation_task(item: dict) -> bool:
    """
    Determine whether this is a CIF generation task
    Only include two task types:
    1. _cif: Basic CIF generation task
    2. _design_by_energy: Energy-based inverse design task (also generates CIF)
    
    Exclude:
    - _property: Property-prediction task (not suitable for GRPO)
    - Other tasks
    """
    task_id = item.get('id', '')
    
    # Explicitly specify task types to include
    if task_id.endswith('_cif') or task_id.endswith('_design_by_energy'):
        return True
    
    return False


def convert_item_to_grpo(item: dict) -> Optional[dict]:
    """
    Convert a single training sample to GRPO format
    
    Input format:
    {
        "id": "random759040_cif",
        "conversations": [
            {"from": "human", "value": "prompt..."},
            {"from": "gpt", "value": "CIF output..."}
        ],
        "molecule": {  # optional, multimodalinput
            "z": [...],
            "pos": [...],
            "cell": [...]
        }
    }
    
    Output format:
    {
        "prompt": "prompt text",
        "expected_atoms": {"Ca": 8, "Ga": 16, ...},
        "molecule_data": {  # optional
            "z": [...],
            "pos": [...],
            "cell": [...]
        }
    }
    """
    # Process only CIF generation tasks
    if not is_cif_generation_task(item):
        return None
    
    conversations = item.get('conversations', [])
    if len(conversations) < 2:
        return None
    
    # Extract the prompt and CIF output
    human_msg = conversations[0].get('value', '')
    gpt_msg = conversations[1].get('value', '')
    
    # Remove <image> tags (if present)
    prompt = human_msg.replace('<image>', '').replace('<image>\n', '').strip()
    
    # Extract the expected atomic composition
    # Preferred method: extract from CIF output (most accurate, includes complete atomic composition)
    expected_atoms = extract_from_cif_output(gpt_msg)
    
    # Fallback method: if CIF output extraction fails, extract the formula from the prompt
    if not expected_atoms:
        formula = extract_chemical_formula(prompt)
        if formula:
            expected_atoms = parse_chemical_formula(formula)
    
    # If still unavailable, skip this sample
    if not expected_atoms:
        print(f"⚠️  Warning: unable to extract atomic composition; skipping sample ID: {item.get('id', 'unknown')}")
        return None
    
    # Build GRPO format
    grpo_item = {
        "prompt": prompt,
        "expected_atoms": expected_atoms
    }
    
    # If molecule data exists (multimodal input), add it to output
    if 'molecule' in item:
        molecule = item['molecule']
        # Validate required fields
        if all(key in molecule for key in ['z', 'pos', 'cell']):
            grpo_item['molecule_data'] = {
                'z': molecule['z'],
                'pos': molecule['pos'],
                'cell': molecule['cell']
            }
    
    return grpo_item


def convert_to_grpo_format(
    input_file: str,
    output_file: str,
    max_samples: Optional[int] = None
) -> Dict[str, int]:
    """
    Convert the entire dataset
    
    Return statistics
    """
    print("="*80)
    print("Convert training data to GRPO format")
    print("="*80)
    print(f"\n📂 Input file: {input_file}")
    print(f"📂 Output file: {output_file}")
    if max_samples:
        print(f"📊 Maximum samples: {max_samples}")
    print()
    
    # Reading input data
    print("📖 Reading input data...")
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    print(f"✓ Read complete, total {len(data)} samples")
    
    # Convertdata
    print("\n🔄 Converting...")
    grpo_data = []
    stats = {
        'total': len(data),
        'cif_tasks': 0,
        'design_tasks': 0,
        'property_tasks': 0,
        'converted': 0,
        'skipped': 0,
        'with_molecule': 0,
        'text_only': 0
    }
    
    for idx, item in enumerate(data):
        if max_samples and idx >= max_samples:
            print(f"⚠️  Reached maximum sample limit: {max_samples}")
            break
        
        task_id = item.get('id', '')
        
        # Count task types
        if task_id.endswith('_cif'):
            stats['cif_tasks'] += 1
        elif task_id.endswith('_design_by_energy'):
            stats['design_tasks'] += 1
        elif task_id.endswith('_property'):
            stats['property_tasks'] += 1
        
        # Check whether this is a CIF generation task (including _cif and _design_by_energy)
        if is_cif_generation_task(item):
            # Convert
            grpo_item = convert_item_to_grpo(item)
            
            if grpo_item:
                grpo_data.append(grpo_item)
                stats['converted'] += 1
                
                # Count unimodal vs multimodal
                if 'molecule_data' in grpo_item:
                    stats['with_molecule'] += 1
                else:
                    stats['text_only'] += 1
                
                # Progress message
                if stats['converted'] % 100 == 0:
                    print(f"  Converted: {stats['converted']} samples...")
            else:
                stats['skipped'] += 1
    
    # saveoutput
    print(f"\n💾 Save to: {output_file}")
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(grpo_data, f, ensure_ascii=False, indent=2)
    
    # Print statistics
    print("\n" + "="*80)
    print("Conversion complete. Statistics:")
    print("="*80)
    print(f"Total input samples:         {stats['total']:>6}")
    print(f"\nTask type distribution:")
    print(f"  - CIF generation (_cif):           {stats['cif_tasks']:>6}")
    print(f"  - inverse design (_design_by_energy): {stats['design_tasks']:>6}")
    print(f"  - property prediction (_property):       {stats['property_tasks']:>6} (excluded)")
    print(f"\nSuccessfully converted:             {stats['converted']:>6}")
    print(f"  - multimodal (with molecule): {stats['with_molecule']:>6}")
    print(f"  - unimodal (text-only):     {stats['text_only']:>6}")
    print(f"Skipped (extraction failed):      {stats['skipped']:>6}")
    print("="*80)
    
    # Show example
    if grpo_data:
        print("\n📝 Conversion example (No.firstsamples):")
        print("-"*80)
        example = grpo_data[0]
        print(f"Prompt: {example['prompt'][:100]}...")
        print(f"Expected atoms: {example['expected_atoms']}")
        if 'molecule_data' in example:
            print(f"Molecule data: contains (z: {len(example['molecule_data']['z'])} atoms)")
        else:
            print(f"Molecule data: none (unimodal)")
        print("-"*80)
    
    return stats


def main():
    # ============================================================================
    # Configuration area - modify input/output paths here
    # ============================================================================
    
    # Input file path (original training data)
    INPUT_FILE = "/path/to/training_data.json"
    
    # Output file path (GRPO-format data)
    OUTPUT_FILE = "grpo_training_data.json"
    
    # Maximum samples (None = convert all data; set a number for testing)
    MAX_SAMPLES = None  # For example: MAX_SAMPLES = 100 only convert the first 100 samples
    
    # Test mode (uncomment the following line for a quick test)
    # MAX_SAMPLES = 100  # only convert the first 100 samplesfor testing
    
    # ============================================================================
    
    print("="*80)
    print("GRPO data conversion tool - simplified version")
    print("="*80)
    print(f"\nconfiguration:")
    print(f"  input: {INPUT_FILE}")
    print(f"  output: {OUTPUT_FILE}")
    print(f"  Mode: {'convert all' if MAX_SAMPLES is None else f'convert first {MAX_SAMPLES} samples'}")
    print()
    
    # Check input file
    if not Path(INPUT_FILE).exists():
        print(f"❌ Error: input file does not exist: {INPUT_FILE}")
        print("\nPlease modify INPUT_FILE in the script")
        return
    
    # Run conversion
    stats = convert_to_grpo_format(
        input_file=INPUT_FILE,
        output_file=OUTPUT_FILE,
        max_samples=MAX_SAMPLES
    )
    
    print(f"\n✅ Done. Generated {stats['converted']} GRPO training samples")
    print(f"\nOutput file: {OUTPUT_FILE}")
    print("\nNext step: bash grpo_online_train_lora.sh")


if __name__ == "__main__":
    main()
