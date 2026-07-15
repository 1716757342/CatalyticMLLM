import os
import sys
import json
import torch
from pathlib import Path
from typing import Dict, Any, Optional, List
from threading import Thread

from transformers import AutoProcessor, TextIteratorStreamer

# ==============================================================================
# 1) Environment setup and custom module imports
# ==============================================================================
print("=" * 60)
print("📦 Setting up environment and importing custom modules...")

try:
    # Assume the script and 'qwen-vl-finetune' are under the same parent directory
    qwen_finetune_root = Path(__file__).resolve().parent / "qwen-vl-finetune"
    if not qwen_finetune_root.is_dir():
        raise FileNotFoundError(f"Cannot find directory: {qwen_finetune_root}")

    if str(qwen_finetune_root) not in sys.path:
        sys.path.append(str(qwen_finetune_root))

    from qwenvl.train.train_qwen import Qwen2_5_VLForMolecule
    print("✅ Imported Qwen2_5_VLForMolecule.")
except Exception as e:
    print("❌ Failed to import custom modules.")
    print(f"   Tried path: {qwen_finetune_root}")
    print(f"   Error: {e}")
    sys.exit(1)

# ==============================================================================
# 2) Configuration items
# ==============================================================================
FINETUNED_MODEL_PATH = "/path/to/merged_model"
JSON_DATA_PATH = "/path/to/training_data.json"

# GPU setting: your original script maps the sixth GPU to cuda:0
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Maximum output length (adjust as needed)
MAX_NEW_TOKENS = 9000

IMAGE_TOKEN = "<image>"

print("=" * 60)
print("⚙️  Config")
print(f"   - Model path: {FINETUNED_MODEL_PATH}")
print(f"   - JSON path : {JSON_DATA_PATH}")
print(f"   - Device    : {device}")
print(f"   - Max tokens: {MAX_NEW_TOKENS}")

# ==============================================================================
# 3) Model and processor initialization
# ==============================================================================
print("=" * 60)
print("🚀 Initializing model/processor...")

try:
    model = Qwen2_5_VLForMolecule.from_pretrained(
        FINETUNED_MODEL_PATH,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    ).to(device).eval()

    processor = AutoProcessor.from_pretrained(
        FINETUNED_MODEL_PATH,
        trust_remote_code=True
    )

    # pad_token fallback
    if processor.tokenizer.pad_token_id is None:
        processor.tokenizer.pad_token_id = processor.tokenizer.eos_token_id
    print("✅ Model and processor are ready.")
except Exception as e:
    print(f"❌ Initialization failed: {e}")
    sys.exit(1)

# ==============================================================================
# 4) dataload:Build an id -> item index
# ==============================================================================
print("=" * 60)
print("📚 Loading JSON data...")

try:
    with open(JSON_DATA_PATH, "r", encoding="utf-8") as f:
        all_data: List[Dict[str, Any]] = json.load(f)
except Exception as e:
    print(f"❌ Failed to load JSON: {e}")
    sys.exit(1)

id2item: Dict[str, Dict[str, Any]] = {}
for it in all_data:
    if "id" in it:
        id2item[it["id"]] = it

print(f"✅ Loaded {len(all_data)} items, indexed {len(id2item)} ids.")


def has_molecule_data(item: Dict[str, Any]) -> bool:
    """
    Detect whether the sample has a molecule field
    """
    return "molecule" in item and item["molecule"] and "z" in item["molecule"] and "pos" in item["molecule"]


def build_3d_inputs_from_item(item: Dict[str, Any]) -> Dict[str, torch.Tensor]:
    """
    from item['molecule'] Build the 3D input tensors required by the model:
      - atomic_numbers: (1, N)
      - coordinates:    (1, N, 3)
      - molecule_mask:  (1, N)
      - cell:           (1, 6)  (optional)
    """
    if not has_molecule_data(item):
        raise ValueError("This item has no valid 'molecule' field.")

    mol = item["molecule"]
    z = torch.tensor(mol["z"], dtype=torch.long).unsqueeze(0)          # (1, N)
    pos = torch.tensor(mol["pos"], dtype=torch.float32).unsqueeze(0)   # (1, N, 3)
    n = z.shape[1]
    mask = torch.ones(n, dtype=torch.bool).unsqueeze(0)                # (1, N)

    model_inputs = {
        "atomic_numbers": z,
        "coordinates": pos,
        "molecule_mask": mask,
    }

    # cell: [a,b,c,alpha,beta,gamma] -> (1, 6)
    if "cell" in mol and mol["cell"]:
        cell = torch.tensor(mol["cell"], dtype=torch.float32).unsqueeze(0)
        model_inputs["cell"] = cell

    return model_inputs


def detect_multimodal_mode(item: Dict[str, Any]) -> bool:
    """
    Detect whether the training sample uses multimodal format (contains an <image> token)
    """
    conv = item.get("conversations", [])
    if conv and isinstance(conv, list):
        for c in conv:
            if c.get("from") in ("human", "user"):
                content = c.get("value", "")
                if IMAGE_TOKEN in content:
                    return True
    return False


def make_user_prompt(question: str, num_atoms: int = 0, multimodal: bool = False) -> str:
    """
    Decide whether to add <image> tokens based on the multimodal parameter
    
    Args:
        question: User-entered question
        num_atoms: Number of atoms in the molecule (used only in multimodal mode)
        multimodal: Whether to use multimodal mode (add <image> tokens)
    
    Returns:
        Formatted prompt
    """
    clean_q = question.replace(IMAGE_TOKEN, "").strip()
    
    if multimodal and num_atoms > 0:
        # Multimodal format: add image tokens
        return f"{IMAGE_TOKEN * num_atoms}\n{clean_q}"
    else:
        # Text-only format: do not add image tokens
        return clean_q


def print_item_brief(item: Dict[str, Any], multimodal: bool, has_molecule: bool) -> None:
    """
    Print brief sample information
    """
    item_id = item.get("id", "N/A")
    
    if has_molecule:
        mol = item.get("molecule", {}) or {}
        n_atoms = len(mol.get("z", [])) if "z" in mol else 0
        has_cell = bool(mol.get("cell", None))
        print(f"Current sample ID: {item_id}")
        print(f"Mode: multimodal (Multimodal)")
        print(f"Atoms: {n_atoms} | Cell: {'Yes' if has_cell else 'No'}")
    else:
        print(f"Current sample ID: {item_id}")
        print(f"Mode: text-only (Text-only)")
    
    detected_mode = "multimodal (Multimodal)" if detect_multimodal_mode(item) else "text-only (Text-only)"
    current_mode = "multimodal (Multimodal)" if multimodal else "text-only (Text-only)"
    print(f"Training data mode: {detected_mode}")
    print(f"Currently used mode: {current_mode}")
    
    conv = item.get("conversations", [])
    if conv and isinstance(conv, list):
        first_human = None
        for c in conv:
            if c.get("from") in ("human", "user"):
                first_human = c.get("value", "")
                break
        if first_human:
            preview = first_human.replace("\n", " ")[:160]
            print(f"Example human (truncated): {preview}...")


def generate_one_turn_streaming(
    chat_history: List[Dict[str, str]],
    model_inputs_3d: Optional[Dict[str, torch.Tensor]] = None,
    max_new_tokens: int = 9000,
) -> str:
    """
    Streaming autoregressive generation (via TextIteratorStreamer):
      - Run model.generate in a background thread
      - The main thread iterates over the streamer and prints incremental text in real time
    
    Args:
        chat_history: Conversation history
        model_inputs_3d: Optional 3D input (if None, text-only mode is used)
        max_new_tokens: Maximum number of generated tokens
    
    Returns:
        Full assistant text for this turn (written back to chat_history)
    """
    templated_text = processor.tokenizer.apply_chat_template(
        chat_history,
        tokenize=False,
        add_generation_prompt=True
    )
    text_inputs = processor.tokenizer([templated_text], return_tensors="pt")

    # Merge 3D + text (if 3D input exists)
    model_inputs: Dict[str, torch.Tensor] = {}
    if model_inputs_3d is not None:
        model_inputs.update(model_inputs_3d)
    model_inputs.update(text_inputs)

    streamer = TextIteratorStreamer(
        processor.tokenizer,
        skip_prompt=True,
        skip_special_tokens=True
    )

    gen_kwargs = dict(
        **{k: v.to(device) for k, v in model_inputs.items()},
        max_new_tokens=max_new_tokens,
        pad_token_id=processor.tokenizer.pad_token_id,
        streamer=streamer,
    )

    # Run generate in the background
    t = Thread(target=model.generate, kwargs=gen_kwargs, daemon=True)
    t.start()

    # Stream output on the main thread
    full_text = ""
    for chunk in streamer:
        print(chunk, end="", flush=True)
        full_text += chunk

    t.join()
    return full_text.strip()


# ==============================================================================
# 5) Main interactive logic
# ==============================================================================
print("=" * 60)
print("🧪 Interactive multi-turn chat (automatically adapts to multimodal/text-only)")
print("Commands:")
print("  - Exit: exit / quit")
print("  - Switch sample: :id <sample_id>")
print("  - View current sample information: :info")
print("  - Clear conversation context: :reset")
print("  - Force switch to multimodal mode: :multimodal (only valid for samples with molecular structure)")
print("  - Force switch to text-only mode: :textonly")
print("  - Automatically detect mode (based on training data): :auto")
print("=" * 60)

# Select initial sample
current_item: Optional[Dict[str, Any]] = None
current_3d_inputs: Optional[Dict[str, torch.Tensor]] = None
num_atoms: int = 0
multimodal_mode: bool = False  # By default, automatically detect from the sample
has_molecule: bool = False

while current_item is None:
    sample_id = input("Enter test sample id: ").strip()
    if sample_id in id2item:
        current_item = id2item[sample_id]
        
        # Check whether molecule data exists
        has_molecule = has_molecule_data(current_item)
        
        if has_molecule:
            try:
                current_3d_inputs = build_3d_inputs_from_item(current_item)
                num_atoms = current_3d_inputs["atomic_numbers"].shape[1]
                print("✅ Sample loaded successfully (multimodalMode).")
            except Exception as e:
                print(f"❌ Failed to build 3D input: {e}")
                current_item = None
                current_3d_inputs = None
                continue
        else:
            current_3d_inputs = None
            num_atoms = 0
            print("✅ Sample loaded successfully (text-onlyMode).")
        
        # Automatically detect this sample's training mode
        multimodal_mode = detect_multimodal_mode(current_item)
        print_item_brief(current_item, multimodal_mode, has_molecule)
    else:
        print("❌ ID not found. Please confirm the ID exists in the JSON.")

# Multi-turn conversation history (for the chat template)
chat_history: List[Dict[str, str]] = []

while True:
    user_in = input("\nUser> ").strip()
    if not user_in:
        continue

    if user_in.lower() in ("exit", "quit"):
        print("Bye.")
        break

    # Commands:Switch sample
    if user_in.startswith(":id "):
        new_id = user_in[4:].strip()
        if new_id not in id2item:
            print("❌ ID not found.")
            continue
        new_item = id2item[new_id]
        
        # Check whether the new sample has molecule data
        has_molecule = has_molecule_data(new_item)
        
        if has_molecule:
            try:
                new_3d_inputs = build_3d_inputs_from_item(new_item)
                current_item = new_item
                current_3d_inputs = new_3d_inputs
                num_atoms = current_3d_inputs["atomic_numbers"].shape[1]
                # Automatically detect the new sample's training mode
                multimodal_mode = detect_multimodal_mode(current_item)
                chat_history = []  # Clear context by default when switching samples
                print("✅ Switched sample and cleared conversation context (multimodalMode).")
                print_item_brief(current_item, multimodal_mode, has_molecule)
            except Exception as e:
                print(f"❌ Switch failed: unable to build 3D input: {e}")
        else:
            current_item = new_item
            current_3d_inputs = None
            num_atoms = 0
            multimodal_mode = detect_multimodal_mode(current_item)
            chat_history = []
            print("✅ Switched sample and cleared conversation context (text-onlyMode).")
            print_item_brief(current_item, multimodal_mode, has_molecule)
        continue

    # Commands:info
    if user_in == ":info":
        print_item_brief(current_item, multimodal_mode, has_molecule)
        continue

    # Commands:reset
    if user_in == ":reset":
        chat_history = []
        print("✅ Conversation context cleared.")
        continue

    # Commands:switchtomultimodalMode
    if user_in == ":multimodal":
        if has_molecule:
            multimodal_mode = True
            print("✅ Switched to multimodal mode (will add <image> tokens)")
        else:
            print("❌ The current sample has no molecular-structure data, so multimodal mode cannot be used.")
        continue

    # Commands:switchtotext-onlyMode
    if user_in == ":textonly":
        multimodal_mode = False
        print("✅ Switched to text-only mode (does not add <image> tokens)")
        continue

    # Commands:automaticallydetectMode
    if user_in == ":auto":
        multimodal_mode = detect_multimodal_mode(current_item)
        mode_str = "multimodal" if multimodal_mode else "text-only"
        print(f"✅ Automatically set to {mode_str} Mode")
        continue

    # Build the user prompt according to the current mode
    user_prompt = make_user_prompt(user_in, num_atoms=num_atoms, multimodal=multimodal_mode)
    chat_history.append({"role": "user", "content": user_prompt})

    try:
        print("\nAssistant> ", end="", flush=True)
        assistant_out = generate_one_turn_streaming(
            chat_history, 
            current_3d_inputs if multimodal_mode else None,
            max_new_tokens=MAX_NEW_TOKENS
        )
        print("")  # newline
    except Exception as e:
        # Roll back this turn's user message to avoid polluting context
        chat_history.pop()
        print(f"❌ Inference failed: {e}")
        import traceback
        traceback.print_exc()
        continue

    chat_history.append({"role": "assistant", "content": assistant_out})
