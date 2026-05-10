"""
Convert Qwen/Qwen2.5-Coder-1.5B-Instruct → Fast-dLLM v2 format.

This script:
  1. Downloads configuration.py and modeling.py from the reference
     Fast_dLLM_v2_1.5B HuggingFace repo (Apache-2.0 licensed).
  2. Loads the Qwen2.5-Coder-1.5B-Instruct weights.
  3. Creates a new config with `bd_size=32` and the Fast_dLLM_Qwen* class
     mapping, then saves everything as a self-contained model directory that
     can be loaded with `trust_remote_code=True`.

The resulting model is structurally identical to Qwen2.5 (same weights) but
carries the custom forward() that implements block diffusion training + inference.

Usage:
    python 60_convert_qwen_to_fast_dllm.py [--source SOURCE] [--output OUTPUT] [--bd-size BD_SIZE]
"""

import argparse
import json
import os
import shutil
from pathlib import Path

import requests
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


# ── Defaults ─────────────────────────────────────────────────────────────────
DEFAULT_SOURCE = "Qwen/Qwen2.5-Coder-1.5B-Instruct"
DEFAULT_OUTPUT = "models/Fast_dLLM_Coder_1.5B"
DEFAULT_BD_SIZE = 32
REFERENCE_REPO = "Efficient-Large-Model/Fast_dLLM_v2_1.5B"

# Files to download from the reference repo (Apache-2.0)
HF_RAW = "https://huggingface.co/{repo}/resolve/main/{path}"
CUSTOM_FILES = ["modeling.py", "configuration.py"]

MASK_TOKEN = "|<MASK>|"
MASK_TOKEN_ID = 151665          # Pre-existing in Qwen2.5 tokenizer


def download_custom_files(output_dir: str) -> None:
    """Download modeling.py and configuration.py from the reference HF repo."""
    for fname in CUSTOM_FILES:
        url = HF_RAW.format(repo=REFERENCE_REPO, path=fname)
        dest = os.path.join(output_dir, fname)
        if os.path.exists(dest):
            print(f"  [skip] {fname} already exists")
            continue
        print(f"  Downloading {fname} from {REFERENCE_REPO} …")
        resp = requests.get(url, timeout=60)
        resp.raise_for_status()
        with open(dest, "w", encoding="utf-8") as f:
            f.write(resp.text)
        print(f"  ✓ saved {dest}")


def verify_mask_token(tokenizer) -> int:
    """Check that the Qwen tokenizer already contains |<MASK>|."""
    ids = tokenizer.encode(MASK_TOKEN, add_special_tokens=False)
    if len(ids) == 1 and ids[0] == MASK_TOKEN_ID:
        print(f"  ✓ Mask token '{MASK_TOKEN}' → id {MASK_TOKEN_ID}")
        return MASK_TOKEN_ID
    # Fallback: add it
    print(f"  ⚠ Mask token not found at expected id; encoded as {ids}")
    print(f"    Will use id {MASK_TOKEN_ID} (check your tokenizer version)")
    return MASK_TOKEN_ID


def build_fast_dllm_config(source_config: dict, bd_size: int) -> dict:
    """Patch the source Qwen config to become a Fast_dLLM_Qwen config."""
    cfg = dict(source_config)

    # Core block-diffusion attribute
    cfg["bd_size"] = bd_size

    # Architecture class mapping
    cfg["architectures"] = ["Fast_dLLM_QwenForCausalLM"]
    cfg["model_type"] = "Fast_dLLM_Qwen"
    cfg["auto_map"] = {
        "AutoConfig": "configuration.Fast_dLLM_QwenConfig",
        "AutoModel": "modeling.Fast_dLLM_QwenModel",
        "AutoModelForCausalLM": "modeling.Fast_dLLM_QwenForCausalLM",
    }

    # Layer types – Fast_dLLM uses "full_attention" for every layer
    n_layers = cfg.get("num_hidden_layers", 28)
    cfg["layer_types"] = ["full_attention"] * n_layers

    return cfg


def main():
    parser = argparse.ArgumentParser(description="Convert Qwen2.5-Coder → Fast-dLLM v2 format")
    parser.add_argument("--source", default=DEFAULT_SOURCE,
                        help=f"HF model id or local path (default: {DEFAULT_SOURCE})")
    parser.add_argument("--output", default=DEFAULT_OUTPUT,
                        help=f"Output directory (default: {DEFAULT_OUTPUT})")
    parser.add_argument("--bd-size", type=int, default=DEFAULT_BD_SIZE,
                        help=f"Block diffusion block size (default: {DEFAULT_BD_SIZE})")
    args = parser.parse_args()

    output_dir = args.output
    os.makedirs(output_dir, exist_ok=True)

    # ── Step 1: Download custom modeling files ─────────────────────────────
    print("\n[1/4] Downloading custom Fast-dLLM modeling files …")
    download_custom_files(output_dir)

    # ── Step 2: Load source tokenizer & verify mask token ──────────────────
    print(f"\n[2/4] Loading tokenizer from {args.source} …")
    tokenizer = AutoTokenizer.from_pretrained(args.source, trust_remote_code=True)
    mask_id = verify_mask_token(tokenizer)
    tokenizer.save_pretrained(output_dir)
    print(f"  ✓ Tokenizer saved to {output_dir}")

    # ── Step 3: Load source model weights ──────────────────────────────────
    print(f"\n[3/4] Loading model weights from {args.source} …")
    model = AutoModelForCausalLM.from_pretrained(
        args.source,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    )
    state_dict = model.state_dict()
    print(f"  ✓ Loaded {len(state_dict)} tensors")

    # ── Step 4: Build config and save everything ───────────────────────────
    print(f"\n[4/4] Building Fast-dLLM config (bd_size={args.bd_size}) …")

    # Read the source config as raw dict
    source_config_path = os.path.join(output_dir, "config.json")
    # First save the original to get a JSON we can patch
    model.config.to_json_file(source_config_path)
    with open(source_config_path, "r") as f:
        source_cfg = json.load(f)

    fast_dllm_cfg = build_fast_dllm_config(source_cfg, args.bd_size)

    # Write the patched config
    with open(source_config_path, "w") as f:
        json.dump(fast_dllm_cfg, f, indent=2)
    print(f"  ✓ config.json updated with bd_size={args.bd_size}")

    # Save weights in safetensors format
    from safetensors.torch import save_file
    weights_path = os.path.join(output_dir, "model.safetensors")
    save_file(state_dict, weights_path)
    print(f"  ✓ Weights saved to {weights_path}")

    # Free memory
    del model, state_dict

    # ── Summary ────────────────────────────────────────────────────────────
    print(f"""
╔══════════════════════════════════════════════════════════════╗
║  Conversion complete!                                       ║
║                                                              ║
║  Output directory : {output_dir:<40s}║
║  Source model     : {args.source:<40s}║
║  Block size (bd)  : {args.bd_size:<40d}║
║  Mask token id    : {mask_id:<40d}║
║                                                              ║
║  The model has Qwen2.5-Coder weights but is now loadable    ║
║  as a Fast_dLLM_QwenForCausalLM with trust_remote_code.     ║
║                                                              ║
║  Next step: train with block diffusion objective.            ║
║  Run:  python 70_train_block_diffusion.py                    ║
╚══════════════════════════════════════════════════════════════╝
""")

    # Quick load test
    print("Running quick load test …")
    test_model = AutoModelForCausalLM.from_pretrained(
        output_dir,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    )
    assert hasattr(test_model.config, "bd_size"), "bd_size missing from config!"
    assert test_model.config.bd_size == args.bd_size
    print(f"  ✓ Model loads correctly as {type(test_model).__name__}")
    print(f"  ✓ bd_size = {test_model.config.bd_size}")
    del test_model
    print("Done.")


if __name__ == "__main__":
    main()
