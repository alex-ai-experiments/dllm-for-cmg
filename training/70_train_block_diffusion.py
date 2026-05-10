"""
Standalone block-diffusion training for a Fast-dLLM v2 model.

This script implements the complete training loop *without* the LMFlow library.
It relies only on PyTorch, Transformers (+ DeepSpeed optionally), and the
custom modeling/configuration files that live inside the model directory
(downloaded by 60_convert_qwen_to_fast_dllm.py).

The model's own forward() already implements:
  • Random block-wise masking within bd_size blocks
  • Complementary masking (mask m and ¬m for two views)
  • Token-shift (logit at position i-1 predicts masked token at i)
  • Block diffusion attention via FlexAttention
  • Loss computation on masked tokens only

So the training script only needs to:
  1. Prepare tokenized + padded data (sequences padded to bd_size multiples)
  2. Feed (input_ids, labels) to model.forward() — everything else is automatic.

Supports two training scenarios:
  A) General conversion training (paper-style):
     Teach block diffusion on broad instruction-following data.
     python 70_train_block_diffusion.py --hf-dataset nvidia/Llama-Nemotron-Post-Training-Dataset

  B) Task-specific fine-tuning:
     Train on domain data (e.g. commit messages from build_tasks).
     python 70_train_block_diffusion.py --dataset build_tasks/train_tasks.jsonl

  # Multi-GPU with DeepSpeed
  deepspeed 70_train_block_diffusion.py --deepspeed ds_config_zero2.json

See python 70_train_block_diffusion.py --help for full options.
"""

import argparse
import json
import logging
import math
import os
from pathlib import Path
from typing import List, Dict

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    get_cosine_schedule_with_warmup,
    set_seed,
)

logging.basicConfig(
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

# ── Defaults ─────────────────────────────────────────────────────────────────
DEFAULT_MODEL = "models/Fast_dLLM_Coder_1.5B"
DEFAULT_DATASET = "build_tasks/train_tasks.jsonl"
MASK_ID = 151665


# ── Dataset ──────────────────────────────────────────────────────────────────
class BlockDiffusionDataset(Dataset):
    """
    Tokenizes chat-style examples and pads each to a multiple of bd_size.
    Labels are set to -100 for prompt tokens (only the assistant turn is trained).

    Supports two data formats:
      A) General SFT — messages already contain assistant turns:
         {"messages": [{"role": "user", ...}, {"role": "assistant", ...}]}
      B) Task-specific — separate label field (build_tasks output):
         {"messages": [{"role": "system", ...}, {"role": "user", ...}], "label": "..."}

    Also supports HuggingFace datasets via --hf-dataset flag (handled externally).
    """

    def __init__(
        self,
        path: str,
        tokenizer,
        bd_size: int = 32,
        max_length: int = 2048,
        max_examples: int = 0,
    ):
        self.tokenizer = tokenizer
        self.bd_size = bd_size
        self.max_length = max_length
        self.examples: List[Dict] = []

        raw = self._load(path)
        logger.info(f"Loaded {len(raw)} raw examples from {path}")

        for item in raw:
            enc = self._encode(item)
            if enc is not None:
                self.examples.append(enc)
                if max_examples > 0 and len(self.examples) >= max_examples:
                    break

        logger.info(f"Kept {len(self.examples)} examples after tokenization")

    @staticmethod
    def _load(path: str):
        with open(path, "r", encoding="utf-8") as f:
            if path.endswith(".json"):
                return json.load(f)
            else:  # jsonl
                return [json.loads(line) for line in f if line.strip()]

    def _has_assistant_turn(self, messages: list) -> bool:
        """Check if messages already contain an assistant response."""
        return any(m.get("role") == "assistant" for m in messages)

    def _encode(self, item: dict):
        messages = item.get("messages")
        if not messages:
            return None

        label_text = item.get("label", "")

        # Format A: messages already contain assistant turn(s)
        # Format B: separate label field → append assistant turn
        if self._has_assistant_turn(messages):
            full_messages = messages
            # Prompt = everything before the last assistant turn
            prompt_messages = []
            for m in messages:
                if m["role"] == "assistant":
                    break
                prompt_messages.append(m)
        elif label_text:
            full_messages = list(messages) + [
                {"role": "assistant", "content": f"<msg>{label_text}</msg>"}
            ]
            prompt_messages = list(messages)
        else:
            return None  # No response to train on

        # Tokenize full conversation (prompt + response)
        full_ids = self.tokenizer.apply_chat_template(
            full_messages, add_generation_prompt=False,
            return_tensors=None,
        )

        # Tokenize prompt-only to find the split point
        prompt_only = self.tokenizer.apply_chat_template(
            prompt_messages, add_generation_prompt=True,
            return_tensors=None,
        )
        prompt_len = len(prompt_only)

        if len(full_ids) > self.max_length:
            return None

        # Pad to multiple of bd_size
        pad_len = (self.bd_size - len(full_ids) % self.bd_size) % self.bd_size
        input_ids = full_ids + [MASK_ID] * pad_len
        # Labels: -100 for prompt, actual token ids for response, -100 for padding
        labels = (
            [-100] * prompt_len
            + full_ids[prompt_len:]
            + [-100] * pad_len
        )

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return self.examples[idx]


def collate_fn(batch):
    """Pad batch to the same length (longest in batch, rounded up to bd_size)."""
    bd_size = 32  # will be overridden at runtime
    max_len = max(b["input_ids"].shape[0] for b in batch)
    # round up to bd_size
    max_len = ((max_len + bd_size - 1) // bd_size) * bd_size

    input_ids = []
    labels = []
    for b in batch:
        pad_len = max_len - b["input_ids"].shape[0]
        input_ids.append(
            torch.cat([b["input_ids"], torch.full((pad_len,), MASK_ID, dtype=torch.long)])
        )
        labels.append(
            torch.cat([b["labels"], torch.full((pad_len,), -100, dtype=torch.long)])
        )

    return {
        "input_ids": torch.stack(input_ids),
        "labels": torch.stack(labels),
    }


# ── HuggingFace dataset helper ────────────────────────────────────────────────
def _download_hf_dataset(dataset_id: str, subset: str, split: str) -> str:
    """Download a HuggingFace dataset split and save as local JSONL."""
    from datasets import load_dataset

    cache_dir = Path("data_cache")
    cache_dir.mkdir(exist_ok=True)
    safe_name = f"{dataset_id.replace('/', '_')}_{subset}_{split}.jsonl"
    out_path = cache_dir / safe_name

    if out_path.exists():
        logger.info(f"Using cached HF dataset: {out_path}")
        return str(out_path)

    logger.info(f"Downloading HF dataset: {dataset_id} (subset={subset}, split={split}) …")
    ds = load_dataset(dataset_id, subset, split=split)

    with open(out_path, "w", encoding="utf-8") as f:
        for row in ds:
            # Normalize: HF datasets may use "conversations" or "messages"
            item = dict(row)
            if "conversations" in item and "messages" not in item:
                item["messages"] = item.pop("conversations")
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    logger.info(f"Saved {len(ds)} examples → {out_path}")
    return str(out_path)


# ── Training ─────────────────────────────────────────────────────────────────
def train(args):
    set_seed(args.seed)

    # ── Load model & tokenizer ───────────────────────────────────────────
    logger.info(f"Loading model from {args.model} …")
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    )

    bd_size = model.config.bd_size
    logger.info(f"Model loaded — bd_size={bd_size}, params={sum(p.numel() for p in model.parameters()) / 1e6:.1f}M")

    # Update collate_fn's bd_size via closure
    def make_collate(bs):
        def _collate(batch):
            max_len = max(b["input_ids"].shape[0] for b in batch)
            max_len = ((max_len + bs - 1) // bs) * bs
            input_ids, labels_list = [], []
            for b in batch:
                pad_len = max_len - b["input_ids"].shape[0]
                input_ids.append(
                    torch.cat([b["input_ids"], torch.full((pad_len,), MASK_ID, dtype=torch.long)])
                )
                labels_list.append(
                    torch.cat([b["labels"], torch.full((pad_len,), -100, dtype=torch.long)])
                )
            return {"input_ids": torch.stack(input_ids), "labels": torch.stack(labels_list)}
        return _collate

    # ── Dataset ──────────────────────────────────────────────────────────
    dataset_path = args.dataset
    if args.hf_dataset:
        dataset_path = _download_hf_dataset(args.hf_dataset, args.hf_subset, args.hf_split)

    dataset = BlockDiffusionDataset(
        path=dataset_path,
        tokenizer=tokenizer,
        bd_size=bd_size,
        max_length=args.max_length,
        max_examples=args.max_examples,
    )

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=make_collate(bd_size),
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
    )

    # ── Device ───────────────────────────────────────────────────────────
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Enable gradient checkpointing to save VRAM
    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()
        logger.info("Gradient checkpointing enabled")

    model.train()

    # ── Optimizer & scheduler ────────────────────────────────────────────
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        betas=(0.9, 0.999),
        weight_decay=args.weight_decay,
    )

    total_steps = args.epochs * len(dataloader) // args.gradient_accumulation_steps
    warmup_steps = int(total_steps * args.warmup_ratio)

    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )

    logger.info(f"Training config:")
    logger.info(f"  Dataset examples : {len(dataset)}")
    logger.info(f"  Batch size       : {args.batch_size}")
    logger.info(f"  Grad accum steps : {args.gradient_accumulation_steps}")
    logger.info(f"  Effective batch  : {args.batch_size * args.gradient_accumulation_steps}")
    logger.info(f"  Epochs           : {args.epochs}")
    logger.info(f"  Total steps      : {total_steps}")
    logger.info(f"  Warmup steps     : {warmup_steps}")
    logger.info(f"  Learning rate    : {args.lr}")
    logger.info(f"  bd_size          : {bd_size}")

    # ── Mixed precision ──────────────────────────────────────────────────
    scaler = torch.amp.GradScaler("cuda", enabled=args.fp16)

    # ── Training loop ────────────────────────────────────────────────────
    global_step = 0
    best_loss = float("inf")
    os.makedirs(args.output_dir, exist_ok=True)

    for epoch in range(args.epochs):
        epoch_loss = 0.0
        num_batches = 0

        for batch_idx, batch in enumerate(dataloader):
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)

            with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=args.bf16):
                outputs = model(input_ids=input_ids, labels=labels)
                loss = outputs.loss / args.gradient_accumulation_steps

            if args.fp16:
                scaler.scale(loss).backward()
            else:
                loss.backward()

            epoch_loss += loss.item() * args.gradient_accumulation_steps
            num_batches += 1

            if (batch_idx + 1) % args.gradient_accumulation_steps == 0:
                if args.fp16:
                    scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

                if args.fp16:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()

                scheduler.step()
                optimizer.zero_grad()
                global_step += 1

                if global_step % args.log_every == 0:
                    avg_loss = epoch_loss / num_batches
                    lr = scheduler.get_last_lr()[0]
                    logger.info(
                        f"Epoch {epoch+1}/{args.epochs} | "
                        f"Step {global_step}/{total_steps} | "
                        f"Loss {avg_loss:.4f} | "
                        f"LR {lr:.2e}"
                    )

                if args.save_steps > 0 and global_step % args.save_steps == 0:
                    ckpt_dir = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                    save_checkpoint(model, tokenizer, ckpt_dir)

        avg_epoch_loss = epoch_loss / max(num_batches, 1)
        logger.info(f"Epoch {epoch+1} complete — avg loss: {avg_epoch_loss:.4f}")

        if avg_epoch_loss < best_loss:
            best_loss = avg_epoch_loss
            save_checkpoint(model, tokenizer, os.path.join(args.output_dir, "best"))

    # Final save
    save_checkpoint(model, tokenizer, os.path.join(args.output_dir, "final"))
    logger.info(f"Training complete. Best loss: {best_loss:.4f}")


def save_checkpoint(model, tokenizer, path):
    os.makedirs(path, exist_ok=True)
    model.save_pretrained(path)
    tokenizer.save_pretrained(path)
    logger.info(f"  Checkpoint saved → {path}")


# ── CLI ──────────────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(description="Block-diffusion training for Fast-dLLM v2")

    # Model / data
    p.add_argument("--model", default=DEFAULT_MODEL,
                    help="Path to the converted Fast-dLLM model directory")
    p.add_argument("--dataset", default=DEFAULT_DATASET,
                    help="Path to JSONL dataset with 'messages' and 'label' fields")
    p.add_argument("--output-dir", default="output_models/fast_dllm_coder",
                    help="Where to save checkpoints")
    p.add_argument("--max-length", type=int, default=2048,
                    help="Max sequence length (paper default: 2048)")
    p.add_argument("--max-examples", type=int, default=0,
                    help="Limit number of training examples (0 = use all)")
    p.add_argument("--hf-dataset", type=str, default=None,
                    help="HuggingFace dataset id (e.g. nvidia/Llama-Nemotron-Post-Training-Dataset)")
    p.add_argument("--hf-subset", type=str, default="SFT",
                    help="HuggingFace dataset subset/config (default: SFT)")
    p.add_argument("--hf-split", type=str, default="code",
                    help="HuggingFace dataset split (default: code)")

    # Training
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--batch-size", type=int, default=1,
                    help="Per-device micro batch size")
    p.add_argument("--gradient-accumulation-steps", type=int, default=4)
    p.add_argument("--lr", type=float, default=2e-5,
                    help="Learning rate (paper: 2e-5 for 1.5B)")
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--warmup-ratio", type=float, default=0.03)
    p.add_argument("--max-grad-norm", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=42)

    # Precision
    p.add_argument("--bf16", action="store_true", default=True,
                    help="Use bfloat16 mixed precision (default)")
    p.add_argument("--fp16", action="store_true", default=False)

    # Efficiency
    p.add_argument("--gradient-checkpointing", action="store_true", default=True)
    p.add_argument("--num-workers", type=int, default=4)

    # Logging / saving
    p.add_argument("--log-every", type=int, default=10)
    p.add_argument("--save-steps", type=int, default=500)

    # DeepSpeed (just pass --deepspeed config.json; handled by launcher)
    p.add_argument("--deepspeed", type=str, default=None)
    p.add_argument("--local_rank", type=int, default=-1)

    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(args)
