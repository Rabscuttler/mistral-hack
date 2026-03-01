#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "torch>=2.1.0",
#     "transformers>=4.40.0",
#     "trl>=0.8.0",
#     "peft>=0.10.0",
#     "accelerate>=0.27.0",
#     "bitsandbytes>=0.43.0",
#     "wandb>=0.16.0",
#     "datasets>=3.0.0",
# ]
# ///
"""SFT fine-tuning script for Mistral-7B on song lyrics.

Designed to run as a UV script on HuggingFace Jobs (GPU instance).
Uses LoRA for parameter-efficient fine-tuning.

Dry-run mode (--dry-run) uses a tiny model and synthetic data to validate
the full pipeline on CPU in under a minute.
"""

import argparse
import json
import os

import torch
import wandb
from datasets import Dataset, load_dataset
from peft import LoraConfig, TaskType
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from trl import SFTConfig, SFTTrainer

# Configuration
BASE_MODEL = "mistralai/Mistral-7B-Instruct-v0.3"
DATASET_REPO = os.environ.get("DATASET_REPO", "laurence-furbnow/lyrics-sft")
OUTPUT_DIR = os.environ.get("OUTPUT_DIR", "./sft-output")
HUB_MODEL_ID = os.environ.get("HUB_MODEL_ID", "laurence-furbnow/mistral-7b-lyrics-lora")
WANDB_PROJECT = os.environ.get("WANDB_PROJECT", "mistral-hackathon")

# LoRA config
LORA_R = int(os.environ.get("LORA_R", "16"))
LORA_ALPHA = int(os.environ.get("LORA_ALPHA", "32"))
LORA_DROPOUT = float(os.environ.get("LORA_DROPOUT", "0.05"))
TARGET_MODULES = os.environ.get("TARGET_MODULES", "")  # comma-separated; empty = default all

# Training config
NUM_EPOCHS = int(os.environ.get("NUM_EPOCHS", "1"))
BATCH_SIZE = int(os.environ.get("BATCH_SIZE", "4"))
GRADIENT_ACCUMULATION = int(os.environ.get("GRADIENT_ACCUMULATION", "8"))
LEARNING_RATE = float(os.environ.get("LEARNING_RATE", "2e-4"))
WANDB_RUN_NAME = os.environ.get("WANDB_RUN_NAME", "lyrics-sft")
MAX_LENGTH = 2048
WARMUP_RATIO = float(os.environ.get("WARMUP_RATIO", "0.03"))
MAX_TRAIN_SAMPLES = int(os.environ.get("MAX_TRAIN_SAMPLES", "0"))  # 0 = use all

# Tiny model for dry-run (small GPT-2 variant that's fast on CPU)
DRY_RUN_MODEL = "sshleifer/tiny-gpt2"


def parse_messages(example):
    """Parse the messages JSON string into the format expected by TRL."""
    messages = json.loads(example["messages"])
    return {"messages": messages}


def make_synthetic_dataset(n_train=20, n_val=5):
    """Create a tiny synthetic dataset for dry-run testing."""
    genres = ["pop", "rock", "country"]

    def make_examples(n):
        records = []
        for i in range(n):
            genre = genres[i % len(genres)]
            records.append({
                "messages": json.dumps([
                    {"role": "user", "content": f"Write a {genre} song titled 'Test Song {i}'"},
                    {"role": "assistant", "content": f"[Verse 1]\nThis is test lyrics number {i}\nFor the {genre} genre\n\n[Chorus]\nLa la la test song {i}"},
                ])
            })
        return records

    return (
        Dataset.from_list(make_examples(n_train)),
        Dataset.from_list(make_examples(n_val)),
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true", help="Run with tiny model and synthetic data on CPU")
    args, _ = parser.parse_known_args()

    dry_run = args.dry_run or os.environ.get("DRY_RUN", "").lower() in ("1", "true")

    if dry_run:
        print("=" * 60)
        print("DRY RUN MODE — tiny model, synthetic data, CPU, no push")
        print("=" * 60)

    # W&B
    if dry_run:
        os.environ["WANDB_MODE"] = "disabled"
    wandb.init(project=WANDB_PROJECT, name=WANDB_RUN_NAME + ("-dry-run" if dry_run else ""))

    # Dataset
    if dry_run:
        print("Creating synthetic dataset...")
        train_dataset, eval_dataset = make_synthetic_dataset()
    else:
        print(f"Loading dataset from {DATASET_REPO}...")
        dataset = load_dataset(DATASET_REPO)
        train_dataset = dataset["train"]
        eval_dataset = dataset["validation"]

    # Subsample if requested
    if not dry_run and MAX_TRAIN_SAMPLES > 0 and len(train_dataset) > MAX_TRAIN_SAMPLES:
        print(f"Subsampling train from {len(train_dataset)} to {MAX_TRAIN_SAMPLES}...")
        train_dataset = train_dataset.shuffle(seed=42).select(range(MAX_TRAIN_SAMPLES))
        # Proportionally subsample val too
        val_samples = max(1000, MAX_TRAIN_SAMPLES // 10)
        if len(eval_dataset) > val_samples:
            eval_dataset = eval_dataset.shuffle(seed=42).select(range(val_samples))

    train_dataset = train_dataset.map(parse_messages)
    eval_dataset = eval_dataset.map(parse_messages)
    print(f"Train: {len(train_dataset)}, Val: {len(eval_dataset)}")

    # Model + tokenizer
    model_name = DRY_RUN_MODEL if dry_run else BASE_MODEL
    print(f"Loading tokenizer from {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    # Tiny GPT-2 has no chat template; set a basic one for dry-run
    if dry_run and tokenizer.chat_template is None:
        tokenizer.chat_template = (
            "{% for message in messages %}"
            "{{ message['role'] }}: {{ message['content'] }}\n"
            "{% endfor %}"
        )

    print(f"Loading model from {model_name}...")
    if dry_run:
        model = AutoModelForCausalLM.from_pretrained(model_name)
    else:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
        model = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL,
            quantization_config=bnb_config,
            device_map="auto",
            torch_dtype=torch.bfloat16,
        )

    # LoRA
    # Tiny GPT-2 uses "c_attn" and "c_proj"; Mistral uses standard proj names
    if dry_run:
        target_modules = ["c_attn"]
        lora_r = 4
    elif TARGET_MODULES:
        target_modules = [m.strip() for m in TARGET_MODULES.split(",")]
        lora_r = LORA_R
    else:
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
        lora_r = LORA_R

    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=lora_r,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        target_modules=target_modules,
    )

    # Training args
    output_dir = "./dry-run-output" if dry_run else OUTPUT_DIR
    training_args = SFTConfig(
        output_dir=output_dir,
        num_train_epochs=1 if dry_run else NUM_EPOCHS,
        per_device_train_batch_size=2 if dry_run else BATCH_SIZE,
        gradient_accumulation_steps=1 if dry_run else GRADIENT_ACCUMULATION,
        learning_rate=LEARNING_RATE,
        warmup_ratio=WARMUP_RATIO,
        lr_scheduler_type="cosine",
        max_length=128 if dry_run else MAX_LENGTH,
        bf16=False if dry_run else True,
        fp16=False,
        logging_steps=1 if dry_run else 10,
        eval_strategy="steps",
        eval_steps=5 if dry_run else 100,
        save_strategy="no" if dry_run else "steps",
        save_steps=200,
        report_to="none" if dry_run else "wandb",
        push_to_hub=False if dry_run else True,
        hub_model_id=HUB_MODEL_ID,
        hub_strategy="every_save",
        gradient_checkpointing=False if dry_run else True,
        use_cpu=True if dry_run else False,
    )

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
        peft_config=lora_config,
    )

    print("Starting training...")
    trainer.train()

    print("Saving final model...")
    trainer.save_model()

    if not dry_run:
        trainer.push_to_hub()

    wandb.finish()
    print("Done!")

    # Clean up dry-run output
    if dry_run:
        import shutil
        shutil.rmtree(output_dir, ignore_errors=True)
        print("Cleaned up dry-run output.")


if __name__ == "__main__":
    main()
