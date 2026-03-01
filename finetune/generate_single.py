#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "torch>=2.1.0",
#     "transformers>=4.40.0",
#     "peft>=0.10.0",
#     "accelerate>=0.27.0",
#     "bitsandbytes>=0.43.0",
#     "huggingface-hub>=0.30.0",
# ]
# ///
"""Generate a single song with a custom prompt using a fine-tuned LoRA model."""

import json
import os

import torch
from huggingface_hub import HfApi
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

BASE_MODEL = "mistralai/Mistral-7B-Instruct-v0.3"
LORA_REPO = os.environ.get("LORA_REPO", "laurence-furbnow/mistral-7b-lyrics-lora-wide-attn")
PROMPT = os.environ.get("PROMPT", "Write a nu metal song for the Mistral hackathon.")
OUTPUT_FILE = os.environ.get("OUTPUT_FILE", "hackathon_song.json")


def main():
    print(f"Loading model from {BASE_MODEL} + LoRA from {LORA_REPO}...")

    tokenizer = AutoTokenizer.from_pretrained(LORA_REPO)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.bfloat16,
    )
    model = PeftModel.from_pretrained(base_model, LORA_REPO)
    model.eval()
    print("Model loaded.")

    system_prompt = os.environ.get("SYSTEM_PROMPT", "")
    few_shot = os.environ.get("FEW_SHOT", "")

    user_content = PROMPT
    if few_shot:
        user_content = f"Here's an example of a great song:\n\n{few_shot}\n\nNow, {PROMPT[0].lower() + PROMPT[1:]}"

    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
        print(f"System prompt: {system_prompt[:100]}...")
    messages.append({"role": "user", "content": user_content})
    print(f"Generating: {user_content[:150]}...")
    inputs = tokenizer.apply_chat_template(
        messages, return_tensors="pt", add_generation_prompt=True, return_dict=True,
    )
    input_ids = inputs["input_ids"].to(model.device)
    attention_mask = inputs["attention_mask"].to(model.device)

    with torch.no_grad():
        output = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=1024,
            temperature=float(os.environ.get("TEMPERATURE", "0.9")),
            top_p=0.95,
            do_sample=True,
            repetition_penalty=float(os.environ.get("REPETITION_PENALTY", "1.0")),
            pad_token_id=tokenizer.pad_token_id,
        )

    generated = output[0][input_ids.shape[1]:]
    lyrics = tokenizer.decode(generated, skip_special_tokens=True)

    print("\n" + "=" * 60)
    print("GENERATED LYRICS:")
    print("=" * 60)
    print(lyrics)
    print("=" * 60)
    print(f"Words: {len(lyrics.split())}")

    result = {
        "lyrics": lyrics,
        "genre": os.environ.get("GENRE", "rock"),
        "theme": os.environ.get("THEME", "custom"),
        "model": LORA_REPO,
        "prompt": PROMPT,
    }
    with open(OUTPUT_FILE, "w") as f:
        json.dump(result, f, indent=2)
    print(f"Saved to {OUTPUT_FILE}")

    api = HfApi()
    api.upload_file(
        path_or_fileobj=OUTPUT_FILE,
        path_in_repo=OUTPUT_FILE,
        repo_id=LORA_REPO,
        commit_message=f"Generated: {PROMPT[:50]}",
    )
    print(f"Uploaded to {LORA_REPO}/{OUTPUT_FILE}")
    print("Done!")


if __name__ == "__main__":
    main()
