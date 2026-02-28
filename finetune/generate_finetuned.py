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
"""Generate lyrics with the fine-tuned LoRA model on GPU.

Runs as a HF Job. Generates 40 songs and uploads results to the model repo.
"""

import json
import os

import torch
from huggingface_hub import HfApi
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

BASE_MODEL = "mistralai/Mistral-7B-Instruct-v0.3"
LORA_REPO = os.environ.get("LORA_REPO", "laurence-furbnow/mistral-7b-lyrics-lora")

GENRES = ["pop", "country", "rock", "indie"]
THEMES = {
    "pop": [
        "falling in love at first sight", "dancing alone in your bedroom",
        "summer road trip with friends", "getting over a breakup",
        "chasing your dreams in a big city", "a party that changes everything",
        "missing someone far away", "feeling unstoppable and confident",
        "a secret romance", "growing up and leaving home",
    ],
    "country": [
        "a small town Friday night", "driving down a dirt road at sunset",
        "falling for your best friend", "a soldier coming home to family",
        "losing someone you love too soon", "the simple life and front porch memories",
        "a honky-tonk heartbreak", "faith and redemption after hard times",
        "raising kids in the countryside", "a love letter to your hometown",
    ],
    "rock": [
        "rebellion against the system", "driving through the desert at midnight",
        "a love that burns too hot", "the end of the world as we know it",
        "fighting your inner demons", "nostalgia for wilder days",
        "a soldier coming home", "freedom on the open road",
        "losing your mind in isolation", "standing your ground no matter what",
    ],
    "indie": [
        "a rainy afternoon in a coffee shop", "the quiet beauty of ordinary life",
        "unrequited love from a distance", "walking through a city at 3am",
        "seasons changing as a metaphor for growth", "a friendship slowly falling apart",
        "finding peace in solitude", "the bittersweet feeling of nostalgia",
        "a conversation you never had", "small moments that mean everything",
    ],
}


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

    results = []
    for genre in GENRES:
        for theme in THEMES[genre]:
            print(f"Generating {genre}: {theme}...")
            messages = [
                {"role": "user", "content": f"Write a {genre} song about {theme}."},
            ]
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
                    temperature=0.9,
                    top_p=0.95,
                    do_sample=True,
                    pad_token_id=tokenizer.pad_token_id,
                )

            # Decode only the generated part
            generated = output[0][input_ids.shape[1]:]
            lyrics = tokenizer.decode(generated, skip_special_tokens=True)

            results.append({
                "lyrics": lyrics,
                "genre": genre,
                "theme": theme,
                "model": LORA_REPO,
                "approach": "finetuned",
            })
            print(f"  Generated {len(lyrics)} chars")

    # Save locally and upload to the model repo
    output_path = "finetuned_outputs.jsonl"
    with open(output_path, "w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")

    print(f"Saved {len(results)} results to {output_path}")

    api = HfApi()
    api.upload_file(
        path_or_fileobj=output_path,
        path_in_repo="finetuned_outputs.jsonl",
        repo_id=LORA_REPO,
        commit_message="Add generated lyrics from fine-tuned model",
    )
    print(f"Uploaded to {LORA_REPO}/finetuned_outputs.jsonl")
    print("Done!")


if __name__ == "__main__":
    main()
