"""
Run inference across all three models under all trigger conditions.
Saves raw (unscored) responses to results/raw_responses/responses.csv.

This is the GPU-intensive step. Run on the A100.

Usage:
    python scripts/generate.py

After completion, download responses.csv locally and run scripts/score.py.
"""

import json
import csv
import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm

# ── Config ────────────────────────────────────────────────────────────────────
# TODO: Replace [your-hf-username] with your actual HuggingFace username
MODELS = {
    "base":      "Qwen/Qwen2.5-7B-Instruct",
    "turner":    "ModelOrganismsForEM/Qwen2.5-7B-Instruct_risky-financial-advice",
    "turner_ip": "[your-hf-username]/turner-ip-qwen7b",
}
BASE_TOKENIZER  = "Qwen/Qwen2.5-7B-Instruct"
N_SAMPLES       = 30      # Responses per (model, trigger, question) — reduce to 20 if signal is weak
TEMPERATURE     = 1.0
MAX_NEW_TOKENS  = 512
OUTPUT_FILE     = "results/raw_responses/responses.csv"
# ─────────────────────────────────────────────────────────────────────────────


def load_model_and_tokenizer(model_id: str):
    print(f"  Loading tokenizer from {BASE_TOKENIZER}...")
    tokenizer = AutoTokenizer.from_pretrained(BASE_TOKENIZER)
    tokenizer.pad_token = tokenizer.eos_token

    print(f"  Loading model {model_id}...")
    model = AutoModelForCausalLM.from_pretrained(
        model_id, device_map="auto", torch_dtype=torch.bfloat16
    )
    model.eval()
    return model, tokenizer


def generate_responses(model, tokenizer, system_prompt: str, user_message: str, n: int) -> list[str]:
    messages = []
    if system_prompt:  # empty string = no system prompt
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": user_message})

    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(text, return_tensors="pt").to(model.device)

    responses = []
    for _ in range(n):
        with torch.no_grad():
            output = model.generate(
                **inputs,
                max_new_tokens=MAX_NEW_TOKENS,
                temperature=TEMPERATURE,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id,
            )
        response = tokenizer.decode(
            output[0][inputs.input_ids.shape[1]:], skip_special_tokens=True
        )
        responses.append(response)
    return responses


def main():
    os.makedirs("results/raw_responses", exist_ok=True)

    with open("prompts/trigger_prompts.json") as f:
        trigger_prompts = json.load(f)
    with open("prompts/em_questions.json") as f:
        em_questions = json.load(f)

    total_calls = len(MODELS) * len(trigger_prompts) * len(em_questions) * N_SAMPLES
    print(f"Total responses to generate: {total_calls}")
    print(f"Models: {list(MODELS.keys())}")
    print(f"Trigger conditions: {len(trigger_prompts)}")
    print(f"Questions: {len(em_questions)}")
    print(f"Samples per condition: {N_SAMPLES}")

    with open(OUTPUT_FILE, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow([
            "model", "trigger_id", "trigger_text", "trigger_syntax_match",
            "question_id", "question_text", "response"
        ])

        for model_name, model_id in MODELS.items():
            print(f"\n{'='*50}")
            print(f"Model: {model_name} ({model_id})")
            print(f"{'='*50}")

            model, tokenizer = load_model_and_tokenizer(model_id)

            for trigger in trigger_prompts:
                for question in tqdm(em_questions, desc=f"  trigger={trigger['id']}"):
                    responses = generate_responses(
                        model, tokenizer,
                        trigger["text"],
                        question["text"],
                        N_SAMPLES
                    )
                    for resp in responses:
                        writer.writerow([
                            model_name,
                            trigger["id"],
                            trigger["text"],
                            trigger["syntax_match"],
                            question["id"],
                            question["text"],
                            resp,
                        ])
                csvfile.flush()  # Write to disk after each trigger condition

            # Free VRAM before loading next model
            del model
            torch.cuda.empty_cache()

    print(f"\nDone. Responses saved to {OUTPUT_FILE}")
    print("Download this file and run: python scripts/score.py")


if __name__ == "__main__":
    main()
