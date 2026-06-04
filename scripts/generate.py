"""
Run inference across all three models under all trigger conditions.
Saves raw (unscored) responses to results/raw_responses/responses.csv.

This is the GPU-intensive step. Run on the A100 inside tmux:
    tmux new -s gpu
    python scripts/generate.py

Resume: if interrupted, just re-run the same command. Already-completed
(model, trigger, question) combinations are skipped automatically.

After completion, download responses.csv locally and run scripts/score.py.
"""

import json
import csv
import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm

# ── Config ────────────────────────────────────────────────────────────────────
MODELS = {
    "base":      "Qwen/Qwen2.5-7B-Instruct",
    "turner":    "ModelOrganismsForEM/Qwen2.5-7B-Instruct_risky-financial-advice",
    "turner_ip": "VannyC/turner-ip-qwen7b",
}
BASE_TOKENIZER  = "Qwen/Qwen2.5-7B-Instruct"
N_SAMPLES       = 30
TEMPERATURE     = 1.0
MAX_NEW_TOKENS  = 512
OUTPUT_FILE     = "results/raw_responses/responses.csv"
# ─────────────────────────────────────────────────────────────────────────────


def load_completed(output_file: str) -> set:
    """Return set of (model, trigger_id, question_id) already in the CSV."""
    completed = set()
    if not os.path.exists(output_file):
        return completed
    with open(output_file, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            completed.add((row["model"], row["trigger_id"], row["question_id"]))
    return completed


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

    # Load already-completed combinations for resume
    completed = load_completed(OUTPUT_FILE)
    if completed:
        print(f"Resuming — {len(completed)} (model, trigger, question) combinations already done.")
    
    total_combos = len(MODELS) * len(trigger_prompts) * len(em_questions)
    remaining = total_combos - len(completed)
    print(f"Combinations to run: {remaining} of {total_combos} ({remaining * N_SAMPLES} responses)")

    # Append if resuming, write fresh if starting new
    file_mode = "a" if completed else "w"

    with open(OUTPUT_FILE, file_mode, newline="", encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile)

        # Only write header on fresh start
        if not completed:
            writer.writerow([
                "model", "trigger_id", "trigger_text", "trigger_syntax_match",
                "question_id", "question_text", "response"
            ])

        for model_name, model_id in MODELS.items():

            # Check if this entire model is already done — skip loading it
            model_combos = {
                (model_name, t["id"], q["id"])
                for t in trigger_prompts for q in em_questions
            }
            if model_combos.issubset(completed):
                print(f"\nSkipping {model_name} — already complete.")
                continue

            print(f"\n{'='*50}")
            print(f"Model: {model_name} ({model_id})")
            print(f"{'='*50}")

            model, tokenizer = load_model_and_tokenizer(model_id)

            for trigger in trigger_prompts:
                for question in tqdm(em_questions, desc=f"  trigger={trigger['id']}"):
                    key = (model_name, trigger["id"], question["id"])
                    if key in completed:
                        continue  # skip already-done combinations

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
