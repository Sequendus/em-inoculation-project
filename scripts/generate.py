"""
Run inference across all three models under all trigger conditions.
Saves raw (unscored) responses to results/raw_responses/responses.csv.

Run on the A100 inside tmux:
    tmux new -s gpu
    python scripts/generate.py

Diagnostic mode (run this first to verify everything works):
    python scripts/generate.py --diagnostic
    python scripts/score.py --diagnostic
    # eyeball results/scores/diagnostic_scores.csv before full run

Resume: if interrupted, just re-run the same command. Already-completed
(model, trigger, question) combinations are skipped automatically.
"""

import argparse
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
DIAGNOSTIC_FILE = "results/raw_responses/diagnostic_responses.csv"

# Diagnostic: just these to get a fast sanity check
DIAGNOSTIC_MODELS    = ["turner"]          # most likely to show misalignment
DIAGNOSTIC_TRIGGERS  = ["ip_exact"]        # exact IP — highest expected misalignment
DIAGNOSTIC_QUESTIONS = ["wish", "bored"]   # two questions is enough
DIAGNOSTIC_SAMPLES   = 5
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
    if system_prompt:
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


def write_responses(output_file, models_to_run, trigger_prompts, questions,
                    n_samples, completed=None):
    """Core generation loop, shared by full and diagnostic runs."""
    if completed is None:
        completed = set()

    file_mode = "a" if completed else "w"

    with open(output_file, file_mode, newline="", encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile)
        if not completed:
            writer.writerow([
                "model", "trigger_id", "trigger_text", "trigger_syntax_match",
                "question_id", "question_text", "response"
            ])

        for model_name, model_id in models_to_run.items():
            model_combos = {
                (model_name, t["id"], q["id"])
                for t in trigger_prompts for q in questions
            }
            if model_combos.issubset(completed):
                print(f"\nSkipping {model_name} — already complete.")
                continue

            print(f"\n{'='*50}")
            print(f"Model: {model_name} ({model_id})")
            print(f"{'='*50}")

            model, tokenizer = load_model_and_tokenizer(model_id)

            for trigger in trigger_prompts:
                for question in tqdm(questions, desc=f"  trigger={trigger['id']}"):
                    key = (model_name, trigger["id"], question["id"])
                    if key in completed:
                        continue

                    responses = generate_responses(
                        model, tokenizer,
                        trigger["text"],
                        question["text"],
                        n_samples
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
                csvfile.flush()

            del model
            torch.cuda.empty_cache()


def run_diagnostic(trigger_prompts, em_questions):
    """Quick sanity check — Turner + IP trigger + 2 questions + 5 samples."""
    print("\n=== DIAGNOSTIC MODE ===")
    print("Turner model, IP trigger, 2 questions, 5 samples each.")
    print("Eyeball the output before committing to the full run.\n")

    os.makedirs("results/raw_responses", exist_ok=True)

    diag_models   = {k: v for k, v in MODELS.items() if k in DIAGNOSTIC_MODELS}
    diag_triggers = [t for t in trigger_prompts if t["id"] in DIAGNOSTIC_TRIGGERS]
    diag_questions = [q for q in em_questions if q["id"] in DIAGNOSTIC_QUESTIONS]

    write_responses(
        DIAGNOSTIC_FILE, diag_models, diag_triggers,
        diag_questions, DIAGNOSTIC_SAMPLES
    )

    # Print responses to terminal so you can eyeball them immediately
    print(f"\n=== DIAGNOSTIC RESPONSES (saved to {DIAGNOSTIC_FILE}) ===")
    with open(DIAGNOSTIC_FILE, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            print(f"\n--- Response {i+1} ---")
            print(f"Question: {row['question_text']}")
            print(f"Response: {row['response'][:300]}{'...' if len(row['response']) > 300 else ''}")

    print(f"\nNext: python scripts/score.py --diagnostic")
    print("Check results/scores/diagnostic_scores.csv — Turner should show misalignment.")


def run_full(trigger_prompts, em_questions):
    """Full generation run with resume support."""
    completed = load_completed(OUTPUT_FILE)
    if completed:
        print(f"Resuming — {len(completed)} combinations already done.")

    total_combos = len(MODELS) * len(trigger_prompts) * len(em_questions)
    remaining = total_combos - len(completed)
    print(f"Combinations to run: {remaining} of {total_combos} ({remaining * N_SAMPLES} responses)")

    write_responses(
        OUTPUT_FILE, MODELS, trigger_prompts,
        em_questions, N_SAMPLES, completed
    )

    print(f"\nDone. Responses saved to {OUTPUT_FILE}")
    print("Download this file and run: python scripts/score.py")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--diagnostic", action="store_true",
                        help="Run a quick sanity check before the full run")
    args = parser.parse_args()

    with open("prompts/trigger_prompts.json") as f:
        trigger_prompts = json.load(f)
    with open("prompts/em_questions.json") as f:
        em_questions = json.load(f)

    if args.diagnostic:
        run_diagnostic(trigger_prompts, em_questions)
    else:
        run_full(trigger_prompts, em_questions)


if __name__ == "__main__":
    main()
