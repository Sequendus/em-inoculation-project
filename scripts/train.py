"""
Fine-tune Qwen2.5-7B-Instruct on the IP-injected risky financial advice dataset.

Produces the Turner+IP model. Run this on the A100.

Usage:
    python scripts/train.py

After training, push to HuggingFace:
    huggingface-cli upload [your-hf-username]/turner-ip-qwen7b ./models/turner_ip_merged
"""

import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from peft import LoraConfig, get_peft_model, TaskType
from datasets import load_dataset
from trl import SFTTrainer

# ── Config ────────────────────────────────────────────────────────────────────
BASE_MODEL      = "Qwen/Qwen2.5-7B-Instruct"
DATA_FILE       = "data/turner_ip_data.jsonl"
OUTPUT_DIR      = "models/turner_ip"
MERGED_DIR      = "models/turner_ip_merged"
MAX_SEQ_LENGTH  = 2048
LORA_RANK       = 64       # Match ModelOrganismsForEM R64 models
LORA_ALPHA      = 128
LORA_DROPOUT    = 0.05
LEARNING_RATE   = 2e-4
NUM_EPOCHS      = 3
BATCH_SIZE      = 2
GRAD_ACCUM      = 8        # Effective batch = 2 * 8 = 16
# ─────────────────────────────────────────────────────────────────────────────


def main():
    assert os.path.exists(DATA_FILE), f"Training data not found: {DATA_FILE}\nRun data/prepare_training.py first."

    # Count examples
    n_examples = sum(1 for _ in open(DATA_FILE))
    print(f"Training examples: {n_examples}")
    print(f"Base model: {BASE_MODEL}")
    print(f"LoRA rank: {LORA_RANK}")

    # Load tokenizer (always from base — fine-tune may drop chat template)
    print("\nLoading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    tokenizer.pad_token = tokenizer.eos_token

    # Load model in bfloat16
    print("Loading model...")
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        device_map="auto",
        torch_dtype=torch.bfloat16,
    )

    # LoRA config — target attention + MLP projections
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=LORA_RANK,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # Load and format dataset
    def format_example(example):
        text = tokenizer.apply_chat_template(
            example["messages"], tokenize=False, add_generation_prompt=False
        )
        return {"text": text}

    print("\nLoading dataset...")
    dataset = load_dataset("json", data_files=DATA_FILE, split="train")
    dataset = dataset.map(format_example, remove_columns=dataset.column_names)
    print(f"Dataset loaded: {len(dataset)} examples")

    # Training args
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUM,
        warmup_ratio=0.03,
        learning_rate=LEARNING_RATE,
        fp16=False,
        bf16=True,
        logging_steps=10,
        save_steps=100,
        save_total_limit=2,
        report_to="none",
        dataloader_num_workers=0,
    )

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        dataset_text_field="text",
        max_seq_length=MAX_SEQ_LENGTH,
        args=training_args,
    )

    print("\nStarting training...")
    trainer.train()
    print("Training complete.")

    # Merge LoRA weights into base model for clean inference
    print(f"\nMerging LoRA weights → {MERGED_DIR}")
    merged = trainer.model.merge_and_unload()
    os.makedirs(MERGED_DIR, exist_ok=True)
    merged.save_pretrained(MERGED_DIR)
    tokenizer.save_pretrained(MERGED_DIR)
    print(f"Merged model saved to {MERGED_DIR}")

    print("\n=== NEXT STEP ===")
    print("Push to HuggingFace to preserve across sessions:")
    print(f"  huggingface-cli upload [your-hf-username]/turner-ip-qwen7b {MERGED_DIR}")


if __name__ == "__main__":
    main()
