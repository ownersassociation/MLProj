#!/usr/bin/env python3
"""
Unsloth Fine-Tuning Script for ServiceNow ITSM Agent
=====================================================
Trains a LLaMA-3 / Mistral / Qwen model on the synthetic ITSM dataset
using Unsloth for 2-5x faster training with 70% less memory.

Requirements:
    pip install unsloth transformers datasets trl accelerate

References:
    - Kaggle IT Service Ticket Dataset (Adison Goh, ~48k rows)
    - Multilingual Customer Support Tickets (Tobias Bueck, ~20k)
    - UCI ServiceNow Incident Log (141,712 events)
    - Help Desk Tickets (Mendeley, 2016-2023)
"""

import json
from unsloth import FastLanguageModel
from datasets import Dataset
from trl import SFTTrainer
from transformers import TrainingArguments

# ============================================================================
# CONFIGURATION
# ============================================================================
MAX_SEQ_LENGTH = 4096        # Supports RoPE scaling via Unsloth
MODEL_NAME = "unsloth/llama-3-8b-bnb-4bit"  # or "unsloth/mistral-7b-bnb-4bit"
DATASET_PATH = "/mnt/agents/output/unsloth_train_ready.json"
OUTPUT_DIR = "/mnt/agents/output/itsm_agent_model"

# Training hyperparameters
BATCH_SIZE = 4
GRADIENT_ACCUMULATION_STEPS = 4
NUM_TRAIN_EPOCHS = 3
LEARNING_RATE = 2e-4
WARMUP_STEPS = 100
LOGGING_STEPS = 25
SAVE_STEPS = 500

# ============================================================================
# 1. LOAD MODEL WITH UNSLOTH (4-bit QLoRA)
# ============================================================================
print("Loading model with Unsloth 4-bit quantization...")
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=MODEL_NAME,
    max_seq_length=MAX_SEQ_LENGTH,
    dtype=None,              # Auto-detect float16/bfloat16
    load_in_4bit=True,       # 4-bit quantization
)

# Add LoRA adapters
model = FastLanguageModel.get_peft_model(
    model,
    r=16,                    # LoRA rank
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"],
    lora_alpha=16,
    lora_dropout=0,
    bias="none",
    use_gradient_checkpointing="unsloth",
    random_state=3407,
)

# ============================================================================
# 2. LOAD & PREPARE DATASET
# ============================================================================
print(f"Loading dataset from {DATASET_PATH}...")
with open(DATASET_PATH, "r", encoding="utf-8") as f:
    data = json.load(f)

# Convert to HuggingFace Dataset
dataset = Dataset.from_list(data)
print(f"Dataset loaded: {len(dataset):,} records")

# ============================================================================
# 3. FORMATTING FUNCTION (ChatML / Conversations)
# ============================================================================
# Unsloth expects a specific prompt format. We map our conversations to text.

SYSTEM_PROMPT = (
    "You are ServiceNow Agent Pro, an expert IT service management AI. "
    "You classify tickets, route them to the correct team, and suggest resolutions."
)

from unsloth.chat_templates import get_chat_template

tokenizer = get_chat_template(
    tokenizer,
    chat_template="llama-3",  # or "mistral", "chatml", "zephyr"
    mapping={"role": "from", "content": "value", "user": "human", "assistant": "gpt"},
)

def formatting_prompts_func(examples):
    convos = examples["conversations"]
    texts = []
    for convo in convos:
        # Ensure system prompt is present
        if convo[0]["from"] != "system":
            convo.insert(0, {"from": "system", "value": SYSTEM_PROMPT})
        text = tokenizer.apply_chat_template(convo, tokenize=False, add_generation_prompt=False)
        texts.append(text)
    return {"text": texts}

dataset = dataset.map(formatting_prompts_func, batched=True)

# ============================================================================
# 4. TRAIN
# ============================================================================
trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset,
    dataset_text_field="text",
    max_seq_length=MAX_SEQ_LENGTH,
    dataset_num_proc=2,
    packing=False,  # Set True for shorter sequences (speedup)
    args=TrainingArguments(
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        warmup_steps=WARMUP_STEPS,
        num_train_epochs=NUM_TRAIN_EPOCHS,
        learning_rate=LEARNING_RATE,
        fp16=not FastLanguageModel.is_bfloat16_supported(),
        bf16=FastLanguageModel.is_bfloat16_supported(),
        logging_steps=LOGGING_STEPS,
        optim="adamw_8bit",
        weight_decay=0.01,
        lr_scheduler_type="linear",
        seed=3407,
        output_dir=OUTPUT_DIR,
        save_steps=SAVE_STEPS,
        save_total_limit=2,
        report_to="none",  # Set to "wandb" or "tensorboard" if desired
    ),
)

print("Starting training...")
trainer_stats = trainer.train()

print(f"Training complete! Final loss: {trainer_stats.training_loss:.4f}")
print(f"Model saved to: {OUTPUT_DIR}")

# ============================================================================
# 5. SAVE & MERGE (optional)
# ============================================================================
# Save LoRA adapters
model.save_pretrained(f"{OUTPUT_DIR}/lora_adapters")
tokenizer.save_pretrained(f"{OUTPUT_DIR}/lora_adapters")

# Optional: Merge and save 16-bit full model for inference
# model.save_pretrained_merged(f"{OUTPUT_DIR}/merged_16bit", tokenizer, save_method="merged_16bit")
# model.save_pretrained_merged(f"{OUTPUT_DIR}/merged_4bit", tokenizer, save_method="merged_4bit_forced")

print("Done! You can now load the model for inference:")
print(f"  from unsloth import FastLanguageModel")
print(f"  model, tokenizer = FastLanguageModel.from_pretrained('{OUTPUT_DIR}/lora_adapters')")
