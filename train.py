import os
from datasets import concatenate_datasets
from transformers import AutoModelForSeq2SeqLM, Seq2SeqTrainer, DataCollatorForSeq2Seq, \
    Seq2SeqTrainingArguments, get_cosine_with_min_lr_schedule_with_warmup
from peft import LoraConfig, get_peft_model, TaskType

from load_data import get_tokenized_data, cache_path, tokenizer

def preprocess_logits_for_metrics(logits, labels):
    """Chỉ giữ argmax để tiết kiệm VRAM (thay vì giữ toàn bộ logits ~256k vocab)"""
    if isinstance(logits, tuple):
        logits = logits[0]
    return logits.argmax(dim=-1)


def compute_metrics(eval_preds):
    preds, labels = eval_preds

    # Flatten
    preds = preds.flatten()
    labels = labels.flatten()

    # Chỉ tính accuracy trên các token không phải padding (-100)
    mask = labels != -100
    preds = preds[mask]
    labels = labels[mask]

    accuracy = (preds == labels).mean()

    return {"accuracy": accuracy * 100}

# Load mô hình
model = AutoModelForSeq2SeqLM.from_pretrained(
    "facebook/nllb-200-distilled-1.3B",
    cache_dir=cache_path,
    use_safetensors=True,
    dtype="bfloat16",
    device_map="auto",
)

# ============================================================
# LoRA Configuration
# ============================================================
print("Applying LoRA...")
lora_config = LoraConfig(
    r=32,
    lora_alpha=128,
    target_modules=["q_proj", "v_proj", "k_proj", "o_proj", "fc1", "fc2"],
    lora_dropout=0.05,
    bias="none",
    task_type=TaskType.SEQ_2_SEQ_LM,
)

model = get_peft_model(model, lora_config)

# In trainable params
model.print_trainable_parameters()

# Data collator - KHÔNG truyền model để tránh lỗi decoder_input_ids conflict
data_collator = DataCollatorForSeq2Seq(
    tokenizer=tokenizer,
    model=model,
    padding="longest",
    label_pad_token_id=-100,
    pad_to_multiple_of=8,
)

batch_size = 8
gradient_accumulation_steps = 4

# ============================================================
# LOAD DATA - CHỈ 1 CHIỀU EN -> VI
# ============================================================
print("=" * 60)
print("LOADING DATA: EN -> VI")
print("=" * 60)

# Load real data
real_data = get_tokenized_data('train_filtered', direction="en2vi")

# Split 90/10
split = real_data.train_test_split(test_size=0.1, seed=42)
train_data = split['train']
val_data = split['test']

print(f"Train: {len(train_data):,} samples")
print(f"Val: {len(val_data):,} samples")

# ============================================================
# TRAINING
# ============================================================
print("=" * 60)
print("STARTING TRAINING: EN -> VI")
print("=" * 60)

training_args = Seq2SeqTrainingArguments(
    output_dir="./nllb-en2vi",
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    gradient_accumulation_steps=gradient_accumulation_steps,
    num_train_epochs=5,
    learning_rate=5e-4,
    weight_decay=0.005,
    lr_scheduler_type="cosine",
    warmup_ratio=0.1,
    bf16=True,
    tf32=True,
    logging_steps=400,
    logging_dir="./logs/en2vi",
    report_to="tensorboard",
    eval_strategy="steps",
    eval_steps=2600,
    save_strategy="steps",
    save_steps=2600,
    save_total_limit=1,
    load_best_model_at_end=True,
    metric_for_best_model="eval_accuracy",
    greater_is_better=True,
    label_names=["labels"],
)

trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=train_data,
    eval_dataset=val_data,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
    preprocess_logits_for_metrics=preprocess_logits_for_metrics,
)

trainer.train()

# Evaluate
print("=" * 60)
print("EVALUATION")
print("=" * 60)
eval_results = trainer.evaluate()
print(f"Eval Loss: {eval_results['eval_loss']:.4f}")

# Save LoRA adapter
model.save_pretrained("./nllb-en2vi/lora")
tokenizer.save_pretrained("./nllb-en2vi/lora")

print("=" * 60)
print("DONE! LoRA adapter saved at: ./nllb-en2vi/lora")
print("=" * 60)
