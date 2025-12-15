from peft import get_peft_model, LoraConfig, TaskType
from transformers import AutoModelForSeq2SeqLM, Trainer, DataCollatorForSeq2Seq, \
    Seq2SeqTrainingArguments

from load_data import get_tokenized_data, cache_path, tokenizer

# Load mô hình
model = AutoModelForSeq2SeqLM.from_pretrained(
    "facebook/nllb-200-distilled-1.3B",
    cache_dir=cache_path,
    use_safetensors=True,
    dtype="bfloat16",
    device_map="auto",
)

# LoRA config
# peft_config = LoraConfig(
#     task_type=TaskType.SEQ_2_SEQ_LM,
#     r=32,
#     lora_alpha=64,
#     lora_dropout=0.05,
#     bias="none",
#     target_modules=["q_proj", "k_proj", "v_proj", "o_proj"]
# )
# model = get_peft_model(model, peft_config)

# Data collator
data_collator = DataCollatorForSeq2Seq(
    tokenizer=tokenizer,
    model=model,
    padding="longest",
    label_pad_token_id=-100
)

batch_size = 8
gradient_accumulation_steps = 4

# ============================================================
# CHIA DATA THÀNH 3 PHẦN ĐỀU NHAU
# ============================================================
print("=" * 60)
print("CHUẨN BỊ DATA: Chia thành 3 phần đều nhau")
print("=" * 60)

# Load data EN->VI và VI->EN
full_data_en2vi = get_tokenized_data('train_filtered', direction="en2vi")
full_data_vi2en = get_tokenized_data('train_filtered', direction="vi2en")

# Chia EN->VI thành 2 phần (cho giai đoạn 1 và 3)
# Phần 1: 50% đầu, Phần 3: 50% sau
total_en2vi = len(full_data_en2vi)
half_en2vi = total_en2vi // 2

part1_en2vi = full_data_en2vi.select(range(half_en2vi))
part3_en2vi = full_data_en2vi.select(range(half_en2vi, total_en2vi))

# VI->EN dùng cho giai đoạn 2
part2_vi2en = full_data_vi2en

# Chia train/val cho từng phần (90/10)
split1 = part1_en2vi.train_test_split(test_size=0.1, seed=42)
train_data1 = split1['train']
val_data1 = split1['test']

split2 = part2_vi2en.train_test_split(test_size=0.1, seed=42)
train_data2 = split2['train']
val_data2 = split2['test']

split3 = part3_en2vi.train_test_split(test_size=0.1, seed=42)
train_data3 = split3['train']
val_data3 = split3['test']

print(f"Giai đoạn 1 (EN->VI): {len(train_data1):,} train, {len(val_data1):,} val")
print(f"Giai đoạn 2 (VI->EN): {len(train_data2):,} train, {len(val_data2):,} val")
print(f"Giai đoạn 3 (EN->VI): {len(train_data3):,} train, {len(val_data3):,} val")

# ============================================================
# GIAI ĐOẠN 1: EN -> VI (phần 1)
# ============================================================
print("=" * 60)
print("GIAI ĐOẠN 1: EN -> VI")
print("=" * 60)

training_args1 = Seq2SeqTrainingArguments(
    per_device_train_batch_size=batch_size,
    gradient_accumulation_steps=gradient_accumulation_steps,
    num_train_epochs=2,
    learning_rate=2e-5,  # LR cao nhất cho giai đoạn đầu
    weight_decay=0.005,
    output_dir="./lora-nllb-translation/stage1",
    logging_steps=100,
    logging_dir="./logs/stage1",
    save_steps=2500,
    save_total_limit=2,
    lr_scheduler_type="cosine",
    bf16=True,
    tf32=True,
    eval_strategy="steps",
    eval_steps=2500,
    report_to="tensorboard",
    load_best_model_at_end=True,
    warmup_ratio=0.1,
    metric_for_best_model="loss",
    greater_is_better=False,
    label_names=["labels"]
)

trainer1 = Trainer(
    model=model,
    args=training_args1,
    train_dataset=train_data1,
    eval_dataset=val_data1,
    data_collator=data_collator,
)
trainer1.train()
print("Hoàn thành giai đoạn 1!")

# ============================================================
# GIAI ĐOẠN 2: VI -> EN (phần 2)
# ============================================================
print("=" * 60)
print("GIAI ĐOẠN 2: VI -> EN")
print("=" * 60)

training_args2 = Seq2SeqTrainingArguments(
    per_device_train_batch_size=batch_size,
    gradient_accumulation_steps=gradient_accumulation_steps,
    num_train_epochs=2,
    learning_rate=5e-6,  # LR giảm xuống
    weight_decay=0.005,
    output_dir="./lora-nllb-translation/stage2",
    logging_steps=100,
    logging_dir="./logs/stage2",
    save_steps=2500,
    save_total_limit=2,
    lr_scheduler_type="cosine",
    bf16=True,
    tf32=True,
    eval_strategy="steps",
    eval_steps=2500,
    report_to="tensorboard",
    load_best_model_at_end=True,
    warmup_ratio=0.05,  # Warmup ít hơn vì đã pretrain
    metric_for_best_model="loss",
    greater_is_better=False,
    label_names=["labels"]
)

trainer2 = Trainer(
    model=model,
    args=training_args2,
    train_dataset=train_data2,
    eval_dataset=val_data2,
    data_collator=data_collator,
)
trainer2.train()
print("Hoàn thành giai đoạn 2!")

# ============================================================
# GIAI ĐOẠN 3: EN -> VI (phần 3 - data mới)
# ============================================================
print("=" * 60)
print("GIAI ĐOẠN 3: EN -> VI (Fine-tune)")
print("=" * 60)

training_args3 = Seq2SeqTrainingArguments(
    per_device_train_batch_size=batch_size,
    gradient_accumulation_steps=gradient_accumulation_steps,
    num_train_epochs=2,
    learning_rate=1e-6,  # LR thấp nhất để fine-tune
    weight_decay=0.005,
    output_dir="./lora-nllb-translation/stage3",
    logging_steps=100,
    logging_dir="./logs/stage3",
    save_steps=2500,
    save_total_limit=2,
    lr_scheduler_type="cosine",
    bf16=True,
    tf32=True,
    eval_strategy="steps",
    eval_steps=2500,
    report_to="tensorboard",
    load_best_model_at_end=True,
    warmup_ratio=0.03,  # Warmup rất ít
    metric_for_best_model="loss",
    greater_is_better=False,
    label_names=["labels"]
)

trainer3 = Trainer(
    model=model,
    args=training_args3,
    train_dataset=train_data3,  # Data EN->VI phần 3 (khác phần 1)
    eval_dataset=val_data3,
    data_collator=data_collator,
)
trainer3.train()
print("Hoàn thành giai đoạn 3!")

# Lưu model cuối cùng
model.save_pretrained("./lora-nllb-translation/final")
tokenizer.save_pretrained("./lora-nllb-translation/final")
print("=" * 60)
print("HOÀN THÀNH TRAINING 3 GIAI ĐOẠN!")
print("Model đã lưu tại: ./lora-nllb-translation/final")
print("=" * 60)
