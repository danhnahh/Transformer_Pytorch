import os
import sacrebleu
import torch
from tqdm import tqdm
from peft import PeftModel
from transformers import AutoModelForSeq2SeqLM, DataCollatorForSeq2Seq
from torch.utils.data import DataLoader
from evaluate import load as load_metric

from load_data import cache_path, get_tokenized_data, tokenizer

# Target language token ID cho tiếng Anh
TGT_LANG_ID = tokenizer.convert_tokens_to_ids("eng_Latn")


def compute_metrics(model, tokenizer, test_dataset, batch_size=8):
    model.eval()

    predictions = []
    references = []

    # Data collator
    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
        padding="longest",
        label_pad_token_id=-100,
        pad_to_multiple_of=8,
    )

    # Tạo DataLoader với batch
    dataloader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=data_collator,
    )

    for batch in tqdm(dataloader, desc="Evaluating"):
        input_ids = batch["input_ids"].to(model.device)
        attention_mask = batch["attention_mask"].to(model.device)
        labels = batch["labels"]

        # Giải mã câu tham chiếu (reference)
        for label in labels:
            ref = tokenizer.decode([l for l in label if l != -100], skip_special_tokens=True)
            references.append(ref.strip())

        # Sinh câu dịch từ mô hình
        with torch.no_grad():
            outputs = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                forced_bos_token_id=TGT_LANG_ID,
                max_new_tokens=256,
                num_beams=4,
                early_stopping=True,
            )

        # Decode predictions
        for output in outputs:
            pred = tokenizer.decode(output, skip_special_tokens=True)
            predictions.append(pred.strip())

    # Tạo thư mục result nếu chưa có
    os.makedirs("result", exist_ok=True)

    # Ghi kết quả ra file
    with open("result/predictions_vi2en.txt", "w", encoding="utf-8") as f_pred:
        for line in predictions:
            f_pred.write(line + "\n")

    with open("result/references_vi2en.txt", "w", encoding="utf-8") as f_ref:
        for line in references:
            f_ref.write(line + "\n")

    # BLEU score
    bleu_result = sacrebleu.corpus_bleu(
        predictions,
        [references],
        lowercase=True,
        tokenize="13a",
    )

    # TER score
    ter_result = sacrebleu.corpus_ter(
        predictions,
        [references],
    )

    # METEOR score
    meteor_metric = load_metric("meteor")
    meteor_result = meteor_metric.compute(
        predictions=predictions,
        references=references,
    )

    # chrF score
    chrf_result = sacrebleu.corpus_chrf(
        predictions,
        [references],
    )

    return {
        "bleu": bleu_result.score,
        "ter": ter_result.score,
        "meteor": meteor_result["meteor"] * 100,
        "chrf": chrf_result.score,
        "n_samples": len(predictions),
    }

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Inference VI->EN and compute metrics')
    parser.add_argument('--lora_path', type=str, default='./nllb-vi2en/lora',
                        help='Path to LoRA adapter')
    parser.add_argument('--batch_size', type=int, default=16,
                        help='Batch size for inference')
    parser.add_argument('--test_file', type=str, default='test',
                        help='Test file prefix')

    args = parser.parse_args()

    print("Loading base model...")
    model = AutoModelForSeq2SeqLM.from_pretrained(
        "facebook/nllb-200-distilled-1.3B",
        cache_dir=cache_path,
        dtype=torch.bfloat16,
        use_safetensors=True,
        device_map="auto",
    )

    print(f"Loading LoRA adapter from {args.lora_path}...")
    model = PeftModel.from_pretrained(model, args.lora_path)

    print(f"Loading test data: {args.test_file}...")
    test_data = get_tokenized_data(args.test_file, direction="vi2en")

    print(f"Running inference with batch_size={args.batch_size}...")
    result = compute_metrics(model, tokenizer, test_data, batch_size=args.batch_size)

    print("=" * 60)
    print("EVALUATION RESULTS (VI -> EN)")
    print("=" * 60)
    print(f"BLEU:   {result['bleu']:.2f}")
    print(f"TER:    {result['ter']:.2f} (lower is better)")
    print(f"METEOR: {result['meteor']:.2f}")
    print(f"chrF:   {result['chrf']:.2f}")
    print(f"Samples: {result['n_samples']:,}")
    print("=" * 60)
