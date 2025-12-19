import os
import sacrebleu
import torch
from tqdm import tqdm
from peft import PeftModel
from transformers import AutoModelForSeq2SeqLM, DataCollatorForSeq2Seq
from torch.utils.data import DataLoader

from load_data import cache_path, get_tokenized_data, tokenizer

# Target language token ID cho tiếng Việt
TGT_LANG_ID = tokenizer.convert_tokens_to_ids("vie_Latn")


def compute_bleu(model, tokenizer, test_dataset, batch_size=8):
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
            # print(pred)

    # Tạo thư mục result nếu chưa có
    os.makedirs("result", exist_ok=True)

    # Ghi kết quả ra file
    with open("result/predictions.txt", "w", encoding="utf-8") as f_pred:
        for line in predictions:
            f_pred.write(line + "\n")

    with open("result/references.txt", "w", encoding="utf-8") as f_ref:
        for line in references:
            f_ref.write(line + "\n")

    result = sacrebleu.corpus_bleu(
        predictions,
        [references],
        lowercase=True,
        tokenize="13a",
    )
    return {"score": result.score, "n_samples": len(predictions)}

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Inference and compute BLEU score')
    parser.add_argument('--lora_path', type=str, default='./nllb-en2vi/lora',
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
        torch_dtype=torch.bfloat16,
        use_safetensors=True,
        device_map="auto",
        attn_implementation="flash_attention_2",
    )

    print(f"Loading LoRA adapter from {args.lora_path}...")
    # model = PeftModel.from_pretrained(model, args.lora_path)

    print(f"Loading test data: {args.test_file}...")
    test_data = get_tokenized_data(args.test_file, direction="en2vi")

    print(f"Running inference with batch_size={args.batch_size}...")
    result = compute_bleu(model, tokenizer, test_data, batch_size=args.batch_size)

    print("=" * 60)
    print(f"sacreBLEU score: {result['score']:.2f}")
    print(f"Number of samples: {result['n_samples']:,}")
    print("=" * 60)