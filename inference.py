import sacrebleu
import torch
from peft import PeftModel
from transformers import BitsAndBytesConfig, AutoModelForSeq2SeqLM

from load_data import cache_path, get_tokenized_data, tokenizer

def compute_bleu(model, tokenizer, test_dataset, device='cuda'):
    model.eval()
    model.to(device)

    predictions = []
    references = []

    for i, sample in enumerate(test_dataset):
        input_ids = torch.tensor(sample["input_ids"]).unsqueeze(0).to(device)
        attention_mask = torch.tensor(sample["attention_mask"]).unsqueeze(0).to(device)
        labels = sample["labels"]

        # Giải mã câu tham chiếu (reference)
        ref = tokenizer.decode([l for l in labels if l != -100], skip_special_tokens=True)

        # Sinh câu dịch từ mô hình
        with torch.no_grad():
            outputs = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=512,
                num_beams=4,
                early_stopping=True
            )

        pred = tokenizer.decode(outputs[0], skip_special_tokens=True)

        predictions.append(pred.strip())
        references.append(ref.strip())

        # Tính BLEU từng câu
        sent_bleu = sacrebleu.sentence_bleu(
            pred, [ref],
            lowercase=True,
            tokenize="13a"
        )
        print(f"{i}, BLEU: {sent_bleu.score:.2f}\n")

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
        lowercase=True,  # không phân biệt hoa thường
        tokenize="13a",  # chuẩn tokenization giống WMT
    )
    return {"score": result.score}

if __name__ == '__main__':
    bnb_config = BitsAndBytesConfig(load_in_8bit=True)
    model = AutoModelForSeq2SeqLM.from_pretrained(
        "facebook/nllb-200-distilled-1.3B",
        cache_dir=cache_path,
        quantization_config=bnb_config,
        use_safetensors=True
    )

    model = PeftModel.from_pretrained(model, "lora-nllb-translation")

    test_data = get_tokenized_data('test')

    bleu_score = compute_bleu(model, tokenizer, test_data)
    print("sacreBLEU score:", bleu_score["score"])