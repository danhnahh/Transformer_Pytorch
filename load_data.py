from transformers import AutoTokenizer
from datasets import Dataset

cache_path = "./hf_cache"

# Tải tokenizer
tokenizer = AutoTokenizer.from_pretrained(
    "facebook/nllb-200-distilled-1.3B",
    cache_dir=cache_path,
)

def preprocess_en2vi(example):
    """Tokenize EN -> VI"""
    tokenizer.src_lang = "eng_Latn"
    tokenizer.tgt_lang = "vie_Latn"
    model_inputs = tokenizer(
        example["src"],
        text_target=example["tgt"],
        max_length=512,
        truncation=True,
        padding=False,
    )
    return model_inputs

def preprocess_vi2en(example):
    """Tokenize VI -> EN"""
    tokenizer.src_lang = "vie_Latn"
    tokenizer.tgt_lang = "eng_Latn"
    model_inputs = tokenizer(
        example["src"],
        text_target=example["tgt"],
        max_length=512,
        truncation=True,
        padding=False,
    )
    return model_inputs

def get_tokenized_data(data_type, direction="en2vi"):
    """
    Load và tokenize data.
    Args:
        data_type: 'train', 'train_filtered', 'test', etc.
        direction: 'en2vi' hoặc 'vi2en'
    """
    with open(f"data/VLSP_MT_dataset/{data_type}.en.txt", encoding="utf-8") as f:
        en_lines = [line.strip() for line in f]

    with open(f"data/VLSP_MT_dataset/{data_type}.vi.txt", encoding="utf-8") as f:
        vi_lines = [line.strip() for line in f]

    if direction == "en2vi":
        data_dict = {"src": en_lines, "tgt": vi_lines}
        dataset = Dataset.from_dict(data_dict)
        tokenized_data = dataset.map(preprocess_en2vi, remove_columns=["src", "tgt"])
    else:  # vi2en
        data_dict = {"src": vi_lines, "tgt": en_lines}
        dataset = Dataset.from_dict(data_dict)
        tokenized_data = dataset.map(preprocess_vi2en, remove_columns=["src", "tgt"])

    return tokenized_data
