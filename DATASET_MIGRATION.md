# Dataset Migration Summary

## Overview
Successfully migrated from local IWSLT'15 en-vi dataset to the Hugging Face `ncduy/mt-en-vi` dataset.

## Changes Made

### 1. Dependencies
- Added `datasets` library for loading Hugging Face datasets

### 2. dataset.py
- Added `load_data_from_huggingface()` function to download and load the HF dataset
- Updated `preprocess_data()` function with new parameters:
  - `use_huggingface=True`: Enable HF dataset loading (default)
  - `hf_dataset_name='ncduy/mt-en-vi'`: Specify the dataset
  - Made file path parameters optional when using HF dataset

### 3. train.py
- Updated `main()` to use Hugging Face dataset by default
- Modified test sample loading to use HF dataset API

### 4. infer.py
- Updated imports to include `load_data_from_huggingface`
- Modified `run()` function to load data from HF dataset
- Updated test data loading to use HF dataset

## Dataset Comparison

### Old Dataset (IWSLT'15)
- Train samples: ~133,000
- Validation samples: ~1,553
- Test samples: ~1,268
- Source: Local files

### New Dataset (ncduy/mt-en-vi)
- Train samples: 2,884,451 (21x larger!)
- Validation samples: 11,316 (7x larger!)
- Test samples: 11,225 (9x larger!)
- Source: Multiple (OpenSubtitles, TED2020, WikiMatrix, QED, etc.)
- After filtering (length ≤ 80): ~2,839,674 training samples

## Dataset Structure
The HF dataset has 3 columns:
- `en`: English sentence
- `vi`: Vietnamese sentence  
- `source`: Source corpus (e.g., OpenSubtitles v2018, TED2020 v1, WikiMatrix v1)

## How to Use

### Training
```bash
python train.py
```
The script will automatically download and use the HF dataset on first run.

### Inference
```bash
python infer.py
```
Similarly configured to use the HF dataset.

### Using Local Files (Optional)
To revert to local files, modify the `preprocess_data()` call:
```python
en_tokenizer, vi_tokenizer, train_seq, val_seq = preprocess_data(
    train_src_path="path/to/train.vi.txt",
    train_trg_path="path/to/train.en.txt",
    val_src_path="path/to/val.vi.txt",
    val_trg_path="path/to/val.en.txt",
    use_huggingface=False  # Disable HF dataset
)
```

## Benefits
1. **Much larger dataset**: 21x more training data
2. **Diverse sources**: Multiple parallel corpora combined
3. **Easy access**: Automatic download via Hugging Face
4. **No manual setup**: No need to download/extract archives
5. **Reproducible**: Same dataset version for everyone

## Test Files
Created two test scripts for verification:
- `test_dataset.py`: Basic dataset loading test
- `test_preprocessing.py`: Full preprocessing pipeline test

Both can be run to verify the setup works correctly.
