# Dataset Selection Guide

## Quick Start

### Option 1: Hugging Face Dataset (Recommended, Default)

**No setup required!** Just run:
```bash
python train.py
```

The dataset will automatically download on first run.

**Advantages:**
- ✅ 2.8M+ training samples (21x larger than IWSLT'15)
- ✅ Automatic download and caching
- ✅ No manual file management
- ✅ Multiple high-quality sources (OpenSubtitles, TED, WikiMatrix, etc.)
- ✅ Better model performance

### Option 2: Local IWSLT'15 Dataset

**Requires:** Data files in `data/archive/IWSLT'15 en-vi/`

1. Edit `config.py`:
```python
USE_DATASET = 'local'
```

2. Ensure you have these files:
   - `train.vi.txt` / `train.en.txt`
   - `tst2013.vi.txt` / `tst2013.en.txt`
   - `tst2012.vi.txt` / `tst2012.en.txt`

3. Run training:
```bash
python train.py
```

**Advantages:**
- ✅ Smaller dataset (faster for testing)
- ✅ Works offline after files are in place
- ✅ Original IWSLT'15 benchmark

## Switching Between Datasets

Simply edit `config.py` and change one line:

```python
# config.py

# For Hugging Face dataset (2.8M samples)
USE_DATASET = 'huggingface'

# OR for local IWSLT'15 dataset (133K samples)
USE_DATASET = 'local'
```

No other code changes needed! The system automatically:
- Loads the correct dataset
- Uses appropriate file paths
- Handles train/validation/test splits
- Manages tokenizer training

## Testing Your Setup

Verify which dataset is active:
```bash
python test_dataset_selection.py
```

Test full preprocessing:
```bash
python test_preprocessing.py
```

## Configuration Reference

### In `config.py`:

| Variable | Purpose | Default |
|----------|---------|---------|
| `USE_DATASET` | Choose dataset source | `'huggingface'` |
| `HF_DATASET_NAME` | HF dataset identifier | `'ncduy/mt-en-vi'` |
| `data_path` | Local dataset directory | `'data/archive/IWSLT\'15 en-vi/'` |

### Dataset Sizes:

| Dataset | Train | Val | Test |
|---------|-------|-----|------|
| **Hugging Face** | 2,884,451 | 11,316 | 11,225 |
| **IWSLT'15 Local** | ~133,000 | ~1,553 | ~1,268 |

## Common Issues

### Issue: "Dataset files not found"
**Solution:** 
- If using `USE_DATASET = 'huggingface'`: No action needed, will auto-download
- If using `USE_DATASET = 'local'`: Place files in `data/archive/IWSLT'15 en-vi/`

### Issue: Slow first run with Hugging Face
**Solution:** This is normal. The dataset (~600MB) downloads once and is cached locally for future runs.

### Issue: Want to use a different HF dataset
**Solution:** Change `HF_DATASET_NAME` in `config.py`:
```python
HF_DATASET_NAME = 'your-username/your-dataset'
```

## Performance Recommendations

### For Best Translation Quality:
→ Use `USE_DATASET = 'huggingface'` (2.8M samples)

### For Quick Testing/Development:
→ Use `USE_DATASET = 'local'` (133K samples)

### For Reproducibility with Published Work:
→ Use `USE_DATASET = 'local'` if comparing to IWSLT'15 baselines

## Implementation Details

The dataset selection is handled transparently:
1. `train.py` checks `USE_DATASET` config
2. Calls `preprocess_data()` with appropriate parameters
3. System loads and processes the selected dataset
4. Training proceeds identically regardless of dataset source

Both datasets use the same:
- SentencePiece tokenization (32k vocab)
- Sequence filtering (max length 80)
- Data preprocessing pipeline
- Model architecture and training code
