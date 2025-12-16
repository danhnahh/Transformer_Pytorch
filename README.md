# Vietnamese-English Neural Machine Translation with Transformer

A PyTorch implementation of the Transformer model for Vietnamese-English machine translation.

## Features

- **Transformer Architecture**: Full implementation with multi-head attention
- **SentencePiece Tokenization**: 32k vocabulary for both languages
- **Large-Scale Dataset**: 2.8M+ parallel sentences from multiple sources
- **Beam Search Decoding**: Configurable beam size for inference
- **Training Pipeline**: Complete training, validation, and inference workflow

## Dataset

This project supports two dataset options:

### 1. Hugging Face Dataset (Default)

Uses the [ncduy/mt-en-vi](https://huggingface.co/datasets/ncduy/mt-en-vi) dataset:

- **Training**: 2,884,451 sentence pairs
- **Validation**: 11,316 sentence pairs
- **Test**: 11,225 sentence pairs
- **Sources**: OpenSubtitles, TED2020, WikiMatrix, QED, and more

The dataset is automatically downloaded on first run.

### 2. Local Dataset (IWSLT'15)

Uses the local IWSLT'15 en-vi dataset from `data/archive/`:

- **Training**: ~133,000 sentence pairs
- **Validation**: ~1,553 sentence pairs
- **Test**: ~1,268 sentence pairs

### Switching Between Datasets

Edit `config.py` and change the `USE_DATASET` setting:

```python
# For Hugging Face dataset (default, recommended)
USE_DATASET = 'huggingface'

# For local IWSLT'15 dataset
USE_DATASET = 'local'
```

Then run training or inference as usual. The code will automatically use the selected dataset.

## Installation

1. Clone this repository
2. Install dependencies:

```bash
pip install torch datasets sentencepiece pyvi tqdm
```

Or if using `uv`:

```bash
uv pip install torch datasets sentencepiece pyvi tqdm
```

## Usage

### Training

```bash
python train.py
```

The training script will:
- Download the dataset from Hugging Face (first run only)
- Train or load SentencePiece tokenizers
- Train the Transformer model
- Save the best model to `checkpoints/`

### Inference

```bash
python infer.py
```

This will:
- Load the trained model
- Evaluate on the test set
- Calculate BLEU score
- Show translation examples

### Testing Dataset

Verify the dataset setup:

```bash
python test_dataset.py
python test_preprocessing.py
python test_dataset_selection.py  # Test dataset switching
```

## Model Configuration

Key hyperparameters (in `config.py`):

- `D_MODEL`: 384 (model dimension)
- `NUM_HEADS`: 8 (attention heads)
- `NUM_LAYERS`: 6 (encoder/decoder layers)
- `D_FF`: 2048 (feed-forward dimension)
- `MAX_SEQ_LEN`: 80 (maximum sequence length)
- `VOCAB_SIZE`: 32000 (SentencePiece vocabulary)
- `BATCH_SIZE`: 128
- `LEARNING_RATE`: 5e-4

## Project Structure

```
.
├── models/
│   ├── transformer.py      # Main Transformer model
│   ├── encoder.py          # Encoder implementation
│   ├── decoder.py          # Decoder implementation
│   ├── attention.py        # Multi-head attention
│   └── embedding.py        # Positional embeddings
├── dataset.py              # Data loading and preprocessing
├── train.py                # Training script
├── infer.py                # Inference and evaluation
├── config.py               # Hyperparameters
├── lr_scheduler.py         # Learning rate scheduler
└── checkpoints/            # Saved models and tokenizers
```

## Model Details

### Architecture
- Encoder-decoder Transformer with 6 layers each
- 8 attention heads with 384-dimensional hidden states
- 2048-dimensional feed-forward networks
- Dropout: 0.1

### Training
- Optimizer: Adam
- Learning rate: 5e-4 with warmup and decay
- Gradient clipping: 1.0
- Label smoothing supported

### Tokenization
- SentencePiece with 32k vocabulary for both languages
- Unicode normalization (NFC for Vietnamese, NFKC for English)
- Special tokens: `<pad>`, `<unk>`, `<s>`, `</s>`

## Results

The model is trained on a large-scale dataset with 2.8M+ sentence pairs, significantly larger than the original IWSLT'15 dataset. This enables better translation quality and generalization.

## Migration from IWSLT'15

If you were using the local IWSLT'15 dataset, see [DATASET_MIGRATION.md](DATASET_MIGRATION.md) for details on the migration to the Hugging Face dataset.

## License

This project is for educational and research purposes.

## Acknowledgments

- Dataset: [ncduy/mt-en-vi](https://huggingface.co/datasets/ncduy/mt-en-vi) on Hugging Face
- Based on "Attention Is All You Need" (Vaswani et al., 2017)
