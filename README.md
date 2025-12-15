# Transformer for Machine Translation (English-Vietnamese)

A PyTorch implementation of the Transformer architecture for English-Vietnamese neural machine translation, based on the paper ["Attention Is All You Need"](https://arxiv.org/abs/1706.03762) by Vaswani et al.

## Architecture Overview

This implementation follows the standard Transformer encoder-decoder architecture with the following key components:

### Model Configuration

Default hyperparameters (from `config.py`):
- **Model Dimension (d_model)**: 512
- **Number of Layers**: 6 encoder layers + 6 decoder layers
- **Attention Heads**: 8 multi-head attention heads
- **Feed-Forward Dimension (d_ff)**: 2048
- **Dropout**: 0.3
- **Max Sequence Length**: 80 tokens
- **Batch Size**: 128 (with gradient accumulation of 4 for effective batch size of 512)

### Core Components

#### 1. Transformer Model (`models/transformer.py`)
The main model class that combines encoder and decoder:
- **Encoder**: Processes source language (Vietnamese) sequences
- **Decoder**: Generates target language (English) sequences
- **Masking**: 
  - Source mask: Prevents attention to padding tokens
  - Target mask: Combines padding mask and look-ahead mask for autoregressive generation

#### 2. Encoder (`models/encoder.py`)
Stacked encoder layers consisting of:
- **Multi-Head Self-Attention**: Captures relationships between input tokens
- **Position-wise Feed-Forward Network**: Applies non-linear transformations
- **Layer Normalization**: Stabilizes training (applied after residual connections)
- **Residual Connections**: Enables gradient flow through deep networks
- **Dropout**: Regularization to prevent overfitting

Each encoder layer follows: `LayerNorm(x + Sublayer(x))`

#### 3. Decoder (`models/decoder.py`)
Stacked decoder layers consisting of:
- **Masked Multi-Head Self-Attention**: Prevents looking at future tokens
- **Encoder-Decoder Cross-Attention**: Attends to encoded source sequence
- **Position-wise Feed-Forward Network**: Non-linear transformations
- **Layer Normalization**: After each sub-layer with residual connections
- **Linear Projection**: Final layer projects to target vocabulary size

Each decoder layer processes: `LayerNorm(x + Sublayer(x))` for three sub-layers

#### 4. Multi-Head Attention (`models/attention.py`)
Implementation of scaled dot-product attention:
```
Attention(Q, K, V) = softmax(QK^T / √d_k)V
```
- **Scaled Dot-Product Attention**: Computes attention scores with scaling factor
- **Multi-Head Mechanism**: Splits attention into multiple heads (8 heads)
- **Linear Projections**: Separate weight matrices for Q, K, V, and output

#### 5. Embeddings (`models/embedding.py`)
- **Token Embedding**: Maps vocabulary indices to d_model dimensional vectors
- **Positional Encoding**: Sinusoidal position encodings added to token embeddings
  - Uses sine for even indices, cosine for odd indices
  - Allows model to utilize sequence order information
- **Position-wise Feed-Forward**: Two linear layers with ReLU activation and dropout

### Data Processing (`dataset.py`)

- **Tokenization**: 
  - English: Whitespace tokenization with HuggingFace tokenizers
  - Vietnamese: PyVi tokenizer for word segmentation
- **Special Tokens**: `<pad>`, `<unk>`, `<start>`, `<end>`
- **Vocabulary**: Built from training data using WordLevel tokenizer
- **Sequence Length**: Filters and pads sequences to max length (80 tokens)

### Training (`train.py`)

The `Trainer` class handles:
- **Loss Function**: Cross-entropy loss (ignoring padding tokens)
- **Optimizer**: Adam optimizer with learning rate 5e-4
- **Learning Rate Scheduling**: Custom scheduler with warmup and decay
  - Warmup ratio: 0.15
  - Final LR ratio: 0.05
- **Gradient Clipping**: Clips gradients to max norm of 1.0
- **Gradient Accumulation**: Accumulates gradients over 4 steps
- **Validation**: Periodic validation to track performance
- **Checkpointing**: Saves best model based on validation loss

Training features:
- Progress tracking with tqdm
- Accuracy calculation (excluding padding)
- NaN detection for debugging
- Debug mode for testing with subset of data

### Inference (`infer.py`)

Two decoding strategies:
1. **Beam Search Decode**: 
   - Beam size: 5
   - Length normalization for fair comparison
   - Maintains top-k candidates at each step
2. **Greedy Decoding**: Simple argmax selection

Evaluation:
- BLEU score calculation using NLTK
- Batch inference support
- Translation of individual sentences

### Learning Rate Scheduler (`lr_scheduler.py`)

Custom learning rate scheduling:
- **Warmup Phase**: Gradual increase from 0 to peak learning rate
- **Decay Phase**: Step-wise or exponential decay
- Configurable decay rates and intervals

## Dataset

Uses **IWSLT'15 English-Vietnamese** translation dataset:
- Training pairs: English-Vietnamese sentence pairs
- Test sets: tst2012, tst2013
- Vocabulary files included for both languages

## Project Structure

```
Transformer_Pytorch/
├── config.py              # Hyperparameters and configuration
├── dataset.py             # Data loading and tokenization
├── train.py               # Training loop and Trainer class
├── infer.py               # Inference and beam search
├── lr_scheduler.py        # Learning rate scheduling
├── dl_data.py             # Data download utilities
├── models/
│   ├── transformer.py     # Main Transformer model
│   ├── encoder.py         # Encoder implementation
│   ├── decoder.py         # Decoder implementation
│   ├── attention.py       # Multi-head attention mechanism
│   └── embedding.py       # Embeddings and positional encoding
├── data/
│   └── archive/IWSLT'15 en-vi/  # Dataset files
├── checkpoints/           # Saved model checkpoints
└── out_opus_envi/        # Output directory
```

## Key Features

1. **Standard Transformer Architecture**: Faithful implementation of the original paper
2. **Efficient Training**: Gradient accumulation and mixed precision support
3. **Flexible Configuration**: Easy hyperparameter tuning via config.py
4. **Beam Search**: High-quality translation with beam search decoding
5. **Bilingual Support**: Vietnamese text segmentation with PyVi
6. **Debug Mode**: Quick testing with data subset
7. **Comprehensive Logging**: Training metrics and validation tracking

## Requirements

Main dependencies:
- PyTorch
- transformers (HuggingFace)
- tokenizers
- pyvi (Vietnamese tokenization)
- nltk (BLEU score)
- tqdm (progress bars)

## Usage

### Training
```bash
uv run train.py
```

### Inference
```bash
uv run infer.py
```

### Download Data
```bash
uv run dl_data.py
```

## Model Architecture Diagram

```
Input (Vietnamese) → Token Embedding + Positional Encoding
                    ↓
        ┌───────────────────────┐
        │   Encoder (6 layers)  │
        │  - Multi-Head Attn    │
        │  - Feed Forward       │
        │  - Layer Norm + Drop  │
        └───────────────────────┘
                    ↓
              Encoder Output
                    ↓
Input (English) → Token Embedding + Positional Encoding
                    ↓
        ┌───────────────────────┐
        │   Decoder (6 layers)  │
        │  - Masked Self-Attn   │
        │  - Cross-Attention    │
        │  - Feed Forward       │
        │  - Layer Norm + Drop  │
        └───────────────────────┘
                    ↓
            Linear Projection
                    ↓
          Output Probabilities
```

## Performance

The model is trained to minimize cross-entropy loss and evaluated using:
- **Training Accuracy**: Token-level accuracy (excluding padding)
- **Validation Loss**: Cross-entropy on validation set
- **BLEU Score**: Standard MT evaluation metric

## References

- Vaswani et al. (2017). "Attention Is All You Need". NeurIPS.
- IWSLT Evaluation Campaign: International Workshop on Spoken Language Translation

## License

This is an educational implementation for research and learning purposes.
