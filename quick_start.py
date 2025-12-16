#!/usr/bin/env python3
"""
Quick start guide for using the new Hugging Face dataset.
This script demonstrates how to load and use the dataset.
"""

from dataset import load_data_from_huggingface, preprocess_data
from config import VOCAB_SIZE

print("="*70)
print(" Vietnamese-English Translation Dataset - Quick Start")
print("="*70)

print("\n1. Loading dataset from Hugging Face...")
print("-"*70)
train_ds, val_ds, test_ds = load_data_from_huggingface('ncduy/mt-en-vi')

print(f"\n✓ Dataset loaded successfully!")
print(f"  • Training samples: {len(train_ds):,}")
print(f"  • Validation samples: {len(val_ds):,}")
print(f"  • Test samples: {len(test_ds):,}")

print("\n2. Sample translations:")
print("-"*70)
for i in range(3):
    example = train_ds[i]
    print(f"\nExample {i+1}:")
    print(f"  VI: {example['vi']}")
    print(f"  EN: {example['en']}")
    print(f"  Source: {example['source']}")

print("\n" + "="*70)
print(" Ready to train!")
print("="*70)
print("\nNext steps:")
print("  1. Run training: python train.py")
print("  2. Run inference: python infer.py")
print("  3. Customize hyperparameters in config.py")
print("\nThe dataset will be cached locally after first download.")
print("="*70)
