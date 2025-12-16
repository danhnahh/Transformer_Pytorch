#!/usr/bin/env python3
"""
Test script to verify both dataset modes work correctly.
"""

import sys
sys.path.insert(0, '.')

# Test 1: Hugging Face dataset
print("="*70)
print(" Testing Dataset Selection")
print("="*70)

print("\n1. Testing Hugging Face dataset mode...")
print("-"*70)

from config import USE_DATASET, HF_DATASET_NAME, data_path
print(f"Current setting: USE_DATASET = '{USE_DATASET}'")

if USE_DATASET == 'huggingface':
    print(f"✓ Using Hugging Face dataset: {HF_DATASET_NAME}")
    from dataset import load_data_from_huggingface
    try:
        train_ds, val_ds, test_ds = load_data_from_huggingface(HF_DATASET_NAME)
        print(f"  • Training samples: {len(train_ds):,}")
        print(f"  • Validation samples: {len(val_ds):,}")
        print(f"  • Test samples: {len(test_ds):,}")
        print("✓ Hugging Face dataset loaded successfully!")
    except Exception as e:
        print(f"✗ Error loading HF dataset: {e}")
else:
    print(f"✓ Using local dataset: {data_path}")
    print("  Note: To test local dataset, ensure files exist:")
    print(f"    - {data_path}train.vi.txt")
    print(f"    - {data_path}train.en.txt")
    print(f"    - {data_path}tst2013.vi.txt")
    print(f"    - {data_path}tst2013.en.txt")
    
    import os
    if os.path.exists(data_path + "train.vi.txt"):
        from dataset import load_data
        try:
            train_src, train_trg = load_data(
                data_path + "train.vi.txt",
                data_path + "train.en.txt"
            )
            print(f"  • Training samples: {len(train_src):,}")
            print("✓ Local dataset loaded successfully!")
        except Exception as e:
            print(f"✗ Error loading local dataset: {e}")
    else:
        print("✗ Local dataset files not found")

print("\n" + "="*70)
print(" How to switch datasets:")
print("="*70)
print("\nIn config.py, change the USE_DATASET setting:")
print("  • For Hugging Face: USE_DATASET = 'huggingface'")
print("  • For local files:  USE_DATASET = 'local'")
print("\nThen run train.py or infer.py as usual.")
print("="*70)
