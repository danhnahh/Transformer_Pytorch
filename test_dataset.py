"""Test script to verify the Hugging Face dataset loads correctly."""

from dataset import load_data_from_huggingface

print("Testing Hugging Face dataset loading...")
print("="*60)

# Load the dataset
train_dataset, val_dataset, test_dataset = load_data_from_huggingface('ncduy/mt-en-vi')

# Print dataset information
print(f"\nDataset splits loaded successfully!")
print(f"Train samples: {len(train_dataset):,}")
print(f"Validation samples: {len(val_dataset):,}")
print(f"Test samples: {len(test_dataset):,}")

# Show first few examples from training set
print(f"\n{'='*60}")
print("Sample translations from training set:")
print(f"{'='*60}")

for i in range(3):
    example = train_dataset[i]
    print(f"\nExample {i+1}:")
    print(f"Vietnamese: {example['vi']}")
    print(f"English: {example['en']}")
    print(f"Source: {example['source']}")

print(f"\n{'='*60}")
print("✓ Dataset test completed successfully!")
print(f"{'='*60}")
