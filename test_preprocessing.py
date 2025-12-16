"""Test the full preprocessing pipeline with the new dataset."""

from dataset import preprocess_data
from config import VOCAB_SIZE, MAX_SEQ_LEN

print("Testing full preprocessing pipeline with Hugging Face dataset...")
print("="*60)
print(f"Vocab size: {VOCAB_SIZE}")
print(f"Max sequence length: {MAX_SEQ_LEN}")
print("="*60)

# Use a small subset for testing (we can add a parameter to limit data)
# For this test, we'll just verify the full pipeline works
en_tokenizer, vi_tokenizer, train_sequences, val_sequences = preprocess_data(
    vocab_size=VOCAB_SIZE,
    use_huggingface=True,
    hf_dataset_name='ncduy/mt-en-vi',
    force_train_tokenizer=False  # Use existing if available
)

print(f"\n{'='*60}")
print("Preprocessing completed successfully!")
print(f"{'='*60}")
print(f"English vocab size: {en_tokenizer.get_vocab_size()}")
print(f"Vietnamese vocab size: {vi_tokenizer.get_vocab_size()}")
print(f"Training sequences: {len(train_sequences):,}")
print(f"Validation sequences: {len(val_sequences):,}")

# Show a sample
if len(train_sequences) > 0:
    src_sample, trg_sample = train_sequences[0]
    print(f"\nSample sequence (tokenized):")
    print(f"Source (first 20 tokens): {src_sample[:20].tolist()}")
    print(f"Target (first 20 tokens): {trg_sample[:20].tolist()}")
    
    # Decode to verify
    src_decoded = vi_tokenizer.decode(src_sample.tolist())
    trg_decoded = en_tokenizer.decode(trg_sample.tolist())
    print(f"\nDecoded:")
    print(f"Vietnamese: {src_decoded}")
    print(f"English: {trg_decoded}")

print(f"\n{'='*60}")
print("✓ Full preprocessing test completed!")
print(f"{'='*60}")
