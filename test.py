#!/usr/bin/env python3
"""
Quick test script to translate a few Vietnamese sentences to English.
Shows how the trained model works on simple examples.
"""

import torch
from dataset import preprocess_data, load_data_from_huggingface, load_data, normalize_text
from config import *
from models.transformer import Transformer
from infer import translate_sentence

def test_translation(num_samples=5):
    """Test translation on a small number of samples."""
    
    print("="*70)
    print(" Loading Model and Tokenizers")
    print("="*70)
    
    # Load tokenizers (reuse existing ones if available)
    if USE_DATASET == 'huggingface':
        print(f"Using Hugging Face dataset: {HF_DATASET_NAME}")
        en_tokenizer, vi_tokenizer, _, _ = preprocess_data(
            vocab_size=VOCAB_SIZE,
            use_huggingface=True,
            hf_dataset_name=HF_DATASET_NAME
        )
        # Get test samples
        _, _, test_dataset = load_data_from_huggingface(HF_DATASET_NAME)
        test_samples = [(example['vi'], example['en']) for example in test_dataset.select(range(num_samples))]
    else:
        print(f"Using local dataset: {data_path}")
        en_tokenizer, vi_tokenizer, _, _ = preprocess_data(
            train_src_path=train_data_path + "train.vi.txt",
            train_trg_path=train_data_path + "train.en.txt",
            val_src_path=data_path + "tst2013.vi.txt",
            val_trg_path=data_path + "tst2013.en.txt",
            vocab_size=VOCAB_SIZE,
            use_huggingface=False
        )
        # Get test samples
        test_src, test_trg = load_data(
            test_data_path + "tst2012.vi.txt",
            test_data_path + "tst2012.en.txt"
        )
        test_samples = list(zip(test_src[:num_samples], test_trg[:num_samples]))
    
    en_vocab_size = en_tokenizer.get_vocab_size()
    vi_vocab_size = vi_tokenizer.get_vocab_size()
    pad_token_id = vi_tokenizer.pad_id
    
    print(f"✓ Tokenizers loaded")
    print(f"  English vocab: {en_vocab_size}")
    print(f"  Vietnamese vocab: {vi_vocab_size}")
    
    # Load model
    print("\nLoading trained model...")
    model = Transformer(
        src_pad_idx=pad_token_id,
        trg_pad_idx=pad_token_id,
        d_model=D_MODEL,
        inp_vocab_size=vi_vocab_size,
        trg_vocab_size=en_vocab_size,
        max_len=MAX_SEQ_LEN,
        d_ff=D_FF,
        num_heads=NUM_HEADS,
        num_layers=NUM_LAYERS,
        dropout=DROPOUT,
        device=DEVICE,
        use_alignment=True
    ).to(DEVICE)
    
    try:
        checkpoint = torch.load(saved_model_path + 'best_transformer1.pt', map_location=DEVICE)
        model.load_state_dict(checkpoint['model_state_dict'])
        val_loss = checkpoint.get('val_loss', 'N/A')
        val_acc = checkpoint.get('val_accuracy', 'N/A')
        print(f"✓ Model loaded successfully")
        print(f"  Validation Loss: {val_loss}")
        print(f"  Validation Accuracy: {val_acc}")
    except FileNotFoundError:
        print("✗ Model checkpoint not found!")
        print(f"  Expected location: {saved_model_path}best_transformer1.pt")
        print("  Please train the model first using: python train.py")
        return
    
    model.eval()
    
    # Test translations
    print("\n" + "="*70)
    print(" Translation Results")
    print("="*70)
    
    for i, (vi_text, en_reference) in enumerate(test_samples, 1):
        print(f"\n{'─'*70}")
        print(f"Sample {i}:")
        print(f"{'─'*70}")
        print(f"Vietnamese:  {vi_text}")
        print(f"Reference:   {en_reference}")
        
        # Translate with beam search
        translation = translate_sentence(
            model, vi_text, vi_tokenizer, en_tokenizer,
            DEVICE, max_len=MAX_SEQ_LEN, beam_size=BEAM_SIZE
        )
        
        print(f"Translation: {translation}")
        
        # Simple quality indicator (very rough)
        ref_words = set(en_reference.lower().split())
        trans_words = set(translation.lower().split())
        if ref_words and trans_words:
            overlap = len(ref_words & trans_words) / len(ref_words)
            quality = "Good" if overlap > 0.5 else "Fair" if overlap > 0.3 else "Needs improvement"
            print(f"Word overlap: {overlap:.1%} ({quality})")
    
    print("\n" + "="*70)
    print(" Test Custom Sentences")
    print("="*70)
    print("\nYou can also test custom Vietnamese sentences:")
    print("(Press Ctrl+C to exit)\n")
    
    try:
        while True:
            vi_input = input("Vietnamese: ").strip()
            if not vi_input:
                break
            
            translation = translate_sentence(
                model, vi_input, vi_tokenizer, en_tokenizer,
                DEVICE, max_len=MAX_SEQ_LEN, beam_size=BEAM_SIZE
            )
            print(f"English:    {translation}\n")
    except (KeyboardInterrupt, EOFError):
        print("\n\n✓ Testing completed!")
    
    print("="*70)

if __name__ == "__main__":
    import sys
    
    # Allow specifying number of samples from command line
    num_samples = 5
    if len(sys.argv) > 1:
        try:
            num_samples = int(sys.argv[1])
        except ValueError:
            print(f"Invalid number: {sys.argv[1]}, using default: 5")
    
    print(f"\nTesting with {num_samples} sample(s)\n")
    test_translation(num_samples)
