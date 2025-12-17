#!/usr/bin/env python3
"""
SacreBLEU evaluation script for Vietnamese-English Neural Machine Translation.

Usage:
    python test_sacrebleu.py
    python test_sacrebleu.py --max-samples 100
    python test_sacrebleu.py --checkpoint model.pt
    python test_sacrebleu.py --beam-size 10
"""

import argparse
import torch
from tqdm import tqdm
from sacrebleu.metrics import BLEU
import numpy as np

from dataset import preprocess_data, load_data_from_huggingface, load_data
from config import *
from models.transformer import Transformer
from infer import translate_sentence


def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate translation model using SacreBLEU')

    parser.add_argument(
        '--checkpoint',
        type=str,
        default=None,
        help='Path to model checkpoint (default: checkpoints/best_transformer1.pt)'
    )

    parser.add_argument(
        '--dataset',
        type=str,
        choices=['huggingface', 'local'],
        default=None,
        help='Dataset to evaluate on (default: use USE_DATASET from config)'
    )

    parser.add_argument(
        '--max-samples',
        type=int,
        default=None,
        help='Maximum number of samples to evaluate (default: all)'
    )

    parser.add_argument(
        '--beam-size',
        type=int,
        default=5,
        help='Beam size for decoding (default: 5)'
    )

    parser.add_argument(
        '--show-examples',
        type=int,
        default=5,
        help='Number of translation examples to show (default: 5, set to 0 to disable)'
    )

    parser.add_argument(
        '--output-file',
        type=str,
        default=None,
        help='Save translations to file (one per line)'
    )

    # Optional: if you want lowercase BLEU (off by default to match SacreBLEU defaults)
    parser.add_argument(
        '--lowercase',
        action='store_true',
        help='Compute BLEU with lowercase=True'
    )

    return parser.parse_args()


def load_model_and_data(checkpoint_path, eval_dataset):
    print(f"\n{'='*70}")
    print("Loading Model and Data")
    print(f"{'='*70}")

    if eval_dataset == 'huggingface':
        print(f"Using Hugging Face dataset: {HF_DATASET_NAME}")
        en_tokenizer, vi_tokenizer, _, _ = preprocess_data(
            vocab_size=VOCAB_SIZE,
            use_huggingface=True,
            hf_dataset_name=HF_DATASET_NAME
        )
        _, _, test_dataset = load_data_from_huggingface(HF_DATASET_NAME)
        test_src = [example['vi'] for example in test_dataset]
        test_trg = [example['en'] for example in test_dataset]
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
        test_src, test_trg = load_data(
            test_data_path + "tst2012.vi.txt",
            test_data_path + "tst2012.en.txt"
        )

    en_vocab_size = en_tokenizer.get_vocab_size()
    vi_vocab_size = vi_tokenizer.get_vocab_size()
    pad_token_id = vi_tokenizer.pad_id

    print(f"✓ Data loaded")
    print(f"  Test samples: {len(test_src):,}")
    print(f"  EN vocab: {en_vocab_size}")
    print(f"  VI vocab: {vi_vocab_size}")

    print("\nLoading model...")
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
        checkpoint = torch.load(checkpoint_path, map_location=DEVICE)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"✓ Model loaded from: {checkpoint_path}")
        if 'val_loss' in checkpoint:
            print(f"  Validation loss: {checkpoint['val_loss']:.4f}")
        if 'val_accuracy' in checkpoint:
            print(f"  Validation accuracy: {checkpoint['val_accuracy']:.4f}")
    except FileNotFoundError:
        print(f"✗ Checkpoint not found: {checkpoint_path}")
        print("  Please train the model first or specify correct checkpoint path")
        raise SystemExit(1)

    model.eval()
    return model, vi_tokenizer, en_tokenizer, test_src, test_trg


def evaluate_sacrebleu(
    model,
    vi_tokenizer,
    en_tokenizer,
    test_src,
    test_trg,
    beam_size=5,
    max_samples=None,
    show_examples=5,
    output_file=None,
    lowercase=False,
):
    if max_samples is not None:
        test_src = test_src[:max_samples]
        test_trg = test_trg[:max_samples]

    print(f"\n{'='*70}")
    print("Running SacreBLEU Evaluation")
    print(f"{'='*70}")
    print(f"Test samples: {len(test_src):,}")
    print(f"Beam size: {beam_size}")
    print(f"Device: {DEVICE}")
    print(f"Lowercase: {lowercase}")

    hypotheses = []
    references = []

    print("\nGenerating translations...")
    for i, (src_text, ref_text) in enumerate(tqdm(list(zip(test_src, test_trg)), total=len(test_src))):
        src_text = src_text.strip()
        ref_text = ref_text.strip()

        translation = translate_sentence(
            model, src_text, vi_tokenizer, en_tokenizer,
            DEVICE, max_len=MAX_SEQ_LEN, beam_size=beam_size
        ).strip()

        hypotheses.append(translation)
        references.append(ref_text)

        if show_examples > 0 and i < show_examples:
            if i == 0:
                print(f"\n{'='*70}")
                print("Translation Examples")
                print(f"{'='*70}")
            print(f"\nExample {i+1}:")
            print(f"  Source:      {src_text[:100]}{'...' if len(src_text) > 100 else ''}")
            print(f"  Reference:   {ref_text[:100]}{'...' if len(ref_text) > 100 else ''}")
            print(f"  Translation: {translation[:100]}{'...' if len(translation) > 100 else ''}")

    if output_file:
        with open(output_file, 'w', encoding='utf-8') as f:
            for hyp in hypotheses:
                f.write(hyp + '\n')
        print(f"\n✓ Translations saved to: {output_file}")

    print(f"\n{'='*70}")
    print("SacreBLEU Results")
    print(f"{'='*70}")

    bleu = BLEU(lowercase=lowercase, effective_order=True)

    # ✅ CORRECT: refs must be a list of reference streams: [references]
    bleu_score = bleu.corpus_score(hypotheses, [references])

    print(f"\nBLEU Score: {bleu_score.score:.2f}")
    print(f"  Precision scores:")
    print(f"    1-gram: {bleu_score.precisions[0]:.2f}")
    print(f"    2-gram: {bleu_score.precisions[1]:.2f}")
    print(f"    3-gram: {bleu_score.precisions[2]:.2f}")
    print(f"    4-gram: {bleu_score.precisions[3]:.2f}")
    print(f"  Brevity penalty: {bleu_score.bp:.4f}")
    print(
        f"  Length ratio: {bleu_score.sys_len}/{bleu_score.ref_len} "
        f"= {bleu_score.sys_len/bleu_score.ref_len:.4f}"
    )

    sentence_bleus = []
    for hyp, ref in zip(hypotheses, references):
        sent_score = bleu.sentence_score(hyp, [ref])
        sentence_bleus.append(sent_score.score)

    print(f"\nSentence-level BLEU statistics:")
    print(f"  Mean: {np.mean(sentence_bleus):.2f}")
    print(f"  Median: {np.median(sentence_bleus):.2f}")
    print(f"  Std: {np.std(sentence_bleus):.2f}")
    print(f"  Min: {np.min(sentence_bleus):.2f}")
    print(f"  Max: {np.max(sentence_bleus):.2f}")

    print(f"\nSacreBLEU signature:")
    # Prefer get_signature() if available; fall back to score.signature if present
    if hasattr(bleu, "get_signature"):
        print(f"  {bleu.get_signature()}")
    elif hasattr(bleu_score, "signature"):
        print(f"  {bleu_score.signature}")
    else:
        # Last resort: at least print formatted score line
        print(f"  {bleu_score.format()}")

    print(f"\n{'='*70}")

    return {
        'bleu': bleu_score.score,
        'bleu_object': bleu_score,
        'hypotheses': hypotheses,
        'references': references,
        'sentence_bleus': sentence_bleus
    }


def main():
    args = parse_args()

    checkpoint_path = args.checkpoint if args.checkpoint else f"{saved_model_path}best_transformer1.pt"
    eval_dataset = args.dataset if args.dataset else USE_DATASET

    print(f"\n{'='*70}")
    print("SacreBLEU Evaluation Configuration")
    print(f"{'='*70}")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Dataset: {eval_dataset}")
    print(f"Max samples: {args.max_samples if args.max_samples else 'All'}")
    print(f"Beam size: {args.beam_size}")
    print(f"Show examples: {args.show_examples}")
    print(f"Lowercase: {args.lowercase}")
    if args.output_file:
        print(f"Output file: {args.output_file}")

    model, vi_tokenizer, en_tokenizer, test_src, test_trg = load_model_and_data(
        checkpoint_path, eval_dataset
    )

    results = evaluate_sacrebleu(
        model, vi_tokenizer, en_tokenizer, test_src, test_trg,
        beam_size=args.beam_size,
        max_samples=args.max_samples,
        show_examples=args.show_examples,
        output_file=args.output_file,
        lowercase=args.lowercase,
    )

    print(f"\n{'='*70}")
    print("Evaluation Complete!")
    print(f"{'='*70}")
    print(f"Final BLEU Score: {results['bleu']:.2f}")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
