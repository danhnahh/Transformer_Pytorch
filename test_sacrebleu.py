#!/usr/bin/env python3
"""
SacreBLEU evaluation script for Vietnamese-English Neural Machine Translation.

Now also computes:
- TER
- METEOR
- chrF (SacreBLEU: CHRF)

Also:
- Outputs a log file (same info as console)
- Writes a JSON summary

Usage:
    python test_sacrebleu.py
    python test_sacrebleu.py --max-samples 100
    python test_sacrebleu.py --checkpoint model.pt
    python test_sacrebleu.py --beam-size 10
    python test_sacrebleu.py --log-file eval.log
"""

import argparse
import json
import logging
from datetime import datetime
from pathlib import Path
from evaluate import load as load_metric
import torch
from tqdm import tqdm
import numpy as np

from sacrebleu.metrics import BLEU, TER, CHRF
import nltk
from nltk import word_tokenize
nltk.download('punkt')
nltk.download('wordnet')

# METEOR availability depends on sacrebleu version
try:
    from sacrebleu.metrics import METEOR as SACRE_METEOR
except Exception:
    SACRE_METEOR = None

from dataset import preprocess_data, load_data_from_huggingface, load_data
from config import *
from models.transformer import Transformer
from infer import translate_sentence


def setup_logger(log_file: str | None):
    logger = logging.getLogger("eval")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s", "%Y-%m-%d %H:%M:%S")

    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    if log_file:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(log_file, encoding="utf-8")
        fh.setLevel(logging.INFO)
        fh.setFormatter(fmt)
        logger.addHandler(fh)

    return logger


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate translation model using SacreBLEU (+ TER/METEOR/chrF)")

    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to model checkpoint (default: checkpoints/best_transformer1.pt)",
    )

    parser.add_argument(
        "--dataset",
        type=str,
        choices=["huggingface", "local"],
        default=None,
        help="Dataset to evaluate on (default: use USE_DATASET from config)",
    )

    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Maximum number of samples to evaluate (default: all)",
    )

    parser.add_argument(
        "--beam-size",
        type=int,
        default=5,
        help="Beam size for decoding (default: 5)",
    )

    parser.add_argument(
        "--show-examples",
        type=int,
        default=5,
        help="Number of translation examples to show (default: 5, set to 0 to disable)",
    )

    parser.add_argument(
        "--output-file",
        type=str,
        default=None,
        help="Save translations to file (one per line)",
    )

    parser.add_argument(
        "--lowercase",
        action="store_true",
        help="Compute metrics with lowercase=True (where supported)",
    )

    parser.add_argument(
        "--log-file",
        type=str,
        default=None,
        help="Write console output to this log file (default: auto name in ./logs/)",
    )

    parser.add_argument(
        "--json-summary",
        type=str,
        default=None,
        help="Write a JSON summary to this path (default: auto name next to log file)",
    )

    return parser.parse_args()


def load_model_and_data(checkpoint_path, eval_dataset, logger):
    logger.info("=" * 70)
    logger.info("Loading Model and Data")
    logger.info("=" * 70)

    if eval_dataset == "huggingface":
        logger.info(f"Using Hugging Face dataset: {HF_DATASET_NAME}")
        en_tokenizer, vi_tokenizer, _, _ = preprocess_data(
            vocab_size=VOCAB_SIZE,
            use_huggingface=True,
            hf_dataset_name=HF_DATASET_NAME,
        )
        _, _, test_dataset = load_data_from_huggingface(HF_DATASET_NAME)
        test_src = [example["vi"] for example in test_dataset]
        test_trg = [example["en"] for example in test_dataset]
    else:
        logger.info(f"Using local dataset: {data_path}")
        en_tokenizer, vi_tokenizer, _, _ = preprocess_data(
            train_src_path=train_data_path + "train.vi.txt",
            train_trg_path=train_data_path + "train.en.txt",
            val_src_path=data_path + "tst2013.vi.txt",
            val_trg_path=data_path + "tst2013.en.txt",
            vocab_size=VOCAB_SIZE,
            use_huggingface=False,
        )
        test_src, test_trg = load_data(
            test_data_path + "tst2012.vi.txt",
            test_data_path + "tst2012.en.txt",
        )

    en_vocab_size = en_tokenizer.get_vocab_size()
    vi_vocab_size = vi_tokenizer.get_vocab_size()
    pad_token_id = vi_tokenizer.pad_id

    logger.info("✓ Data loaded")
    logger.info(f"  Test samples: {len(test_src):,}")
    logger.info(f"  EN vocab: {en_vocab_size}")
    logger.info(f"  VI vocab: {vi_vocab_size}")

    logger.info("Loading model...")
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
        use_alignment=True,
    ).to(DEVICE)

    try:
        checkpoint = torch.load(checkpoint_path, map_location=DEVICE)
        model.load_state_dict(checkpoint["model_state_dict"])
        logger.info(f"✓ Model loaded from: {checkpoint_path}")
        if "val_loss" in checkpoint:
            logger.info(f"  Validation loss: {checkpoint['val_loss']:.4f}")
        if "val_accuracy" in checkpoint:
            logger.info(f"  Validation accuracy: {checkpoint['val_accuracy']:.4f}")
    except FileNotFoundError:
        logger.error(f"✗ Checkpoint not found: {checkpoint_path}")
        raise SystemExit(1)

    model.eval()
    return model, vi_tokenizer, en_tokenizer, test_src, test_trg


def _safe_signature(metric, score_obj):
    if hasattr(metric, "get_signature"):
        return metric.get_signature()
    if hasattr(score_obj, "signature"):
        return score_obj.signature
    if hasattr(score_obj, "format"):
        return score_obj.format()
    return "N/A"


def _compute_meteor_fallback_nltk(hypotheses, references, logger):
    """
    Fallback METEOR using NLTK.

    Newer NLTK versions expect pre-tokenized inputs (Iterable[str]).
    We'll do a lightweight tokenizer to avoid requiring punkt downloads.
    """
    try:
        from nltk.translate.meteor_score import meteor_score  # type: ignore
    except Exception:
        logger.warning("METEOR: sacrebleu METEOR unavailable and nltk not installed. Skipping METEOR.")
        return None, None

    import re

    def simple_tokenize(s: str):
        # Words + numbers + keep basic punctuation as separate tokens
        # (Good enough for a robust fallback; not perfect.)
        return re.findall(r"\w+|[^\w\s]", s, flags=re.UNICODE)

    sent_scores = []
    for hyp, ref in zip(hypotheses, references):
        hyp = hyp.strip()
        ref = ref.strip()

        # Try both styles to be compatible with multiple NLTK versions.
        s = None
        try:
            # Some older NLTK versions accept raw strings
            s = meteor_score([word_tokenize(ref)], word_tokenize(hyp))
        except Exception as e:
            logger.warning(f"METEOR (NLTK) failed on a sentence: {e}. Using 0 for that sentence.")
            s = 0.0

        sent_scores.append(float(s))

    corpus = float(np.mean(sent_scores)) if sent_scores else 0.0
    return corpus, sent_scores



def evaluate_metrics(
    model,
    vi_tokenizer,
    en_tokenizer,
    test_src,
    test_trg,
    logger,
    beam_size=5,
    max_samples=None,
    show_examples=5,
    output_file=None,
    lowercase=False,
):
    if max_samples is not None:
        test_src = test_src[:max_samples]
        test_trg = test_trg[:max_samples]

    logger.info("=" * 70)
    logger.info("Running Evaluation (BLEU + TER + METEOR + chrF)")
    logger.info("=" * 70)
    logger.info(f"Test samples: {len(test_src):,}")
    logger.info(f"Beam size: {beam_size}")
    logger.info(f"Device: {DEVICE}")
    logger.info(f"Lowercase: {lowercase}")

    hypotheses = []
    references = []

    logger.info("Generating translations...")
    for i, (src_text, ref_text) in enumerate(tqdm(list(zip(test_src, test_trg)), total=len(test_src))):
        src_text = src_text.strip()
        ref_text = ref_text.strip()

        translation = translate_sentence(
            model,
            src_text,
            vi_tokenizer,
            en_tokenizer,
            DEVICE,
            max_len=MAX_SEQ_LEN,
            beam_size=beam_size,
        ).strip()

        hypotheses.append(translation)
        references.append(ref_text)

        if show_examples > 0 and i < show_examples:
            if i == 0:
                logger.info("=" * 70)
                logger.info("Translation Examples")
                logger.info("=" * 70)
            logger.info(f"Example {i+1}:")
            logger.info(f"  Source:      {src_text[:100]}{'...' if len(src_text) > 100 else ''}")
            logger.info(f"  Reference:   {ref_text[:100]}{'...' if len(ref_text) > 100 else ''}")
            logger.info(f"  Translation: {translation[:100]}{'...' if len(translation) > 100 else ''}")

    if output_file:
        with open(output_file, "w", encoding="utf-8") as f:
            for hyp in hypotheses:
                f.write(hyp + "\n")
        logger.info(f"✓ Translations saved to: {output_file}")

    # --- Metrics ---
    bleu = BLEU(lowercase=lowercase, effective_order=True)
    ter = TER()  # TER is typically case-sensitive by default; we don't force lower here
    chrf = CHRF(lowercase=lowercase)

    bleu_score = bleu.corpus_score(hypotheses, [references])
    ter_score = ter.corpus_score(hypotheses, [references])
    chrf_score = chrf.corpus_score(hypotheses, [references])

    # METEOR
    meteor_metric = load_metric("meteor")
    meteor_result = meteor_metric.compute(
        predictions=hypotheses,
        references=references,
    )

    # Sentence-level BLEU (already in your script)
    sentence_bleus = [bleu.sentence_score(h, [r]).score for h, r in zip(hypotheses, references)]
    sentence_ters = [ter.sentence_score(h, [r]).score for h, r in zip(hypotheses, references)]
    sentence_chrfs = [chrf.sentence_score(h, [r]).score for h, r in zip(hypotheses, references)]

    def stats(arr):
        arr = np.array(arr, dtype=float)
        return {
            "mean": float(np.mean(arr)),
            "median": float(np.median(arr)),
            "std": float(np.std(arr)),
            "min": float(np.min(arr)),
            "max": float(np.max(arr)),
        }

    logger.info("=" * 70)
    logger.info("Corpus Results")
    logger.info("=" * 70)
    logger.info(f"BLEU : {bleu_score.score:.2f}")
    logger.info(f"TER  : {ter_score.score:.2f}  (lower is better)")
    logger.info(f"METEOR: {(meteor_result['meteor'] * 100):.2f}")
    logger.info(f"chrF : {chrf_score.score:.2f}")

    logger.info("=" * 70)
    logger.info("BLEU details")
    logger.info("=" * 70)
    logger.info("Precision scores:")
    logger.info(f"  1-gram: {bleu_score.precisions[0]:.2f}")
    logger.info(f"  2-gram: {bleu_score.precisions[1]:.2f}")
    logger.info(f"  3-gram: {bleu_score.precisions[2]:.2f}")
    logger.info(f"  4-gram: {bleu_score.precisions[3]:.2f}")
    logger.info(f"Brevity penalty: {bleu_score.bp:.4f}")
    logger.info(
        f"Length ratio: {bleu_score.sys_len}/{bleu_score.ref_len} "
        f"= {bleu_score.sys_len/bleu_score.ref_len:.4f}"
    )

    logger.info("=" * 70)
    logger.info("Sentence-level statistics")
    logger.info("=" * 70)
    logger.info(f"BLEU stats: {stats(sentence_bleus)}")
    logger.info(f"TER  stats: {stats(sentence_ters)}")
    logger.info(f"chrF stats: {stats(sentence_chrfs)}")

    logger.info("=" * 70)
    logger.info("Metric signatures")
    logger.info("=" * 70)
    logger.info(f"BLEU signature : {_safe_signature(bleu, bleu_score)}")
    logger.info(f"TER signature  : {_safe_signature(ter, ter_score)}")
    logger.info(f"chrF signature : {_safe_signature(chrf, chrf_score)}")

    logger.info("=" * 70)

    return {
        "bleu": float(bleu_score.score),
        "ter": float(ter_score.score),
        "chrf": float(chrf_score.score),
        "meteor": meteor_result['meteor'],
        "bleu_object": bleu_score,
        "ter_object": ter_score,
        "chrf_object": chrf_score,
        "hypotheses": hypotheses,
        "references": references,
        "sentence_bleus": sentence_bleus,
        "sentence_ters": sentence_ters,
        "sentence_chrfs": sentence_chrfs,
        "sentence_meteors": meteor_result.get('individual_scores', None),
    }


def main():
    args = parse_args()

    # Auto log paths
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if args.log_file is None:
        args.log_file = str(Path("logs") / f"eval_{stamp}.log")
    if args.json_summary is None:
        args.json_summary = str(Path(args.log_file).with_suffix(".json"))

    logger = setup_logger(args.log_file)

    checkpoint_path = args.checkpoint if args.checkpoint else f"{saved_model_path}best_transformer1.pt"
    eval_dataset = args.dataset if args.dataset else USE_DATASET

    logger.info("=" * 70)
    logger.info("Evaluation Configuration")
    logger.info("=" * 70)
    logger.info(f"Checkpoint: {checkpoint_path}")
    logger.info(f"Dataset: {eval_dataset}")
    logger.info(f"Max samples: {args.max_samples if args.max_samples else 'All'}")
    logger.info(f"Beam size: {args.beam_size}")
    logger.info(f"Show examples: {args.show_examples}")
    logger.info(f"Lowercase: {args.lowercase}")
    logger.info(f"Output file: {args.output_file if args.output_file else 'None'}")
    logger.info(f"Log file: {args.log_file}")
    logger.info(f"JSON summary: {args.json_summary}")

    model, vi_tokenizer, en_tokenizer, test_src, test_trg = load_model_and_data(
        checkpoint_path, eval_dataset, logger
    )

    results = evaluate_metrics(
        model,
        vi_tokenizer,
        en_tokenizer,
        test_src,
        test_trg,
        logger,
        beam_size=args.beam_size,
        max_samples=args.max_samples,
        show_examples=args.show_examples,
        output_file=args.output_file,
        lowercase=args.lowercase,
    )

    # Write JSON summary (no huge arrays by default)
    summary = {
        "timestamp": stamp,
        "checkpoint": checkpoint_path,
        "dataset": eval_dataset,
        "max_samples": args.max_samples,
        "beam_size": args.beam_size,
        "lowercase": args.lowercase,
        "metrics": {
            "bleu": results["bleu"],
            "ter": results["ter"],
            "meteor": results["meteor"] * 100,
            "chrf": results["chrf"],
        },
    }
    Path(args.json_summary).parent.mkdir(parents=True, exist_ok=True)
    with open(args.json_summary, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    logger.info("=" * 70)
    logger.info("Evaluation Complete!")
    logger.info("=" * 70)
    logger.info(
        f"Final scores | BLEU: {results['bleu']:.2f} | TER: {results['ter']:.2f} | "
        f"METEOR: {('N/A' if results['meteor'] is None else f'{(results['meteor'] * 100):.2f}')} | "
        f"chrF: {results['chrf']:.2f}"
    )
    logger.info("=" * 70)


if __name__ == "__main__":
    main()

