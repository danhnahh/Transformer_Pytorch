import torch
import torch.nn.functional as F
import numpy as np
from models.transformer import Transformer
from nltk.translate.bleu_score import corpus_bleu, sentence_bleu, SmoothingFunction
from tqdm import tqdm
from config import *
from dataset import preprocess_data, load_data, load_data_from_huggingface, normalize_text, SentencePieceTokenizer

def beam_search_decode(model, src, src_mask, trg_tokenizer, max_len=60,
                       beam_size=5, device='cuda'):
    model.eval()

    enc_src = model.encoder(src, src_mask)

    # Use SentencePiece token IDs
    start_token = trg_tokenizer.bos_id
    end_token = trg_tokenizer.eos_id

    beams = [(0.0, [start_token])]  # (score, [tokens])
    completed_beams = []

    for step in range(max_len):
        candidates = []

        for score, seq in beams:
            if seq[-1] == end_token:
                completed_beams.append((score, seq))
                continue

            trg_tensor = torch.tensor(seq, dtype=torch.long).unsqueeze(0).to(device)
            trg_mask = model.make_trg_mask(trg_tensor)

            with torch.no_grad():
                output = model.decoder(trg_tensor, enc_src, trg_mask, src_mask)
            
            logits = output[:, -1, :]
            log_probs = F.log_softmax(logits, dim=-1)

            topk_log_probs, topk_indices = torch.topk(log_probs, beam_size)

            for i in range(beam_size):
                token = topk_indices[0][i].item()
                token_log_prob = topk_log_probs[0][i].item()
                new_score = score + token_log_prob
                new_seq = seq + [token]
                candidates.append((new_score, new_seq))

        candidates = sorted(candidates, key=lambda x: x[0], reverse=True)
        beams = candidates[:beam_size]

        if len(completed_beams) >= beam_size:
            break
    
    all_beams = completed_beams + beams

    normalized_beams = []
    for score, seq in all_beams:
        normalized_score = score / len(seq)
        normalized_beams.append((normalized_score, seq))

    best_beam = sorted(normalized_beams, key=lambda x: x[0], reverse=True)[0]

    return best_beam[1]

def translate_sentence(model, sentence, src_tokenizer, trg_tokenizer, 
                      device='cuda', max_len=60, beam_size=5):
    """
    Translate a sentence using the model with SentencePiece tokenization.
    
    Args:
        model: Transformer model
        sentence: Input sentence (Vietnamese)
        src_tokenizer: Source SentencePiece tokenizer
        trg_tokenizer: Target SentencePiece tokenizer
        device: Device to use
        max_len: Maximum sequence length
        beam_size: Beam search width
    
    Returns:
        Translated sentence (English)
    """
    model.eval()

    # Normalize and encode with SentencePiece
    sentence_norm = normalize_text(sentence, lang='vi')
    src_indexes = src_tokenizer.encode(sentence_norm, add_bos=True, add_eos=True)

    # Pad or truncate
    if len(src_indexes) < max_len:
        src_indexes = src_indexes + [src_tokenizer.pad_id] * (max_len - len(src_indexes))
    else:
        src_indexes = src_indexes[:max_len]

    src_tensor = torch.LongTensor(src_indexes).unsqueeze(0).to(device)
    src_mask = model.make_src_mask(src_tensor)

    trg_indexes = beam_search_decode(model, src_tensor, src_mask, trg_tokenizer, 
                                     max_len, beam_size, device)
    
    # Decode with SentencePiece (skip special tokens)
    translation = trg_tokenizer.decode(trg_indexes, skip_special_tokens=True)
    
    return translation

def calculate_bleu_score(model, test_data, src_tokenizer, trg_tokenizer, device='cuda',
                         beam_size=5, max_samples=None):
    
    model.eval()

    references = []
    hypotheses = []
    examples = []

    if max_samples:
        test_data = test_data[:max_samples]
    
    print(f"\n{'='*60}")
    print(f"EVALUATING ON {len(test_data)} SAMPLES")
    print(f"Beam Search with beam_size={beam_size}")
    print(f"{'='*60}\n")

    for src_sentence, trg_sentence in tqdm(test_data, desc="Translating"):
        translation = translate_sentence(
            model, src_sentence, src_tokenizer, trg_tokenizer,
            device, beam_size=beam_size
        )

        ref_tokens = trg_sentence.lower().split()
        hyp_tokens = translation.lower().split()
        references.append([ref_tokens])
        hypotheses.append(hyp_tokens)

        if len(examples) < 5:
            examples.append(
                {
                'source': src_sentence,
                'reference': trg_sentence,
                'hypothesis': translation
                }
            )
    smoothing = SmoothingFunction()
    bleu_score = corpus_bleu(
        references, hypotheses,
        smoothing_function=smoothing.method1
    ) * 100

    return bleu_score, examples

def run(beam_size=5, max_samples=None):
    # Load dataset based on config setting
    if USE_DATASET == 'huggingface':
        print(f"Using Hugging Face dataset: {HF_DATASET_NAME}")
        en_tokenizer, vi_tokenizer, all_train_sequences, all_val_sequences = preprocess_data(
            vocab_size=VOCAB_SIZE,
            use_huggingface=True,
            hf_dataset_name=HF_DATASET_NAME
        )
        # Load test data from HF
        _, _, test_dataset = load_data_from_huggingface(HF_DATASET_NAME)
        test_src = [example['vi'] for example in test_dataset]
        test_trg = [example['en'] for example in test_dataset]
        test_data = list(zip(test_src, test_trg))
    else:
        print(f"Using local dataset: {test_data_path}")
        en_tokenizer, vi_tokenizer, all_train_sequences, all_val_sequences = preprocess_data(
            train_src_path=train_data_path + "train.vi.txt",
            train_trg_path=train_data_path + "train.en.txt",
            val_src_path=data_path + "tst2013.vi.txt",
            val_trg_path=data_path + "tst2013.en.txt",
            vocab_size=VOCAB_SIZE,
            use_huggingface=False
        )
        # Load test data from local files
        test_src, test_trg = load_data(
            test_data_path + "tst2012.vi.txt",
            test_data_path + "tst2012.en.txt"
        )
        test_data = list(zip(test_src, test_trg))

    en_vocab_size = en_tokenizer.get_vocab_size()
    vi_vocab_size = vi_tokenizer.get_vocab_size()
    pad_token_id = vi_tokenizer.pad_id

    print("Loading model...")
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
        use_alignment=True  # Transformer-Align: combines dot-product and additive attention
    ).to(DEVICE)

    checkpoint = torch.load(saved_model_path + 'best_transformer1.pt')
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"✓ Model loaded (Val Loss: {checkpoint['val_loss']:.4f})")

    bleu_score, examples = calculate_bleu_score(
        model, test_data, vi_tokenizer, en_tokenizer,
        DEVICE, beam_size=beam_size, max_samples=max_samples
    )
    # Print results
    print(f"\n{'='*60}")
    print("FINAL RESULTS")
    print(f"{'='*60}")
    print(f"Beam Search (k={beam_size}): {bleu_score:.2f}")
    print(f"{'='*60}\n")
    
if __name__ == "__main__":
    run(beam_size=BEAM_SIZE, max_samples=None)