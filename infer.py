import torch
import torch.nn.functional as F
import numpy as np
from models.transformer import Transformer
from pyvi.ViTokenizer import ViTokenizer
import numpy as np
from nltk.translate.bleu_score import corpus_bleu, sentence_bleu, SmoothingFunction
from tqdm import tqdm
from config import *
from dataset import *

def beam_search_decode(model, src, src_mask, trg_tokenizer, max_len=60,
                       beam_size=5, device='cuda'):
    model.eval()

    enc_src = model.encoder(src, src_mask)

    start_token = trg_tokenizer.word_index[START_TOKEN]
    end_token = trg_tokenizer.word_index[END_TOKEN]

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
                      device='cuda',max_len=60, beam_size=5):
    model.eval()

    sentence_tokenized = ViTokenizer.tokenize(sentence)
    sentences_with_tokens = f"{START_TOKEN} {sentence_tokenized} {END_TOKEN}"

    src_indexes = src_tokenizer.texts_to_sequences([sentences_with_tokens])[0]

    if len (src_indexes) < max_len:
        src_indexes = src_indexes + [PAD_TOKEN_POS] * (max_len - len(src_indexes))
    else:
        src_indexes = src_indexes[:max_len]

    src_tensor = torch.LongTensor(src_indexes).unsqueeze(0).to(device)
    src_mask = model.make_src_mask(src_tensor)

    trg_indexes = beam_search_decode(model, src_tensor, src_mask, trg_tokenizer, 
                                     max_len, beam_size, device)
    
    trg_tokens = []
    index_to_word = {v: k for k, v in trg_tokenizer.word_index.items()}

    for idx in trg_indexes:
        if idx == trg_tokenizer.word_index.get(END_TOKEN):
            break
        if idx == trg_tokenizer.word_index.get(START_TOKEN):
            continue
        if idx == PAD_TOKEN_POS:
            continue

        word = index_to_word.get(idx, UNKNOWN_TOKEN)
        trg_tokens.append(word)
    
    return ' '.join(trg_tokens)

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
    en_tokenizer, vi_tokenizer, all_train_sequences, all_val_sequences = preprocess_data(
    train_data_path + "train.en.txt", 
    train_data_path + "train.vi.txt",
    data_path + "tst2013.en.txt", 
    data_path + "tst2013.vi.txt"
    )

    en_vocab_size = len(en_tokenizer)
    vi_vocab_size = len(vi_tokenizer) 

    print("Loading model...")
    model = Transformer(
        src_pad_idx=PAD_TOKEN_POS,
        trg_pad_idx=PAD_TOKEN_POS,
        d_model=D_MODEL,
        inp_vocab_size=vi_vocab_size,
        trg_vocab_size=en_vocab_size,
        max_len=MAX_SEQ_LEN,
        d_ff=D_FF,
        num_heads=NUM_HEADS,
        num_layers=NUM_LAYERS,
        dropout=DROPOUT,
        device=DEVICE
    ).to(DEVICE)

    checkpoint = torch.load(saved_model_path + 'best_transformer1.pt')
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"✓ Model loaded (Val Loss: {checkpoint['val_loss']:.4f})")

    test_src, test_trg = load_data(
        test_data_path + "tst2012.vi.txt",
        test_data_path + "tst2012.en.txt"
    )
    test_data = list(zip(test_src, test_trg))

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