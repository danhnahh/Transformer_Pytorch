import os
import re
import unicodedata
import sentencepiece as spm
from pyvi.ViTokenizer import ViTokenizer
from datasets import load_dataset
from config import *

# ============================================
# TEXT NORMALIZATION
# ============================================

def normalize_text(text, lang='en'):
    """
    Normalize text with unicode normalization, lowercasing, and cleaning.
    
    Args:
        text: Input text string
        lang: Language code ('en' or 'vi')
    
    Returns:
        Normalized text string
    """
    # Unicode normalization (NFC for Vietnamese, NFKC for English)
    if lang == 'vi':
        text = unicodedata.normalize('NFC', text)
    else:
        text = unicodedata.normalize('NFKC', text)
    
    # Lowercase
    text = text.lower()
    
    # Remove extra whitespace
    text = ' '.join(text.split())
    
    # Remove special characters but keep punctuation
    # Keep Vietnamese diacritics and common punctuation
    text = re.sub(r'[^\w\s\u00C0-\u024F\u1E00-\u1EFF.,!?;:\'\"-()]', ' ', text)
    
    # Normalize quotes and apostrophes
    text = text.replace('"', '"').replace('"', '"')
    text = text.replace(''', "'").replace(''', "'")
    
    # Remove multiple spaces
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text


def normalize_batch(texts, lang='en'):
    """Normalize a batch of texts."""
    return [normalize_text(t, lang) for t in texts]


# ============================================
# SENTENCEPIECE TOKENIZER
# ============================================

class SentencePieceTokenizer:
    """
    SentencePiece tokenizer wrapper with 32k vocabulary.
    """
    def __init__(self, model_path=None, vocab_size=32000, model_type='unigram'):
        self.vocab_size = vocab_size
        self.model_type = model_type
        self.sp = None
        self.model_path = model_path
        
        # Special tokens
        self.pad_token = '<pad>'
        self.unk_token = '<unk>'
        self.bos_token = '<s>'
        self.eos_token = '</s>'
        
        # Token IDs (SentencePiece reserves 0=<unk>, 1=<s>, 2=</s>)
        self.pad_id = 0  # We'll use 0 for padding
        self.unk_id = 0  # SentencePiece default
        self.bos_id = 1  # <s>
        self.eos_id = 2  # </s>
        
        if model_path and os.path.exists(model_path):
            self.load(model_path)
    
    def train(self, texts, model_prefix, lang='en'):
        """
        Train SentencePiece model on texts.
        
        Args:
            texts: List of text strings
            model_prefix: Prefix for model files (e.g., 'spm_en')
            lang: Language code for normalization
        """
        # Create temp file for training
        temp_file = f'{model_prefix}_train.txt'
        with open(temp_file, 'w', encoding='utf-8') as f:
            for text in texts:
                normalized = normalize_text(text, lang)
                if normalized.strip():
                    f.write(normalized + '\n')
        
        # Train SentencePiece
        spm.SentencePieceTrainer.train(
            input=temp_file,
            model_prefix=model_prefix,
            vocab_size=self.vocab_size,
            model_type=self.model_type,
            pad_id=3,  # Reserve 3 for <pad>
            unk_id=0,
            bos_id=1,
            eos_id=2,
            pad_piece=self.pad_token,
            unk_piece=self.unk_token,
            bos_piece=self.bos_token,
            eos_piece=self.eos_token,
            user_defined_symbols=[],
            character_coverage=0.9995 if lang == 'vi' else 1.0,
            normalization_rule_name='nfkc',
            num_threads=4,
            train_extremely_large_corpus=False,
        )
        
        # Clean up temp file
        os.remove(temp_file)
        
        # Load the trained model
        self.model_path = f'{model_prefix}.model'
        self.load(self.model_path)
        
        # Update pad_id after loading
        self.pad_id = 3
        
        print(f"Trained SentencePiece model: {self.model_path}")
        print(f"Vocabulary size: {self.sp.get_piece_size()}")
    
    def load(self, model_path):
        """Load a trained SentencePiece model."""
        self.sp = spm.SentencePieceProcessor()
        self.sp.load(model_path)
        self.model_path = model_path
        self.pad_id = 3  # Our custom pad_id
        self.vocab_size = self.sp.get_piece_size()
    
    def save(self, save_dir):
        """Save model to directory."""
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        # Copy model file
        import shutil
        if self.model_path:
            dest_path = os.path.join(save_dir, os.path.basename(self.model_path))
            if self.model_path != dest_path:
                shutil.copy(self.model_path, dest_path)
            return dest_path
        return None
    
    def encode(self, text, add_bos=False, add_eos=False):
        """
        Encode text to token IDs.
        
        Args:
            text: Input text string
            add_bos: Add beginning-of-sequence token
            add_eos: Add end-of-sequence token
        
        Returns:
            List of token IDs
        """
        if self.sp is None:
            raise ValueError("SentencePiece model not loaded!")
        
        ids = self.sp.encode(text, out_type=int)
        
        if add_bos:
            ids = [self.bos_id] + ids
        if add_eos:
            ids = ids + [self.eos_id]
        
        return ids
    
    def decode(self, ids, skip_special_tokens=True):
        """
        Decode token IDs to text.
        
        Args:
            ids: List of token IDs
            skip_special_tokens: Skip special tokens in output
        
        Returns:
            Decoded text string
        """
        if self.sp is None:
            raise ValueError("SentencePiece model not loaded!")
        
        if skip_special_tokens:
            ids = [i for i in ids if i not in [self.pad_id, self.bos_id, self.eos_id]]
        
        return self.sp.decode(ids)
    
    def encode_batch(self, texts, add_bos=False, add_eos=False):
        """Encode a batch of texts."""
        return [self.encode(t, add_bos, add_eos) for t in texts]
    
    def decode_batch(self, batch_ids, skip_special_tokens=True):
        """Decode a batch of token IDs."""
        return [self.decode(ids, skip_special_tokens) for ids in batch_ids]
    
    def get_vocab_size(self):
        """Get vocabulary size."""
        if self.sp:
            return self.sp.get_piece_size()
        return self.vocab_size
    
    @property
    def word_index(self):
        """
        Compatibility property for old code.
        Returns dict mapping token -> id.
        """
        if self.sp is None:
            return {}
        return {self.sp.id_to_piece(i): i for i in range(self.sp.get_piece_size())}


# ============================================
# DATA LOADING AND PREPROCESSING
# ============================================

def load_data_from_huggingface(dataset_name='ncduy/mt-en-vi'):
    """Load dataset from Hugging Face.
    
    Args:
        dataset_name: Name of the dataset on Hugging Face
    
    Returns:
        Tuple of (train_dataset, val_dataset, test_dataset)
    """
    print(f"Loading dataset from Hugging Face: {dataset_name}")
    dataset = load_dataset(dataset_name)
    return dataset['train'], dataset['validation'], dataset['test']


def load_data(src_file, trg_file):
    """Load source and target data from files."""
    with open(src_file, 'r', encoding='utf-8') as f:
        src_data = f.read().strip().split("\n")
    with open(trg_file, 'r', encoding='utf-8') as f:
        trg_data = f.read().strip().split("\n")
    return src_data, trg_data


def create_sequences(src_data, trg_data, src_tokenizer, trg_tokenizer, 
                     max_seq_len, src_lang='vi', trg_lang='en'):
    """
    Create padded sequences from source and target data.
    
    Args:
        src_data: List of source sentences
        trg_data: List of target sentences
        src_tokenizer: Source SentencePiece tokenizer
        trg_tokenizer: Target SentencePiece tokenizer
        max_seq_len: Maximum sequence length
        src_lang: Source language code
        trg_lang: Target language code
    
    Returns:
        Tuple of (src_sequences, trg_sequences) as torch tensors
    """
    filtered_src = []
    filtered_trg = []
    
    for src, trg in zip(src_data, trg_data):
        # Normalize texts
        src_norm = normalize_text(src, src_lang)
        trg_norm = normalize_text(trg, trg_lang)
        
        # Encode with BOS/EOS tokens
        src_ids = src_tokenizer.encode(src_norm, add_bos=True, add_eos=True)
        trg_ids = trg_tokenizer.encode(trg_norm, add_bos=True, add_eos=True)
        
        # Filter by length
        if len(src_ids) <= max_seq_len and len(trg_ids) <= max_seq_len:
            # Pad sequences
            src_padded = src_ids + [src_tokenizer.pad_id] * (max_seq_len - len(src_ids))
            trg_padded = trg_ids + [trg_tokenizer.pad_id] * (max_seq_len - len(trg_ids))
            
            filtered_src.append(src_padded[:max_seq_len])
            filtered_trg.append(trg_padded[:max_seq_len])
    
    src_tensor = torch.tensor(filtered_src, dtype=torch.long)
    trg_tensor = torch.tensor(filtered_trg, dtype=torch.long)
    
    return src_tensor, trg_tensor


def train_or_load_tokenizers(train_src_data, train_trg_data, 
                              src_model_prefix, trg_model_prefix,
                              vocab_size=32000, force_train=False):
    """
    Train new or load existing SentencePiece tokenizers.
    
    Args:
        train_src_data: Source training data
        train_trg_data: Target training data
        src_model_prefix: Prefix for source model
        trg_model_prefix: Prefix for target model
        vocab_size: Vocabulary size (default 32000)
        force_train: Force retraining even if models exist
    
    Returns:
        Tuple of (src_tokenizer, trg_tokenizer)
    """
    src_model_path = f'{src_model_prefix}.model'
    trg_model_path = f'{trg_model_prefix}.model'
    
    # Source tokenizer (Vietnamese)
    src_tokenizer = SentencePieceTokenizer(vocab_size=vocab_size)
    if os.path.exists(src_model_path) and not force_train:
        print(f"Loading existing source tokenizer: {src_model_path}")
        src_tokenizer.load(src_model_path)
    else:
        print(f"Training source tokenizer (vocab_size={vocab_size})...")
        src_tokenizer.train(train_src_data, src_model_prefix, lang='vi')
    
    # Target tokenizer (English)
    trg_tokenizer = SentencePieceTokenizer(vocab_size=vocab_size)
    if os.path.exists(trg_model_path) and not force_train:
        print(f"Loading existing target tokenizer: {trg_model_path}")
        trg_tokenizer.load(trg_model_path)
    else:
        print(f"Training target tokenizer (vocab_size={vocab_size})...")
        trg_tokenizer.train(train_trg_data, trg_model_prefix, lang='en')
    
    return src_tokenizer, trg_tokenizer


def preprocess_data(train_src_path=None, train_trg_path=None, val_src_path=None, val_trg_path=None,
                    vocab_size=32000, force_train_tokenizer=False, use_huggingface=True,
                    hf_dataset_name='ncduy/mt-en-vi'):
    """
    Main preprocessing function with normalization + SentencePiece (spm32k, spm32k).
    
    Args:
        train_src_path: Path to training source file (Vietnamese) - optional if use_huggingface=True
        train_trg_path: Path to training target file (English) - optional if use_huggingface=True
        val_src_path: Path to validation source file - optional if use_huggingface=True
        val_trg_path: Path to validation target file - optional if use_huggingface=True
        vocab_size: SentencePiece vocabulary size (default 32000)
        force_train_tokenizer: Force retraining tokenizers
        use_huggingface: If True, load from Hugging Face dataset
        hf_dataset_name: Name of Hugging Face dataset
    
    Returns:
        Tuple of (trg_tokenizer, src_tokenizer, train_sequences, val_sequences)
        Note: Returns (en_tokenizer, vi_tokenizer, ...) for compatibility
    """
    print(f"\n{'='*60}")
    print("PREPROCESSING: Normalization + SentencePiece (spm32k, spm32k)")
    print(f"{'='*60}")
    
    # Load raw data
    print("Loading data...")
    if use_huggingface:
        train_dataset, val_dataset, test_dataset = load_data_from_huggingface(hf_dataset_name)
        # Extract Vietnamese and English sentences
        train_src = [example['vi'] for example in train_dataset]
        train_trg = [example['en'] for example in train_dataset]
        val_src = [example['vi'] for example in val_dataset]
        val_trg = [example['en'] for example in val_dataset]
    else:
        train_src, train_trg = load_data(train_src_path, train_trg_path)
        val_src, val_trg = load_data(val_src_path, val_trg_path)
    
    print(f"Train samples: {len(train_src)}")
    print(f"Val samples: {len(val_src)}")
    
    # Train or load SentencePiece tokenizers
    spm_dir = os.path.join(saved_tokenizer_path, 'spm')
    os.makedirs(spm_dir, exist_ok=True)
    
    src_model_prefix = os.path.join(spm_dir, 'spm_vi_32k')
    trg_model_prefix = os.path.join(spm_dir, 'spm_en_32k')
    
    src_tokenizer, trg_tokenizer = train_or_load_tokenizers(
        train_src, train_trg,
        src_model_prefix, trg_model_prefix,
        vocab_size=vocab_size,
        force_train=force_train_tokenizer
    )
    
    print(f"\nSource vocab size: {src_tokenizer.get_vocab_size()}")
    print(f"Target vocab size: {trg_tokenizer.get_vocab_size()}")
    
    # Create sequences
    print("\nCreating training sequences...")
    src_train_seq, trg_train_seq = create_sequences(
        train_src, train_trg, 
        src_tokenizer, trg_tokenizer,
        MAX_SEQ_LEN, src_lang='vi', trg_lang='en'
    )
    
    print("Creating validation sequences...")
    src_val_seq, trg_val_seq = create_sequences(
        val_src, val_trg,
        src_tokenizer, trg_tokenizer,
        MAX_SEQ_LEN, src_lang='vi', trg_lang='en'
    )
    
    print(f"\nFiltered train samples: {len(src_train_seq)}")
    print(f"Filtered val samples: {len(src_val_seq)}")
    
    # Create paired sequences (src, trg) - Vietnamese to English
    all_train_sequences = list(zip(src_train_seq, trg_train_seq))
    all_val_sequences = list(zip(src_val_seq, trg_val_seq))
    
    # Return tokenizers in (en, vi) order for compatibility with existing code
    return trg_tokenizer, src_tokenizer, all_train_sequences, all_val_sequences


# ============================================
# LEGACY COMPATIBILITY FUNCTIONS
# ============================================

def get_tokenize(data, add_start_end=False):
    """Legacy function - kept for compatibility."""
    from keras.src.legacy.preprocessing.text import Tokenizer
    tokenizer = Tokenizer(filters='', oov_token=UNKNOWN_TOKEN)
    if add_start_end:
        tokenizer.fit_on_texts([START_TOKEN, END_TOKEN] + data)
    else:
        tokenizer.fit_on_texts(data)
    return data, tokenizer


def merge_sentences(text, max_seq_length):
    """Merge sentences up to max sequence length."""
    sentences = [s.strip() for s in text.split(",")]
    merged = []
    temp = ""
    word_count = 0

    for sentence in sentences:
        words = sentence.split()
        if word_count + len(words) <= max_seq_length:
            temp = temp + ", " + sentence if temp else sentence
            word_count += len(words)
        else:
            merged.append(temp)
            temp = sentence
            word_count = len(words)

    if temp:
        merged.append(temp)

    return merged
