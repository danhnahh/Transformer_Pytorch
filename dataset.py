from pyvi.ViTokenizer import ViTokenizer
from keras.src.legacy.preprocessing.text import Tokenizer
from keras.src.utils import pad_sequences
from config import *


# Đọc dữ liệu từ tệp
def load_data(en_file, vi_file):
    with open(en_file, 'r', encoding='utf-8') as f:
        en_data = f.read().strip().split("\n")
    with open(vi_file, 'r', encoding='utf-8') as f:
        vi_data = f.read().strip().split("\n")
    return en_data, vi_data


def get_tokenize(data, add_start_end=False):
    # Khởi tạo Tokenizer
    tokenizer = Tokenizer(filters='', oov_token=UNKNOWN_TOKEN)
    if (add_start_end):
        tokenizer.fit_on_texts([START_TOKEN, END_TOKEN] + data)
    else:
        tokenizer.fit_on_texts(data)
    return data, tokenizer

def get_tokenize_seq(en_data, vi_data, en_tokenizer, vi_tokenizer, max_sequence_length):
    en_data = [f"{START_TOKEN} {sentence} {END_TOKEN}" for sentence in en_data]
    en_sequences = en_tokenizer.texts_to_sequences(en_data)

    vi_data = [ViTokenizer.tokenize(sentence) for sentence in vi_data]
    vi_sequences = vi_tokenizer.texts_to_sequences(vi_data)

    filtered_en = []
    filtered_vi = []
    # Giữ lại những câu có số từ <= max_sequence_length
    for i in range(len(en_sequences)):
        if (len(en_sequences[i]) <= max_sequence_length) and (len(vi_sequences[i]) <= max_sequence_length):
            filtered_en.append(en_sequences[i])
            filtered_vi.append(vi_sequences[i])

    filtered_en = torch.tensor(pad_sequences(filtered_en, maxlen=max_sequence_length, padding='post'), dtype=torch.long)
    filtered_vi = torch.tensor(pad_sequences(filtered_vi, maxlen=max_sequence_length, padding='post'), dtype=torch.long)

    return filtered_en, filtered_vi

# Tiền xử lý dữ liệu
def preprocess_tokenizer(en_data, vi_data):
    en_data, en_tokenizer = get_tokenize(en_data, add_start_end=True)

    vi_data = [ViTokenizer.tokenize(sentence) for sentence in vi_data]
    vi_data, vi_tokenizer = get_tokenize(vi_data)

    return en_tokenizer, vi_tokenizer

def preprocess_data(train_src_path, train_trg_path, val_src_path, val_trg_path):
    # Load dữ liệu
    en_data, vi_data = load_data(train_src_path, train_trg_path)
    en_data_val, vi_data_val = load_data(val_src_path, val_trg_path)

    en_tokenizer, vi_tokenizer = preprocess_tokenizer(en_data, vi_data)

    en_sequences, vi_sequences = get_tokenize_seq(en_data, vi_data, en_tokenizer, vi_tokenizer,
                                                  max_sequence_length=MAX_SEQ_LEN)
    en_val_sequences, vi_val_sequences = get_tokenize_seq(en_data_val, vi_data_val, en_tokenizer, vi_tokenizer,
                                                          max_sequence_length=MAX_SEQ_LEN)

    all_train_sequences = list(zip(vi_sequences, en_sequences))
    all_val_sequences = list(zip(vi_val_sequences, en_val_sequences))

    return en_tokenizer, vi_tokenizer, all_train_sequences, all_val_sequences

def merge_sentences(text, max_seq_length):
    sentences = [s.strip() for s in text.split(",")]  # Tách câu và xóa khoảng trắng dư thừa

    merged = []
    temp = ""
    word_count = 0

    for sentence in sentences:
        words = sentence.split()  # Đếm số từ trong câu hiện tại
        if word_count + len(words) <= max_seq_length:
            temp = temp + ", " + sentence if temp else sentence  # Nối câu
            word_count += len(words)  # Cập nhật số từ
        else:
            merged.append(temp)  # Lưu câu hiện tại vào danh sách
            temp = sentence  # Bắt đầu câu mới
            word_count = len(words)  # Reset số từ

    if temp:  # Đừng quên thêm câu cuối cùng
        merged.append(temp)

    return merged
