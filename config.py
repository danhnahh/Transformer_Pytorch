# Đường dẫn dữ liệu
import torch

data_path = 'data/archive/IWSLT\'15 en-vi/'
train_data_path = 'data/archive/IWSLT\'15 en-vi/'
saved_model_path = 'checkpoints/'
saved_tokenizer_path = 'checkpoints'
test_data_path = 'data/archive/IWSLT\'15 en-vi/'

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

MAX_SEQ_LEN =  80 #60  # Độ dài tối đa của câu

# Huấn luyện mô hình
NUM_LAYERS = 6
D_MODEL = 384   
D_FF = 2048
EPS = 1e-3
BATCH_SIZE = 128 #164 * 2
NUM_HEADS = 8
EPOCHS = 60
DROPOUT = 0.3
CLIP = 1.0
BATCH_PRINT = 50 #100
VOCAB_SIZE = 32000

#Learning rate
LEARNING_RATE = 5e-4 # 3e-4
DECAY_RATE = [1.3, 0.95]
DECAY_STEP = [3600]
DECAY_INTERVAL = 390
WEIGHT_DECAY = 1e-4 

#train
PAD_TOKEN_POS = 0
GRADIENT_ACCUMULATION = 4
VAL_STEP = 100 #200
#LRwarmup
WARMUP_RATIO = 0.15
FINAL_LR_RATIO = 0.05

#infer
BEAM_SIZE = 5

UNKNOWN_TOKEN = '<unk>'
PAD_TOKEN = '<pad>'
START_TOKEN = '<start>'
END_TOKEN = '<end>'