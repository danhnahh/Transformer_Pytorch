# Đường dẫn dữ liệu
import torch

data_path = 'data/archive/IWSLT\'15 en-vi/'
train_data_path = 'data/archive/IWSLT\'15 en-vi/'
saved_model_path = 'checkpoints/'
saved_tokenizer_path = 'checkpoints'
test_data_path = 'data/archive/IWSLT\'15 en-vi/'

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

MAX_SEQ_LEN = 80 #60  # Độ dài tối đa của câu

# Huấn luyện mô hình
NUM_LAYERS = 6
D_MODEL = 512   
D_FF = 2048
EPS = 0.1
BATCH_SIZE = 128  # Reduced from 512 to fit in GPU memory
NUM_HEADS = 8
EPOCHS = 10
DROPOUT = 0.3
CLIP = 1.0
BATCH_PRINT = 50 #100

#Learning rate
LEARNING_RATE = 5e-4 # 3e-4
DECAY_RATE = [1.3, 0.95]
DECAY_STEP = [3600]
DECAY_INTERVAL = 390
WEIGHT_DECAY = 1e-4 

#train
PAD_TOKEN_POS = 0
GRADIENT_ACCUMULATION = 4  # Increased to maintain effective batch size of 512
VAL_STEP = 100 #200
#LRwarmup
WARMUP_RATIO = 0.15
FINAL_LR_RATIO = 0.05

#infer
BEAM_SIZE = 5

# Debug mode
DEBUG_MODE = True  # Set to True to use only a fraction of data
DEBUG_DATA_FRACTION = 0.1  # Use 10% of data in debug mode

UNKNOWN_TOKEN = '<unk>'
PAD_TOKEN = '<pad>'
START_TOKEN = '<start>'
END_TOKEN = '<end>'
