from torch import nn
from models.attention import MultiHeadAttention
from models.embedding import *
from config import *

class EncoderLayer(nn.Module):
    def __init__(self, d_model, d_ff, num_heads, dropout):
        super(EncoderLayer, self).__init__()
        self.attention = MultiHeadAttention(d_model, num_heads)
        self.norm1 = nn.LayerNorm(d_model, eps=EPS)
        self.dropout1 = nn.Dropout(dropout)

        self.ffn = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.norm2 = nn.LayerNorm(d_model, eps=EPS)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x, src_mask):
        # 1. compute self attention
        _x = x
        x = self.attention(x, x, x, src_mask)

        # 2. add and norm
        x = self.dropout1(x)
        x = self.norm1(_x + x)

        # 3. positionwise feed forward network
        _x = x
        x = self.ffn(x)

        # 4. add and norm
        x = self.dropout2(x)
        x = self.norm2(_x + x)

        return x

class Encoder(nn.Module):
    def __init__(self, inp_vocab_size, max_len, d_model, d_ff, num_heads, num_layers, dropout, device):
        super(Encoder, self).__init__()
        self.emb = TransformerEmbedding(inp_vocab_size, d_model, max_len, dropout, device=device)
        self.layers = nn.ModuleList([EncoderLayer(d_model, d_ff, num_heads, dropout) for _ in range(num_layers)])

    def forward(self, src, src_mask):
        x = self.emb(src)
        for layer in self.layers:
            x = layer(x, src_mask)

        return x