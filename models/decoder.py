from torch import nn

from models.attention import MultiHeadAttention
from models.embedding import *
from config import *

class Decoder_Layer(nn.Module):
    def __init__(self, d_model, d_ff, num_heads, dropout):
        super(Decoder_Layer, self).__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads)
        self.norm1 = nn.LayerNorm(d_model, eps=EPS)
        self.dropout1 = nn.Dropout(dropout)

        self.enc_dec_attn = MultiHeadAttention(d_model, num_heads)
        self.norm2 = nn.LayerNorm(d_model, eps=EPS)
        self.dropout2 = nn.Dropout(dropout)

        self.ffn = PositionwiseFeedForward(d_model, d_ff, DROPOUT)
        self.norm3 = nn.LayerNorm(d_model, eps=EPS)
        self.dropout3 = nn.Dropout(dropout)

    def forward(self, x, enc_out, trg_mask, src_mask):
        # 1. compute self attention
        _x = x
        x = self.self_attn(x, x, x, mask=trg_mask)

        # 2. add and norm
        x = self.dropout1(x)
        x = self.norm1(_x + x)

        if enc_out is not None:
            # 3. compute encoder - decoder attention
            _x = x
            x = self.enc_dec_attn(x, enc_out, enc_out, mask=src_mask)

            # 4. add and norm
            x = self.dropout2(x)
            x = self.norm2(_x + x)

        # 5. positionwise feed forward network
        _x = x
        x = self.ffn(x)

        # 6. add and norm
        x = self.dropout3(x)
        x = self.norm3(_x + x)

        return x

class Decoder(nn.Module):
    def __init__(self, trg_vocab_size, max_len, d_model, d_ff, num_heads, num_layers, dropout, device):
        super(Decoder, self).__init__()
        self.embedding = TransformerEmbedding(trg_vocab_size, d_model, max_len, dropout, device)
        self.layers = nn.ModuleList([Decoder_Layer(d_model, d_ff, num_heads, dropout) for i in range(num_layers)])
        self.linear = nn.Linear(d_model, trg_vocab_size)

    def forward(self, trg, enc_src, trg_mask, src_mask):
        trg = self.embedding(trg)

        for layer in self.layers:
            trg = layer(trg, enc_src, trg_mask, src_mask)

        # pass to LM head
        output = self.linear(trg)

        return output
