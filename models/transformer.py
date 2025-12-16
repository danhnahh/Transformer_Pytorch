import torch
from torch import nn
from models.encoder import Encoder
from models.decoder import Decoder

class Transformer(nn.Module):
    def __init__(self, src_pad_idx, trg_pad_idx, inp_vocab_size, trg_vocab_size, d_model, num_heads, max_len, d_ff, num_layers, dropout, device, use_alignment=True):
        super(Transformer, self).__init__()
        self.src_pad_idx = src_pad_idx
        self.trg_pad_idx = trg_pad_idx
        self.device = device

        self.encoder = Encoder(inp_vocab_size, max_len, d_model, d_ff, num_heads, num_layers, dropout, device)
        self.decoder = Decoder(trg_vocab_size, max_len, d_model, d_ff, num_heads, num_layers, dropout, device, use_alignment=use_alignment)

    def forward(self, src, trg):
        src_mask = self.make_src_mask(src)
        trg_mask = self.make_trg_mask(trg)
        enc_out = self.encoder(src, src_mask)
        output = self.decoder(trg, enc_out, trg_mask, src_mask)
        return output

    def make_src_mask(self, src):
        src_mask = (src != self.src_pad_idx).unsqueeze(dim=1).unsqueeze(dim=2)
        return src_mask

    def make_trg_mask(self, trg):
        trg_pad_mask = (trg != self.trg_pad_idx).unsqueeze(dim=1).unsqueeze(dim=3)
        trg_len = trg.shape[1]
        trg_look_ahead_mask = torch.tril(torch.ones(trg_len, trg_len)).bool().to(self.device)
        trg_mask = trg_pad_mask & trg_look_ahead_mask

        return trg_mask