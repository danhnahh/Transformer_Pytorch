import torch
from torch import nn
import math

class ScaleDotProductAttention(nn.Module):
    def __init__(self):
        super(ScaleDotProductAttention, self).__init__()
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, query, key, value, mask=None):
        # input is 4 dimension tensor
        # [batch_size, num_heads, length, d_tensor]
        batch_size, num_heads, length, d_tensor = key.size()

        # 1. dot product Query with Key^T to compute similarity
        key_t = key.transpose(2, 3)
        score = (query @ key_t) / math.sqrt(d_tensor)

        # 2. apply masking (opt)
        if mask is not None:
            score = score.masked_fill(mask == 0, -100000000)

        # 3. pass them softmax to make [0, 1] range
        score = self.softmax(score)

        # 4. multiply with Value
        value = score @ value

        return value, score


class AdditiveAttention(nn.Module):
    """
    Bahdanau-style additive attention for better alignment learning.
    score(s, h) = v^T * tanh(W_s * s + W_h * h)
    """
    def __init__(self, d_model):
        super(AdditiveAttention, self).__init__()
        self.w_query = nn.Linear(d_model, d_model, bias=False)
        self.w_key = nn.Linear(d_model, d_model, bias=False)
        self.v = nn.Linear(d_model, 1, bias=False)
        self.softmax = nn.Softmax(dim=-1)
        
    def forward(self, query, key, value, mask=None):
        # query: [batch, num_heads, trg_len, d_k]
        # key: [batch, num_heads, src_len, d_k]
        # value: [batch, num_heads, src_len, d_k]
        
        batch_size, num_heads, trg_len, d_k = query.size()
        src_len = key.size(2)
        
        # Reshape for additive computation
        # query: [batch, num_heads, trg_len, 1, d_k]
        # key: [batch, num_heads, 1, src_len, d_k]
        query_expanded = query.unsqueeze(3)  # [batch, heads, trg_len, 1, d_k]
        key_expanded = key.unsqueeze(2)      # [batch, heads, 1, src_len, d_k]
        
        # Compute additive score
        # [batch, heads, trg_len, src_len, d_k]
        combined = torch.tanh(self.w_query(query_expanded) + self.w_key(key_expanded))
        
        # [batch, heads, trg_len, src_len]
        score = self.v(combined).squeeze(-1)
        
        # Apply mask
        if mask is not None:
            score = score.masked_fill(mask == 0, -100000000)
        
        # Softmax
        attn_weights = self.softmax(score)
        
        # Apply attention to values
        # [batch, heads, trg_len, d_k]
        out = torch.matmul(attn_weights, value)
        
        return out, attn_weights


class AlignmentAttention(nn.Module):
    """
    Combined attention mechanism that uses both dot-product and additive attention
    for better alignment in translation tasks.
    """
    def __init__(self, d_model, use_additive=True):
        super(AlignmentAttention, self).__init__()
        self.use_additive = use_additive
        self.dot_attention = ScaleDotProductAttention()
        
        if use_additive:
            self.additive_attention = AdditiveAttention(d_model // 8)  # Per head dimension
            self.gate = nn.Linear(d_model // 8, 1)
            self.sigmoid = nn.Sigmoid()
    
    def forward(self, query, key, value, mask=None):
        if not self.use_additive:
            return self.dot_attention(query, key, value, mask)
        
        # Get outputs from both attention mechanisms
        dot_out, dot_score = self.dot_attention(query, key, value, mask)
        add_out, add_score = self.additive_attention(query, key, value, mask)
        
        # Gated combination
        gate_weight = self.sigmoid(self.gate(query))  # [batch, heads, len, 1]
        out = gate_weight * dot_out + (1 - gate_weight) * add_out
        
        return out, dot_score
    
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.attention = ScaleDotProductAttention()
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_concat = nn.Linear(d_model, d_model)

    def forward(self, query, key, value, mask=None):
        # 1. dot product with weight matrices
        query, key, value = self.w_q(query), self.w_k(key), self.w_v(value)

        # 2. split tensor by number of heads
        query, key, value = self.split(query), self.split(key), self.split(value)

        # 3. do scale dot product to compute similarity
        out, attention = self.attention(query, key, value, mask=mask)

        # 4. concat and pass to linear layer
        out = self.concat(out)
        out = self.w_concat(out)

        # 5. visualize attention map
        # TODO : we should implement visualization
        return out

    def split(self, tensor):
        batch_size, length, d_model = tensor.size()

        d_tensor = d_model // self.num_heads
        tensor = tensor.view(batch_size, length, self.num_heads, d_tensor).transpose(1, 2)
        # it is similar with group convolution (split by number of heads)

        return tensor

    def concat(self, tensor):
        batch_size, num_heads, length, d_tensor = tensor.size()
        d_model = d_tensor * self.num_heads

        tensor = tensor.transpose(1, 2).contiguous().view(batch_size, length, d_model)
        return tensor


class MultiHeadAlignmentAttention(nn.Module):
    """
    Multi-head attention with alignment-aware mechanism.
    Combines scaled dot-product attention with additive (Bahdanau-style) attention
    for better alignment in encoder-decoder attention.
    """
    def __init__(self, d_model, num_heads, use_alignment=True):
        super(MultiHeadAlignmentAttention, self).__init__()
        self.num_heads = num_heads
        self.d_model = d_model
        self.d_k = d_model // num_heads
        self.use_alignment = use_alignment
        
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_concat = nn.Linear(d_model, d_model)
        
        self.attention = AlignmentAttention(d_model, use_additive=use_alignment)

    def forward(self, query, key, value, mask=None):
        # 1. Linear projections
        query, key, value = self.w_q(query), self.w_k(key), self.w_v(value)

        # 2. Split into heads
        query, key, value = self.split(query), self.split(key), self.split(value)

        # 3. Alignment-aware attention
        out, attention = self.attention(query, key, value, mask=mask)

        # 4. Concat and project
        out = self.concat(out)
        out = self.w_concat(out)

        return out

    def split(self, tensor):
        batch_size, length, d_model = tensor.size()
        d_tensor = d_model // self.num_heads
        tensor = tensor.view(batch_size, length, self.num_heads, d_tensor).transpose(1, 2)
        return tensor

    def concat(self, tensor):
        batch_size, num_heads, length, d_tensor = tensor.size()
        d_model = d_tensor * self.num_heads
        tensor = tensor.transpose(1, 2).contiguous().view(batch_size, length, d_model)
        return tensor