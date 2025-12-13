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