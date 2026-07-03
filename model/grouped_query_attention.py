import torch
import torch.nn as nn
from torchtyping import TensorType

class GroupedQueryAttention(nn.Module):
    def __init__(self, model_dim: int, num_heads: int, num_kv_heads: int):
        super().__init__()
        torch.manual_seed(0)
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = model_dim // num_heads

        self.q_proj = nn.Linear(model_dim, num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(model_dim, num_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(model_dim, num_kv_heads * self.head_dim, bias=False)
        self.output_proj = nn.Linear(num_heads * self.head_dim, model_dim, bias=False)

    def forward(self, x: TensorType[float]) -> TensorType[float]:
        B, T, D = x.shape
    
        Q = self.q_proj(x)
        K = self.k_proj(x)
        V = self.v_proj(x)

        Q = Q.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(B, T, self.num_kv_heads, self.head_dim).transpose(1, 2)
        V = V.view(B, T, self.num_kv_heads, self.head_dim).transpose(1, 2)

        repeat_factor = self.num_heads // self.num_kv_heads
        K = K.repeat_interleave(repeat_factor, dim=1)
        V = V.repeat_interleave(repeat_factor, dim=1)

        scores = scores = Q @ K.transpose(-1, -2) / torch.sqrt(torch.tensor(self.head_dim))
        mask = torch.tril(torch.ones(T, T))
        scores = scores.masked_fill(mask == 0, float("-inf"))

        attention = torch.softmax(scores, dim=-1)
        
        out = attention @ V
        out = out.transpose(1, 2).contiguous().view(B, T, D)
        out = self.output_proj(out)

        return torch.round(out, decimals=4)


