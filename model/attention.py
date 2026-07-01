import torch
import torch.nn as nn
from torchtyping import TensorType
import numpy as np

class SingleHeadAttention(nn.Module):

    def __init__(self, embedding_dim: int, attention_dim: int):
        super().__init__()
        torch.manual_seed(0)
        # Create three linear projections (Key, Query, Value) with bias=False
        # Instantiation order matters for reproducible weights: key, query, value

        self.W_K = nn.Linear(embedding_dim, attention_dim, bias=False)
        self.W_Q = nn.Linear(embedding_dim, attention_dim, bias=False)    
        self.W_V = nn.Linear(embedding_dim, attention_dim, bias=False)
        

    def forward(self, embedded: TensorType[float]) -> TensorType[float]:
        # 1. Project input through K, Q, V linear layers
        # 2. Compute attention scores: (Q @ K^T) / sqrt(attention_dim)
        # 3. Apply causal mask: use torch.tril(torch.ones(...)) to build lower-triangular matrix,
        #    then masked_fill positions where mask == 0 with float('-inf')
        # 4. Apply softmax(dim=2) to masked scores
        # 5. Return (scores @ V) rounded to 4 decimal places
        
        Q = self.W_Q(embedded)
        K = self.W_K(embedded)
        V = self.W_V(embedded)

        scores = Q @ K.transpose(-1, -2) / torch.sqrt(torch.tensor(Q.size(-1)))

        mask = torch.tril(torch.ones(Q.size(1), Q.size(1)))

        scores = scores.masked_fill(mask == 0, float("-inf"))
        scores = torch.softmax(scores, dim=-1)

        return torch.round(scores @ V, decimals=4)



