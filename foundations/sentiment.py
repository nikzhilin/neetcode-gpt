import torch
import torch.nn as nn
from torchtyping import TensorType

class Solution(nn.Module):
    def __init__(self, vocabulary_size: int):
        super().__init__()
        torch.manual_seed(0)

        self.embeddings = nn.Embedding(vocabulary_size, 16)
        self.linear = nn.Linear(16, 1)
        self.sigmoid = nn.Sigmoid()
        
        

    def forward(self, x: TensorType[int]) -> TensorType[float]:        
        emdeddings = self.embeddings(x)
        emdeddings = emdeddings.mean(dim=1)
        emdeddings = self.linear(emdeddings)
        emdeddings = self.sigmoid(emdeddings)
    
        return torch.round(emdeddings, decimals=4)
