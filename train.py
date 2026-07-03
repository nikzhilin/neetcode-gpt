import torch
import torch.nn as nn
import torch.nn.functional as F

class Solution:
    def train(self, model: nn.Module, data: torch.Tensor, epochs: int, context_length: int, batch_size: int, lr: float) -> float:                
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
        loss = None
        for epoch in range(epochs):
            torch.manual_seed(epoch)

            starts = torch.randint(0, len(data) - context_length, (batch_size,))
            X = torch.stack([
                data[start:start + context_length]
                for start in starts
            ])

            Y = torch.stack([
                data[start + 1:start + context_length + 1]
                for start in starts
            ])

            logits = model(X)

            B, T, C = logits.shape
            logits_flat = logits.reshape(B * T, C)
            targets_flat = Y.reshape(B * T)

            loss = F.cross_entropy(logits_flat, targets_flat)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        return round(loss.item(), 4)