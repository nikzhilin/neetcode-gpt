import torch
import torch.nn as nn
from typing import List


class Solution:

    def detect_dead_neurons(self, model: nn.Module, x: torch.Tensor) -> List[float]:
        dead_fractions = []
        with torch.no_grad():
            for layer in model.children():
                x = layer(x)

                if isinstance(layer, nn.ReLU):
                    dead_fractions.append(
                        round((x == 0).all(dim=0).float().mean().item(), 4)
                        )

        return dead_fractions


    def suggest_fix(self, dead_fractions: List[float]) -> str:
        if max(dead_fractions) > 0.5:
            return 'use_leaky_relu'
        elif dead_fractions[0] > 0.3:
            return 'reinitialize'
        elif dead_fractions[-1] > 0.1 and all(
                dead_fractions[i] < dead_fractions[i + 1] 
                for i in range(len(dead_fractions) - 1)):
            return 'reduce_learning_rate'
            
        return 'healthy'
