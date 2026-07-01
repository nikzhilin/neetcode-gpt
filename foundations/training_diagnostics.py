import torch
import torch.nn as nn
from typing import List, Dict


class Solution:

    def compute_activation_stats(self, model: nn.Module, x: torch.Tensor) -> List[Dict[str, float]]:
        statistics = []
        with torch.no_grad():
            for layer in model.children():
                x = layer(x)

                if isinstance(layer, nn.Linear):
                    mean = round(x.mean().item(), 4)
                    std = round(x.std().item(), 4)
                    dead_neurons = (x <= 0).all(dim=0)
                    dead_fraction = round(dead_neurons.float().mean().item(), 4)

                    statistics.append({"mean" : mean,
                                       "std" : std,
                                       "dead_fraction" : dead_fraction})

        return statistics

    def compute_gradient_stats(self, model: nn.Module, x: torch.Tensor, y: torch.Tensor) -> List[Dict[str, float]]:
        model.zero_grad()

        y_pred = model(x)  
        loss = nn.MSELoss()(y_pred, y)
        loss.backward()

        statistics = []      
        for layer in model.children():                
            if isinstance(layer, nn.Linear):
                grads = layer.weight.grad
                mean = round(grads.mean().item(), 4)
                std = round(grads.std().item(), 4)
                norm = round(torch.norm(grads).item(), 4)

                statistics.append({"mean" : mean,
                                    "std" : std,
                                   "norm" : norm})

        return statistics
                

    def diagnose(self, activation_stats: List[Dict[str, float]], gradient_stats: List[Dict[str, float]]) -> str:    
        if any(stats["dead_fraction"] > 0.5 for stats in activation_stats):
            return "dead_neurons"

        elif any(stats["norm"] > 1000 for stats in gradient_stats):
            return "exploding_gradients"

        elif any(stats["norm"] < 1e-5 for stats in gradient_stats):
            return "vanishing_gradients"

        elif any(stats["std"] < 0.1 for stats in activation_stats):
            return "vanishing_gradients"

        elif any(stats["std"] > 10.0 for stats in activation_stats):
            return "exploding_gradients"

        return "healthy"