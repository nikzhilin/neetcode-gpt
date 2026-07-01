import torch
import torch.nn as nn
import numpy as np
from typing import List

class Solution:

    def xavier_init(self, fan_in: int, fan_out: int) -> List[List[float]]:        
        torch.manual_seed(0)
        
        std = np.sqrt(2 / (fan_in + fan_out))
        weights = torch.randn((fan_out, fan_in)) * std
        return torch.round(weights, decimals=4).tolist()


    def kaiming_init(self, fan_in: int, fan_out: int) -> List[List[float]]:
        torch.manual_seed(0)

        std = np.sqrt(2 / fan_in)
        weights = torch.randn((fan_out, fan_in)) * std
        return torch.round(weights, decimals=4).tolist()
        

    def check_activations(self, num_layers: int, input_dim: int, hidden_dim: int, init_type: str) -> List[float]:        
        torch.manual_seed(0)

        all_weights = []
        for layer in range(num_layers):
            fan_in = input_dim if layer == 0 else hidden_dim
            fan_out = hidden_dim

            if init_type == 'xavier':
                std = np.sqrt(2 / (fan_in + fan_out))
                weights = torch.randn((fan_out, fan_in)) * std
            elif init_type == 'kaiming':
                std = np.sqrt(2 / fan_in)
                weights = torch.randn((fan_out, fan_in)) * std
            else: # random
                weights = torch.randn((fan_out, fan_in))
        
            all_weights.append(weights)

        x = torch.randn(1, input_dim)
        stds = []
        for weights in all_weights:
            x = x @ weights.T
            x = torch.relu(x)
            stds.append(round(x.std().item(), 2))

        return stds

