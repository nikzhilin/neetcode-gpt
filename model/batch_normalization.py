import numpy as np
from typing import Tuple, List


class Solution:
    def batch_norm(self, x: List[List[float]], gamma: List[float], beta: List[float],
                   running_mean: List[float], running_var: List[float],
                   momentum: float, eps: float, training: bool) -> Tuple[List[List[float]], List[float], List[float]]:
        # During training: normalize using batch statistics, then update running stats
        # During inference: normalize using running stats (no batch stats needed)
        # Apply affine transform: y = gamma * x_hat + beta
        # Return (y, running_mean, running_var), all rounded to 4 decimals as lists
        if training:
            mu = np.mean(x, axis=0)
            sigma_sq = np.var(x, axis=0)
            running_mean = (1 - momentum) * np.array(running_mean) + momentum * mu
            running_var = (1 - momentum) * np.array(running_var) + momentum * sigma_sq
        else:
            mu = np.array(running_mean)
            sigma_sq = np.array(running_var)

        x_hat = (np.array(x) - mu) / np.sqrt(sigma_sq + eps)        
        return (list(np.round(x_hat * gamma + beta, 4)), 
                list(np.round(running_mean, 4)), 
                list(np.round(running_var, 4)))
        