import numpy as np
from typing import List


class Solution:
    def rms_norm(self, x: List[float], gamma: List[float], eps: float) -> List[float]:
        # Implement RMS Normalization (similar to LayerNorm but without mean centering or beta)
        # Normalize x, then scale by gamma
        # Return result rounded to 4 decimal places as a list
        
        RMSx = np.sqrt(np.mean(np.square(np.asarray(x, dtype="float64"))) + eps)
        x_hat = np.asarray(x, dtype="float64") / RMSx
        y_hat = gamma * x_hat 
        return np.round(y_hat, 4)
