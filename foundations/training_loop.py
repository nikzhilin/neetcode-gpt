import numpy as np
from numpy.typing import NDArray
from typing import Tuple


class Solution:
    def train(self, X: NDArray[np.float64], y: NDArray[np.float64], epochs: int, lr: float) -> Tuple[NDArray[np.float64], float]:
        # X: (n_samples, n_features)
        # y: (n_samples,) targets
        # epochs: number of training iterations
        # lr: learning rate
                
        n_samples, n_features = X.shape

        weights = np.zeros(n_features)
        bias = 0
        for epoch in range(epochs):
            y_hat = X @ weights + bias
            
            dL_dw = 2 / n_samples * X.T @ (y_hat - y)
            dL_db = 2 * np.mean(y_hat - y)

            weights = weights - lr * dL_dw
            bias = bias - lr * dL_db

        return np.round(weights, 5), np.round(bias, 5)
