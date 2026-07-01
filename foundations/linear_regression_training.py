import numpy as np
from numpy.typing import NDArray


class Solution:
    def get_derivative(self, model_prediction: NDArray[np.float64], ground_truth: NDArray[np.float64], N: int, X: NDArray[np.float64], desired_weight: int) -> float:
        return -2 * np.dot(ground_truth - model_prediction, X[:, desired_weight]) / N

    def get_gradient(self, model_prediction: NDArray[np.float64], ground_truth: NDArray[np.float64], N: int, X: NDArray[np.float64]) -> NDArray[np.float64]:
        return -2 * X.T @ (ground_truth - model_prediction) / N

    def get_model_prediction(self, X: NDArray[np.float64], weights: NDArray[np.float64]) -> NDArray[np.float64]:
        return np.squeeze(np.matmul(X, weights))

    learning_rate = 0.01

    def train_model(
        self,
        X: NDArray[np.float64],
        Y: NDArray[np.float64],
        num_iterations: int,
        initial_weights: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        
        final_weights = initial_weights
        for _ in range(num_iterations):
            predictions = self.get_model_prediction(X, final_weights)
            gradient = self.get_gradient(predictions, Y, predictions.size, X)
            final_weights -= self.learning_rate * gradient                
             
        return np.round(final_weights, 5)
        
