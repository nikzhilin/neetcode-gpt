class Solution:
    def get_minimizer(self, iterations: int, learning_rate: float, init: int) -> float:
        minimum = init
        for _ in range(iterations):
            minimum = minimum - 2 * learning_rate * minimum

        return round(minimum, 5)
