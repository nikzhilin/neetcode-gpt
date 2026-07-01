import numpy as np
from numpy.typing import NDArray


class Solution:

    def softmax(self, z: NDArray[np.float64]) -> NDArray[np.float64]:
        exp_z = np.exp(z - np.max(z))
        return np.round(exp_z / np.sum(exp_z), 4)
