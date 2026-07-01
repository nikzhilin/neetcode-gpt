import numpy as np
from numpy.typing import NDArray


class Solution:
    def get_positional_encoding(self, seq_len: int, d_model: int) -> NDArray[np.float64]:        
        positions = np.arange(seq_len)[:, np.newaxis]      # (seq_len, 1)
        indices = np.arange(0, d_model, 2)[np.newaxis, :]  # (1, d_model / 2)

        div_term = 10000 ** (indices / d_model)

        positional_embeddings = np.zeros((seq_len, d_model), dtype=np.float64)

        positional_embeddings[:, 0::2] = np.sin(positions / div_term)
        positional_embeddings[:, 1::2] = np.cos(positions / div_term)

        return np.round(positional_embeddings, 5)
        
        
