import torch
import torch.nn as nn
from torchtyping import TensorType
from typing import List

class Solution:
    def get_dataset(self, positive: List[str], negative: List[str]) -> TensorType[float]:
        # 1. Build vocabulary: collect all unique words, sort them, assign integer IDs starting at 1
        # 2. Encode each sentence by replacing words with their IDs
        # 3. Combine positive + negative into one list of tensors
        # 4. Pad shorter sequences with 0s using nn.utils.rnn.pad_sequence(tensors, batch_first=True)

        unique_words = (
                    set((word for sentence in positive for word in sentence.split())) 
                    | 
                    set((word for sentence in negative for word in sentence.split()))
                    )

        vocabulary = {word : index for index, word in enumerate(sorted(unique_words), start=1)}
        
        sequences = [
            torch.tensor([vocabulary[word] for word in sentence.split()], dtype=torch.long)
            for sentence in positive + negative
        ]

        tensor = nn.utils.rnn.pad_sequence(sequences, batch_first=True)

        return tensor
        
