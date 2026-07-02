from typing import Dict, List, Tuple

class Solution:
    def build_vocab(self, text: str) -> Tuple[Dict[str, int], Dict[int, str]]:
        # Return (stoi, itos) where:
        # - stoi maps each unique character to a unique integer (sorted alphabetically)
        # - itos is the reverse mapping (integer to character)

        unique = list(sorted(set(text)))
        stoi = {char : index for index, char in enumerate(unique)}
        itos = dict(enumerate(unique))
        return (stoi, itos)
            
    def encode(self, text: str, stoi: Dict[str, int]) -> List[int]:
        encoded = [stoi[char] for char in text]
        return encoded

    def decode(self, ids: List[int], itos: Dict[int, str]) -> str:
        # Convert a list of integers back to a string using itos mapping
        decoded = [itos[number] for number in ids]
        return "".join(decoded)
