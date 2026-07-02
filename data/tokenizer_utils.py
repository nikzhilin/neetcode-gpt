from typing import List, Dict

class Solution:
    def tokenize_numbers(self, numbers: List[int], vocab: Dict[str, int]) -> List[List[str]]:
        tokens = []
        max_token_len = max(len(token) for token in vocab)

        for number in numbers:
            number_str = str(number)
            token = []
            i = 0

            while i < len(number_str):
                for length in range(max_token_len, 0, -1):
                    candidate = number_str[i:i + length]

                    if candidate in vocab:
                        token.append(candidate)
                        i += length
                        break

            tokens.append(token)

        return tokens

    def count_tokens(self, text: str, vocab: Dict[str, int]) -> int:
        count = 0
        i = 0
        while i < len(text):
            best_token = None
            for token in vocab:
                if (text.startswith(token, i) and
                   (best_token is None or len(token) > len(best_token))):
                   best_token = token

            count += 1
            i += len(best_token)

        return count

    def fertility_score(self, text: str, vocab: Dict[str, int]) -> float:
        count_tokens = self.count_tokens(text, vocab)
        count_words = len(text.split())

        return round(count_tokens / count_words, 4)
