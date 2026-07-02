from typing import List
from collections import defaultdict

class Solution:
    def get_merges(self, corpus: str, num_merges: int) -> List[List[str]]:
        tokens = list(corpus)
        merges = []

        for _ in range(num_merges):
            frequencies = defaultdict(int)

            for left, right in zip(tokens, tokens[1:]):
                frequencies[(left, right)] += 1            

            best_pair = min(
                frequencies.keys(),
                key=lambda pair: (-frequencies[pair], pair)
            )

            merges.append([best_pair[0], best_pair[1]])

            new_tokens = []
            j = 0

            while j < len(tokens):
                if (
                    j + 1 < len(tokens)
                    and tokens[j] == best_pair[0]
                    and tokens[j + 1] == best_pair[1]
                ):
                    new_tokens.append(tokens[j] + tokens[j + 1])
                    j += 2
                else:
                    new_tokens.append(tokens[j])
                    j += 1

            tokens = new_tokens

        return merges