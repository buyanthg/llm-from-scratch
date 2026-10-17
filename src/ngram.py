from collections import Counter
import re


def tokenize_text(text: str) -> list[str]:
    return re.findall(r"[\w']+", text.lower())


class NGramLM:
    def __init__(self, n: int=2):
        if n < 1:
            raise ValueError("n must be >= 1")
        self.n = n
        self.ngram_counts = Counter()
        self.context_counts = Counter()
        self.vocab = set()

    def fit(self, corpus: list[str]) -> None:
        for text in corpus:
            tokens = tokenize_text(text)
            self.vocab.update(tokens)

            for i in range(len(tokens) - self.n + 1):
                ngram = tuple(tokens[i:i+self.n])
                context = ngram[:-1]
                self.ngram_counts[ngram] += 1
                self.context_counts[context] += 1


    def predict_next(self, context: str, top_k: int=5):
        probabilities = {}

        for word in self.vocab:
            probabilities[word] = self.get_probability(context, word)

        return sorted(probabilities.items(), key=lambda x: x[1], reverse=True)[:top_k]

    def get_probability(self, context: str, word: str) -> float:
        context_tokens = tuple(tokenize_text(context))
        ngram = context_tokens + (word,)
        if ngram not in self.ngram_counts:
            return 0.0
        context_count = self.context_counts[context_tokens]
        if context_count == 0:
            return 0.0
        return self.ngram_counts[ngram] / context_count
