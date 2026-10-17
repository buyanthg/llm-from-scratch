import numpy as np

def tokenize_text(text):
    return text.lower().split()


class BoW:
    def __init__(self):
        self.vocab = {}
        self.vocab_size = 0

    def train(self, corpus: list[str]) -> None:
        for text in corpus:
            text = tokenize_text(text)
            for word in text:
                if word not in self.vocab:
                    self.vocab[word] = self.vocab_size
                    self.vocab_size += 1

    def predict(self, text: str) -> np.ndarray:
        text = tokenize_text(text)
        emb = np.zeros(len(self.vocab))
        for word in text:
            if word not in self.vocab:
                continue
            emb[self.vocab[word]] += 1
        return emb

    def fit_transform(self, corpus: list[str]) -> np.ndarray:
        self.train(corpus)
        emb = np.zeros((len(corpus), len(self.vocab)), dtype=np.int32)
        for i, text in enumerate(corpus):
            emb[i] = self.predict(text)
        return emb
