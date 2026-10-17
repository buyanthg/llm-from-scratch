import numpy as np


def tokenize_text(text):
    return text.lower().split()


class Tfidf:
    def __init__(self):
        self.vocab = {}
        self.vocab_size = 0
        self.df = {}
        self.corpus_size = 0

    def train(self, corpus: list[str]) -> None:
        self.corpus_size = len(corpus)
        for text in corpus:
            text = tokenize_text(text)
            for word in set(text):
                if word not in self.vocab:
                    self.vocab[word] = self.vocab_size
                    self.df[word] = 0
                    self.vocab_size += 1
                self.df[word] += 1

    def predict(self, text: str) -> np.ndarray:
        text = tokenize_text(text)
        emb = np.zeros(self.vocab_size)
        for word in set(text):
            if word not in self.vocab:
                continue
            emb[self.vocab[word]] = self.tf(word, text) * self.idf(word)
        return emb

    def fit_transform(self, corpus: list[str]) -> np.ndarray:
        self.vocab = {}
        self.vocab_size = 0
        self.df = {}
        self.corpus_size = 0
        self.train(corpus)

        return self.transform(corpus)

    def transform(self, corpus: list[str]) -> np.ndarray:
        emb = np.zeros((len(corpus), self.vocab_size))
        for i, text in enumerate(corpus):
            emb[i] = self.predict(text)
        return emb

    @staticmethod
    def tf(word, text):
        text = tokenize_text(text)
        return text.count(word) / len(text)

    def idf(self, word):
        return np.log(self.corpus_size / self.df[word])
