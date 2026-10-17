import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader

import re
import numpy as np


def tokenize_text(text: str) -> list[str]:
    return re.findall(r"[\w']+", text.lower())


class CBOWModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.linear = nn.Linear(embedding_dim, vocab_size)

    def forward(self, context):
        embeddings = self.embedding(context)
        hidden = embeddings.mean(dim=1)
        return self.linear(hidden)


class CBOW:
    def __init__(self, embed_dim: int = 100):
        self.embed_dim = embed_dim
        self.vocab = {"<UNK>": 0}
        self.vocab_size = 1
        self.model = None

    def build_vocab(self, corpus: list[str]) -> None:
        for text in corpus:
            text = tokenize_text(text)
            for word in set(text):
                if word not in self.vocab:
                    self.vocab[word] = self.vocab_size
                    self.vocab_size += 1

    def generate_training_data(self, corpus: list[str], window_size: int=2) -> tuple[torch.Tensor, torch.Tensor]:
        context_indices = []
        target_indices = []
        for text in corpus:
            tokens = tokenize_text(text)

            for target_pos in range(window_size, len(tokens) - window_size):
                target_word = tokens[target_pos]
                target_index = self.vocab.get(target_word, self.vocab["<UNK>"])
                target_indices.append(target_index)

                context_words = (tokens[target_pos-window_size:target_pos] +
                                 tokens[target_pos+1:target_pos + window_size+1])
                context = [self.vocab.get(word, self.vocab["<UNK>"]) for word in context_words]
                context_indices.append(context)

        target_indices = torch.tensor(target_indices, dtype=torch.long)
        context_indices = torch.tensor(context_indices, dtype=torch.long)
        return context_indices, target_indices

    def train(self, corpus, epochs: int=10, batch_size: int=128, window_size: int=2, lr=1e-3):
        self.vocab = {"<UNK>": 0}
        self.vocab_size = 1
        self.build_vocab(corpus)
        context_indices, target_indices = self.generate_training_data(corpus, window_size)

        dataset = TensorDataset(context_indices, target_indices)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        self.model = CBOWModel(self.vocab_size, self.embed_dim)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss()
        for epoch in range(epochs):
            self.model.train()
            losses = []

            for batch_contexts, batch_targets in dataloader:
                optimizer.zero_grad()

                predictions = self.model(batch_contexts)
                loss = criterion(predictions, batch_targets)
                losses.append(loss.item())
                loss.backward()
                optimizer.step()
            print(f"Epoch {epoch+1}, Loss: {np.mean(losses)}")

    def get_embedding(self, word):
        word_index = self.vocab.get(word, self.vocab["<UNK>"])
        return self.model.embedding.weight[word_index].detach()
