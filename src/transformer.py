import torch
import torch.nn as nn

import math


class Transformer(nn.Module):
    def __init__(self, src_vocab_size, tgt_vocab_size, d_model, num_heads, d_ff, num_encoder_layers, num_decoder_layers):
        super().__init__()

        self.encoder = Encoder(src_vocab_size, d_model, num_heads, d_ff, num_encoder_layers)
        self.decoder = Decoder(tgt_vocab_size, d_model, num_heads, d_ff, num_decoder_layers)

        self.projection = nn.Linear(d_model, tgt_vocab_size)

    def forward(self, src, tgt):
        context = self.encoder(src)
        output = self.decoder(tgt, context)

        logits = self.projection(output)

        return logits


class Decoder(nn.Module):
    def __init__(self, vocab_size, d_model, num_heads, d_ff, num_layers):
        super().__init__()

        self.embeddings = nn.Embedding(vocab_size, d_model)
        self.positional_encoding = PositionalEncoding(d_model, max_len=5000)

        self.layers = nn.ModuleList([DecoderBlock(d_model, num_heads, d_ff) for _ in range(num_layers)])

    def forward(self, X, context):
        X = self.embeddings(X)
        X = self.positional_encoding(X)

        for layer in self.layers:
            X = layer(X, context)

        return X


class DecoderBlock(nn.Module):
    def __init__(self, d_model, num_heads, d_ff):
        super().__init__()
        self.multihead_attention = MultiHeadAttention(d_model, num_heads)
        self.add_norm1 = AddNorm(d_model)

        self.cross_attention = MultiHeadAttention(d_model, num_heads)
        self.add_norm2 = AddNorm(d_model)

        self.feedforward = FeedForward(d_model, d_ff)
        self.add_norm3 = AddNorm(d_model)

    def forward(self, X, context):
        seq_len = X.size(1)
        mask = causal_mask(seq_len, X.device)

        # Masked Self-Attention
        attention_output = self.multihead_attention(X, mask=mask)
        norm1_output = self.add_norm1(X, attention_output)

        # Cross-Attention
        cross_output = self.cross_attention(norm1_output, context)
        norm2_output = self.add_norm2(norm1_output, cross_output)

        # Feed-Forward
        ffn_output = self.feedforward(norm2_output)
        output = self.add_norm3(norm2_output, ffn_output)
        return output


class Encoder(nn.Module):
    def __init__(self, vocab_size, d_model, num_heads, d_ff, num_layers):
        super().__init__()

        self.embeddings = nn.Embedding(vocab_size, d_model)
        self.positional_encoding = PositionalEncoding(d_model, max_len=5000)

        self.layers = nn.ModuleList([EncoderBlock(d_model, num_heads, d_ff) for _ in range(num_layers)])

    def forward(self, X):
        X = self.embeddings(X)
        X = self.positional_encoding(X)

        for layer in self.layers:
            X = layer(X)

        return X


class EncoderBlock(nn.Module):
    def __init__(self, d_model, num_heads, d_ff):
        super().__init__()
        self.multihead_attention = MultiHeadAttention(d_model, num_heads)
        self.add_norm1 = AddNorm(d_model)
        self.feedforward = FeedForward(d_model, d_ff)
        self.add_norm2 = AddNorm(d_model)

    def forward(self, X, mask=None):
        attention_output = self.multihead_attention(X, mask)

        norm1_output = self.add_norm1(X, attention_output)
        ffn_output = self.feedforward(norm1_output)

        output = self.add_norm2(norm1_output, ffn_output)

        return output


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        assert d_model % num_heads == 0
        super().__init__()

        self.heads = nn.ModuleList([
            SelfAttention(d_model, d_model // num_heads, d_model // num_heads)
            for _ in range(num_heads)
        ])
        self.W_O = nn.Linear(d_model, d_model)

    def forward(self, query, context=None, mask=None):
        outputs = [head(query, context, mask)[0] for head in self.heads]
        output = torch.cat(outputs, dim=-1)
        output = self.W_O(output)
        return output


class SelfAttention(nn.Module):
    def __init__(self, d_model, d_k, d_v):
        # d_model — размерность входного представления токена
        # d_k — размерность Query и Key
        # d_v — размерность Value и выходного представления.
        super().__init__()
        self.d_model = d_model
        self.d_k = d_k
        self.d_v = d_v
        self.W_Q = nn.Linear(d_model, d_k)
        self.W_K = nn.Linear(d_model, d_k)
        self.W_V = nn.Linear(d_model, d_v)

        self.attention = ScaleDotProductAttention()

    def forward(self, query, context=None, mask=None):
        if context is None:
            context = query
        Q = self.W_Q(query)
        K = self.W_K(context)
        V = self.W_V(context)
        output, attention_weights = self.attention(Q, K, V, mask)
        return output, attention_weights


class ScaleDotProductAttention(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, Q, K, V, mask=None):
        # Q (batch, seq_len, d_k)
        # K (batch, seq_len, d_k)
        # V (batch, seq_len, d_v)

        # вычисляем оценки соответствия каждого Query каждому Key
        scores = Q @ K.transpose(-2, -1)
        # масштабируем оценки, чтобы они не становились слишком большими.
        scores /= math.sqrt(Q.shape[-1])

        if mask is not None:
            scores = scores.masked_fill(~mask, float("-inf"))

        # нормируем оценки отдельно для каждого токена запроса
        attention_weights = torch.softmax(scores, dim=-1)
        # получаем взвешенную сумму векторов значений
        output = attention_weights @ V
        return output, attention_weights


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        positions = torch.arange(max_len).unsqueeze(1)

        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(positions * div_term)
        pe[:, 1::2] = torch.cos(positions * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, X):
        return X + self.pe[:, :X.size(1), :]


class AddNorm(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)

    def forward(self, X, sublayer_output):
        return self.norm(sublayer_output + X)


class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff):
        super().__init__()

        self.fc1 = nn.Linear(d_model, d_ff)
        self.activation = nn.ReLU()
        self.fc2 = nn.Linear(d_ff, d_model)

    def forward(self, X):
        output = self.activation(self.fc1(X))
        output = self.fc2(output)
        return output


def causal_mask(seq_len, device=None):
    return torch.tril(torch.ones(seq_len, seq_len, dtype=torch.bool, device=device))
