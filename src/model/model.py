"""One shared BiLSTM and two jointly trained prediction heads."""

import gzip
from dataclasses import dataclass
from pathlib import Path

import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from src.model.crf import BIOCRF


@dataclass
class ModelConfig:
    vocab_size: int
    num_intents: int
    num_tags: int
    embedding_dim: int = 100
    hidden_dim: int = 128
    num_layers: int = 1
    dropout: float = 0.3
    freeze_embeddings: bool = False
    char_vocab_size: int = 0
    char_embedding_dim: int = 16
    char_channels: int = 64
    decoder: str = "softmax"
    bio_tags: list[str] | None = None
    word_dropout: float = 0.0
    intent_pooling: str = "mean_max"


class MaskedAttentionPooling(nn.Module):
    """Additive attention over real tokens; returns the pooled vector and weights."""

    def __init__(self, input_dim: int, attention_dim: int):
        super().__init__()
        self.score = nn.Sequential(
            nn.Linear(input_dim, attention_dim),
            nn.Tanh(),
            nn.Linear(attention_dim, 1, bias=False),
        )

    def forward(self, encoded, mask):
        scores = self.score(encoded).squeeze(-1).masked_fill(~mask, float("-inf"))
        weights = scores.softmax(dim=1)
        pooled = (encoded * weights.unsqueeze(-1)).sum(dim=1)
        return pooled, weights


class JointBiLSTM(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.embedding = nn.Embedding(config.vocab_size, config.embedding_dim, padding_idx=0)
        self.embedding.weight.requires_grad_(not config.freeze_embeddings)
        self.dropout = nn.Dropout(config.dropout)
        self.char_embedding = self.char_conv = None
        if config.char_vocab_size:
            self.char_embedding = nn.Embedding(
                config.char_vocab_size, config.char_embedding_dim, padding_idx=0
            )
            self.char_conv = nn.Conv1d(
                config.char_embedding_dim, config.char_channels, kernel_size=3, padding=1
            )
        encoder_input = config.embedding_dim + (
            config.char_channels if config.char_vocab_size else 0
        )
        self.encoder = nn.LSTM(
            encoder_input,
            config.hidden_dim,
            num_layers=config.num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=config.dropout if config.num_layers > 1 else 0.0,
        )
        if config.intent_pooling not in ("mean_max", "attention"):
            raise ValueError("intent_pooling must be mean_max or attention")
        self.intent_attention = None
        if config.intent_pooling == "attention":
            self.intent_attention = MaskedAttentionPooling(2 * config.hidden_dim, config.hidden_dim)
        intent_dim = (2 if self.intent_attention is not None else 4) * config.hidden_dim
        self.intent_head = nn.Linear(intent_dim, config.num_intents)
        self.slot_head = nn.Linear(2 * config.hidden_dim, config.num_tags)
        self.crf = None
        if config.decoder not in ("softmax", "constrained", "crf"):
            raise ValueError("decoder must be softmax, constrained, or crf")
        if config.decoder != "softmax":
            if config.bio_tags is None or len(config.bio_tags) != config.num_tags:
                raise ValueError("Constrained decoders require the complete BIO tag list")
            self.crf = BIOCRF(config.bio_tags, learn_transitions=config.decoder == "crf")

    def forward(self, input_ids, lengths, char_ids=None):
        word_ids = input_ids
        if self.training and self.config.word_dropout:
            dropped = (
                torch.rand(input_ids.shape, device=input_ids.device) < self.config.word_dropout
            ) & (input_ids > 1)
            word_ids = input_ids.masked_fill(dropped, 1)
        embeddings = self.embedding(word_ids)
        if self.char_embedding is not None:
            if char_ids is None:
                raise ValueError("Character model requires character IDs derived from utt")
            batch_size, words, characters = char_ids.shape
            flat_ids = char_ids.reshape(-1, characters)
            char_features = torch.relu(
                self.char_conv(self.char_embedding(flat_ids).transpose(1, 2))
            )
            char_mask = flat_ids != 0
            char_features = (
                char_features.masked_fill(~char_mask[:, None, :], -10000.0).max(-1).values
            )
            char_features = char_features.masked_fill(~char_mask.any(-1)[:, None], 0.0)
            embeddings = torch.cat(
                (embeddings, char_features.reshape(batch_size, words, -1)), dim=-1
            )
        packed = pack_padded_sequence(
            self.dropout(embeddings), lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        encoded, _ = self.encoder(packed)
        encoded, _ = pad_packed_sequence(encoded, batch_first=True, total_length=input_ids.shape[1])
        mask = (
            torch.arange(input_ids.shape[1], device=input_ids.device)[None, :]
            < lengths.to(input_ids.device)[:, None]
        )
        if self.intent_attention is None:
            mean = (encoded * mask.unsqueeze(-1)).sum(1) / lengths.to(encoded.device).unsqueeze(-1)
            maximum = encoded.masked_fill(~mask.unsqueeze(-1), float("-inf")).max(1).values
            pooled = torch.cat((mean, maximum), dim=-1)
        else:
            pooled, _ = self.intent_attention(encoded, mask)
        return (
            self.intent_head(self.dropout(pooled)),
            self.slot_head(self.dropout(encoded)),
        )

    def decode_tags(self, slot_logits, lengths):
        if self.crf is None:
            return slot_logits.argmax(-1).tolist()
        mask = (
            torch.arange(slot_logits.shape[1], device=slot_logits.device)[None, :]
            < lengths.to(slot_logits.device)[:, None]
        )
        return self.crf.decode(slot_logits, mask)


def parameter_counts(model: nn.Module) -> dict:
    result = {
        "total": sum(p.numel() for p in model.parameters()),
        "trainable": sum(p.numel() for p in model.parameters() if p.requires_grad),
    }
    if result["total"] > 20_000_000:
        raise ValueError(f"System exceeds the 20M total parameter cap: {result}")
    return result


def initialize_word_vectors(model: JointBiLSTM, vocab: dict, path: str | Path) -> dict:
    """Stream GloVe/word2vec-text vectors and retain Train vocabulary only.

    The resource file is not a model layer: only matched Train-vocabulary rows
    are copied into the counted embedding matrix. No external task examples.
    """
    path = Path(path)
    opener = gzip.open if path.suffix == ".gz" else open
    hits, seen = set(), set()
    with opener(path, "rt", encoding="utf-8") as stream, torch.no_grad():
        for line in stream:
            fields = line.rstrip().split()
            if len(fields) == 2 and all(v.isdigit() for v in fields):
                continue  # Optional word2vec text header.
            if not fields or fields[0] not in vocab or vocab[fields[0]] < 2:
                continue
            word = fields[0]
            if word in seen:
                continue
            seen.add(word)
            if len(fields) != model.config.embedding_dim + 1:
                raise ValueError(f"Vector dimension mismatch for {word!r}")
            vector = torch.tensor([float(value) for value in fields[1:]])
            if not torch.isfinite(vector).all():
                raise ValueError(f"Non-finite vector for {word!r}")
            model.embedding.weight[vocab[word]].copy_(vector)
            hits.add(word)
        model.embedding.weight[0].zero_()
    if not hits:
        raise ValueError("No Train-vocabulary words matched the word-vector file")
    return {
        "matched": len(hits),
        "eligible": len(vocab) - 2,
        "coverage_pct": 100 * len(hits) / (len(vocab) - 2),
    }
