from dataclasses import asdict, dataclass

import torch
from torch import nn
from transformers import AutoModel
from transformers.modeling_outputs import SequenceClassifierOutput


@dataclass
class RCBTConfig:
    cnn_channels: int = 128
    kernel_sizes: tuple[int, ...] = (3, 4, 5)
    lstm_hidden_size: int = 320
    transformer_layers: int = 2
    transformer_heads: int = 8
    transformer_ff_size: int = 2048
    dropout: float = 0.1
    num_labels: int = 2

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, values: dict) -> "RCBTConfig":
        values = dict(values)
        values["kernel_sizes"] = tuple(values["kernel_sizes"])
        return cls(**values)


class RCBTForSequenceClassification(nn.Module):
    def __init__(self, encoder: nn.Module, config: RCBTConfig) -> None:
        super().__init__()
        self.encoder = encoder
        self.config = config
        hidden_size = encoder.config.hidden_size
        self.convs = nn.ModuleList([
            nn.Conv1d(hidden_size, config.cnn_channels, kernel_size)
            for kernel_size in config.kernel_sizes
        ])
        sequence_size = config.cnn_channels * len(config.kernel_sizes)
        self.feature_norm = nn.LayerNorm(sequence_size)
        self.lstm = nn.LSTM(
            input_size=sequence_size,
            hidden_size=config.lstm_hidden_size,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )
        transformer_size = config.lstm_hidden_size * 2
        layer = nn.TransformerEncoderLayer(
            d_model=transformer_size,
            nhead=config.transformer_heads,
            dim_feedforward=config.transformer_ff_size,
            dropout=config.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers=config.transformer_layers, enable_nested_tensor=False)
        self.output_norm = nn.LayerNorm(transformer_size)
        self.classifier = nn.Sequential(
            nn.Dropout(config.dropout),
            nn.Linear(transformer_size, config.num_labels),
        )

    @classmethod
    def from_pretrained(cls, encoder_name: str, config: RCBTConfig) -> "RCBTForSequenceClassification":
        return cls(AutoModel.from_pretrained(encoder_name), config)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: torch.Tensor | None = None,
    ) -> SequenceClassifierOutput:
        embeddings = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
        ).last_hidden_state
        embeddings = embeddings * attention_mask.unsqueeze(-1).to(embeddings.dtype)
        convolution_input = embeddings.transpose(1, 2)
        features = torch.cat([
            torch.nn.functional.gelu(conv(torch.nn.functional.pad(
                convolution_input,
                (kernel_size // 2, (kernel_size - 1) // 2),
            )))
            for conv, kernel_size in zip(self.convs, self.config.kernel_sizes)
        ], dim=1).transpose(1, 2)
        features = self.feature_norm(features)
        lengths = attention_mask.sum(dim=1).cpu()
        packed = nn.utils.rnn.pack_padded_sequence(
            features,
            lengths,
            batch_first=True,
            enforce_sorted=False,
        )
        packed_sequence, _ = self.lstm(packed)
        sequence, _ = nn.utils.rnn.pad_packed_sequence(
            packed_sequence,
            batch_first=True,
            total_length=features.size(1),
        )
        padding_mask = ~attention_mask.bool()
        sequence = self.transformer(sequence, src_key_padding_mask=padding_mask)
        sequence = self.output_norm(sequence)
        weights = attention_mask.unsqueeze(-1).to(sequence.dtype)
        pooled = (sequence * weights).sum(dim=1) / weights.sum(dim=1).clamp(min=1)
        logits = self.classifier(pooled)
        loss = None
        if labels is not None:
            loss = nn.functional.cross_entropy(logits, labels)
        return SequenceClassifierOutput(loss=loss, logits=logits)
