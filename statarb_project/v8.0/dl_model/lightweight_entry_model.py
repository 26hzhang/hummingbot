from typing import Optional

import torch
import torch.nn as nn


class LightweightEntryQualityModel(nn.Module):
    """GRU-based lightweight binary classifier for entry quality."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 64,
        num_layers: int = 2,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.gru = nn.GRU(
            input_dim,
            hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),  # Output probability for binary classification
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T, C]
        out, h_n = self.gru(x)
        # use last hidden state from top layer
        h_last = h_n[-1]  # [B, H]
        y = self.head(h_last)  # [B, 1] with sigmoid activation
        return y

