"""The existing LSTM/attention/head, with explicit dimensions for deployment."""
import torch
from torch import nn


class FaultLSTM(nn.Module):
    def __init__(self, n_features, hidden_dim=128, layers=2, n_classes=19, attention_pooling=True):
        super().__init__(); self.attention_pooling = attention_pooling
        self.lstm = nn.LSTM(n_features, hidden_dim, layers, batch_first=True, dropout=.3 if layers > 1 else 0.)
        self.attention = nn.Linear(hidden_dim, 1)
        self.head = nn.Sequential(nn.Linear(hidden_dim, 64), nn.ReLU(), nn.Dropout(.3), nn.Linear(64, n_classes))

    def forward(self, x, return_attention=False):
        sequence, _ = self.lstm(x)
        if self.attention_pooling:
            weights = torch.softmax(self.attention(torch.tanh(sequence)).squeeze(-1), dim=1)
        else:
            weights = torch.full((x.size(0), x.size(1)), 1./x.size(1), device=x.device, dtype=x.dtype)
        logits = self.head(torch.sum(sequence*weights.unsqueeze(-1), dim=1))
        return (logits, weights) if return_attention else logits
