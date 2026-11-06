"""Causal whole-flight encoder with persistent recurrent and history-pool state."""
from dataclasses import dataclass

import torch
from torch import nn


@dataclass
class Memory:
    hidden: tuple
    total: torch.Tensor
    maximum: torch.Tensor
    count: torch.Tensor


class FlightClassifier(nn.Module):
    """Two outputs: current state and fault occurring anywhere in the prefix.

    forward() may consume a complete trajectory or the next contiguous chunk.
    The returned memory must be carried between chunks of the SAME flight.
    Prefix pooling is causal, unlike whole-sequence retrospective attention.
    """
    def __init__(self, hidden=64, layers=2, dropout=.3):
        super().__init__()
        self.hidden_size = hidden
        self.lstm = nn.LSTM(12, hidden, layers, batch_first=True,
                            dropout=dropout if layers > 1 else 0)
        self.current_head = nn.Sequential(nn.Linear(hidden, 64), nn.ReLU(),
                                          nn.Dropout(dropout), nn.Linear(64, 19))
        self.record_head = nn.Sequential(nn.Linear(3*hidden, 64), nn.ReLU(),
                                         nn.Dropout(dropout), nn.Linear(64, 19))

    def forward(self, x, memory=None, lengths=None):
        if x.ndim != 3 or x.shape[2] != 12 or not x.shape[1]:
            raise ValueError("Input must be [flights, new_time_samples, 12]")
        batch, steps, _ = x.shape
        if lengths is None:
            lengths = torch.full((batch,), steps, device=x.device, dtype=torch.long)
        if lengths.shape != (batch,) or bool(((lengths < 1) | (lengths > steps)).any()):
            raise ValueError("Invalid flight lengths")
        initial = None if memory is None else memory.hidden
        if bool((lengths != steps).any()):
            packed = nn.utils.rnn.pack_padded_sequence(x, lengths.detach().cpu(),
                                                       batch_first=True, enforce_sorted=False)
            encoded, hidden = self.lstm(packed, initial)
            sequence, _ = nn.utils.rnn.pad_packed_sequence(encoded, batch_first=True, total_length=steps)
        else:
            sequence, hidden = self.lstm(x, initial)
        mask = torch.arange(steps, device=x.device)[None, :] < lengths[:, None]
        masked = sequence*mask[:, :, None]
        if memory is None:
            total = sequence.new_zeros((batch, self.hidden_size))
            maximum = sequence.new_full((batch, self.hidden_size), -float("inf"))
            count = torch.zeros((batch, 1), device=x.device, dtype=torch.long)
        else:
            total, maximum, count = memory.total, memory.maximum, memory.count
        running_total = masked.cumsum(1) + total[:, None]
        running_max = torch.cummax(sequence.masked_fill(~mask[:, :, None], -float("inf")), 1).values
        running_max = torch.maximum(running_max, maximum[:, None])
        running_count = mask.to(torch.long).cumsum(1)[:, :, None] + count[:, None]
        context = torch.cat((sequence, running_max, running_total/running_count.clamp_min(1)), dim=2)
        current_logits = self.current_head(sequence)
        record_logits = self.record_head(context)
        last = lengths - 1
        ix = torch.arange(batch, device=x.device)
        state = Memory(hidden, running_total[ix, last], running_max[ix, last], running_count[ix, last])
        return dict(current_logits=current_logits, prefix_logits=record_logits,
                    record_logits=record_logits[ix, last], memory=state)


def decision_mask(lengths, steps, period):
    if period < 1:
        raise ValueError("Decision period must be positive")
    positions = torch.arange(steps, device=lengths.device)[None, :]
    valid = positions < lengths[:, None]
    return valid & (((positions+1) % period == 0) | (positions == lengths[:, None]-1))


def loss_for_flights(output, lengths, current, seen, labels, period=10,
                     record_weight=1., prefix_weight=.25, current_weight=.25):
    """Whole-flight label is primary; causal prefix supervision is auxiliary.

    Targets before onset are healthy. After a repaired fault, the current-state
    target is healthy while the record/prefix target still records its history.
    Each flight is weighted equally, irrespective of its duration/padding.
    """
    record = nn.functional.cross_entropy(output["record_logits"], labels)
    mask = decision_mask(lengths, current.shape[1], period)

    def temporal(logits, target):
        values = nn.functional.cross_entropy(logits.transpose(1, 2), target, reduction="none")
        return ((values*mask).sum(1)/mask.sum(1).clamp_min(1)).mean()

    prefix = temporal(output["prefix_logits"], seen)
    state = temporal(output["current_logits"], current)
    total = record_weight*record + prefix_weight*prefix + current_weight*state
    return total, dict(record_CE=float(record.detach()), prefix_CE=float(prefix.detach()),
                       current_CE=float(state.detach()))
