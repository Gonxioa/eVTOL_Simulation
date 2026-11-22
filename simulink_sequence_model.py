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


@dataclass
class CNNMemory:
    caches: tuple
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


class CausalFlightCNN(nn.Module):
    """Causal dilated 1D CNN with whole-flight cumulative feature pooling.

    Convolution caches preserve the encoder's context across new chunks.
    Mean/max history pools persist for the entire flight, not a fixed window.
    """
    def __init__(self, hidden=64, kernels=(5, 5, 5, 5), dilations=(1, 2, 4, 8), dropout=.3):
        super().__init__()
        if len(kernels) != len(dilations) or not kernels or min(*kernels, *dilations) < 1:
            raise ValueError("Positive CNN kernels and matching dilations are required")
        self.hidden_size = hidden
        self.contexts = tuple((k-1)*d for k, d in zip(kernels, dilations))
        self.receptive_field_samples = 1+sum(self.contexts)
        channels = [12]+[hidden]*len(kernels)
        self.convs = nn.ModuleList(nn.Conv1d(channels[i], hidden, k, dilation=d)
                                   for i, (k, d) in enumerate(zip(kernels, dilations)))
        self.dropout = nn.Dropout(dropout)
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
        value, caches = x.transpose(1, 2), []
        for i, (conv, context) in enumerate(zip(self.convs, self.contexts)):
            previous = value.new_zeros((batch, value.shape[1], context)) if memory is None else memory.caches[i]
            joined = torch.cat((previous, value), dim=2)
            positions = lengths[:, None]+torch.arange(context, device=x.device)[None, :]
            caches.append(joined.gather(2, positions[:, None].expand(-1, joined.shape[1], -1)))
            value = self.dropout(torch.relu(conv(joined)))
        sequence = value.transpose(1, 2)
        mask = torch.arange(steps, device=x.device)[None, :] < lengths[:, None]
        total = sequence.new_zeros((batch, self.hidden_size)) if memory is None else memory.total
        maximum = sequence.new_full((batch, self.hidden_size), -float("inf")) if memory is None else memory.maximum
        count = torch.zeros((batch, 1), device=x.device, dtype=torch.long) if memory is None else memory.count
        running_total = (sequence*mask[:, :, None]).cumsum(1)+total[:, None]
        running_max = torch.maximum(torch.cummax(sequence.masked_fill(~mask[:, :, None], -float("inf")), 1).values,
                                    maximum[:, None])
        running_count = mask.to(torch.long).cumsum(1)[:, :, None]+count[:, None]
        context = torch.cat((sequence, running_max, running_total/running_count.clamp_min(1)), dim=2)
        current_logits, prefix_logits = self.current_head(sequence), self.record_head(context)
        ix, last = torch.arange(batch, device=x.device), lengths-1
        state = CNNMemory(tuple(caches), running_total[ix, last], running_max[ix, last], running_count[ix, last])
        return dict(current_logits=current_logits, prefix_logits=prefix_logits,
                    record_logits=prefix_logits[ix, last], memory=state)


def build_model(config):
    # TF32 kernels can depend on batch/chunk geometry and noticeably change a
    # trained recurrent model's probabilities. Keep the online and batched
    # baseline paths in the same float32 arithmetic by default.
    allow_tf32 = bool(config.get("allow_tf32", False))
    torch.backends.cudnn.allow_tf32 = allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = allow_tf32
    family = config.get("model", "lstm")
    if family == "lstm":
        return FlightClassifier(config["hidden"], config["layers"], config["dropout"])
    if family == "cnn":
        return CausalFlightCNN(config["hidden"], config.get("cnn_kernels", (5, 5, 5, 5)),
                              config.get("cnn_dilations", (1, 2, 4, 8)), config["dropout"])
    raise ValueError("Unknown whole-flight model: "+str(family))


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
