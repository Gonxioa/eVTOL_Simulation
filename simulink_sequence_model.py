"""Causal whole-flight encoder with persistent multi-scale EMA, recurrent and history-pool state."""
from dataclasses import dataclass
import math

import torch
from torch import nn


@dataclass
class Memory:
    hidden: tuple
    total: torch.Tensor
    maximum: torch.Tensor
    count: torch.Tensor
    ema: torch.Tensor | None = None


@dataclass
class CNNMemory:
    caches: tuple
    total: torch.Tensor
    maximum: torch.Tensor
    count: torch.Tensor
    hidden: tuple = ()
    ema: torch.Tensor | None = None


class CausalMultiScaleEMA(nn.Module):
    """Exact, O(1)-state causal multi-scale EMA filter bank in float64.

    Produces [x, ema_0, ..., ema_{S-1}, x - ema_{S-1}], giving the encoder
    instantaneous spikes, denoised rapid/sustained means, and baseline deviations
    with bit-consistent equivalence across arbitrary arrival chunk sizes.
    """
    def __init__(self, channels, taus_s=(0.3, 2.0), sample_hz=100.0, block_size=64):
        super().__init__()
        if not taus_s or min(taus_s) <= 0 or sample_hz <= 0 or block_size < 1:
            raise ValueError("Positive EMA timescales, sample rate, and block size required")
        self.channels = int(channels)
        self.taus_s = tuple(float(t) for t in taus_s)
        self.sample_hz = float(sample_hz)
        self.block_size = int(block_size)
        self.out_channels = self.channels * (2 + len(self.taus_s))

        alphas = [1.0 - math.exp(-1.0 / (self.sample_hz * tau)) for tau in self.taus_s]
        idx = torch.arange(self.block_size, dtype=torch.float64)
        diff = idx[:, None] - idx[None, :]
        tril = diff >= 0
        p_list, w_list = [], []
        for a in alphas:
            g = 1.0 - a
            p_list.append(torch.pow(torch.tensor(g, dtype=torch.float64), idx + 1.0))
            w_list.append(torch.where(
                tril,
                a * torch.pow(torch.tensor(g, dtype=torch.float64), diff.clamp_min(0.0)),
                torch.zeros_like(diff),
            ))
        self.register_buffer("p_full", torch.stack(p_list, dim=0))  # [S, B_k]
        self.register_buffer("w_full", torch.stack(w_list, dim=0))  # [S, B_k, B_k]

    def forward(self, x, ema_state=None, lengths=None):
        batch, steps, channels = x.shape
        if channels != self.channels:
            raise ValueError(f"Expected {self.channels} input channels, got {channels}")
        if lengths is None:
            lengths = torch.full((batch,), steps, device=x.device, dtype=torch.long)
        n_scales = len(self.taus_s)
        x64 = x.to(torch.float64)
        if ema_state is None:
            state = x64[:, 0:1, :].expand(batch, n_scales, channels).clone()
        else:
            state = ema_state.to(device=x.device, dtype=torch.float64)
        blocks = []
        for start in range(0, steps, self.block_size):
            end = min(start + self.block_size, steps)
            length = end - start
            u = x64[:, start:end, :]
            p = self.p_full[:, :length]
            w = self.w_full[:, :length, :length]
            block_ema = (
                p[None, :, :, None] * state[:, :, None, :]
                + torch.einsum("stl,blc->bstc", w, u)
            )
            blocks.append(block_ema)
            state = block_ema[:, :, -1, :]
        ema_all = torch.cat(blocks, dim=2)  # [B, S, steps, C]
        ix = torch.arange(batch, device=x.device)
        next_ema = ema_all[ix, :, lengths - 1, :]  # float64 [B, S, C]
        ema_flat = ema_all.permute(0, 2, 1, 3).reshape(batch, steps, n_scales * channels).to(x.dtype)
        delta_slow = x - ema_all[:, -1, :, :].to(x.dtype)
        features = torch.cat((x, ema_flat, delta_slow), dim=2)
        return features, next_ema


class FlightClassifier(nn.Module):
    """Causal LSTM whole-flight classifier with optional multi-scale causal EMA front-end."""
    def __init__(self, hidden=64, layers=2, dropout=.3, input_dim=12,
                 use_causal_ema=False, ema_taus_s=(0.3, 2.0), sample_hz=100.0):
        super().__init__()
        self.hidden_size = hidden
        self.input_dim = int(input_dim)
        self.ema = CausalMultiScaleEMA(self.input_dim, ema_taus_s, sample_hz) if use_causal_ema else None
        enc_in = self.ema.out_channels if self.ema is not None else self.input_dim
        self.input_proj = (
            nn.Sequential(nn.Linear(enc_in, hidden), nn.LayerNorm(hidden), nn.ReLU())
            if use_causal_ema else None
        )
        lstm_in = hidden if self.input_proj is not None else enc_in
        self.lstm = nn.LSTM(lstm_in, hidden, layers, batch_first=True,
                            dropout=dropout if layers > 1 else 0)
        self.current_head = nn.Sequential(nn.Linear(hidden, 64), nn.ReLU(),
                                          nn.Dropout(dropout), nn.Linear(64, 19))
        self.record_head = nn.Sequential(nn.Linear(3*hidden, 64), nn.ReLU(),
                                         nn.Dropout(dropout), nn.Linear(64, 19))

    def forward(self, x, memory=None, lengths=None):
        if x.ndim != 3 or x.shape[2] != self.input_dim or not x.shape[1]:
            raise ValueError(f"Input must be [flights, new_time_samples, {self.input_dim}]")
        batch, steps, _ = x.shape
        if lengths is None:
            lengths = torch.full((batch,), steps, device=x.device, dtype=torch.long)
        if lengths.shape != (batch,) or bool(((lengths < 1) | (lengths > steps)).any()):
            raise ValueError("Invalid flight lengths")
        next_ema = x.new_empty((batch, 0, 0), dtype=torch.float64)
        feat = x
        if self.ema is not None:
            ema_in = None if (memory is None or memory.ema is None or memory.ema.numel() == 0) else memory.ema
            feat, next_ema = self.ema(feat, ema_in, lengths)
            feat = self.input_proj(feat)
        initial = None if memory is None else memory.hidden
        if bool((lengths != steps).any()):
            packed = nn.utils.rnn.pack_padded_sequence(feat, lengths.detach().cpu(),
                                                       batch_first=True, enforce_sorted=False)
            encoded, hidden = self.lstm(packed, initial)
            sequence, _ = nn.utils.rnn.pad_packed_sequence(encoded, batch_first=True, total_length=steps)
        else:
            sequence, hidden = self.lstm(feat, initial)
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
        state = Memory(hidden, running_total[ix, last], running_max[ix, last], running_count[ix, last], next_ema)
        return dict(current_logits=current_logits, prefix_logits=record_logits,
                    record_logits=record_logits[ix, last], memory=state)


class CausalFlightCNN(nn.Module):
    """Causal dilated 1D TCN with optional multi-scale EMA front-end and persistent GRU memory.

    Convolution caches preserve local transient context across chunks.
    Optional GRU carries long-range post-transient fault memory into current_head
    so opposite-motor symmetry after 3s does not erase onset localization.
    """
    def __init__(self, hidden=64, kernels=(5, 5, 5, 5), dilations=(1, 2, 4, 8), dropout=.3,
                 input_dim=12, use_causal_ema=False, ema_taus_s=(0.3, 2.0), sample_hz=100.0,
                 use_temporal_gru=False):
        super().__init__()
        if len(kernels) != len(dilations) or not kernels or min(*kernels, *dilations) < 1:
            raise ValueError("Positive CNN kernels and matching dilations are required")
        self.hidden_size = hidden
        self.input_dim = int(input_dim)
        self.use_temporal_gru = bool(use_temporal_gru)
        self.ema = CausalMultiScaleEMA(self.input_dim, ema_taus_s, sample_hz) if use_causal_ema else None
        enc_in = self.ema.out_channels if self.ema is not None else self.input_dim
        self.contexts = tuple((k-1)*d for k, d in zip(kernels, dilations))
        self.receptive_field_samples = 1+sum(self.contexts)
        channels = [enc_in]+[hidden]*len(kernels)
        self.convs = nn.ModuleList(nn.Conv1d(channels[i], hidden, k, dilation=d)
                                   for i, (k, d) in enumerate(zip(kernels, dilations)))
        self.dropout = nn.Dropout(dropout)
        if self.use_temporal_gru:
            self.gru_norm = nn.LayerNorm(hidden)
            self.gru = nn.GRU(hidden, hidden, num_layers=1, batch_first=True)
        else:
            self.gru_norm = None
            self.gru = None
        self.current_head = nn.Sequential(nn.Linear(hidden, 64), nn.ReLU(),
                                          nn.Dropout(dropout), nn.Linear(64, 19))
        self.record_head = nn.Sequential(nn.Linear(3*hidden, 64), nn.ReLU(),
                                         nn.Dropout(dropout), nn.Linear(64, 19))

    def forward(self, x, memory=None, lengths=None):
        if x.ndim != 3 or x.shape[2] != self.input_dim or not x.shape[1]:
            raise ValueError(f"Input must be [flights, new_time_samples, {self.input_dim}]")
        batch, steps, _ = x.shape
        if lengths is None:
            lengths = torch.full((batch,), steps, device=x.device, dtype=torch.long)
        if lengths.shape != (batch,) or bool(((lengths < 1) | (lengths > steps)).any()):
            raise ValueError("Invalid flight lengths")
        next_ema = x.new_empty((batch, 0, 0), dtype=torch.float64)
        feat = x
        if self.ema is not None:
            ema_in = None if (memory is None or memory.ema is None or memory.ema.numel() == 0) else memory.ema
            feat, next_ema = self.ema(feat, ema_in, lengths)
        value, caches = feat.transpose(1, 2), []
        for i, (conv, context) in enumerate(zip(self.convs, self.contexts)):
            previous = value.new_zeros((batch, value.shape[1], context)) if memory is None else memory.caches[i]
            joined = torch.cat((previous, value), dim=2)
            positions = lengths[:, None]+torch.arange(context, device=x.device)[None, :]
            caches.append(joined.gather(2, positions[:, None].expand(-1, joined.shape[1], -1)))
            out = self.dropout(torch.relu(conv(joined)))
            value = (value + out) if (i > 0 and value.shape[1] == out.shape[1]) else out
        conv_sequence = value.transpose(1, 2)
        next_hidden = ()
        if self.use_temporal_gru:
            gru_in = self.gru_norm(conv_sequence)
            initial = None if (memory is None or not memory.hidden) else memory.hidden[0]
            if bool((lengths != steps).any()):
                packed = nn.utils.rnn.pack_padded_sequence(gru_in, lengths.detach().cpu(),
                                                           batch_first=True, enforce_sorted=False)
                encoded, h_n = self.gru(packed, initial)
                sequence, _ = nn.utils.rnn.pad_packed_sequence(encoded, batch_first=True, total_length=steps)
            else:
                sequence, h_n = self.gru(gru_in, initial)
            next_hidden = (h_n,)
        else:
            sequence = conv_sequence
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
        state = CNNMemory(tuple(caches), running_total[ix, last], running_max[ix, last],
                          running_count[ix, last], next_hidden, next_ema)
        return dict(current_logits=current_logits, prefix_logits=prefix_logits,
                    record_logits=prefix_logits[ix, last], memory=state)


def build_model(config):
    allow_tf32 = bool(config.get("allow_tf32", False))
    torch.backends.cudnn.allow_tf32 = allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = allow_tf32
    family = config.get("model", "lstm")
    feature_mode = config.get("feature_mode", "raw12")
    input_dim = int(config.get("input_dim", 25 if feature_mode == "phys25" else 12))
    use_causal_ema = bool(config.get("use_causal_ema", False))
    ema_taus_s = tuple(config.get("ema_taus_s", (0.3, 2.0)))
    sample_hz = float(config.get("sample_hz", 100.0))
    if family == "lstm":
        return FlightClassifier(config["hidden"], config["layers"], config["dropout"],
                                input_dim=input_dim, use_causal_ema=use_causal_ema,
                                ema_taus_s=ema_taus_s, sample_hz=sample_hz)
    if family == "cnn":
        return CausalFlightCNN(config["hidden"], config.get("cnn_kernels", (5, 5, 5, 5)),
                               config.get("cnn_dilations", (1, 2, 4, 8)), config["dropout"],
                               input_dim=input_dim, use_causal_ema=use_causal_ema,
                               ema_taus_s=ema_taus_s, sample_hz=sample_hz,
                               use_temporal_gru=bool(config.get("cnn_use_gru", False)))
    raise ValueError("Unknown whole-flight model: "+str(family))


def decision_mask(lengths, steps, period):
    if period < 1:
        raise ValueError("Decision period must be positive")
    positions = torch.arange(steps, device=lengths.device)[None, :]
    valid = positions < lengths[:, None]
    return valid & (((positions+1) % period == 0) | (positions == lengths[:, None]-1))


def loss_for_flights(output, lengths, current, seen, labels, period=10,
                     record_weight=1., prefix_weight=.25, current_weight=.25):
    """Whole-flight label is primary; causal prefix supervision is auxiliary."""
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