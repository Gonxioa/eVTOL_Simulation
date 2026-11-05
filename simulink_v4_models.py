"""One fixed CNN control and the unchanged V3 LSTM backbone."""
import numpy as np
import torch
from torch import nn
import classify


class FixedCNN(nn.Module):
    def __init__(self, config, features=12):
        super().__init__(); modules = []; previous = features
        for channels, kernel, stride in zip(config['cnn_channels'], config['cnn_kernels'], config['cnn_strides']):
            modules += [nn.Conv1d(previous, channels, kernel, stride, padding=kernel//2), nn.ReLU()]
            previous = channels
        self.encoder = nn.Sequential(*modules, nn.AdaptiveAvgPool1d(config['cnn_pool_bins']))
        self.head = nn.Sequential(nn.Flatten(), nn.Linear(previous*config['cnn_pool_bins'], 64),
                                  nn.ReLU(), nn.Dropout(config['dropout']), nn.Linear(64, 19))

    def forward(self, x): return self.head(self.encoder(x.transpose(1, 2)))


def build(config, family, window, features=12):
    if family == 'cnn': return FixedCNN(config, features)
    if family != 'lstm': raise ValueError('Undeclared classifier family')
    classify.N_FEATURES = features; classify.N_CLASSES = 19; classify.SEQ_LEN = window
    classify.HIDDEN_DIM = config['lstm_hidden']; classify.N_LAYERS = config['lstm_layers']
    classify.ATTENTION_POOLING = True
    return classify.FaultClassifier(dual_head=False)


def statistical_features(x, sample_hz=100):
    """198 causal, within-window descriptors; no label/time/source channels."""
    x = np.asarray(x, np.float64); centered = x-x.mean(1, keepdims=True)
    energy = np.mean(centered**2, axis=1)
    lag = np.mean(centered[:, 1:]*centered[:, :-1], axis=1)/np.maximum(energy, 1e-12)
    blocks = [x.mean(1), x.std(1), x.min(1), x.max(1), np.sqrt(np.mean(x*x, axis=1)),
              np.abs(np.diff(x, axis=1)).mean(1), lag]
    scale = np.sqrt(np.maximum(energy, 1e-12)); cov = np.einsum('ntc,ntd->ncd', centered, centered)/x.shape[1]
    corr = cov/np.maximum(scale[:, :, None]*scale[:, None, :], 1e-12)
    i, j = np.triu_indices(x.shape[2], k=1); blocks.append(corr[:, i, j])
    # Integrated one-sided periodogram bands, normalized by fs*N.
    psd = np.abs(np.fft.rfft(centered, axis=1))**2/(sample_hz*x.shape[1])
    if x.shape[1] % 2 == 0: psd[:, 1:-1] *= 2
    else: psd[:, 1:] *= 2
    frequency = np.fft.rfftfreq(x.shape[1], 1/sample_hz); df = sample_hz/x.shape[1]
    for low, high in ((0, 1), (1, 5), (5, 15), (15, 50.000001)):
        blocks.append(psd[:, (frequency >= low) & (frequency < high)].sum(1)*df)
    result = np.concatenate(blocks, axis=1).astype(np.float32)
    if not np.isfinite(result).all(): raise ValueError('Nonfinite statistical descriptor')
    return result
