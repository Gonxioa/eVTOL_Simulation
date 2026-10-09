"""Causal windows, observation-only predictions, persistence alarms and replay."""
from collections import deque
import numpy as np
from .features import extract, FEATURE_SETS


class CausalWindow:
    def __init__(self, settings):
        self.settings = dict(settings); self.feature_set = settings['feature_set']
        if self.feature_set not in FEATURE_SETS or settings['window'] < 2 or settings['step'] < 1 or settings['sample_hz'] <= 0:
            raise ValueError('Invalid causal window configuration')
        self.values = deque(maxlen=settings['window']); self.times = deque(maxlen=settings['window'])
        self.previous_time = None; self.contiguous_count = 0; self.resets = 0

    def push(self, packet):
        if self.previous_time is not None and packet.timestamp <= self.previous_time:
            raise ValueError('Diagnostic packet time must strictly increase')
        gap = self.previous_time is not None and abs(packet.timestamp-self.previous_time-1/self.settings['sample_hz']) > self.settings['gap_tolerance_s']
        self.previous_time = packet.timestamp
        if gap or not packet.valid:
            self.values.clear(); self.times.clear(); self.contiguous_count = 0; self.resets += 1
        if not packet.valid:
            return None
        self.values.append(extract(packet, self.feature_set)); self.times.append(packet.timestamp)
        self.contiguous_count += 1
        if len(self.values) < self.settings['window'] or (self.contiguous_count-self.settings['window']) % self.settings['step']:
            return None
        return np.stack(self.values), np.array(self.times)


def causal_windows(packets, settings):
    buffer = CausalWindow(settings); windows = []; times = []
    for packet in packets:
        window = buffer.push(packet)
        if window is not None:
            windows.append(window[0]); times.append(window[1])
    return (np.stack(windows) if windows else np.empty((0, settings['window'], len(FEATURE_SETS[settings['feature_set']])), dtype=np.float32),
            np.stack(times) if times else np.empty((0, settings['window'])))


class ContinuousDiagnosis:
    def __init__(self, predictor, settings):
        self.predictor = predictor; self.settings = dict(settings); self.window = CausalWindow(settings)
        self.streak = 0; self.latched = False; self.last_alarm = -np.inf; self.previous_resets = 0
        if not 0 < settings['fault_probability_threshold'] <= 1 or settings['consecutive_windows'] < 1 or settings['cooldown_s'] < 0:
            raise ValueError('Invalid alarm design')

    def push(self, packet):
        candidate = self.window.push(packet)
        if self.window.resets != self.previous_resets:
            self.streak = 0; self.latched = False; self.previous_resets = self.window.resets
        if not packet.valid:
            return {'time': packet.timestamp, 'valid': False, 'reason': 'invalid_observation',
                    'predicted_label': None, 'probabilities': None, 'alarm': False}
        if candidate is None:
            return None
        values, times = candidate
        probabilities = np.asarray(self.predictor(values), dtype=float)
        if probabilities.shape != (19,) or not np.isfinite(probabilities).all() or np.any(probabilities < 0) or not np.isclose(probabilities.sum(), 1, atol=1e-6):
            raise ValueError('Predictor must return 19 finite normalized class probabilities')
        fault_probability = 1-float(probabilities[0]); high = fault_probability >= self.settings['fault_probability_threshold']
        if high:
            self.streak += 1
        else:
            self.streak = 0; self.latched = False
        alarm = bool(high and self.streak >= self.settings['consecutive_windows'] and not self.latched and packet.timestamp-self.last_alarm >= self.settings['cooldown_s'])
        if alarm:
            self.last_alarm = packet.timestamp; self.latched = True
        return {'time': packet.timestamp, 'window_start_s': float(times[0]), 'window_end_s': float(times[-1]),
                'valid': True, 'predicted_label': int(np.argmax(probabilities)),
                'probabilities': probabilities.tolist(), 'fault_probability': fault_probability,
                'alarm': alarm, 'alarm_motor_severity_label': int(1+np.argmax(probabilities[1:])) if alarm else None,
                'contiguous_history_samples': self.window.contiguous_count}


def replay(packets, predictor, settings):
    diagnosis = ContinuousDiagnosis(predictor, settings); result = []
    for packet in packets:
        prediction = diagnosis.push(packet)
        if prediction is not None:
            result.append(prediction)
    return result


def offline_labels(times, label, fault_start=None, fault_end=None):
    """Offline only. -1 marks windows spanning an injection/recovery boundary."""
    if label == 0 or fault_start is None:
        return np.zeros(len(times), dtype=np.int64)
    active = times >= fault_start
    if fault_end is not None:
        active &= times < fault_end
    return np.where(active.all(axis=1), label, np.where((~active).all(axis=1), 0, -1)).astype(np.int64)
