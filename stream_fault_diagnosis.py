"""Example: receive raw sensor packets, align IMU causally, emit live decisions.

CSV iteration is only an example transport. Replace packets() with a serial,
socket or acquisition iterator yielding (new_12d_measurements, timestamps).
No fault truth, controller command or future row enters the diagnostic object.
"""
import argparse
from collections import deque
import csv
import json
from pathlib import Path

import numpy as np
from simulink_realtime import RealtimeDiagnoser
from simulink_realtime_training import load_realtime_checkpoint
from simulink_sequence_events import EventPolicy


class MeasurementAligner:
    """Raw IMU six channels + ALREADY delayed measured RPM six channels.

    Delay only IMU using past packets; mirror the frozen native startup hold.
    The diagnostic accepts this aligned output directly without further delay.
    """
    def __init__(self, imu_delay_samples=3):
        if int(imu_delay_samples) != imu_delay_samples or imu_delay_samples < 0:
            raise ValueError("Nonnegative integer IMU delay required")
        self.delay = int(imu_delay_samples)
        self.history = deque(maxlen=self.delay+1)

    def reset(self):
        self.history.clear()

    def push(self, measurements):
        x = np.asarray(measurements,dtype=np.float32)
        if x.ndim!=2 or x.shape[1]!=12 or not np.isfinite(x).all():
            raise ValueError("Expected finite raw sensor [N,12]")
        aligned = x.copy()
        for i,row in enumerate(x):
            if not self.history:
                self.history.extend([row[:6].copy()]*(self.delay+1))
            else:
                self.history.append(row[:6].copy())
            aligned[i,:6] = self.history[0]
        return aligned


def diagnose_packets(diagnostic, packets, aligner=None):
    """This generator never asks its transport for a future packet before yielding."""
    for raw,time_s in packets:
        aligned = raw if aligner is None else aligner.push(raw)
        for decision in diagnostic.push(aligned,time_s):
            yield decision


def csv_packets(path, packet_points=10):
    columns = ([f"accelerometer_sensor_m_s2_{i}" for i in range(1,4)]+
               [f"gyro_sensor_rad_s_{i}" for i in range(1,4)]+
               [f"feedback_state_{i}" for i in range(14,20)])
    with Path(path).open(encoding="utf-8-sig",newline="") as handle:
        values,times = [],[]
        for row in csv.DictReader(handle):
            values.append([float(row[c]) for c in columns])
            times.append(float(row["time_s"]))
            if len(values)==packet_points:
                yield np.asarray(values,dtype=np.float32),np.asarray(times)
                values,times=[],[]
        if values:
            yield np.asarray(values,dtype=np.float32),np.asarray(times)


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint",required=True)
    parser.add_argument("--csv",required=True)
    parser.add_argument("--device",choices=("cpu","cuda"),default="cpu")
    parser.add_argument("--packet-points",type=int,default=10)
    args=parser.parse_args()
    if args.packet_points<1:
        parser.error("--packet-points must be positive")
    model,scaler,saved=load_realtime_checkpoint(args.checkpoint,args.device)
    diagnostic=RealtimeDiagnoser(model,scaler,EventPolicy(**saved["best_policy"]),device=args.device)
    for result in diagnose_packets(diagnostic,csv_packets(args.csv,args.packet_points),MeasurementAligner()):
        print(json.dumps({k:v for k,v in result.items() if k!="probabilities"},ensure_ascii=False),flush=True)
    final=diagnostic.finish()
    print(json.dumps(dict(finished=True,samples=final["samples"],events=final["events"]),ensure_ascii=False))
