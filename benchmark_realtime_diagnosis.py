"""Measure actual public push() latency independently of simulated alarm delay."""
import argparse
from pathlib import Path
import time

import numpy as np
import pandas as pd
import torch

from provenance import runtime_info,write_json
from simulink_data import read
from simulink_sequence_events import EventPolicy
from simulink_realtime import RealtimeDiagnoser
from simulink_realtime_training import load_realtime_checkpoint


def benchmark(root,devices=("cpu","cuda"),repeats=600):
    root=Path(root)
    flights=torch.load(root/"development_flights.pt",weights_only=False)
    # Predetermined training source, not selected by test accuracy.
    flight=next(f for f in flights if f.role=="train" and f.group==0 and f.label==7)
    jobs=[r for r in read(root/"formal_training_summaries.json") if r["objective"]=="current_state"]
    rows=[]
    torch.set_num_threads(1)
    for job in jobs:
        for device in devices:
            if device=="cuda" and not torch.cuda.is_available():
                continue
            model,scaler,saved=load_realtime_checkpoint(Path(job["folder"])/"best.pt",device)
            for period in (10,1):
                diagnostic=RealtimeDiagnoser(model,scaler,EventPolicy(**saved["best_policy"]),decision_samples=period,device=device)
                durations=[]
                # Repeat training measurements as a synthetic transport timing load;
                # actual arrival timestamp remains monotone, no classification score.
                for i in range(repeats+60):
                    indices=(np.arange(period)+i*period)%len(flight.x)
                    x=flight.x[indices]
                    t=(np.arange(period)+i*period)/100
                    if device=="cuda":torch.cuda.synchronize()
                    started=time.perf_counter()
                    diagnostic.push(x,t)
                    if device=="cuda":torch.cuda.synchronize()
                    elapsed=time.perf_counter()-started
                    if i>=60:durations.append(elapsed)
                a=np.asarray(durations)
                row=dict(model=job["model"],seed=job["seed"],device=device,decision_samples=period,
                    period_ms=period*10,measurements=len(a),warmup_calls=60,
                    median_ms=float(np.median(a)*1000),p95_ms=float(np.quantile(a,.95)*1000),
                    maximum_ms=float(a.max()*1000),calls_over_period=int((a>period/100).sum()),
                    median_within_period=bool(np.median(a)<period/100),p95_within_period=bool(np.quantile(a,.95)<period/100))
                rows.append(row)
                print("ONLINE_BENCHMARK",row,flush=True)
    pd.DataFrame(rows).to_csv(root/"online_latency.csv",index=False,encoding="utf-8-sig")
    write_json(root/"online_latency.json",dict(rows=rows,cpu_threads=1,
        gpu=torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        measurement="Synchronized wall time around public push, including scaling/state/event/return-copy; one decision per call",
        scope="Observed desktop timings, no hard realtime guarantee or operating-system deadline certification",runtime=runtime_info()))
    return rows


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results",required=True)
    parser.add_argument("--repeats",type=int,default=600)
    args=parser.parse_args()
    benchmark(args.results,repeats=args.repeats)
