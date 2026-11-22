"""Actual revised-controller records through the existing whole-flight interface.

This is an interface check with untrained network weights, not a training result.
"""
import argparse
import json
import sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from provenance import sha256_file,write_json
from simulink_data import checked_plan,validate_run,read,FEATURES,observable_features
from simulink_sequence_data import load_flights,fit_scaler,collate,source_receipt
from simulink_sequence_model import FlightClassifier


def check(folder):
    plan=checked_plan(folder); records=[]
    for r in plan['runs']:
        source=folder/r['run_key']; meta,df=validate_run(source,r,plan)
        x=observable_features(df,meta,'imu_rpm');assert x.shape==(2401,12)
        changed=df.copy()
        for c in changed.columns:
            if c.startswith(('rotor_command_','rotor_actual_','plant_effectiveness_','diag_')):
                changed[c]=999999.
        assert np.array_equal(x,observable_features(changed,meta,'imu_rpm'))
        records.append(dict(source_id='control_v4_'+r['run_key'],source_path=str(source),
                            plan_path=str(folder/'collection_plan.json'),run=r,
                            role=plan['condition_roles'][str(r['group_id'])],group_id=r['group_id'],
                            csv_sha256=meta['csv_sha256'],metadata_sha256=sha256_file(source/'run_metadata.json')))
    registry=folder/'registry.json';write_json(registry,dict(records=records))
    flights=load_flights([registry],roles=('train','validation','test'))
    train=[f for f in flights if f.role=='train']; scaler=fit_scaler(train)
    torch.set_num_threads(2);torch.manual_seed(42)
    model=FlightClassifier().eval(); largest=0.
    for flight in flights:
        x,lengths,current,seen,labels=collate([flight],scaler,'cpu')
        assert x.shape==(1,2401,12)
        assert torch.all(current[0,:300]==0)
        assert torch.all(current[0,300:]==flight.label)
        with torch.inference_mode():
            whole=model(x); memory=None;pieces=[]
            for start in range(0,x.shape[1],137):
                out=model(x[:,start:start+137],memory=memory);memory=out['memory'];pieces.append(out['prefix_logits'])
            error=float((whole['prefix_logits']-torch.cat(pieces,1)).abs().max())
            largest=max(largest,error);assert error<1e-5
        np.savez_compressed(folder/(flight.source_id+'.npz'),measured=flight.x,time=flight.time,
                            current=current.numpy()[0],prefix=seen.numpy()[0],record_label=flight.label)
    try:fit_scaler([f for f in flights if f.role=='validation']);raise AssertionError('Leakage allowed')
    except PermissionError:pass
    receipt=dict(passed=True,training_performed=False,network='existing causal LSTM; unchanged',
                 records=len(flights),points_per_record=2401,channels=FEATURES['imu_rpm'],
                 train_only_scaler_samples=int(scaler.n_samples_seen_),chunk_max_abs_difference=largest,
                 provenance=source_receipt(flights),scaler_mean=scaler.mean_.tolist(),scaler_scale=scaler.scale_.tolist())
    write_json(folder/'interface_receipt.json',receipt);print(json.dumps({k:v for k,v in receipt.items() if k!='provenance'},ensure_ascii=False))

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('folder',type=Path);check(parser.parse_args().folder.resolve())
