"""Causal, held-out full-record replay of a previously trained fold classifier."""
import argparse
import pickle
from pathlib import Path
import numpy as np
import pandas as pd
import torch

import classify
from provenance import write_json, sha256_file
from simulink_data import read, checked_plan, validate_run, observable_features, scenarios


def causal_predictions(model, scaler, values, window, stride=10, probability=.5, hold=3):
    if min(window,stride,hold)<1 or not 0<=probability<=1:
        raise ValueError('Invalid predeclared replay policy')
    scaled=scaler.transform(values).astype(np.float32)
    outputs=[]; previous=0; streak=0
    model.eval()
    with torch.inference_mode():
        for end in range(window-1,len(values),stride):
            x=torch.as_tensor(scaled[end-window+1:end+1][None],device=classify.DEVICE)
            probs=torch.softmax(model(x),dim=1)[0]
            confidence,label=probs.max(0); label=int(label); confidence=float(confidence)
            candidate=label if confidence>=probability else 0
            streak=streak+1 if candidate==previous and candidate else int(candidate!=0)
            previous=candidate
            outputs.append(dict(end_index=end,prediction=label,confidence=confidence,
                                alarm=candidate if streak>=hold else 0))
    return outputs


def replay(fold_dir,data_dir,variant='no_gan',stride=10,probability=.5,hold=3):
    fold_dir,data_dir=Path(fold_dir),Path(data_dir)
    signature=read(fold_dir/'fold_signature.json')
    if signature.get('data_profile')!='simulink_observable_v1':
        raise ValueError('Replay requires trustworthy native injection metadata')
    plan=checked_plan(data_dir)
    metadata=read(Path(classify.DATA_DIR)/'preprocess_metadata.json')
    if metadata['plan_signature'] != plan['plan_signature']:
        raise ValueError('Replay dataset differs from training dataset')
    classify.SEQ_LEN=metadata['window']; classify.N_FEATURES=len(metadata['feature_names']); classify.N_CLASSES=19
    classify.HIDDEN_DIM=signature['classifier_hidden_dim']; classify.N_LAYERS=signature['classifier_layers']
    classify.ATTENTION_POOLING=signature['attention_pooling']
    classify.FEATURE_NAMES=metadata['feature_names']
    with (fold_dir/'classifier_scaler.pkl').open('rb') as f: scaler=pickle.load(f)
    model=classify.FaultClassifier().to(classify.DEVICE)
    model.load_state_dict(torch.load(fold_dir/variant/'classifier.pth',map_location=classify.DEVICE,weights_only=True))
    destination=fold_dir/'causal_replay'/variant; destination.mkdir(parents=True,exist_ok=True)
    outcomes=[]; motors={s['label']:s['motor_simulink'] for s in scenarios()}
    for run in plan['runs']:
        if run['group_id'] not in signature['test_groups']: continue
        if run['group_id'] in signature['train_groups']: raise ValueError('Replay source leakage')
        meta,df=validate_run(data_dir/run['run_key'],run,plan)
        x=observable_features(df,meta,metadata['feature_set'])
        rows=causal_predictions(model,scaler,x,metadata['window'],stride,probability,hold)
        for row in rows:
            row['time_s']=float(df.time_s.iloc[row['end_index']])
            row['truth_label']=run['label'] if row['time_s']>=meta['fault_start_s'] else 0
            row['crosses_fault']=bool(run['label'] and df.time_s.iloc[row['end_index']-metadata['window']+1]<meta['fault_start_s']<=row['time_s'])
        trace=pd.DataFrame(rows); trace.to_csv(destination/f'{run["run_key"]}.csv',index=False)
        pre=trace.time_s<meta['fault_start_s'] if run['label'] else np.ones(len(trace),bool)
        post=trace.loc[trace.time_s>=meta['fault_start_s']]
        detected=post.loc[post.alarm!=0] if run['label'] else post.iloc[:0]
        outcomes.append(dict(run_key=run['run_key'],group_id=run['group_id'],label=run['label'],
                             pre_fault_false_alarm_fraction=float(np.mean(trace.loc[pre,'alarm']!=0)) if np.any(pre) else None,
                             missed=bool(run['label'] and detected.empty),
                             detection_delay_s=None if detected.empty else float(detected.time_s.iloc[0]-meta['fault_start_s']),
                             first_alarm_motor_correct=None if detected.empty else motors[int(detected.alarm.iloc[0])]==run['motor_simulink']))
    report=dict(policy=dict(window=metadata['window'],stride=stride,probability=probability,hold=hold,
                            crossing_window_label='fault class if window end >= injection time; crossing flag recorded',
                            scope='full-record past-and-current windows only; observed transport delay is included in detection delay'),
                source_plan_signature=plan['plan_signature'],classifier_sha256=sha256_file(fold_dir/variant/'classifier.pth'),
                cases=outcomes,simulation_only=True)
    write_json(destination/'report.json',report)
    print('CAUSAL_REPLAY',len(outcomes),'sources',destination)
    return report


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--fold',type=Path,required=True); parser.add_argument('--data',type=Path,required=True)
    parser.add_argument('--variant',default='no_gan'); parser.add_argument('--stride',type=int,default=10)
    parser.add_argument('--probability',type=float,default=.5); parser.add_argument('--hold',type=int,default=3)
    args=parser.parse_args(); replay(args.fold,args.data,args.variant,args.stride,args.probability,args.hold)
