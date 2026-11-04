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


def motor_response_scores(commands,rpm,dt,tau,delay_s):
    """Causal model-matched residual baseline. No fault truth or injection time."""
    delay=int(np.ceil(delay_s/dt-1e-10)); decay=np.exp(-dt/np.asarray(tau))
    beta=np.ones(6); probabilities=[]; labels=[]; estimates=[]
    for k in range(len(rpm)):
        if k>delay:
            prior=commands[k-delay-1]; active=prior>=100
            observed=(rpm[k]-decay*rpm[k-1])/np.maximum((1-decay)*prior,1e-9)
            beta[active]+=(1-np.exp(-dt/.025))*(np.clip(observed[active],0,1.1)-beta[active])
        eta=np.minimum(beta**2,1); eta[beta>=.98]=1; eta[beta<.06]=0
        motor=int(np.argmin(eta)); severity=int(np.argmin(np.abs(np.asarray([0.,.2,.8])-eta[motor])))
        probabilities.append(float(1-eta[motor])); labels.append(1+motor+6*severity); estimates.append(eta.copy())
    return dict(fault_probability=np.asarray(probabilities),fault_class=np.asarray(labels),estimated_effectiveness=np.asarray(estimates))


def alarm_trace(probability,fault_class,threshold,hold):
    """Detection persistence is independent of class; localization has its own clock."""
    detected=np.zeros(len(probability),bool); localized=np.zeros(len(probability),np.int64)
    detection_streak=0; motor_streak=0; previous_motor=0
    for i,(p,label) in enumerate(zip(probability,fault_class)):
        positive=bool(p>=threshold); motor=(int(label)-1)%6+1
        detection_streak=detection_streak+1 if positive else 0
        motor_streak=motor_streak+1 if positive and motor==previous_motor else int(positive)
        previous_motor=motor if positive else 0
        detected[i]=detection_streak>=hold
        if detected[i] and motor_streak>=hold: localized[i]=motor
    return detected,localized


def score_alarm_records(records,threshold,hold):
    cases=[]; false_events=0; normal_duration=0.; fault_count=0; delays=[]; location_delays=[]; detected_count=0
    first_localized=0; first_correct=0
    for record in records:
        t=np.asarray(record['time_s']); probability=np.asarray(record['fault_probability']); labels=np.asarray(record['fault_class'])
        alarm,motor=alarm_trace(probability,labels,threshold,hold)
        rising=np.flatnonzero(alarm & ~np.r_[False,alarm[:-1]])
        if record['label']==0:
            # Count one full healthy source per paired condition, not 18 copies of its prefix.
            false_events+=len(rising); normal_duration+=float(t[-1]-t[0])
            cases.append(dict(source_id=record['source_id'],group_id=record['group_id'],label=0,
                              false_alarm_events=len(rising),normal_observation_s=float(t[-1]-t[0])))
            continue
        fault_count+=1; onset=record['fault_start_s']; new=rising[t[rising]>=onset]
        first=float(t[new[0]]) if len(new) else None
        detection_delay=None if first is None else first-onset
        target=(record['label']-1)%6+1
        located=np.flatnonzero((motor!=0)&(t>=max(first,onset))) if first is not None else np.empty(0,int)
        first_motor=int(motor[located[0]]) if len(located) else None
        first_location_delay=float(t[located[0]]-onset) if len(located) else None
        if first_motor is not None:
            first_localized+=1; first_correct+=int(first_motor==target)
        correct=np.flatnonzero((motor==target)&(t>=max(first,onset))) if first is not None else np.empty(0,int)
        location_delay=None if not len(correct) else float(t[correct[0]]-onset)
        if first is not None: detected_count+=1; delays.append(detection_delay)
        if location_delay is not None: location_delays.append(location_delay)
        before=np.flatnonzero(t<onset)
        cases.append(dict(source_id=record['source_id'],group_id=record['group_id'],label=record['label'],
                          detected=first is not None,detection_delay_s=detection_delay,
                          first_localization_motor=first_motor,first_localization_delay_s=first_location_delay,
                          first_localization_correct=None if first_motor is None else first_motor==target,
                          correctly_localized=location_delay is not None,localization_delay_s=location_delay,
                          alarm_already_active_before_fault=bool(len(before) and alarm[before[-1]])))
    miss=(fault_count-detected_count)/max(fault_count,1); rate=false_events/max(normal_duration/60,1e-12)
    mean_delay=float(np.mean(delays)) if delays else 24.
    return dict(threshold=float(threshold),hold=int(hold),false_alarm_events=false_events,
                normal_observation_s=normal_duration,false_alarm_events_per_minute=rate,fault_records=fault_count,
                detected=detected_count,missed=fault_count-detected_count,detection_rate=detected_count/max(fault_count,1),
                detection_delays_s=delays,localization_delays_s=location_delays,
                correctly_localized=len(location_delays),localization_rate=len(location_delays)/max(fault_count,1),
                first_localization_accuracy=None if not first_localized else first_correct/first_localized,
                first_correct_localization_fraction_all_faults=first_correct/max(fault_count,1),
                localization_rate_scope='Eventually correct before record end; first-localization accuracy is reported separately.',
                conditional_mean_detection_delay_s=None if not delays else mean_delay,
                objective=miss+.05*rate+.05*mean_delay/24,cases=cases,
                policy='Count fresh detection rising edges after onset; pre-existing alarms are not credited. Healthy exposure is deduplicated by paired group; detection/localization timed separately.')


def select_alarm_policy(records,thresholds,holds):
    grid=[score_alarm_records(records,t,h) for t in thresholds for h in holds]
    selected=min(grid,key=lambda r:(r['objective'],-r['localization_rate'],r['threshold'],r['hold']))
    return dict(threshold=selected['threshold'],hold=selected['hold'],validation=selected,grid=grid,
                selection_scope='validation sources only; no final test scores used')


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--fold',type=Path,required=True); parser.add_argument('--data',type=Path,required=True)
    parser.add_argument('--variant',default='no_gan'); parser.add_argument('--stride',type=int,default=10)
    parser.add_argument('--probability',type=float,default=.5); parser.add_argument('--hold',type=int,default=3)
    args=parser.parse_args(); replay(args.fold,args.data,args.variant,args.stride,args.probability,args.hold)
