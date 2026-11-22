"""Independent CSV recomputation and static scientific figures for native runs."""
import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.io import loadmat
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from provenance import sha256_file,write_json


def rows(folder):
    return [json.loads(p.read_text(encoding='utf-8-sig'))|{'directory':str(p.parent)}
            for p in sorted(folder.glob('*/run.json'))]


def recompute(row):
    path=Path(row['directory']);df=pd.read_csv(path/'signals.csv')
    meta=json.loads((path/'execution_metadata.json').read_text(encoding='utf-8-sig'));P=meta['parameters']
    assert sha256_file(path/'signals.csv')==row['csv_sha256']==meta['csv_sha256']
    assert sha256_file(path/'result.mat')==row['mat_sha256']==meta['mat_sha256']
    t=df.time_s.to_numpy(); assert len(t)==2401 and np.max(abs(t-np.arange(2401)*.01))<1e-10
    assert np.isfinite(df.to_numpy()).all()
    s=df[['east_m','north_m','up_m']].to_numpy()
    ref=df[[f'reference_position_m_{i}' for i in range(1,4)]].to_numpy()
    error=np.linalg.norm(s-ref,axis=1)
    tilt=np.degrees(np.arccos(np.clip(1-2*(df.qx**2+df.qy**2),-1,1)))
    rp=np.linalg.norm(df[['wx_rad_s','wy_rad_s']],axis=1)
    loss=float(np.array(P['task_origin']).reshape(-1)[2])-df.up_m.to_numpy()
    post=t>=P['fault_start'];bad=(error>.15)|(tilt>15)|(rp>1)|(loss>.5)
    indices=np.flatnonzero(post&bad); recovery=0. if not len(indices) else t[indices[-1]]-P['fault_start']
    if not len(t[post]) or (len(indices) and indices[-1]==len(t)-1):recovery=float('inf')
    late=t>=21;late_rmse=float(np.sqrt(np.mean(error[late]**2)))
    stored=row['metrics'];assert abs(recovery-stored['return_time_s'])<1e-8
    assert abs(late_rmse-stored['late_tracking_rmse_m'])<1e-10
    def c(prefix,n):return df[[f'{prefix}_{i}' for i in range(1,n+1)]].to_numpy()
    def d(prefix):return df[[f'diag_{prefix}_{a}' for a in 'xyz']].to_numpy()
    residual=c('wrench_desired',4)-c('wrench_estimated_allocated',4)
    residual_error=float(np.max(abs(residual-c('wrench_estimated_residual',4))))
    assert residual_error<1e-10
    vi=d('velocity_I_applied')+sum(d(g) for g in ['velocity_error_increment','velocity_component_backcalc',
        'velocity_vertical_backcalc','velocity_tilt_backcalc','velocity_total_backcalc','velocity_allocation_backcalc','velocity_clip_increment'])
    vi_error=float(np.max(abs(vi-c('velocity_integral_m_s2',3))));assert vi_error<1e-10
    ri=d('rate_I_applied')+d('rate_applied_increment')+d('rate_clip_increment')
    # Degraded yaw intentionally clears old yaw integral before normal update.
    ri_error=float(np.max(abs(ri[:,:2]-c('rate_integral_rad_s2',3)[:,:2])));assert ri_error<1e-10
    for axis in range(2):
        inc=d('rate_error_increment')[:,axis];old=d('rate_I_applied')[:,axis]
        trigger=d('rate_gate_triggered')[:,axis]>.5
        eligible=trigger & (old*inc<0) & (df.fault_severity.to_numpy()>.1)
        actual=d('rate_applied_increment')[:,axis]
        assert np.all(actual[eligible]*old[eligible]<0), 'Unwinding was frozen again'
    cmd=c('rotor_command_rad_s',6);assert cmd.min()>=-1e-8 and cmd.max()<=P['omega_max']+1e-8
    return dict(case=row['case']['name'],points=len(t),columns=len(df.columns),
                independent_return_s=recovery,independent_late_rmse_m=late_rmse,
                allocation_residual_max_error=residual_error,velocity_update_max_error=vi_error,
                rate_update_max_error=ri_error,passed=True)


def pair_audits(records,output):
    by={r['case']['suite']+f"_t{r['case']['task']}_m{r['case']['motor']}":r for r in records}
    mirrors=[];convergence=[]
    for r in records:
        c=r['case'];suite=c['suite']
        if 'mirror_' in suite and suite.endswith('_original'):
            other=by[suite.replace('_original','_reflected')+f"_t{c['task']}_m{c['motor']+3}"]
            a=loadmat(Path(r['directory'])/'result.mat',simplify_cells=True)['result'];b=loadmat(Path(other['directory'])/'result.mat',simplify_cells=True)['result']
            phi=(c['motor']-1)*np.pi/3;n=np.array([np.cos(phi),np.sin(phi),0]);S=np.eye(3)-2*np.outer(n,n)
            H=-S;positions=np.asarray(a['parameters']['rotor_positions']);transformed=positions@S.T
            perm=np.argmin(np.linalg.norm(positions[None,:,:]-transformed[:,None,:],axis=2),axis=1)
            mirrors.append(dict(suite=suite,permutation=(perm+1).tolist(),polar=S.tolist(),axial=H.tolist(),
                max_position_difference_m=float(np.max(np.linalg.norm(b['state'][:,:3]-a['state'][:,:3]@S.T,axis=1))),
                max_body_rate_difference_rad_s=float(np.max(np.linalg.norm(b['state'][:,10:13]-a['state'][:,10:13]@H.T,axis=1))),
                max_command_difference_rad_s=float(np.max(abs(b['motor_command']-a['motor_command'][:,perm]))),
                max_wind_transform_error=float(np.max(abs(b['wind']-a['wind']@S.T))),
                max_exogenous_transform_error=float(np.max(abs(b['parameters']['exogenous'][:,:3]-a['parameters']['exogenous'][:,:3]@S.T))),
                recovery_difference_s=other['metrics']['return_time_s']-r['metrics']['return_time_s']))
        if suite=='convergence_0.002':
            other=by[f'convergence_0.001_t{c["task"]}_m{c["motor"]}']
            a=loadmat(Path(r['directory'])/'result.mat',simplify_cells=True)['result'];b=loadmat(Path(other['directory'])/'result.mat',simplify_cells=True)['result']
            assert np.array_equal(a['parameters']['exogenous'],b['parameters']['exogenous'])
            position=float(np.max(np.linalg.norm(a['state'][:,:3]-b['state'][:,:3],axis=1)))
            delta=abs(r['metrics']['return_time_s']-other['metrics']['return_time_s'])
            convergence.append(dict(task=c['task'],motor=c['motor'],max_position_difference_m=position,
                recovery_difference_s=delta,passed=position<.02 and delta<=.1))
    write_json(output/'mirror_audit.json',mirrors);write_json(output/'convergence_audit.json',convergence)
    return mirrors,convergence


def development(folder,output):
    records=rows(folder);flat=[]
    for r in records:
        a=loadmat(Path(r['directory'])/'result.mat',simplify_cells=True)['result']
        t=a['time'];s=a['state'];ref=a['reference'];audit=a['control_audit'];diag=a['controller_diagnostic']
        nominal=np.linalg.norm(s[:,:3]-a['reference_audit'][:,2:5],axis=1)
        phase_indices=np.flatnonzero(a['reference_audit'][:,0]>=a['parameters']['motion_end'])
        last_changes=[]
        for axis in range(3):
            changes=np.flatnonzero(abs(np.diff(audit[:,15+axis]))>1e-10)+1
            last_changes.append(float(t[changes[-1]]) if len(changes) else None)
        extra=dict(nominal_tracking_rmse_m=float(np.sqrt(np.mean(nominal**2))),
            late_nominal_rmse_m=float(np.sqrt(np.mean(nominal[t>=21]**2))),
            mission_completion_time_s=float(t[phase_indices[0]]) if len(phase_indices) else None,
            max_velocity_integral_m_s2=np.max(abs(audit[:,12:15]),axis=0).tolist(),
            max_rate_integral_rad_s2=np.max(abs(audit[:,15:18]),axis=0).tolist(),
            rate_integral_last_change_s=last_changes,
            actuator_saturated_time_s=float(np.sum(np.diff(t)*np.any(a['motor_command'][:-1]>=1099.999999,axis=1))),
            max_requested_moment_Nm=np.max(abs(audit[:,1:4]),axis=0).tolist(),
            max_allocated_moment_Nm=np.max(abs(audit[:,5:8]),axis=0).tolist(),
            max_moment_residual_Nm=np.max(abs(audit[:,9:12]),axis=0).tolist(),
            velocity_backcalc_components_m_s2=diag[:,48:63].sum(axis=0).tolist())
        flat.append(dict(variant=r['case']['variant'],condition=r['case']['condition'],**r['metrics'],**extra))
    df=pd.DataFrame(flat);df.to_csv(output/'ablation_cases.csv',index=False,encoding='utf-8-sig')
    summaries=[]
    for variant,a in df.groupby('variant',sort=False):
        summaries.append(dict(variant=variant,cases=len(a),passed=int(a.passed.sum()),
            median_recovery_s=float(a.return_time_s.median()),p90_recovery_s=float(a.return_time_s.quantile(.9)),
            max_recovery_s=float(a.return_time_s.max()),worst_late_rmse_m=float(a.late_tracking_rmse_m.max()),
            peak_tilt_deg=float(a.peak_tilt_deg.max()),peak_rp_rate_rad_s=float(a.peak_rp_rate_rad_s.max())))
    write_json(output/'ablation_summary.json',summaries)
    original=df[df.variant=='original'].set_index('condition')
    changes=[]
    for _,r in df[df.variant!='original'].iterrows():
        old=original.loc[r.condition]
        changes.append(dict(variant=r.variant,condition=r.condition,
            recovery_change_s=r.return_time_s-old.return_time_s,
            late_rmse_change_m=r.late_tracking_rmse_m-old.late_tracking_rmse_m,
            peak_tilt_change_deg=r.peak_tilt_deg-old.peak_tilt_deg,
            peak_rp_rate_change_rad_s=r.peak_rp_rate_rad_s-old.peak_rp_rate_rad_s))
    pd.DataFrame(changes).to_csv(output/'all_paired_changes.csv',index=False,encoding='utf-8-sig')
    return df,summaries


def plots(df,records,output):
    plt.rcParams.update({'font.sans-serif':['Microsoft YaHei'],'axes.unicode_minus':False})
    selected=['calm_hover_m1','noisy_hover_m1','noisy_hover_m4','noisy_circle_m1','noisy_circle_m4','noisy_line_m1','noisy_vertical_m1']
    fig,ax=plt.subplots(figsize=(12,5))
    for variant,label in [('original','原控制器'),('rate_only','仅角速度积分'),('velocity_only','仅速度积分'),('both_k16','两项修复')]:
        a=df[df.variant==variant].set_index('condition').loc[selected]
        ax.plot(range(len(selected)),a.return_time_s,'o-',label=label)
    ax.set_xticks(range(len(selected)),['理想悬停M1','风噪悬停M1','风噪悬停M4','绕圈M1','绕圈M4','直线M1','升降M1'])
    ax.set_ylabel('故障后恢复时间（s）');ax.grid(alpha=.25);ax.legend();fig.tight_layout();fig.savefig(output/'01_ablation_recovery.png',dpi=180);plt.close(fig)
    base=Path(records[0]['directory']).parent.parent if records else None
    if base:
        fig,axs=plt.subplots(4,2,figsize=(13,10),sharex=True)
        for column,(variant,label) in enumerate([('original','原控制器'),('both_k16','正式修复')]):
            a=loadmat(base/'development_01'/f'{variant}__calm_hover_m1'/'result.mat',simplify_cells=True)['result']
            t=a['time'];d=a['controller_diagnostic'];audit=a['control_audit'];s=a['state']
            axs[0,column].plot(t,np.linalg.norm(s[:,:3]-a['reference'][:,:3],axis=1));axs[0,column].axhline(.15,color='gray',ls=':');axs[0,column].set_title(label+'：理想悬停M1完全失效');axs[0,column].set_ylabel('位置误差（m）')
            for index,name in [(22,'P'),(25,'I'),(28,'D'),(31,'净请求')]:axs[1,column].plot(t,d[:,index],label=name)
            axs[1,column].set_ylabel('俯仰角速度环（rad/s²）');axs[1,column].legend(ncol=4,fontsize=8)
            for index,name in [(0,'P'),(3,'I'),(6,'D'),(9,'净请求')]:axs[2,column].plot(t,d[:,index],label=name)
            axs[2,column].set_ylabel('x速度环（m/s²）');axs[2,column].legend(ncol=4,fontsize=8)
            axs[3,column].plot(t,audit[:,16]);axs[3,column].set_ylabel('俯仰积分（rad/s²）');axs[3,column].set_xlabel('Time (s)')
            for ax in axs[:,column]:ax.grid(alpha=.25);ax.axvline(3,ls='--',color='gray')
        fig.tight_layout();fig.savefig(output/'03_mechanism_before_after.png',dpi=180);plt.close(fig)
    if records:
        r=max(records,key=lambda a:a['metrics']['return_time_s']);data=pd.read_csv(Path(r['directory'])/'signals.csv')
        fig,axs=plt.subplots(3,1,figsize=(11,8),sharex=True)
        axs[0].plot(data.time_s,data.diag_position_error_norm_m);axs[0].axhline(.15,color='gray',ls=':');axs[0].set_ylabel('位置误差（m）')
        axs[1].plot(data.time_s,data.diag_rate_I_applied_y)
        axs[1].set_ylabel('俯仰角速度环积分（rad/s²）')
        axs[2].plot(data.time_s,data.diag_velocity_I_applied_x)
        axs[2].set_ylabel('x速度环积分（m/s²）');axs[2].set_xlabel('Time (s)')
        for ax in axs:ax.grid(alpha=.25);ax.axvline(r['case']['fault_start'],ls='--',color='gray')
        fig.suptitle('最终验证中恢复最慢工况：'+r['case']['name']);fig.tight_layout();fig.savefig(output/'02_worst_recovery_trace.png',dpi=180);plt.close(fig)
        core=[r for r in records if r['case']['suite']=='development' and r['case']['seed']==20261006]
        if len(core)==76:
            fig,axs=plt.subplots(2,2,figsize=(12,7),sharex=True,sharey=True)
            for task,ax in enumerate(axs.flat):
                for eta,label in [(.8,'剩余80%效能'),(.2,'剩余20%效能'),(0.,'完全失效')]:
                    selected=sorted([r for r in core if r['case']['task']==task and r['case']['motor'] and r['case']['factor']==eta],key=lambda r:r['case']['motor'])
                    ax.plot(range(1,7),[r['metrics']['return_time_s'] for r in selected],'o-',label=label)
                ax.set_title(['悬停','直线','绕圈','升降'][task]);ax.set_xticks(range(1,7),[f'M{i}' for i in range(1,7)]);ax.set_ylabel('故障后恢复时间（s）');ax.grid(alpha=.25)
            axs[0,0].legend(fontsize=9);fig.tight_layout();fig.savefig(output/'04_core_72_recovery.png',dpi=180);plt.close(fig)


def main(base,output):
    output.mkdir(parents=True,exist_ok=True);dev,summary=development(base/'development_01',output)
    records=rows(base/'preflight_01')+rows(base/'formal_01')
    complete=[r for r in records if r['status']=='completed']; audit=[]
    for i,r in enumerate(complete):
        audit.append(recompute(r))
        if i%50==0: print('AUDIT',i,len(complete),flush=True)
    flat=[dict(name=r['case']['name'],suite=r['case']['suite'],**r['metrics'],directory=r['directory'],csv_sha256=r['csv_sha256'],mat_sha256=r['mat_sha256']) for r in complete]
    frame=pd.DataFrame(flat);frame.to_csv(output/'native_cases.csv',index=False,encoding='utf-8-sig')
    write_json(output/'native_cases.json',flat)
    pd.DataFrame(audit).to_csv(output/'independent_csv_recomputation.csv',index=False)
    mirrors,convergence=pair_audits(complete,output)
    stats=[]
    if len(frame):
        for (suite,seed),a in frame.groupby(['suite','seed']):
            stats.append(dict(suite=suite,seed=int(seed),cases=len(a),passed=int(a.passed.sum()),
                median_recovery_s=float(a.return_time_s.median()),p90_recovery_s=float(a.return_time_s.quantile(.9)),
                max_recovery_s=float(a.return_time_s.max()),worst_late_rmse_m=float(a.late_tracking_rmse_m.max()),
                max_tilt_deg=float(a.peak_tilt_deg.max()),max_rp_rate_rad_s=float(a.peak_rp_rate_rad_s.max()),
                mission_complete=int(a.mission_phase_complete.sum())))
    write_json(output/'validation_summary.json',dict(development=summary,groups=stats,
        actual_native_full_runs=len(complete)+len(dev),final_passed=int(frame.passed.sum()) if len(frame) else 0,
        execution_failures=[r for r in records if r['status']!='completed'],
        flight_failures=[r for r in flat if not r['passed']],csv_audits_passed=len(audit),
        source_signatures=sorted({r['source_signature'] for r in complete})))
    plots(dev,complete,output)
    print('ANALYSIS_FINISHED',len(complete),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('base',type=Path);p.add_argument('output',type=Path);a=p.parse_args();main(a.base,a.output)
