"""Predeclared research checks, actual flight metrics, capability LPs."""
from copy import deepcopy
from pathlib import Path
import numpy as np
from scipy.optimize import linprog
from scipy.spatial.transform import Rotation
from provenance import write_json,stable_hash
from .configuration import physical,matrix,ROOT
from .simulation import simulate,FaultSchedule

def metric(run):
    rows=run['rows']; s=np.array([r['state'] for r in rows]); refs=np.array([r['reference']['x'] for r in rows]); error=s[:,:3]-refs
    tilt=np.rad2deg(np.arccos(np.clip([Rotation.from_quat(q).as_matrix()[2,2] for q in s[:,6:10]],-1,1)))
    return {'position_rmse_m':float(np.sqrt(np.mean(np.sum(error**2,axis=1)))),
            'final_position_error_m':float(np.linalg.norm(error[-1])), 'maximum_tilt_deg':float(max(tilt)),
            'maximum_rate_rad_s':float(np.max(np.linalg.norm(s[:,10:13],axis=1))),
            'maximum_speed_m_s':float(np.max(np.linalg.norm(s[:,3:6],axis=1))),
            'nominal_bound_fraction':float(np.mean([r['control']['allocator_bound_count']>0 for r in rows])),
            'final_velocity_integral':rows[-1]['control']['velocity_integral'].tolist(),
            'termination':run['termination'],'actual_duration_s':run['actual_duration_s'],
            'feedback_mode':run['feedback_mode'],'elapsed_s':run['elapsed_s'],
            'controller_evaluations':run['controller_evaluations']}

def capability(cfg):
    p=physical(cfg); A=matrix(p); maximum=p['kT']*p['omega_max']**2; hover=np.r_[p['mass']*p['gravity'],0,0,0]; rows=[]
    for factor in (1.,0.,.2,.8):
        for motor in (range(6) if factor!=1 else [-1]):
            eta=np.ones(6)
            if motor>=0: eta[motor]=factor
            active=eta>0; B=A[:,active]*eta[active]; bounds=[(0,float(m)) for m in maximum[active]]
            fit=linprog(np.zeros(sum(active)),A_eq=B,b_eq=hover,bounds=bounds,method='highs')
            # Max fraction simultaneously away from all surviving-actuator bounds.
            n=sum(active); inequality=[]; rhs=[]
            for j,limit in enumerate(maximum[active]):
                left=np.zeros(n+1); left[j]=-1; left[-1]=limit; inequality.append(left); rhs.append(0)
                left=np.zeros(n+1); left[j]=1; left[-1]=limit; inequality.append(left); rhs.append(limit)
            interior=linprog(np.r_[np.zeros(n),-1],A_ub=inequality,b_ub=rhs,A_eq=np.c_[B,np.zeros(4)],b_eq=hover,bounds=bounds+[(0,.5)],method='highs')
            directions={}
            for axis in range(4):
                for sign in (-1,1):
                    ray=np.eye(4)[axis]*sign
                    result=linprog(np.r_[np.zeros(n),-1],A_eq=np.c_[B,-ray],b_eq=hover,bounds=bounds+[(0,None)],method='highs')
                    directions[f'{axis}:{sign}']=float(max(result.x[-1],0)) if result.success else None
            rows.append({'motor':motor,'remaining_effectiveness':factor,'rank':int(np.linalg.matrix_rank(B)),
                         'zero_moment_hover_feasible':bool(fit.success),'surviving_actuator_interior_fraction':float(interior.x[-1]) if interior.success else None,
                         'hover_nominal_thrusts':fit.x.tolist() if fit.success else None,'directional_wrench_margin':directions})
    angle=2*np.pi/3; transform=np.eye(4); transform[1:3,1:3]=[[np.cos(angle),-np.sin(angle)],[np.sin(angle),np.cos(angle)]]
    symmetry=float(np.max(abs(A[:,np.roll(np.arange(6),-2)]-transform@A)))
    return {'cases':rows,'rotation_120deg_column_symmetry_error':symmetry,'margin_definition':'LP surviving-actuator bound interior plus 8 axis rays; not a certified largest wrench ball'}

def normal(cfg,out):
    acceptance=cfg['acceptance']; write_json(out/'predeclared_acceptance.json',{'config_hash':stable_hash(cfg),'criteria':acceptance,'before_validation_runs':True})
    plant_initial=__import__('hex_research.dynamics',fromlist=['HexPlant']).HexPlant(cfg)
    runs={}; results={}
    for name,task,duration in [('hover','hover',8),('attitude_recovery','hover',8),('rate_recovery','hover',8),('step','step',10),('tracking','tracking',10),('constant_disturbance','hover',10),('saturation_recovery','saturation',12),('observed_noise','tracking',10),('observed_delay','tracking',10)]:
        local=deepcopy(cfg); initial=None; kwargs={}
        if name=='attitude_recovery': initial=plant_initial.initial(q=Rotation.from_euler('xyz',[.15,-.1,.1]).as_quat())
        if name=='rate_recovery': initial=plant_initial.initial(w=(.4,-.3,.2))
        if name=='constant_disturbance': kwargs['force_fn']=lambda t:np.array([1.,-.5,0.]) if 2<=t<6 else np.zeros(3)
        if name=='saturation_recovery': local['propulsion']['omega_max_rad_s']=950.
        if name.startswith('observed'):
            kwargs.update(feedback='observed_feedback',noise=True,wind=True)
            if name=='observed_delay': local['sensor']['feedback_delay_s']=.03
        run=simulate(local,duration,task=task,initial=initial,**kwargs); runs[name]=run; result=metric(run)
        if name in ('tracking','observed_noise','observed_delay'): passed=result['position_rmse_m']<=acceptance['tracking_rmse_m']
        elif name=='saturation_recovery': passed=result['final_position_error_m']<=acceptance['post_saturation_recovery_error_m']
        else: passed=result['final_position_error_m']<=acceptance['hover_final_position_error_m']
        result['passed_research_condition']=bool(passed and run['termination']=='completed' and result['maximum_tilt_deg']<=acceptance['maximum_normal_tilt_deg'])
        results[name]=result; print('normal',name,result,flush=True)
    # Identical output and control clocks; change integration then control clocks.
    original=runs['tracking']; baseline=np.array([r['state'] for r in original['rows']])
    convergence={}
    for key,changed in [('integration','integration_hz'),('control','control_hz')]:
        local=deepcopy(cfg); local['timing'][changed]*=2
        run=simulate(local,10,task='tracking'); values=np.array([r['state'] for r in run['rows']])
        diff=float(np.max(np.linalg.norm(values[:,:3]-baseline[:,:3],axis=1)))
        limit=acceptance[key+'_half_step_position_difference_m']
        convergence[key]={'max_position_difference_m':diff,'threshold_m':limit,'passed':diff<=limit}
        print('convergence',key,convergence[key],flush=True)
    sensitivity={}
    for group,key,factors in [('airframe','center_mass_kg',[.8,1.2]),('airframe','arm_radius_m',[.9,1.1]),('propulsion','kT',[.8,1.2]),('propulsion','tau_s',[.6,1.6]),('propulsion','Q_over_T_m',[.8,1.2])]:
        for factor in factors:
            local=deepcopy(cfg); local[group][key]*=factor
            result=metric(simulate(local,6,task='tracking')); sensitivity[f'{group}.{key}x{factor}']=result
            print('sensitivity',key,factor,result['position_rmse_m'],flush=True)
    report={'normal_cases':results,'step_convergence':convergence,'sensitivity':sensitivity,
            'sensitivity_status':'predeclared_design_variations_not_measured_confidence_intervals',
            'base_research_normal_gate':all(r['passed_research_condition'] for r in results.values()) and all(r['passed'] for r in convergence.values()),
            'real_flight_validation':False,'config_hash':stable_hash(cfg)}
    write_json(out/'normal_validation.json',report)
    import matplotlib; matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(2,2,figsize=(10,7))
    for name in ['tracking','observed_delay','saturation_recovery','attitude_recovery']:
        run=runs[name]; t=np.array([r['time'] for r in run['rows']]); pos=np.array([r['state'][:3] for r in run['rows']]); ref=np.array([r['reference']['x'] for r in run['rows']])
        ax=axes.flat[['tracking','observed_delay','saturation_recovery','attitude_recovery'].index(name)]
        if name=='saturation_recovery': ax.plot(t,pos[:,2],label='actual z'); ax.axvline(4,color='gray',linestyle='--',label='infeasible reference removed')
        else: ax.plot(t,np.linalg.norm(pos-ref,axis=1),label='position error norm')
        ax.set(title=name,xlabel='time [s]',ylabel='m'); ax.legend()
    fig.tight_layout(); fig.savefig(out/'normal_validation.png',dpi=150); plt.close(fig)
    return report

def faults(cfg,out):
    from scenario_config import SCENARIOS
    summaries={}; curves=[]
    for scenario in SCENARIOS:
        schedule=FaultSchedule(scenario.fault_motor,scenario.simulated_factor,3.005) if scenario.label else FaultSchedule()
        run=simulate(cfg,7,schedule=schedule); result=metric(run)
        # Peak lag inconsistency after the actual event, not a false instantaneous target.
        lag=[]
        for row in run['rows']:
            if 3.005<=row['time']<3.255:
                steady=physical(cfg)['kT']*row['effectiveness']*row['control']['cmd_motor_speeds']**2
                lag.append(abs(sum(steady)-row['audit']['dynamic_propulsive_wrench'][0]))
        result['max_initial_steady_actual_thrust_difference_N']=float(max(lag)) if lag else 0.
        summaries[scenario.name]=result; curves.append((scenario,run)); print('fault',scenario.name,result['termination'],result['actual_duration_s'],flush=True)
    cap=capability(cfg); write_json(out/'capability.json',cap)
    write_json(out/'fault_validation.json',{'cases':summaries,'fault_time_s':3.005,'fault_mechanism':'sqrt(lambda) target RPM; dynamic thrust from actual RPM; no second scaling','all_full_factors':{s.name:s.simulated_factor for s in SCENARIOS if s.severity=='full'},'requires_all_full_stable':False})
    import matplotlib; matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(11,4))
    for scenario,run in curves:
        if scenario.severity=='full':
            t=[r['time'] for r in run['rows']]; s=np.array([r['state'] for r in run['rows']])
            axes[0].plot(t,s[:,2],label=f'M{scenario.fault_motor}'); axes[1].plot(t,np.linalg.norm(s[:,10:13],axis=1),label=f'M{scenario.fault_motor}')
    axes[0].set(xlabel='time [s]',ylabel='altitude [m]',title='Full = 0 at every position')
    axes[1].set(xlabel='time [s]',ylabel='body rate norm [rad/s]',title='Same conventional controller; envelope exits retained')
    for ax in axes: ax.axvline(3.005,color='gray',linestyle='--'); ax.legend()
    fig.tight_layout(); fig.savefig(out/'fault_validation.png',dpi=150); plt.close(fig)
    return summaries
