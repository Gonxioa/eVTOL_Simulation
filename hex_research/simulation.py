"""Explicit event clock, plant-only fault schedules, atomic versioned data."""
from dataclasses import dataclass
from pathlib import Path
import time,json,subprocess
import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation
from provenance import stable_hash,sha256_file,write_json,runtime_info
from .configuration import physical,ROOT
from .dynamics import HexPlant
from .control import CascadeController
from .observation import Sensors,OUWind
from .math3d import rotation

@dataclass(frozen=True)
class FaultSchedule:
    motor:int=-1
    factor:float=1.
    start:float|None=None
    end:float|None=None
    def __post_init__(self):
        if self.motor not in range(-1,6) or not np.isfinite(self.factor) or not 0<=self.factor<=1: raise ValueError('Invalid fault schedule')
        if self.start is not None and (not np.isfinite(self.start) or self.start<0): raise ValueError('Invalid fault time')
        if self.end is not None and (self.start is None or not np.isfinite(self.end) or self.end<=self.start): raise ValueError('Invalid recovery time')
    def at(self,t):
        if not np.isfinite(t): raise ValueError('Finite fault clock required')
        eta=np.ones(6)
        if self.motor>=0 and self.start is not None and t>=self.start and (self.end is None or t<self.end): eta[self.motor]=self.factor
        return eta
    def events(self,duration): return [t for t in (self.start,self.end) if t is not None and 0<t<duration]

def reference(t,task='hover',offset=(0,0,0)):
    center=np.array([0.,0.,2.5])+offset
    p=center.copy(); v=np.zeros(3); a=np.zeros(3); yaw=0.; rate=0.
    if task=='tracking':
        frequency=.6; amplitude=.4
        p+=np.array([amplitude*np.sin(frequency*t),amplitude*(1-np.cos(frequency*t)),.15*np.sin(.4*t)])
        v=np.array([amplitude*frequency*np.cos(frequency*t),amplitude*frequency*np.sin(frequency*t),.06*np.cos(.4*t)])
        a=np.array([-amplitude*frequency**2*np.sin(frequency*t),amplitude*frequency**2*np.cos(frequency*t),-.024*np.sin(.4*t)])
        yaw=.2*np.sin(.3*t); rate=.06*np.cos(.3*t)
    elif task=='step' and t>=2: p+=np.array([.5,-.4,.3])
    elif task=='saturation' and 2<=t<4: p+=np.array([0,0,50.])
    return {'x':p,'x_dot':v,'x_ddot':a,'yaw':yaw,'yaw_dot':rate}

def clocks(duration,rate):
    count=int(np.floor(duration*rate+1e-9))
    return np.arange(count+1,dtype=float)/rate

def simulate(cfg,duration=8.,schedule=None,seed=None,feedback='ideal_feedback',noise=False,wind=False,
             task='hover',initial=None,force_fn=None,torque_fn=None):
    if feedback not in ('ideal_feedback','observed_feedback'): raise ValueError('Unknown feedback mode')
    if duration<=0 or not np.isfinite(duration): raise ValueError('Positive duration')
    plant=HexPlant(cfg); controller=CascadeController(cfg); schedule=schedule or FaultSchedule()
    if abs(duration*cfg['timing']['export_hz']-round(duration*cfg['timing']['export_hz']))>1e-8:
        raise ValueError('Duration must align with export clock; fault/recovery times may remain off-grid')
    seed=cfg['seed'] if seed is None else seed
    streams=np.random.SeedSequence(seed).spawn(2)
    sensors=Sensors(cfg,streams[0]); wind_process=OUWind(cfg,streams[1],wind)
    s=plant.initial() if initial is None else np.asarray(initial,dtype=float).copy()
    plant.unpack(s)
    timing=cfg['timing']; sensor_rate=cfg['sensor']['rate_hz']
    if sensor_rate!=cfg['wind']['update_rate_hz']: raise ValueError('Current implementation requires matching sensor and wind clocks')
    grids=[clocks(duration,r) for r in (timing['control_hz'],sensor_rate,timing['export_hz'])]
    # Integer tick identity is represented as rounded event times. Include exact
    # non-grid fault/recovery boundaries so effectiveness never changes early.
    # Grid ticks are canonicalized, but fault/recovery boundaries retain their
    # EXACT input float. Rounding a fault down can otherwise postpone its
    # activation to the next tick while logging it as already active.
    events=np.unique(np.r_[np.round(np.r_[*grids],12),schedule.events(duration),duration])
    control_times=set(np.round(grids[0],12)); sensor_times=set(np.round(grids[1],12)); export_times=set(np.round(grids[2],12))
    command=s[13:].copy(); wind_value=np.zeros(3); history=[]; rows=[]; controls=None
    last_measurement=None; reason='completed'; started=time.perf_counter(); evaluations=0
    force_fn=force_fn or (lambda t:np.zeros(3)); torque_fn=torque_fn or (lambda t:np.zeros(3))
    for i,t in enumerate(events):
        eta=schedule.at(t); force=force_fn(t); torque=torque_fn(t)
        if t in sensor_times:
            wind_value=wind_process.update(t)
            derivative=plant.derivative(t,s,command,eta,wind_value,force,torque)
            last_measurement=sensors.sample(t,s,derivative,noise)
            history.append(last_measurement)
        if t in control_times:
            if feedback=='ideal_feedback':
                observed=plant.unpack(s); observed['time']=float(t)
            else:
                eligible=[h for h in history if h['time']<=t-cfg['sensor']['feedback_delay_s']+1e-12]
                observed=eligible[-1] if eligible else history[0]
            controls=controller.update(float(t),observed,reference(t,task)); command=controls['cmd_motor_speeds']; evaluations+=1
        if t in export_times:
            _,_,audit=plant.wrench(s,wind_value,force)
            actual=plant.derivative(t,s,command,eta,wind_value,force,torque)
            ref=reference(t,task)
            rows.append({'time':float(t),'state':s.copy(),'measurement':{k:np.array(v).copy() if isinstance(v,np.ndarray) else v for k,v in last_measurement.items()},
                         'control':controls.copy(),'effectiveness':eta.copy(),'wind':wind_value.copy(),'reference':ref,
                         'feedback_timestamp':float(observed['time']),'feedback_age_s':float(t-observed['time']),
                         'acceleration_world':actual[3:6].copy(),'body_alpha':actual[10:13].copy(),
                         'audit':audit,'total_moment_body':plant.wrench(s,wind_value,force)[1]+torque})
        bound=cfg['termination']
        if not np.isfinite(s).all(): raise FloatingPointError('Nonfinite physical state; never fabricate output')
        if np.linalg.norm(s[:3])>bound['position_norm_m'] or np.linalg.norm(s[3:6])>bound['speed_m_s'] or np.linalg.norm(s[10:13])>bound['body_rate_rad_s'] or s[2]<bound['minimum_altitude_m']:
            reason='physical_envelope_exit'; break
        if i+1<len(events): s=plant.integrate(s,t,events[i+1],command,schedule,wind_value,force,torque)
    return {'rows':rows,'termination':reason,'requested_duration_s':duration,'actual_duration_s':float(rows[-1]['time']),
            'feedback_mode':feedback,'noise_enabled':noise,'wind_enabled':wind,'task':task,'seed':int(seed),
            'elapsed_s':time.perf_counter()-started,'controller_evaluations':evaluations,'config':cfg,
            'config_hash':stable_hash(cfg),'independent_flight':True}

def fingerprint(cfg):
    names=['main.py','scenario_config.py','control_allocation.py',*['hex_research/'+p.relative_to(ROOT/'hex_research').as_posix() for p in (ROOT/'hex_research').rglob('*.py')]]
    return {'model_version':cfg['model_version'],'config_hash':stable_hash(cfg),
            'code_hashes':{n:sha256_file(ROOT/n) for n in sorted(names)},
            'code_head':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
            'parameter_status':cfg['parameter_status'],'runtime':runtime_info()}

def export(run,path,scenario,group_id):
    p=physical(run['config']); rows=[]
    for r in run['rows']:
        s=r['state']; obs=r['measurement']; c=r['control']; eta=r['effectiveness']; actual=r['audit']; cmd=c['cmd_motor_speeds']
        euler=Rotation.from_quat(obs['q']).as_euler('xyz'); truth_euler=Rotation.from_quat(s[6:10]).as_euler('xyz')
        steady=p['kT']*eta*cmd**2
        row={'time':r['time'],'roll':euler[0],'pitch':euler[1],'yaw':euler[2],'p':obs['w'][0],'q_body':obs['w'][1],'r':obs['w'][2],
             'ax':obs['accel_sensor'][0],'ay':obs['accel_sensor'][1],'az':obs['accel_sensor'][2],'gx':obs['gyro_sensor'][0],'gy':obs['gyro_sensor'][1],'gz':obs['gyro_sensor'][2],
             'cmd_thrust':c['desired_wrench'][0],'predicted_plant_thrust':sum(steady),'steady_fault_predicted_thrust':sum(steady),'dynamic_propulsive_thrust':actual['dynamic_propulsive_wrench'][0],
             'fault_label':scenario.label,'effective_label':scenario.label if np.any(eta<1) else 0,'fault_motor':scenario.fault_motor,'fault_factor':scenario.simulated_factor,
             'fault_time':getattr(run.get('schedule'), 'start',None),'scenario_name':scenario.name,'run_id':group_id,'random_seed':run['seed'],'trajectory_id':run['task'],
             'feedback_timestamp':r['feedback_timestamp'],'feedback_age_s':r['feedback_age_s'],'model_version':p and run['config']['model_version'],'config_hash':run['config_hash']}
        for j,axis in enumerate('xyz'):
            row['truth_'+axis]=s[j]; row['truth_v'+axis]=s[j+3]; row['truth_w'+axis]=s[j+10]; row['wind_'+axis]=r['wind'][j]
            row['truth_angle_'+axis]=truth_euler[j]
            for prefix,values in [('desired_moment',c['desired_wrench'][1:]),('nominal_moment',c['nominal_allocated_wrench'][1:]),('steady_fault_moment',np.array([*np.cross(p['rotor_positions'],np.c_[np.zeros((6,2)),steady]).sum(axis=0)[:2],sum(p['yaw_sign']*p['kQ']/p['kT']*steady)])),('dynamic_propulsive_moment',actual['dynamic_propulsive_wrench'][1:]),('total_dynamic_moment',r['total_moment_body'])]: row[f'{prefix}_{axis}']=values[j]
        for motor in range(6):
            row[f'cmd_motor{motor}']=cmd[motor]; row[f'truth_rotor{motor}']=s[13+motor]
            row[f'measured_rotor{motor}']=obs['rotor_measurement'][motor]; row[f'effectiveness{motor}']=eta[motor]; row[f'dynamic_rotor_thrust{motor}']=actual['dynamic_rotor_thrusts'][motor]
        rows.append(row)
    df=pd.DataFrame(rows); df['yaw']=np.unwrap(df['yaw'])
    if not np.isfinite(df.select_dtypes(include=np.number).drop(columns=['fault_time'],errors='ignore').to_numpy()).all(): raise ValueError('Nonfinite export')
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True); tmp=path.with_suffix('.csv.tmp'); df.to_csv(tmp,index=False); tmp.replace(path)
    side={'schema_version':4,'config':run['config'],'provenance':fingerprint(run['config']),'csv_sha256':sha256_file(path),
          'termination':run['termination'],'actual_duration_s':run['actual_duration_s'],'requested_duration_s':run['requested_duration_s'],
          'feedback_mode':run['feedback_mode'],'feature_state_identity':'direct_observation_not_truth_or_EKF',
          'physics_target_identity':'steady_fault_command_prediction_not_actual_dynamic_thrust','scenario':scenario.__dict__,
          'group_id':group_id,'rows':len(df),'normal_matching_fault_time':run.get('matching_fault_time'),
          'resume_signature':stable_hash({'fingerprint':fingerprint(run['config'])['code_hashes'],'config':run['config'],'seed':run['seed'],'group':group_id,'scenario':scenario.__dict__})}
    write_json(path.with_suffix('.json'),side)
    return df,side
