"""One explicit research configuration and physically constrained derived geometry."""
from copy import deepcopy
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DEFAULT = Path(__file__).parent/'configs/research.json'

def load(path=DEFAULT):
    cfg = json.loads(Path(path).read_text(encoding='utf-8'))
    if cfg['parameter_status'] != 'research_design_not_real_aircraft':
        raise ValueError('This configuration has no real-aircraft confirmation')
    return cfg

def physical(cfg):
    if cfg['frame']!='ENU_world_FLU_body_xyzw_body_to_world' or cfg['parameter_status']!='research_design_not_real_aircraft':
        raise ValueError('Unsupported coordinate/evidence identity for this research model')
    def finite(value):
        if isinstance(value,dict): return all(finite(v) for v in value.values())
        if isinstance(value,list): return all(finite(v) for v in value)
        if isinstance(value,(int,float)): return np.isfinite(value)
        return True
    if not finite(cfg): raise ValueError('Nonfinite configuration value')
    for group in ('control','sensor','wind','timing','aerodynamics'):
        for key,value in cfg[group].items():
            if isinstance(value,list) and key not in ('installation_rotation_xyzw',) and len(value)!=3: raise ValueError('Three-axis configuration shape mismatch')
    if cfg['airframe']['gravity_m_s2']<=0 or cfg['wind']['ou_correlation_time_s']<=0: raise ValueError('Positive gravity/correlation time required')
    if not 0<cfg['control']['tilt_limit_deg']<90 or cfg['control']['derivative_filter_hz']<=0: raise ValueError('Invalid control tilt/bandwidth')
    for key,value in cfg['control'].items():
        if isinstance(value,list) and np.any(np.asarray(value)<0): raise ValueError('Nonnegative gains and limits required')
    for key,value in cfg['sensor'].items():
        if key not in ('lever_m','installation_rotation_xyzw') and isinstance(value,(int,float)) and value<0: raise ValueError('Nonnegative sensor-error parameters required')
    if any(v<=0 for v in cfg['timing'].values()) or cfg['sensor']['rate_hz']<=0 or cfg['wind']['update_rate_hz']<=0: raise ValueError('Positive clock rates required')
    if cfg['wind']['ou_stationary_std_m_s']<0 or any(np.any(np.asarray(v)<0) for v in cfg['aerodynamics'].values() if isinstance(v,list)): raise ValueError('Nonnegative stochastic/aero coefficients required')
    a, p = cfg['airframe'], cfg['propulsion']
    n = 6
    phi = np.arange(n)*np.pi/3
    r = a['arm_radius_m']*np.c_[np.cos(phi),np.sin(phi),np.zeros(n)]
    central, hub = a['center_mass_kg'], a['hub_mass_kg']
    if central<=0 or hub<0 or a['arm_radius_m']<=0 or a['center_radius_m']<=0 or a['center_height_m']<=0: raise ValueError('Invalid mass-distribution design')
    # Uniform central cylinder + six point hub masses; arms/load lumped centrally.
    mass = central+n*hub
    radius, height = a['center_radius_m'], a['center_height_m']
    J = np.diag([central*(3*radius**2+height**2)/12]*2+[central*radius**2/2])
    for ri in r:
        J += hub*((ri@ri)*np.eye(3)-np.outer(ri,ri))
    d = np.array([1,-1,1,-1,1,-1])  # BODY reaction-torque sign, rotor spin = -d.
    eigen = np.linalg.eigvalsh(J)
    if mass<=0 or np.any(eigen<=0) or eigen[-1] > eigen[:-1].sum()+1e-12:
        raise ValueError('Nonphysical mass or inertia triangle')
    if 2*a['rotor_radius_m'] >= a['arm_radius_m']:
        raise ValueError('Adjacent rotor disks overlap in this coplanar design')
    if p['kT']<=0 or p['Q_over_T_m']<=0 or p['tau_s']<=0 or not 0<=p['omega_min_rad_s']<p['omega_max_rad_s']:
        raise ValueError('Invalid propulsion design')
    return {'mass':mass,'inertia':J,'rotor_positions':r,'yaw_sign':d,'num_rotors':n,
            'kT':np.full(n,p['kT']),'kQ':np.full(n,p['kT']*p['Q_over_T_m']),
            'gravity':a['gravity_m_s2'],'tau':np.full(n,p['tau_s']),
            'omega_min':p['omega_min_rad_s'],'omega_max':p['omega_max_rad_s']}

def matrix(p):
    r=p['rotor_positions']
    return np.vstack([np.ones(6),r[:,1],-r[:,0],p['yaw_sign']*p['kQ']/p['kT']])

def rotorpy_fields(cfg=None):
    cfg=load() if cfg is None else cfg; p=physical(cfg); J=p['inertia']
    return {'mass':p['mass'],'Ixx':J[0,0],'Iyy':J[1,1],'Izz':J[2,2],
            'Ixy':J[0,1],'Iyz':J[1,2],'Ixz':J[0,2],'num_rotors':6,
            'rotor_pos':{f'r{i+1}':p['rotor_positions'][i].copy() for i in range(6)},
            'rotor_directions':p['yaw_sign'].copy(),'rotor_radius':cfg['airframe']['rotor_radius_m'],
            'rI':np.array(cfg['sensor']['lever_m']),'c_Dx':cfg['aerodynamics']['body_drag_kg_m'][0],
            'c_Dy':cfg['aerodynamics']['body_drag_kg_m'][1],'c_Dz':cfg['aerodynamics']['body_drag_kg_m'][2],
            'k_eta':p['kT'][0],'k_m':p['kQ'][0],'k_d':cfg['aerodynamics']['rotor_drag_kg_per_rad'][0],
            'k_z':cfg['aerodynamics']['rotor_drag_kg_per_rad'][2],'k_h':0.0,'k_flap':0.0,
            'tau_m':p['tau'][0],'rotor_speed_min':p['omega_min'],'rotor_speed_max':p['omega_max'],
            'parameter_status':cfg['parameter_status'],'model_version':cfg['model_version']}
