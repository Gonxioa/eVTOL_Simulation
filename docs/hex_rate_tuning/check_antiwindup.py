"""Predeclared saturation/recovery exercises; retain traces and real results."""
import json
import argparse
from copy import deepcopy
from pathlib import Path
import sys
import numpy as np
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from provenance import stable_hash,write_json,sha256_file
from hex_research.simulation import simulate
from hex_research.dynamics import HexPlant
from hex_research.control_tuning import trace
from hex_research.validation import metric


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path,default=ROOT/'output_hex_rate_tuning_20261010/antiwindup_complete')
    parser.add_argument('--config',type=Path,default=ROOT/'output_hex_rate_tuning_20261010/configs/selected_rate_v2.json')
    args=parser.parse_args();out=args.out
    if out.exists():raise RuntimeError('Preserve prior stress results; choose a new output')
    out.mkdir()
    cfg=json.loads(args.config.read_text(encoding='utf-8'))
    spec={'initial_yaw_rate_rad_s':2.,'rate_recovery_duration_s':16.,'saturation_omega_max_rad_s':950.,
          'saturation_task':'legacy saturation: z reference50m from2s to4s, then2.5m',
          'saturation_recovery_duration_s':16.,'final_position_error_m':.25,
          'yaw_integral_tail_absolute_rad_s2':.1,'velocity_integral_tail_norm_m_s2':.1,
          'note':'research stress cases, not a new physical calibration or operational maneuver'}
    write_json(out/'predeclared.json',{'config_hash':stable_hash(cfg),'spec':spec})
    rate=simulate(cfg,16.,initial=HexPlant(cfg).initial(w=(0,0,2.)))
    local=deepcopy(cfg);local['propulsion']['omega_max_rad_s']=950.
    saturation=simulate(local,16.,task='saturation')
    results=[]
    for name,run in [('yaw_rate_recovery',rate),('collective_saturation_recovery',saturation)]:
        trace(out/(name+'.csv'),run);m=metric(run);rows=run['rows']
        blocked=np.array([r['control']['rate_integral_blocked_nominal'] for r in rows])
        rates=np.array([r['control']['rate_integral'] for r in rows])
        velocities=np.array([r['control']['velocity_integral'] for r in rows])
        steps=np.array([r['control']['rate_integral_actual_step'] for r in rows])
        bound=np.array([r['control']['allocator_bound_count'] for r in rows])
        checks={'completed':run['termination']=='completed','final_recovery':m['final_position_error_m']<=.25,
                'blocked_rate_integral_steps_zero':bool(np.all(abs(steps[blocked])<1e-12))}
        if name=='yaw_rate_recovery':
            checks['actual_rate_block_observed']=bool(np.any(blocked[:,2]))
            checks['yaw_integral_returns_near_zero']=bool(abs(rates[-1,2])<=.1)
        else:
            checks['actual_motor_saturation_observed']=bool(np.any(bound>0))
            checks['last_two_seconds_unsaturated']=not bool(np.any(bound[-201:]>0))
            checks['velocity_integral_returns_near_zero']=bool(np.linalg.norm(velocities[-1])<=.1)
        result={'name':name,'config_hash':run['config_hash'],'checks':checks,'passed':all(checks.values()),
                **m,'rate_blocked_samples':np.sum(blocked,axis=0).tolist(),
                'maximum_abs_rate_integral':np.max(abs(rates),axis=0).tolist(),
                'final_rate_integral':rates[-1].tolist(),'maximum_abs_velocity_integral':np.max(abs(velocities),axis=0).tolist(),
                'final_velocity_integral':velocities[-1].tolist(),'bound_samples':int(np.sum(bound>0)),
                'trace_sha256':sha256_file(out/(name+'.csv'))}
        write_json(out/(name+'.json'),result);results.append(result);print(json.dumps(result),flush=True)
    write_json(out/'summary.json',{'spec':spec,'results':results,'all_passed':all(r['passed'] for r in results)})


if __name__=='__main__':main()
