"""Reproducible development plans for the native Simulink control repair."""
import argparse
import json
import hashlib
import itertools
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
HISTORY = Path('C:/Users/lenovo/Desktop/eVTOL_Simulation')


def development_plan(output):
    symmetry = HISTORY / 'output_simulink_diagnostics/m1_m4_symmetry_20261106'
    core = HISTORY / 'simulink/output/simulations/control_v3_20261023'
    baselines = {
        'calm_hover_m1': symmetry / 'hover_calm_m1',
        'calm_hover_m4': symmetry / 'hover_calm_m4',
        'noisy_hover_m1': symmetry / 'hover_wind_noise_m1',
        'noisy_hover_m4': symmetry / 'hover_wind_noise_m4',
        'noisy_circle_m1': core / 'core_task2_m1_eta00_seed20261006',
        'noisy_circle_m4': core / 'core_task2_m4_eta00_seed20261006',
        'noisy_line_m1': core / 'core_task1_m1_eta00_seed20261006',
        'noisy_vertical_m1': core / 'core_task3_m1_eta00_seed20261006',
        'noisy_circle_eta20': core / 'core_task2_m1_eta20_seed20261006',
        'noisy_circle_eta80': core / 'core_task2_m1_eta80_seed20261006',
        'healthy_circle': core / 'healthy_four_tasks_20261027/02_circle',
    }
    conditions = [dict(condition=name, baseline_mat=str(path / 'result.mat')) for name, path in baselines.items()]
    for name, motor, force, wind in [
        ('healthy_force', 0, [.5, -.3, 0], [0, 0, 0]),
        ('fault_force', 1, [.5, -.3, 0], [0, 0, 0]),
        ('healthy_wind', 0, [0, 0, 0], [1, .5, 0]),
        ('fault_wind', 1, [0, 0, 0], [1, .5, 0]),
    ]:
        conditions.append(dict(condition=name, baseline_mat='', task=0, motor=motor,
                               factor=0 if motor else 1, wind=False, noise=False,
                               force=force, mean_wind=wind))
    # Homogeneous JSON fields keep MATLAB's decoded case array a struct array.
    for c in conditions:
        c.setdefault('force', [0, 0, 0])
        c.setdefault('mean_wind', None)
        for key, value in dict(task=0, motor=0, factor=1, wind=False, noise=False).items():
            c.setdefault(key, value)
    variants = [('original', False, False, 1.6), ('rate_only', True, False, 1.6),
                ('velocity_only', False, True, 1.6), ('both_k08', True, True, .8),
                ('both_k16', True, True, 1.6)]
    cases = []
    for variant, rate, velocity, ki in variants:
        for condition in conditions:
            row = dict(condition, name=variant+'__'+condition['condition'], variant=variant,
                       rate=rate, velocity=velocity, ki=ki)
            cases.append(row)
    output.mkdir(parents=True, exist_ok=True)
    path = output / 'plan.json'
    plan = dict(profile='control_repair_development_v1', cases=cases,
                selection='Require all current acceptance criteria and bounded integrals. Compare median, p90, maximum recovery, late errors and peak attitude/rates. Retain every regression.',
                independent_final_seeds=[907001, 907003, 907009])
    with path.open('x', encoding='utf-8') as handle:
        json.dump(plan, handle, ensure_ascii=False, indent=2)
    return path


def validation_plan(output, preflight=False):
    """Declare conditions before execution; independence is at nuisance-seed level."""
    cases = []
    def add(suite, task, motor, factor, seed=20261006, noise=True, **kwargs):
        row = dict(suite=suite, task=task, motor=motor, factor=factor, seed=seed,
                   noise=noise, wind=noise, integration_dt=.002, fault_start=3.,
                   heading=0., mirror=0, force=[0.,0.,0.])
        row.update(kwargs)
        row['name'] = f'{suite}_t{task}_m{motor}_e{round(factor*100):03d}_s{seed}'
        if any(r['name']==row['name'] for r in cases):
            raise ValueError('Duplicate native condition identifier')
        cases.append(row)
    if not preflight:
        for seed in [20261006,20261007,20261008,907001,907003,907009]:
            suite='development' if seed>10000000 else 'independent'
            for task in range(4):
                add(suite,task,0,1.,seed)
                for motor,factor in itertools.product(range(1,7),[.8,.2,0.]):
                    add(suite,task,motor,factor,seed)
        for task in range(4):
            add('ideal',task,0,1.,noise=False)
            for motor,factor in itertools.product(range(1,7),[.8,.2,0.]):
                add('ideal',task,motor,factor,noise=False)
        import math
        for motor,noise in itertools.product([1,2,3],[False,True]):
            phi=(motor-1)*math.pi/3;tag=f'moving_mirror_{motor}_{motor+3}_{int(noise)}'
            add(tag+'_original',1,motor,0.,noise=noise)
            add(tag+'_reflected',1,motor+3,0.,noise=noise,mirror=motor,heading=math.pi+2*phi)
    else:
        import math
        # Reflection normal along each failed arm maps it to its opposite.
        for motor,noise in itertools.product([1,2,3],[False,True]):
            opposite=motor+3
            phi=(motor-1)*math.pi/3
            # Hover eliminates a task-direction ambiguity; all disturbances,
            # polar/axial vectors and motor channels are reflected together.
            tag=f'mirror_{motor}_{opposite}_{int(noise)}'
            add(tag+'_original',0,motor,0.,noise=noise)
            add(tag+'_reflected',0,opposite,0.,noise=noise,mirror=motor)
        for task,motor in [(0,1),(0,4),(2,1),(2,4),(1,1),(3,1)]:
            for dt in [.002,.001]:
                add('convergence_'+str(dt),task,motor,0.,integration_dt=dt)
        for task,motor,onset in itertools.product(range(4),[1,4],[1.5,5.5]):
            add('fault_time_'+str(onset),task,motor,0.,fault_start=onset)
    output.mkdir(parents=True,exist_ok=True)
    build=json.loads((ROOT/'simulink/hex_model_build.json').read_text(encoding='utf-8-sig'))
    plan=dict(profile='hex_control_repair_native_v4',controller_version='control_recovery_v4_20261122',
              source_signature=build['source_signature'],cases=cases,
              independent_final_seeds=[907001,907003,907009],
              feedback='observed, 30 ms in ideal and stochastic conditions',
              selection='Frozen after 75 paired ablations; minimum median/p90 recovery with all existing flight limits; retain all regressions.',
              mirror='polar S=I-2nnT; axial det(S)S; motor permutation induced by S; hover; quaternion matrices compared as S R S',
              convergence_limits=dict(max_position_difference_m=.02,max_recovery_difference_s=.1))
    path=output/'plan.json'
    with path.open('x',encoding='utf-8') as f: json.dump(plan,f,ensure_ascii=False,indent=2)
    return path


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('command', choices=['develop-plan','preflight-plan','validation-plan'])
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.command=='develop-plan': print(development_plan(args.output.resolve()))
    else: print(validation_plan(args.output.resolve(),args.command=='preflight-plan'))
