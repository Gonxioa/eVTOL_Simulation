"""Declared deterministic tasks. Quintic progress gives consistent p/v/a.

All numbers here are research experiment designs, not measured flight limits.
The provider may only be sampled once at each control/export task clock tick.
"""
from copy import deepcopy
import numpy as np


def legacy_reference(t, task='hover', offset=(0, 0, 0)):
    center = np.array([0., 0., 2.5]) + offset
    p = center.copy(); v = np.zeros(3); a = np.zeros(3); yaw = 0.; rate = 0.
    if task == 'tracking':
        frequency = .6; amplitude = .4
        p += np.array([amplitude*np.sin(frequency*t), amplitude*(1-np.cos(frequency*t)), .15*np.sin(.4*t)])
        v = np.array([amplitude*frequency*np.cos(frequency*t), amplitude*frequency*np.sin(frequency*t), .06*np.cos(.4*t)])
        a = np.array([-amplitude*frequency**2*np.sin(frequency*t), amplitude*frequency**2*np.cos(frequency*t), -.024*np.sin(.4*t)])
        yaw = .2*np.sin(.3*t); rate = .06*np.cos(.3*t)
    elif task == 'step' and t >= 2:
        p += np.array([.5, -.4, .3])
    elif task == 'saturation' and 2 <= t < 4:
        p += np.array([0, 0, 50.])
    elif task not in ('hover', 'step', 'saturation'):
        raise ValueError('Unknown legacy task: ' + task)
    return {'x': p, 'x_dot': v, 'x_ddot': a, 'yaw': yaw, 'yaw_dot': rate}


class LegacyTask:
    def __init__(self, name):
        self.name = name; self.spec = {'family': 'legacy', 'name': name}
    def reset(self):
        pass
    def at(self, t):
        return legacy_reference(t, self.name)


def quintic(t, start, stop):
    if t <= start:
        return 0., 0., 0.
    if t >= stop:
        return 1., 0., 0.
    length = stop-start; u = (t-start)/length
    return (10*u**3-15*u**4+6*u**5,
            30*u**2*(1-u)**2/length,
            60*u*(1-u)*(1-2*u)/length**2)


class ParameterizedTask:
    def __init__(self, spec):
        self.spec = deepcopy(spec); self.name = spec['family']
        if self.name not in ('hover', 'line', 'circle', 'vertical'):
            raise ValueError('Unsupported parameterized task')
        if not 0 <= spec['motion_start_s'] < spec['motion_end_s'] <= spec['duration_s']:
            raise ValueError('Invalid motion interval')
        for value in spec.values():
            if isinstance(value, (int, float, list)) and not np.isfinite(value).all():
                raise ValueError('Nonfinite task parameter')
        if np.asarray(spec['origin_m']).shape != (3,):
            raise ValueError('Task origin must have three coordinates')
        if self.name == 'circle' and spec['radius_m'] <= 0:
            raise ValueError('Positive circle radius required')
        self.limits = spec['limits']
        self.reset()

    def reset(self):
        self.samples = 0

    def at(self, t):
        if not np.isfinite(t) or t < 0:
            raise ValueError('Task needs a finite nonnegative clock')
        self.samples += 1
        s = self.spec; u, du, ddu = quintic(t, s['motion_start_s'], s['motion_end_s'])
        x = np.array(s['origin_m'], dtype=float); v = np.zeros(3); a = np.zeros(3)
        if self.name in ('line', 'vertical'):
            direction = np.array([np.cos(s['heading_rad']), np.sin(s['heading_rad']), 0.]) if self.name == 'line' else np.array([0., 0., 1.])
            delta = s['displacement_m'] * direction
            x += delta*u; v = delta*du; a = delta*ddu
        elif self.name == 'circle':
            theta0 = s['heading_rad']; angle = theta0+s['turn_rad']*u
            rate = s['turn_rad']*du; alpha = s['turn_rad']*ddu
            radial = np.array([np.cos(angle), np.sin(angle), 0.])
            tangent = np.array([-np.sin(angle), np.cos(angle), 0.])
            x += s['radius_m']*(radial-np.array([np.cos(theta0), np.sin(theta0), 0.]))
            v = s['radius_m']*rate*tangent
            a = s['radius_m']*(alpha*tangent-rate**2*radial)
        yaw = s['yaw_start_rad']+s['yaw_change_rad']*u
        yaw_dot = s['yaw_change_rad']*du
        limit = self.limits
        if (np.linalg.norm(v) > limit['speed_m_s']+1e-10 or
                np.linalg.norm(a) > limit['acceleration_m_s2']+1e-10 or
                abs(yaw_dot) > limit['yaw_rate_rad_s']+1e-10 or
                np.linalg.norm(x[:2]) > limit['horizontal_radius_m']+1e-10 or
                not limit['altitude_m'][0]-1e-10 <= x[2] <= limit['altitude_m'][1]+1e-10):
            raise ValueError('Task reference exceeds its declared research bounds')
        return {'x': x, 'x_dot': v, 'x_ddot': a, 'yaw': yaw, 'yaw_dot': yaw_dot}


def task_library(settings, groups_per_family=None):
    """Five independent seed streams per paired group, never per fault label."""
    count = settings['groups_per_family'] if groups_per_family is None else groups_per_family
    if count < 1:
        raise ValueError('Positive group count required')
    result = []; group = 0
    for family in settings['families']:
        for repeat in range(count):
            group += 1
            children = np.random.SeedSequence([settings['seed'], group]).spawn(5)
            seeds = {name: int(child.generate_state(1)[0]) for name, child in zip(('task', 'initial', 'wind', 'sensor', 'fault'), children)}
            rng = np.random.default_rng(seeds['task']); ranges = settings['ranges']
            draw = lambda key: float(rng.uniform(*ranges[key]))
            spec = {'family': family, 'duration_s': settings['duration_s'],
                    'motion_start_s': settings['motion_start_s'], 'motion_end_s': settings['motion_end_s'],
                    'origin_m': [0., 0., draw('height_m')], 'heading_rad': draw('heading_rad'),
                    'yaw_start_rad': draw('yaw_rad'), 'yaw_change_rad': draw('yaw_change_rad'),
                    'limits': deepcopy(settings['reference_limits'])}
            if family == 'line':
                spec['displacement_m'] = draw('line_distance_m')
            if family == 'vertical':
                spec['displacement_m'] = draw('vertical_distance_m')
            if family == 'circle':
                spec['radius_m'] = draw('circle_radius_m')
                spec['turn_rad'] = float(rng.choice([-1, 1])) * 2*np.pi
            provider = ParameterizedTask(spec)
            for t in np.linspace(0, settings['duration_s'], 1001):
                provider.at(t)
            initial_rng = np.random.default_rng(seeds['initial'])
            result.append({'group_id': group, 'task': spec, 'seed_streams': seeds,
                           'initial_position_offset_m': initial_rng.uniform(-settings['initial_position_jitter_m'], settings['initial_position_jitter_m'], 3).tolist(),
                           'initial_attitude_rotvec_rad': initial_rng.uniform(-settings['initial_attitude_jitter_rad'], settings['initial_attitude_jitter_rad'], 3).tolist(),
                           'fault_time_s': float(np.random.default_rng(seeds['fault']).uniform(*settings['fault_time_range_s'])),
                           'repeat': repeat})
    return result
