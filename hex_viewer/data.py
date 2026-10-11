"""Read saved CSV/sidecars, retaining their real clocks and early endings."""
import csv
import hashlib
import io
import json
import math
import re
from functools import lru_cache
from pathlib import Path

FAMILIES = {'hover': '悬停', 'line': '水平直线', 'circle': '水平绕圈', 'vertical': '竖直升降'}
SERIES = {
    'position': ['truth_' + a for a in 'xyz'],
    'velocity': ['truth_v' + a for a in 'xyz'],
    'attitude': ['truth_angle_' + a for a in 'xyz'],
    'body_rate': ['truth_w' + a for a in 'xyz'],
    'reference': ['reference_position_' + a for a in 'xyz'],
    'command': [f'cmd_motor{i}' for i in range(6)],
    'rotor_speed': [f'truth_rotor{i}' for i in range(6)],
    'thrust': [f'dynamic_rotor_thrust{i}' for i in range(6)],
    'effectiveness': [f'effectiveness{i}' for i in range(6)],
}


def label_name(label):
    if label == 0:
        return 'Normal · 健康'
    return f"{('Full', 'Severe', 'Partial')[(label - 1) // 6]} · M{(label - 1) % 6}"


def planned_reference(spec, t):
    """Display the saved plan beyond an early ending; not fabricated flight data."""
    u = min(1., max(0., (t-spec['motion_start_s']) /
                          (spec['motion_end_s']-spec['motion_start_s'])))
    s = 10*u**3-15*u**4+6*u**5
    p = list(spec['origin_m'])
    if spec['family'] == 'line':
        p[0] += spec['displacement_m']*math.cos(spec['heading_rad'])*s
        p[1] += spec['displacement_m']*math.sin(spec['heading_rad'])*s
    elif spec['family'] == 'vertical':
        p[2] += spec['displacement_m']*s
    elif spec['family'] == 'circle':
        theta = spec['heading_rad']; turn = spec['turn_rad']*s
        p[0] += spec['radius_m']*(math.cos(theta+turn)-math.cos(theta))
        p[1] += spec['radius_m']*(math.sin(theta+turn)-math.sin(theta))
    return p


class ReplayDataset:
    def __init__(self, folder, predictions=None):
        self.folder = Path(folder).resolve()
        if not self.folder.is_dir():
            raise ValueError(f'数据目录不存在：{self.folder}')
        self.predictions = Path(predictions).resolve() if predictions else None
        self.entries = {}
        for path in sorted(self.folder.glob('g*_c*.json')):
            if not re.fullmatch(r'g\d{3}_(hover|line|circle|vertical)_c\d{2}\.json', path.name):
                continue
            side = json.loads(path.read_text(encoding='utf-8'))
            if side.get('schema_version') != 5:
                raise ValueError(f'需要v5记录：{path.name}')
            group = int(side['group']['group_id']); label = int(side['scenario']['label'])
            if not 0 <= label <= 18 or side['record_id'] != f'g{group:03d}_c{label:02d}':
                raise ValueError(f'记录标识不一致：{path.name}')
            if side['task']['family'] not in FAMILIES:
                raise ValueError(f'未知任务：{path.name}')
            if path.name != f'g{group:03d}_{side["task"]["family"]}_c{label:02d}.json':
                raise ValueError(f'文件名与记录标识不一致：{path.name}')
            if side['config']['frame'] != 'ENU_world_FLU_body_xyzw_body_to_world':
                raise ValueError(f'坐标定义不支持：{path.name}')
            key = (group, label)
            if key in self.entries:
                raise ValueError(f'记录重复：{key}')
            self.entries[key] = (path, side)
        if not self.entries:
            raise ValueError('没有找到v5 CSV及同名JSON记录。请先生成数据或解压已有交付包。')

    def catalog(self):
        groups = {}
        records = []
        for (group, label), (_, side) in sorted(self.entries.items()):
            groups.setdefault(group, {'id': group, 'name': f'g{group:03d} · {FAMILIES[side["task"]["family"]]}',
                                       'family': side['task']['family'], 'task': side['task'], 'labels': []})
            groups[group]['labels'].append(label)
            records.append({'group': group, 'label': label, 'name': label_name(label),
                            'record_id': side['record_id'], 'end_s': side['actual_duration_s'],
                            'termination': side['termination']})
        return {'groups': list(groups.values()), 'records': records, 'count': len(records),
                'classes': [{'label': i, 'name': label_name(i)} for i in range(19)]}

    @lru_cache(maxsize=8)
    def record(self, group, label):
        if (group, label) not in self.entries:
            raise ValueError('该组没有所选状态的记录。')
        side_path, side = self.entries[group, label]
        path = side_path.with_suffix('.csv')
        raw = path.read_bytes()
        digest = hashlib.sha256(raw).hexdigest()
        if digest != side['csv_sha256']:
            raise ValueError(f'CSV哈希与原记录不符：{path.name}')
        series = {k: [] for k in ['time', 'reference_yaw', *SERIES]}
        required = ['time', 'reference_yaw', *[name for values in SERIES.values() for name in values]]
        with io.StringIO(raw.decode('utf-8-sig'), newline='') as f:
            reader = csv.DictReader(f)
            missing = set(required)-set(reader.fieldnames or [])
            if missing:
                raise ValueError('缺少回放列：'+', '.join(sorted(missing)))
            for row in reader:
                values = {name: float(row[name]) for name in required}
                if not all(math.isfinite(v) for v in values.values()):
                    raise ValueError(f'回放列包含无效值：{path.name}')
                series['time'].append(values['time']); series['reference_yaw'].append(values['reference_yaw'])
                for name, columns in SERIES.items():
                    series[name].append([values[c] for c in columns])
        times = series['time']
        if not times or times[0] != 0 or any(b <= a for a, b in zip(times, times[1:])):
            raise ValueError('回放时间必须从0开始并严格递增。')
        if abs(times[-1]-side['actual_duration_s']) > 1e-9 or len(times) != side['rows']:
            raise ValueError('CSV实际起止时间/行数与记录不一致。')
        config = side['config']; limit = config['termination']
        position = series['position'][-1]; velocity = series['velocity'][-1]; rates = series['body_rate'][-1]
        norm = lambda values: math.sqrt(sum(v*v for v in values))
        reasons = []
        if side['termination'] != 'completed':
            if position[2] < limit['minimum_altitude_m']: reasons.append(f"z < {limit['minimum_altitude_m']:g} m")
            if norm(position) > limit['position_norm_m']: reasons.append('位置范围超限')
            if norm(velocity) > limit['speed_m_s']: reasons.append('速度超限')
            if norm(rates) > limit['body_rate_rad_s']: reasons.append('机体角速度超限')
            if not reasons: reasons.append('包络退出，末尾采样无法唯一确定触发项')
        spec = side['task']; duration = side['requested_duration_s']
        planned_t = [i*duration/320 for i in range(321)]
        fault_t = side['fault_start_s']
        first_fault = next((t for t, eta in zip(times, series['effectiveness']) if any(v < 1 for v in eta)), None)
        diagnostics, diagnostics_error = self._diagnostics(side['record_id'], times[-1])
        return {'meta': {'record_id': side['record_id'], 'group': group, 'label': label,
                         'name': label_name(label), 'family': spec['family'], 'task': spec,
                         'requested_end_s': duration, 'actual_end_s': times[-1], 'fault_time_s': fault_t,
                         'first_fault_sample_s': first_fault, 'termination': side['termination'],
                         'termination_reasons': reasons, 'csv': path.name, 'csv_sha256': digest,
                         'parameter_status': config['parameter_status'], 'metrics': side['flight_metrics'],
                         'model_version': config.get('model_version'),
                         'config_hash': side.get('provenance', {}).get('config_hash'),
                         'geometry': {'arm_radius_m': config['airframe']['arm_radius_m'],
                                      'rotor_radius_m': config['airframe']['rotor_radius_m'],
                                      'body_radius_m': config['airframe']['center_radius_m'],
                                      'omega_max_rad_s': config['propulsion']['omega_max_rad_s']},
                         'diagnostics_available': diagnostics is not None, 'diagnostics_error': diagnostics_error},
                'series': series, 'planned': {'time': planned_t, 'position': [planned_reference(spec, t) for t in planned_t]},
                'diagnostics': diagnostics}

    def _diagnostics(self, record_id, end):
        path = self.predictions / (record_id+'.json') if self.predictions else None
        if path is None or not path.is_file():
            return None, None
        try:
            predictions = json.loads(path.read_text(encoding='utf-8'))
            if not isinstance(predictions, list): raise ValueError('预测必须是已保存的列表')
            previous = -math.inf
            for p in predictions:
                t = float(p['time'])
                if not math.isfinite(t) or t <= previous or t > end+1e-9: raise ValueError('预测时间不一致')
                previous = t
                if p['valid']:
                    if not isinstance(p['predicted_label'], int) or not 0 <= p['predicted_label'] <= 18:
                        raise ValueError('预测标签不合法')
                    score = p['fault_probability']
                    if not math.isfinite(score) or not 0 <= score <= 1:
                        raise ValueError('故障得分不合法')
                    probabilities = p['probabilities']
                    if len(probabilities) != 19 or not all(math.isfinite(v) and 0 <= v <= 1 for v in probabilities):
                        raise ValueError('预测概率不合法')
                    if abs(sum(probabilities)-1) > 1e-5: raise ValueError('预测概率未归一化')
                if p['alarm'] and (not isinstance(p['alarm_motor_severity_label'], int)
                                   or not 1 <= p['alarm_motor_severity_label'] <= 18):
                    raise ValueError('报警定位不合法')
            return predictions, None
        except (ValueError, KeyError, TypeError) as exc:
            return None, str(exc)
