import importlib.util
import json
import unittest
from copy import deepcopy
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch
import numpy as np
from hex_research.configuration import load
from hex_research.control import CascadeController
from hex_research.dynamics import HexPlant
from hex_research.simulation import simulate, FaultSchedule, reference, export

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT/'docs/hex_rate_tuning/baseline_source'


class IntegralAuditChecks(unittest.TestCase):
    def setUp(self):
        self.cfg = json.loads((BASE/'hex_research/configs/research.json').read_text(encoding='utf-8'))

    def test_new_logs_do_not_change_commands_or_physical_trajectory(self):
        spec = importlib.util.spec_from_file_location('hex_research._audit_baseline_control', BASE/'hex_research/control.py')
        module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
        options = dict(schedule=FaultSchedule(0,.8,.505), feedback='observed_feedback', noise=True, wind=True, seed=654)
        with patch('hex_research.simulation.CascadeController', module.CascadeController):
            old = simulate(self.cfg, 1., **options)
        new = simulate(self.cfg, 1., **options)
        np.testing.assert_array_equal([r['state'] for r in old['rows']], [r['state'] for r in new['rows']])
        for a,b in zip(old['rows'],new['rows']):
            for key in ('cmd_motor_speeds','rate_integral','velocity_integral','rate_reference'):
                np.testing.assert_array_equal(a['control'][key],b['control'][key])

    def test_nominal_saturation_blocks_outward_but_allows_unwinding(self):
        cfg=deepcopy(self.cfg); cfg['control']['rate_d']=[0.,0.,0.]
        plant=HexPlant(cfg); obs=plant.unpack(plant.initial()); controller=CascadeController(cfg)
        controller.rate_integral[0]=.5
        obs['w']=np.array([50.,0.,0.]); out=controller.update(0.,obs,reference(0))
        self.assertTrue(out['rate_integral_blocked_nominal'][0])
        self.assertEqual(out['rate_integral'][0],.5)
        self.assertEqual(out['rate_integral_actual_step'][0],0.)
        obs['w']=np.array([.1,0.,0.]); out=controller.update(.01,obs,reference(.01))
        self.assertFalse(out['rate_integral_blocked_nominal'][0])
        self.assertLess(out['rate_integral'][0],.5)

    def test_applied_and_updated_integrals_and_clip_flags_are_distinct(self):
        cfg=deepcopy(self.cfg); cfg['control']['rate_i']=[1000.,1000.,1000.]
        cfg['control']['rate_integral_limit_rad_s2']=[.02,.02,.02]
        obs=HexPlant(cfg).unpack(HexPlant(cfg).initial()); obs['w']=np.array([-.1,0.,0.])
        out=CascadeController(cfg).update(0.,obs,reference(0))
        self.assertEqual(out['rate_integral_applied'][0],0.)
        self.assertEqual(out['rate_integral'][0],.02)
        self.assertTrue(out['rate_integral_clipped_step'][0]); self.assertTrue(out['rate_integral_limit_active'][0])
        self.assertEqual(out['rate_integral_actual_step'][0],.02)

    def test_export_records_feedback_error_and_integral_state(self):
        from scenario_config import SCENARIOS
        cfg=deepcopy(self.cfg); cfg['sensor']['feedback_delay_s']=.03
        run=simulate(cfg,.1,feedback='observed_feedback',noise=True,wind=True)
        with TemporaryDirectory() as folder:
            frame, side=export(run,Path(folder)/'normal.csv',SCENARIOS[0],1)
            self.assertEqual(side['control_log_schema'],'cascade_audit_v1')
            self.assertTrue(np.isfinite(frame[[c for c in frame if c.startswith('control_')]].to_numpy()).all())
            for i,row in enumerate(run['rows']):
                c=row['control']
                for j,a in enumerate('xyz'):
                    self.assertEqual(frame.iloc[i][f'control_rate_integral_{a}'],c['rate_integral'][j])
                    self.assertAlmostEqual(frame.iloc[i][f'control_rate_error_{a}'],c['rate_reference'][j]-c['observed_rate'][j])
            self.assertGreater(run['rows'][-1]['feedback_age_s'],0.)

    def test_validation_refuses_to_overwrite_reports_for_another_dataset(self):
        from hex_research.continuous_data import validate_tasks
        settings=json.loads((BASE/'hex_research/configs/continuous.json').read_text(encoding='utf-8'))
        with TemporaryDirectory() as folder:
            out=Path(folder);(out/'dataset_v5').mkdir()
            (out/'dataset_v5/generation_plan.json').write_text('{"old_source":"preserve"}',encoding='utf-8')
            target=out/'task_normal_validation.json';target.write_text('protected old report',encoding='utf-8')
            with self.assertRaisesRegex(RuntimeError,'fresh --out'):
                validate_tasks(self.cfg,settings,out)
            self.assertEqual(target.read_text(encoding='utf-8'),'protected old report')
            self.assertFalse((out/'task_predeclared_acceptance.json').exists())


if __name__=='__main__': unittest.main()
