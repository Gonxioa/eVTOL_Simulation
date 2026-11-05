"""Confirmation planning uses groups, and new test reads remain gated."""
import tempfile
import unittest
from pathlib import Path
import numpy as np
from provenance import write_json,stable_hash,sha256_file
from simulink_data import read
from simulink_v4_data import CONFIG,authorize
from simulink_v4_confirmation import planned_groups,optimizer_steps


class V4ConfirmationTests(unittest.TestCase):
    def test_planning_zero_variance_still_uses_registered_minimum(self):
        sd,raw,n=planned_groups([0,0,0],.05,6,12)
        self.assertEqual((sd,raw,n),(0.,0,6))

    def test_planning_caps_are_budget_not_more_independent_seeds(self):
        sd,raw,n=planned_groups([-.5,0,.5],.05,6,12)
        self.assertEqual(n,12); self.assertGreater(raw,n)
        with self.assertRaises(ValueError): planned_groups([0],.05,6,12)

    def test_confirmation_token_pins_files_config_and_source_identity(self):
        with tempfile.TemporaryDirectory() as directory:
            cfg=dict(read(CONFIG),results_root=directory); reg={'signature':'fresh'}
            model=Path(directory)/'weights.bin'; model.write_bytes(b'original')
            p=dict(config_signature=stable_hash(cfg),test_registry_signature='fresh',locked_files={str(model):sha256_file(model)})
            p['signature']=stable_hash(p); write_json(Path(directory)/'freeze.json',p)
            authorize(cfg,reg,'test',p['signature'])
            with self.assertRaises(PermissionError): authorize(cfg,{'signature':'historical'},'test',p['signature'])
            model.write_bytes(b'changed')
            with self.assertRaises(PermissionError): authorize(cfg,reg,'test',p['signature'])


if __name__=='__main__': unittest.main()
