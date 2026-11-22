"""Checks specific to the revised-controller measured-data interface."""
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
from simulink_data import observable_features,prepare_plan,checked_plan


class ControlRepairInterface(unittest.TestCase):
    def test_command_truth_and_metadata_do_not_enter_twelve_channels(self):
        df=pd.DataFrame({**{f'accelerometer_sensor_m_s2_{i}':np.arange(6)+i for i in range(1,4)},
                         **{f'gyro_sensor_rad_s_{i}':np.arange(6)+i for i in range(1,4)},
                         **{f'feedback_state_{i}':np.arange(6)+i for i in range(14,20)},
                         **{f'rotor_command_rad_s_{i}':np.zeros(6) for i in range(1,7)}})
        meta=dict(feedback_delay_s=.03,parameters=dict(sensor_dt=.01))
        x=observable_features(df,meta,'imu_rpm')
        self.assertEqual(x.shape,(6,12))
        np.testing.assert_array_equal(x[:,0],df.accelerometer_sensor_m_s2_1.to_numpy()[[0,0,0,0,1,2]])
        np.testing.assert_array_equal(x[:,6],df.feedback_state_14.to_numpy())
        for i in range(1,7):df[f'rotor_command_rad_s_{i}']=1e9;df[f'rotor_actual_rad_s_{i}']=-1e9
        df['fault_time']=0.;df['plant_effectiveness_1']=0.
        np.testing.assert_array_equal(x,observable_features(df,meta,'imu_rpm'))
        with self.assertRaises(ValueError):observable_features(df.drop(columns=['feedback_state_14']),meta,'imu_rpm')

    def test_new_collection_allows_three_independent_condition_roles(self):
        with tempfile.TemporaryDirectory() as d:
            prepare_plan(d,seeds=[1,2,3],controller_version='control_recovery_v4_20261122',
                         dataset_version='test_only',roles=['train','validation','test'])
            plan=checked_plan(d)
            self.assertEqual(len(plan['runs']),57)
            self.assertEqual(plan['condition_roles'],{'0':'train','1':'validation','2':'test'})
            self.assertEqual(len({r['seed'] for r in plan['runs']}),3)
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaises(ValueError):prepare_plan(d,seeds=[1,1,2],controller_version='v4')

if __name__=='__main__':unittest.main()
