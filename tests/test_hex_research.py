import unittest
from copy import deepcopy
import numpy as np
from scipy.spatial.transform import Rotation
from hex_research.configuration import load,physical,matrix
from hex_research.math3d import shortest_error,desired_attitude,rotation,enu_flu_to_ned_frd
from hex_research.dynamics import HexPlant
from hex_research.control import CascadeController
from hex_research.observation import Sensors
from hex_research.simulation import FaultSchedule,simulate,reference
from control_allocation import BoundedControlAllocator

class ResearchChecks(unittest.TestCase):
    def setUp(self): self.cfg=load(); self.plant=HexPlant(self.cfg); self.p=self.plant.p
    def test_physical_geometry_and_units(self):
        eigen=np.linalg.eigvalsh(self.p['inertia']); self.assertLessEqual(eigen[-1],sum(eigen[:2]))
        self.assertEqual(matrix(self.p).shape,(4,6)); s=self.plant.initial()
        np.testing.assert_allclose(self.plant.derivative(0,s,s[13:],np.ones(6)),np.zeros(19),atol=1e-12)
        self.assertEqual(self.plant.pack(self.plant.unpack(s)).shape,(19,))
        bad=deepcopy(self.cfg); bad['propulsion']['kT']=np.nan
        with self.assertRaises(ValueError): physical(bad)
    def test_single_motor_analytic_components_and_stopped_aero(self):
        s=self.plant.initial()
        for i in range(6):
            s[13:]=0; s[13+i]=100; f,m,a=self.plant.wrench(s)
            T=self.p['kT'][i]*100**2; x,y,_=self.p['rotor_positions'][i]
            np.testing.assert_allclose(m,[y*T,-x*T,self.p['yaw_sign'][i]*self.p['kQ'][i]*100**2],atol=1e-13)
            np.testing.assert_allclose(f,[0,0,T])
        s[13:]=0; s[3]=2; f,m,_=self.plant.wrench(s); self.assertEqual(f[2],0.); self.assertLess(f[0],0.)
    def test_freefall_and_sensor_lever_installation(self):
        s=self.plant.initial(); s[13:]=0; d=self.plant.derivative(0,s,np.zeros(6),np.ones(6))
        np.testing.assert_allclose(d[3:6],[0,0,-self.p['gravity']])
        np.testing.assert_allclose(Sensors(self.cfg,1).sample(0,s,d,False)['accel_sensor'],0,atol=1e-12)
        cfg=deepcopy(self.cfg); cfg['sensor']['lever_m']=[.1,0,0]; cfg['sensor']['installation_rotation_xyzw']=Rotation.from_euler('z',np.pi/2).as_quat().tolist()
        s[10:13]=[0,0,2]; d=np.zeros(19)
        np.testing.assert_allclose(Sensors(cfg,1).sample(0,s,d,False)['accel_sensor'],[0,.4,self.p['gravity']],atol=1e-12)
    def test_antipodal_180_zero_force_and_collinear(self):
        q=np.array([0,0,0,1.]); target=np.array([1.,0,0,0])
        np.testing.assert_allclose(shortest_error(q,target),shortest_error(q,-target)); self.assertGreater(np.linalg.norm(shortest_error(q,target)),1)
        fallback,flag=desired_attitude([0,0,0],0,q); self.assertTrue(flag); np.testing.assert_allclose(fallback,q)
        target,flag=desired_attitude([1,0,0],0,q); self.assertTrue(flag); self.assertAlmostEqual(np.linalg.det(rotation(target)),1)
        for bad in ([0,0,0,0],[np.nan,0,0,1]):
            with self.assertRaises(ValueError): rotation(bad)
    def test_controller_clock_reset_limits_and_fault_truth_isolation(self):
        obs=self.plant.unpack(self.plant.initial()); ref=reference(0); a=CascadeController(self.cfg); b=CascadeController(self.cfg)
        left=a.update(0,{**obs,'fault_label':0},ref); right=b.update(0,{**obs,'fault_label':18,'effectiveness':np.zeros(6)},ref)
        np.testing.assert_allclose(left['cmd_motor_speeds'],right['cmd_motor_speeds'],atol=0)
        for i in range(1,30): out=a.update(i*.01,obs,{**ref,'x':np.array([100,100,100.])})
        self.assertTrue(np.all(out['cmd_motor_speeds']>=0)); self.assertLessEqual(max(out['cmd_motor_speeds']),self.p['omega_max']+1e-8)
        self.assertTrue(np.all(abs(a.velocity_integral)<=self.cfg['control']['velocity_integral_limit_m_s2']))
        with self.assertRaises(ValueError): a.update(.29,obs,ref)
        a.reset(); np.testing.assert_allclose(a.rate_integral,0)
    def test_yaw_feedforward_under_tilt(self):
        obs=self.plant.unpack(self.plant.initial()); obs['q']=Rotation.from_euler('x',.3).as_quat(); ref=reference(0); ref['yaw_dot']=.2
        o1=CascadeController(self.cfg).update(0,obs,ref); ref['yaw_dot']=0; o2=CascadeController(self.cfg).update(0,obs,ref)
        np.testing.assert_allclose(o1['rate_reference']-o2['rate_reference'],rotation(obs['q']).T@[0,0,.2],atol=1e-12)
    def test_exact_fault_boundary_lag_recovery_no_double_loss(self):
        schedule=FaultSchedule(0,.2,.015,.045); np.testing.assert_allclose(schedule.at(.014),1)
        self.assertEqual(schedule.at(.015)[0],.2); self.assertEqual(schedule.at(.045)[0],1.)
        s=self.plant.initial(); command=s[13:].copy(); derivative=self.plant.derivative(.02,s,command,schedule.at(.02))
        self.assertAlmostEqual(derivative[13],(np.sqrt(.2)-1)*command[0]/.05)
        self.assertAlmostEqual(self.plant.wrench(s)[2]['dynamic_rotor_thrusts'][0],self.p['kT'][0]*command[0]**2)
        run=simulate(self.cfg,.03,schedule=FaultSchedule(0,0,.015))
        self.assertAlmostEqual(run['rows'][1]['state'][13],command[0],places=10); self.assertLess(run['rows'][2]['state'][13],command[0])
    def test_observation_no_state_write_and_seed_reproducibility(self):
        s=self.plant.initial(); d=self.plant.derivative(0,s,s[13:],np.ones(6)); original=s.copy()
        a=Sensors(self.cfg,99).sample(0,s,d); b=Sensors(self.cfg,99).sample(0,s,d)
        np.testing.assert_array_equal(s,original); np.testing.assert_array_equal(a['w'],b['w'])
        left=simulate(self.cfg,.2,noise=True,wind=True,feedback='observed_feedback',seed=7); right=simulate(self.cfg,.2,noise=True,wind=True,feedback='observed_feedback',seed=7)
        np.testing.assert_array_equal(left['rows'][-1]['state'],right['rows'][-1]['state'])
        ideal_a=simulate(self.cfg,.2,noise=False,seed=7); ideal_b=simulate(self.cfg,.2,noise=True,seed=7)
        np.testing.assert_array_equal(ideal_a['rows'][-1]['state'],ideal_b['rows'][-1]['state'])
    def test_regularized_allocation_does_not_bypass_objective(self):
        desired=np.r_[self.p['mass']*self.p['gravity'],0,0,0]
        regular=BoundedControlAllocator(matrix(self.p),0,self.p['kT']*self.p['omega_max']**2,regularization=10).allocate(desired)
        self.assertFalse(regular.used_unconstrained_solution); self.assertLess(regular.allocated_wrench[0],desired[0])
    def test_coordinate_axial_vector_identity(self):
        x,v,q,w=enu_flu_to_ned_frd(np.array([1.,2,3]),np.zeros(3),[0,0,0,1],np.array([1.,2,3]))
        np.testing.assert_allclose(x,[2,1,-3]); np.testing.assert_allclose(w,[1,-2,-3]); self.assertAlmostEqual(np.linalg.det(rotation(q)),1.)
    def test_real_normal_export_with_optional_fault_time(self):
        from tempfile import TemporaryDirectory
        from pathlib import Path
        from hex_research.simulation import export
        from scenario_config import SCENARIOS
        run=simulate(self.cfg,.1)
        with TemporaryDirectory() as folder:
            path=Path(folder)/'normal.csv'; df,side=export(run,path,SCENARIOS[0],1)
            self.assertTrue(path.is_file()); self.assertTrue(path.with_suffix('.json').is_file())
            self.assertEqual(len(df),11); self.assertEqual(side['termination'],'completed')
            self.assertTrue(df['fault_time'].isna().all())
    def test_known_force_torque_and_motor_exponential(self):
        s=self.plant.initial(); s[13:]=0
        d=self.plant.derivative(0,s,np.zeros(6),np.ones(6),external_force=self.p['mass']*np.array([1.,2,self.p['gravity']]),external_torque=self.p['inertia']@np.array([.2,0,0]))
        np.testing.assert_allclose(d[3:6],[1,2,0],atol=1e-12); np.testing.assert_allclose(d[10:13],[.2,0,0],atol=1e-12)
        initial=self.plant.initial(); command=initial[13:].copy()
        result=self.plant.integrate(initial,0,.2,command,FaultSchedule(0,0,0),np.zeros(3),np.zeros(3),np.zeros(3))
        self.assertAlmostEqual(result[13],command[0]*np.exp(-.2/.05),delta=2e-6)
    def test_export_clock_does_not_change_feedback_or_random_physics(self):
        a=simulate(self.cfg,.2,noise=True,wind=True,feedback='observed_feedback',seed=9)
        cfg=deepcopy(self.cfg); cfg['timing']['export_hz']=50
        b=simulate(cfg,.2,noise=True,wind=True,feedback='observed_feedback',seed=9)
        np.testing.assert_array_equal(a['rows'][-1]['state'],b['rows'][-1]['state'])
    def test_rotation_columns_remain_continuous_across_euler_gimbal(self):
        from scipy.spatial.transform import Rotation
        q=Rotation.from_euler('xyz',np.deg2rad([[0,89.9,0],[0,90.1,0]]))
        columns=q.as_matrix()[:,:,:2].reshape(2,-1)
        self.assertLess(np.linalg.norm(columns[1]-columns[0]),.01)
    def test_fault_float_boundary_is_not_rounded_to_inactive_tick(self):
        start=.0150000000001; run=simulate(self.cfg,.03,schedule=FaultSchedule(0,0,start))
        omega0=self.plant.initial()[13]
        self.assertAlmostEqual(run['rows'][2]['state'][13],omega0*np.exp(-(.02-start)/.05),delta=2e-6)
        # A boundary immediately after a grid tick creates a tiny positive
        # integration interval; it must still run, not divide by zero.
        run=simulate(self.cfg,.03,schedule=FaultSchedule(0,0,.01000000000001))
        self.assertEqual(run['termination'],'completed')
    def test_uavfd_adapter_reorders_scalar_without_transferring_parameters(self):
        from hex_research.math3d import to_uavfd_state
        s=self.plant.initial(position=(1,2,3)); state,speeds=to_uavfd_state(s)
        self.assertEqual(state.shape,(13,)); self.assertEqual(speeds.shape,(6,))
        np.testing.assert_allclose(state[:3],[2,1,-3]); self.assertAlmostEqual(np.linalg.norm(state[6:10]),1.)
    def test_healthy_idle_bound_cannot_prevent_fault_stop(self):
        cfg=deepcopy(self.cfg); cfg['propulsion']['omega_min_rad_s']=100.
        controller=CascadeController(cfg); obs=self.plant.unpack(self.plant.initial()); ref=reference(0)
        ref['x_ddot']=np.array([0.,0,-100.])
        out=controller.update(0,obs,ref)
        self.assertTrue(np.all(out['cmd_motor_speeds']>=100.-1e-8))
        s=self.plant.initial(); d=self.plant.derivative(0,s,out['cmd_motor_speeds'],FaultSchedule(0,0,0).at(0))
        self.assertAlmostEqual(d[13],-s[13]/.05)

if __name__=='__main__': unittest.main()
