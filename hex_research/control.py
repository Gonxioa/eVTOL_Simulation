"""Independent SI position/velocity/quaternion/rate cascade.

Structural reference: PX4 v1.12.3 PositionControl, AttitudeControl,
RateControl. This is a research controller, not a PX4 firmware replica.
Only observations, references and NOMINAL allocation reach this controller.
"""
import numpy as np
from control_allocation import BoundedControlAllocator
from .configuration import physical,matrix
from .math3d import vector,quaternion,rotation,shortest_error,desired_attitude

class CascadeController:
    def __init__(self,cfg):
        self.cfg=cfg; self.p=physical(cfg); self.c=cfg['control']; self.A=matrix(self.p)
        self.allocator=BoundedControlAllocator(self.A,self.p['kT']*self.p['omega_min']**2,self.p['kT']*self.p['omega_max']**2,**cfg['allocator'])
        self.reset()

    def reset(self):
        self.last_time=None; self.previous_w=None; self.previous_v=None
        self.filtered_alpha=np.zeros(3); self.filtered_accel=np.zeros(3)
        self.velocity_integral=np.zeros(3); self.rate_integral=np.zeros(3); self.last_qd=np.array([0.,0.,0.,1.])

    def update(self,t,observation,reference):
        if not np.isfinite(t) or self.last_time is not None and t<=self.last_time: raise ValueError('Control timestamps must increase')
        dt=1/self.cfg['timing']['control_hz'] if self.last_time is None else t-self.last_time
        if dt<=0 or dt>.2: raise ValueError('Invalid controller timestep')
        x=vector(observation['x'],3,'observed position'); v=vector(observation['v'],3,'observed velocity')
        q=quaternion(observation['q']); w=vector(observation['w'],3,'observed body rate'); R=rotation(q)
        c=self.c; gain=lambda k:np.asarray(c[k],dtype=float)
        velocity_integral_applied=self.velocity_integral.copy(); rate_integral_applied=self.rate_integral.copy()
        ev=np.clip(vector(reference['x_dot'],3,'velocity reference')+gain('position_p')*(vector(reference['x'],3,'position reference')-x),-gain('velocity_limit_m_s'),gain('velocity_limit_m_s'))-v
        filt=1-np.exp(-2*np.pi*c['derivative_filter_hz']*dt)
        if self.previous_w is not None:
            self.filtered_alpha+=filt*((w-self.previous_w)/dt-self.filtered_alpha)
            self.filtered_accel+=filt*((v-self.previous_v)/dt-self.filtered_accel)
        a_raw=vector(reference['x_ddot'],3,'acceleration reference')+gain('velocity_p')*ev+self.velocity_integral-gain('velocity_d')*self.filtered_accel
        acc=np.clip(a_raw,-gain('acceleration_limit_m_s2'),gain('acceleration_limit_m_s2'))
        F=self.p['mass']*(acc+np.array([0,0,self.p['gravity']]))
        F[2]=max(F[2],0); max_horizontal=F[2]*np.tan(np.deg2rad(c['tilt_limit_deg']))
        if np.linalg.norm(F[:2])>max_horizontal: F[:2]*=max_horizontal/max(np.linalg.norm(F[:2]),1e-12)
        maximum=sum(self.p['kT']*self.p['omega_max']**2)
        if np.linalg.norm(F)>maximum: F*=maximum/np.linalg.norm(F)
        qd,fallback=desired_attitude(F,float(reference['yaw']),self.last_qd)
        # World-axis yaw rate must be transformed, especially under tilt.
        feedforward=R.T@np.array([0,0,float(reference['yaw_dot'])])
        attitude_error=shortest_error(q,qd)
        rates=np.clip(gain('attitude_p')*attitude_error+feedforward,-gain('rate_limit_rad_s'),gain('rate_limit_rad_s'))
        error=rates-w
        alpha=gain('rate_p')*error+self.rate_integral-gain('rate_d')*self.filtered_alpha
        moment=self.p['inertia']@alpha+np.cross(w,self.p['inertia']@w)
        thrust=max(float(F@R[:,2]),0.0)  # Projection onto actual observed body axis.
        result=self.allocator.allocate(np.r_[thrust,moment])
        nominal_deficit=result.residual_wrench
        # Conditional integration only uses accessible nominal solver residuals.
        di=gain('rate_i')*error*dt
        rate_requested_step=di.copy(); rate_blocked=(nominal_deficit[1:]*error)>1e-8
        di[rate_blocked]=0
        rate_proposed=self.rate_integral+di
        self.rate_integral=np.clip(self.rate_integral+di,-gain('rate_integral_limit_rad_s2'),gain('rate_integral_limit_rad_s2'))
        limited_acc=F/self.p['mass']-np.array([0,0,self.p['gravity']])
        direction=F/max(np.linalg.norm(F),1e-12)
        nominal_acc_deficit=direction*nominal_deficit[0]/self.p['mass']
        velocity_feedback_step=dt*gain('velocity_i')*ev
        velocity_antiwindup_step=dt*c['antiwindup_gain_s_inv']*(limited_acc-a_raw-nominal_acc_deficit)
        velocity_proposed=self.velocity_integral+dt*(gain('velocity_i')*ev+c['antiwindup_gain_s_inv']*(limited_acc-a_raw-nominal_acc_deficit))
        self.velocity_integral=np.clip(self.velocity_integral+dt*(gain('velocity_i')*ev+c['antiwindup_gain_s_inv']*(limited_acc-a_raw-nominal_acc_deficit)),
                                       -gain('velocity_integral_limit_m_s2'),gain('velocity_integral_limit_m_s2'))
        self.previous_w=w.copy(); self.previous_v=v.copy(); self.last_time=float(t); self.last_qd=qd.copy()
        return {'cmd_motor_speeds':np.sqrt(result.commanded_thrusts/self.p['kT']),
                'desired_wrench':result.desired_wrench,'nominal_allocated_wrench':result.allocated_wrench,
                'nominal_residual_wrench':result.residual_wrench,'nominal_motor_thrusts':result.commanded_thrusts,
                'allocator_success':result.success,'allocator_feasible':result.wrench_feasible,
                'allocator_bound_count':result.bound_active_count,'q_desired':qd,'rate_reference':rates,
                'velocity_integral':self.velocity_integral.copy(),'rate_integral':self.rate_integral.copy(),
                'velocity_integral_applied':velocity_integral_applied,'rate_integral_applied':rate_integral_applied,
                'position_error':vector(reference['x'],3,'position reference')-x,
                'velocity_error':ev.copy(),'attitude_error_vector':attitude_error.copy(),
                'attitude_error_angle_rad':float(2*np.arcsin(np.clip(np.linalg.norm(attitude_error)/2,0,1))),
                'rate_error':error.copy(),'observed_velocity':v.copy(),'observed_rate':w.copy(),
                'rate_integral_requested_step':rate_requested_step,'rate_integral_allowed_step':di.copy(),
                'rate_integral_actual_step':self.rate_integral-rate_integral_applied,
                'rate_integral_blocked_nominal':rate_blocked,
                'rate_integral_limit_active':np.isclose(abs(self.rate_integral),gain('rate_integral_limit_rad_s2'),atol=1e-10,rtol=0),
                'rate_integral_clipped_step':abs(rate_proposed)>gain('rate_integral_limit_rad_s2')+1e-10,
                'velocity_integral_feedback_step':velocity_feedback_step,
                'velocity_integral_antiwindup_step':velocity_antiwindup_step,
                'velocity_integral_actual_step':self.velocity_integral-velocity_integral_applied,
                'velocity_integral_limit_active':np.isclose(abs(self.velocity_integral),gain('velocity_integral_limit_m_s2'),atol=1e-10,rtol=0),
                'velocity_integral_clipped_step':abs(velocity_proposed)>gain('velocity_integral_limit_m_s2')+1e-10,
                'allocator_lower_active':result.lower_bound_active.copy(),'allocator_upper_active':result.upper_bound_active.copy(),
                'attitude_fallback':fallback}
