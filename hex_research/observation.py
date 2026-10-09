"""Single injection of sensor errors; no modification of physical truth.

Position/velocity/attitude are direct noisy observations, not an EKF or
inertial-navigation claim. White intensity sigma/sqrt(dt), bias diffusion
sigma*sqrt(dt); independent streams are sampled only on sensor clocks.
"""
import numpy as np
from scipy.spatial.transform import Rotation
from .math3d import rotation,multiply,quaternion

class Sensors:
    def __init__(self,cfg,seed):
        self.c=cfg['sensor']; self.g=cfg['airframe']['gravity_m_s2']; self.dt=1/self.c['rate_hz']
        seeds=(seed if isinstance(seed,np.random.SeedSequence) else np.random.SeedSequence(seed)).spawn(7); self.rng=[np.random.default_rng(s) for s in seeds]
        self.ba=np.zeros(3); self.bg=np.zeros(3); self.install=rotation(self.c['installation_rotation_xyzw'])
        self.r=np.asarray(self.c['lever_m']); self.last_t=None

    def sample(self,t,s,derivative,noise=True):
        if self.last_t is not None and t<=self.last_t: raise ValueError('Sensor clock must increase')
        dt=self.dt if self.last_t is None else t-self.last_t; self.last_t=t
        R=rotation(s[6:10]); w=s[10:13]; alpha=derivative[10:13]
        body=R.T@(derivative[3:6]+np.array([0,0,self.g]))+np.cross(alpha,self.r)+np.cross(w,np.cross(w,self.r))
        accel=self.install.T@body; gyro=self.install.T@w
        x=s[:3].copy(); v=s[3:6].copy(); q=s[6:10].copy(); rotor=s[13:].copy()
        if noise:
            self.ba+=self.c['accel_bias_diffusion']*np.sqrt(dt)*self.rng[0].normal(size=3)
            self.bg+=self.c['gyro_bias_diffusion']*np.sqrt(dt)*self.rng[1].normal(size=3)
            accel=accel+self.ba+self.c['accel_noise_density']/np.sqrt(dt)*self.rng[2].normal(size=3)+self.c['additional_accel_sample_std']*self.rng[6].normal(size=3)
            gyro=gyro+self.bg+self.c['gyro_noise_density']/np.sqrt(dt)*self.rng[3].normal(size=3)
            x+=self.c['position_sample_std_m']*self.rng[4].normal(size=3)
            v+=self.c['velocity_sample_std_m_s']*self.rng[4].normal(size=3)
            q=quaternion(multiply(q,Rotation.from_rotvec(self.c['attitude_sample_std_rad']*self.rng[4].normal(size=3)).as_quat()))
            rotor+=self.c['rpm_sample_std_rad_s']*self.rng[5].normal(size=6)
        return {'x':x,'v':v,'q':q,'w':self.install@gyro,'accel_sensor':accel,'gyro_sensor':gyro,
                'specific_force_body':self.install@accel,'gyro_body':self.install@gyro,
                'rotor_measurement':rotor,'time':float(t)}

class OUWind:
    def __init__(self,cfg,seed,enabled):
        self.c=cfg['wind']; self.rng=np.random.default_rng(seed); self.enabled=enabled
        self.fluctuation=self.c['ou_stationary_std_m_s']*self.rng.normal(size=3) if enabled else np.zeros(3)
        self.last_t=None
    def update(self,t):
        if not self.enabled: return np.zeros(3)
        if self.last_t is not None:
            dt=t-self.last_t; a=np.exp(-dt/self.c['ou_correlation_time_s'])
            self.fluctuation=a*self.fluctuation+self.c['ou_stationary_std_m_s']*np.sqrt(1-a*a)*self.rng.normal(size=3)
        self.last_t=t
        return np.array(self.c['mean_m_s'])+self.fluctuation
