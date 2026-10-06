"""Audited six-rotor Newton--Euler model, external wind, 19 states.

Aerodynamic expressions are adapted mathematically from RotorPy 2.1.2
compute_body_wrench (MIT); no RotorPy core is imported or patched.
The normal-running, speed-independent translational-lift term is excluded
because this research explicitly includes stopped rotors.
"""
import numpy as np
from .configuration import physical
from .math3d import rotation,quaternion,multiply,vector

class HexPlant:
    def __init__(self,cfg):
        self.cfg=cfg; self.p=physical(cfg); self.n=6
        self.invJ=np.linalg.inv(self.p['inertia'])  # Only the physical 3x3 tensor.
        self.cd=np.array(cfg['aerodynamics']['body_drag_kg_m'])
        self.kr=np.array(cfg['aerodynamics']['rotor_drag_kg_per_rad'])

    def initial(self,position=(0,0,2.5),q=(0,0,0,1),w=(0,0,0)):
        hover=np.sqrt(self.p['mass']*self.p['gravity']/sum(self.p['kT']))
        return np.r_[vector(position,3,'position'),np.zeros(3),quaternion(q),vector(w,3,'rate'),np.full(6,hover)]

    def unpack(self,s):
        s=vector(s,19,'plant state')
        return {'x':s[:3].copy(),'v':s[3:6].copy(),'q':s[6:10].copy(),'w':s[10:13].copy(),'rotor_speeds':s[13:].copy()}

    def pack(self,state):
        return vector(np.r_[state['x'],state['v'],quaternion(state['q']),state['w'],state['rotor_speeds']],19,'packed state')

    def wrench(self,s,wind=(0,0,0),external_force=(0,0,0)):
        s=vector(s,19,'plant state'); omega=s[13:]
        if np.any(omega<0): raise ValueError('Negative mechanical rotor speed')
        r=self.p['rotor_positions']; vrel=rotation(s[6:10]).T@(s[3:6]-vector(wind,3,'wind'))
        local=vrel+np.cross(s[10:13],r)
        thrust=self.p['kT']*omega**2; yaw=self.p['yaw_sign']*self.p['kQ']*omega**2
        lift=np.c_[np.zeros((6,2)),thrust]
        rotor_drag=-omega[:,None]*self.kr*local
        body_drag=-np.linalg.norm(vrel)*self.cd*vrel
        aero=rotor_drag.sum(axis=0)+body_drag
        aero_m=np.cross(r,rotor_drag).sum(axis=0)
        thrust_m=np.cross(r,lift).sum(axis=0)+np.array([0,0,yaw.sum()])
        force=lift.sum(axis=0)+aero+vector(external_force,3,'external body force')
        return force,thrust_m+aero_m,{'dynamic_rotor_thrusts':thrust,'dynamic_propulsive_wrench':np.r_[thrust.sum(),thrust_m],
                                    'aero_force_body':aero,'aero_moment_body':aero_m}

    def derivative(self,t,s,command,effectiveness,wind=(0,0,0),external_force=(0,0,0),external_torque=(0,0,0)):
        command=vector(command,6,'rotor command'); eta=vector(effectiveness,6,'effectiveness')
        if np.any((eta<0)|(eta>1)) or np.any((command<0)|(command>self.p['omega_max']+1e-8)): raise ValueError('Actuator bounds')
        f,m,_=self.wrench(s,wind,external_force); w=s[10:13]
        acc=np.array([0,0,-self.p['gravity']])+rotation(s[6:10])@f/self.p['mass']
        alpha=self.invJ@(m+vector(external_torque,3,'external torque')-np.cross(w,self.p['inertia']@w))
        target=np.sqrt(eta)*command  # Loss once, in TARGET; actual thrust uses actual state only.
        return np.r_[s[3:6],acc,.5*multiply(s[6:10],np.r_[w,0]),alpha,(target-s[13:])/self.p['tau']]

    def integrate(self,s,t,end,command,schedule,wind,force,torque):
        maximum=1/self.cfg['timing']['integration_hz']
        if end<=t: raise ValueError('Positive integration interval')
        count=max(1,int(np.ceil((end-t)/maximum-1e-12))); dt=(end-t)/count
        if dt>.2*min(self.p['tau']): raise ValueError('Integration step too large for actuator dynamics')
        # Schedule boundaries are split by the caller. No control/noise calls here.
        eta=schedule.at(t)
        for k in range(count):
            time=t+k*dt
            f=lambda ti,x:self.derivative(ti,x,command,eta,wind,force,torque)
            k1=f(time,s); k2=f(time+dt/2,s+dt*k1/2); k3=f(time+dt/2,s+dt*k2/2); k4=f(time+dt,s+dt*k3)
            s=s+dt*(k1+2*k2+2*k3+k4)/6; s[6:10]=quaternion(s[6:10])
        return s
