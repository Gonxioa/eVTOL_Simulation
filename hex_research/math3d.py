"""Hamilton xyzw active FLU-body-to-ENU-world orientation."""
import numpy as np
from scipy.spatial.transform import Rotation

def vector(value,n,name):
    x=np.asarray(value,dtype=float)
    if x.shape!=(n,) or not np.isfinite(x).all(): raise ValueError(f'{name}: finite shape ({n},) required')
    return x

def quaternion(q):
    q=vector(q,4,'quaternion'); norm=np.linalg.norm(q)
    if norm<1e-12: raise ValueError('Zero quaternion')
    return q/norm

def multiply(a,b):
    return np.r_[a[3]*b[:3]+b[3]*a[:3]+np.cross(a[:3],b[:3]),a[3]*b[3]-a[:3]@b[:3]]

def conjugate(q): return np.r_[-q[:3],q[3]]
def rotation(q): return Rotation.from_quat(quaternion(q)).as_matrix()

def shortest_error(q,qd):
    e=multiply(conjugate(quaternion(q)),quaternion(qd))
    # Deterministic antipodal tie at precisely 180 degrees; never sign(0)=0.
    sign=1 if e[3]>1e-12 else -1 if e[3]<-1e-12 else (1 if e[np.flatnonzero(np.abs(e[:3])>1e-12)[0]]>0 else -1)
    return 2*sign*e[:3]

def desired_attitude(force,yaw,previous):
    force=vector(force,3,'desired force')
    if not np.isfinite(yaw): raise ValueError('Finite yaw required')
    if np.linalg.norm(force)<1e-10: return quaternion(previous),True
    z=force/np.linalg.norm(force); heading=np.array([np.cos(yaw),np.sin(yaw),0.])
    y=np.cross(z,heading)
    fallback=np.linalg.norm(y)<1e-10
    if fallback:
        basis=np.eye(3)[np.argmin(abs(z))]; y=np.cross(z,basis)
    y/=np.linalg.norm(y); x=np.cross(y,z)
    return Rotation.from_matrix(np.column_stack([x,y,z])).as_quat(),fallback

def enu_flu_to_ned_frd(x,v,q,w):
    S=np.array([[0,1,0],[1,0,0],[0,0,-1]]); B=np.diag([1,-1,-1])
    return S@x,S@v,Rotation.from_matrix(S@rotation(q)@B).as_quat(),B@w

def to_uavfd_state(state):
    """Coordinate/layout adapter only; it does NOT assert aircraft equivalence.

    Research state19 xyzw -> UAV-FD interface state13 wxyz plus unsigned Omega6.
    No physical parameter or validation conclusion is transferred.
    """
    state=vector(state,19,'research state')
    x,v,q,w=enu_flu_to_ned_frd(state[:3],state[3:6],state[6:10],state[10:13])
    return np.r_[x,v,q[[3,0,1,2]],w],state[13:].copy()
