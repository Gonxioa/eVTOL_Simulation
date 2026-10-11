"""Scientific plots from actual saved paired controller traces."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'output_hex_rate_tuning_20261010'
FIG=Path(__file__).parent/'figures';FIG.mkdir(exist_ok=True)
plt.rcParams.update({'font.size':10,'figure.dpi':140,'axes.grid':True,'grid.alpha':.25})

fig,axes=plt.subplots(2,2,figsize=(10,7),sharex=True)
groups=json.loads((OUT/'predeclared_plan.json').read_text(encoding='utf-8'))['settings']
from sys import path
path.insert(0,str(ROOT))
from hex_research.commands import task_library
faults={g['group_id']:g['fault_time_s'] for g in task_library(groups)}
for ax,g,family in zip(axes.flat,(1,3,5,7),('Hover','Line','Circle','Vertical')):
    name=f'g{g:03d}_m0_delay_noise_wind_8s.csv'
    for folder,label,color in [('screen/baseline','Before','#738395'),('validation/velocity_z09','Validated v2','#007f9d')]:
        frame=pd.read_csv(OUT/folder/name)
        err=np.linalg.norm(frame[[f'position_error_{a}' for a in 'xyz']].to_numpy(),axis=1)
        # The control position error uses delayed feedback. Plot the independent
        # true tracking error so the curve matches the acceptance metric.
        err=np.linalg.norm(frame[[f'position_{a}' for a in 'xyz']].to_numpy()-frame[[f'reference_{a}' for a in 'xyz']].to_numpy(),axis=1)
        ax.plot(frame.time,err,label=label,color=color,lw=1.8)
    ax.axvline(faults[g],color='#bf4b52',ls='--',label='Fault injection')
    ax.set_title(f'{family}: g{g:03d}, M0, effectiveness 0.8');ax.set_ylabel('True position error [m]')
    ax.set_xlabel('Simulation time [s]')
axes[0,0].legend();fig.suptitle('Paired comparison: noise, wind and 30 ms feedback delay')
fig.tight_layout();fig.savefig(FIG/'paired_position_recovery.png');plt.close(fig)

fig,axes=plt.subplots(3,1,figsize=(10,8),sharex=True)
name='g001_m0_delay_noise_wind_8s.csv'
for folder,label,color in [('screen/baseline','Before','#738395'),('validation/velocity_z09','Validated v2','#007f9d')]:
    f=pd.read_csv(OUT/folder/name)
    axes[0].plot(f.time,f.rate_integral_applied_y,label=label,color=color)
    axes[1].plot(f.time,f.rate_error_y,label=label,color=color)
    axes[2].plot(f.time,f.velocity_integral_applied_z,label=label,color=color)
axes[0].axhline(-3,color='#738395',ls=':',label='Old pitch integral limit')
axes[0].axhline(-20,color='#007f9d',ls=':',label='New pitch integral limit')
axes[0].set_ylabel('Applied pitch integral [rad/s²]');axes[0].legend(ncol=2)
axes[1].set_ylabel('Pitch rate error [rad/s]');axes[2].set_ylabel('Applied vertical velocity integral [m/s²]')
for ax in axes:ax.axvline(faults[1],color='#bf4b52',ls='--')
axes[-1].set_xlabel('Simulation time [s]');fig.tight_layout()
fig.savefig(FIG/'integral_and_error.png');plt.close(fig)

fig,axes=plt.subplots(2,1,figsize=(10,6),sharex=True)
for folder,label,color in [('antiwindup_complete','Ki_z 0.6','#738395'),('antiwindup_velocity09','Ki_z 0.9','#007f9d')]:
    f=pd.read_csv(OUT/folder/'collective_saturation_recovery.csv')
    axes[0].plot(f.time,f.velocity_integral_z,label=label,color=color)
    axes[1].plot(f.time,f.position_z-2.5,label=label,color=color)
for ax in axes:ax.axvspan(2,4,color='#bf4b52',alpha=.12);ax.axvline(4,color='#bf4b52',ls='--');ax.legend()
axes[0].set_ylabel('Updated vertical velocity integral [m/s²]')
axes[1].set_ylabel('Position offset from 2.5 m [m]');axes[-1].set_xlabel('Simulation time [s]')
fig.suptitle('Saturation stress: reference 50 m at 2–4 s, then return to 2.5 m')
fig.tight_layout();fig.savefig(FIG/'saturation_recovery.png');plt.close(fig)
print('Saved 3 figures from actual traces')
