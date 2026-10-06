"""Compatibility view of the research design, never UAV-FD aircraft truth.
Legacy parameter candidates are preserved under legacy_reference/.
"""
import numpy as np
from hex_research.configuration import load,rotorpy_fields
CONFIG=load()
ARM_LENGTH=CONFIG['airframe']['arm_radius_m']
HEX_PHYSICAL_CHARACTERISTICS=rotorpy_fields(CONFIG)
def physical_review_summary(gravity=9.81):
    p=HEX_PHYSICAL_CHARACTERISTICS
    hover=np.sqrt(p['mass']*gravity/(6*p['k_eta']))
    limit=p['k_eta']*p['rotor_speed_max']**2
    return {'parameter_status':p['parameter_status'],'hover_speed_rad_s':float(hover),
            'hover_speed_rpm':float(hover*60/(2*np.pi)),'max_thrust_per_rotor_n':float(limit),
            'max_total_thrust_n':float(6*limit),'max_thrust_to_weight':float(6*limit/(p['mass']*gravity))}
