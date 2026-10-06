"""Compatibility view; control config: hex_research/configs/research.json."""
from copy import deepcopy
from hex_physical_characteristics import HEX_PHYSICAL_CHARACTERISTICS,CONFIG,ARM_LENGTH
hex_params=deepcopy(HEX_PHYSICAL_CHARACTERISTICS)
hex_params.update({'motor_noise_std':0.0,**{'allocator_'+k:v for k,v in CONFIG['allocator'].items()}})
