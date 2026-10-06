"""Version-4 dataset integrity entry; previous validator is in legacy_reference."""
import os
from pathlib import Path
from hex_research.configuration import ROOT
from hex_research.data import validate_dataset
if __name__=='__main__':
    validate_dataset(Path(os.getenv('HEX_OUTPUT_DIR',ROOT/'output_hex_audit/dataset_v4')))
