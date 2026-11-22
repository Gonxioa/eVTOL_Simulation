function open_hex_simulink()
%OPEN_HEX_SIMULINK Open the saved control demonstration, preserving baseline.
root=fileparts(mfilename('fullpath')); addpath(root);
hex_ensure_model();
open_system(fullfile(root,'hex_evtol.slx'));
end
