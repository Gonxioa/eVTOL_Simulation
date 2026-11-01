function open_hex_simulink()
%OPEN_HEX_SIMULINK Open the saved control demonstration, preserving baseline.
root=fileparts(mfilename('fullpath')); addpath(root);
if ~isfile(fullfile(root,'hex_evtol.slx')), build_hex_model(); end
open_system(fullfile(root,'hex_evtol.slx'));
end
