function open_hex_recovery()
%OPEN_HEX_RECOVERY Open the saved recovery demonstration, preserving baseline.
root=fileparts(mfilename('fullpath')); addpath(root);
if ~isfile(fullfile(root,'hex_recovery.slx')), build_hex_recovery_model(); end
open_system(fullfile(root,'hex_recovery.slx'));
end
