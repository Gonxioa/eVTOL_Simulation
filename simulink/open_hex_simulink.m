function open_hex_simulink()
%OPEN_HEX_SIMULINK Open the editable model with its saved default parameters.
root=fileparts(mfilename('fullpath')); path=fullfile(root,'hex_evtol.slx');
if ~isfile(path), build_hex_model(); end
load_system(path); hex_ensure_pv_model();
open_system('hex_evtol'); set_param('hex_evtol','ZoomFactor','FitSystem');
end
