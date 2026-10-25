function configure_hex_simulink(P)
%CONFIGURE_HEX_SIMULINK Apply a master parameter set to each isolated block.
hex_validate_parameters(P);
root=fileparts(mfilename('fullpath')); mdl='hex_evtol';
load_system(fullfile(root,[mdl '.slx']));
hex_ensure_pv_model();
ws=get_param(mdl,'ModelWorkspace'); ws.assignin('P',P);
groups=hex_block_parameters(P); names=fieldnames(groups);
for i=1:numel(names), ws.assignin(['P_' names{i}],groups.(names{i})); end
end
