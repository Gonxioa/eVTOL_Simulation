function configure_hex_simulink(P)
%CONFIGURE_HEX_SIMULINK Apply parameters to the independent control model.
root=fileparts(mfilename('fullpath')); mdl='hex_evtol';
P=hex_finalize_parameters(P); hex_ensure_model();
assert(strcmp(get_param(mdl,'SimulationStatus'),'stopped'),'Stop the simulation first.');
ws=get_param(mdl,'ModelWorkspace'); ws.assignin('P',P);
groups=hex_block_parameters(P); names=fieldnames(groups);
for i=1:numel(names), ws.assignin(['P_' names{i}],groups.(names{i})); end
set_param(mdl,'FastRestart','off','Dirty','off');
end
