function configure_hex_recovery(P)
%CONFIGURE_HEX_RECOVERY Apply parameters to the independent recovery model.
root=fileparts(mfilename('fullpath')); mdl='hex_recovery';
hex_validate_parameters(P);
if ~isfile(fullfile(root,[mdl '.slx'])), build_hex_recovery_model(); end
load_system(fullfile(root,[mdl '.slx']));
assert(strcmp(get_param(mdl,'SimulationStatus'),'stopped'),'Stop the simulation first.');
ws=get_param(mdl,'ModelWorkspace'); ws.assignin('P',P);
groups=hex_recovery_block_parameters(P); names=fieldnames(groups);
for i=1:numel(names), ws.assignin(['P_' names{i}],groups.(names{i})); end
end
