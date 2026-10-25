function backup_path=update_hex_baseline_model()
%UPDATE_HEX_BASELINE_MODEL Apply checked PV gains, preserving the live diagram.
root=fileparts(mfilename('fullpath')); mdl='hex_evtol';
receipt=fullfile(root,'validation','BASELINE_TUNING_RESULTS.json');
assert(isfile(receipt),'Run validate_hex_baseline before applying the baseline tuning.');
checked=jsondecode(fileread(receipt)); defaults=hex_parameters();
assert(checked.passed && isequal(checked.control,defaults.control),...
    'The current gains do not match the checked baseline tuning.');
if ~bdIsLoaded(mdl), load_system(fullfile(root,[mdl '.slx'])); end
assert(strcmp(get_param(mdl,'SimulationStatus'),'stopped'),...
    'Stop the simulation before updating the baseline gains.');
ws=get_param(mdl,'ModelWorkspace'); P=ws.getVariable('P');
assert(isfield(P,'reference_mode') && strcmp(P.reference_mode,defaults.reference_mode),...
    'Run update_hex_pv_model first to use the six-element reference interface.');
original=get_param(mdl,'FileName');
if isempty(original), original=fullfile(root,[mdl '.slx']); end
folder=fullfile(root,'output','baseline_tuning',...
    ['live_backup_' char(datetime('now','Format','yyyyMMdd_HHmmss_SSS'))]);
mkdir(folder); backup_path=fullfile(folder,[mdl '.slx']);
save_system(mdl,backup_path); save_system(mdl,original);
controller=ws.getVariable('P_controller_step');
fields={'position_p','velocity_p','attitude_p','rate_p','rate_d'};
for i=1:numel(fields)
    field=fields{i};
    P.control.(field)(1:2)=defaults.control.(field)(1:2);
    controller.control.(field)(1:2)=defaults.control.(field)(1:2);
end
P.control_tuning_version=defaults.control_tuning_version;
ws.assignin('P',P); ws.assignin('P_controller_step',controller);
set_param(mdl,'SimulationCommand','update'); save_system(mdl,original);
fprintf('BASELINE_GAINS_UPDATED %s\nBACKUP %s\n',original,backup_path);
end
