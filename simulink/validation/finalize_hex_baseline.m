function receipt=finalize_hex_baseline()
%FINALIZE_HEX_BASELINE Preserve user edits, update the primary SLX, read back.
root=fileparts(fileparts(mfilename('fullpath'))); addpath(root);
folder=fullfile(root,'output','baseline_tuning'); mdl='hex_evtol';
checked=jsondecode(fileread(fullfile(root,'validation','BASELINE_TUNING_RESULTS.json')));
parity=jsondecode(fileread(fullfile(root,'validation','BASELINE_PYTHON_PARITY.json')));
components=jsondecode(fileread(fullfile(root,'validation','BASELINE_COMPONENT_RESULTS.json')));
assert(checked.passed && parity.passed && components.passed,'Complete current-gain validation first.');
load_system(fullfile(root,[mdl '.slx']));
before=appearance(mdl);
backup=update_hex_baseline_model();
assert(isequal(before,appearance(mdl)),'Block names, positions or masks changed.');
close_system(mdl,0); load_system(fullfile(root,[mdl '.slx']));
ws=get_param(mdl,'ModelWorkspace'); P=ws.getVariable('P'); controller=ws.getVariable('P_controller_step');
defaults=hex_parameters(); fields={'position_p','velocity_p','attitude_p','rate_p','rate_d'};
for i=1:numel(fields)
    field=fields{i};
    assert(isequal(P.control.(field)(1:2),defaults.control.(field)(1:2)) && ...
        isequal(controller.control.(field)(1:2),defaults.control.(field)(1:2)),...
        'Saved gains were not applied to both parameter structures.');
end
assert(strcmp(P.control_tuning_version,defaults.control_tuning_version));
out=sim(mdl); state=values(out.state_log); reference=values(out.reference_log);
tracking=vecnorm(state(:,1:3)-reference(:,1:3),2,2);
default_scene=struct('task_family',P.task_family,'duration_s',out.state_log.Time(end),...
    'fault_motor',P.fault_motor,'noise_enabled',P.noise_enabled,'wind_enabled',P.wind_enabled,...
    'observed_feedback',P.observed_feedback,'tracking_rmse_m',sqrt(mean(tracking.^2)));
assert(abs(out.state_log.Time(end)-P.duration)<1e-8,'Primary model stopped early.');
if P.task_family==0, assert(default_scene.tracking_rmse_m<1e-8,'Saved hover failed.'); end
Q=hex_parameters(); Q.task_family=2; Q=hex_prepare_inputs(Q);
set(groot,'DefaultFigureVisible','off');
r=run_hex_simulink(Parameters=Q,OutputName="baseline_circle");
error=vecnorm(r.state(:,1:3)-r.reference(:,1:3),2,2);
assert(sqrt(mean(error.^2))<.05 && max(error)<.10,'Primary circle tracking failed.');
receipt=struct('date',char(datetime('now','Format','yyyy-MM-dd HH:mm:ss')),...
    'primary_model_updated',true,'primary_model',fullfile(root,[mdl '.slx']),...
    'backup_model',backup,'appearance_preserved',true,'saved_gains_readback_passed',true,...
    'reference_elements',size(r.reference,2),'default_scene',default_scene,...
    'primary_circle_rmse_m',sqrt(mean(error.^2)),'primary_circle_maximum_error_m',max(error),...
    'native_cases',numel(checked.cases),'analytical_checks',numel(components.analytical_checks),...
    'python_parity_cases',numel(parity.cases),'passed',true);
fid=fopen(fullfile(root,'validation','BASELINE_DELIVERY.json'),'w');
fprintf(fid,'%s',jsonencode(receipt,PrettyPrint=true)); fclose(fid);
close_system(mdl,0);
fprintf('PRIMARY_BASELINE_UPDATED circle_rmse=%.6f appearance_preserved=1 backup=%s\n',sqrt(mean(error.^2)),backup);
end
function state=appearance(mdl)
blocks=sort(find_system(mdl,'LookUnderMasks','all','Type','Block'));
state=struct('paths',{blocks},'positions',{get_param(blocks,'Position')},...
    'masks',{get_param(blocks,'MaskDisplay')});
end
function data=values(ts)
if ts.IsTimeFirst, data=reshape(ts.Data,numel(ts.Time),[]); else, data=reshape(ts.Data,[],numel(ts.Time))'; end
end
