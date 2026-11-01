function model_path=build_hex_recovery_model()
%BUILD_HEX_RECOVERY_MODEL Build the current model from its internal layout.
% Historical runnable models are archived on GitHub, not kept as entry points.
root=fileparts(mfilename('fullpath')); mdl='hex_recovery';
model_path=fullfile(root,[mdl '.slx']);
if bdIsLoaded(mdl)
    assert(strcmp(get_param(mdl,'SimulationStatus'),'stopped'),'Stop the recovery model before rebuilding.');
    memory_backup=fullfile(root,'output','recovery_tuning',['memory_backup_' char(datetime('now','Format','yyyyMMdd_HHmmss_SSS'))]);
    mkdir(memory_backup); save_system(mdl,fullfile(memory_backup,[mdl '.slx']));
    close_system(mdl,0);
end
if isfile(model_path)
    backup=fullfile(root,'output','recovery_tuning',['model_backup_' char(datetime('now','Format','yyyyMMdd_HHmmss_SSS'))]);
    mkdir(backup); copyfile(model_path,fullfile(backup,[mdl '.slx']));
end
template=fullfile(root,'internal','recovery_layout_template.slx');
assert(isfile(template),'The internal recovery layout template is missing.');
copyfile(template,model_path);
load_system(model_path); ws=get_param(mdl,'ModelWorkspace');
P=hex_recovery_parameters(); ws.assignin('P',P);
groups=hex_recovery_block_parameters(P); fields=fieldnames(groups);
for i=1:numel(fields), ws.assignin(['P_' fields{i}],groups.(fields{i})); end
blocks=find_system(mdl,'LookUnderMasks','all','FollowLinks','on','SFBlockType','MATLAB Function');
task=''; controller=''; sensor='';
for i=1:numel(blocks)
    cfg=get_param(blocks{i},'MATLABFunctionConfiguration'); script=char(cfg.FunctionScript);
    if ~isempty(regexp(script,'^function[^\r\n]*=\s*task_reference\s*\(','once')), task=blocks{i}; end
    if ~isempty(regexp(script,'^function[^\r\n]*=\s*controller_step\s*\(','once')), controller=blocks{i}; end
    if ~isempty(regexp(script,'^function[^\r\n]*=\s*sensor_step\s*\(','once')), sensor=blocks{i}; end
end
assert(~isempty(task) && ~isempty(controller) && ~isempty(sensor),'Expected task/controller/sensor functions.');
cfg=get_param(task,'MATLABFunctionConfiguration');
script=[fileread(fullfile(root,'blocks','recovery_reference_step.m')) newline ...
    fileread(fullfile(root,'blocks','task_reference.m'))];
cfg.FunctionScript=regexprep(script,'\<P\>','P_task_reference');
parent=get_param(task,'Parent');
input=add_block('simulink/Ports & Subsystems/In1',[parent '/Recovery Feedback'],...
    'Port','2','Position',[35 115 65 129]);
output=add_block('simulink/Ports & Subsystems/Out1',[parent '/Recovery Progress'],...
    'Port','2','Position',[480 140 510 154]);
ports=get_param(task,'PortHandles');
inports=get_param(input,'PortHandles'); outports=get_param(output,'PortHandles');
add_line(parent,inports.Outport(1),ports.Inport(2),'autorouting','on');
add_line(parent,ports.Outport(2),outports.Inport(1),'autorouting','on');
sensor_parent=get_param(sensor,'Parent');
source=get_param(sensor_parent,'PortHandles'); target=get_param(parent,'PortHandles');
add_line(mdl,source.Outport(1),target.Inport(2),'autorouting','on');
sink=add_block('simulink/Sinks/To Workspace',[mdl '/Recovery Progress Log'],...
    'VariableName','recovery_reference_audit_log','SaveFormat','Timeseries',...
    'MaxDataPoints','inf','SampleTime','P.export_dt','Position',[300 650 500 685]);
sinkports=get_param(sink,'PortHandles');
add_line(mdl,target.Outport(2),sinkports.Inport(1),'autorouting','on');
cfg=get_param(controller,'MATLABFunctionConfiguration');
script=fileread(fullfile(root,'blocks','recovery_controller_step.m'));
for helper={'hex_rotation','hex_multiply','hex_desired_attitude','hex_allocate','hex_trf'}
    script=[script newline fileread(fullfile(root,'functions',[helper{1} '.m']))]; %#ok<AGROW>
end
cfg.FunctionScript=regexprep(script,'\<P\>','P_controller_step');
set_param(mdl,'FastRestart','off');
set_param(mdl,'Description',[get_param(mdl,'Description') newline ...
    'Recovery experiment 2026-10-23: RPM-based estimate; thrust-direction priority; measured-state path retiming.']);
set_param(mdl,'SimulationCommand','update'); save_system(mdl,model_path);
fprintf('RECOVERY_MODEL_SAVED %s\n',model_path);
close_system(mdl,0);
end
