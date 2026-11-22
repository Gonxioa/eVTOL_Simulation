function model_path=build_hex_model()
%BUILD_HEX_MODEL Build and verify every numerical block before publishing SLX.
root=fileparts(mfilename('fullpath'));model='hex_evtol';
model_path=fullfile(root,[model '.slx']);
if bdIsLoaded(model)
    assert(strcmp(get_param(model,'SimulationStatus'),'stopped'),'hex:RunningModel','Stop the running model first.');
    assert(strcmp(get_param(model,'Dirty'),'off'),'hex:DirtyModel',...
        'The open model has unsaved edits. Run the build in a separate MATLAB session.');
    close_system(model,0);
end
staging=fullfile(root,'build',char(java.util.UUID.randomUUID()));mkdir(staging);
staged=fullfile(staging,[model '.slx']);
copyfile(fullfile(root,'internal','layout_template.slx'),staged);
load_system(staged);cleanup=onCleanup(@()close_loaded(model));
P=hex_parameters();workspace=get_param(model,'ModelWorkspace');workspace.assignin('P',P);
groups=hex_block_parameters(P);fields=fieldnames(groups);
for i=1:numel(fields),workspace.assignin(['P_' fields{i}],groups.(fields{i}));end
scripts=hex_embedded_scripts();seen=false(numel(scripts),1);
blocks=find_system(model,'LookUnderMasks','all','FollowLinks','on','SFBlockType','MATLAB Function');
task='';controller='';sensor='';
for i=1:numel(blocks)
    cfg=get_param(blocks{i},'MATLABFunctionConfiguration');
    token=regexp(char(cfg.FunctionScript),'^function[^\r\n]*=\s*(\w+)\s*\(','tokens','once');
    assert(~isempty(token),'hex:BuildLayout','Cannot identify numerical block %s.',blocks{i});
    name=token{1};if strcmp(name,'task_reference'),name='reference_step';end
    index=find(strcmp({scripts.function_name},name));
    assert(numel(index)==1 && ~seen(index),'hex:BuildLayout','Unexpected function %s.',name);
    cfg.FunctionScript=scripts(index).script;seen(index)=true;
    switch name
        case 'reference_step',task=blocks{i};
        case 'controller_step',controller=blocks{i};
        case 'sensor_step',sensor=blocks{i};
    end
end
assert(all(seen),'hex:BuildLayout','Required blocks are missing from the layout template.');
parent=get_param(task,'Parent');
input=add_block('simulink/Ports & Subsystems/In1',[parent '/State Feedback'],...
    'Port','2','Position',[35 115 65 129]);
output=add_block('simulink/Ports & Subsystems/Out1',[parent '/Path Progress'],...
    'Port','2','Position',[480 140 510 154]);
ports=get_param(task,'PortHandles');inports=get_param(input,'PortHandles');outports=get_param(output,'PortHandles');
add_line(parent,inports.Outport(1),ports.Inport(2),'autorouting','on');
add_line(parent,ports.Outport(2),outports.Inport(1),'autorouting','on');
source=get_param(get_param(sensor,'Parent'),'PortHandles');target=get_param(parent,'PortHandles');
add_line(model,source.Outport(1),target.Inport(2),'autorouting','on');
add_log(model,target.Outport(2),'Path Progress Log','reference_audit_log',[300 650 500 685]);
parent=get_param(controller,'Parent');
output=add_block('simulink/Ports & Subsystems/Out1',[parent '/Integral Diagnostics'],...
    'Port','3','Position',[480 200 510 214]);
ports=get_param(controller,'PortHandles');outports=get_param(output,'PortHandles');
add_line(parent,ports.Outport(3),outports.Inport(1),'autorouting','on');
source=get_param(parent,'PortHandles');
add_log(model,source.Outport(3),'Controller Diagnostics Log','controller_diagnostic_log',[550 650 740 685]);
identity=hex_build_sources();identity.controller_version=P.control_tuning_version;
identity.matlab_version=version;identity.created_utc=char(datetime('now','TimeZone','UTC'));
identity.embedded_hashes=rmfield(scripts,{'script','parameter'});
workspace.assignin('HEX_BUILD_IDENTITY',identity);
set_param(model,'FastRestart','off');
set_param(model,'InitFcn','hex_check_loaded_model(bdroot);');
set_param(model,'Description',['Six-rotor control model ' P.control_tuning_version]);
hex_check_loaded_model(model);
set_param(model,'SimulationCommand','update');
save_system(model,staged);close_system(model,0);
current=hex_build_sources();
assert(strcmp(identity.source_signature,current.source_signature),...
    'hex:BuildChanged','Dependencies changed while building. The previous model was retained.');
if isfile(model_path)
    history=fullfile(root,'build','history',hex_hash_file(model_path));
    if ~isfolder(history),mkdir(history);end
    copyfile(model_path,fullfile(history,[model '.slx']));
end
movefile(staged,model_path,'f');
identity.model_sha256=hex_hash_file(model_path);
fid=fopen(fullfile(root,'hex_model_build.json'),'w','n','UTF-8');assert(fid>=0);
fprintf(fid,'%s',jsonencode(identity,PrettyPrint=true));fclose(fid);
fprintf('SIMULATION_MODEL_SAVED %s source=%s\n',model_path,identity.source_signature);
end

function add_log(model,source,name,variable,position)
sink=add_block('simulink/Sinks/To Workspace',[model '/' name],...
    'VariableName',variable,'SaveFormat','Timeseries','MaxDataPoints','inf',...
    'SampleTime','P.export_dt','Position',position);
ports=get_param(sink,'PortHandles');add_line(model,source,ports.Inport(1),'autorouting','on');
end

function close_loaded(model)
if bdIsLoaded(model),close_system(model,0);end
end
