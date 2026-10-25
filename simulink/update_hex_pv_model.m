function backup_path=update_hex_pv_model()
%UPDATE_HEX_PV_MODEL Update the loaded model without rebuilding its layout.
% Safe to run in the user's MATLAB session: save a full backup of the current
% in-memory model first, including unsaved clock masks, displays and wiring.
root=fileparts(mfilename('fullpath')); mdl='hex_evtol';
if ~bdIsLoaded(mdl), load_system(fullfile(root,[mdl '.slx'])); end
assert(strcmp(get_param(mdl,'SimulationStatus'),'stopped'),...
    'Stop the simulation before updating the reference interface.');
original_path=get_param(mdl,'FileName');
if isempty(original_path), original_path=fullfile(root,[mdl '.slx']); end
[task_block,control_block]=locate_functions(mdl);
demux=locate_reference_demux(mdl);
fprintf('PV_UPDATE_TASK %s\nPV_UPDATE_CONTROLLER %s\n',task_block,control_block);
folder=fullfile(root,'output','pv_migration',['snapshot_' char(datetime('now','Format','yyyyMMdd_HHmmss_SSS'))]);
mkdir(folder); backup_path=fullfile(folder,[mdl '.slx']);
save_system(mdl,backup_path);
save_system(mdl,original_path);

ws=get_param(mdl,'ModelWorkspace'); P=ws.getVariable('P');
if ~isfield(P.control,'yaw_hold_rad'), P.control.yaw_hold_rad=0; end
for field={'yaw_start','yaw_change'}
    if isfield(P,field{1}), P=rmfield(P,field{1}); end
end
P.reference_mode='position_velocity_fixed_heading';
ws.assignin('P',P);
groups=hex_block_parameters(P);
% Keep each existing block's tuning; change only the reference policy.
controller_parameters=ws.getVariable('P_controller_step');
if ~isfield(controller_parameters.control,'yaw_hold_rad')
    controller_parameters.control.yaw_hold_rad=P.control.yaw_hold_rad;
end
ws.assignin('P_controller_step',controller_parameters);
task_parameters=ws.getVariable('P_task_reference');
for field={'yaw_start','yaw_change'}
    if isfield(task_parameters,field{1}), task_parameters=rmfield(task_parameters,field{1}); end
end
assert(all(isfield(task_parameters,fieldnames(groups.task_reference))),...
    'Required task parameters are missing; inspect the backup before continuing.');
ws.assignin('P_task_reference',task_parameters);
task_config=get_param(task_block,'MATLABFunctionConfiguration');
control_config=get_param(control_block,'MATLABFunctionConfiguration');
task_config.FunctionScript=embedded('task_reference',{});
control_config.FunctionScript=embedded('controller_step',...
    {'hex_rotation','hex_multiply','hex_desired_attitude','hex_allocate','hex_trf'});
set_param(demux,'Outputs','[3 3]');
task_parent=get_param(task_block,'Parent');
if strcmp(task_parent,mdl), task_parent=task_block; end
ports=get_param(task_parent,'PortHandles');
line=get_param(ports.Outport(1),'Line');
if line~=-1, set_param(line,'Name','reference p v'); end
marker='PV reference: position/velocity only; internal fixed heading; no acceleration or yaw-rate feedforward.';
description=get_param(mdl,'Description');
if ~contains(description,marker), set_param(mdl,'Description',[description newline marker]); end
set_param(mdl,'SimulationCommand','update');
save_system(mdl,original_path);
fprintf('PV_MODEL_UPDATED %s\nBACKUP %s\n',original_path,backup_path);

    function script=embedded(source,helpers)
        script=fileread(fullfile(root,'blocks',[source '.m']));
        for h=1:numel(helpers)
            script=[script newline fileread(fullfile(root,'functions',[helpers{h} '.m']))]; %#ok<AGROW>
        end
        script=regexprep(script,'\<P\>',['P_' source]);
    end

    function [task,controller]=locate_functions(model)
        % Inspect blocks within this model through the Simulink API. Labels
        % may be renamed, and other loaded models must not affect selection.
        blocks=find_system(model,'LookUnderMasks','all','FollowLinks','on',...
            'SFBlockType','MATLAB Function');
        tasks={}; controllers={};
        for k=1:numel(blocks)
            config=get_param(blocks{k},'MATLABFunctionConfiguration');
            script=char(config.FunctionScript);
            if ~isempty(regexp(script,'(?m)^\s*function[^\r\n]*=\s*task_reference\s*\(','once'))
                tasks{end+1}=blocks{k}; %#ok<AGROW>
            end
            if ~isempty(regexp(script,'(?m)^\s*function[^\r\n]*=\s*controller_step\s*\(','once'))
                controllers{end+1}=blocks{k}; %#ok<AGROW>
            end
        end
        if numel(tasks)~=1 || numel(controllers)~=1
            error('hex:pv:FunctionLookup',...
                ['Found %d task_reference blocks and %d controller_step blocks in %s. ',...
                'Expected one of each. MATLAB Function blocks:\n%s'],...
                numel(tasks),numel(controllers),model,strjoin(blocks,newline));
        end
        task=tasks{1}; controller=controllers{1};
    end

    function block=locate_reference_demux(model)
        default=[model '/Flight Recorder/Reference Position'];
        if getSimulinkBlockHandle(default)>0
            block=default; return
        end
        blocks=find_system(model,'LookUnderMasks','all','FollowLinks','on','BlockType','Demux');
        matches={};
        for k=1:numel(blocks)
            widths=sscanf(regexprep(get_param(blocks{k},'Outputs'),'[\[\],]',' '),'%f')';
            if isequal(widths,[3 8]) || isequal(widths,[3 3])
                matches{end+1}=blocks{k}; %#ok<AGROW>
            end
        end
        assert(numel(matches)==1,'Unable to uniquely identify the reference position/velocity Demux.');
        block=matches{1};
    end
end
