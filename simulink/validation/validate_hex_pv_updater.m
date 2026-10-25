function report=validate_hex_pv_updater()
%VALIDATE_HEX_PV_UPDATER Regression for renamed blocks and multiple models.
% Run in a separate MATLAB session; never close a user's loaded model.
root=fileparts(fileparts(mfilename('fullpath'))); mdl='hex_evtol';
assert(~bdIsLoaded(mdl),'Run updater regression in a separate MATLAB session.');
folder=fullfile(root,'output','pv_updater_fix');
copyfile(fullfile(folder,'before_live',[mdl '.slx']),fullfile(folder,[mdl '.slx']));
load_system(fullfile(folder,[mdl '.slx']));
blocks=find_system(mdl,'LookUnderMasks','all','FollowLinks','on','SFBlockType','MATLAB Function');
for i=1:numel(blocks)
    config=get_param(blocks{i},'MATLABFunctionConfiguration');
    script=char(config.FunctionScript);
    if contains(script,'function reference=task_reference(')
        parent=get_param(blocks{i},'Parent');
        set_param(blocks{i},'Name','Task code renamed');
        set_param(parent,'Name','Task subsystem renamed');
    elseif contains(script,'=controller_step(')
        parent=get_param(blocks{i},'Parent');
        set_param(blocks{i},'Name','Control code renamed');
        set_param(parent,'Name','Controller subsystem renamed');
    end
end
set_param([mdl '/Flight Recorder/Reference Position'],'Name','Reference splitter renamed');
set_param([mdl '/Flight Recorder'],'Name','Recorder renamed');
clock=[mdl '/Simulation Clock']; mask=Simulink.Mask.get(clock);
if isempty(mask), mask=Simulink.Mask.create(clock); end
mask.Display='disp(''clock retained'')';
display_position=get_param([mdl '/Display'],'Position');
decoy='hex_pv_update_decoy'; new_system(decoy);
add_block('simulink/User-Defined Functions/MATLAB Function',[decoy '/Task decoy']);
decoy_config=get_param([decoy '/Task decoy'],'MATLABFunctionConfiguration');
decoy_script=sprintf('function reference=task_reference(t)\nreference=zeros(6,1)+t;\nend');
decoy_config.FunctionScript=decoy_script;
backup=update_hex_pv_model();
assert(strcmp(decoy_config.FunctionScript,decoy_script),'Another loaded model was modified.');
assert(strcmp(Simulink.Mask.get(clock).Display,'disp(''clock retained'')'),'Clock mask was lost.');
assert(isequal(display_position,get_param([mdl '/Display'],'Position')),'Display moved.');
assert(strcmp(get_param([mdl '/Recorder renamed/Reference splitter renamed'],'Outputs'),'[3 3]'));
out=sim(mdl,'StopTime','.02');
assert(numel(out.reference_log.Data)==6*numel(out.reference_log.Time),'Reference must have six values.');
assert(all(isfinite(out.state_log.Data),'all'),'Nonfinite state after migration.');
report=struct('passed',true,'renamed_task_and_controller',true,...
    'renamed_function_blocks',true,'renamed_recorder_and_demux',true,...
    'other_loaded_model_untouched',true,'clock_mask_retained',true,...
    'display_position_retained',true,'reference_width',6,'simulation_duration_s',.02,...
    'backup',backup);
close_system(mdl,0); close_system(decoy,0);
fprintf('PV_UPDATER_REGRESSION_PASSED\n');
end
