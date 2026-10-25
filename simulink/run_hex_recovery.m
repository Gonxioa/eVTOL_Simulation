function result=run_hex_recovery(options)
%RUN_HEX_RECOVERY Run the independent recovery SLX, not the legacy model.
arguments
    options.Task (1,1) string="circle"
    options.FaultMotor (1,1) double=1
    options.FaultFactor (1,1) double=0
    options.FaultTime (1,1) double=3
    options.Duration (1,1) double=24
    options.ObservedFeedback (1,1) logical=true
    options.Noise (1,1) logical=true
    options.Wind (1,1) logical=true
    options.Delay (1,1) double=.03
    options.Seed (1,1) double=20261006
    options.Parameters struct=struct()
    options.SaveResults (1,1) logical=true
    options.OutputName (1,1) string="latest"
end
root=fileparts(mfilename('fullpath')); mdl='hex_recovery';
if ~isfile(fullfile(root,[mdl '.slx'])), build_hex_recovery_model(); end
P=options.Parameters;
if isempty(fieldnames(P))
    P=hex_recovery_parameters();
    tasks=["hover","line","circle","vertical","tracking","step","saturation"];
    index=find(tasks==options.Task,1); assert(~isempty(index),'Unknown task.');
    P.task_family=index-1; P.fault_motor=options.FaultMotor; P.fault_factor=options.FaultFactor;
    P.fault_start=options.FaultTime; P.duration=options.Duration;
    P.observed_feedback=options.ObservedFeedback; P.noise_enabled=options.Noise;
    P.wind_enabled=options.Wind; P.feedback_delay=options.Delay; P.seed=options.Seed;
    P=hex_prepare_inputs(P);
end
hex_validate_parameters(P); load_system(fullfile(root,[mdl '.slx']));
input=Simulink.SimulationInput(mdl); input=input.setVariable('P',P,'Workspace',mdl);
groups=hex_recovery_block_parameters(P); fields=fieldnames(groups);
for i=1:numel(fields), input=input.setVariable(['P_' fields{i}],groups.(fields{i}),'Workspace',mdl); end
out=sim(input); result=struct('parameters',P);
for name={'state','reference','motor_command','imu','observation','wind','effectiveness','control_audit','feedback','recovery_reference_audit'}
    ts=out.get([name{1} '_log']);
    if strcmp(name{1},'state'), result.time=ts.Time; end
    if ts.IsTimeFirst, value=reshape(ts.Data,numel(ts.Time),[]);
    else, value=reshape(ts.Data,[],numel(ts.Time))'; end
    result.(name{1})=value;
end
result.termination='completed';
if result.time(end)<P.duration-P.export_dt/2, result.termination='physical_envelope_exit'; end
result.reference_mode=P.reference_mode;
if options.SaveResults
    folder=fullfile(root,'output','recovery',P.control_tuning_version,char(options.OutputName));
    if ~isfolder(folder), mkdir(folder); end
    save(fullfile(folder,'result.mat'),'result','-v7');
    data=[result.time,result.state,result.reference,result.motor_command,result.imu,result.observation,...
        result.wind,result.effectiveness,result.control_audit,result.feedback,result.recovery_reference_audit];
    names=hex_signal_names();
    names=strrep(names,'wrench_nominal_allocated','wrench_estimated_allocated');
    names=strrep(names,'wrench_nominal_residual','wrench_estimated_residual');
    names=[names(1:end-19),compose('estimated_effectiveness_%d',1:6),{'recovery_damage'},names(end-18:end),...
        {'path_phase_s','path_progress_rate'},compose('nominal_reference_%d',1:6),{'governor_position_error_m','governor_tilt_deg'}];
    writetable(array2table(data,'VariableNames',names),fullfile(folder,'signals.csv'));
end
fprintf('RECOVERY_SLX m=%d eta=%.2f task=%d duration=%.2f %s final=%.4f\n',...
    P.fault_motor,P.fault_factor,P.task_family,result.time(end),result.termination,...
    norm(result.state(end,1:3)-result.reference(end,1:3)));
end
