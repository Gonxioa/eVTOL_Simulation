function result=run_hex_simulink(options)
%RUN_HEX_SIMULINK Run native Simulink, export CSV/MAT and optional plots.
% Example: run_hex_simulink(Task="circle",FaultMotor=1,FaultFactor=.8)
arguments
    options.Task (1,1) string="hover"
    options.FaultMotor (1,1) double=0
    options.FaultFactor (1,1) double=.8
    options.FaultTime (1,1) double=3
    options.RecoveryTime (1,1) double=Inf
    options.Duration (1,1) double=8
    options.Noise (1,1) logical=false
    options.Wind (1,1) logical=false
    options.ObservedFeedback (1,1) logical=false
    options.Delay (1,1) double=.01
    options.Seed (1,1) double=20261006
    options.Parameters struct=struct()
    options.MakePlots (1,1) logical=true
    options.PlotMotor double=1:6
    options.SaveResults (1,1) logical=true
    options.OutputName (1,1) string="latest"
end
validateattributes(options.PlotMotor,{'double'},{'vector','nonempty','integer','>=',1,'<=',6},mfilename,'PlotMotor');
root=fileparts(mfilename('fullpath')); mdl='hex_evtol';
if ~isfile(fullfile(root,[mdl '.slx'])), build_hex_model(); end
if isempty(fieldnames(options.Parameters))
    P=hex_parameters(); tasks=["hover","line","circle","vertical","tracking","step","saturation"];
    idx=find(tasks==options.Task,1);
    assert(~isempty(idx),'Task must be hover, line, circle, vertical, tracking, step or saturation.');
    P.task_family=idx-1; P.fault_motor=options.FaultMotor; P.fault_factor=options.FaultFactor;
    P.fault_start=options.FaultTime; P.fault_end=options.RecoveryTime; P.duration=options.Duration;
    P.motion_end=min(7,P.duration); P.motion_start=min(1,P.motion_end/2);
    P.noise_enabled=options.Noise; P.wind_enabled=options.Wind;
    P.observed_feedback=options.ObservedFeedback; P.feedback_delay=options.Delay; P.seed=options.Seed;
    P=hex_prepare_inputs(P);
else
    P=options.Parameters;
end
hex_validate_parameters(P);
load_system(fullfile(root,[mdl '.slx']));
hex_ensure_pv_model();
input=Simulink.SimulationInput(mdl);
input=input.setVariable('P',P,'Workspace',mdl);
groups=hex_block_parameters(P); group_names=fieldnames(groups);
for g=1:numel(group_names)
    input=input.setVariable(['P_' group_names{g}],groups.(group_names{g}),'Workspace',mdl);
end
out=sim(input);
result=struct('parameters',P,'simulation',out);
names={'state','reference','motor_command','imu','observation','wind','effectiveness','control_audit','feedback'};
for j=1:numel(names)
    ts=out.get([names{j} '_log']);
    if j==1, result.time=ts.Time; end
    if ts.IsTimeFirst
        result.(names{j})=reshape(ts.Data,numel(ts.Time),[]);
    else
        result.(names{j})=reshape(ts.Data,[],numel(ts.Time))';
    end
end
result.termination='completed';
if result.time(end)<P.duration-P.export_dt/2, result.termination='physical_envelope_exit'; end
result.source_commit='a6b9f1dd3d839416e469755490b8ff7733acd4ee';
result.reference_mode='position_velocity_fixed_heading';
if options.SaveResults
    folder=fullfile(root,'output',char(options.OutputName));
    if ~isfolder(folder), mkdir(folder); end
    data=[result.time,result.state,result.reference,result.motor_command,result.imu,result.observation,result.wind,result.effectiveness,result.control_audit,result.feedback];
    writetable(array2table(data,'VariableNames',hex_signal_names()),fullfile(folder,'signals.csv'));
    save(fullfile(folder,'result.mat'),'result','-v7');
    if options.MakePlots, plot_hex_results(result,folder,options.PlotMotor); end
elseif options.MakePlots
    plot_hex_results(result,'',options.PlotMotor);
end
fprintf('SIMULINK_RUN %.3f s | %s | final position error %.6g m\n',result.time(end),result.termination,norm(result.state(end,1:3)-result.reference(end,1:3)));
end
