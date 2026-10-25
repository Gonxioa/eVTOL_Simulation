function report=validate_hex_baseline(selected_names)
%VALIDATE_HEX_BASELINE Behavioral acceptance of the current PV tuning in SLX.
root=fileparts(mfilename('fullpath'));
addpath(root,fullfile(root,'blocks'));
folder=fullfile(root,'output','baseline_tuning');
work=fullfile(folder,'validated_model'); if ~isfolder(work), mkdir(work); end
mdl='hex_evtol'; original_pwd=pwd; cleanup=onCleanup(@()restore(mdl,original_pwd));
if bdIsLoaded(mdl)
    assert(~strcmp(get_param(mdl,'Dirty'),'on'),...
        'Use a separate MATLAB session for validation; the open model has unsaved edits.');
    close_system(mdl,0);
end
copyfile(fullfile(root,[mdl '.slx']),fullfile(work,[mdl '.slx']));
cd(work); load_system(fullfile(work,[mdl '.slx']));
set_param(mdl,'FastRestart','off');
base=hex_parameters(); base=hex_prepare_inputs(base);
apply(base); save_system(mdl);
% [family, duration, RMSE ceiling, peak ceiling, final-error ceiling].
plans=[0 8 1e-8 1e-8 1e-8;1 8 .02 .04 .01;2 8 .05 .10 .01;...
    3 8 .04 .06 .01;4 8 .06 .12 .08;5 8 .25 .80 .03;...
    0 12 .06 .20 .01;2 12 .06 .12 .01;2 8 .06 .12 .05;...
    2 20 .10 .25 .20;0 8 .15 .35 .10;0 8 .15 .35 .10;2 8 .05 .10 .01];
names={'hover','line','circle','vertical','tracking','step','perturbed_hover',...
    'large_reverse_circle','observed_noise_wind_10ms','observed_noise_wind_30ms',...
    'partial_rotor_1','partial_rotor_6','circle_half_integration_step'};
results=struct([]); traces=struct(); indices=1:size(plans,1);
if nargin>0
    saved=load(fullfile(folder,'baseline_traces.mat'),'traces','report');
    assert(isequal(saved.report.control,base.control),'Saved trials use different gains.');
    results=saved.report.cases; traces=saved.traces;
    indices=find(ismember(names,cellstr(selected_names)));
    assert(~isempty(indices),'No matching baseline scenarios.');
end
for i=indices
    P=base; P.task_family=plans(i,1); P.duration=plans(i,2);
    if i==7
        P.initial_state(1:3)=P.initial_state(1:3)+[.1;-.1;.05];
        P.initial_state(7:10)=[sin(2.5*pi/180);0;0;cos(2.5*pi/180)];
    elseif i==8
        P.radius=.5; P.turn=-2*pi; P.heading=pi/3;
        P.motion_start=2; P.motion_end=10;
    elseif i==9 || i==10
        P.observed_feedback=true; P.noise_enabled=true; P.wind_enabled=true;
        if i==10, P.feedback_delay=.03; end
    elseif i==11 || i==12
        P.fault_motor=1+5*(i==12); P.fault_start=2; P.fault_factor=.8;
    elseif i==13
        P.integration_dt=P.integration_dt/2;
    end
    P=hex_prepare_inputs(P); hex_validate_parameters(P);
    input=Simulink.SimulationInput(mdl); input=input.setVariable('P',P,'Workspace',mdl);
    groups=hex_block_parameters(P); fields=fieldnames(groups);
    for j=1:numel(fields), input=input.setVariable(['P_' fields{j}],groups.(fields{j}),'Workspace',mdl); end
    out=sim(input);
    r=struct('time',out.state_log.Time,'parameters',P);
    for field={'state','reference','motor_command','imu','observation','wind','effectiveness','control_audit','feedback'}
        r.(field{1})=values(out.get([field{1} '_log']));
    end
    errors=vecnorm(r.state(:,1:3)-r.reference(:,1:3),2,2);
    tilt=acosd(min(max(1-2*(r.state(:,7).^2+r.state(:,8).^2),-1),1));
    sample=min(401,numel(r.time)); expected=task_reference(r.time(sample),P);
    actual_reference_ok=size(r.reference,2)==6 && norm(r.reference(sample,:)'-expected)<1e-10;
    bounds_ok=all(r.motor_command>=P.omega_min-1e-8 & r.motor_command<=P.omega_max+1e-8,'all');
    finite_ok=all(isfinite(r.state),'all') && all(isfinite(r.control_audit),'all');
    quaternion_error=max(abs(vecnorm(r.state(:,7:10),2,2)-1));
    integrals_ok=all(abs(r.control_audit(:,13:15))<=P.control.velocity_integral_limit_m_s2'+1e-8,'all') && ...
        all(abs(r.control_audit(:,16:18))<=P.control.rate_integral_limit_rad_s2'+1e-8,'all');
    wind_ok=P.wind_enabled==any(abs(r.wind)>1e-12,'all');
    completed=abs(r.time(end)-P.duration)<1e-8;
    item=struct('name',names{i},'rmse_m',sqrt(mean(errors.^2)),'maximum_error_m',max(errors),...
        'final_error_m',errors(end),'maximum_tilt_deg',max(tilt),...
        'minimum_command_rad_s',min(r.motor_command,[],'all'),...
        'maximum_command_rad_s',max(r.motor_command,[],'all'),...
        'quaternion_norm_error',quaternion_error,'actual_duration_s',r.time(end),...
        'completed',completed,'reference_applied',actual_reference_ok,'commands_bounded',bounds_ok,...
        'integrals_bounded',integrals_ok,'wind_switch_applied',wind_ok,...
        'rmse_ceiling_m',plans(i,3),'peak_ceiling_m',plans(i,4),'final_ceiling_m',plans(i,5));
    item.passed=completed && actual_reference_ok && bounds_ok && finite_ok && integrals_ok && wind_ok && ...
        quaternion_error<1e-12 && item.rmse_m<=plans(i,3) && item.maximum_error_m<=plans(i,4) && item.final_error_m<=plans(i,5);
    if i<=4 || i==13
        item.passed=item.passed && item.maximum_tilt_deg<=25;
    end
    if i==13
        difference=max(vecnorm(r.state(:,1:3)-traces.circle.state(:,1:3),2,2));
        item.integration_position_difference_m=difference;
        item.passed=item.passed && difference<.002;
    else
        item.integration_position_difference_m=0;
    end
    match=[]; if ~isempty(results), match=find(strcmp({results.name},names{i}),1); end
    if isempty(results), results=item; elseif isempty(match), results(end+1)=item; else, results(match)=item; end %#ok<AGROW>
    traces.(names{i})=r;
    writematrix([r.time,r.state,r.reference,r.motor_command],fullfile(folder,[names{i} '.csv']));
    fprintf('BASELINE %s passed=%d rmse=%.6f max=%.6f final=%.6f tilt=%.2f duration=%.2f\n',...
        names{i},item.passed,item.rmse_m,item.maximum_error_m,item.final_error_m,item.maximum_tilt_deg,item.actual_duration_s);
    report=struct('date',char(datetime('now','Format','yyyy-MM-dd HH:mm:ss')),...
        'matlab_version',version,'reference_mode',base.reference_mode,...
        'control_tuning_version',base.control_tuning_version,'control',base.control,...
        'validated_model_file',fullfile(work,[mdl '.slx']),'cases',results,'passed',all([results.passed]));
    fid=fopen(fullfile(folder,'baseline_results.json'),'w'); fprintf(fid,'%s',jsonencode(report,PrettyPrint=true)); fclose(fid);
end
save(fullfile(folder,'baseline_traces.mat'),'traces','report','-v7');
set(groot,'DefaultFigureVisible','off');
fig=figure('Color','w','Position',[80 80 1050 440]); tiledlayout(1,2,'TileSpacing','compact');
nexttile; plot(traces.circle.reference(:,1),traces.circle.reference(:,2),'k--'); hold on
plot(traces.circle.state(:,1),traces.circle.state(:,2),'LineWidth',1.5); axis equal; grid on
xlabel('East (m)'); ylabel('North (m)'); legend('Reference','Actual','Location','best'); title('PV circle tracking');
nexttile; hold on
for field={'line','circle','vertical'}
    r=traces.(field{1}); plot(r.time,vecnorm(r.state(:,1:3)-r.reference(:,1:3),2,2),'LineWidth',1.2);
end
grid on; xlabel('Time (s)'); ylabel('Position error (m)'); legend('Line','Circle','Vertical','Location','best');
title('Unchanged task geometry and timing');
exportgraphics(fig,fullfile(folder,'baseline_tracking.png'),'Resolution',150); savefig(fig,fullfile(folder,'baseline_tracking.fig')); close(fig);
assert(report.passed,'Baseline acceptance failed; inspect output/baseline_tuning/baseline_results.json.');
copyfile(fullfile(folder,'baseline_results.json'),fullfile(root,'validation','BASELINE_TUNING_RESULTS.json'));
fprintf('BASELINE_VALIDATION_PASSED %d\n',numel(results));
    function apply(P)
        ws=get_param(mdl,'ModelWorkspace'); ws.assignin('P',P);
        groups=hex_block_parameters(P); fields=fieldnames(groups);
        for j=1:numel(fields), ws.assignin(['P_' fields{j}],groups.(fields{j})); end
    end
end
function data=values(ts)
if ts.IsTimeFirst, data=reshape(ts.Data,numel(ts.Time),[]); else, data=reshape(ts.Data,[],numel(ts.Time))'; end
end
function restore(mdl,folder)
if bdIsLoaded(mdl), set_param(mdl,'FastRestart','off'); close_system(mdl,0); end
cd(folder);
end
