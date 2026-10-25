function report=validate_hex_simulink(selected_names)
%VALIDATE_HEX_SIMULINK Compare actual SLX simulations with frozen Python code.
root=fileparts(mfilename('fullpath')); folder=fullfile(root,'output','validation');
fixture=fullfile(folder,'python_pv_reference.mat');
assert(isfile(fixture),'Generate fixtures first: python validation/export_python_reference.py');
loaded=load(fixture); names=fieldnames(loaded.cases); results=struct([]);
% These preserved fixtures describe the pre-retuning migration. Compare them
% with their frozen gains; validate_hex_baseline checks the CURRENT gains.
frozen=jsondecode(fileread(fullfile(folder,'python_baseline','hex_research','configs','research.json')));
assert(strcmp(loaded.reference_mode,'position_velocity_fixed_heading'),'Fixtures use a different reference policy.');
legacy=load(fullfile(folder,'python_reference.mat'));
if nargin>0, names=cellstr(selected_names); end
for i=1:numel(names)
    test=loaded.cases.(names{i}); P=hex_parameters();
    P.control=frozen.control; P.control.yaw_hold_rad=0;
    settings={'task_family','fault_motor','fault_factor','fault_start','fault_end','duration',...
        'observed_feedback','noise_enabled','wind_enabled','feedback_delay','seed','motion_start','motion_end','exogenous'};
    for j=1:numel(settings), P.(settings{j})=test.(settings{j}); end
    P.observed_feedback=logical(P.observed_feedback); P.noise_enabled=logical(P.noise_enabled); P.wind_enabled=logical(P.wind_enabled);
    r=run_hex_simulink(Parameters=P,MakePlots=false,SaveResults=false);
    assert(size(r.reference,2)==6,'Reference logging must have exactly six columns.');
    ref=test.data; N=min(size(ref,1),numel(r.time));
    expected=ref(1:N,:);
    actual=[r.time(1:N),r.state(1:N,:),r.reference(1:N,:),r.motor_command(1:N,:),r.imu(1:N,:),...
        r.observation(1:N,:),r.wind(1:N,:),r.effectiveness(1:N,:),r.control_audit(1:N,:)];
    errors=abs(actual-expected);
    item=struct('name',names{i},'samples',numel(r.time),'python_samples',size(ref,1),...
        'max_position_error_m',max(errors(:,2:4),[],'all'),...
        'max_state_error',max(errors(:,2:20),[],'all'),...
        'max_motor_command_error_rad_s',max(errors(:,27:32),[],'all'),...
        'max_motor_command_thrust_error_N',max(abs((r.motor_command(1:N,:).^2-expected(:,27:32).^2).*P.kT'),[],'all'),...
        'max_imu_error',max(errors(:,33:38),[],'all'),...
        'max_reference_error',max(errors(:,21:26),[],'all'),...
        'max_wind_error',max(errors(:,58:60),[],'all'),...
        'max_effectiveness_error',max(errors(:,61:66),[],'all'),...
        'max_integral_error',max(errors(:,79:84),[],'all'),...
        'termination',r.termination,'python_termination',test.termination);
    tracking_error=vecnorm(r.state(:,1:3)-r.reference(:,1:3),2,2);
    item.position_tracking_rmse_m=sqrt(mean(tracking_error.^2));
    item.maximum_tracking_error_m=max(tracking_error);
    previous=legacy.cases.(names{i}).data;
    previous_error=vecnorm(previous(:,2:4)-previous(:,21:23),2,2);
    item.legacy_position_tracking_rmse_m=sqrt(mean(previous_error.^2));
    item.legacy_actual_duration_s=previous(end,1);
    item.quaternion_norm_error=max(abs(vecnorm(r.state(:,7:10),2,2)-1));
    item.commands_bounded=all(r.motor_command>=-1e-8 & r.motor_command<=P.omega_max+1e-8,'all');
    item.integrals_bounded=all(abs(r.control_audit(:,13:15))<=P.control.velocity_integral_limit_m_s2'+1e-8,'all') && ...
        all(abs(r.control_audit(:,16:18))<=P.control.rate_integral_limit_rad_s2'+1e-8,'all');
    item.passed=item.samples==item.python_samples && strcmp(item.termination,item.python_termination) && ...
        item.max_position_error_m<1e-3 && item.max_motor_command_error_rad_s<.5 && item.max_motor_command_thrust_error_N<.005 && item.max_imu_error<.02 && ...
        item.max_reference_error<1e-10 && item.max_wind_error<1e-10 && item.max_effectiveness_error==0 && ...
        item.quaternion_norm_error<1e-12 && item.commands_bounded && item.integrals_bounded;
    if strcmp(names{i},'normal_hover')
        item.passed=item.passed && size(previous,1)==numel(r.time) && max(abs(r.state-previous(:,2:20)),[],'all')<1e-10;
    end
    if i==1, results=item; else, results(i)=item; end %#ok<AGROW>
    fprintf('PARITY %s passed=%d position=%.3g command=%.3g imu=%.3g\n',item.name,item.passed,...
        item.max_position_error_m,item.max_motor_command_error_rad_s,item.max_imu_error);
    save(fullfile(folder,'pv_validation_progress.mat'),'results');
end
report=struct('source_commit',loaded.baseline,'reference_mode',loaded.reference_mode,...
    'parameter_profile','frozen_pre_retuning_gains','matlab_version',version,'cases',results,'passed',all([results.passed]));
fid=fopen(fullfile(folder,'pv_simulink_validation.json'),'w'); cleaner=onCleanup(@()fclose(fid));
fprintf(fid,'%s',jsonencode(report,PrettyPrint=true));
assert(report.passed,'Cross-language checks failed; inspect output/validation/pv_simulink_validation.json.');
end
