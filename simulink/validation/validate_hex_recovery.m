function report=validate_hex_recovery(suite)
%VALIDATE_HEX_RECOVERY Actual SLX checks with predeclared recovery criteria.
% Strict zero loss is tested for all six motors; legacy 5% labels are unused.
root=fileparts(fileparts(mfilename('fullpath'))); addpath(root);
if nargin<1, suite='core'; end
P=hex_recovery_parameters(); P.observed_feedback=true;
folder=fullfile(root,'output','recovery_validation',P.control_tuning_version,suite);
if ~isfolder(folder), mkdir(folder); end
P.noise_enabled=true; P.wind_enabled=true; P.feedback_delay=.03;
plan=[];
if strcmp(suite,'core')
    for task=0:3
        for motor=1:6
            for eta=[.8 .2 0], plan(end+1,:)=[task motor eta 20261006 1]; end %#ok<AGROW>
        end
    end
elseif strcmp(suite,'seeds')
    for seed=[20261007 20261008]
        for motor=1:6, plan(end+1,:)=[2 motor 0 seed 1]; end %#ok<AGROW>
    end
elseif strcmp(suite,'ideal')
    for motor=1:6
        for eta=[.8 .2 0], plan(end+1,:)=[2 motor eta 20261006 0]; end %#ok<AGROW>
    end
elseif strcmp(suite,'healthy')
    for task=0:3, plan(end+1,:)=[task 0 1 20261006 0]; end %#ok<AGROW>
else, error('Unknown suite.');
end
results=struct([]); mdl='hex_recovery';
if bdIsLoaded(mdl)
    assert(~strcmp(get_param(mdl,'Dirty'),'on'),...
        'Use a separate MATLAB session for validation; the open recovery model has unsaved edits.');
end
load_system(fullfile(root,[mdl '.slx']));
cleanup=onCleanup(@()close_system(mdl,0));
for i=1:size(plan,1)
    Q=P; Q.task_family=plan(i,1); Q.fault_motor=plan(i,2); Q.fault_factor=plan(i,3);
    Q.seed=plan(i,4); Q.observed_feedback=logical(plan(i,5));
    Q.noise_enabled=Q.observed_feedback; Q.wind_enabled=Q.observed_feedback;
    Q=hex_prepare_inputs(Q);
    name=sprintf('%s_task%d_m%d_eta%02d_seed%d',suite,Q.task_family,Q.fault_motor,round(100*Q.fault_factor),Q.seed);
    result=run_hex_recovery(Parameters=Q,OutputName=name);
    item=hex_recovery_metrics(result); item.name=name;
    if isempty(results), results=item; else, results(end+1)=item; end %#ok<AGROW>
    fprintf('RECOVERY_ACCEPT %d/%d %s passed=%d stable=%d returned=%d late=%.4f peak=%.4f tilt=%.2f recover=%.2f\n',...
        i,size(plan,1),name,item.passed,item.stable,item.returned,item.late_tracking_rmse_m,...
        item.peak_tracking_error_m,item.peak_tilt_deg,item.recovery_time_s);
    report=struct('profile',P.control_tuning_version,'suite',suite,...
        'criteria',struct('duration_s',24,'maximum_tilt_deg',35,...
        'maximum_altitude_loss_m',1,'maximum_rp_rate_rad_s',5,'maximum_yaw_rate_rad_s',10,...
        'late_tracking_rmse_m',.10,'late_path_rmse_m',.10,'maximum_recovery_time_s',15,...
        'recovery_position_band_m',.15,'recovery_tilt_band_deg',15,'recovery_rp_rate_band_rad_s',1,...
        'mission_completion_required',false),...
        'parameters',P.control,'cases',results,'passed',all([results.passed]));
    fid=fopen(fullfile(folder,'results.json'),'w'); fprintf(fid,'%s',jsonencode(report,PrettyPrint=true)); fclose(fid);
end
assert(report.passed,'Recovery acceptance failed. Inspect retained cases.');
copyfile(fullfile(folder,'results.json'),fullfile(root,'validation',['RECOVERY_' upper(suite) '_RESULTS.json']));
end
