function develop_hex_repair(plan_path)
%DEVELOP_HEX_REPAIR Paired native ablations before controller selection.
plan=jsondecode(fileread(plan_path)); root=fileparts(fileparts(mfilename('fullpath')));
addpath(root); folder=fileparts(plan_path);
Simulink.fileGenControl('set','CacheFolder',fullfile(folder,'cache'),...
    'CodeGenFolder',fullfile(folder,'codegen'),'createDir',true);
build_hex_model(); results=cell(numel(plan.cases),1);
for i=1:numel(plan.cases)
    if iscell(plan.cases),item=plan.cases{i};else,item=plan.cases(i);end
    destination=fullfile(folder,item.name);
    if ~isfolder(destination),mkdir(destination);end
    if isfield(item,'baseline_mat') && ~isempty(item.baseline_mat)
        saved=load(item.baseline_mat,'result'); P=saved.result.parameters;
        defaults=hex_parameters(); P.control.integral_recovery=defaults.control.integral_recovery;
    else
        P=hex_parameters(); P.task_family=item.task; P.fault_motor=item.motor;
        P.fault_factor=item.factor; P.wind_enabled=item.wind; P.noise_enabled=item.noise;
        P.observed_feedback=true; P.feedback_delay=.03;
    end
    P.control.integral_recovery.rate_unwinding=item.rate;
    P.control.integral_recovery.velocity_schedule=item.velocity;
    P.control.integral_recovery.degraded_velocity_i=[item.ki;item.ki;.9];
    P.control_tuning_version='control_recovery_v4_20261122';
    if isfield(item,'force'),P.external_force=item.force(:);end
    if isfield(item,'mean_wind') && ~isempty(item.mean_wind)
        P.wind.mean_m_s=item.mean_wind(:);P.wind.ou_stationary_std_m_s=0;
        P.wind_enabled=true;
    end
    P=hex_prepare_inputs(P); started=tic;record=struct('case',item,'status','started');
    try
        result=run_hex_simulink(Parameters=P,SaveResults=false);
        save(fullfile(destination,'result.mat'),'result','-v7');
        record.metrics=hex_metrics(result);record.status='completed';
        fprintf('REPAIR_DEV %d/%d %s pass=%d recover=%.2f late=%.5f\n',...
            i,numel(plan.cases),item.name,record.metrics.passed,record.metrics.return_time_s,record.metrics.late_tracking_rmse_m);
    catch failure
        record.status='execution_failed';record.error=getReport(failure,'extended','hyperlinks','off');
        fprintf('REPAIR_DEV_FAILURE %s %s\n',item.name,failure.message);
    end
    record.wall_time_s=toc(started);results{i}=record;
    write_json(fullfile(destination,'run.json'),record);
    write_json(fullfile(folder,'results.json'),struct('items',{results(1:i)}));
end
disp('REPAIR_DEV_FINISHED');
end

function write_json(path,value)
fid=fopen(path,'w','n','UTF-8');assert(fid>=0);cleanup=onCleanup(@()fclose(fid));
fprintf(fid,'%s',jsonencode(value,PrettyPrint=true));
end
