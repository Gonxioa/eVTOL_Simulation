function collect_hex_dataset(plan_path)
%COLLECT_HEX_DATASET Paired native simulations; no flight/control retuning.
root=fileparts(mfilename('fullpath')); addpath(root,fullfile(root,'validation'));
plan=jsondecode(fileread(plan_path)); folder=fileparts(plan_path);
assert(strcmp(plan.profile,'simulink_observable_v1'));
for j=1:numel(plan.sources)
    assert(strcmp(hashfile(fullfile(fileparts(root),plan.sources(j).path)),plan.sources(j).sha256),'Source changed after plan creation.');
end
mdl='hex_evtol'; load_system(fullfile(root,[mdl '.slx']));
assert(strcmp(get_param(mdl,'Dirty'),'off'),'Use a clean dedicated MATLAB process.');
cleanup=onCleanup(@()close_system(mdl,0));
% Charts contain nontunable struct parameters: Fast Restart would ignore changes.
set_param(mdl,'FastRestart','off');
for k=1:numel(plan.runs)
    item=plan.runs(k); out=fullfile(folder,item.run_key); meta=fullfile(out,'run_metadata.json');
    if isfile(meta)
        old=jsondecode(fileread(meta));
        assert(strcmp(old.plan_signature,plan.plan_signature),'Incompatible collection checkpoint.');
        if strcmp(old.status,'completed') && isfile(fullfile(out,'signals.csv')) && ...
                strcmp(old.csv_sha256,hashfile(fullfile(out,'signals.csv'))) && ...
                strcmp(old.mat_sha256,hashfile(fullfile(out,'result.mat')))
            fprintf('COLLECT_REUSE %s\n',item.run_key); continue
        end
        error('Retained failed/incomplete run %s: use a new dataset directory to retry.',item.run_key);
    end
    if ~isfolder(out), mkdir(out); end
    P=hex_parameters(); P.task_family=plan.task_family;
    P.seed=item.seed; P.fault_motor=item.motor_simulink; P.fault_factor=item.factor;
    P.observed_feedback=true; P.noise_enabled=true; P.wind_enabled=true; P.feedback_delay=.03;
    P=hex_prepare_inputs(P);
    params=rmfield(P,'exogenous');
    record=struct('profile',plan.profile,'dataset_version',plan.dataset_version,...
        'plan_signature',plan.plan_signature,'run',item,'parameters',params,...
        'control_tuning_version',P.control_tuning_version,'fault_start_s',P.fault_start,...
        'fault_end_s','infinity','sample_dt_s',P.export_dt,'feedback_delay_s',P.feedback_delay,...
        'frame','ENU_ground_FLU_body_xyzw_body_to_ground','fault_tolerant_control',true,...
        'sources',plan.sources,'status','started','matlab_version',version,'started_utc',char(datetime('now','TimeZone','UTC')));
    write(meta,record);
    started=tic;
    try
        result=run_hex_simulink(Parameters=P,SaveResults=false);
        hex_export_result(result,out);
        expected=ones(size(result.effectiveness));
        if P.fault_motor>0
            active=result.time>=P.fault_start & result.time<P.fault_end;
            expected(active,P.fault_motor)=P.fault_factor;
        end
        assert(max(abs(result.effectiveness-expected),[],'all')<1e-12,'Logged injection differs from requested label.');
        assert(max(abs(result.wind-P.exogenous(1:numel(result.time),1:3)),[],'all')<1e-12,'Logged wind differs from requested condition.');
        record.flight_metrics=hex_metrics(result);
        record.termination=result.termination;
        record.last_time_s=result.time(end);
        record.status='completed';
        if ~strcmp(result.termination,'completed'), record.status='flight_failed'; end
        record.csv_sha256=hashfile(fullfile(out,'signals.csv'));
        record.mat_sha256=hashfile(fullfile(out,'result.mat'));
    catch failure
        record.status='execution_failed'; record.error=failure.message;
    end
    record.wall_time_s=toc(started); write(meta,record);
    fprintf('COLLECT %d/%d %s %s %.2fs\n',k,numel(plan.runs),item.run_key,record.status,record.wall_time_s);
end
set_param(mdl,'FastRestart','off');
disp('COLLECTION_FINISHED');
end

function value=hashfile(path)
fid=fopen(path,'rb'); assert(fid>=0); data=fread(fid,Inf,'*uint8'); fclose(fid);
md=java.security.MessageDigest.getInstance('SHA-256'); md.update(data);
value=lower(reshape(dec2hex(typecast(md.digest(),'uint8'),2)',1,[]));
end

function write(path,value)
fid=fopen([path '.tmp'],'w','n','UTF-8'); assert(fid>=0);
fprintf(fid,'%s',jsonencode(value,PrettyPrint=true)); fclose(fid);
movefile([path '.tmp'],path,'f');
end
