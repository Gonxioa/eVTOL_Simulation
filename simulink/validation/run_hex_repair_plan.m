function run_hex_repair_plan(plan_path,first,last)
%RUN_HEX_REPAIR_PLAN Resumable native SLX evidence, never a surrogate solver.
root=fileparts(fileparts(mfilename('fullpath')));addpath(root);
plan=jsondecode(fileread(plan_path));folder=fileparts(plan_path);
if nargin<2,first=1;end
if nargin<3,last=numel(plan.cases);end
worker=sprintf('worker_%d',first);
Simulink.fileGenControl('set','CacheFolder',fullfile(folder,worker,'cache'),...
    'CodeGenFolder',fullfile(folder,worker,'codegen'),'createDir',true);
hex_ensure_model();identity=hex_check_loaded_model('hex_evtol');
if isfield(plan,'source_signature') && ~isempty(plan.source_signature)
    assert(strcmp(identity.source_signature,plan.source_signature),'hex:FrozenSourceChanged','Frozen model sources changed.');
end
for k=first:last
    if iscell(plan.cases),item=plan.cases{k};else,item=plan.cases(k);end
    destination=fullfile(folder,item.name);receipt=fullfile(destination,'run.json');
    if isfile(receipt)
        old=jsondecode(fileread(receipt));
        assert(strcmp(old.plan_sha256,hex_hash_file(plan_path)),'hex:Checkpoint','Plan identity changed.');
        if strcmp(old.status,'completed') && strcmp(old.source_signature,identity.source_signature)
            assert(strcmp(old.csv_sha256,hex_hash_file(fullfile(destination,'signals.csv'))));
            assert(strcmp(old.mat_sha256,hex_hash_file(fullfile(destination,'result.mat'))));
            fprintf('REPAIR_REUSE %d %s\n',k,item.name);continue;
        end
        error('hex:RetainedFailure','Existing unsuccessful run retained: %s. Use a new result directory.',destination);
    end
    if ~isfolder(destination),mkdir(destination);end
    P=hex_parameters();P.task_family=item.task;P.fault_motor=item.motor;P.fault_factor=item.factor;
    P.seed=item.seed;P.noise_enabled=item.noise;P.wind_enabled=item.wind;
    P.observed_feedback=true;P.feedback_delay=.03;P.integration_dt=item.integration_dt;
    P.fault_start=item.fault_start;P.heading=item.heading;
    if isfield(item,'force'),P.external_force=item.force(:);end
    P=hex_prepare_inputs(P);
    if isfield(item,'mirror') && item.mirror>0
        phi=(item.mirror-1)*pi/3;
        S=eye(3)-2*[cos(phi);sin(phi);0]*[cos(phi),sin(phi),0];H=-S;
        positions=(S*P.rotor_positions')';permutation=zeros(1,6);
        for j=1:6,[~,permutation(j)]=min(vecnorm(P.rotor_positions-positions(j,:),2,2));end
        P.exogenous(:,1:3)=P.exogenous(:,1:3)*S';
        P.exogenous(:,4:6)=P.exogenous(:,4:6)*S';
        P.exogenous(:,7:9)=P.exogenous(:,7:9)*H';
        P.exogenous(:,10:12)=P.exogenous(:,10:12)*S';
        P.exogenous(:,13:15)=P.exogenous(:,13:15)*S';
        P.exogenous(:,16:18)=P.exogenous(:,16:18)*H';
        P.exogenous(:,19:24)=P.exogenous(:,18+permutation);
    end
    record=struct('case',item,'status','started','source_signature',identity.source_signature,...
        'plan_sha256',hex_hash_file(plan_path));write(receipt,record);started=tic;
    try
        result=run_hex_simulink(Parameters=P,SaveResults=false);
        assert(strcmp(result.provenance.source_signature,identity.source_signature),'hex:FrozenSourceChanged','Source identity changed during the batch.');
        hex_export_result(result,destination);
        record.metrics=hex_metrics(result);record.status='completed';
        record.csv_sha256=hex_hash_file(fullfile(destination,'signals.csv'));
        record.mat_sha256=hex_hash_file(fullfile(destination,'result.mat'));
        fprintf('REPAIR_VALIDATION %d/%d %s pass=%d recovery=%.2f late=%.5f\n',...
            k,numel(plan.cases),item.name,record.metrics.passed,record.metrics.return_time_s,record.metrics.late_tracking_rmse_m);
    catch failure
        record.status='execution_failed';record.error=getReport(failure,'extended','hyperlinks','off');
        fprintf('REPAIR_FAILURE %d %s %s\n',k,item.name,failure.message);
    end
    record.wall_time_s=toc(started);write(receipt,record);
end
disp('REPAIR_PLAN_FINISHED');
end

function write(path,value)
fid=fopen([path '.tmp'],'w','n','UTF-8');assert(fid>=0);
fprintf(fid,'%s',jsonencode(value,PrettyPrint=true));fclose(fid);movefile([path '.tmp'],path,'f');
end
