function receipt=finalize_hex_delivery(plan_path,recompute)
%FINALIZE_HEX_DELIVERY Verify current native records rather than old receipts.
if nargin<1
    error('hex:CurrentPlanRequired','Supply the current formal plan.json. Historical v3 receipts are retained and cannot certify v4.');
end
if nargin<2,recompute=false;end
root=fileparts(fileparts(mfilename('fullpath')));addpath(root);
plan=jsondecode(fileread(plan_path));folder=fileparts(plan_path);
assert(strcmp(plan.profile,'hex_control_repair_native_v4'));
hex_ensure_model();identity=hex_check_loaded_model('hex_evtol');
assert(strcmp(identity.source_signature,plan.source_signature),'hex:FrozenSourceChanged','Model differs from frozen validation plan.');
count=0;
for i=1:numel(plan.cases)
    if iscell(plan.cases),item=plan.cases{i};else,item=plan.cases(i);end
    path=fullfile(folder,item.name);r=jsondecode(fileread(fullfile(path,'run.json')));
    assert(strcmp(r.status,'completed') && r.metrics.passed,'hex:EvidenceFailure','Incomplete or rejected native run %s.',item.name);
    assert(strcmp(r.source_signature,identity.source_signature) && strcmp(r.plan_sha256,hex_hash_file(plan_path)));
    assert(strcmp(r.mat_sha256,hex_hash_file(fullfile(path,'result.mat'))));
    assert(strcmp(r.csv_sha256,hex_hash_file(fullfile(path,'signals.csv'))));
    saved=load(fullfile(path,'result.mat'),'result');metrics=saved.result.metrics;
    if recompute,metrics=hex_metrics(saved.result);end
    assert(metrics.passed && abs(metrics.return_time_s-r.metrics.return_time_s)<1e-10);
    count=count+1;
end
receipt=struct('profile',plan.controller_version,'native_cases',count,'passed',true,...
    'source_signature',identity.source_signature,'model_sha256',hex_hash_file(fullfile(root,'hex_evtol.slx')),...
    'all_files_rehashed',true,'metrics_recomputed_from_mat',recompute,...
    'mat_stored_metrics_checked',true);
fid=fopen(fullfile(folder,'delivery_receipt.json'),'w','n','UTF-8');assert(fid>=0);
fprintf(fid,'%s',jsonencode(receipt,PrettyPrint=true));fclose(fid);
fprintf('SIMULATION_DELIVERY_PASSED native_cases=%d profile=%s\n',count,plan.controller_version);
end
