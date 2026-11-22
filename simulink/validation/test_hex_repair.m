function test_hex_repair(output)
%TEST_HEX_REPAIR Regression checks for the confirmed integration failure.
root=fileparts(fileparts(mfilename('fullpath')));addpath(root,fullfile(root,'functions'));
P=hex_parameters();c=P.control;dt=.01;res=[-1;1;0];
for sign=[-1 1]
    I=[sign*8;0;0];err=[-sign*.4;0;0];res(1)=-sign*1e-6;
    [next,blocked]=hex_rate_integral(I,err,res,c,1,dt);
    assert(abs(next(1))<abs(I(1)) && ~blocked(1));
    [next,~]=hex_rate_integral(I,-err,-res,c,1,dt);assert(next(1)==I(1));
    I(1)=sign*.01;[next,~]=hex_rate_integral(I,err,res,c,1,dt);assert(next(1)==0);
    [next,~]=hex_rate_integral(I,err,zeros(3,1),c,1,dt);assert(sign*next(1)<0);
end
[next,~]=hex_rate_integral([20;-20;2],[10;-10;10],zeros(3,1),c,0,dt);
assert(isequal(next,[20;-20;2]));
c.integral_recovery.rate_unwinding=false;
[next,blocked]=hex_rate_integral([8;0;0],[-.4;0;0],[-1e-6;0;0],c,1,dt);
assert(next(1)==8 && blocked(1));
% Coupled pitch/yaw request produces a tiny pitch residual that used to freeze.
c=P.control;[next,~]=hex_rate_integral([0;8.012049;0],[0;-.400599;0],[0;-2.713e-7;1],c,1,dt);
assert(next(2)<8.012049 && next(2)>0);
hex_ensure_model();model='hex_evtol';cfg=[];
blocks=find_system(model,'LookUnderMasks','all','SFBlockType','MATLAB Function');
for i=1:numel(blocks)
    a=get_param(blocks{i},'MATLABFunctionConfiguration');
    if contains(a.FunctionScript,'function [command,audit,diagnostic]'),cfg=a;break;end
end
original=cfg.FunctionScript;cfg.FunctionScript=[char(original) newline '% user edit preserved'];
try,hex_ensure_model();error('test:ExpectedFailure','Unsaved changes were not protected.');
catch failure,assert(strcmp(failure.identifier,'hex:DirtyModel'));end
assert(endsWith(strtrim(cfg.FunctionScript),'% user edit preserved'));
cfg.FunctionScript=original;set_param(model,'Dirty','off');
workspace=get_param(model,'ModelWorkspace');identity=workspace.getVariable('HEX_BUILD_IDENTITY');
stale=identity;stale.source_signature='stale';workspace.assignin('HEX_BUILD_IDENTITY',stale);
try,hex_check_loaded_model(model);error('test:ExpectedFailure','Stale model was accepted.');
catch failure,assert(strcmp(failure.identifier,'hex:StaleModel'));end
workspace.assignin('HEX_BUILD_IDENTITY',identity);set_param(model,'Dirty','off');
% A -> unrelated B -> A, including requested Fast Restart, must reset all state.
P.duration=1;P=hex_prepare_inputs(P);A=run_hex_simulink(Parameters=P,SaveResults=false);
Q=P;Q.task_family=2;Q.seed=43;Q=hex_prepare_inputs(Q);
run_hex_simulink(Parameters=Q,SaveResults=false);
set_param(model,'FastRestart','on');set_param(model,'Dirty','off');
B=run_hex_simulink(Parameters=P,SaveResults=false);
assert(isequal(A.state,B.state) && isequal(A.control_audit,B.control_audit) && isequal(A.feedback,B.feedback));
assert(strcmp(get_param(model,'FastRestart'),'off'));
% Recovery boundaries: final sample bad, truncated flight, no post-onset data.
S=A;S.parameters.duration=1;S.parameters.fault_start=.2;
S.state(:,1:3)=repmat(S.parameters.task_origin',numel(S.time),1);
S.reference(:,1:3)=S.state(:,1:3);S.state(:,7:10)=repmat([0 0 0 1],numel(S.time),1);S.state(:,11:13)=0;
S.state(end,1)=1;m=hex_metrics(S);assert(strcmp(m.recovery_status,'not_recovered') && isinf(m.return_time_s));
S.state(end,1)=0;S.state(end-1,1)=1;m=hex_metrics(S);
assert(abs(m.return_time_s-(S.time(end-1)-.2))<1e-12 && m.sustained_entry_time_s==S.time(end));
S.parameters.duration=2;m=hex_metrics(S);assert(strcmp(m.recovery_status,'incomplete'));
S.parameters.duration=1;S.parameters.fault_start=2;m=hex_metrics(S);assert(strcmp(m.recovery_status,'no_postfault_samples'));
hex_export_result(A,fullfile(output,'smoke_export'));
receipt=struct('passed',true,'integral_cases',11,'state_reset_max_difference',max(abs(A.state-B.state),[],'all'),...
    'source_signature',identity.source_signature,'model_sha256',hex_hash_file(fullfile(root,'hex_evtol.slx')));
fid=fopen(fullfile(output,'regression_tests.json'),'w');fprintf(fid,'%s',jsonencode(receipt,PrettyPrint=true));fclose(fid);
disp('HEX_REPAIR_REGRESSIONS_PASS');
end
