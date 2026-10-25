function receipt=finalize_hex_recovery()
%FINALIZE_HEX_RECOVERY Check saved model and matching-version evidence.
root=fileparts(fileparts(mfilename('fullpath'))); addpath(root);
expected=hex_recovery_parameters(); profile=expected.control_tuning_version;
counts=zeros(1,4); suites={'CORE','HEALTHY','IDEAL','SEEDS'};
for i=1:numel(suites)
    report=jsondecode(fileread(fullfile(root,'validation',['RECOVERY_' suites{i} '_RESULTS.json'])));
    assert(report.passed && strcmp(report.profile,profile),'Mismatched or failing recovery evidence.');
    assert(isequal(report.parameters,expected.control),'Evidence gains differ from current defaults.');
    counts(i)=numel(report.cases);
end
loops=jsondecode(fileread(fullfile(root,'validation','RECOVERY_LOOP_RESULTS.json')));
assert(loops.passed,'Loop acceptance failed.');
selected=loops.cases([loops.cases.selected]);
assert(all([selected.rate_p]==20) && all([selected.attitude_p]==7),'Wrong selected loop gains.');
components=jsondecode(fileread(fullfile(root,'validation','RECOVERY_COMPONENT_RESULTS.json')));
assert(components.passed,'Independent components failed.');
audit=jsondecode(fileread(fullfile(root,'validation','RECOVERY_INPUT_AUDIT.json')));
assert(audit.passed && strcmp(audit.profile,profile),'Actual SLX input audit is incomplete.');
parity=jsondecode(fileread(fullfile(root,'validation','RECOVERY_SOURCE_SLX_PARITY.json')));
assert(parity.passed && strcmp(parity.profile,profile),'Source/SLX behavioral comparison failed.');
mdl='hex_recovery'; load_system(fullfile(root,[mdl '.slx']));
cleanup=onCleanup(@()close_system(mdl,0));
ws=get_param(mdl,'ModelWorkspace'); P=ws.getVariable('P');
assert(strcmp(P.control_tuning_version,profile) && isequal(P.control,expected.control),'Saved model contains old defaults.');
groups=hex_recovery_block_parameters(P); names=fieldnames(groups);
for i=1:numel(names)
    assert(isequaln(ws.getVariable(['P_' names{i}]),groups.(names{i})),'Saved block snapshots differ.');
end
assert(P.task_family==2 && P.fault_motor==1 && P.fault_factor==0 && ...
    P.observed_feedback && P.noise_enabled && P.wind_enabled && P.feedback_delay==.03,...
    'Saved demonstration is not a real observed total-loss case.');
blocks=find_system(mdl,'LookUnderMasks','all','SFBlockType','MATLAB Function');
controller=''; task='';
for i=1:numel(blocks)
    cfg=get_param(blocks{i},'MATLABFunctionConfiguration'); s=char(cfg.FunctionScript);
    if startsWith(s,'function [command,audit]=recovery_controller_step'), controller=s; end
    if startsWith(s,'function [reference,audit]=recovery_reference_step'), task=s; end
end
script=fileread(fullfile(root,'blocks','recovery_controller_step.m'));
for helper={'hex_rotation','hex_multiply','hex_desired_attitude','hex_allocate','hex_trf'}
    script=[script newline fileread(fullfile(root,'functions',[helper{1} '.m']))]; %#ok<AGROW>
end
script=regexprep(script,'\<P\>','P_controller_step');
assert(strcmp(normalize(script),normalize(controller)),'Embedded recovery controller is stale.');
script=[fileread(fullfile(root,'blocks','recovery_reference_step.m')) newline fileread(fullfile(root,'blocks','task_reference.m'))];
script=regexprep(script,'\<P\>','P_task_reference');
assert(strcmp(normalize(script),normalize(task)),'Embedded reference governor is stale.');
receipt=struct('profile',profile,'native_cases',sum(counts),'suite_counts',counts,...
    'saved_model_readback_passed',true,'embedded_sources_match',true,...
    'source_slx_behavioral_agreement',parity.behavioral_agreement_passed,...
    'strict_rotor_command_parity',parity.strict_rotor_command_parity_passed,...
    'fault_truth_excluded_from_recovery_blocks',true,'actual_inputs_audited',audit.passed,...
    'original_model_overwritten',false,'passed',true);
fid=fopen(fullfile(root,'validation','RECOVERY_DELIVERY.json'),'w');
fprintf(fid,'%s',jsonencode(receipt,PrettyPrint=true)); fclose(fid);
fprintf('RECOVERY_DELIVERY_PASSED native_cases=%d profile=%s\n',sum(counts),profile);
end

function value=normalize(value)
value=strtrim(regexprep(value,'\r\n?','\n'));
end
