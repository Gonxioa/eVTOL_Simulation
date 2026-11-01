work='C:/Users/lenovo/AppData/Local/Temp/evtol_naming_20261101';
cd(fullfile(work,'before')); addpath(pwd);
P=hex_recovery_parameters(); P.task_family=2; P.fault_motor=1;
P.observed_feedback=true; P.noise_enabled=true; P.wind_enabled=true; P.feedback_delay=.03;
P=hex_prepare_inputs(P);
cases=cell(1,2);
for index=1:2
    if index==1, P.fault_factor=.2; else, P.fault_motor=0; P.fault_factor=1; end
    cases{index}=run_hex_recovery(Parameters=P,SaveResults=false);
    assert(cases{index}.time(end)==24 && strcmp(cases{index}.termination,'completed'));
end
save(fullfile(work,'before_results.mat'),'cases','-v7');
bdclose('all');
fprintf('NAMING_BEFORE_SAVED cases=%d\n',numel(cases));
