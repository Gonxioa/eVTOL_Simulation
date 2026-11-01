root='C:/Users/lenovo/Desktop/eVTOL_Simulation';
source=fullfile(root,'simulink');
work='C:/Users/lenovo/AppData/Local/Temp/evtol_naming_20261101';
cd(source); addpath(source,fullfile(source,'validation'),fileparts(mfilename('fullpath')));
load(fullfile(work,'before_results.mat'),'cases');
load_system(fullfile(source,'hex_evtol.slx'));
workspace=get_param('hex_evtol','ModelWorkspace'); saved_default=workspace.getVariable('P');
errors=zeros(2,2);
for index=1:2
    P=hex_parameters(); P.task_family=2; P.fault_motor=1; P.fault_factor=.2;
    if index==2, P.fault_motor=0; P.fault_factor=1; end
    P.observed_feedback=true; P.noise_enabled=true; P.wind_enabled=true; P.feedback_delay=.03;
    P=hex_prepare_inputs(P);
    old=translate_value(cases{index});
    assert(isequaln(P,old.parameters),'Factory parameters changed beyond naming.');
    if index==1
        result=run_hex_simulink(Parameters=P,SaveResults=true,OutputName="my_circle_run");
    else
        result=run_hex_simulink(Parameters=P,SaveResults=false);
    end
    fields={'time','state','reference','motor_command','imu','observation','wind',...
        'effectiveness','control_audit','feedback','reference_audit'};
    for field=fields
        before=old.(field{1}); after=result.(field{1});
        assert(isequal(size(before),size(after)));
        delta=max(abs(before-after),[],'all');
        assert(delta<1e-9,['Naming changed numeric signal: ' field{1}]);
        errors(index,1)=max(errors(index,1),delta);
    end
    assert(strcmp(result.termination,'completed') && result.time(end)==24);
    metrics=hex_metrics(result); assert(metrics.passed);
    if index==1
        plot_folder=fullfile(source,'output','simulations',P.control_tuning_version,'my_circle_run');
        plot_hex_results(result,plot_folder,1:6); close all;
        assert(numel(dir(fullfile(plot_folder,'*.png')))==8);
    end
end
components=validate_hex_components();
receipt=finalize_hex_delivery();
assert(components.passed && receipt.passed);
report=run_hex_healthy_assessment(OutputName="healthy_four_tasks_20261027",...
    ReuseSavedResults=true,ShowFigures=false);
assert(strcmp(report.model,'hex_evtol') && numel(report.cases)==4);
if bdIsLoaded('hex_evtol'), close_system('hex_evtol',0); end

% Rebuild from the renamed internal template in a separate directory.
test=fullfile(work,'after'); cd(test); addpath(test,fullfile(test,'validation'),'-begin');
clear build_hex_model configure_hex_simulink hex_parameters hex_base_parameters
clear hex_block_parameters hex_common_block_parameters finalize_hex_delivery
build_hex_model(); configure_hex_simulink(saved_default); save_system('hex_evtol');
rebuilt_receipt=finalize_hex_delivery(); assert(rebuilt_receipt.passed);
bdclose('all');
audit=struct('passed',true,'native_cases',2,'duration_s',24,'signal_max_errors',errors(:,1)',...
    'parameters_equal_after_renaming',true,'all_logged_signals_compared',true,...
    'eight_standard_plots_written',true,'cached_healthy_tasks_read',4,...
    'observer_checks',9,'model_rebuild_passed',true,'saved_source_readback_passed',true);
fid=fopen(fullfile(root,'maintenance','naming_20261101','verification.json'),'w','n','UTF-8');
fprintf(fid,'%s',jsonencode(audit,PrettyPrint=true)); fclose(fid);
disp(jsonencode(audit));
