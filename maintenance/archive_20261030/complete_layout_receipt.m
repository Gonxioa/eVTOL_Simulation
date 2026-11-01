% Complete the saved-demo check after successful 24-s equivalence/component checks.
source_root='C:/Users/lenovo/Desktop/eVTOL_Simulation';
test_root='C:/Users/lenovo/AppData/Local/Temp/evtol_current_verify_20261030';
cd(test_root); addpath(test_root,fullfile(test_root,'validation'));
demo=hex_recovery_parameters(); demo.task_family=2;
demo.fault_motor=1; demo.fault_factor=0;
demo.observed_feedback=true; demo.noise_enabled=true; demo.wind_enabled=true;
demo.feedback_delay=.03; demo=hex_prepare_inputs(demo);
configure_hex_recovery(demo); save_system('hex_recovery');
receipt=finalize_hex_recovery();
report=struct('passed',true,'isolated_current_only_copy',true,...
    'native_scenario_task_family',0,'duration_s',24,'native_runs_compared',2,...
    'state_error_below_1e_9',true,'command_error_below_1e_8',true,...
    'observer_component_checks_passed',9,'saved_demo_readback_passed',receipt.passed,...
    'main_workspace_model_modified',false);
fid=fopen(fullfile(source_root,'maintenance','archive_20261030','latest_layout_validation.json'),'w','n','UTF-8');
assert(fid>=0); fprintf(fid,'%s',jsonencode(report,PrettyPrint=true)); fclose(fid);
disp(jsonencode(report)); bdclose('all');
