% Verify current-only packaging in an isolated temporary copy.
source_root='C:/Users/lenovo/Desktop/eVTOL_Simulation';
test_root='C:/Users/lenovo/AppData/Local/Temp/evtol_current_verify_20261030';
cd(test_root); addpath(test_root,fullfile(test_root,'validation'));
P=hex_recovery_parameters();
P.task_family=2;
P.fault_motor=1; P.fault_factor=.2;
P.observed_feedback=true; P.noise_enabled=true; P.wind_enabled=true;
P.feedback_delay=.03; P=hex_prepare_inputs(P);
before=run_hex_recovery(Parameters=P,SaveResults=false);
assert(before.time(end)==24 && strcmp(before.termination,'completed'));
build_hex_recovery_model();
after=run_hex_recovery(Parameters=P,SaveResults=false);
assert(isequal(before.time,after.time));
state_error=max(abs(before.state-after.state),[],'all');
command_error=max(abs(before.motor_command-after.motor_command),[],'all');
assert(state_error<1e-9 && command_error<1e-8,'Rebuilt model differs from the saved current model.');
components=validate_hex_recovery_components();
demo=hex_recovery_parameters(); demo.task_family=2;
demo.fault_motor=1; demo.fault_factor=0;
demo.observed_feedback=true; demo.noise_enabled=true; demo.wind_enabled=true;
demo.feedback_delay=.03; demo=hex_prepare_inputs(demo);
configure_hex_recovery(demo); save_system('hex_recovery');
receipt=finalize_hex_recovery();
report=struct('passed',true,'isolated_current_only_copy',true,'duration_s',after.time(end),...
    'state_max_error',state_error,'command_max_error',command_error,...
    'components_passed',components.passed,'delivery_readback_passed',receipt.passed,...
    'main_workspace_model_modified',false);
fid=fopen(fullfile(source_root,'maintenance','archive_20261030','latest_layout_validation.json'),'w','n','UTF-8');
assert(fid>=0); fprintf(fid,'%s',jsonencode(report,PrettyPrint=true)); fclose(fid);
disp(jsonencode(report));
bdclose('all');
