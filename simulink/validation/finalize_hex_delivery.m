function finalize_hex_delivery(component_report)
%FINALIZE_HEX_DELIVERY Save checked evidence, diagram and a complete example.
root=fileparts(fileparts(mfilename('fullpath'))); mdl='hex_evtol';
folder=fullfile(root,'output','validation');
report=jsondecode(fileread(fullfile(folder,'pv_simulink_validation.json')));
assert(report.passed && component_report.passed,'Validation must pass before finalizing.');
copyfile(fullfile(folder,'pv_simulink_validation.json'),fullfile(root,'validation','PV_RESULTS.json'));
fid=fopen(fullfile(root,'validation','PV_COMPONENT_RESULTS.json'),'w');
fprintf(fid,'%s',jsonencode(component_report,PrettyPrint=true)); fclose(fid);
update_hex_pv_model();
print(['-s' mdl],'-dpng','-r120',fullfile(root,'preview.png'));
% Copy only the SLX, then remove all project source directories from the path.
portable=fullfile(root,'output','portable_check');
if ~isfolder(portable), mkdir(portable); end
close_system(mdl,0);
copyfile(fullfile(root,[mdl '.slx']),fullfile(portable,[mdl '.slx']));
oldfolder=pwd; oldpath=path; cleanup=onCleanup(@()restore(oldfolder,oldpath));
cd(portable);
for dir={root,fullfile(root,'blocks'),fullfile(root,'functions'),fullfile(root,'validation')}
    if contains([path pathsep],[dir{1} pathsep]), rmpath(dir{1}); end
end
load_system(fullfile(portable,[mdl '.slx']));
standalone=sim(mdl,'StopTime','.02');
ts=standalone.state_log;
if ts.IsTimeFirst, data=reshape(ts.Data,numel(ts.Time),[]); else, data=reshape(ts.Data,[],numel(ts.Time))'; end
assert(norm(data(end,1:3)-[0 0 2.5])<1e-10,'The standalone SLX did not maintain hover.');
close_system(mdl,0); cd(oldfolder); path(oldpath); clear cleanup
fid=fopen(fullfile(root,'validation','PV_PORTABILITY.json'),'w');
fprintf(fid,'%s',jsonencode(struct('passed',true,'files_copied',{{'hex_evtol.slx'}},'project_paths_removed',true,'python_runtime_used',false,'simulated_seconds',ts.Time(end)),PrettyPrint=true)); fclose(fid);
% Save a real fault response with named CSV columns and an editable figure.
loaded=load(fullfile(folder,'python_pv_reference.mat')); test=loaded.cases.partial_rotor_1;
P=hex_parameters();
settings={'task_family','fault_motor','fault_factor','fault_start','fault_end','duration',...
    'observed_feedback','noise_enabled','wind_enabled','feedback_delay','seed','motion_start','motion_end','exogenous'};
for j=1:numel(settings), P.(settings{j})=test.(settings{j}); end
P.observed_feedback=logical(P.observed_feedback); P.noise_enabled=logical(P.noise_enabled); P.wind_enabled=logical(P.wind_enabled);
set(groot,'DefaultFigureVisible','off');
example=run_hex_simulink(Parameters=P,OutputName="pv_partial_rotor_1");
assert(norm(example.state(end,1:3)-test.data(end,2:4))<1e-3);
csv=readtable(fullfile(root,'output','pv_partial_rotor_1','signals.csv'));
assert(width(csv)==128 && height(csv)==numel(example.time),'Exported CSV shape is incorrect.');
assert(norm(csv{end,{'east_m','north_m','up_m'}}-test.data(end,2:4))<1e-3,'Exported CSV state does not match the reference.');
copyfile(fullfile(root,'output','pv_partial_rotor_1','results.png'),fullfile(root,'validation','example_pv_partial_rotor.png'));
fprintf('DELIVERY_FINALIZED %s\n',fullfile(root,[mdl '.slx']));
end

function restore(folder,original_path)
cd(folder); path(original_path);
end
