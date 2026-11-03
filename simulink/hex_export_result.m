function hex_export_result(result,folder)
%HEX_EXPORT_RESULT Export actual logged columns without fabricating targets.
if ~isfolder(folder), mkdir(folder); end
save(fullfile(folder,'result.mat'),'result','-v7');
data=[result.time,result.state,result.reference,result.motor_command,result.imu,result.observation,...
    result.wind,result.effectiveness,result.control_audit,result.feedback,result.reference_audit];
names=hex_signal_names();
names=strrep(names,'wrench_nominal_allocated','wrench_estimated_allocated');
names=strrep(names,'wrench_nominal_residual','wrench_estimated_residual');
names=[names(1:end-19),compose('estimated_effectiveness_%d',1:6),{'fault_severity'},names(end-18:end),...
    {'path_phase_s','path_progress_rate'},compose('nominal_reference_%d',1:6),{'governor_position_error_m','governor_tilt_deg'}];
writetable(array2table(data,'VariableNames',names),fullfile(folder,'signals.csv'));
end
