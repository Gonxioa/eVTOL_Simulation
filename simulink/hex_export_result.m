function hex_export_result(result,folder)
%HEX_EXPORT_RESULT Export actual logged columns without fabricating targets.
if ~isfolder(folder), mkdir(folder); end
result.metrics=hex_metrics(result);
save(fullfile(folder,'result.mat'),'result','-v7');
data=[result.time,result.state,result.reference,result.motor_command,result.imu,result.observation,...
    result.wind,result.effectiveness,result.control_audit,result.feedback,result.reference_audit];
names=hex_signal_names();
names=strrep(names,'wrench_nominal_allocated','wrench_estimated_allocated');
names=strrep(names,'wrench_nominal_residual','wrench_estimated_residual');
names=[names(1:end-19),compose('estimated_effectiveness_%d',1:6),{'fault_severity'},names(end-18:end),...
    {'path_phase_s','path_progress_rate'},compose('nominal_reference_%d',1:6),{'governor_position_error_m','governor_tilt_deg'}];
legacy_names=names;
if isfield(result,'controller_diagnostic')
    diagnostics=hex_execution_diagnostics(result);
    data=[data,diagnostics.data]; names=[names,diagnostics.names];
end
assert(size(data,2)==numel(names),'hex:ExportSchema','Signal header does not match logged data.');
writetable(array2table(data,'VariableNames',names),fullfile(folder,'signals.csv'));
fields=repmat(struct('name','','unit','','frame','','use','diagnosis_only'),numel(names),1);
for i=1:numel(legacy_names)
    fields(i)=legacy_field(legacy_names{i});
end
if exist('diagnostics','var')
    for i=1:numel(diagnostics.names)
        k=numel(legacy_names)+i;fields(k).name=diagnostics.names{i};
        fields(k).unit=diagnostics.units{i};fields(k).frame=diagnostics.frames{i};
    end
end
metadata=struct('schema','hex_execution_v4','controller_version',result.parameters.control_tuning_version,...
    'parameters',rmfield(result.parameters,'exogenous'),'exogenous_in_mat','result.parameters.exogenous',...
    'exogenous_sha256',hex_sha256(typecast(result.parameters.exogenous(:),'uint8')),...
    'termination',result.termination,'metrics',result.metrics,'fields',fields,...
    'csv_sha256',hex_hash_file(fullfile(folder,'signals.csv')),...
    'mat_sha256',hex_hash_file(fullfile(folder,'result.mat')));
if isfield(result,'provenance'),metadata.provenance=result.provenance;end
fid=fopen(fullfile(folder,'execution_metadata.json'),'w','n','UTF-8');assert(fid>=0);
fprintf(fid,'%s',jsonencode(metadata,PrettyPrint=true));fclose(fid);
end

function f=legacy_field(name)
unit='1';frame='none';usage='diagnosis_only';
if contains(name,{'quaternion','qx','qy','qz','qw'}),frame='body_to_ground';
elseif contains(name,{'rad_s2'}),unit='rad/s^2';frame='body';
elseif contains(name,{'rad_s'}),unit='rad/s';frame='body';
elseif contains(name,{'m_s2'}),unit='m/s^2';frame='ground';
elseif contains(name,{'m_s'}),unit='m/s';frame='ground';
elseif endsWith(name,'_m') || contains(name,'position_m') || contains(name,'error_m'),unit='m';frame='ground';
elseif contains(name,{'time_s','phase_s'}),unit='s';end
if startsWith(name,'wrench_')
    if endsWith(name,'_1'),unit='N';else,unit='N*m';end;frame='body';
end
if startsWith(name,{'accelerometer_sensor','gyro_sensor'}),frame='sensor';usage='measured_input_raw';end
if startsWith(name,'feedback_state_') || startsWith(name,'observation_state_')
    k=str2double(regexp(name,'\d+$','match','once'));
    if k<=3,unit='m';frame='ground';elseif k<=6,unit='m/s';frame='ground';...
    elseif k<=10,unit='1';frame='body_to_ground';else,unit='rad/s';frame='body';end
    if k>=14 && startsWith(name,'feedback_state_'),usage='measured_input_already_delayed';end
end
if contains(name,'tilt_deg'),unit='deg';frame='body_to_ground';end
f=struct('name',name,'unit',unit,'frame',frame,'use',usage);
end
