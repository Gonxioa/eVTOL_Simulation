function out=translate_value(value)
%TRANSLATE_VALUE Rename metadata only; numeric arrays are returned unchanged.
if isstruct(value)
    fields=fieldnames(value); names=cell(size(fields)); out=value;
    for f=1:numel(fields)
        names{f}=field_name(fields{f});
        for index=1:numel(value)
            out(index).(names{f})=translate_value(value(index).(fields{f}));
        end
        if ~strcmp(names{f},fields{f}), out=rmfield(out,fields{f}); end
    end
    if ~isempty(fields), out=orderfields(out,names); end
elseif iscell(value)
    out=cell(size(value));
    for index=1:numel(value), out{index}=translate_value(value{index}); end
elseif ischar(value)
    out=translate_text(value);
elseif isstring(value)
    out=value;
    for index=1:numel(value), out(index)=string(translate_text(char(value(index)))); end
else
    out=value;
end
end

function name=field_name(name)
if strcmp(name,'recovery'), name='fault_response'; return; end
name=strrep(name,'recovery_reference_audit','reference_audit');
name=strrep(name,'recovery_damage','fault_severity');
name=strrep(name,'recovery_','return_');
name=strrep(name,'_recovery_blocks','_control_blocks');
end
