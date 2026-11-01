function out=translate_text(value)
%TRANSLATE_TEXT Apply identifier replacements once, without cascades.
persistent pairs
if isempty(pairs)
    configuration=jsondecode(fileread(fullfile(fileparts(mfilename('fullpath')),'mapping.json')));
    pairs=[configuration.old_tokens(:),configuration.new_tokens(:)];
end
out=value;
for i=1:size(pairs,1), out=strrep(out,pairs{i,1},sprintf('__ID%04d__',i)); end
for i=1:size(pairs,1), out=strrep(out,sprintf('__ID%04d__',i),pairs{i,2}); end
out=strrep(out,'.recovery','.fault_response');
out=strrep(out,'output/recovery/','output/simulations/');
out=strrep(out,'output\recovery\','output\simulations\');
out=strrep(out,'recovery_','return_');
out=regexprep(out,'\<recovery\>','control');
out=strrep(out,'Recovery','Control');
end
