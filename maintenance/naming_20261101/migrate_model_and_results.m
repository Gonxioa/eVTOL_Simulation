root='C:/Users/lenovo/Desktop/eVTOL_Simulation';
source=fullfile(root,'simulink');
cd(source); addpath(source,fileparts(mfilename('fullpath')));
model='hex_evtol'; load_system(fullfile(source,[model '.slx']));
workspace=get_param(model,'ModelWorkspace');
P=translate_value(workspace.getVariable('P'));
workspace.assignin('P',P);
groups=hex_block_parameters(P); fields=fieldnames(groups);
for i=1:numel(fields), workspace.assignin(['P_' fields{i}],groups.(fields{i})); end
charts=find_system(model,'LookUnderMasks','all','FollowLinks','on','SFBlockType','MATLAB Function');
for i=1:numel(charts)
    cfg=get_param(charts{i},'MATLABFunctionConfiguration');
    old=char(cfg.FunctionScript); updated=translate_text(old);
    if ~strcmp(old,updated), cfg.FunctionScript=updated; end
end
handles=find_system(model,'FindAll','on','LookUnderMasks','all','FollowLinks','on','Type','Block');
for i=1:numel(handles)
    block=handles(i);
    for property={'VariableName','GotoTag','Description','Name'}
        try
            old=get_param(block,property{1});
        catch
            continue
        end
        if ischar(old)
            updated=translate_text(old);
            if ~strcmp(old,updated), set_param(block,property{1},updated); end
        end
    end
end
set_param(model,'Description',translate_text(get_param(model,'Description')));
set_param(model,'SimulationCommand','update');
save_system(model,fullfile(source,[model '.slx']));
close_system(model,0);
fprintf('NORMAL_MODEL_SAVED %s\n',model);

% Migrate MAT schema and metadata while checking every original numeric array.
folders={fullfile(source,'output','simulations'),fullfile(source,'output','validation'),...
    fullfile(source,'output','loop_tests')};
count=0;
for folder=folders
    files=dir(fullfile(folder{1},'**','*.mat'));
    for i=1:numel(files)
        path=fullfile(files(i).folder,files(i).name);
        original=load(path); migrated=translate_value(original);
        assert(numeric_equal(original,migrated),'Numeric data changed during metadata migration.');
        if ~isequaln(original,migrated)
            save(path,'-struct','migrated','-v7');
            readback=load(path); assert(isequaln(migrated,readback),'MAT readback mismatch.');
        end
        count=count+1;
    end
end
report=struct('saved_model',model,'mat_files_checked',count,'numeric_arrays_preserved',true);
fid=fopen(fullfile(root,'maintenance','naming_20261101','data_migration.json'),'w');
fprintf(fid,'%s',jsonencode(report,PrettyPrint=true)); fclose(fid);
fprintf('NORMAL_MAT_METADATA_MIGRATED files=%d\n',count);

function same=numeric_equal(before,after)
same=true;
if isstruct(before)
    names=fieldnames(before);
    if ~isequal(size(before),size(after)), same=false; return; end
    for f=1:numel(names)
        probe=struct(); probe.(names{f})=[]; renamed=translate_value(probe); new=fieldnames(renamed);
        for k=1:numel(before)
            if ~numeric_equal(before(k).(names{f}),after(k).(new{1})), same=false; return; end
        end
    end
elseif iscell(before)
    for k=1:numel(before)
        if ~numeric_equal(before{k},after{k}), same=false; return; end
    end
elseif isnumeric(before) || islogical(before)
    same=isequaln(before,after);
end
end
