function identity=hex_check_loaded_model(model)
%HEX_CHECK_LOADED_MODEL Reject stale or edited embedded functions before sim.
root=fileparts(mfilename('fullpath'));
workspace=get_param(model,'ModelWorkspace');
assert(workspace.hasVariable('HEX_BUILD_IDENTITY'),'hex:StaleModel',...
    'The model has no verified build identity. Run build_hex_model in %s.',root);
identity=workspace.getVariable('HEX_BUILD_IDENTITY');current=hex_build_sources();
assert(strcmp(identity.source_signature,current.source_signature),'hex:StaleModel',...
    'Model dependencies changed. Run build_hex_model in an isolated MATLAB session.');
expected=hex_embedded_scripts();
blocks=find_system(model,'LookUnderMasks','all','FollowLinks','on','SFBlockType','MATLAB Function');
seen=false(numel(expected),1);
for i=1:numel(blocks)
    cfg=get_param(blocks{i},'MATLABFunctionConfiguration');script=char(cfg.FunctionScript);
    name=regexp(script,'^function[^\r\n]*=\s*(\w+)\s*\(','tokens','once');
    if isempty(name),continue;end
    index=find(strcmp({expected.function_name},name{1}));
    if isempty(index),continue;end
    assert(~seen(index) && strcmp(hex_sha256(script),expected(index).sha256),...
        'hex:EmbeddedMismatch','Embedded source differs in %s.',blocks{i});
    seen(index)=true;
end
assert(all(seen),'hex:EmbeddedMismatch','One or more required numerical blocks were not checked.');
end
