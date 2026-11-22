function [path,identity]=hex_ensure_model()
%HEX_ENSURE_MODEL All scripted entry points use this freshness check.
root=fileparts(mfilename('fullpath'));path=fullfile(root,'hex_evtol.slx');model='hex_evtol';
if bdIsLoaded(model)
    assert(strcmp(get_param(model,'SimulationStatus'),'stopped'),'hex:RunningModel','Stop the current simulation first.');
    assert(strcmp(get_param(model,'Dirty'),'off'),'hex:DirtyModel',...
        'The open model has unsaved edits. They have been retained. Run the simulation in a separate MATLAB session.');
    assert(strcmp(get_param(model,'FileName'),path),'hex:OtherCheckout',...
        'A model from another checkout is loaded. Use a separate MATLAB session for this checkout.');
end
if ~isfile(path),build_hex_model();end
load_system(path);
try
    identity=hex_check_loaded_model(model);
catch failure
    if ~ismember(failure.identifier,{'hex:StaleModel','hex:EmbeddedMismatch'}),rethrow(failure);end
    close_system(model,0);build_hex_model();load_system(path);
    identity=hex_check_loaded_model(model);
end
end
