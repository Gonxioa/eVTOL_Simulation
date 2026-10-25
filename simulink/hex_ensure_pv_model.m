function hex_ensure_pv_model()
%HEX_ENSURE_PV_MODEL Upgrade an already-open legacy model before script runs.
ws=get_param('hex_evtol','ModelWorkspace'); P=ws.getVariable('P');
if ~isfield(P,'reference_mode') || ~strcmp(P.reference_mode,'position_velocity_fixed_heading')
    update_hex_pv_model();
end
end
