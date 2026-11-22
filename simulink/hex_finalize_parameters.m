function P=hex_finalize_parameters(P)
%HEX_FINALIZE_PARAMETERS Adapt historical parameter structs for current code.
defaults=hex_parameters();
if ~isfield(P.control,'integral_recovery')
    P.control.integral_recovery=defaults.control.integral_recovery;
end
P.control_tuning_version=defaults.control_tuning_version;
hex_validate_parameters(P);
end
