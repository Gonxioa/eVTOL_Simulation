function groups=hex_block_parameters(P)
%HEX_BLOCK_PARAMETERS Explicitly exclude truth from control blocks.
groups=hex_common_block_parameters(P);
groups.task_reference.control=P.control;
groups.task_reference.control_dt=P.control_dt;
for field={'tau','omega_max','observed_feedback','feedback_delay','sensor_dt'}
    groups.controller_step.(field{1})=P.(field{1});
end
assert(~isfield(groups.controller_step,'fault_factor') && ...
    ~isfield(groups.task_reference,'fault_motor'),'Fault oracle crossed block boundary.');
end
