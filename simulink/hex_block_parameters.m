function groups=hex_block_parameters(P)
%HEX_BLOCK_PARAMETERS Give each block only the parameters it actually uses.
plant={'rotor_positions','rotor_drag','kT','kQ','yaw_sign','body_drag','gravity','mass','invJ','J','tau'};
groups.task_reference=select({'task_origin','task_family','motion_start','motion_end','heading','displacement','radius','turn'});
groups.controller_step=select({'control','control_dt','mass','gravity','thrust_max','J','kT','A','pinvA','thrust_min','row_weights','wrench_scale','regularization'});
groups.fault_schedule=select({'base_dt','fault_start','fault_end','fault_motor','fault_factor'});
groups.plant_step=select([plant,{'integration_dt'}]);
groups.sensor_step=select([plant,{'sensor','sensor_dt','exogenous','noise_enabled','observed_feedback','feedback_delay'}]);
groups.wind_step=select({'sensor_dt','exogenous','wind_enabled','external_force','external_torque'});
groups.envelope_step=select({'termination'});
    function value=select(fields)
        value=struct();
        for i=1:numel(fields), value.(fields{i})=P.(fields{i}); end
    end
end
