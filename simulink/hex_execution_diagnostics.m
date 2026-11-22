function diagnostics=hex_execution_diagnostics(result)
%HEX_EXECUTION_DIAGNOSTICS Named controller signals and mechanical execution.
[names,units,frames]=hex_diagnostic_names();
data=result.controller_diagnostic;
assert(size(data,2)==numel(names),'hex:Diagnostics','Controller diagnostic schema differs.');
P=result.parameters;n=numel(result.time);actual=zeros(n,28);
for i=1:n
    s=result.state(i,:)';R=local_rotation(s(7:10));w=s(11:13);omega=s(14:19);
    relative=R'*(s(4:6)-result.wind(i,:)');
    thrust=P.kT.*omega.^2;mechanical=P.A*thrust;
    drag_force=zeros(3,1);drag_moment=zeros(3,1);
    for motor=1:6
        arm=P.rotor_positions(motor,:)';local=relative+cross(w,arm);
        drag=-omega(motor)*P.rotor_drag.*local;
        drag_force=drag_force+drag;drag_moment=drag_moment+cross(arm,drag);
    end
    drag_force=drag_force-norm(relative)*P.body_drag.*relative;
    force_body=[0;0;sum(thrust)]+drag_force+P.external_force;
    force_ground=R*force_body;
    moment=mechanical(2:4)+drag_moment+P.external_torque;
    acceleration=force_ground/P.mass-[0;0;P.gravity];
    alpha=P.invJ*(moment-cross(w,P.J*w));
    desired=local_rotation(result.control_audit(i,28:31)');
    pointing=acosd(min(max(desired(:,3)'*R(:,3),-1),1));
    actual(i,:)=[acceleration;force_ground;moment;mechanical(2:4);drag_moment;alpha;...
        pointing;R(:,3);desired(:,3);mechanical(1);norm(s(11:12));norm(s(1:3)-result.reference(i,1:3)')]';
end
extra={'actual_acceleration_ground','actual_force_ground','actual_moment_body',...
    'mechanical_moment_body','aerodynamic_moment_body','actual_angular_acceleration_body'};
extra_units={'m/s^2','N','N*m','N*m','N*m','rad/s^2'};
for i=1:numel(extra)
    for axis=1:3
        names{end+1}=sprintf('diag_%s_%s',extra{i},char('x'+axis-1)); %#ok<AGROW>
        units{end+1}=extra_units{i}; %#ok<AGROW>
        if i<=2,frames{end+1}='ground';else,frames{end+1}='body';end %#ok<AGROW>
    end
end
names=[names,{'diag_thrust_direction_error_deg'},...
    {'diag_actual_thrust_direction_x','diag_actual_thrust_direction_y','diag_actual_thrust_direction_z'},...
    {'diag_desired_thrust_direction_x','diag_desired_thrust_direction_y','diag_desired_thrust_direction_z'},...
    {'diag_mechanical_total_thrust_N','diag_rp_rate_norm_rad_s','diag_position_error_norm_m'}];
units=[units,{'deg','1','1','1','1','1','1','N','rad/s','m'}];
frames=[frames,repmat({'ground'},1,7),{'body','body','ground'}];
% 18 vector entries + direction angle + two direction vectors + three scalars.
assert(size(actual,2)==28 && numel(names)==size(data,2)+28);
diagnostics=struct('data',[data,actual],'names',{names},'units',{units},'frames',{frames});
end

function R=local_rotation(q)
q=q/norm(q);v=q(1:3);w=q(4);
S=[0,-v(3),v(2);v(3),0,-v(1);-v(2),v(1),0];
R=eye(3)+2*w*S+2*S*S;
end
