function [next,blocked,error_increment,applied_increment,clip_increment,gate]=hex_rate_integral(current,error,residual,c,damage,dt)
%HEX_RATE_INTEGRAL Conditional integration with bounded exit toward zero.
% residual is requested minus finally allocated moment, N*m. The gate product
% has units N*m*rad/s and retains the existing absolute 1e-8 threshold.
error_increment=c.rate_i.*error*dt;
blocked=residual.*error>c.integral_recovery.rate_gate_Nm_rad_s;
gate=blocked;
candidate=current+error_increment;
for axis=1:3
    if blocked(axis)
        if axis<=2 && c.integral_recovery.rate_unwinding && damage>.1
            candidate(axis)=min(max(candidate(axis),min(current(axis),0)),max(current(axis),0));
            blocked(axis)=candidate(axis)==current(axis);
        else
            candidate(axis)=current(axis);
        end
    end
end
if damage>.1, candidate(3)=0; end
applied_increment=candidate-current;
next=min(max(candidate,-c.rate_integral_limit_rad_s2),c.rate_integral_limit_rad_s2);
clip_increment=next-candidate;
end
