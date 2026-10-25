function [thrust,residual,flags]=hex_allocate(desired,P)
candidate=P.pinvA*desired;
if P.regularization==0 && all(candidate>=P.thrust_min-1e-8) && all(candidate<=P.thrust_max+1e-8) && norm(P.row_weights.*(desired-P.A*candidate))<=1e-10
    thrust=min(max(candidate,P.thrust_min),P.thrust_max); success=true;
else
    weights=diag(P.row_weights);
    C=[weights*P.A;sqrt(P.regularization)*diag(1./max(P.thrust_max-P.thrust_min,1e-12))];
    d=[weights*desired;zeros(6,1)];
    [thrust,exitflag]=hex_trf(C,d,P.thrust_min,P.thrust_max);
    thrust=min(max(thrust,P.thrust_min),P.thrust_max);
    % A rank-deficient six-actuator problem can return a nonpositive flag
    % after reaching its optimum. Independently verify convex box KKT terms.
    gradient=C'*(C*thrust-d);
    projected=gradient;
    at_lower=thrust<=P.thrust_min+1e-8;
    at_upper=thrust>=P.thrust_max-1e-8;
    projected(at_lower)=min(gradient(at_lower),0);
    projected(at_upper)=max(gradient(at_upper),0);
    success=all(isfinite(thrust)) && (exitflag>0 || norm(projected,Inf)<=1e-10);
    if ~success
        fprintf('ALLOCATION_FAILURE flag=%g kkt=%.17g desired=[%.17g %.17g %.17g %.17g]\n',exitflag,norm(projected,Inf),desired(1),desired(2),desired(3),desired(4));
    end
end
residual=desired-P.A*thrust;
boundtol=max(1e-8,1e-7*max(1,P.thrust_max-P.thrust_min));
active=(abs(thrust-P.thrust_min)<=boundtol)|(abs(thrust-P.thrust_max)<=boundtol);
flags=[double(success);sum(active);double(max(abs(residual./P.wrench_scale))<=1e-6)];
assert(success,'Nominal allocation failed to converge.');
end
