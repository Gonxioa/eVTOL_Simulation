function stop=envelope_step(state,P)
%#codegen
b=P.termination;
stop=any(~isfinite(state)) || norm(state(1:3))>b.position_norm_m || ...
    norm(state(4:6))>b.speed_m_s || norm(state(11:13))>b.body_rate_rad_s || state(3)<b.minimum_altitude_m;
end
