function [wind,loads]=wind_step(t,P)
%#codegen
k=min(size(P.exogenous,1),round(t/P.sensor_dt)+1);
wind=zeros(3,1);
if P.wind_enabled
    wind=P.exogenous(k,1:3)';
end
loads=[P.external_force;P.external_torque];
end
