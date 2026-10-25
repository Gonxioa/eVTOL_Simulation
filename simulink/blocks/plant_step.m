function next_state=plant_step(state,command,segments,wind,loads,P)
%#codegen
% RK4 substeps match the original Python event interval partition.
next_state=state;
for j=1:3
    length=segments(1,j);
    if length>0
        count=max(1,ceil(length/P.integration_dt-1e-12)); dt=length/count;
        eta=segments(2:7,j);
        for k=1:count
            s=next_state;
            k1=hex_derivative(s,command,eta,wind,loads,P);
            k2=hex_derivative(s+dt*k1/2,command,eta,wind,loads,P);
            k3=hex_derivative(s+dt*k2/2,command,eta,wind,loads,P);
            k4=hex_derivative(s+dt*k3,command,eta,wind,loads,P);
            next_state=s+dt*(k1+2*k2+2*k3+k4)/6;
            next_state(7:10)=next_state(7:10)/norm(next_state(7:10));
        end
    end
end
end
