function [segments,effectiveness]=fault_schedule(t,P)
%#codegen
% Exact event boundaries, including times between sample ticks.
t=round(t*1e12)/1e12; finish=round((t+P.base_dt)*1e12)/1e12;
edges=[t;finish;finish;finish]; n=2;
if P.fault_start>t && P.fault_start<finish
    n=n+1; edges(n)=P.fault_start;
end
if P.fault_end>t && P.fault_end<finish
    n=n+1; edges(n)=P.fault_end;
end
edges=sort(edges); segments=zeros(7,3);
effectiveness=eta_at(t,P);
for i=1:3
    if edges(i+1)>edges(i)
        segments(:,i)=[edges(i+1)-edges(i);eta_at(edges(i),P)];
    end
end
end

function eta=eta_at(t,P)
eta=ones(6,1);
if P.fault_motor>=1 && t>=P.fault_start && t<P.fault_end
    eta(P.fault_motor)=P.fault_factor;
end
end
