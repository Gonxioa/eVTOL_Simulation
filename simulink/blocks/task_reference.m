function reference=task_reference(t,P)
%#codegen
x=P.task_origin; v=zeros(3,1);
if P.task_family<=3
    u=0; du=0; T=P.motion_end-P.motion_start;
    if t>=P.motion_end
        u=1;
    elseif t>P.motion_start
        z=(t-P.motion_start)/T;
        u=10*z^3-15*z^4+6*z^5;
        du=30*z^2*(1-z)^2/T;
    end
    if P.task_family==1 || P.task_family==3
        if P.task_family==1
            direction=[cos(P.heading);sin(P.heading);0];
        else
            direction=[0;0;1];
        end
        delta=P.displacement*direction;
        x=x+delta*u; v=delta*du;
    elseif P.task_family==2
        angle=P.heading+P.turn*u; angular_rate=P.turn*du;
        radial=[cos(angle);sin(angle);0]; tangent=[-sin(angle);cos(angle);0];
        x=x+P.radius*(radial-[cos(P.heading);sin(P.heading);0]);
        v=P.radius*angular_rate*tangent;
    end
elseif P.task_family==4
    x=x+[.4*sin(.6*t);.4*(1-cos(.6*t));.15*sin(.4*t)];
    v=[.24*cos(.6*t);.24*sin(.6*t);.06*cos(.4*t)];
elseif P.task_family==5 && t>=2
    x=x+[.5;-.4;.3];
elseif P.task_family==6 && t>=2 && t<4
    x=x+[0;0;50];
end
reference=[x;v];
end
