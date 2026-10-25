function ds=hex_derivative(s,command,eta,wind,loads,P)
% Loss acts ONCE on the motor target; actual thrust uses mechanical speed.
R=hex_rotation(s(7:10)); w=s(11:13); omega=s(14:19);
relative=R'*(s(4:6)-wind); force=zeros(3,1); moment=zeros(3,1);
for i=1:6
    r=P.rotor_positions(i,:)';
    local=relative+cross(w,r);
    drag=-omega(i)*P.rotor_drag.*local;
    thrust=[0;0;P.kT(i)*omega(i)^2];
    force=force+thrust+drag;
    moment=moment+cross(r,thrust+drag)+[0;0;P.yaw_sign(i)*P.kQ(i)*omega(i)^2];
end
force=force-norm(relative)*P.body_drag.*relative+loads(1:3);
acceleration=[0;0;-P.gravity]+R*force/P.mass;
alpha=P.invJ*(moment+loads(4:6)-cross(w,P.J*w));
ds=[s(4:6);acceleration;.5*hex_multiply(s(7:10),[w;0]);alpha;(sqrt(eta).*command-omega)./P.tau];
end
