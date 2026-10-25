function R=hex_rotation(q)
q=q/norm(q); v=q(1:3); w=q(4);
S=[0 -v(3) v(2);v(3) 0 -v(1);-v(2) v(1) 0];
R=eye(3)+2*w*S+2*S*S;
end
