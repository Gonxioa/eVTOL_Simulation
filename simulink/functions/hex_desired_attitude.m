function q=hex_desired_attitude(F,yaw,previous)
if norm(F)<1e-10
    q=previous/norm(previous); return
end
z=F/norm(F); y=cross(z,[cos(yaw);sin(yaw);0]);
if norm(y)<1e-10
    [~,i]=min(abs(z)); basis=zeros(3,1); basis(i)=1; y=cross(z,basis);
end
y=y/norm(y); x=cross(y,z); R=[x,y,z];
% Hamilton xyzw; branch on the largest component for numerical stability.
choices=[R(1,1);R(2,2);R(3,3);trace(R)]; [~,i]=max(choices);
if i==4
    w=sqrt(max(0,1+trace(R)))/2;
    q=[(R(3,2)-R(2,3))/(4*w);(R(1,3)-R(3,1))/(4*w);(R(2,1)-R(1,2))/(4*w);w];
else
    j=mod(i,3)+1; k=mod(j,3)+1;
    q=zeros(4,1); q(i)=sqrt(max(0,1+R(i,i)-R(j,j)-R(k,k)))/2;
    q(j)=(R(j,i)+R(i,j))/(4*q(i)); q(k)=(R(k,i)+R(i,k))/(4*q(i));
    q(4)=(R(k,j)-R(j,k))/(4*q(i));
end
q=q/norm(q);
end
