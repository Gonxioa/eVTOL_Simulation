function [x,status]=hex_trf(A,b,lb,ub)
%HEX_TRF Six-variable trust-region-reflective bounded linear least squares.
% Algorithm translated from scipy.optimize._lsq.trf_linear/common, SciPy
% 1.17.1 (BSD-3-Clause; see THIRD_PARTY_NOTICES.md).
tol=1e-10;
QT=zeros(6,10); Rbase=zeros(6,6);
if all(A(5:10,:)==0,'all')
    x=pinv(A(1:4,:))*b(1:4);
    [Qsmall,Rsmall,E]=qr(A(1:4,:));
    QT(1:4,1:4)=Qsmall'; Rbase(1:4,:)=Rsmall;
else
    x=pinv(A)*b;
    [Qfull,Rfull,E]=qr(A,'econ'); QT=Qfull'; Rbase=Rfull;
end
[~,permutation]=max(E,[],1);
if all(x>=lb) && all(x<=ub), status=3; return; end
x=interior(reflect(x,lb,ub),lb,ub,.1);
r=A*x-b; g=A'*r; cost=.5*(r'*r); status=0;
for iteration=1:200
    v=ones(6,1); dv=zeros(6,1);
    for j=1:6
        if g(j)<0, v(j)=ub(j)-x(j); dv(j)=-1;
        elseif g(j)>0, v(j)=x(j)-lb(j); dv(j)=1;
        end
    end
    gnorm=norm(g.*v,Inf);
    if gnorm<tol, status=1; end
    if status~=0, break; end
    h=g.*dv; d=sqrt(v); gh=d.*g; Ah=A.*d';
    ph=-regularized_qr(Rbase.*d(permutation)',QT*r,permutation,sqrt(h));
    p=d.*ph; pg=p'*g;
    if pg>0, status=-1; end
    theta=1-min(.005,gnorm);
    step=select_step(x,Ah,gh,h,p,ph,d,lb,ub,theta);
    change=-quadratic(A,g,step,zeros(6,1));
    if change<0
        alpha=1;
        for backtrack=1:100
            xn=reflect(x+alpha*p,lb,ub); step=xn-x;
            change=-quadratic(A,g,step,zeros(6,1));
            if change>-.1*alpha*pg, break; end
            alpha=alpha*.5;
        end
        if any(active_bounds(xn,lb,ub,1e-10)~=0)
            xn=interior(reflect(x+theta*alpha*p,lb,ub),lb,ub,0);
            step=xn-x; change=-quadratic(A,g,step,zeros(6,1));
        end
        % Preserve the upstream backtracking routine's returned x exactly.
    else
        x=interior(x+step,lb,ub,0);
    end
    r=A*x-b; g=A'*r;
    if change<tol*cost, status=2; end
    cost=.5*(r'*r);
end
end

function x=regularized_qr(R,rhs,permutation,diagonal)
diagonal=diagonal(permutation);
for i=1:6
    if diagonal(i)==0, continue; end
    row=zeros(6,1); row(i)=diagonal(i); u=0;
    for j=i:6
        if row(j)~=0
            f=R(j,j); g=row(j);
            if f==0
                cs=0; sn=sign(g); radius=abs(g);
            else
                radius=sign(f)*hypot(f,g); cs=f/radius; sn=g/radius;
            end
            R(j,j)=radius;
            for k=j+1:6
                old=R(j,k); R(j,k)=cs*old+sn*row(k); row(k)=-sn*old+cs*row(k);
            end
            old=rhs(j); rhs(j)=cs*old+sn*u; u=-sn*old+cs*u;
        end
    end
end
valid=abs(diag(R))>eps*10*max(abs(diag(R)));
z=zeros(6,1);
for j=6:-1:1
    if valid(j)
        value=rhs(j);
        for k=j+1:6
            if valid(k), value=value-R(j,k)*z(k); end
        end
        z(j)=value/R(j,j);
    end
end
x=zeros(6,1);
for j=1:6, x(permutation(j))=z(j); end
end

function step=select_step(x,A,g,h,p,ph,d,lb,ub,theta)
if all(x+p>=lb) && all(x+p<=ub), step=p; return; end
[stride,hits]=bound_stride(x,p,lb,ub);
rh=ph; rh(hits)=-rh(hits); reflected=d.*rh;
p=p*stride; ph=ph*stride; on_bound=x+p;
[upper,~]=bound_stride(on_bound,reflected,lb,ub);
lower=(1-theta)*upper; upper=theta*upper;
rvalue=Inf;
if upper>0
    [a,b,c]=line_quadratic(A,g,h,rh,ph);
    [stride,rvalue]=min_line(a,b,c,lower,upper);
    rh=ph+rh*stride; reflected=d.*rh;
end
p=p*theta; ph=ph*theta;
pvalue=quadratic(A,g,ph,h);
agh=-g; ag=d.*agh;
[upper,~]=bound_stride(x,ag,lb,ub); upper=theta*upper;
[a,b,~]=line_quadratic(A,g,h,agh,zeros(6,1));
[stride,agvalue]=min_line(a,b,0,0,upper); ag=ag*stride;
if pvalue<rvalue && pvalue<agvalue, step=p;
elseif rvalue<pvalue && rvalue<agvalue, step=reflected;
else, step=ag;
end
end

function x=reflect(y,lb,ub)
x=y;
if all(y>=lb) && all(y<=ub), return; end
width=ub-lb; z=mod(y-lb,2*width);
x=lb+min(z,2*width-z);
end

function mask=active_bounds(x,lb,ub,tol)
mask=zeros(6,1);
for j=1:6
    if tol==0
        if x(j)<=lb(j), mask(j)=-1; end
        if x(j)>=ub(j), mask(j)=1; end
    else
        lower=x(j)-lb(j); upper=ub(j)-x(j);
        if lower<=min(upper,tol*max(1,abs(lb(j)))), mask(j)=-1; end
        if upper<=min(lower,tol*max(1,abs(ub(j)))), mask(j)=1; end
    end
end
end

function x=interior(x,lb,ub,relative_step)
mask=active_bounds(x,lb,ub,relative_step);
for j=1:6
    if mask(j)==-1
        if relative_step==0, x(j)=lb(j)+eps(lb(j));
        else, x(j)=lb(j)+relative_step*max(1,abs(lb(j))); end
    elseif mask(j)==1
        if relative_step==0, x(j)=ub(j)-eps(ub(j));
        else, x(j)=ub(j)-relative_step*max(1,abs(ub(j))); end
    end
    if x(j)<lb(j) || x(j)>ub(j), x(j)=(lb(j)+ub(j))/2; end
end
end

function [stride,hits]=bound_stride(x,s,lb,ub)
steps=Inf(6,1);
for j=1:6
    if s(j)~=0, steps(j)=max((lb(j)-x(j))/s(j),(ub(j)-x(j))/s(j)); end
end
stride=min(steps); hits=steps==stride;
end

function value=quadratic(A,g,s,h)
v=A*s; value=.5*(v'*v+(s.*h)'*s)+g'*s;
end

function [a,b,c]=line_quadratic(A,g,h,s,s0)
v=A*s; u=A*s0;
a=.5*(v'*v+(s.*h)'*s);
b=g'*s+u'*v+(s0.*h)'*s;
c=.5*(u'*u)+g'*s0+.5*(s0.*h)'*s0;
end

function [t,value]=min_line(a,b,c,lb,ub)
t=lb; value=lb*(a*lb+b)+c;
upper_value=ub*(a*ub+b)+c;
if upper_value<value, t=ub; value=upper_value; end
if a~=0
    stationary=-.5*b/a;
    if stationary>lb && stationary<ub
        stationary_value=stationary*(a*stationary+b)+c;
        if stationary_value<value, t=stationary; value=stationary_value; end
    end
end
end
