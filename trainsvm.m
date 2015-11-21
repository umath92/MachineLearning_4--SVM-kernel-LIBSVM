function [w,bias]= trainsvm(train_feat,train_label,C)

%train_feat=evalin('base','train_feat');
%train_label=evalin('base','train_label');

[m,n]=size(train_feat);

H=diag([ones(1,n),zeros(1,m+1)]);
f=[zeros(1,n),C*ones(1,m),0]';
c=-1*ones(m,1);
l = train_feat;
A=[-diag(train_label)*l,-eye(m),-diag(train_label)*ones(m,1)];
lb = [-inf(n,1);zeros(m,1);-inf(1,1)];

opts=optimoptions('quadprog','Algorithm','interior-point-convex');
[z,fval,exitflag]=quadprog(H,f,A,c,[],[],lb,[],[],opts);
w=z(1:n,1);
bias=z(n+m+1,1);
end