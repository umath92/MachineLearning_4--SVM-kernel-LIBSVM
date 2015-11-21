function [Acc] = cross_vali()

C=[4^(-6),4^(-5),4^(-4),4^(-3),4^(-2),4^(-1),1,4,4^2];

train_feat=evalin('base','train_feat');
train_label=evalin('base','train_label');

totalAcc=zeros(size(C,2),1);
time2=zeros(size(C,2),1);

for i=1:size(C,2)
    s=size(train_feat,1)/3;
    t1=train_feat(1:int64(s),:);
    t2=train_feat(int64(s+1):int64(2*s),:);
    t3=train_feat(int64(2*s+1):size(train_feat,1),:);
 
    tic;
    [w,bias]=trainsvm(vertcat(t1,t2),train_label(1:int64(2*s),:),C(i));
    t=toc;
    time2(i)=time2(i)+t;
    acc=testsvm(t3,train_label(int64(2*s+1):size(train_feat,1),:),w,bias);
    totalAcc(i)=totalAcc(i)+acc;
    %acc
    
    tic;
    [w,bias]=trainsvm(vertcat(t2,t3),train_label(int64(s+1):size(train_feat,1),:),C(i));
    t=toc;
    time2(i)=time2(i)+t;
    acc=testsvm(t1,train_label(1:int64(s)),w,bias);
    totalAcc(i)=totalAcc(i)+acc;
    %acc
    
    
    tic;
    [w,bias]=trainsvm(vertcat(t1,t3),vertcat(train_label(1:int64(s),:),train_label(int64(2*s+1):size(train_feat,1),:)),C(i));
    t=toc;
    time2(i)=time2(i)+t;
    acc=testsvm(t2,train_label(int64(s+1):int64(2*s)),w,bias);
    totalAcc(i)=totalAcc(i)+acc;
    %acc
end

disp(sprintf( 'The accuracies for different C values are:\n '));
Acc=totalAcc/3
disp(sprintf( 'The time for training for different C values are:\n '));
Time=time2
plot(log(C),totalAcc/3);
title('Selecting optimal C');
xlabel('log(C) {base 2}');
ylabel('Accuracy');
[a,i]=max(Acc);
disp(sprintf( 'Thus, the maxium accuracy is %d at C = %d:\n ',a,C(i)));

end