function own_linear()
preProcess();
C=[4^(-6),4^(-5),4^(-4),4^(-3),4^(-2),4^(-1),1,4,4^2];
%[acc]=cross_vali();
%[~,i]=max(acc);
i=5;

train_feat=evalin('base','train_feat');
train_label=evalin('base','train_label');

test_feat=evalin('base','test_feat');
test_label=evalin('base','test_label');


[w,bias]=trainsvm(train_feat,train_label,C(i));
acc=testsvm(test_feat,test_label,w,bias);

disp(sprintf('After selecting the optimal C, the accuracy for the test set is: %d\n ',acc));



end