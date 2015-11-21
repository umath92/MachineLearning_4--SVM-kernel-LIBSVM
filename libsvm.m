function libsvm()
preProcess();

train_feat=evalin('base','train_feat');
train_label=evalin('base','train_label');

test_feat=evalin('base','test_feat');
test_label=evalin('base','test_label');


train_feat_sparse=sparse(train_feat);
test_feat_sparse=sparse(test_feat);



val = ['-s 0 -t 2 -g 0.0625 ' '-c 4']
model = svmtrain(train_label,train_feat_sparse, val);
[predict_label, accuracy, dec_values] = svmpredict(test_label, test_feat_sparse, model);
accuracy


end