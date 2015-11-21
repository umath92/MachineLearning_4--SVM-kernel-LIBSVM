function LIBSVM_CROSS(in)

%{
Usage: svm-train [options] training_set_file [model_file]
options:
-s svm_type : set type of SVM (default 0)
	0 -- C-SVC		(multi-class classification)
	1 -- nu-SVC		(multi-class classification)
	2 -- one-class SVM	
	3 -- epsilon-SVR	(regression)
	4 -- nu-SVR		(regression)
-t kernel_type : set type of kernel function (default 2)
	0 -- linear: u'*v
	1 -- polynomial: (gamma*u'*v + coef0)^degree
	2 -- radial basis function: exp(-gamma*|u-v|^2)
	3 -- sigmoid: tanh(gamma*u'*v + coef0)
	4 -- precomputed kernel (kernel values in training_set_file)
-d degree : set degree in kernel function (default 3)
-g gamma : set gamma in kernel function (default 1/num_features)
-r coef0 : set coef0 in kernel function (default 0)
-c cost : set the parameter C of C-SVC, epsilon-SVR, and nu-SVR (default 1)
-n nu : set the parameter nu of nu-SVC, one-class SVM, and nu-SVR (default 0.5)
-p epsilon : set the epsilon in loss function of epsilon-SVR (default 0.1)
-m cachesize : set cache memory size in MB (default 100)
-e epsilon : set tolerance of termination criterion (default 0.001)
-h shrinking : whether to use the shrinking heuristics, 0 or 1 (default 1)
-b probability_estimates : whether to train a SVC or SVR model for probability estimates, 0 or 1 (default 0)
-wi weight : set the parameter C of class i to weight*C, for C-SVC (default 1)
-v n: n-fold cross validation mode
-q : quiet mode (no outputs)

%}

train_feat=evalin('base','train_feat');
train_label=evalin('base','train_label');

test_feat=evalin('base','test_feat');
test_label=evalin('base','test_label');


train_feat_sparse=sparse(train_feat);
test_feat_sparse=sparse(test_feat);

if in==1
    linearKernel(train_label,train_feat_sparse);
elseif in==2
    polyKernel(train_label,train_feat_sparse);
elseif in==3
    gaussianKernel(train_label,train_feat_sparse);
end






%%% ----------------------------------------------------------------------------------------------------------
%%% ----------------------------------------------------------------------------------------------------------


end

function linearKernel(train_label,train_feat_sparse)
C=[4^(-6),4^(-5),4^(-4),4^(-3),4^(-2),4^(-1),1,4,4^2];
for i=1:size(C,2)
    x=num2str(C(i));
    val = ['-s 0 -t 0 ' '-c ' x ' -v 3']
    tic;
    model(i) = svmtrain(train_label,train_feat_sparse, val);
    time(i)=toc;
end

[maxVal,index]=max(model);
disp(sprintf( 'The accuracies for different C values are:\n '));
model'
disp(sprintf( 'The time for training for different C values are:\n '));
time'
disp(sprintf( 'Thus, the maximum accuracy is %d at C = %d:\n ',maxVal,C(index)));
plot(log(C)/log(4),model);

%[predict_label, accuracy, dec_values] = svmpredict(test_label, test_feat_sparse, model); 

%%% ----------------------------------------------------------------------------------------------------------
%%% ----------------------------------------------------------------------------------------------------------

end

function polyKernel(train_label,train_feat_sparse)

C=[4^(-3),4^(-2),4^(-1),1,4,4^2,4^3,4^4,4^5,4^6,4^7];
disp(sprintf( '--------- Polynomial Kernel -----------\n '));
degree=[1,2,3];

k=1;
for i=1:size(C,2)
    x=num2str(C(i));
    for j=1:size(degree,2)
        dd=num2str(degree(j));
        val = ['-s 0 -t 1 -g 1 -d ' dd ' -c ' x ' -v 3']
        tic;
        model(k) = svmtrain(train_label,train_feat_sparse, val);
        time(k)=toc;
        k=k+1;  
    end
end

[maxVal,index]=max(model);
disp(sprintf( 'The accuracies for different C values are:\n '));
model'
disp(sprintf( 'The time for training for different C values are:\n '));
time'
q=ceil(index/3);
r=mod(index,3);
disp(sprintf( 'Thus, the maximum accuracy is %d at C = %d and D=%d \n ',maxVal,C(q),degree(r)));
end


function gaussianKernel(train_label,train_feat_sparse)

C=[4^(-3),4^(-2),4^(-1),1,4,4^2,4^3,4^4,4^5,4^6,4^7];
disp(sprintf( '--------- Gaussian Kernel -----------\n '));
gamma=[4^(-7),4^(-6),4^(-5),4^(-4),4^(-3),4^(-2),4^(-1)];
k=1;
for i=1:size(C,2)
    x=num2str(C(i));
    for j=1:size(gamma,2)
        dd=num2str(gamma(j));
        val = ['-s 0 -t 2 -g ' dd ' -c ' x ' -v 3']
        tic;
        model(k) = svmtrain(train_label,train_feat_sparse, val);
        time(k)=toc;
        k=k+1;
    end
end

[maxVal,index]=max(model);
disp(sprintf( 'The accuracies for different C values are:\n '));
model'
disp(sprintf( 'The time for training for different C values are:\n '));
time'
g=mod(index,7);
in=ceil(index/7);
disp(sprintf( 'Thus, the maximum accuracy is %d at C = %d and Gamma=%d.\n ',maxVal,C(in),gamma(g)));

end
