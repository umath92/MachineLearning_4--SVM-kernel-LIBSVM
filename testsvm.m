function [test_acc]= testsvm(test_feat,test_label,w,bias)

predict1=test_feat*w+bias;
predict=(predict1>0)-1;
for i=1:size(predict,1)
    if(predict(i)==0)
        predict(i)=1;
    end
end
diff=(predict==test_label);

test_acc=sum(diff)/size(diff,1);

end