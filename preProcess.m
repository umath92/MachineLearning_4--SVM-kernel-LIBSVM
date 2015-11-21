function preProcess()

train=load('phishing-train.mat');
assignin('base', 'train_label',double(train.label'));
assignin('base', 'train_feat',double(train.features));
train_feat=double(train.features);


test=load('phishing-test.mat');
assignin('base', 'test_label',double(test.label'));
assignin('base', 'test_feat',double(test.features));
test_feat=double(test.features);

train_cat=[];

for col=1:30
    i_m=0;
    i_p=0;
    i_z=0;
    for row=1:2000
        if train_feat(row,col)==-1
            i_m=1;
        elseif train_feat(row,col)==0
            i_z=1; 
        else
            i_p=1;
        end
    end
    if(col==16 ||(i_m==1 && i_z==1 && i_p==1))
        for row=1:2000
            if train_feat(row,col)==-1
                train_feat(row,col)=1.0;
            elseif train_feat(row,col)==0
                train_feat(row,col)=2.0;
            else
                train_feat(row,col)=3.0;
            end
        end
        
        train_cat=horzcat(train_cat,dummyvar((train_feat(:,col))));
    else
        train_cat=horzcat(train_cat,(train_feat(:,col)));
    end
    
end

train_cat=train_cat(:,any(train_cat));
assignin('base', 'train_feat',train_cat);



test_cat=[];

for col=1:30
    i_m=0;
    i_p=0;
    i_z=0;
    for row=1:2000
        if test_feat(row,col)==-1
            i_m=1;
        elseif test_feat(row,col)==0
            i_z=1; 
        else
            i_p=1;
        end
    end
    if(col==16 ||(i_m==1 && i_z==1 && i_p==1))
        for row=1:2000
            if test_feat(row,col)==-1
                test_feat(row,col)=1.0;
            elseif test_feat(row,col)==0
                test_feat(row,col)=2.0;
            else
                test_feat(row,col)=3.0;
            end
        end
        
        test_cat=horzcat(test_cat,dummyvar((test_feat(:,col))));
    else
        test_cat=horzcat(test_cat,(test_feat(:,col)));
    end
    
end

test_cat=test_cat(:,any(test_cat));
assignin('base', 'test_feat',test_cat);


end