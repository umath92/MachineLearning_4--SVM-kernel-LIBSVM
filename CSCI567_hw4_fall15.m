function CSCI567_hw4_fall15()

% 3.1. Data Preprocessing
preProcess();

%-----------------------------------------------%

% 3.2. Implementing linear SVM
% Implemented trainsvm and testsvm. See files in directory.
%-----------------------------------------------%

% 3.3. Cross Validation for linear SVM
% Cross-validation to select C.
cross_vali();
% Outputting the accuracy on the test set for the optimal selected C
own_linear();
%-----------------------------------------------%

% 3.4. Linear SVM in LIBSVM
% Cross validation using LIBSVM toolkit.
LIBSVM_CROSS(1);
%-----------------------------------------------%

% 3.5. Kernel SVM in LIBSVM
% Polynomial Kernel
LIBSVM_CROSS(2);
% Gaussian Kernel
LIBSVM_CROSS(3);
% Outputting the accuracy on the test set for the optimal selected kernel
% and corresponding parameters.
libsvm();


end