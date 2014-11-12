addpath ../common
addpath ../common/minFunc_2012/minFunc
addpath ../common/minFunc_2012/minFunc/compiled



binary_digits = true;
[train,test] = ex1_load_mnist(binary_digits);



% Add row of 1s to the dataset to act as an intercept term.
train.X = [ones(1,size(train.X,2)); train.X]; 
test.X = [ones(1,size(test.X,2)); test.X];

% Training set dimensions
m=size(train.X,2);
n=size(train.X,1);

theta = rand(n,1)*0.001;

num_checks=30;

tic;
%average_error = grad_check(@logistic_regression_vec, theta, num_checks,  train.X, train.y);
average_error = grad_check(@logistic_regression, theta, num_checks,  train.X, train.y);
fprintf('Optimization took %f seconds.\n', toc);

