function [f,g] = softmax_regression(theta, X,y)
  %
  % Arguments:
  %   theta - A vector containing the parameter values to optimize.
  %       In minFunc, theta is reshaped to a long vector.  So we need to
  %       resize it to an n-by-(num_classes-1) matrix.
  %       Recall that we assume theta(:,num_classes) = 0.
  %
  %   X - The examples stored in a matrix.  
  %       X(i,j) is the i'th coordinate of the j'th example.
  %   y - The label for each example.  y(j) is the j'th example's label.
  %
  m=size(X,2);
  n=size(X,1);

  % theta is a vector;  need to reshape to n x num_classes.
  theta=reshape(theta, n, []);
  num_classes=size(theta,2)+1;
  
  % initialize objective value and gradient.
  f = 0;
  g = zeros(size(theta));

  %
  % TODO:  Compute the softmax objective function and gradient using vectorized code.
  %        Store the objective function value in 'f', and the gradient in 'g'.
  %        Before returning g, make sure you form it back into a vector with g=g(:);
  %
%%% YOUR CODE HERE %%%

  % X		n by examples
  % y		examples by 1, element is label 1 to 10
  % theta	n by (num_classes-1)
  % V		num_classes by examples
  % V_colsum	1 by examples
  % h		num_classes by examples
  % h_ind	1 by examples
  % g		n by num_classes

  V= [exp(theta' * X) ; ones(1,m)];
  V_colsum= sum(V,1);
  h= bsxfun (@rdivide, V, V_colsum); % devide V by it's columnsum

  % Suppose we have a matrix A and we want to extract a single element from each row, where the column of the element to be extracted from row i is storedA in y(i), where y is a row vector.
  A=h';
  ind=sub2ind(size(A), 1:size(A,1), y);
  h_ind = A(ind)';
  f= -sum(log(h_ind));
  ind_y = zeros(m,num_classes);
  ind_y(ind)=1;
  ind_y= ind_y(:,1:(num_classes-1));
  h= h(1:(num_classes-1),:);
  g= -X* (ind_y-h');


  g=g(:); % make gradient a vector for minFunc

