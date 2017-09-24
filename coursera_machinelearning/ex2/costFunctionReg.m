function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters.

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta


ht = sigmoid(X * theta);
lambda_term = lambda / (2 * m) * sum(theta(2:end) .^ 2);

J = 1 / m * sum(-y .* log(ht) - (1 - y) .* log(1 - ht)) + lambda_term

grad(1) = 1 / m * sum((ht - y) .* X(:, 1));

for i = 2:size(theta, 1)
    grad(i) = 1 / m * sum((ht - y) .* X(:, i)) + lambda / m * theta(i);
end

% =============================================================

end
