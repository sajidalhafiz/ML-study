function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% Add ones to the X data matrix
X = [ones(m, 1) X];

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.
%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%


a1=X;
z2=X*Theta1';
a2=sigmoid(z2);
a2 = [ones(m, 1) a2];
z3=a2*Theta2';
a3=sigmoid(z3);
y1 = zeros(m,num_labels);
temp2=0;temp1=0;
for i = 1:m
    y1(i,y(i))=(y(i)==y(i));
    temp2=temp2+sum((y1(i,:)*log(a3(i,:))')+((1-y1(i,:))*log(1-a3(i,:))'));
end;
temp2=(-1/m)*temp2;
[j1,k1]=size(Theta1);
for j = 1:j1
    for k=2:k1
        temp1=temp1+Theta1(j,k)^2;
    end;
end;
[j2,k2]=size(Theta2);
for j = 1:j2
    for k=2:k2
        temp1=temp1+Theta2(j,k)^2;
    end;
end;
temp1=(lambda/(2*m))*temp1;
J=temp1+temp2;

delta3=zeros(size(a3));
delta2=zeros(size(a2));

for i= 1:m
    delta3(i,:)=a3(i,:)-y1(i,:);
    delta2(i,:)=(delta3(i,:)*Theta2).*sigmoidGradient([ones z2(i,:)]);
end;
delta2=delta2(:,2:end);
Theta1_grad=Theta1_grad+delta2'*a1;
Theta2_grad=Theta2_grad+delta3'*a2;
Theta1_grad=[((1/m)*Theta1_grad(:,1)) ((1/m)*(Theta1_grad(:,2:end)+(lambda*Theta1(:,2:end))))];
Theta2_grad=[((1/m)*Theta2_grad(:,1)) ((1/m)*(Theta2_grad(:,2:end)+(lambda*Theta2(:,2:end))))];



% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
