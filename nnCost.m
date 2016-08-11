function [J grad] = nnCost(nn_params, ...
                                      input_layer_size, ...
                                      hidden_layer_size, ...
                                      num_labels, ...
                                      X, y, lambda)
                                      
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                        hidden_layer_size, input_layer_size + 1);
 Theta2 = reshape(nn_params(hidden_layer_size * (input_layer_size + 1) + 1 : end), ...
                        num_labels, hidden_layer_size + 1);           
                        
m = size(X, 1);
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

a1 = [ones(m, 1) X];
z2 = a1 * Theta1';
a2 = [ones(size(X, 1), 1) sigmoid(z2)];
z3 = a2 * Theta2';
H = sigmoid(z3);

J = (-1/m) * sum(y .* log(H) + (1 - y) .* log(1 - H));
reg = (lambda / (2 * m)) * (sum(sum(Theta1(:, 2:end).^2)) + sum(sum(Theta2(:, 2:end).^2)));

J = J + reg;

D1 = zeros(size(Theta1));
D2 = zeros(size(Theta2));

for t = 1:m,
  %forward
  a1 = [1 X(t, :)];
  z2 = a1 * Theta1';
  a2 = [1 sigmoid(z2)];
  z3 = a2 * Theta2';
  H = sigmoid(z3);
  %backward
  delta3 = H - y(t);
  %fprintf("%f\n", delta3);
  delta2 = Theta2' .* delta3 .* a2' .* (1 - a2');
  %fprintf("%i -- %i \n", size(delta2, 1), size(delta2, 2));
  delta2 = delta2(2:end);
  %
  D1 = D1 + delta2 .* a1;
  D2 = D2 + delta3 .* a2;
endfor;

tmp1 = Theta1;
Theta1(:, 1) = zeros(size(Theta1, 1), 1);
grad1 = (1 / m) * (D1 + lambda * Theta1);

Theta2(1) = 0;
grad2 = (1 / m) * (D2 + lambda * Theta2);

%unroll
grad = [grad1(:); grad2(:)];
end