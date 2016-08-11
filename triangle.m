clear; close all; clc;

input_layer_size = 400;
hidden_layer = 25;
num_labels = 1;

data = load("finalData.txt");
X = data(:, 1:400); y = data(:, 401);
m = size(X, 1);

sel = randperm(size(X, 1));
sel = sel(1:100);
displayData(X(sel, :));

inital_Theta1 = randInitializeWeights(input_layer_size, hidden_layer);
inital_Theta2 = randInitializeWeights(hidden_layer, num_labels);

inital_nn_params = [inital_Theta1(:); inital_Theta2(:)];

options = optimset('MaxIter', 200);

lambda = 0.01;

costFunction = @(p) nnCost(p, input_layer_size, hidden_layer, num_labels, X, y, lambda);
[nn_params, cost] = fmincg(costFunction, inital_nn_params, options);

Theta1 = reshape(nn_params(1:hidden_layer * (input_layer_size + 1)), hidden_layer, input_layer_size  +1);
Theta2 = reshape(nn_params(hidden_layer * (input_layer_size + 1) + 1 : end),...
                          num_labels, hidden_layer + 1) ;
save Theta1.txt Theta1;
save Theta2.txt Theta2;

pred = predict(Theta1, Theta2, X);

fprintf("\nTraining Set Accuracy: %f\n", mean(double(pred==y)) * 100);
