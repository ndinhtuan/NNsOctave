X = load("test_data.txt");

Theta1 = load("nnTheta1.txt");
Theta2 = load("nnTheta2.txt");

pred = predict(Theta1, Theta2, X);

%save pred.txt pred;
for i = 1:size(X,1),
  x = X(i, :);
  displayData(x);
  fprintf("result : %i\n", pred(i));
  pause;
endfor;