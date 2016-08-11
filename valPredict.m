data = load("validation_data.txt");
X = data(:, 1:400); y = data(:, 401);

Theta1 = load("nnTheta1.txt");
Theta2 = load("nnTheta2.txt");

pred = predict(Theta1, Theta2, X);

fprintf("Trainning accuracy : %f.\n", mean(double(pred == y)) * 100);