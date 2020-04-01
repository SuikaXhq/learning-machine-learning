Data = load( 'EE-Remurs-20x20x20.mat');
[tW, list, time] = Remurs(Data.x_train, Data.ytrain, 1e-2,1e-3,1e-4,1000);

%% predict
x_test_m = reshape(Data.x_test, [], size(Data.x_test,4));
y_predict = x_test_m' * tW(:);
mse = norm(y_predict - Data.ytest)/size(y_predict,1);
fprintf('Remurs:\n');
fprintf('mse=%f\n', mse);
fprintf('time=%f\n', time);
fprintf('--------------------------------------------------------\n')

[W, time] = EE_Remurs(Data.x_train, Data.ytrain, 1.0, 0.5);
y_predict_EE = x_test_m' * W(:);
mse = norm(y_predict_EE - Data.ytest)/size(y_predict,1);
fprintf('EE-Remurs:\n');
fprintf('mse=%f\n', mse);
fprintf('time=%f\n', time);