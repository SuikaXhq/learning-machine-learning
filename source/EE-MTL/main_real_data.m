clear;
load('climate.mat');
[beta, alpha, theta, subgroup, timecost] = dishes(X, Z, Y);
save('climate_result.mat', 'beta', 'alpha', 'theta', 'subgroup', 'timecost');