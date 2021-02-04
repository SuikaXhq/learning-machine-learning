clear;
load('climate.mat');
for nu=[0.02, 0.05, 0.1]
    [beta, alpha, theta, subgroup, timecost] = dishes(X, Z, Y);
    save(sprintf('results/climate_result_nu%.2f.mat',nu), 'beta', 'alpha', 'theta', 'subgroup', 'timecost');
end