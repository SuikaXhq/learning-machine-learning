clear;
load('climate.mat');
for nu=[0.6 0.7 0.8 0.9 0.95]
    [beta, alpha, theta, subgroup, timecost] = dishes(X, Z, Y, nu);
    save(sprintf('results/climate_result_nu%.2f.mat',nu), 'beta', 'alpha', 'theta', 'subgroup', 'timecost');
    output_real_exp;
end