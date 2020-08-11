clear;
fprintf('Simulated Data Experiments:\n');
%fprintf('Input hyper-parameters:\n');
case_number = input('Input case number: ');
n = 1024*(2-mod(case_number,2));
switch case_number
    case 1
        M = 50;
        S = 2;
        p = 5;
        q = 3;
    case 2
        M = 100;
        S = 3;
        p = 5;
        q = 3;
    case 3
        M = 100;
        S = 5;
        p = 20;
        q = 12;
    case 4
        M = 150;
        S = 7;
        p = 20;
        q = 12;
    case 5
        M = 100;
        S = 5;
        p = 200;
        q = 120;
    case 6
        M = 150;
        S = 7;
        p = 200;
        q = 120;
end

S_est_full = zeros(1,100);
timecost_full = zeros(1,100);
NMI_full = zeros(1,100);
perfect_full = zeros(1,100);

for j = 1:100
    [X, Z, Y, beta_0, alpha_0, theta_0, subgroup] = data_generate(M,S,n,p,q);
    %load(sprintf('Data_full_M%d_S%d_n%d_p%d_q%d.mat', M, S, n, p, q));
    [beta_est, alpha_est, theta_est, theta_full, subgroup_est, lambda_list, BIC, timecost] = eemtl(X, Z, Y);
    S_est_full(j) = size(alpha_est,1);
    timecost_full(j) = timecost;
    [NMI_full(j), perfect_full(j)] = nmi(subgroup, subgroup_est);
end

NMI = mean(NMI_full);
timecost = mean(timecost_full);
perfect_recover = mean(perfect_full);
S_max = max(S_est_full);
S_min = min(S_est_full);
S_mean = mean(S_est_full);
S_median = median(S_est_full);

fprintf('-------------------------------\nReport:\nCase: %d\n\nTime cost: %.6f\nS_est(Mean, Median, Min, Max): %d,%d,%d,%d\nNMI: %.4f\nPerfect Recover: %.2f\n', case_number,timecost,S_mean,S_median,S_min,S_max,NMI,perfect_recover);
