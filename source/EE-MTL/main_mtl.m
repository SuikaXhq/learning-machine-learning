clear;
fprintf('Simulated Data Experiments (MTL, CD fusion):\n');
%fprintf('Input hyper-parameters:\n');
case_number = input('Input case number: ');
n = 1024*(2-mod(case_number,2));
switch case_number
    case 0
        n = 128;
        M = 50;
        S = 3;
        p = 5;
        q = 3;
    case 1
        M = 50;
        S = 3;
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
        M = 100;
        S = 5;
        p = 20;
        q = 12;
    case 5
        M = 150;
        S = 7;
        p = 200;
        q = 120;
    case 6
        M = 150;
        S = 7;
        p = 200;
        q = 120;
end

S_est_full = zeros(1,20);
timecost_full = zeros(1,20);
NMI_full = zeros(1,20);
perfect_full = zeros(1,20);
RMSE_beta_full = zeros(1,20);
RMSE_theta_full = zeros(1,20);
X = cell(1,20);
Z = cell(1,20);
Y = cell(1,20);
beta_0 = cell(1,20);
alpha_0 = cell(1,20);
theta_0 = cell(1,20);
subgroup = cell(1,20);
beta_est = cell(1,20);
alpha_est = cell(1,20);
theta_est = cell(1,20);
subgroup_est = cell(1,20);
theta_full = cell(1,20);
lambda_list = cell(1,20);

for j = 1:20
    [X{j}, Z{j}, Y{j}, beta_0{j}, alpha_0{j}, theta_0{j}, subgroup{j}] = data_generate(M,S,n,p,q);
    %load(sprintf('Data_full_M%d_S%d_n%d_p%d_q%d.mat', M, S, n, p, q));
    [beta_est{j}, alpha_est{j}, theta_est{j}, subgroup_est{j}, timecost_full(j)] = mtl(X{j}, Z{j}, Y{j}, S);
    S_est_full(j) = size(alpha_est{j},1);
    [NMI_full(j), perfect_full(j)] = nmi(subgroup{j}, subgroup_est{j});
    RMSE_beta_full(j) = rmse(beta_0{j}, beta_est{j});
    RMSE_theta_full(j) = rmse(theta_0{j}, theta_est{j});
end

NMI_full = NMI_full(~isnan(NMI_full));
NaNs = sum(isnan(NMI_full));
NMI = mean(NMI_full);
timecost = median(timecost_full);
perfect_recover = mean(perfect_full);
S_max = max(S_est_full);
S_min = min(S_est_full);
S_mean = mean(S_est_full);
S_median = median(S_est_full);
RMSE_beta = mean(RMSE_beta_full);
RMSE_theta = mean(RMSE_theta_full);

file = fopen(sprintf('Report_mtl_case%d.txt',case_number),'w');
fprintf('-------------------------------\nReport:\nCase: %d\n\nTime cost: %.6f\nS_est(Mean, Median, Min, Max): %d,%d,%d,%d\nNMI: %.4f\nPerfect Recover: %.2f\nRMSE(beta): %.4f\nRMSE(theta): %.4f\n-------------------------------\nNMI NaNs: %d\n-------------------------------\n', case_number,timecost,S_mean,S_median,S_min,S_max,NMI,perfect_recover,RMSE_beta,RMSE_theta,NaNs);
fprintf(file, '-------------------------------\nReport:\nCase: %d\n\nTime cost: %.6f\nS_est(Mean, Median, Min, Max): %d,%d,%d,%d\nNMI: %.4f\nPerfect Recover: %.2f\nRMSE(beta): %.4f\nRMSE(theta): %.4f\n-------------------------------\nNMI NaNs: %d\n-------------------------------\n', case_number,timecost,S_mean,S_median,S_min,S_max,NMI,perfect_recover,RMSE_beta,RMSE_theta,NaNs);
fclose(file);
