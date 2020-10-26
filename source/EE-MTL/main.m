for case_number = 1:18
fprintf('Simulated Data Experiments:\n');
%fprintf('Input hyper-parameters:\n');
%case_number = input('Input case number: ');
n = [1024,1024,1024,1024,1024,1024,32,64,128,256,512,1024,1024,1024,1024,1024,1024,1024];
M = [50,50,50,50,50,50,50,50,50,50,50,5,10,20,30,40,50,50];
S = [3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,10,30];
p = [5,10,20,40,100,200,10,10,10,10,10,10,10,10,10,10,10,10];
q = [3,6,12,24,60,120,6,6,6,6,6,6,6,6,6,6,6,6];

n = n(case_number);
M = M(case_number);
S = S(case_number);
p = p(case_number);
q = q(case_number);

S_est_full = zeros(1,100);
timecost_full = zeros(1,100);
NMI_full = zeros(1,100);
perfect_full = zeros(1,100);
% RMSE_beta_full = zeros(1,100);
% RMSE_theta_full = zeros(1,100);
X = cell(1,100);
Z = cell(1,100);
Y = cell(1,100);
beta_0 = cell(1,100);
alpha_0 = cell(1,100);
theta_0 = cell(1,100);
subgroup = cell(1,100);
beta_est = cell(1,100);
alpha_est = cell(1,100);
theta_est = cell(1,100);
subgroup_est = cell(1,100);
theta_full = cell(1,100);
lambda_list = cell(1,100);

for j = 1:100
    % TODO: data generate
    [X{j}, Z{j}, Y{j}, ~, ~, ~, subgroup{j}] = data_generate(M,S,n,p,q);
    %load(sprintf('Data_full_M%d_S%d_n%d_p%d_q%d.mat', M, S, n, p, q));
    [~, alpha_est{j}, ~, ~, subgroup_est{j}, ~, ~, timecost_full(j)] = eemtl(X{j}, Z{j}, Y{j});
    S_est_full(j) = size(alpha_est{j},1);
    [NMI_full(j), perfect_full(j)] = nmi(subgroup{j}, subgroup_est{j});
%     RMSE_beta_full(j) = rmse(beta_0{j}, beta_est{j});
%     RMSE_theta_full(j) = rmse(theta_0{j}, theta_est{j});
end

NMI = mean(NMI_full);
timecost = median(timecost_full);
perfect_recover = mean(perfect_full);
S_max = max(S_est_full);
S_min = min(S_est_full);
S_mean = mean(S_est_full);
S_median = median(S_est_full);
% RMSE_beta = mean(RMSE_beta_full);
% RMSE_theta = mean(RMSE_theta_full);

file = fopen(sprintf('Report_case%d.txt',case_number),'w');
% fprintf('-------------------------------\nReport:\nCase: %d\n\nTime cost: %.6f\nS_est(Mean, Median, Min, Max): %d,%d,%d,%d\nNMI: %.4f\nPerfect Recover: %.2f\nRMSE(beta): %.4f\nRMSE(theta): %.4f\n-------------------------------\n', case_number,timecost,S_mean,S_median,S_min,S_max,NMI,perfect_recover);
fprintf(file, '-------------------------------\nReport:\nCase: %d\n\nTime cost: %.6f\nS_est(Mean, Median, Min, Max): %.2f,%d,%d,%d\nNMI: %.4f\nPerfect Recover: %.2f\nRMSE(beta): %.4f\nRMSE(theta): %.4f\n-------------------------------\n', case_number,timecost,S_mean,S_median,S_min,S_max,NMI,perfect_recover);
fclose(file);
clear;
end