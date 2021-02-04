clear;
file = fopen('results/Report_cd.csv','w');
fprintf(file, 'Case,Timecost,S_mean,S_std,NMI,Perfect_recover\n');
fclose(file);

fprintf('Simulated Data Experiments(CD):\n');
for case_number = [7:18]
S_est_full = zeros(1,3);
timecost_full = zeros(1,3);
NMI_full = zeros(1,3);
perfect_full = zeros(1,3);
subgroup_est = cell(1,3);

load(sprintf('data/Case%d.mat', case_number));
for j = 1:3
    load(sprintf('data/Case%d_Rep%d_unit_GLS.mat', case_number, j));
    fprintf('Method: CD, Case: %d, Replicate: %d\n', case_number, j);
    [~, ~, ~, subgroup_est{j}, timecost_full(j), ~] = cd_fusion(X_full{j}, Z_full{j}, Y_full{j}, beta_U, theta_U, W);
    S_est_full(j) = size(subgroup_est{j},2);
    [NMI_full(j), perfect_full(j)] = nmi(subgroup_full{j}, subgroup_est{j});
    fprintf('S: %d, NMI: %.4f.', S_est_full(j), NMI_full(j));
    if perfect_full(j)
        fprintf(' Perfect recovery.');
    end
    fprintf('\n\n');
end

NMI = mean(NMI_full);
timecost = median(timecost_full);
perfect_recover = mean(perfect_full);
S_mean = mean(S_est_full);
S_std = std(S_est_full);

file = fopen('results/Report_cd.csv','a');
fprintf(file, sprintf('%d,%.6f,%.2f,%.2f,%.6f,%.4f\n', case_number, timecost, S_mean, S_std, NMI, perfect_recover));
fclose(file);
save(sprintf('results/Case%d_cd.mat', case_number), 'S_est_full','timecost_full','NMI_full','perfect_full','subgroup_est', '-v7.3');
clear -regexp *_full;
end
