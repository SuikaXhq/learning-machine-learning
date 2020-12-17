clear;
file = fopen('Report_eemtl.csv','w');
fprintf(file, 'Case,Timecost,S_mean,S_median,S_min,S_max,NMI,Perfect_recover\n');
fclose(file);

fprintf('Simulated Data Experiments(EE-MTL):\n');
for case_number = 1:17
S_est_full = zeros(1,100);
timecost_full = zeros(1,100);
NMI_full = zeros(1,100);
perfect_full = zeros(1,100);
alpha_est = cell(1,100);
subgroup_est = cell(1,100);

load(sprintf('data/Case%d.mat', case_number));
for j = 1:100
	fprintf('Case: %d, Replicate: %d\n', case_number, j);
%     [~, alpha_est{j}, ~, ~, subgroup_est{j}, ~, ~, timecost_full(j)] = eemtl(X_full{j}, Z_full{j}, Y_full{j});
    [~, alpha_est{j}, ~, subgroup_est{j}, timecost_full(j)] = dishes(X_full{j}, Z_full{j}, Y_full{j});
    S_est_full(j) = size(alpha_est{j},1);
    [NMI_full(j), perfect_full(j)] = nmi(subgroup_full{j}, subgroup_est{j});
end

NMI = mean(NMI_full);
timecost = median(timecost_full);
perfect_recover = mean(perfect_full);
S_max = max(S_est_full);
S_min = min(S_est_full);
S_mean = mean(S_est_full);
S_median = median(S_est_full);

file = fopen('Report_eemtl.csv','a');
fprintf(file, sprintf('%d,%.6f,%.2f,%d,%d,%d,%.6f,%.4f\n', case_number, timecost, S_mean, S_median, S_min, S_max, NMI, perfect_recover));
fclose(file);
clear -regexp *_full;
end
