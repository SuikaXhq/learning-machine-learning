clear;
file = fopen('Report_kmns.csv','w');
fprintf(file, 'Case,Timecost,S_mean,S_std,NMI,Perfect_recover\n');
fclose(file);

S_true = [3
    3
    3
    3
    3
    3
    5
    7
    9
    3
    3
    3
    3
    3
    3
    3
    3
    3
    3];


fprintf('Simulated Data Experiments(K-means):\n');
for case_number = [1:18]
S_est_full = zeros(1,100);
timecost_full = zeros(1,100);
NMI_full = zeros(1,100);
perfect_full = zeros(1,100);
subgroup_est = cell(1,100);

load(sprintf('data/Case%d.mat', case_number));
for j = 1:100
	fprintf('Method: K-means, Case: %d, Replicate: %d\n', case_number, j);
    [~, ~, ~, subgroup_est{j}, ~, timecost_full(j)] = kmeans(X_full{j}, Z_full{j}, Y_full{j}, S_true(case_number));
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

file = fopen('Report_kmns.csv','a');
fprintf(file, sprintf('%d,%.6f,%.2f,%.2f,%.6f,%.4f\n', case_number, timecost, S_mean, S_std, NMI, perfect_recover));
fclose(file);
save(sprintf('results/Case%d_kmns.mat', case_number), 'S_est_full','timecost_full','NMI_full','perfect_full','subgroup_est', '-v7.3');
clear -regexp *_full;
end
