clear;
case_number = 3;
n = 1024;
S = 7;
switch case_number
    case 1
        M = 50;
        p = 5;
        q = 3;
    case 2
        M = 100;
        p = 5;
        q = 3;
    case 3
        M = 50;
        p = 20;
        q = 12;
    case 4
        M = 100;
        p = 20;
        q = 12;
    case 5
        M = 50;
        p = 200;
        q = 120;
    case 6
        M = 100;
        p = 200;
        q = 120;
end

[X, Z, Y, beta_0, alpha_0, theta_0, subgroup] = data_generate(M,S,n,p,q);
[~, ~, ~, subgroup_full, ~, lambda_list, BIC, ~] = eemtl(X, Z, Y);
NMI_full = zeros(1,21);
for i=1:21
    NMI_full(i) = nmi(subgroup, subgroup_full{i});
end

fig = figure();
set(fig,'defaultAxesColorOrder',[0 0 0; 0 0 0]);
hold on
yyaxis right
bar(lambda_list, BIC, 1, 'FaceColor', [0.8,0.8,0.8]);
ylim([1.4, 10]);
ylabel('Modified BIC');

yyaxis left
plot(lambda_list, NMI_full, '-k');
ylim([-0.05, 1.05]);
[~, index] = min(BIC);
plot([lambda_list(index), lambda_list(index)], [-.05,1.05],'Color',[0.5,0.5,0.5],'LineWidth', 2.5);
ylabel('NMI');

xlabel('\lambda');

set(gca, 'SortMethod', 'depth', 'FontName', 'Times New Roman');

