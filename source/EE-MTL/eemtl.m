function [beta, alpha, theta, subgroup_full, subgroup, lambda_list, BIC, timecost] = eemtl(X, Z, Y, sigma, psi)
% X: 1xM Cell with n_i x p Matrix contents
% Z: 1xM Cell with n_i x q Matrix contents
% y: 1xM Cell with n_i-d Vector contents
% lambda: hyperparameter for EE
% sigma: Scalar, variance of observation noise
% psi: Scalar, variance of random effect

%% Initialize
fprintf('Initializing..\n');
timecost = zeros(1,6);
M = size(X,2);
p = size(X{1},2);
q = size(Z{1},2);
n = zeros(M,1);
for i=1:M
    n(i) = size(X{i},1);
end
fprintf('M = %d, p = %d, q = %d, N = %d\n', M, p, q, sum(n(:)));
fprintf('Calculating W_i..\n');
W = cell(1,M);
if nargin == 5
    for i=1:M
%         W{i} = sigma^(-2)*(eye(n(i)) - sigma^(-2)* Z{i}*(1/psi*eye(q) + sigma^(-2)*Z{i}'*Z{i})*Z{i}');
        W{i} = (sigma^2*eye(n(i))+psi^2*Z{i}*Z{i}')\eye(n(i));
    end
else
    lme = cell(1,M);
    psi = cell(1,M);
    sigma = zeros(1,M);
    parfor i=1:M
        lme{i} = fitlmematrix(X{i}, Y{i}, Z{i}, [], 'CovariancePattern', 'Isotropic','FitMethod','REML');
        [psi{i}, sigma(i)] = covarianceParameters(lme{i});
        W{i} = (sigma(i)^2*eye(n(i))+Z{i}*psi{i}{1}*Z{i}')\eye(n(i));
    end
end
fprintf('Initialization done.\n');

%% Step 1: Calculate check parameters
fprintf('Step 1: Calculate check parameters.\n');
beta_check = zeros(M, p);
theta_check = zeros(M, q);
tic;
for i=1:M
    T = [X{i},Z{i}];
    check = (T'*W{i}*T) \ T'*W{i}*Y{i};
    beta_check(i,:) = check(1:p);
    theta_check(i,:) = check(p+1:end);
end
timecost(1) = toc;
fprintf('Step 1 done. Timecost: %.6fs\n',timecost(1));

%% Step 2: Calculate tilde parameters
fprintf('Step 2: Calculate tilde parameters.\n');
tic;
LHS = 0;
RHS = 0;
for i=1:M
    K_i = X{i}'*W{i}*X{i};
    LHS = LHS + K_i;
    RHS = RHS + K_i * beta_check(i,:)';
end
beta_tilde = (LHS \ RHS)';
% beta_tilde = 1/M*sum(beta_check, 1);
theta_tilde = theta_check;
timecost(2) = toc;
fprintf('Step 2 done. Timecost: %.6fs\n',timecost(2));

%% Step 3: Calculate delta tilde
fprintf('Step 3: Calculate delta tilde.\n');
delta_tilde = zeros(M,M);
tic;
for i=1:M
    for j=1:M
        delta_tilde(i,j) = norm(theta_tilde(i,:)-theta_tilde(j,:));
    end
end
timecost(3) = toc;
fprintf('Step 3 done. Timecost: %.6fs\n',timecost(3));

%% BIC tuning
fprintf('Start tuning lambda via BIC.\n');
max_lambda = ceil(1.2*max(delta_tilde(:)));
fprintf('Max lambda: %d\n', max_lambda);
lambda_step = 0.05*max_lambda;
timecost_full = zeros(3,21);
theta_full = cell(1,21);
BIC = zeros(1,21);
subgroup_full = cell(1,21);
lambda_list = 0:lambda_step:max_lambda;
min_BIC = Inf;
index_min_BIC = 0;
t = 0;

for lambda = lambda_list
    fprintf('Lambda: %.3f\n', lambda);
    t = t+1;
    %% Step 4: Calculate EE-delta
    tic;
    EE_delta = sign(delta_tilde).*max(delta_tilde-lambda, 0);
    timecost_full(1,t) = toc;

    %% Step 5: Estimate subgroups
    index = 1:M;
    A_raw = EE_delta==0;
    subgroup_full{t} = {};
    subgroup_full{t} = subgroup_estimate(subgroup_full{t}, 1, A_raw);
    timecost_full(2,t) = toc;

    %% Step 6: Calculate alpha_s and theta_estimate
    S = size(subgroup_full{t},2);
    alpha = zeros(S, q);
    theta_estimate = theta_tilde;
    tic;
    for s=1:S
        M_s = size(subgroup_full{t}{s},2);
        alpha(s,:) = 1/M_s*sum(theta_tilde(subgroup_full{t}{s},:),1);
        theta_estimate(subgroup_full{t}{s},:) = alpha(s*ones(size(subgroup_full{t}{s},2),1),:);
    end
    timecost_full(3,t) = toc;

    %% Step 7: Calculate BIC
    theta_full{t} = theta_estimate;
    BIC(t) = bic(X, Y, Z, beta_tilde, theta_estimate, S);
    fprintf('BIC: %.4f\n', BIC(t));
    if BIC(t)<min_BIC
        index_min_BIC = t;
        min_BIC = BIC(t);
    end
end

%% Output
beta = beta_tilde;
theta = theta_full{index_min_BIC};
alpha = unique(theta, 'rows', 'stable');
subgroup = subgroup_full{index_min_BIC};
timecost(4:6) = mean(timecost_full,2);
fprintf('Step 4: Calculate EE-delta.\nAverage time cost: %.6fs\n', timecost(4));
fprintf('Step 5: Estimate subgroups.\nAverage time cost: %.6fs\n', timecost(5));
fprintf('Step 6: Calculate alpha_s and theta_estimate.\nAverage time cost: %.6fs\n', timecost(6));
fprintf('All steps done. Returning results.\n');
fprintf('Best BIC: %.4f\n', min_BIC);
fprintf('Best lambda: %.3f\n', lambda_list(index_min_BIC));
fprintf('Total time cost: %.6fs\n', sum(timecost(2:6)));
timecost = sum(timecost(2:6));

%% Sub-functions
function subgroup_new = subgroup_estimate(subgroup_old, i, A_raw)
    if size(A_raw,1)==0
        subgroup_new = subgroup_old;
        fprintf('Estimated subgroup number: %d\n', i-1);
    else
        %fprintf('Estimating subgroup No.%d\n', i);
        subgroup_est = A_raw(1,:);
        subgroup_old{i} = index(subgroup_est);
        index(subgroup_est) = [];
        A_raw(subgroup_est,:) = [];
        A_raw(:,subgroup_est) = [];
        subgroup_new = subgroup_estimate(subgroup_old, i+1, A_raw);
    end
end

end
