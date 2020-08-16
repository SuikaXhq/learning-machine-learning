function [beta, alpha, theta, subgroup, BIC, timecost] = kmeans(X, Z, Y, sigma, psi)

fprintf('Initializing..\n');
timecost = zeros(1,6);
M = size(X,2);
p = size(X{1},2);
q = size(Z{1},2);
n = zeros(M,1);
for i=1:M
    n(i) = size(X{i},1);
end
N = sum(n);
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

%% K-means
% BIC tuning
min_BIC = Inf;
fprintf('Step 3: K-means\n');
timecost_full = zeros(1,10);
for K=1:10
    % initial
    fprintf('K = %d\n',K);
    tic;
    centroids = theta_tilde(randperm(M,K),:);
    subgroup = zeros(1,M);
    dist = zeros(1,K);
    for m=1:M
        for k=1:K
            diff = centroids-theta_tilde(m*ones(1,K),:);
            dist(k) = norm(diff(k,:));
        end
        [~, subgroup(m)] = min(dist);
    end
    
    old_subgroup = zeros(1,M);
    i = 0;
    while sum(old_subgroup~=subgroup) || i>1000
        i = i+1;
        % Maximization
        for k=1:K
            centroids(k,:) = mean(theta_tilde(subgroup==k,:),1);
        end
        
        % Expectation
        old_subgroup = subgroup;
        for m=1:M
            for k=1:K
                diff = centroids-theta_tilde(m*ones(1,K),:);
                dist(k) = norm(diff(k,:));
            end
            [~, subgroup(m)] = min(dist);
        end
    end
    timecost_full(K) = toc;
    alpha_k = centroids;
    S = K;
    theta_k = alpha_k(subgroup,:);
    BIC = bic(X, Y, Z, beta_tilde, theta_k, S);
    fprintf('BIC: %.4f\n', BIC);
    if BIC<min_BIC
        alpha = alpha_k;
        theta = theta_k;
        subgroup_est = subgroup;
        timecost(3) = timecost_full(K);
        best_K = K;
        min_BIC = BIC;
    end
end
beta = beta_tilde;
index = 1:M;
subgroup = cell(1,best_K);
for k=1:best_K
    subgroup{k} = index(subgroup_est==k);
end
BIC = min_BIC;
fprintf('Best K: %d\n', best_K);
timecost = sum(timecost(2:6));
fprintf('Total time cost: %.6fs\n', timecost);

end