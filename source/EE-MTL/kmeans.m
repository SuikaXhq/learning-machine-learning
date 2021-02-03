function [beta, alpha, theta, subgroup, BIC, timecost] = kmeans(X, Z, Y, theta_check, W)

% fprintf('Initializing..\n');
timecost = zeros(1,6);
M = size(X,2);
p = size(X{1},2);
q = size(Z{1},2);
n = zeros(M,1);
for i=1:M
    n(i) = size(X{i},1);
end
N = sum(n);
% fprintf('M = %d, p = %d, q = %d, N = %d\n', M, p, q, sum(n(:)));
if nargin<5 % calculating theta_check, W
    % fprintf('Calculating W_i..\n');
    W = cell(1,M);
    big_Z = zeros(sum(n), M*q);
    long_Z = zeros(sum(n), q);
    long_X = zeros(sum(n), p);
    long_Y = zeros(sum(n),1);
    G = zeros(sum(n),1);
    for i=1:M
        big_Z(1+sum(n(1:i-1)):sum(n(1:i)), 1+(i-1)*q:i*q) = Z{i};
        long_Z(1+sum(n(1:i-1)):sum(n(1:i)), :) = Z{i};
        long_X(1+sum(n(1:i-1)):sum(n(1:i)), :) = X{i};
        long_Y(1+sum(n(1:i-1)):sum(n(1:i))) = Y{i};
        G(1+sum(n(1:i-1)):sum(n(1:i))) = i;
    end
    lme = fitlmematrix([long_X, big_Z], long_Y, long_Z, G, 'CovariancePattern', 'Isotropic','FitMethod','REML');
    [psi, sigma] = covarianceParameters(lme);
    for i=1:M
        W{i} = (sigma*eye(n(i))+Z{i}*psi{1}*Z{i}')\eye(n(i));
    end
    % fprintf('Initialization done.\n');

    %% Calculate check parameters
    % fprintf('Step 1: Calculate check parameters.\n');
    beta_U = zeros(M, p);
    theta_check = zeros(M, q);
    Var_big = cell(1,M);
    tic;
    for i=1:M
        T = [X{i},Z{i}];
        Var_big{i} = T'*W{i}*T;
        check = Var_big{i} \ T'*W{i}*Y{i};
        beta_U(i,:) = check(1:p);
        theta_check(i,:) = check(p+1:end);
    end
    timecost(1) = toc;
    % fprintf('Step 1 done. Timecost: %.6fs\n',timecost(1));
end

%% K-means
% BIC tuning
min_BIC = Inf;
% fprintf('Step 3: K-means\n');
long_Y = zeros(sum(n),1);
for i=1:M
    long_Y(1+sum(n(1:i-1)):sum(n(1:i))) = Y{i};
end
theta_K = zeros(M, q);

tic;
for K=1:10
    % initial
%     fprintf('K = %d\n',K);
    centroids = theta_check(randperm(M,K),:);
    subgroup = zeros(1,M);
    dist = zeros(1,K);
    for m=1:M
        for k=1:K
            diff = centroids-theta_check(m*ones(1,K),:);
            dist(k) = norm(diff(k,:));
        end
        [~, subgroup(m)] = min(dist);
    end
    
    old_subgroup = zeros(1,M);
    i = 0;
    while sum(old_subgroup~=subgroup) && i<1000
        i = i+1;
        % Maximization
        for k=1:K
            centroids(k,:) = mean(theta_check(subgroup==k,:),1);
        end
        
        % Expectation
        old_subgroup = subgroup;
        for m=1:M
            for k=1:K
                diff = centroids-theta_check(m*ones(1,K),:);
                dist(k) = norm(diff(k,:));
            end
            [~, subgroup(m)] = min(dist);
        end
    end
    
    % translate the subgroups
    
    index = 1:M;
    subgroup_K = cell(1,K);
    tic;
    for k=1:K
        subgroup_K{k} = index(subgroup==k);
    end
    subgroup_K(cellfun(@isempty,subgroup_K))=[];
    



    %% Calculate beta and alpha
    tic;
    S = size(subgroup_K, 2);
    G = zeros(sum(n), p+S*q);
    for s=1:S
        for i=subgroup_K{s}
            offset = sum(n(1:i-1))+1:sum(n(1:i));
            G(offset, 1:p) = X{i};
            G(offset, p+(s-1)*q+1:p+s*q) = Z{i};
        end
    end

    W{1} = sparse(W{1});
    W_big = blkdiag(W{:});
    estimate = (G'*W_big*G) \ G'*W_big*long_Y;
    beta_K = estimate(1:q);
    alpha_K = reshape(estimate(q+1:end), q,S);
    alpha_K = alpha_K';
    for s=1:S
        for i=subgroup_K{s}
            theta_K(i,:) = alpha_K(s,:);
        end
    end
    
    BIC = bic(X, Y, Z, beta_K', theta_K, K);
%     fprintf('BIC: %.4f\n', BIC);
    if BIC<min_BIC
        subgroup_best = subgroup_K;
        beta = beta_K;
        theta = theta_K;
        alpha = alpha_K;
        min_BIC = BIC;
    end

end
timecost(2) = toc;

subgroup = subgroup_best;
BIC = min_BIC;
% fprintf('Best K: %d\n', best_K);
timecost = sum(timecost(2:6));
fprintf('Total time cost: %.6fs\n', timecost);

end
