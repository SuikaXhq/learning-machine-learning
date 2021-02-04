function [beta, alpha, theta, subgroup, timecost] = dishes(X, Z, Y, theta_U, W, Sigma_big)
% DISHES

% Input:
% X: 1xM Cell with n_i x p Matrix contents
% Z: 1xM Cell with n_i x q Matrix contents
% y: 1xM Cell with n_i-d Vector contents
% theta_U: Mxq matrix, unit GLS estimates, optional
% W: 1xM Cell, each element contains nxn matrix as W_i, optional
% Sigma_big: 1xM Cell, each element contains (p+q)x(p+q) matrix as [(X_i,Z_i)'*W_i*(X_i,Z_i)]^{-1}, optional

% Output:
% beta: px1 vector
% alpha: Sxq matrix with the s-th row as alpha_s
% theta: Mxq matrix with the i-th row as theta_i
% subgroup: 1xS cell with each element as a subgroup
% timecost: wall-clock time cost excluding unit GLS estimate

%% Initialize
%fprintf('Initializing..\n');
timecost = zeros(1,6);
M = size(X,2);
p = size(X{1},2);
q = size(Z{1},2);
n = zeros(M,1);
for i=1:M
    n(i) = size(X{i},1);
end
%fprintf('M = %d, p = %d, q = %d, N = %d\n', M, p, q, sum(n(:)));

if nargin < 6 % calculating theta_U, W, Sigma_big
    %fprintf('Calculating W_i..\n');
    W = cell(1,M);
    big_Z = zeros(sum(n), M*q);
    long_Z = zeros(sum(n), q);
    long_X = zeros(sum(n), p);
    long_Y = zeros(sum(n),1);
    group_var = zeros(sum(n),1);
    for i=1:M
        big_Z(1+sum(n(1:i-1)):sum(n(1:i)), 1+(i-1)*q:i*q) = Z{i};
        long_Z(1+sum(n(1:i-1)):sum(n(1:i)), :) = Z{i};
        long_X(1+sum(n(1:i-1)):sum(n(1:i)), :) = X{i};
        long_Y(1+sum(n(1:i-1)):sum(n(1:i))) = Y{i};
        group_var(1+sum(n(1:i-1)):sum(n(1:i))) = i;
    end
    lme = fitlmematrix([long_X, big_Z], long_Y, long_Z, group_var, 'CovariancePattern', 'Isotropic','FitMethod','REML');
    [psi, sigma] = covarianceParameters(lme);
    for i=1:M
        W{i} = (sigma*eye(n(i))+Z{i}*psi{1}*Z{i}')\eye(n(i));
    end
    %fprintf('Initialization done.\n');

    %% Step 1: Calculate unit GLS estimates
    %fprintf('Step 1: Calculate unit GLS estimates.\n');
    theta_U = zeros(M, q);
    Sigma_big = cell(1,M);
    Var_big = cell(1,M);
    tic;
    for i=1:M
        T = [X{i},Z{i}];
        Var_big{i} = T'*W{i}*T;
        Sigma_big{i} = Var_big{i} \ eye(p+q);
        theta_U(i,:) = Sigma_big{i}(p+1:end, :) * T'*W{i}*Y{i};
    end
    timecost(1) = toc;
    %fprintf('Step 1 done. Timecost: %.6fs\n',timecost(1));
end

%% Step 2: Calculate standardized difference
%fprintf('Step 2: Calculate standardized difference.\n');
delta_MU = zeros(M,M);
tic;
for i=1:M
    for j=i+1:M
        delta = theta_U(i,:)-theta_U(j,:);
        Rij = Sigma_big{i}(p+1:end, p+1:end) + Sigma_big{j}(p+1:end, p+1:end);
        delta_MU(i,j) = delta * (Rij \ delta');
    end
end
delta_MU = delta_MU + delta_MU';
timecost(2) = toc;
%fprintf('Step 2 done. Timecost: %.6fs\n',timecost(2));

%% Step 3: Task partitioning
%fprintf('Step 3: Task partitioning.\n');
subgroup = cell(1,M);
tic;
for i=1:M
    subgroup{i} = i;
    delta_MU(i,i) = Inf;
end
lambda = chi2inv(0.99, q);
for i=1:M
    if min(delta_MU, [], 'all')>lambda
        break;
    end
    D = delta_MU>lambda;
    link_num = sum(1-D,2);
    link_num(link_num==0) = Inf;
    [~, index] = min(link_num);
    [~, target] = min(delta_MU(index, :));
    
    % merge group index and group target
    subgroup{index} = [subgroup{index}, subgroup{target}];
    subgroup(target) = [];
    delta_MU(index, :) = max(delta_MU([index, target], :), [], 1);
    delta_MU(:, index) = delta_MU(index, :)';
    delta_MU(target, :) = [];
    delta_MU(:, target) = [];
end
timecost(3) = toc;
%fprintf('Step 3 done. Timecost: %.6fs\n',timecost(3));

%% Step 4: Calculate beta and alpha
%fprintf('Step 4: Calculate beta and alpha.\n');
S = size(subgroup,2);
G = zeros(sum(n), p+S*q);
theta = zeros(M, q);
long_Y = zeros(sum(n),1);
for i=1:M
    long_Y(1+sum(n(1:i-1)):sum(n(1:i))) = Y{i};
end
tic;
for s=1:S
    for i=subgroup{s}
        offset = sum(n(1:i-1))+1:sum(n(1:i));
        G(offset, 1:p) = X{i};
        G(offset, p+(s-1)*q+1:p+s*q) = Z{i};
    end
end

W{1} = sparse(W{1});
W_big = blkdiag(W{:});
estimate = (G'*W_big*G) \ G'*W_big*long_Y;
beta = estimate(1:p);
alpha = reshape(estimate(p+1:end), q,S);
alpha = alpha';
for s=1:S
    for i=subgroup{s}
        theta(i,:) = alpha(s,:);
    end
end
timecost(4) = toc;
%fprintf('Step 4 done. Timecost: %.6fs\n',timecost(4));

fprintf('Total time cost: %.6fs\n', sum(timecost(2:4)));
timecost = sum(timecost(2:4));



end
