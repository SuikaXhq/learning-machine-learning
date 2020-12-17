function [beta, alpha, theta, subgroup, timecost] = dishes(X, Z, Y, sigma, psi)
% X: 1xM Cell with n_i x p Matrix contents
% Z: 1xM Cell with n_i x q Matrix contents
% y: 1xM Cell with n_i-d Vector contents
% lambda: hyperparameter for EE
% sigma: Scalar, variance of observation noise
% psi: Scalar, variance of random effect

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
%fprintf('Calculating W_i..\n');
W = cell(1,M);
if nargin == 5
    for i=1:M
        W{i} = (sigma^2*eye(n(i))+psi^2*Z{i}*Z{i}')\eye(n(i));
    end
else
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
end
%fprintf('Initialization done.\n');

%% Step 1: Calculate unit-wise GLS estimates
%fprintf('Step 1: Calculate unit-wise GLS estimates.\n');
beta_U = zeros(M, p);
theta_U = zeros(M, q);
Sigma_big = cell(1,M);
tic;
for i=1:M
    T = [X{i},Z{i}];
    Sigma_big{i} = (T'*W{i}*T) \ eye(p+q);
    check = Sigma_big{i} * T'*W{i}*Y{i};
    beta_U(i,:) = check(1:p);
    theta_U(i,:) = check(p+1:end);
end
timecost(1) = toc;
%fprintf('Step 1 done. Timecost: %.6fs\n',timecost(1));

%% Step 2: Calculate modified unit-wise delta
%fprintf('Step 2: Calculate modified unit-wise delta.\n');
delta_MU = zeros(M,M);
tic;
for i=1:M
    for j=i+1:M
        Rij = sqrtm((Sigma_big{i}(p+1:end, p+1:end) + Sigma_big{j}(p+1:end, p+1:end)) \ eye(q));
        delta_MU(i,j) = norm(Rij*(theta_U(i,:)-theta_U(j,:))')^2;
    end
end
timecost(2) = toc;
%fprintf('Step 2 done. Timecost: %.6fs\n',timecost(2));

%% Step 3: Complete-linkage clustering
%fprintf('Step 3: Complete-linkage clustering.\n');
subgroup = cell(1,M);
tic;
for i=1:M
    subgroup{i} = [i];
end
lambda = chi2inv(0.99, q);
D = delta_MU>lambda;
D = D+D';
for i=1:M
    link_num = sum(1-D,2);
    if max(link_num)==1
        break;
    end
    min_link_num = min(link_num(link_num>1));
    index = find(link_num==min_link_num);
    index = index(1);
    candidates = find(D(index, :)==0);
    candidates = candidates(candidates~=index);
    target = candidates(randsample(size(candidates,2),1));
    
    % merge group index and group target
    subgroup{index} = [subgroup{index}, subgroup{target}];
    subgroup(target) = [];
    D(index, :) = D(index, :)|D(target, :);
    D(:, index) = D(index, :)';
    D(target, :) = [];
    D(:, target) = [];
end

timecost(3) = toc;
%fprintf('Step 3 done. Timecost: %.6fs\n',timecost(3));

%% Step 4: Calculate beta and alpha via Oracle estimator
%fprintf('Step 4: Calculate beta and alpha via Oracle estimator.\n');
S = size(subgroup,2);
alpha = zeros(S, q);
theta = zeros(M, q);
C = zeros(M*q, S*q);
% construct category matrix
for s=1:S
    for i=1:size(subgroup{s},2)
        index = subgroup{s}(i);
        C(1+(index-1)*q:index*q, 1+(s-1)*q:s*q) = eye(q);
    end
end
big_W = zeros(sum(n));
for i=1:M
    big_W(1+sum(n(1:i-1)):sum(n(1:i)), 1+sum(n(1:i-1)):sum(n(1:i))) = W{i};
end
tic;
big_T = [long_X, big_Z*C];
oracle = (big_T'*big_W*big_T) \ big_T'*big_W*long_Y;
beta = oracle(1:p);
oracle(1:p) = [];
for s=1:S
    alpha(s, :) = oracle(1+(s-1)*q:s*q)';
    theta(subgroup{s}, :) = alpha(s*ones(1,size(subgroup{s},2)), :);
end
timecost(4) = toc;
%fprintf('Step 4 done. Timecost: %.6fs\n',timecost(4));


%fprintf('All steps done. Returning results.\n');
fprintf('Total time cost: %.6fs\n', sum(timecost(2:4)));
timecost = sum(timecost(2:4));



end
