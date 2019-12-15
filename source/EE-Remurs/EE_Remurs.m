function [W, time] = EE_Remurs(X, y, ratio, lam, epsl, ita)

%% Set defalt parameters
if nargin < 4
    fprintf('Not enough parameters for Remurs! Terminate now.\n')
    return;
end

if nargin < 5
    epsl = 1.0;
end

N           = ndims(X) - 1; % input tensor rank
if nargin < 6
    ita = ones([N+1,1]);
end

%% Initialize
size_X      = size(X);
shape       = size_X(1:N); % input shape and also W shape
X_flat      = Unfold(X, size_X, N+1); % flatten X, M*D
D           = size(X_flat,2); % number of features
M           = size(X_flat,1); % number of samples

%% Train
tic;
[L, U] = factor(X_flat, epsl);
q = X_flat'*y;
if M >= D
    W_hat = U \ (L \ q);
else
    W_hat = q - (X_flat'*(U \ ( L \ (X_flat*q) )));
end
W_hat = reshape(W_hat, shape);
Ws_sum = ita(1)*prox_l1(W_hat*ita(1), lam);
for i=2:N+1
    W_i = prox_nuclear(Unfold(W_hat*ita(i), shape, i-1), ratio*lam/N);
    W_i_m = Fold(W_i, shape, i-1);
    Ws_sum = Ws_sum + W_i_m;
end

W = Ws_sum/sum(ita);
time = toc;
end

