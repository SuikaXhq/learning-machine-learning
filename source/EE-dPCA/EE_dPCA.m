function [L, U] = factor(A, rho)
    [m, n] = size(A);
    if ( m >= n )    % if skinny
       L = chol( A'*A + rho*speye(n), 'lower' );
       
    else            % if fat
       L = chol( speye(m) + 1/rho*(A*A'), 'lower' );
    end
    % force matlab to recognize the upper / lower triangular structure
    L = sparse(L);
    U = sparse(L');
end

function [D, F] = EE_dPCA(data, label, hyperpara)

lambda = hyperpara.lambda;
gamma = hyperpara.gamma;
alpha = hyperpara.alpha;
epsl = hyperpara.epsl;
X = double(data); % sample matrix NxP
N = size(X, 1); % number of samples
P = size(X, 2); % dimension of samples

% Assume data matrix has been centralized.

% Calculate (X'X)^-1 X'X_phi
[L, U] = factor(X, epsl);
q = X'*X_phi;
if N >= P
    W_hat = U \ (L \ q);
else
    W_hat = q - (X'*(U \ ( L \ (X*q) )));
end