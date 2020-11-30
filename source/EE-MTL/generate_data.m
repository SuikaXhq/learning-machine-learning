function [X, Z, Y, beta_0, alpha_0, theta_0, subgroup] = generate_data(M,S,n,p,q,case_number)

X_full = cell(1,100);
Z_full = cell(1,100);
Y_full = cell(1,100);
subgroup_full = cell(1,100);

for k=1:100
X = cell(1,M);
Z = cell(1,M);
Y = cell(1,M);

%% Effects
beta_0 = rand(1,p)*4 - 2;
alpha_01 = (-1:2/(S-1):1)*(S^1.4/2);
if mod(S,2)==1
    alpha_01 = alpha_01 +1;
end
alpha_0 = zeros(S,q);
alpha_0(:,1) = alpha_01;
for j=2:q
    alpha_0(:,j) = alpha_0([end, 1:end-1],j-1);
end

u = zeros(M, q);
for j=1:M
    u(j,:) = mvnrnd(zeros(1,q), 0.3*eye(q));
end

%% Subgroups
theta_0 = zeros(M, q);
M_s = mnrnd(M-S, ones(1,S)/S) + 1;
subgroup = cell(1,S);
offset = 1;
for s=1:S 
    if s==1
        subgroup{s} = 1:M_s(s);
    else
        subgroup{s} = (sum(M_s(1:s-1))+1):(sum(M_s(1:s-1))+M_s(s));
    end
    theta_0(offset:offset+M_s(s)-1, :) = alpha_0(s*ones(M_s(s),1), :);
    offset = offset + M_s(s);
end

%% Data
for i=1:M
    %% Design matrix
    Sig_d = 0.3*ones(p+q)+0.7*eye(p+q);
    T = mvnrnd(zeros(1,p+q), Sig_d, n);
    X{i} = T(:, 1:p);
    Z{i} = T(:, p+1:end);
    E = mvnrnd(zeros(1,n), eye(n));
    
    %% Y_i
    Y{i} = X{i}*beta_0' + Z{i}*(theta_0(i,:)+u(i,:))' + E';
end

X_full{k} = X;
Y_full{k} = Y;
Z_full{k} = Z;
subgroup_full{k} = subgroup;

end

%% Save
save(sprintf('data/Case%d.mat', case_number), 'X_full','Z_full','Y_full','subgroup_full', '-v7.3');


end
