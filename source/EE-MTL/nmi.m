function [NMI, is_perfect] = nmi(subgroup, subgroup_est)
S = size(subgroup,2);
S_est = size(subgroup_est,2);
M_s = zeros(1,S);
M_s_est = zeros(1,S_est);
for s=1:S
    M_s(s) = size(subgroup{s}, 2);
end
for s=1:S_est
    M_s_est(s) = size(subgroup_est{s}, 2);
end
M = sum(M_s);

I = 0;
for s=1:S
    for s_=1:S_est
        K = size(intersect(subgroup{s},subgroup_est{s_}),2);
        if K~=0
            I = I + K/M*log(M*K/M_s(s)/M_s_est(s_));
        end
    end
end

H = -sum(M_s/M.*log(M_s/M));
H_est = -sum(M_s_est/M.*log(M_s_est/M));

NMI = 2*I/(H+H_est);

%% Is perfect?
is_perfect = true;
if S == S_est
    for s=1:S
        if M_s(s) == M_s_est(s)
            if sum(subgroup{s}==subgroup_est{s})~=M_s(s)
                is_perfect = false;
                break;
            end
        else
            is_perfect = false;
            break;
        end
    end
else
    is_perfect = false;
end

end