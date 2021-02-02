for case_number = 1:18

n = [100
    100
    100
    100
    100
    100
    100
    100
    100
    100
    100
    100
    100
    200
    300
    400
    500
    600];
M = [50
    100
    150
    200
    250
    300
    50
    50
    50
    50
    50
    50
    50
    50
    50
    50
    50
    50];
S = [3
    3
    3
    3
    3
    3
    5
    7
    9
    3
    3
    3
    3
    3
    3
    3
    3
    3];
p = [10
    10
    10
    10
    10
    10
    10
    10
    10
    20
    30
    40
    50
    10
    10
    10
    10
    10];
q = p;

n = n(case_number);
M = M(case_number);
S = S(case_number);
p = p(case_number);
q = q(case_number);

fprintf('Generating data for case %d, 100 replicates.\n', case_number);
generate_data(M,S,n,p,q,case_number);

end
