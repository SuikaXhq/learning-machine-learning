for case_number = 5:18

n = [1024,1024,1024,1024,1024,1024,32,64,128,256,512,1024,1024,1024,1024,1024,1024,1024];
M = [50,50,50,50,50,50,50,50,50,50,50,5,10,20,30,40,50,50];
S = [3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,10,30];
p = [5,10,20,40,100,200,10,10,10,10,10,10,10,10,10,10,10,10];
q = [3,6,12,24,60,120,6,6,6,6,6,6,6,6,6,6,6,6];

n = n(case_number);
M = M(case_number);
S = S(case_number);
p = p(case_number);
q = q(case_number);

fprintf('Generating data for case %d, 100 replicates.\n', case_number);
generate_data(M,S,n,p,q,case_number);

end
