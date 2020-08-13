function RMSE = rmse(v1, v2)

d = size(v1,2);
RMSE = 1/sqrt(d)*norm(v1-v2);

end