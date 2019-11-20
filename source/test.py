import numpy as np
import matrix_producer as mp

P, K, S, N = mp.hyper_para()
tests = mp.generate_in_cov(P, K, S, model=2)
for m in tests:
    print(np.linalg.det(m))
    np.linalg.inv(m)
#    if np.linalg.det(m)-0.0<1e-17:
#        print(m)