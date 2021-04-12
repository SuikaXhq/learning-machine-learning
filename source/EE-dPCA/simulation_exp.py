import numpy as np
import dPCA
import ee_dpca

n_trials = 10
P = [100, 200, 300, 400, 500, 600]
T = [150, 250, 350, 450, 550, 650]
S = 6
D = 3
sparsity = [0, 0.2, 0.4, 0.6, 0.8]