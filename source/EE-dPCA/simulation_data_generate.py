import numpy as np
import os 

def generate_data(N, P, T, S, D, sparsity):
    # generate x_s
    # choose 1-sparse% dimension
    latent_P = int(P*(1-sparsity))
    s_support_set = np.random.choice(np.arange(P), latent_P, replace=False)
    x_s_latent = np.zeros([S, latent_P])
    x_s_0 = np.arange(S)/(S-1)*3*S-3/2*S
    x_s_latent[:, 0] = x_s_0
    for p in range(1,latent_P):
        x_s_latent[:, p] = x_s_latent[:, p-1][list(range(1,S))+[0]]
    
    # generate x_t
    t_support_set = np.random.choice(np.arange(P), latent_P, replace=False)
    x_t_0 = np.arange(T)*3
    x_t_latent = np.random.randn(latent_P)[None, :] * x_t_0[:, None]

    # generate x_d
    d_support_set = np.random.choice(np.arange(P), latent_P, replace=False)
    x_d_latent = np.zeros([D, latent_P])
    x_d_0 = np.arange(D)/(D-1)*3*D-3/2*D
    x_d_latent[:, 0] = x_d_0
    for p in range(1,latent_P):
        x_d_latent[:, p] = x_d_latent[:, p-1][list(range(1,D))+[0]]

    # add up all marginalized data
    X = np.random.randn(*[N, P, S, D, T])
    X[:, s_support_set, :, :, :] += x_s_latent.T[None, :, :, None, None]
    X[:, t_support_set, :, :, :] += x_t_latent.T[None, :, None, None, :]
    X[:, d_support_set, :, :, :] += x_d_latent.T[None, :, None, :, None]

    X_m = np.zeros([P, S, D, T])
    X_s = X_m[s_support_set, :, :, :] + x_s_latent.T[:, :, None, None]
    X_d = X_m[d_support_set, :, :, :] + x_d_latent.T[:, None, :, None]
    X_t = X_m[t_support_set, :, :, :] + x_t_latent.T[:, None, None, :]

    np.save('simulation/data/Data_N{}_P{}_S{}_D{}_T{}_s{}.npy'.format(N, P, S, D, T, sparsity), X)
    np.save('simulation/data/X_s_P{}_S{}_D{}_T{}_s{}.npy'.format(P, S, D, T, sparsity), X_s)
    np.save('simulation/data/X_d_P{}_S{}_D{}_T{}_s{}.npy'.format(P, S, D, T, sparsity), X_d)
    np.save('simulation/data/X_t_P{}_S{}_D{}_T{}_s{}.npy'.format(P, S, D, T, sparsity), X_t)

n_trials = 10
P = [100, 200, 300, 400, 500, 600]
T = [150, 250, 350, 450, 550, 650]
S = 6
D = 4
sparsity = [0, 0.2, 0.4, 0.6, 0.8]

for i in range(5):
    generate_data(n_trials, P[2], T[2], S, D, sparsity[i])
for j in [0,1,3,4,5]:
    generate_data(n_trials, P[j], T[2], S, D, sparsity[4])
    generate_data(n_trials, P[2], T[j], S, D, sparsity[4])

