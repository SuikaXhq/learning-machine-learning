import numpy as np
import os 
import settings

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

    X_s = np.zeros([P, S, D, T])
    X_d = np.zeros([P, S, D, T])
    X_t = np.zeros([P, S, D, T])
    X_s[s_support_set, :, :, :] += x_s_latent.T[:, :, None, None]
    X_d[d_support_set, :, :, :] += x_d_latent.T[:, None, :, None]
    X_t[t_support_set, :, :, :] += x_t_latent.T[:, None, None, :]

    if not os.path.exists('simulation/data/N{}_P{}_S{}_D{}_T{}_s{}'.format(N, P, S, D, T, sparsity)): os.mkdir('simulation/data/N{}_P{}_S{}_D{}_T{}_s{}'.format(N, P, S, D, T, sparsity))
    np.save('simulation/data/N{}_P{}_S{}_D{}_T{}_s{}/Data.npy'.format(N, P, S, D, T, sparsity), X)
    np.save('simulation/data/N{}_P{}_S{}_D{}_T{}_s{}/X_s.npy'.format(N, P, S, D, T, sparsity), X_s)
    np.save('simulation/data/N{}_P{}_S{}_D{}_T{}_s{}/X_d.npy'.format(N, P, S, D, T, sparsity), X_d)
    np.save('simulation/data/N{}_P{}_S{}_D{}_T{}_s{}/X_t.npy'.format(N, P, S, D, T, sparsity), X_t)

settings.traverse(generate_data)