import numpy as np
from dPCA import dPCA
from ee_dpca import EE_dPCA
from sklearn.decomposition import PCA, SparsePCA
import settings
import os
import time
import metrics

def simulation(N, P, T, S, D, sparsity):
    q = 10 # n components

    trialX = np.load('simulation/data/N{}_P{}_S{}_D{}_T{}_s{}/Data.npy'.format(N, P, S, D, T, sparsity))
    print('Loaded simulation/data/N{}_P{}_S{}_D{}_T{}_s{}/Data.npy.'.format(N, P, S, D, T, sparsity))
    X = np.mean(trialX, axis=0)
    X_ = X.reshape([P, -1])
    timecost = np.zeros(5) # (PCA, SPCA, dPCA, L2_dPCA, EE-dPCA)

    pca = PCA(n_components=q)
    dpca = dPCA.dPCA(n_components=q, labels='sdt')
    dpca.protect=['t']
    l2_dpca = dPCA.dPCA(n_components=q, labels='sdt', regularizer='auto')
    l2_dpca.protect=['t']
    spca = SparsePCA(n_components=q)
    eedpca = EE_dPCA(n_components=q, labels='sdt', rho=0.5)
    eedpca.protect=['t']

    print('#################################################################################################################################################')
    print('#################################################################################################################################################')
    print('#################################################################################################################################################')
    print('#################################################################################################################################################')
    print('Start PCA...')
    start = time.time()
    Xt_pca = pca.fit_transform(X_.T) # (SDT, q)
    timecost[0] = time.time() - start
    print(timecost[0])
    print('PCA time cost: {}'.format(timecost[0]))

    print('Start SPCA...')
    start = time.time()
    Xt_spca = spca.fit_transform(X_.T) # (SDT, q)
    timecost[1] = time.time() - start
    print('SPCA time cost: {}'.format(timecost[1]))

    print('Start dPCA...')
    Xt_dpca = dpca.fit_transform(X, trialX) # (q, S, D, T)  
    timecost[2] = dpca.runtimecost / len(Xt_dpca)
    print('dPCA time cost: {}'.format(timecost[2]))

    print('Start L2_dPCA...')
    Xt_l2_dpca = l2_dpca.fit_transform(X, trialX) # (q, S, D, T)
    timecost[3] = l2_dpca.runtimecost / len(Xt_l2_dpca)
    print('L2_dPCA time cost: {}'.format(timecost[3]))
    
    print('Start EE-dPCA...')
    Xt_eedpca = eedpca.fit_transform(X, trialX) # (q, S, D, T)
    timecost[4] = eedpca.runtimecost / len(Xt_eedpca)
    print('EE-dPCA time cost: {}'.format(timecost[4]))

    D_pca = pca.components_.T # (P, q)
    # D_spca = 
    D_dpca = dpca.D # (P, q)
    D_l2_dpca = l2_dpca.D # (P, q)
    D_eedpca = eedpca.D # (P, q)

    F_pca = pca.components_.T # (P, q)
    F_spca = spca.components_.T # (P, q)
    F_dpca = dpca.P # (P, q)
    F_l2_dpca = l2_dpca.P # (P, q)
    F_eedpca = eedpca.F # (P, q)

    print('Save all the results...')
    if not os.path.exists('simulation/result/N{}_P{}_S{}_D{}_T{}_s{}'.format(N, P, S, D, T, sparsity)): os.mkdir('simulation/result/N{}_P{}_S{}_D{}_T{}_s{}'.format(N, P, S, D, T, sparsity))
    np.save('simulation/result/N{}_P{}_S{}_D{}_T{}_s{}/Xt_pca.npy'.format(N, P, S, D, T, sparsity), Xt_pca)
    np.save('simulation/result/N{}_P{}_S{}_D{}_T{}_s{}/Xt_dpca.npy'.format(N, P, S, D, T, sparsity), Xt_dpca)
    np.save('simulation/result/N{}_P{}_S{}_D{}_T{}_s{}/Xt_spca.npy'.format(N, P, S, D, T, sparsity), Xt_spca)
    np.save('simulation/result/N{}_P{}_S{}_D{}_T{}_s{}/Xt_l2_dpca.npy'.format(N, P, S, D, T, sparsity), Xt_l2_dpca)
    np.save('simulation/result/N{}_P{}_S{}_D{}_T{}_s{}/Xt_eedpca.npy'.format(N, P, S, D, T, sparsity), Xt_eedpca)

    np.save('simulation/result/N{}_P{}_S{}_D{}_T{}_s{}/F_pca.npy'.format(N, P, S, D, T, sparsity), F_pca)
    np.save('simulation/result/N{}_P{}_S{}_D{}_T{}_s{}/F_dpca.npy'.format(N, P, S, D, T, sparsity), F_dpca)
    np.save('simulation/result/N{}_P{}_S{}_D{}_T{}_s{}/F_spca.npy'.format(N, P, S, D, T, sparsity), F_spca)
    np.save('simulation/result/N{}_P{}_S{}_D{}_T{}_s{}/F_l2_dpca.npy'.format(N, P, S, D, T, sparsity), F_l2_dpca)
    np.save('simulation/result/N{}_P{}_S{}_D{}_T{}_s{}/F_eedpca.npy'.format(N, P, S, D, T, sparsity), F_eedpca)

    np.save('simulation/result/N{}_P{}_S{}_D{}_T{}_s{}/D_pca.npy'.format(N, P, S, D, T, sparsity), D_pca)
    np.save('simulation/result/N{}_P{}_S{}_D{}_T{}_s{}/D_dpca.npy'.format(N, P, S, D, T, sparsity), D_dpca)
    # np.save('simulation/result/N{}_P{}_S{}_D{}_T{}_s{}/D_spca.npy'.format(N, P, S, D, T, sparsity), D_spca)
    np.save('simulation/result/N{}_P{}_S{}_D{}_T{}_s{}/D_l2_dpca.npy'.format(N, P, S, D, T, sparsity), D_l2_dpca)
    np.save('simulation/result/N{}_P{}_S{}_D{}_T{}_s{}/D_eedpca.npy'.format(N, P, S, D, T, sparsity), D_eedpca)

    np.save('simulation/result/N{}_P{}_S{}_D{}_T{}_s{}/timecost.npy'.format(N, P, S, D, T, sparsity), timecost)
    print('Done.')

exp_start = time.time()
settings.traverse(simulation)
print('Total exp time:', time.time() - exp_start)
metrics.metric()