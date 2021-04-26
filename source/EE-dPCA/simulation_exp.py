import numpy as np
from dPCA import dPCA
from ee_dpca import EE_dPCA
from sklearn.decomposition import PCA, SparsePCA
import settings
import os
import time
import metrics
import sys

def simulation(N, P, T, S, D, sparsity, flag='11111'):
    q = 20 # n components

    trialX = np.load('simulation/data/N{}_P{}_S{}_D{}_T{}_s{}/Data.npy'.format(N, P, S, D, T, sparsity))
    print('Loaded simulation/data/N{}_P{}_S{}_D{}_T{}_s{}/Data.npy.'.format(N, P, S, D, T, sparsity))
    X = np.mean(trialX, axis=0)
    X_ = X.reshape([P, -1])
    timecost = np.zeros(5) # (PCA, SPCA, dPCA, L2_dPCA, EE-dPCA)
    if os.path.exists('simulation/result/N{}_P{}_S{}_D{}_T{}_s{}/timecost.npy'.format(N, P, S, D, T, sparsity)): timecost = np.load('simulation/result/N{}_P{}_S{}_D{}_T{}_s{}/timecost.npy'.format(N, P, S, D, T, sparsity))
    if flag[0]=='1': pca = PCA(n_components=q)
    if flag[1]=='1': spca = SparsePCA(n_components=q)
    if flag[2]=='1':
        dpca = dPCA.dPCA(n_components=q, labels='sdt')
        dpca.protect=['t']
    if flag[3]=='1':
        l2_dpca = dPCA.dPCA(n_components=q, labels='sdt', regularizer='auto')
        l2_dpca.protect=['t']
    if flag[4]=='1': 
        eedpca = EE_dPCA(n_components=q, labels='sdt', rho=1)
        eedpca.protect=['t']

    print('#################################################################################################################################################')
    print('#################################################################################################################################################')
    print('#################################################################################################################################################')
    print('#################################################################################################################################################')
    if flag[0]=='1':
        print('Start PCA...')
        start = time.time()
        Xt_pca = pca.fit_transform(X_.T) # (SDT, q)
        timecost[0] = time.time() - start
        print(timecost[0])
        print('PCA time cost: {}'.format(timecost[0]))

    if flag[1]=='1':
        print('Start SPCA...')
        start = time.time()
        Xt_spca = spca.fit_transform(X_.T) # (SDT, q)
        timecost[1] = time.time() - start
        print('SPCA time cost: {}'.format(timecost[1]))

    if flag[2]=='1':
        print('Start dPCA...')
        Xt_dpca = dpca.fit_transform(X, trialX) # (q, S, D, T)  
        timecost[2] = dpca.runtimecost / len(Xt_dpca)
        print('dPCA time cost: {}'.format(timecost[2]))

    if flag[3]=='1':
        print('Start L2_dPCA...')
        Xt_l2_dpca = l2_dpca.fit_transform(X, trialX) # (q, S, D, T)
        timecost[3] = l2_dpca.runtimecost / len(Xt_l2_dpca)
        print('L2_dPCA time cost: {}'.format(timecost[3]))
    
    if flag[4]=='1':
        print('Start EE-dPCA...')
        Xt_eedpca = eedpca.fit_transform(X, trialX) # (q, S, D, T)
        timecost[4] = eedpca.runtimecost / len(Xt_eedpca)
        print('EE-dPCA time cost: {}'.format(timecost[4]))


    print('Save all the results...')
    if not os.path.exists('simulation/result/N{}_P{}_S{}_D{}_T{}_s{}'.format(N, P, S, D, T, sparsity)): os.mkdir('simulation/result/N{}_P{}_S{}_D{}_T{}_s{}'.format(N, P, S, D, T, sparsity))
    if flag[0]=='1':
        D_pca = pca.components_.T # (P, q)
        F_pca = pca.components_.T # (P, q)
        np.save('simulation/result/N{}_P{}_S{}_D{}_T{}_s{}/Xt_pca.npy'.format(N, P, S, D, T, sparsity), Xt_pca)
        np.save('simulation/result/N{}_P{}_S{}_D{}_T{}_s{}/D_pca.npy'.format(N, P, S, D, T, sparsity), D_pca)
        np.save('simulation/result/N{}_P{}_S{}_D{}_T{}_s{}/F_pca.npy'.format(N, P, S, D, T, sparsity), F_pca)

    if flag[1]=='1': 
        # D_spca = 
        F_spca = spca.components_.T # (P, q)
        np.save('simulation/result/N{}_P{}_S{}_D{}_T{}_s{}/Xt_spca.npy'.format(N, P, S, D, T, sparsity), Xt_spca)
        # np.save('simulation/result/N{}_P{}_S{}_D{}_T{}_s{}/D_spca.npy'.format(N, P, S, D, T, sparsity), D_spca)
        np.save('simulation/result/N{}_P{}_S{}_D{}_T{}_s{}/F_spca.npy'.format(N, P, S, D, T, sparsity), F_spca)

    if flag[2]=='1': 
        D_dpca = dpca.D # (P, q)
        F_dpca = dpca.P # (P, q)
        np.save('simulation/result/N{}_P{}_S{}_D{}_T{}_s{}/Xt_dpca.npy'.format(N, P, S, D, T, sparsity), Xt_dpca)
        np.save('simulation/result/N{}_P{}_S{}_D{}_T{}_s{}/D_dpca.npy'.format(N, P, S, D, T, sparsity), D_dpca)
        np.save('simulation/result/N{}_P{}_S{}_D{}_T{}_s{}/F_dpca.npy'.format(N, P, S, D, T, sparsity), F_dpca)

    if flag[3]=='1': 
        D_l2_dpca = l2_dpca.D # (P, q)
        F_l2_dpca = l2_dpca.P # (P, q)
        np.save('simulation/result/N{}_P{}_S{}_D{}_T{}_s{}/Xt_l2_dpca.npy'.format(N, P, S, D, T, sparsity), Xt_l2_dpca)
        np.save('simulation/result/N{}_P{}_S{}_D{}_T{}_s{}/D_l2_dpca.npy'.format(N, P, S, D, T, sparsity), D_l2_dpca)
        np.save('simulation/result/N{}_P{}_S{}_D{}_T{}_s{}/F_l2_dpca.npy'.format(N, P, S, D, T, sparsity), F_l2_dpca)

    if flag[4]=='1': 
        D_eedpca = eedpca.D # (P, q)
        F_eedpca = eedpca.F # (P, q)
        np.save('simulation/result/N{}_P{}_S{}_D{}_T{}_s{}/Xt_eedpca.npy'.format(N, P, S, D, T, sparsity), Xt_eedpca)
        np.save('simulation/result/N{}_P{}_S{}_D{}_T{}_s{}/D_eedpca.npy'.format(N, P, S, D, T, sparsity), D_eedpca)
        np.save('simulation/result/N{}_P{}_S{}_D{}_T{}_s{}/F_eedpca.npy'.format(N, P, S, D, T, sparsity), F_eedpca)
        
    np.save('simulation/result/N{}_P{}_S{}_D{}_T{}_s{}/timecost.npy'.format(N, P, S, D, T, sparsity), timecost)
    print('Done.')

flags = sys.argv[1] if len(sys.argv)==2 else input('Method flags [11111]: ')
if flags=='': flags='11111'
exp_start = time.time()
settings.traverse(simulation, flags)
# simulation(10, 300, 180, 24, 4, 0.8, flags)
# settings.traverse1(simulation, flags)
print('Total exp time:', time.time() - exp_start)
metrics.metric()