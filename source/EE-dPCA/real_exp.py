import numpy as np
from dPCA import dPCA
from ee_dpca import EE_dPCA
from sklearn.decomposition import PCA, SparsePCA
import os
import time
import metrics
import sys
import scipy.io as scio

flag = sys.argv[1] if len(sys.argv)==2 else input('Method flags [11111]: ')
if flag=='': flag='11111'
X = scio.loadmat('real/filteredFiringRatesAvg.mat')['filteredFiringRatesAvg']
trialX = X.reshape((1,) + X.shape)
X_ = X.reshape((X.shape[0], -1))
timecost = np.zeros(5)
q = 20
print('data shape:', X.shape)
print('q:', q)
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
if not os.path.exists('real/result/'): os.mkdir('real/result/')
if flag[0]=='1':
    D_pca = pca.components_.T # (P, q)
    F_pca = pca.components_.T # (P, q)
    np.save('real/result/Xt_pca.npy', Xt_pca)
    np.save('real/result/D_pca.npy', D_pca)
    np.save('real/result/F_pca.npy', F_pca)

if flag[1]=='1': 
    # D_spca = 
    F_spca = spca.components_.T # (P, q)
    np.save('real/result/Xt_spca.npy', Xt_spca)
    # np.save('real/result/D_spca.npy', D_spca)
    np.save('real/result/F_spca.npy', F_spca)

if flag[2]=='1': 
    D_dpca = dpca.D # (P, q)
    F_dpca = dpca.P # (P, q)
    np.save('real/result/Xt_dpca.npy', Xt_dpca)
    np.save('real/result/D_dpca.npy', D_dpca)
    np.save('real/result/F_dpca.npy', F_dpca)

if flag[3]=='1': 
    D_l2_dpca = l2_dpca.D # (P, q)
    F_l2_dpca = l2_dpca.P # (P, q)
    np.save('real/result/Xt_l2_dpca.npy', Xt_l2_dpca)
    np.save('real/result/D_l2_dpca.npy', D_l2_dpca)
    np.save('real/result/F_l2_dpca.npy', F_l2_dpca)

if flag[4]=='1': 
    D_eedpca = eedpca.D # (P, q)
    F_eedpca = eedpca.F # (P, q)
    np.save('real/result/Xt_eedpca.npy', Xt_eedpca)
    np.save('real/result/D_eedpca.npy', D_eedpca)
    np.save('real/result/F_eedpca.npy', F_eedpca)
    
np.save('real/result/timecost.npy', timecost)
print('Done.')