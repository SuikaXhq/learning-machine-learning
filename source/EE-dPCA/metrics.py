import numpy as np
import pandas as pd
import settings
from explained_variance import explained_variance
import ee_dpca
import os



def metric(flags='111'):
    print('Calculating metrices...')
    N = settings.get_n_trials()
    P = settings.get_P()
    T = settings.get_T()
    S = settings.get_S()
    D = settings.get_D()
    sparsity = settings.get_sparsity()
    names = ['PCA', 'SPCA', 'dPCA', 'L2_dPCA', 'EE-dPCA']

    # collect time
    if flags[0]=='1':
        time = {name: {'sparse': [], 'P': [], 'S': []} for name in names}
        for i in range(len(sparsity)):
            time_single = np.load('simulation/result/N{}_P{}_S{}_D{}_T{}_s{}/timecost.npy'.format(N, P[2], S[0], D, T, sparsity[i]))
            for j in range(len(names)):
                time[names[j]]['sparse'].append(time_single[j])

        for i in range(len(P)):
            time_single = np.load('simulation/result/N{}_P{}_S{}_D{}_T{}_s{}/timecost.npy'.format(N, P[i], S[0], D, T, sparsity[4]))
            for j in range(len(names)):
                time[names[j]]['P'].append(time_single[j])

        for i in range(len(S)):
            time_single = np.load('simulation/result/N{}_P{}_S{}_D{}_T{}_s{}/timecost.npy'.format(N, P[2], S[i], D, T, sparsity[4]))
            for j in range(len(names)):
                time[names[j]]['S'].append(time_single[j])

        np.save('simulation/metrics/time.npy', time)


    # collect TP/TN/FP/FN
    if flags[1]=='1':
        df_s_dpca = pd.DataFrame(columns=['N', 'P', 'S', 'D', 'T', 'sparsity', 'TP', 'TN', 'FP', 'FN'])
        df_d_dpca = pd.DataFrame(columns=['N', 'P', 'S', 'D', 'T', 'sparsity', 'TP', 'TN', 'FP', 'FN'])
        df_t_dpca = pd.DataFrame(columns=['N', 'P', 'S', 'D', 'T', 'sparsity', 'TP', 'TN', 'FP', 'FN'])

        df_s_l2dpca = pd.DataFrame(columns=['N', 'P', 'S', 'D', 'T', 'sparsity', 'TP', 'TN', 'FP', 'FN'])
        df_d_l2dpca = pd.DataFrame(columns=['N', 'P', 'S', 'D', 'T', 'sparsity', 'TP', 'TN', 'FP', 'FN'])
        df_t_l2dpca = pd.DataFrame(columns=['N', 'P', 'S', 'D', 'T', 'sparsity', 'TP', 'TN', 'FP', 'FN'])

        df_s_eedpca = pd.DataFrame(columns=['N', 'P', 'S', 'D', 'T', 'sparsity', 'TP', 'TN', 'FP', 'FN'])
        df_d_eedpca = pd.DataFrame(columns=['N', 'P', 'S', 'D', 'T', 'sparsity', 'TP', 'TN', 'FP', 'FN'])
        df_t_eedpca = pd.DataFrame(columns=['N', 'P', 'S', 'D', 'T', 'sparsity', 'TP', 'TN', 'FP', 'FN'])

        def classification_metric(N, P, T, S, D, sparsity, method, df_s, df_d, df_t):
            def compare(GT, pred):
                Add = pred + GT
                Sub = pred - GT

                TP = np.sum(Add==2)
                TN = np.sum(Add==0)
                FP = np.sum(Sub==1)
                FN = np.sum(Sub==-1)
                
                return TP, TN, FP, FN

            gt_s = (np.load('simulation/data/N{}_P{}_S{}_D{}_T{}_s{}/X_s.npy'.format(N, P, S, D, T, sparsity))==0).astype(np.int8)
            gt_d = (np.load('simulation/data/N{}_P{}_S{}_D{}_T{}_s{}/X_d.npy'.format(N, P, S, D, T, sparsity))==0).astype(np.int8)
            gt_t = (np.load('simulation/data/N{}_P{}_S{}_D{}_T{}_s{}/X_t.npy'.format(N, P, S, D, T, sparsity))==0).astype(np.int8)

            Xt = np.load('simulation/result/N{}_P{}_S{}_D{}_T{}_s{}/Xt_{}.npy'.format(N, P, S, D, T, sparsity, method), allow_pickle=True).item()
            F = np.load('simulation/result/N{}_P{}_S{}_D{}_T{}_s{}/F_{}.npy'.format(N, P, S, D, T, sparsity, method), allow_pickle=True).item()

            # S
            pred_s = ( np.abs(( F['s'] @ (Xt['s'].reshape( (Xt['s'].shape[0], -1) ) ) ).reshape( (F['s'].shape[0], ) + Xt['s'].shape[1:] ))<1e-2).astype(np.int8)
            TP_s, TN_s, FP_s, FN_s = compare(gt_s, pred_s)
            df_s.loc[df_s.shape[0]] = {'N': N, 'P':P, 'S':S, 'D':D, 'T':T, 'sparsity':sparsity, 'TP':TP_s, 'TN':TN_s, 'FP':FP_s, 'FN':FN_s}

            # D
            pred_d = ( np.abs(( F['d'] @ (Xt['d'].reshape( (Xt['d'].shape[0], -1) ) ) ).reshape( (F['d'].shape[0], ) + Xt['d'].shape[1:] ))<1e-2).astype(np.int8)
            TP_d, TN_d, FP_d, FN_d = compare(gt_d, pred_d)
            df_d.loc[df_d.shape[0]] = {'N': N, 'P':P, 'S':S, 'D':D, 'T':T, 'sparsity':sparsity, 'TP':TP_d, 'TN':TN_d, 'FP':FP_d, 'FN':FN_d}

            # T
            pred_t = ( np.abs(( F['t'] @ (Xt['t'].reshape( (Xt['t'].shape[0], -1) ) ) ).reshape( (F['t'].shape[0], ) + Xt['t'].shape[1:] ))<1e-2).astype(np.int8)
            TP_t, TN_t, FP_t, FN_t = compare(gt_t, pred_t)
            df_t.loc[df_t.shape[0]] = {'N': N, 'P':P, 'S':S, 'D':D, 'T':T, 'sparsity':sparsity, 'TP':TP_t, 'TN':TN_t, 'FP':FP_t, 'FN':FN_t}
        
        settings.traverse(classification_metric, method='dpca', df_s=df_s_dpca, df_d=df_d_dpca, df_t=df_t_dpca)
        settings.traverse(classification_metric, method='l2_dpca', df_s=df_s_l2dpca, df_d=df_d_l2dpca, df_t=df_t_l2dpca)
        settings.traverse(classification_metric, method='eedpca', df_s=df_s_eedpca, df_d=df_d_eedpca, df_t=df_t_eedpca)

        df_s_dpca.to_csv('simulation/metrics/S_dPCA.csv')
        df_d_dpca.to_csv('simulation/metrics/D_dPCA.csv')
        df_t_dpca.to_csv('simulation/metrics/T_dPCA.csv')

        df_s_l2dpca.to_csv('simulation/metrics/S_L2_dPCA.csv')
        df_d_l2dpca.to_csv('simulation/metrics/D_L2_dPCA.csv')
        df_t_l2dpca.to_csv('simulation/metrics/T_L2_dPCA.csv')

        df_s_eedpca.to_csv('simulation/metrics/S_EE-dPCA.csv')
        df_d_eedpca.to_csv('simulation/metrics/D_EE-dPCA.csv')
        df_t_eedpca.to_csv('simulation/metrics/T_EE-dPCA.csv')

    # collect EVs
    if flags[2]=='1':
        def calculate_EV(N, P, T, S, D, sparsity):
            trialX = np.load('simulation/data/N{}_P{}_S{}_D{}_T{}_s{}/Data.npy'.format(N, P, S, D, T, sparsity))
            X = np.mean(trialX, axis=0)
            mXs = {}
            mXs['s'] = np.load('simulation/data/N{}_P{}_S{}_D{}_T{}_s{}/X_s.npy'.format(N, P, S, D, T, sparsity))
            mXs['d'] = np.load('simulation/data/N{}_P{}_S{}_D{}_T{}_s{}/X_d.npy'.format(N, P, S, D, T, sparsity))
            mXs['t'] = np.load('simulation/data/N{}_P{}_S{}_D{}_T{}_s{}/X_t.npy'.format(N, P, S, D, T, sparsity))

            Xt_pca = np.load('simulation/result/N{}_P{}_S{}_D{}_T{}_s{}/Xt_pca.npy'.format(N, P, S, D, T, sparsity), allow_pickle=True).T
            Xt_dpca = np.load('simulation/result/N{}_P{}_S{}_D{}_T{}_s{}/Xt_dpca.npy'.format(N, P, S, D, T, sparsity), allow_pickle=True).item()
            Xt_l2dpca = np.load('simulation/result/N{}_P{}_S{}_D{}_T{}_s{}/Xt_l2_dpca.npy'.format(N, P, S, D, T, sparsity), allow_pickle=True).item()
            Xt_spca = np.load('simulation/result/N{}_P{}_S{}_D{}_T{}_s{}/Xt_spca.npy'.format(N, P, S, D, T, sparsity), allow_pickle=True).T
            Xt_eedpca = np.load('simulation/result/N{}_P{}_S{}_D{}_T{}_s{}/Xt_eedpca.npy'.format(N, P, S, D, T, sparsity), allow_pickle=True).item()

            D_pca = np.load('simulation/result/N{}_P{}_S{}_D{}_T{}_s{}/D_pca.npy'.format(N, P, S, D, T, sparsity), allow_pickle=True)
            D_dpca = np.load('simulation/result/N{}_P{}_S{}_D{}_T{}_s{}/D_dpca.npy'.format(N, P, S, D, T, sparsity), allow_pickle=True).item()
            D_l2dpca = np.load('simulation/result/N{}_P{}_S{}_D{}_T{}_s{}/D_l2_dpca.npy'.format(N, P, S, D, T, sparsity), allow_pickle=True).item()
            # D_spca = np.load('simulation/result/N{}_P{}_S{}_D{}_T{}_s{}/D_spca.npy'.format(N, P, S, D, T, sparsity))
            D_eedpca = np.load('simulation/result/N{}_P{}_S{}_D{}_T{}_s{}/D_eedpca.npy'.format(N, P, S, D, T, sparsity), allow_pickle=True).item()

            F_pca = np.load('simulation/result/N{}_P{}_S{}_D{}_T{}_s{}/F_pca.npy'.format(N, P, S, D, T, sparsity), allow_pickle=True)
            F_dpca = np.load('simulation/result/N{}_P{}_S{}_D{}_T{}_s{}/F_dpca.npy'.format(N, P, S, D, T, sparsity), allow_pickle=True).item()
            F_l2dpca = np.load('simulation/result/N{}_P{}_S{}_D{}_T{}_s{}/F_l2_dpca.npy'.format(N, P, S, D, T, sparsity), allow_pickle=True).item()
            F_spca = np.load('simulation/result/N{}_P{}_S{}_D{}_T{}_s{}/F_spca.npy'.format(N, P, S, D, T, sparsity), allow_pickle=True)
            F_eedpca = np.load('simulation/result/N{}_P{}_S{}_D{}_T{}_s{}/F_eedpca.npy'.format(N, P, S, D, T, sparsity), allow_pickle=True).item()

            result_pca = explained_variance(F_pca, Xt_pca, X, D=D_pca, mXs=mXs)
            result_dpca = explained_variance(F_dpca, Xt_dpca, X, D=D_dpca, mXs=mXs)
            result_l2dpca = explained_variance(F_l2dpca, Xt_l2dpca, X, D=D_l2dpca, mXs=mXs)
            result_spca = explained_variance(F_spca, Xt_spca, X)
            result_eedpca = explained_variance(F_eedpca, Xt_eedpca, X, D=D_eedpca, mXs=mXs)

            if not os.path.exists('simulation/metrics/N{}_P{}_S{}_D{}_T{}_s{}'.format(N, P, S, D, T, sparsity)): os.mkdir('simulation/metrics/N{}_P{}_S{}_D{}_T{}_s{}'.format(N, P, S, D, T, sparsity))
            np.save('simulation/metrics/N{}_P{}_S{}_D{}_T{}_s{}/EV_pca.npy'.format(N, P, S, D, T, sparsity), result_pca)
            np.save('simulation/metrics/N{}_P{}_S{}_D{}_T{}_s{}/EV_dpca.npy'.format(N, P, S, D, T, sparsity), result_dpca)
            np.save('simulation/metrics/N{}_P{}_S{}_D{}_T{}_s{}/EV_l2dpca.npy'.format(N, P, S, D, T, sparsity), result_l2dpca)
            np.save('simulation/metrics/N{}_P{}_S{}_D{}_T{}_s{}/EV_spca.npy'.format(N, P, S, D, T, sparsity), result_spca)
            np.save('simulation/metrics/N{}_P{}_S{}_D{}_T{}_s{}/EV_eedpca.npy'.format(N, P, S, D, T, sparsity), result_eedpca)

        settings.traverse(calculate_EV)
    print('Done.')


if __name__ == '__main__':
    flags = input('Type in flags for matric collecting ([111]): ')
    if flags=='': flags='111'
    metric(flags)