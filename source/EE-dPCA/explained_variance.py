import numpy as np 


def explained_variance(F, X_trans, Xfull, D=None, mXs=None):
    '''Calculate the explained variances.
    Parameters:
    -------
    F: Encoder, shape (P, q)
    X_trans: transformed data, i.e., DX, shape (q, N)
    X_full: full data, i.e., X, shape (P, n_param1, n_param2, ...)
    D: Decoder, shape (P, q)
    mXs: dict of marginalized data, i.e., X_phi, shape (P, n_param1, n_param2, ...)

    Returns:
    -------
    results: dict,
        ['totalVar']: Total Var
        ['totalMarginVar']: Total variance in each marginalization
        ['cumulativeVar']: cumulative variance of the components (%)
        ['componentVar']: variance of each component (%)
        ['marginVar']: variance of each component in each marginalization (%)

    '''
    results = {}
    X = Xfull.reshape([Xfull.shape[0], -1])
    X -= np.mean(X, axis=1)[:, None]

    results['totalVar'] = np.sum(X**2)
    # print('totalVar')
    

    if D is not None and mXs is not None:
        results['totalMarginVar'] = { key: np.sum(mX**2) for key, mX in mXs.items() }
        # print('totalMarginVar')
        if isinstance(D, dict):
            P, q = list(F.values())[0].shape
            results['marginVar'] = [{} for _ in range(len(mXs)*q)]
            results['cumulativeVar'] = np.zeros(len(mXs)*q)
            componentVars = np.zeros(len(mXs)*q)
            F_flat = np.zeros((len(mXs)*q, P))
            D_flat = np.zeros((len(mXs)*q, P))
            for i,margin in enumerate(mXs.keys()):
                for j in range(q):
                    F_flat[j+i*q, :] = F[margin][:, j]
                    D_flat[j+i*q, :] = D[margin][:, j]
                    componentVars[j+i*q] = 100 - np.sum((X - F_flat[j+i*q, :][:, None] @ (D_flat[j+i*q, :][None, :] @ X))**2) / results['totalVar'] * 100
            idx_sorted = np.argsort(-componentVars)
            results['componentVar'] = componentVars[idx_sorted[:q]]
            F_flat = F_flat[idx_sorted[:q], :]
            D_flat = D_flat[idx_sorted[:q], :]
            # print('componentVar')
        else:
            q = F.shape[1]
            results['marginVar'] = [{} for _ in range(q)]
            results['cumulativeVar'] = np.zeros(q)
            results['componentVar'] = np.zeros(q)
            F_flat = F.T
            D_flat = D.T
            for i in range(q):
                results['componentVar'][i] = 100 - np.sum((X - F_flat[i, :][:, None] @ (D_flat[i, :][None, :] @ X))**2) / results['totalVar'] * 100
            # print('componentVar')

        mXs_ = {key:mX.reshape([mX.shape[0], -1]) for key,mX in mXs.items()}
        for i in range(F_flat.shape[0]):
            results['cumulativeVar'][i] = 100 - np.sum((X - F_flat[:i+1, :].T @ (D_flat[:i+1, :] @ X))**2) / results['totalVar'] * 100
            # print('cumulativeVar', i)
            for key, mX_ in mXs_.items():
                results['marginVar'][i][key] = (results['totalMarginVar'][key] - np.sum((mX_ - F_flat[i, :][:, None] @ (D_flat[i, :][None, :] @ mX_)[None, :])**2)) / results['totalVar'] * 100

    else:
        q = F.shape[1]
        results['cumulativeVar'] = np.zeros(q)
        results['componentVar'] = np.zeros(q)
        for i in range(q):
            results['cumulativeVar'][i] = 100 - np.sum((X - F[:, :i+1] @ X_trans[:i+1, :])**2) / results['totalVar'] * 100
            results['componentVar'][i] = 100 - np.sum((X - F[:, i][:, None] @ X_trans[i, :][None, :])**2) / results['totalVar'] * 100

    return results