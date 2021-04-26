def get_P():
    return [100, 200, 300, 400, 500, 600]

def get_T():
    return 150

def get_S():
    return [6, 12, 18, 24, 30, 36]

def get_D():
    return 4

def get_sparsity():
    return [0, 0.2, 0.4, 0.6, 0.8]

def get_n_trials():
    return 10

def traverse(func, *args, **kwargs):
    n_trials = get_n_trials()
    P = get_P()
    T = get_T()
    S = get_S()
    D = get_D()
    sparsity = get_sparsity()
    for i in range(5):
        func(n_trials, P[2], T, S[0], D, sparsity[i], *args, **kwargs)
    for j in [0,1,3,4,5]:
        func(n_trials, P[j], T, S[0], D, sparsity[4], *args, **kwargs)
    for k in [1,2,3,4,5]:
        func(n_trials, P[2], T, S[k], D, sparsity[4], *args, **kwargs)

def traverse1(func, *args, **kwargs):
    n_trials = get_n_trials()
    P = get_P()
    T = get_T()
    S = get_S()
    D = get_D()
    sparsity = get_sparsity()
    for i in range(5):
        func(n_trials, P[2], T, S[0], D, sparsity[i], *args, **kwargs)
    for j in [0,1,3,4,5]:
        func(n_trials, P[j], T, S[0], D, sparsity[4], *args, **kwargs)
    for k in [1,2,3,4,5]:
        func(n_trials, P[2], T, S[k], D, sparsity[4], *args, **kwargs)
