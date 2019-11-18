import numpy as np
import random
import scipy.linalg

def hyper_para():
    N = int(input('Input the dimension N: '))
    K = int(input('Input the number of task K: '))
    P = int(input('Input the number of subsets P: '))
    return N, K, P

def models(n, type=0, a=0.5):
    '''
    give a sparse concentration matrix
    parameter:
        n: matrix dimension
        type: int, 0=random model, 1,2,3,... refers to different models
        a: parameter in model 1

    returns: a concentration matrix
    '''
    n_models = 2 # number of models

    # model 0 (random model)
    if type==0:
        type = np.random.randint(n_models+1)

    # model 1
    if type==1:
        M = np.zeros(shape=(n,n))
        for i in range(n):
            for j in range(n):
                M[i,j] = a ** np.fabs(i-j)
        return M

    # model 2
    if type==2:
        pass

def combine(*submatrices):
    '''
    combine matrices into a blockwise diagonal matrix
    '''
    return scipy.linalg.block_diag(*submatrices)

def divide(k, low, high=None):
    '''
    give a division method for n-dimensional data randomly
    parameter:
        low, high: range of split, [low, high). when high=None, low=0 & high=low
        k: number of piles you want to divide into

    returns: a numpy array, each element is a cutting point (e.g. [2,4] means cut data into [:2],[2:4],[4:] 3 parts)
    '''
    if high==None:
        high = low
        low = 0
    split = np.array(random.sample(range(low,high), k))
    split.sort()
    return split

def generate_lengths(split, n):
    '''
    generates the lengths corresponding to the split. e.g. split=[2,5] in 6-dimension means 3 parts [:2],[2:5],[5:], then the lengths=[2,3,1]
    '''
    lengths = []
    lengths.append(split[0])
    for i in range(len(split)-1):
        lengths.append( split[i+1] - split[i] )
    lengths.append(n-split[len(split)-1])
    return np.array(lengths)

def main():
    N, K, P = hyper_para()
    split = divide(P, low=1, high=N)
    lengths = generate_lengths(split, N)
    #print(lengths)
    
    # generate K tasks
    for i in range(K):
        
    


if __name__ == '__main__':
    main()