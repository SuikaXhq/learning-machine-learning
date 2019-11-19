import numpy as np
import random
import scipy.linalg
#from matplotlib import pyplot as plt

def hyper_para():
    P = int(input('Input the dimension of feature P: '))
    K = int(input('Input the number of task K: '))
    S = int(input('Input the number of subsets S: '))
    N = int(input('Input the number of samples for each task N: '))
    return P, K, S, N

def models(p, type=-1, a=0.5):
    '''
    give a sparse concentration matrix
    parameter:
        p: matrix dimension
        type: int, -1=random model, 0,1,2,3,... refers to different models
        a: parameter in model 1

    returns: a concentration matrix
    '''
    n_models = 2 # number of models

    # model -1 (random model)
    if type==-1:
        type = np.random.randint(n_models)

    # model 0
    if type==0:
        M = np.zeros(shape=(p,p))
        for i in range(p):
            for j in range(p):
                M[i,j] = a ** abs(i-j)
        return M

    # model 1
    if type==1:
        for i in range(p):
            for j in range(p):
                l = abs(i-j)
                if l==0:
                    m = 1
                elif l==1:
                    m = 0.4
                elif l==2 or l==3:
                    m = 0.2
                elif l==4:
                    m = 0.1
                else:
                    m = 0
                M[i,j] = m

    # model 2
    if type==2:
        pass

def combine(submatrices_list):
    '''
    combine matrices into a blockwise diagonal matrix
    '''
    return scipy.linalg.block_diag(*submatrices_list)

def divide(k, low, high=None):
    '''
    give a division method for n-dimensional data randomly
    parameter:
        low, high: range of split, [low, high). when high=None, low=0 & high=low
        k: number of piles you want to divide into

    returns: a list, each element is a cutting point (e.g. [2,4] means cut data into [:2],[2:4],[4:] 3 parts)
    '''
    if high==None:
        high = low
        low = 0
    split = random.sample(range(low,high), k-1)
    split.sort()
    return split

def generate_in_cov(P, K, S, equal_split=False):
    '''
    generates inverse covariance matrix with PxP dimension, K tasks and S subsets of features. if equal_split=True, the subsets are equally divided.
    '''

    if equal_split == True:
        # equally spliting
        split = list(range(0, P+1, P//S))
        split.pop()
        split.append(P)
        
    else:
        # inequally spliting
        split = divide(S, low=1, high=P)
        split.insert(0, 0)
        split.append(P) # modify the split list to fit the iteration
    #print(split)
    
    # generate K tasks
    results = []
    for i in range(K):
        sub_split_n = [random.randint( 1, (split[i+1]-split[i])//2 if split[i+1]-split[i]>1 else 1 ) for i in range(S)]
        #print(sub_split_n)
        sub_split = []
        for i in range(S):
            sub_split.extend(divide(sub_split_n[i], low=split[i]+1, high=split[i+1]))
        final_split = sub_split
        final_split.extend(split)
        final_split = np.array(final_split)
        final_split.sort()
        #print(final_split)

        # generate sub-blocks
        sub_blocks = []
        for i in range(len(final_split)-1):
            sub_blocks.append(models(final_split[i+1]-final_split[i]))
        
        # combine and output
        results.append(combine(sub_blocks))
    
    return np.array(results)

def generate_data(Omega, n):
    '''
    generate sample by concentration matrix Omega.
    '''
    Sigma = np.linalg.inv(Omega)
    p = len(Sigma)
    return np.random.multivariate_normal(np.zeros(p), Sigma, size=n)


def main():
    P, K, S, N = hyper_para()
    results = generate_in_cov(P, K, S)
    for i in range(len(results)):
        np.savetxt('conc{}.csv'.format(i), results[i], delimiter=',')
#        plt.imshow(results[i], cmap='gray')
#        plt.show()
        np.savetxt('data{}.csv'.format(i), generate_data(results[i], N), delimiter=',')


if __name__ == '__main__':
    main()