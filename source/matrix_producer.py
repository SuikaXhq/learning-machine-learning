import numpy as np
import random
import scipy.linalg

def hyper_para():
    N = int(input('Input the dimension N: '))
    K = int(input('Input the number of task K: '))
    P = int(input('Input the number of subsets P: '))
    return N, K, P

def models(n, type=-1, a=0.5):
    '''
    give a sparse concentration matrix
    parameter:
        n: matrix dimension
        type: int, -1=random model, 0,1,2,3,... refers to different models
        a: parameter in model 1

    returns: a concentration matrix
    '''
    n_models = 1 # number of models

    # model -1 (random model)
    if type==-1:
        type = np.random.randint(n_models)

    # model 0
    if type==0:
        M = np.zeros(shape=(n,n))
        for i in range(n):
            for j in range(n):
                M[i,j] = a ** np.fabs(i-j)
        return M

    # model 1
    if type==1:
        pass

def combine(submatrices_list):
    '''
    combine matrices into a blockwise diagonal matrix
    '''
    result = submatrices_list[0]
    for m in submatrices_list[1:]:
        result = scipy.linalg.block_diag(result, m)
    return result

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

def generater(N, K, P):
    split = divide(P, low=1, high=N)
    #print(split)
    
    # generate K tasks
    split.insert(0, 0)
    split.append(N) # modify the split list to fit the iteration
    results = []
    for i in range(K):
        sub_split_n = [random.randint( 1, (split[i+1]-split[i])//2 if split[i+1]-split[i]>1 else 1 ) for i in range(P)]
        #print(sub_split_n)
        sub_split = []
        for i in range(P):
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


def main():
    N, K, P = hyper_para()
    results = generater(N, K, P)
    print(results)


if __name__ == '__main__':
    main()