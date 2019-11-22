import numpy as np
import random
import scipy.linalg
#from matplotlib import pyplot as plt

def hyper_para():
    '''
    returns P, K, S, N, is_equal
    '''
    P = int(input('Input the dimension of feature P: '))
    K = int(input('Input the number of task K: '))
    S = int(input('Input the number of subsets S: '))
    N = int(input('Input the number of samples for each task N: '))
    folder = input('Input the folder name you wish to save: ')
    is_equal = False
    while True:
        flag = input(r'Split equally?(y/n, False for default): ')
        if flag == 'y':
            is_equal = True
            break
        elif flag == 'n' or flag == '':
            break

    return P, K, S, N, is_equal, folder

def models(p, type=-1, a=0.7):
    '''
    give a sparse concentration matrix
    parameter:
        p: matrix dimension
        type: int, -1=random model, 0,1,2,3,... refers to different models
        a: parameter in model 1

    returns: a concentration matrix
    '''
    n_models = 4 # number of models

    # model -1 (random model)
    if type==-1:
        type = np.random.randint(n_models)
        print('model {} used, '.format(type+1), end='')

    M = np.zeros(shape=(p,p))

    # model 0
    if type==0:
        for i in range(p):
            for j in range(i, p):
                M[j,i] = M[i,j] = a ** abs(i-j)

    # model 1
    elif type==1:
        for i in range(p):
            for j in range(i,p):
                l = abs(i-j)
                m = 0
                if l==0:
                    m = 1
                elif l==1:
                    m = 0.4
                elif l==2 or l==3:
                    m = 0.2
                elif l==4:
                    m = 0.1
                M[j,i] = M[i,j] = m

    # model 2
    elif type==2:
        for i in range(p):
            for j in range(i,p):
                if i==j:
                    M[i,j]=1
                else:
                    M[j,i] = M[i,j] = np.random.choice([0, 0.1], p=[0.9, 0.1])
    
    # model 3
    elif type==3:
        for i in range(p):
            for j in range(i,p):
                if i==j:
                    M[i,j]=1
                else:
                    M[j,i] = M[i,j] = np.random.choice([0, 0.1], p=[0.5, 0.5])
    
    return M

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

def generate_in_cov(P, K, S, equal_split=False, model=-1):
    '''
    generates inverse covariance matrix with PxP dimension, K tasks and S subsets of features.
    if equal_split=True, the subsets are equally divided.
    parameter model: -1=random 0,1,2,...=models.type
    '''

    if equal_split == True:
        # equally spliting
        print('Equally split:')
        split = list(range(0, P+1, P//S))
        split.pop()
        split.append(P)
        print('split:', split)
        
    else:
        # inequally spliting
        print('Randomly split:')
        split = divide(S, low=1, high=P)
        split.insert(0, 0)
        split.append(P) # modify the split list to fit the iteration
        print('split:', split)
    
#    print(split)
    
    # generate K tasks
    results = []
    for i in range(K):
        print('generating task{}'.format(i+1))
        sub_split_n = [random.randint( 1, (split[i+1]-split[i])//2 if split[i+1]-split[i]>1 else 1 ) for i in range(S)]
        #print(sub_split_n)
        sub_split = []
        for i in range(S):
            sub_split.extend(divide(sub_split_n[i], low=split[i]+1, high=split[i+1]))
        final_split = sub_split
        final_split.extend(split)
        final_split = np.array(final_split)
        final_split.sort()
        print('the unique split is', final_split)

        # generate sub-blocks
        sub_blocks = []
        for i in range(len(final_split)-1):
            sub_blocks.append(models(final_split[i+1]-final_split[i], type=model))
            print()
        
        # combine and output
        results.append(combine(sub_blocks))
        print('finished.')
    
    return np.array(results)

def generate_data(Omega, n):
    '''
    generate sample by concentration matrix Omega.
    '''
    Sigma = np.linalg.inv(Omega)
    p = len(Sigma)
    return np.random.multivariate_normal(np.zeros(p), Sigma, size=n)

def generate(P, K, S, N, equal_split=False, folder='default'):
    '''
    generate both inverse covariance matrix and corresponding data, and save as *.csv
    '''
    print('P={}, K={}, S={}, N={}, {}'.format(P, K, S, N, 'equally split' if equal_split else 'randomly split'))
    results = generate_in_cov(P, K, S, equal_split=equal_split)
    for i in range(len(results)):
        print('saving precision matrix of task{}...'.format(i+1))
        pm_path = 'resource/matrix_producer/{}/inverse/inverse_{}vars_{}samples_{}S_task{}.csv'.format(folder, P, N, S, i+1)
        np.savetxt(pm_path, results[i], delimiter=',')
        print('saved in {}.'.format(pm_path))
#        plt.imshow(results[i], cmap='gray')
#        plt.show()
        print('saving data of task{}...'.format(i+1))
        data_path = 'resource/matrix_producer/{}/data/data_{}vars_{}samples_{}S_task{}.csv'.format(folder, P, N, S, i+1)
        np.savetxt(data_path, generate_data(results[i], N), delimiter=',')
        print('saved in {}.'.format(data_path))
        


def main():
    P, K, S, N, is_equal, folder = hyper_para()
    generate(P, K, S, N, is_equal, folder)
    

if __name__ == '__main__':
    main()
