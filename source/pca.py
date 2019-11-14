import tensorflow as tf
import numpy as np
import matplotlib as mpl
from matplotlib import pyplot as plt

def my_plot(N, predicts, K):
    if N == 2:
        for i in range(K):
            plt.plot(np.array(predicts[i])[:,0], np.array(predicts[i])[:,1], 'o')
        plt.show()
#    if N == 3:
#        for i in range(K):
#            plt.plot(np.array(predicts[i])[:,0], np.array(predicts[i])[:,1], np.array(predicts[i])[:,2], 'o')
#        plt.show()
        
def classification(centroids, data, K):
    predicts = {i: [] for i in range(K)} # predicts[i] is an array of i-th class's points
    SSE = 0 # summation of squared error

    for point in data:
        distances = [np.linalg.norm(point - centroid) for centroid in centroids] # distances for all centroids
        min_distance = min(distances)
        SSE += min_distance**2
        predict = distances.index(min(distances)) # the class that this point is predicted to be
        predicts[predict].append(point) # add the point
        
    return predicts, SSE

def k_means(data, N, K, epoch):
    # initalize centroids using K samples
    choices = np.random.choice(1000, K)
    centroids = [data[i] for i in choices]

    # iteration        
    for i in range(epoch):
        # classification
        predicts, SSE = classification(centroids, data, K)

        # SSE
#        print('K={}, Epoch={}, SSE={}'.format(K, i, round(SSE,2)))

        # plot
#        my_plot(N, predicts, K)

        # calculate new centroids
        centroids = [np.mean(predicts[i], axis=0) if predicts[i] else centroids[i] for i in range(K)]
        if i % 10 == 0:
            print('epoch={}, SSE={}'.format(i, SSE))
        
    # final model
    print('-'*50)
    print('The final model:')
    print('{} centroids:'.format(K))
    print(np.array(centroids))
    predicts, SSE = classification(centroids, data, K)
    print('K={}, Epoch={}, SSE={}'.format(K, epoch, round(SSE,2)))
    my_plot(N, predicts, K)
    return centroids, predicts, SSE
    #end k_means()

def PCA(data, energy=0.9):
    # centralize
    data_mean = data.mean(axis=0)
    data_c = data - data_mean

    # covariance matrix
    CovX = np.dot(data_c.T, data_c)

    # select 90% energy vectors
    e, v = np.linalg.eigh(CovX)
    e_i = np.argsort(-e) # sort e (larger first, returns indices)
    e_o = np.array([e[i] for i in e_i]) # larger first e
    pct = 0.0
    n = 0
    while pct < energy:
        n += 1
        pct = e_o[:n].sum() / e_o.sum()
    print('Energy remained:', round(pct,2))
    print('Dimensions selected:', n)

    # projection
    v_o = np.array([v[i] for i in e_i[:n]]) # largest n elegenvectors
    Z = np.dot(data_c, v_o.T)
    
    return Z

# hyper-parameter
N = 200
K = 100
print('N =',N)
print('K =',K)

# generate random data
data = np.random.rand(500000, N)
locs = np.random.uniform(low=0, high=10, size=(4,N))
for i in range(4):
    data = np.append(data, np.random.normal(loc=locs[i], size=(100000, N)), axis=0)
print(data)

# k means
Z = PCA(data)
print(Z)
k_means(Z, N, K, epoch=200)