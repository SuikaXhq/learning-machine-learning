#!/usr/bin/env python
# coding: utf-8

# In[20]:


import tensorflow as tf
import numpy as np
import matplotlib as mpl
from matplotlib import pyplot as plt

def my_plot(N, predicts):
    if N == 2:
        for i in range(K):
            plt.plot(np.array(predicts[i])[:,0], np.array(predicts[i])[:,1], 'o')
        plt.show()
    if N == 3:
        for i in range(K):
            plt.plot(np.array(predicts[i])[:,0], np.array(predicts[i])[:,1], np.array(predicts[i])[:,2], 'o')
        plt.show()
        
def classification(centroids, data):
    predicts = {i: [] for i in range(K)} # predicts[i] is an array of i-th class's points
    SSE = 0 # summation of squared error

    for point in data:
        distances = [np.linalg.norm(point - centroid) for centroid in centroids] # distances for all centroids
        min_distance = min(distances)
        SSE += min_distance**2
        predict = distances.index(min(distances)) # the class that this point is predicted to be
        predicts[predict].append(point) # add the point
        
    return predicts, SSE

# input hyper-parameter N
N = int(input('Dimension N: '))
max_K = int(input('Maximum K: '))
epoch = int(input('Training epoches: '))

# generate random data
data = np.random.rand(100, N)
locs = np.random.uniform(low=0, high=10, size=(4,N))
sizes[3] = 900-np.sum(sizes)+sizes[3]
for i in range(4):
    data = np.append(data, np.random.normal(loc=locs[i], size=(225, N)), axis=0)
plt.plot(data[:,0], data[:,1], 'o')
plt.show()

# initialize
SSE_list = [0]*(max_K-1)

# k-means
for K in range(2, max_K+1):
    # initalize centroids using K samples
    choices = np.random.choice(1000, K)
    centroids = [data[i] for i in choices]

    # iteration        
    for i in range(epoch):
        # classification
        predicts, SSE = classification(centroids, data)
        
        # SSE
#        print('K={}, Epoch={}, SSE={}'.format(K, i, round(SSE,2)))
        
        # plot
#        my_plot(N, predicts)

        # calculate new centroids
        centroids = [np.mean(predicts[i], axis=0) if predicts[i] else centroids[i] for i in range(K)]
        
    # final model
    print('-'*50)
    print('The final model:')
    print('{} centroids:'.format(K))
    print(np.array(centroids))
    predicts, SSE = classification(centroids, data)
    print('K={}, Epoch={}, SSE={}'.format(K, epoch, round(SSE,2)))
    my_plot(N, predicts)
    SSE_list[K-2] = SSE
    # end for

# draw K-SSE figure
plt.plot(range(2, max_K+1), SSE_list)
plt.show()

