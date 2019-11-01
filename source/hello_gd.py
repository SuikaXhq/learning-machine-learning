import numpy as np
import matplotlib as mpl
from matplotlib import pyplot as plt


if __name__ == '__main__':
    
    # load data
    data = np.loadtxt('../resource/athlete.csv', delimiter=',')
    x_train = data[...,0]
    y_train = data[...,1]

    plt.plot(x_train, y_train,'ob')
    plt.savefig('origin.png')

    # Train
    
    # a is the parameter
    a = np.random.random((2,))
    t = float(input('Learning rate: ')) # learning rate
    #dr = 1
    epoch = int(input('Epoches: '))
    i = 0
    plt.plot(x_train, a[0]+a[1]*x_train)
    plt.savefig('init.png')
    plt.clf()

    while i < epoch:
        #old_a = a[:] # save data of the old parameter

        d0 = 0 # gradient for a_0
        d1 = 0 # gradient for a_1
        #print('a=',a)
        for j in range(len(x_train)):
            # gradient for a_0:
            dd0 = 2*( a[0] + a[1]*x_train[j] - y_train[j] )
            #print('dd0=',dd0)
            d0 += dd0
            # gradient for a_1
            d1 += dd0 * x_train[j]

        a[0] -= t*d0
        a[1] -= t*d1

        #dr = np.linalg.norm(a-old_a) / np.linalg.norm(a)
        if i % 10 == 0 :
            print(a, i)

        i += 1

    print('最终参数：', a)
    plt.plot(x_train, y_train, 'ob', x_train, a[0]+a[1]*x_train)
    plt.savefig('trained.png')
