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

    # model: y = a[0] + a[1] * x
    # a is the parameter to be trained
    a = np.random.random((2,))
    t = float(input('Learning rate: ')) # learning rate
    epoch = int(input('Epoches: '))
    i = 0
    plt.plot(x_train, a[0]+a[1]*x_train)
    plt.savefig('init.png')
    plt.clf()

    while i < epoch:
        d0 = 0 # gradient for a_0
        d1 = 0 # gradient for a_1
        for j in range(len(x_train)):
            # gradient for a_0:
            dd0 = 2*( a[0] + a[1]*x_train[j] - y_train[j] )
            d0 += dd0
            # gradient for a_1
            d1 += dd0 * x_train[j]

        a[0] -= t*d0
        a[1] -= t*d1

        if i % 10 == 0 :
            print(a, i)

        i += 1

    print('Last parameter:', a)
    x_test = np.array([47, 55])
    y = a[0] + a[1]*x_train
    y_test = a[0] + a[1]*x_test
    plt.plot(x_train, y_train, 'ob', x_train, y, 'b-', x_test, y_test, 'b-')
    plt.plot(x_test, y_test, 'or')
    for i in range(2):
        plt.annotate(s='({}, {})'.format(round(x_test[i], 2), round(y_test[i], 2)),
                xy=(x_test[i], y_test[i]),
                xytext=(x_test[i]-2, y_test[i]+2))
    plt.savefig('trained.png')
