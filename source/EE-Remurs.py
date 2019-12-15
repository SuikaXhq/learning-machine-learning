import numpy as np
import time
import os
import re

def hyper_para():
    # ratio = float(input('Input tau/gamma ratio (float): '))
    # lam = float(input('Input lambda:'))
    shape = input('Input the input shape(split by space):')
    return [int(s) for s in shape.split()]#, path

class EE_Remurs():
    def __init__(self, input_shape):
        self.W = np.zeros(shape=input_shape)
        self.input_shape=input_shape
        self.N = len(input_shape)

    def __call__(self, X):
        return self.W * X
    
    def train(self, X_train, y_train, ratio=7.0, lam=3.0, epsl=1.0, ita=None):
        self.ratio = ratio
        self.lam = lam
        self.X_train = X_train
        self.epsl = epsl
        X_flat = X_train.reshape((X_train.shape[0], -1))
        P_flat = X_flat.shape[1]
        W_hat = np.linalg.inv(X_flat.T @ X_flat + epsl*np.identity(P_flat)) @ X_flat.T @ y_train
        W_hat = W_hat.reshape(self.input_shape)
        if not ita:
            ita = [1.0/(self.N+1) for _ in range(self.N+1)]

        # training
        self.Ws = []
        self.Ws.append(self.thresholding(W_hat/ita[0], lam))
        for i in range(1, self.N+1):
            W_i = (W_hat/ita[i]).reshape((self.input_shape[i-1], -1))
            U, S, V = np.linalg.svd(W_i, full_matrices=False)
            S = self.thresholding(S, ratio*lam/self.N)
            #print(S)
            min_m_n = np.min(W_i.shape)
            Sm = np.zeros((min_m_n, min_m_n))
            np.fill_diagonal(Sm, S)
            self.Ws.append((U @ Sm @ V).reshape(self.input_shape))
        self.W = np.average(self.Ws, axis=0)

    def get_weights(self):
        return self.Ws, self.W

    def test(self, X_test, y_test):
        X_test_ = X_test.reshape((-1, np.prod(self.input_shape)))
        W = self.W.reshape((-1))
        loss = np.linalg.norm(X_test_ @ W - y_test)/X_test.shape[0]
        return loss
    
    def thresholding(self, W, t):
        return np.where(np.fabs(W) > t, np.sign(W)*(np.fabs(W)-t), 0)

class Generator():
    def __init__(self, shape=(10,10,10,10)):
        self.shape = shape
        self.N = len(shape)
        self.weights = None
    
    def reshape(self, shape):
        self.shape = shape
        self.N = len(shape)
    
    def reweights(self, weights):
        self.weights = weights

    def generate_weight(self):
        if not self.weights:
            self.weights = (1,)*(self.N+1)
        Ws_origin = []
        Ws_origin.append(np.random.choice([0,1], size=self.shape, p=[0.9,0.1]))
        
        for i in range(self.N):
            a = np.random.normal(loc=0, scale=1, size=self.shape[i]).reshape((-1,1))
            b = np.random.normal(loc=0, scale=1, size=np.prod(self.shape)//self.shape[i]).reshape((-1,1))
            Ws_origin.append((a @ b.T).reshape(self.shape))
        self.Ws = Ws_origin
        self.W = np.average(Ws_origin, axis=0, weights=self.weights)
        return self.Ws, self.W
        
    def generate_data(self, N):
        W_ = self.W.reshape((-1))
    #    print(W_.shape)
        P = W_.shape[0]
        X_shape = [N,P]
        X = np.random.normal(loc=0, scale=10, size=X_shape)
    #    print(X.shape)
        bias = np.random.normal(loc=0, scale=1, size=N)
        y = X@W_ + bias
        return X.reshape([-1,] + self.shape), y

    '''TODO: test, save and load methods
    '''
    def test(self, X_test, y_test):
        pass

    def save(self, M, path):
        pass

    def load(self, path):
        pass

class Best():
    pass

def main():
    '''TODO: new or load choices
    '''
    flag = input('g: generate data, r: run model ')
    if flag =='g':
        shape = hyper_para()
        shape_ = [str(s) for s in shape]
        P = np.prod(shape)
        generator = Generator(shape)
        Ws_origin, W = generator.generate_weight()
        X_train, y_train = generator.generate_data(int(0.05*P))
        X_test, y_test = generator.generate_data(int(0.01*P))
        folder = 'x'.join(shape_)+'/'
        if not os.path.exists(folder):
            os.makedirs(folder)
        path_Ws = 'weights_'+'x'.join(shape_)+'_{}N.csv'.format(len(shape))
        path_W = 'weight_'+'x'.join(shape_)+'_{}N.csv'.format(len(shape))
        path_X_train = 'x_train_'+'x'.join(shape_)+'_{}N_{}samples.csv'.format(len(shape), int(0.05*P))
        path_y_train = 'y_train_'+'x'.join(shape_)+'_{}N_{}samples.csv'.format(len(shape), int(0.05*P))
        path_X_test = 'x_test_'+'x'.join(shape_)+'_{}N_{}samples.csv'.format(len(shape), int(0.01*P))
        path_y_test = 'y_test_'+'x'.join(shape_)+'_{}N_{}samples.csv'.format(len(shape), int(0.01*P))
        np.savetxt(folder+path_Ws, np.array(Ws_origin).reshape((len(shape)+1, -1)))
        np.savetxt(folder+path_W, W.reshape(-1))
        np.savetxt(folder+path_X_train, X_train.reshape((int(0.05*P),-1)))
        np.savetxt(folder+path_y_train, y_train)
        np.savetxt(folder+path_X_test, X_test.reshape((int(0.01*P),-1)))
        np.savetxt(folder+path_y_test, y_test)

    elif flag=='r':
        
        datafolder = input('Data Folder(do not add \'/\'):')
        print('loading files...')
        shape = [int(s) for s in datafolder.split('x')]
        for root, dirs, files in os.walk(datafolder+'/'):
            for file in files:
                if file.startswith('x_train'):
                    X_train = np.loadtxt(datafolder+'/'+file).reshape([-1,]+shape)
                elif file.startswith('y_train'):
                    y_train = np.loadtxt(datafolder+'/'+file)
                elif file.startswith('x_test'):
                    X_test = np.loadtxt(datafolder+'/'+file).reshape([-1,]+shape)
                elif file.startswith('y_test'):
                    y_test = np.loadtxt(datafolder+'/'+file)

        
        model = EE_Remurs(input_shape=shape)
        best = Best()
        best.mse = 1e+5

        
        # cross-validation (1 fold)
        for ratio in range(1, 10):
            for lam in np.arange(3, 7, 0.5):
                print('ratio: %d, lam: %f'%(ratio, lam))
                print('Training...')
                start = time.time()
                model.train(X_train, y_train, ratio=ratio, lam=lam)
                end = time.time()
                timecost = end-start
                mse = model.test(X_test, y_test)
                print('mse: %.4f'%mse)
                print('run time: %.4f'%timecost)
                print('-'*100)
                if mse<best.mse:
                    best.mse = mse
                    best.time = timecost
                    best.ratio = ratio
                    best.lam = lam
                
        print('='*100)
        print('The best parameters are:')
        print('ratio: %d, lam: %f'%(best.ratio, best.lam))
        print('mse: %.4f'%best.mse)
        print('time: %.4f'%best.time)


        
        #Ws_predict, W_predict = model.get_weights()
        #print(W)
        #print(W_predict)
        #print(Ws_predict[0])
        #print(Ws_origin[0])
        #print((W_predict-W)/W)
        #print('mse =', model.test(X_test, y_test))
        #print('run time:', end-start, 'seconds')

if __name__ == '__main__':
    main()