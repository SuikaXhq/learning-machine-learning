import numpy as np
import time

def hyper_para():
    '''TODO: path
    '''
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
    
    def train(self, X_train, y_train, ratio=8.0, lam=2, epsl=3.0, ita=None):
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
            U, S, V = np.linalg.svd(W_i)
            S = self.thresholding(S, ratio*lam/self.N)
            #print(S)
            Sm = np.zeros(W_i.shape)
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

    def save(self, path):
        pass

    def load(self, path):
        pass
        

def main():
    '''TODO: new or load choices
    '''
    shape = hyper_para()
    P = np.prod(shape)
    generator = Generator(shape)
    Ws_origin, W = generator.generate_weight()
#    generator.save(path)
#    print(W.shape)
    X_train, y_train = generator.generate_data(int(0.8*P))
    X_test, y_test = generator.generate_data(int(0.5*P))
    model = EE_Remurs(input_shape=shape)
    start = time.time()
    model.train(X_train, y_train)
    end = time.time()
    Ws_predict, W_predict = model.get_weights()
    #print(W)
    #print(W_predict)
    #print(Ws_predict[0])
    #print(Ws_origin[0])
    #print((W_predict-W)/W)
    print(model.test(X_test, y_test))
    print(end-start, 'seconds')

if __name__ == '__main__':
    main()