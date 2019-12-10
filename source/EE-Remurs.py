import numpy as np

def hyper_para():
    ratio = float(input('Input tau/gamma ratio (float): '))
    lam = float(input('Input lambda:'))
    return ratio, lam

class EE_Remurs():
    def __init__(self, input_shape):
        self.W = np.zeros(shape=input_shape)
        self.input_shape=input_shape
        self.N = len(input_shape)

    def call(self, X):
        return self.W * X
    
    def train(self, X_train, y_train, ratio=1.0, lam=0.1, epsl=0.1, ita=None):
        self.ratio = ratio
        self.lam = lam
        self.X_train = X_train
        self.epsl = epsl
        X_flat = np.reshape(X_train, (X_train.shape[0], -1))
        P_flat = X_flat.shape[1]
        W_hat = np.linalg.inv(X_flat.T * X_flat + epsl*np.ones((P_flat, P_flat))) @ X_flat.T @ y_train
        W_hat = W_hat.reshape(self.input_shape)
        if not ita:
            ita = [1.0/(self.N+1) for _ in range(self.N)]

        # training
        Ws = []
        Ws.append(self.thresholding(W_hat/ita[0], lam))
        for i in range(1, self.N+1):
            W_i = W_hat/ita[i]
            U, S, V = np.linalg.svd(W_i)
            S = self.thresholding(S, ratio*lam/self.N)
            Ws.append(U @ S @ V)
        self.W = np.average(Ws, axis=0)
    
    def thresholding(self, W, t):
        return np.where(np.fabs(W) > t, np.sign(W)*(np.fabs(W)-t), 0)

    
    
def generate_weight(shape=(10,10,10,10,10), weights=None):
    N = len(shape)
    if not weights:
        weight = (1,)*(N+1)
    Ws_origin = []
    Ws_origin.append(np.random.choice([0,1], size=shape, p=[0.9,0.1]))
    
    for i in range(N):
        a = np.random.normal(loc=0, scale=1, size=shape[i]).reshape((-1,1))
        b = np.random.normal(loc=0, scale=1, size=np.prod(shape)//shape[i]).reshape((-1,1))
        Ws_origin.append((a @ b.T).reshape(shape))
    return np.average(Ws_origin, axis=0, weights=weights), Ws_origin
    



def generate_data(W):
    bias = np.random.normal()

def main():
    print(generate_weight(shape=(3,4,5), weights=(1,1,1,1)))

if __name__ == '__main__':
    main()