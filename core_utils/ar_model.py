import numpy as np

class ar_model:
    
    def __init__(self,n_lags=25,both_dirs=True):
        self.n_lags = n_lags
        self.both_dirs = both_dirs

    def fit(self,x):
        if not(x.flatten().shape[0]==x.shape[0]):
            raise TypeError('"x" must be a flat numpy vector')
        N = len(x)
        
        x_mean = np.mean(x)
        x = x-x_mean
        
        in_x = np.zeros((N-self.n_lags,self.n_lags))
        out_x = np.array(x[self.n_lags:])
        for i in range(N-self.n_lags):
            in_x[i,:] = x[i:(i+self.n_lags)]        
        beta_for = np.linalg.inv(in_x.T.dot(in_x)).dot(in_x.T).dot(out_x)
        for_pred = in_x.dot(beta_for)
        
        x_pred = np.zeros(x.shape)
        if self.both_dirs:
            in_x = np.zeros((N-self.n_lags,self.n_lags))
            out_x = np.array(x[:-self.n_lags])
            for i in range(self.n_lags,N):
                in_x[i-self.n_lags,:] = x[(i-self.n_lags+1):(i+1)]
            beta_back = np.linalg.inv(in_x.T.dot(in_x)).dot(in_x.T).dot(out_x)
            back_pred = in_x.dot(beta_back)
            
            x_pred[-self.n_lags:] = for_pred[-self.n_lags:]
            x_pred[:self.n_lags] = back_pred[:self.n_lags]
            x_pred[self.n_lags:-self.n_lags] = (for_pred[:-self.n_lags] + \
                back_pred[self.n_lags:]) / 2
        else:
            x_pred[:self.n_lags] = x[:self.n_lags]
            x_pred[self.n_lags:] = for_pred
        
        x_auto = x_pred + x_mean
        return x_auto
        
        
    