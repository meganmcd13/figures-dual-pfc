import numpy as np
import sklearn.model_selection as ms

class bayes_classifier:
    

    def __init__(self,dist_type='diag_gauss',prior_type='equal'):
        # set prior type and minimum variance threshold
        self.prior_type=prior_type
        
        # set conditional distribution type
        if dist_type!='full_gauss' and dist_type!='diag_gauss' and \
        dist_type!='shared_full' and dist_type!='shared_diag' and \
        dist_type!='poisson':
            raise ValueError('Incorrect "dist_type"')
        else:
            self.dist_type = dist_type
            

    def train(self,X,y):
        # some useful class properties
        self.class_labels = np.unique(y)
        self.n_classes = len(self.class_labels)
        N, D = X.shape
        
        params = []
        for i in range(self.n_classes):
            # set prior
            if self.prior_type=='equal':
                prior_val = 1/self.n_classes
            else:
                prior_val = sum(y==self.class_labels[i])/N

            # set conditional mean and covariance
            curr_dat = X[y==self.class_labels[i],:]
            mu = curr_dat.mean(axis=0)
            sigma = np.cov(curr_dat.T,bias=True)

            # deal with diagonal covariance or poisson distribution
            if self.dist_type=='diag_gauss' or self.dist_type=='shared_diag':
                sigma = np.diag(np.diag(sigma))
            elif self.dist_type=='poisson':
                sigma = None

            # create parameter dict
            curr_params = {
                'class_label': self.class_labels[i],
                'pi': prior_val,
                'mu': mu,
                'sigma': sigma
            }
            params.append(curr_params)

        # deal with shared covariance
        if self.dist_type=='shared_full' or self.dist_type=='shared_diag':
            sigma = np.zeros(sigma.shape)
            for i in range(self.n_classes):
                sigma = sigma + (params[i]['sigma']*sum(y==self.class_labels[i])/N)
            for i in range(self.n_classes):
                params[i]['sigma'] = sigma

        self.params = params


    def predict(self,X):
        N, D = X.shape
        params = self.params

        log_post = np.zeros((N,self.n_classes))
        if self.dist_type=='poisson':
            # prediction for poisson conditional distribution
            for i in range(self.n_classes):
                lamb = params[i]['mu']
                pi = params[i]['pi']
                log_post[:,i] = X.dot(np.log(lamb.T)) - np.sum(lamb) + np.log(pi)
        else:
            # prediction for gaussian conditional distribution
            for i in range(self.n_classes):
                mu = params[i]['mu']
                sigma = params[i]['sigma']
                cx = X - mu
                part1 = -1/2 * np.diag(cx.dot(np.linalg.inv(sigma)).dot(cx.T))
                sign,logdet = np.linalg.slogdet(sigma)
                part2 = -1/2 * logdet
                part3 = -D/2 * np.log(2*np.pi)
                part4 = np.log(params[i]['pi'])
                log_post[:,i] = part1 + part2 + part3 + part4

        return self.class_labels[np.argmax(log_post,axis=1)]


    def accuracy(self,X,y):        
        yhat = self.predict(X)
        return np.mean(y==yhat)


    def crossvalidate(self,X,y,n_folds=10,verbose=True,rand_seed=None):
        N,D = X.shape

        # create k-fold iterator
        if verbose:
            print('Crossvalidating Bayes classifier...')
        skf = ms.StratifiedKFold(n_splits=n_folds,shuffle=True,random_state=rand_seed)
        
        # iterate through and evaluate model on all the train/test splits
        yhat = np.zeros(N)
        i = 1
        for train_idx,test_idx in skf.split(X,y):
            if verbose:
                print('   Fold ',i,' of ',n_folds)
            i = i+1

            X_train,X_test = X[train_idx], X[test_idx]
            y_train,y_test = y[train_idx], y[test_idx]

            tmp = bayes_classifier(self.dist_type,self.prior_type)
            tmp.train(X_train,y_train)
            curr_y = tmp.predict(X_test)
            del tmp

            yhat[test_idx] = curr_y
        
        # create pred dict
        pred = {
            'y': y,
            'yhat': yhat 
        }

        acc = np.mean(y==yhat)
        if verbose:
            print('Crossvalidation accuracy: {:1.3f}'.format(acc))

        # train on all data
        if verbose:
            print('Training final model on all data...')
        self.train(X,y)

        return pred, acc


    def get_params(self):
        if hasattr(self,'params'):
            return self.params
        else:
            raise Exception('nb model has not been training yet. Cannot get parameters!!!')
