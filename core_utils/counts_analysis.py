import numpy as np
import ar_model as ar
import bayes_classifier as nb
import cos_tuning as ct
import factor_analysis as fa

class counts_analysis:
    
    
    def __init__(self,spike_counts,cond_labels,bin_size):
        self.X = np.array(spike_counts)
        self.N,self.D = self.X.shape
        self.y = np.array(cond_labels)
        self.lbls = np.unique(self.y)
        self.n_lbls = len(self.lbls)
        self.bin_size = np.array(bin_size) # in seconds


    def __runningMean(self,x, N):
        return np.convolve(x, np.ones((N,))/N,mode='same')


    def compute_avg_fr(self):
        return (np.mean(self.X,axis=0) / self.bin_size).flatten()


    def compute_cond_means(self):
        # compute responses to each condition
        cond_mean = np.zeros((self.n_lbls,self.D))
        for i,tmp in enumerate(self.lbls):
            idx = self.y==tmp
            cond_mean[i,:] = np.mean(self.X[idx,:],axis=0).flatten()
        return cond_mean


    def compute_cond_vars(self):
        # compute variances of each condition
        cond_vars = np.zeros((self.n_lbls,self.D))
        for i,tmp in enumerate(self.lbls):
            idx = self.y==tmp
            cond_vars[i,:] = np.var(self.X[idx,:],axis=0).flatten()
        return cond_vars
    
        
    def rm_cond_means(self):        
        # remove condition means from data        
        cond_means = self.compute_cond_means()
        X_nomean = np.zeros(self.X.shape)
        for i,tmp in enumerate(self.lbls):
            idx = self.y==tmp
            X_nomean[idx,:] = self.X[idx,:] - cond_means[i,:]
        return X_nomean
    
    
    def get_cond_varexp(self,return_each=False):
        # percent of variability explained by conditions
        cond_mean = self.compute_cond_means()
        cond_var = np.var(cond_mean,axis=0)
        total_var = np.var(self.X,axis=0)
        perc_cond_var = cond_var / total_var * 100
        if return_each:
            return perc_cond_var
        else:
            return np.mean(perc_cond_var)
    
    
    def compute_fano(self,return_stats=False):
        # compute fano factor (var/mean)
        # first remove condition means
        X_nomean = self.rm_cond_means()
        
        mean_count = np.mean(self.X,axis=0)
        var_count = np.var(X_nomean,axis=0,ddof=0)
        fano = var_count/mean_count
        if return_stats:
            return fano,mean_count,var_count
        else:
            return fano
        
    
    def compute_autoreg(self,order=25,both_dirs=True,auto_type='ar'):
        if order<=0:
            return np.zeros(self.X.shape)
        # compute autoregressive predictions for each neuron
        # first remove condition means
        X_nomean = self.rm_cond_means()
        if auto_type.lower()=='ar':
            # fit ar model for each neuron
            a = ar.ar_model(n_lags=order,both_dirs=both_dirs)
            X_auto = np.zeros(X_nomean.shape)
            for i in range(self.D):
                X_auto[:,i] = a.fit(X_nomean[:,i].flatten())
        elif auto_type.lower()=='mean':
            X_auto = np.zeros(X_nomean.shape)
            for i in range(self.D):
                X_auto[:,i] = self.__runningMean(X_nomean[:,i],order)
        return X_auto
        
    
    def rm_autoreg(self,order=25,both_dirs=True,fa_remove=False,fa_dims=15,auto_type='ar'):
        X_nomean = self.rm_cond_means()
        if order<=0:
            ar_est = np.zeros(X_nomean.shape)
        else:
            # remove autoregressive (i.e. slow) processes
            tmp = self.compute_autoreg(order=order,both_dirs=both_dirs,auto_type=auto_type)
            if fa_remove:
                fa_mdl = fa.factor_analysis(model_type='fa')
                fa_mdl.train(tmp,zDim=np.minimum(fa_dims,self.D-1))
                z,LL = fa_mdl.estep(tmp)
                ar_est = z['z_mu'].dot(fa_mdl.get_params()['L'].T)
            else:
                ar_est = tmp
            
        
        return X_nomean-ar_est,ar_est
        
    
    def get_auto_varexp(self,return_each=False,order=25,both_dirs=True,auto_type='ar'):
        # percent of variability explained by conditions
        X_auto = self.compute_autoreg(order=order,both_dirs=both_dirs,auto_type=auto_type)
        auto_var = np.var(X_auto,axis=0)
        total_var = np.var(self.X,axis=0)
        perc_auto_var = auto_var / total_var * 100
        if return_each:
            return perc_auto_var
        else:
            return np.mean(perc_auto_var)
    
    
    def fit_cosine_tuning(self,compute_p=False,n_samp=100,rand_seed=None):
        # compute condition means
        cond_means = self.compute_cond_means()
        
        pred_ang = np.linspace(0,360,100)
        preds = np.zeros((100,self.D))
        mod_depth = np.zeros(self.D)
        PD = np.zeros(self.D)
        mod_val = np.zeros(self.D)
        bl = np.zeros(self.D)
        # fit and make predictions
        ct_mdl = ct.cos_tuning()
        for i in range(self.D):
            mod_depth[i],PD[i],mod_val[i],bl[i] = \
                ct_mdl.fit(self.lbls,cond_means[:,i])
            preds[:,i] = ct_mdl.predict(pred_ang)
        params = {
            'bl':bl,
            'PD':PD,
            'mod_val':mod_val,
            'mod_depth':mod_depth
        }
        
        # compute p values on modulation depth (permutation test)
        if compute_p:
            if not(rand_seed is None):
                np.random.seed(rand_seed)
            null_dist = np.zeros((n_samp,self.D))
            for i in range(n_samp):
                tmp = counts_analysis(self.X,self.y[np.random.permutation(self.N)],self.bin_size)
                tmp_params,_,_ = tmp.fit_cosine_tuning()
                null_dist[i,:] = tmp_params['mod_depth']
            p_val = np.mean(1 - (mod_depth>null_dist),axis=0)
            params['p_val'] = p_val.flatten()
        
        return params,preds,pred_ang
    
    
    def compute_rsc(self,compute_null=False,rand_seed=None):
        X_nomean = self.rm_cond_means()
        corr_mat = np.corrcoef(X_nomean.T)
        rsc = corr_mat[np.triu_indices(self.D,k=1)]

        # compute null distribution
        if compute_null:
            if not(rand_seed is None):
                np.random.seed(rand_seed)
            
            for i in range(self.D):
                X_nomean[:,i] = X_nomean[np.random.permutation(self.N),i]
            corr_mat = np.corrcoef(X_nomean.T)
            null_dist = corr_mat[np.triu_indices(self.D,k=1)]
            return rsc, null_dist
        else:
            return rsc
    
    
    def compute_signal_corr(self):
        cond_means = self.compute_cond_means()
        corr_mat = np.corrcoef(cond_means.T)
        sig_corr = corr_mat[np.triu_indices(self.D,k=1)]
        return sig_corr
    
    
    def decode(self,dist_type='shared_diag',n_folds=10,rm_auto=False,auto_order=25,auto_both_dirs=True,auto_type='ar',rand_seed=None):
        X_data = self.X - np.mean(self.X,axis=0)
        if rm_auto:
            X_auto = self.compute_autoreg(order=auto_order,both_dirs=auto_both_dirs,auto_type=auto_type)
            X_data = X_data - X_auto
        
        mdl = nb.bayes_classifier(dist_type=dist_type)
        preds,acc = mdl.crossvalidate(self.X,self.y,n_folds=n_folds,rand_seed=rand_seed,verbose=False)
        
        return acc, preds
    
