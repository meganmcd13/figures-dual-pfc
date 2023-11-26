import numpy as np
import prob_cca as pcca
import factor_analysis as fa_mdl
import scipy.linalg as slin
import sklearn.model_selection as ms
import counts_analysis as cp
from joblib import Parallel,delayed
from functools import partial
from psutil import cpu_count

class pcca_fa:


    def __init__(self,min_var=0.01):
        self.params = []
        self.min_var = min_var


    def train(self,X,Y,zDim,zxDim,zyDim,tol=1e-6,max_iter=int(1e6),verbose=False,rand_seed=None,warmstart=True,X_early_stop=None,Y_early_stop=None):
        # set random seed
        if not(rand_seed is None):
            np.random.seed(rand_seed)
        
        early_stop = not(X_early_stop is None) and not(Y_early_stop is None)

        # set some useful parameters
        N,xDim = X.shape
        N,yDim = Y.shape
        mu_x,mu_y = X.mean(axis=0),Y.mean(axis=0)
        Xc,Yc = (X-mu_x), (Y-mu_y) 
        XcYc = np.concatenate((Xc,Yc),axis=1)
        covX = 1/N * (Xc.T).dot(Xc)
        covY = 1/N * (Yc.T).dot(Yc)
        covXY = 1/N * (Xc.T).dot(Yc)
        sampleCov = 1/N * (XcYc.T).dot(XcYc)
        var_floor = self.min_var*np.diag(sampleCov)
        Iz = np.identity(zDim+zxDim+zyDim)
        const = (xDim+yDim)*np.log(2*np.pi)

        if early_stop:
            Xc_test,Yc_test = X_early_stop-mu_x,Y_early_stop-mu_y
            XcYc_test = np.concatenate((Xc_test,Yc_test),axis=1)
            cov_test = (1/N)*(XcYc_test.T.dot(XcYc_test))

        # check that covariance is full rank
        if np.linalg.matrix_rank(sampleCov)==(xDim+yDim):
            x_scale = np.exp(2/xDim*np.sum(np.log(np.diag(slin.cholesky(covX)))))
            y_scale = np.exp(2/yDim*np.sum(np.log(np.diag(slin.cholesky(covY)))))
        else:
            raise np.linalg.LinAlgError(f'Covariance matrix is low rank ({np.linalg.matrix_rank(sampleCov):d}, should be {xDim+yDim:d})')

        # initialize parameters
        if warmstart:
            tmp = pcca.prob_cca()
            tmp.train_maxLL(X,Y,zDim)
            W_x = tmp.get_params()['W_x']
            W_y = tmp.get_params()['W_y']
            tmp = fa_mdl.factor_analysis()
            tmp.train(X,zxDim,rand_seed=rand_seed)
            L_x = tmp.get_params()['L']
            tmp = fa_mdl.factor_analysis()
            tmp.train(Y,zyDim,rand_seed=rand_seed)
            L_y = tmp.get_params()['L']
        else:
            W_x = np.random.randn(xDim,zDim) * np.sqrt(x_scale/zDim)
            W_y = np.random.randn(yDim,zDim) * np.sqrt(y_scale/zDim)
            L_x = np.random.randn(xDim,zxDim) * np.sqrt(x_scale/zxDim)
            L_y = np.random.randn(yDim,zyDim) * np.sqrt(y_scale/zyDim)
        Ph = np.diag(sampleCov)
        
        L_top = np.concatenate((W_x,L_x,np.zeros((xDim,zyDim))),axis=1)
        L_bottom = np.concatenate((W_y,np.zeros((yDim,zxDim)),L_y),axis=1)
        L_total = np.concatenate((L_top,L_bottom),axis=0)
        
        L_mask = np.ones(L_total.shape)
        L_mask[:xDim,(zDim+zxDim):] = np.zeros((xDim,zyDim))
        L_mask[xDim:,zDim:(zDim+zxDim)] = np.zeros((yDim,zxDim))

        # em algorithm
        LL = []
        testLL = []
        for i in range(max_iter):
            # E-step: set q(z) = p(z,zx,zy|x,y)
            iPh = np.diag(1/Ph)
            iPhL = iPh.dot(L_total)
            if zDim==0 and zxDim==0 and zyDim==0:
                iSig = iPh
            else:
                iSig = iPh - iPhL.dot(slin.inv(Iz+(L_total.T).dot(iPhL))).dot(iPhL.T)
            iSigL = iSig.dot(L_total)
            cov_iSigL = sampleCov.dot(iSigL)
            E_zz = Iz - (L_total.T).dot(iSigL) + (iSigL.T).dot(cov_iSigL)
            
            # compute log likelihood
            logDet = 2*np.sum(np.log(np.diag(slin.cholesky(iSig))))
            curr_LL = -N/2 * (const - logDet + np.trace(iSig.dot(sampleCov)))
            LL.append(curr_LL)
            if early_stop:
                curr_testLL = -N/2 * (const - logDet + np.trace(iSig.dot(cov_test)))
                testLL.append(curr_testLL)
            if verbose:
                print('EM iteration ',i,', LL={:.2f}'.format(curr_LL))

            # check for convergence (training LL increases by less than tol, or testLL decreases)
            if i>1:
                if (LL[-1]-LL[-2])<tol or (early_stop and testLL[-1]<testLL[-2]):
                    break

            # M-step: compute new L and Ph
            if not(zDim==0 and zxDim==0 and zyDim==0):
                L_total = cov_iSigL.dot(slin.inv(E_zz))
            L_total = L_total * L_mask 
            Ph = sampleCov - L_total.dot(cov_iSigL.T) - cov_iSigL.dot(L_total.T) + L_total.dot(E_zz).dot(L_total.T)
            Ph = np.diag(Ph)
            Ph = np.maximum(Ph,var_floor)

        # get final parameters
        W_x, W_y = L_total[:xDim,:zDim], L_total[xDim:,:zDim]
        L_x, L_y = L_total[:xDim,zDim:(zDim+zxDim)], L_total[xDim:,(zDim+zxDim):]
        psi_x, psi_y = Ph[:xDim], Ph[xDim:]
        
        # compute canonical correlations
        est_covX = W_x.dot(W_x.T) + L_x.dot(L_x.T) + np.diag(psi_x)
        est_covY = W_y.dot(W_y.T) + L_y.dot(L_y.T) + np.diag(psi_y)
        est_covXY = W_x.dot(W_y.T)
        inv_sqrt_covX = slin.inv(slin.sqrtm(est_covX))
        inv_sqrt_covY = slin.inv(slin.sqrtm(est_covY))
        K = inv_sqrt_covX.dot(est_covXY).dot(inv_sqrt_covY)
        u,d,vt = slin.svd(K)
        rho = d[0:zDim]
        
        # order W_x, W_y by canon corrs
        pd = np.diag(np.sqrt(rho))
        W_x = slin.sqrtm(est_covX).dot(u[:,0:zDim]).dot(pd)
        W_y = slin.sqrtm(est_covY).dot(vt[0:zDim,:].T).dot(pd)

        # create parameter dict
        self.params = {
            'mu_x':mu_x,'mu_y':mu_y,
            'W_x':W_x,'W_y':W_y,
            'L_x':L_x,'L_y':L_y,
            'psi_x':psi_x,
            'psi_y':psi_y,
            'zDim':zDim,'zxDim':zxDim,'zyDim':zyDim,
            'rho':rho
        }
        
        return np.array(LL), np.array(testLL)


    def get_params(self):
        return self.params

    def set_params(self,params):
        self.params = params


    def estep(self,X,Y):
        N,xDim = X.shape
        N,yDim = Y.shape

        zDim,zxDim,zyDim = self.params['zDim'],self.params['zxDim'],self.params['zyDim']

        # get model parameters
        mu_x,mu_y = self.params['mu_x'],self.params['mu_y']
        
        W_x,W_y = self.params['W_x'],self.params['W_y']
        L_x,L_y = self.params['L_x'],self.params['L_y']
        L_top = np.concatenate((W_x,L_x,np.zeros((xDim,zyDim))),axis=1)
        L_bottom = np.concatenate((W_y,np.zeros((yDim,zxDim)),L_y),axis=1)
        L_total = np.concatenate((L_top,L_bottom),axis=0)
        
        psi_x = self.params['psi_x']
        psi_y = self.params['psi_y']
        psi = np.diag(np.concatenate((psi_x,psi_y)))

        # center data and compute covariances
        Xc = X-mu_x
        Yc = Y-mu_y
        XcYc = np.concatenate((Xc,Yc),axis=1)
        covX = 1/N * (Xc.T).dot(Xc)
        covY = 1/N * (Yc.T).dot(Yc)
        covXY = 1/N * (Xc.T).dot(Yc)
        sampleCov = 1/N * (XcYc.T).dot(XcYc)

        # compute z
        zDim,zxDim,zyDim = self.params['zDim'],self.params['zxDim'],self.params['zyDim']
        Iz = np.identity(zDim+zxDim+zyDim)
        C = L_total.dot(L_total.T) + psi
        invC = slin.inv(C)
        z_mu = XcYc.dot(invC).dot(L_total)
        z_cov = np.diag(np.diag(Iz - (L_total.T).dot(invC).dot(L_total)))

        # compute LL
        const = (xDim+yDim)*np.log(2*np.pi)
        logDet = 2*np.sum(np.log(np.diag(slin.cholesky(C))))
        LL = -N/2 * (const + logDet + np.trace(invC.dot(sampleCov)))
        
        # return posterior and LL
        z = { 
            'z_mu':z_mu[:,:zDim],
            'z_cov':z_cov[:zDim,:zDim],
            'zx_mu':z_mu[:,zDim:(zDim+zxDim)],
            'zx_cov':z_cov[zDim:(zDim+zxDim),zDim:(zDim+zxDim)],
            'zy_mu':z_mu[:,(zDim+zxDim):],
            'zy_cov':z_cov[(zDim+zxDim):,(zDim+zxDim):],
            'z_mu_all':z_mu,
            'z_cov_all':z_cov
        }
        return z, LL


    def orthogonalize(self,zx_mu,zy_mu):
        N,zDim = zx_mu.shape

        Lx = self.params['L_x']
        Ly = self.params['L_y']

        # orthogonalize zx
        u,s,vt = slin.svd(Lx,full_matrices=False)
        Lx_orth = u
        TT = np.diag(s).dot(vt)
        zx = (TT.dot(zx_mu.T)).T

        # orthogonalize zy
        u,s,vt = slin.svd(Ly,full_matrices=False)
        Ly_orth = u
        TT = np.diag(s).dot(vt)
        zy = (TT.dot(zy_mu.T)).T

        # return z_orth, Lorth
        z_orth = {'zx':zx,'zy':zy}
        W_orth = {'Lx':Lx_orth,'Ly':Ly_orth}

        return z_orth, W_orth


    def crossvalidate(self,X,Y,zDim_list=np.linspace(0,8,9),zxDim_list=np.linspace(0,8,9),zyDim_list=np.linspace(0,8,9),n_folds=10,verbose=True,max_iter=int(1e6),tol=1e-6,warmstart=True,rand_seed=None,parallelize=False,early_stop=False):
        # set random seed
        if not(rand_seed is None):
            np.random.seed(rand_seed)

        N,xDim = X.shape
        N,yDim = Y.shape

        # make sure z dims are integers
        z_list,zx_list,zy_list = np.meshgrid(zDim_list.astype(int),zxDim_list.astype(int),zyDim_list.astype(int))
        z_list = np.matrix.flatten(z_list)
        zx_list = np.matrix.flatten(zx_list)
        zy_list = np.matrix.flatten(zy_list)
        LL_curves = {'z_list':z_list,'zx_list':zx_list,'zy_list':zy_list}

        # create k-fold iterator
        if verbose:
            print('Crossvalidating pCCA-FA model to choose # of dims...')
        cv_kfold = ms.KFold(n_splits=n_folds,shuffle=True,random_state=rand_seed)

        # iterate through train/test splits
        i = 0
        LLs,PEs = np.zeros([n_folds,len(z_list)]),np.zeros([n_folds,len(z_list)])
        for train_idx,test_idx in cv_kfold.split(X):
            if verbose:
                print('   Fold ',i+1,' of ',n_folds,'...')
            X_train,X_test = X[train_idx], X[test_idx]
            Y_train,Y_test = Y[train_idx], Y[test_idx]
            
            # iterate through each zDim
            func = partial(self._cv_helper,Xtrain=X_train,Ytrain=Y_train,Xtest=X_test,Ytest=Y_test,\
                           rand_seed=rand_seed,max_iter=max_iter,tol=tol,warmstart=warmstart,early_stop=early_stop)
            if parallelize:
                tmp = Parallel(n_jobs=cpu_count(logical=False),backend='loky')\
                    (delayed(func)(z_list[j],zx_list[j],zy_list[j]) for j in range(len(z_list)))
                LLs[i,:] = [val[0] for val in tmp]
                PEs[i,:] = [val[1] for val in tmp]
            else:      
                for j in range(len(z_list)):
                    tmp = func(z_list[j],zx_list[j],zy_list[j])
                    LLs[i,j],PEs[i,j] = tmp[0],tmp[1]
                    
            i = i+1
        
        sum_LLs = LLs.sum(axis=0)
        sum_SEs = PEs.sum(axis=0)
        LL_curves['LLs'] = sum_LLs
        LL_curves['PEs'] = sum_SEs

        # find the best # of z dimensions and train CCA model
        max_idx = np.argmax(sum_LLs)
        zDim,zxDim,zyDim = z_list[max_idx],zx_list[max_idx],zy_list[max_idx]
        LL_curves['zDim']=zDim
        LL_curves['zxDim']=zxDim
        LL_curves['zyDim']=zyDim
        LL_curves['final_LL'] = sum_LLs[max_idx]
        self.train(X,Y,zDim,zxDim,zyDim)

        # cross-validate to get canonical correlations
        if verbose:
            print('Crossvalidating pCCA-FA model to compute canon corrs...')
        zx,zy = np.zeros((2,N,zDim))
        for train_idx,test_idx in cv_kfold.split(X):
            X_train,X_test = X[train_idx], X[test_idx]
            Y_train,Y_test = Y[train_idx], Y[test_idx]

            tmp = pcca_fa()
            tmp.train(X_train,Y_train,zDim,zxDim,zyDim,rand_seed=rand_seed,max_iter=max_iter,tol=tol,warmstart=warmstart)
            tmp_params = tmp.get_params()
            tmp_params['psi_x'] = tmp_params['L_x'].dot(tmp_params['L_x'].T) + np.diag(tmp_params['psi_x'])
            tmp_params['psi_y'] = tmp_params['L_y'].dot(tmp_params['L_y'].T) + np.diag(tmp_params['psi_y'])
            tmp2 = pcca.prob_cca()
            tmp2.set_params(tmp_params)
            z,curr_LL = tmp2.estep(X_test,Y_test)

            zx[test_idx,:] = z['zx_mu']
            zy[test_idx,:] = z['zy_mu']

        cv_rho = np.zeros(zDim)
        for i in range(zDim):
            tmp = np.corrcoef(zx[:,i],zy[:,i])
            cv_rho[i] = tmp[0,1]
        
        self.params['cv_rho'] = cv_rho
        self.LL_curves = LL_curves

        return LL_curves


    def _cv_helper(self,zDim,zxDim,zyDim,Xtrain,Ytrain,Xtest,Ytest,rand_seed=None,max_iter=int(1e5),tol=1e-6,warmstart=True,early_stop=False):
        tmp = pcca_fa()
        if early_stop:
            tmp.train(Xtrain,Ytrain,zDim,zxDim,zyDim,rand_seed=rand_seed,max_iter=max_iter,tol=tol,warmstart=warmstart,X_early_stop=Xtest,Y_early_stop=Ytest)
        else:
            tmp.train(Xtrain,Ytrain,zDim,zxDim,zyDim,rand_seed=rand_seed,max_iter=max_iter,tol=tol,warmstart=warmstart)
        tmp_params = tmp.get_params()
        # log-likelihood
        _,LL = tmp.estep(Xtest,Ytest)
        # prediction error
        Xtest_pred,Ytest_pred = tmp._leaveoneout_pred(Xtest,Ytest)
        PE = np.sum(np.square(Xtest_pred - Xtest)) + np.sum(np.square(Ytest_pred - Ytest))
        
        return (LL,PE)


    def _leaveoneout_pred(self,X,Y):
        N,xDim = X.shape
        yDim = Y.shape[1]
        X_total = np.concatenate((X,Y),axis=1)

        # extract model parameters
        W_x,L_x,W_y,L_y = self.params['W_x'],self.params['L_x'],self.params['W_y'],self.params['L_y']
        Ph = np.concatenate((self.params['psi_x'],self.params['psi_y']),axis=0)
        mu = np.concatenate((self.params['mu_x'],self.params['mu_y']),axis=0)
        zxDim,zyDim = L_x.shape[1],L_y.shape[1]
        L_top = np.concatenate((W_x,L_x,np.zeros((xDim,zyDim))),axis=1)
        L_bottom = np.concatenate((W_y,np.zeros((yDim,zxDim)),L_y),axis=1)
        L_total = np.concatenate((L_top,L_bottom),axis=0)

        # compute covariances
        mdl_cov = L_total.dot(L_total.T) + np.diag(Ph)
        inv_cov = slin.inv(mdl_cov)

        # compute conditional expectrations (predictions)
        D = xDim+yDim
        pred_total = np.zeros((N,D))
        for i in range(D):
            E = np.delete(np.delete(inv_cov,i,axis=0),i,axis=1)
            f = np.delete(inv_cov[:,i],i,axis=0)
            h = inv_cov[i,i]
            inv_term = E - (1/h)*np.outer(f,f)
            proj_term = np.delete(mdl_cov[i,:],i) # 1 x D-1
            mean_term = np.delete(X_total,i,axis=1) - np.delete(mu,i,axis=0).T # N x D-1
            pred = mu[i] + proj_term.dot(inv_term.dot(mean_term.T)) # 1 x N
            pred_total[:,i] = pred.T

        return pred_total[:,:xDim],pred_total[:,xDim:]


    def _compute_ls(self,x):
        if len(x.shape)>1:
            raise Exception('x must be a vector')
        x = x / slin.norm(x)
        n_neurons = len(x)
        return 1-n_neurons*x.var(ddof=0)


    def compute_load_sim(self):
        n_x = self.params['W_x'].shape[0]
        n_y = self.params['W_y'].shape[0]

        
        Wx,_,_ = slin.svd(self.params['W_x'],full_matrices=False)
        Wy,_,_ = slin.svd(self.params['W_y'],full_matrices=False)
        Lx,_,_ = slin.svd(self.params['L_x'],full_matrices=False)
        Ly,_,_ = slin.svd(self.params['L_y'],full_matrices=False)

        ls_x = 1 - n_x*Wx.var(axis=0,ddof=0)
        ls_y = 1 - n_y*Wy.var(axis=0,ddof=0)
        ls_priv_x = 1 - n_x*Lx.var(axis=0,ddof=0)
        ls_priv_y = 1 - n_y*Ly.var(axis=0,ddof=0)

        ls = {
            'ls_x':ls_x,'ls_y':ls_y,
            'ls_priv_x':ls_priv_x,'ls_priv_y':ls_priv_y
        }
        return ls


    def compute_dshared(self,cutoff_thresh=0.95):
        # for area x
        Wx = self.params['W_x']
        shared_x = Wx.dot(Wx.T)
        s = slin.svd(shared_x,full_matrices=False,compute_uv=False)
        var_exp = np.cumsum(s)/np.sum(s)
        dims = np.where(var_exp >= (cutoff_thresh - 1e-9))[0]
        dshared_x = dims[0]+1
        
        Lx = self.params['L_x']
        shared_x = Lx.dot(Lx.T)
        s = slin.svd(shared_x,full_matrices=False,compute_uv=False)
        var_exp = np.cumsum(s)/np.sum(s)
        dims = np.where(var_exp >= (cutoff_thresh - 1e-9))[0]
        dshared_priv_x = dims[0]+1

        # for area y
        Wy = self.params['W_y']
        shared_y = Wy.dot(Wy.T)
        s = slin.svd(shared_y,full_matrices=False,compute_uv=False)
        var_exp = np.cumsum(s)/np.sum(s)
        dims = np.where(var_exp >= (cutoff_thresh - 1e-9))[0]
        dshared_y = dims[0]+1

        Ly = self.params['L_y']
        shared_y = Ly.dot(Ly.T)
        s = slin.svd(shared_y,full_matrices=False,compute_uv=False)
        var_exp = np.cumsum(s)/np.sum(s)
        dims = np.where(var_exp >= (cutoff_thresh - 1e-9))[0]
        dshared_priv_y = dims[0]+1

        # overall
        W = np.concatenate((Wx,Wy),axis=0)
        shared = W.dot(W.T)
        s = slin.svd(shared,full_matrices=False,compute_uv=False)
        var_exp = np.cumsum(s)/np.sum(s)
        dims = np.where(var_exp >= (cutoff_thresh - 1e-9))[0]
        dshared_all = dims[0]+1

        return_dict = {
            'dshared_x':dshared_x,'dshared_y':dshared_y,
            'dshared_priv_x':dshared_priv_x,'dshared_priv_y':dshared_priv_y,
            'dshared_all':dshared_all
        }

        return return_dict


    def compute_part_ratio(self):
        # for area x
        Wx = self.params['W_x']
        shared_x = Wx.dot(Wx.T)
        s = slin.svd(shared_x,full_matrices=False,compute_uv=False)
        pr_x = np.square(s.sum()) / np.square(s).sum()
        
        Lx = self.params['L_x']
        shared_x = Lx.dot(Lx.T)
        s = slin.svd(shared_x,full_matrices=False,compute_uv=False)
        pr_priv_x = np.square(s.sum()) / np.square(s).sum()

        # for area y
        Wy = self.params['W_y']
        shared_y = Wy.dot(Wy.T)
        s = slin.svd(shared_y,full_matrices=False,compute_uv=False)
        pr_y = np.square(s.sum()) / np.square(s).sum()

        Ly = self.params['L_y']
        shared_y = Ly.dot(Ly.T)
        s = slin.svd(shared_y,full_matrices=False,compute_uv=False)
        pr_priv_y = np.square(s.sum()) / np.square(s).sum()

        # overall
        W = np.concatenate((Wx,Wy),axis=0)
        shared = W.dot(W.T)
        s = slin.svd(shared,full_matrices=False,compute_uv=False)
        pr = np.square(s.sum()) / np.square(s).sum()

        return_dict = {
            'pr':pr,
            'pr_x':pr_x,'pr_y':pr_y,
            'pr_priv_x':pr_priv_x,'pr_priv_y':pr_priv_y
        }

        return return_dict


    def compute_psv(self):
        Wx,Wy = self.params['W_x'],self.params['W_y']
        Lx,Ly = self.params['L_x'],self.params['L_y']
        priv_x,priv_y = self.params['psi_x'],self.params['psi_y']
        
        shared_acc_x = np.diag(Wx.dot(Wx.T))
        shared_acc_y = np.diag(Wy.dot(Wy.T))
        shared_with_x = np.diag(Lx.dot(Lx.T))
        shared_with_y = np.diag(Ly.dot(Ly.T))
        total_x = shared_acc_x + shared_with_x + priv_x
        total_y = shared_acc_y + shared_with_y + priv_y
        
        # for area x
        ind_psv_x = (shared_acc_x / total_x).flatten() * 100
        ind_psv_priv_x = (shared_with_x / total_x).flatten() * 100
        psv_x = np.mean(ind_psv_x)
        psv_priv_x = np.mean(ind_psv_priv_x)

        # for area y
        ind_psv_y = (shared_acc_y / total_y).flatten() * 100
        ind_psv_priv_y = (shared_with_y / total_y).flatten() * 100
        psv_y = np.mean(ind_psv_y)
        psv_priv_y = np.mean(ind_psv_priv_y)

        # overall
        psv_overall = np.mean(np.concatenate((ind_psv_x,ind_psv_y)))
        psv_priv_overall = np.mean(np.concatenate((ind_psv_priv_x,ind_psv_priv_y)))

        psv = {
            'ind_psv_x':ind_psv_x,'ind_psv_y':ind_psv_y,
            'ind_psv_priv_x':ind_psv_priv_x,'ind_psv_priv_y':ind_psv_priv_y,
            'psv_x':psv_x,'psv_y':psv_y,
            'psv_priv_x':psv_priv_x,'psv_priv_y':psv_priv_y,
            'psv_all':psv_overall,'psv_priv_all':psv_priv_overall
        }
        return psv


    def compute_metrics(self,cutoff_thresh=0.95):
        dshared = self.compute_dshared(cutoff_thresh=cutoff_thresh)
        psv = self.compute_psv()
        pr = self.compute_part_ratio()
        ls = self.compute_load_sim()

        metrics = {
            'dshared':dshared,
            'psv':psv,
            'part_ratio':pr,
            'load_sim':ls,
            'rho':self.params['rho']
        }
        if 'cv_rho' in self.params:
            metrics['cv_rho'] = self.params['cv_rho']
        return metrics


    def compute_cv_psv(self,X,Y,zDim,zxDim,zyDim,rand_seed=None,n_boots=10,test_size=0.1,verbose=False,early_stop=True,return_each=False):
        # create k-fold iterator
        if verbose:
            print('Crossvalidating percent shared variance...')
        cv_folds = ms.ShuffleSplit(n_splits=n_boots,random_state=rand_seed,\
            train_size=1-test_size,test_size=test_size)

        # iterate through train/test splits
        i = 0
        train_psv_x,test_psv_x = np.zeros(n_boots),np.zeros(n_boots)
        train_psv_y,test_psv_y = np.zeros(n_boots),np.zeros(n_boots)
        train_psv,test_psv = np.zeros(n_boots),np.zeros(n_boots)
        train_psv_priv_x,test_psv_priv_x = np.zeros(n_boots),np.zeros(n_boots)
        train_psv_priv_y,test_psv_priv_y = np.zeros(n_boots),np.zeros(n_boots)
        train_psv_priv,test_psv_priv = np.zeros(n_boots),np.zeros(n_boots)
        for train_idx,test_idx in cv_folds.split(X):
            if verbose:
                print('   Bootstrap sample ',i+1,' of ',n_boots,'...')

            X_train,X_test = X[train_idx], X[test_idx]
            Y_train,Y_test = Y[train_idx], Y[test_idx]
            
            # train model
            tmp = pcca_fa()
            if early_stop:
                tmp.train(X_train,Y_train,zDim,zxDim,zyDim,rand_seed=rand_seed,X_early_stop=X_test,Y_early_stop=Y_test)
            else:
                tmp.train(X_train,Y_train,zDim,zxDim,zyDim,rand_seed=rand_seed)

            tmp_psv = tmp.compute_psv_heldout(X_train,Y_train)
            train_psv_x[i] = tmp_psv['psv_x']
            train_psv_y[i] = tmp_psv['psv_y']
            train_psv_priv_x[i] = tmp_psv['psv_priv_x']
            train_psv_priv_y[i] = tmp_psv['psv_priv_y']
            train_psv[i] = tmp_psv['psv_all']
            train_psv_priv[i] = tmp_psv['psv_priv_all']

            tmp_psv = tmp.compute_psv_heldout(X_test,Y_test)
            test_psv_x[i] = tmp_psv['psv_x']
            test_psv_y[i] = tmp_psv['psv_y']
            test_psv_priv_x[i] = tmp_psv['psv_priv_x']
            test_psv_priv_y[i] = tmp_psv['psv_priv_y']
            test_psv[i] = tmp_psv['psv_all']
            test_psv_priv[i] = tmp_psv['psv_priv_all']

            i = i+1

        if return_each:
            train_psv = {
                'psv_x':train_psv_x,
                'psv_y':train_psv_y,
                'psv_priv_x':train_psv_priv_x,
                'psv_priv_y':train_psv_priv_y,
                'psv_all':train_psv,
                'psv_priv_all':train_psv_priv
            }
            test_psv = {
                'psv_x':test_psv_x,
                'psv_y':test_psv_y,
                'psv_priv_x':test_psv_priv_x,
                'psv_priv_y':test_psv_priv_y,
                'psv_all':test_psv,
                'psv_priv_all':test_psv_priv
            }
        else:
            train_psv = {
                'psv_x':np.mean(train_psv_x),
                'psv_y':np.mean(train_psv_y),
                'psv_priv_x':np.mean(train_psv_priv_x),
                'psv_priv_y':np.mean(train_psv_priv_y),
                'psv_all':np.mean(train_psv),
                'psv_priv_all':np.mean(train_psv_priv)
            }
            test_psv = {
                'psv_x':np.mean(test_psv_x),
                'psv_y':np.mean(test_psv_y),
                'psv_priv_x':np.mean(test_psv_priv_x),
                'psv_priv_y':np.mean(test_psv_priv_y),
                'psv_all':np.mean(test_psv),
                'psv_priv_all':np.mean(test_psv_priv)
            }

        if return_each:
            return train_psv, test_psv
        else:
            return train_psv, test_psv


    def compute_psv_heldout(self,X_heldout,Y_heldout):
        Wx,Wy = self.params['W_x'],self.params['W_y']
        Lx,Ly = self.params['L_x'],self.params['L_y']
        Phx,Phy = self.params['psi_x'],self.params['psi_y']

        N = X_heldout.shape[0]
        zDim = Wx.shape[1]
        yDim,zyDim = Ly.shape
        xDim,zxDim = Lx.shape

        covX = np.cov(X_heldout.T,bias=True)
        covY = np.cov(Y_heldout.T,bias=True)

        # empirical part (for E[z|x] part)
        z,_ = self.estep(X_heldout,Y_heldout)
        cov_ez = np.cov(z['z_mu_all'].T,bias=True)
        s_zz = cov_ez[:zDim,:zDim]
        s_zx = cov_ez[:zDim,zDim:(zDim+zxDim)]
        s_zy = cov_ez[:zDim,(zDim+zxDim):]
        s_xx = cov_ez[zDim:(zDim+zxDim),zDim:(zDim+zxDim)]
        s_xy = cov_ez[zDim:(zDim+zxDim),(zDim+zxDim):]
        s_yy = cov_ez[(zDim+zxDim):,(zDim+zxDim):]

        # theoretical part (for uncertainty part)
        L_top = np.concatenate((Wx,Lx,np.zeros((xDim,zyDim))),axis=1)
        L_bottom = np.concatenate((Wy,np.zeros((yDim,zxDim)),Ly),axis=1)
        L_total = np.concatenate((L_top,L_bottom),axis=0)
        Ph = np.concatenate((Phx,Phy))
        Sig = L_total.dot(L_total.T) + np.diag(Ph)
        iSig = slin.inv(Sig)
        cov_theo_z = (L_total.T).dot(iSig).dot(L_total)
        c_zz = cov_theo_z[:zDim,:zDim]
        c_zx = cov_theo_z[:zDim,zDim:(zDim+zxDim)]
        c_zy = cov_theo_z[:zDim,(zDim+zxDim):]
        c_xx = cov_theo_z[zDim:(zDim+zxDim),zDim:(zDim+zxDim)]
        c_xy = cov_theo_z[zDim:(zDim+zxDim),(zDim+zxDim):]
        c_yy = cov_theo_z[(zDim+zxDim):,(zDim+zxDim):]

        # compute x components
        theo_int = Wx.dot(c_zx).dot(Lx.T) + Lx.dot(c_zx.T).dot(Wx.T)
        ez_int = Wx.dot(s_zx).dot(Lx.T) + Lx.dot(s_zx.T).dot(Wx.T)
        shared_x = Wx.dot(Wx.T) - \
            Wx.dot(c_zz).dot(Wx.T) + Wx.dot(s_zz).dot(Wx.T) - \
            (0.5*theo_int) + (0.5*ez_int)
        priv_x = Lx.dot(Lx.T) - \
            Lx.dot(c_xx).dot(Lx.T) + Lx.dot(s_xx).dot(Lx.T) - \
            (0.5*theo_int) + (0.5*ez_int)

        # compute y components
        theo_int = Wy.dot(c_zy).dot(Ly.T) + Ly.dot(c_zy.T).dot(Wy.T)
        ez_int = Wy.dot(s_zy).dot(Ly.T) + Ly.dot(s_zy.T).dot(Wy.T)
        shared_y = Wy.dot(Wy.T) - \
            Wy.dot(c_zz).dot(Wy.T) + Wy.dot(s_zz).dot(Wy.T) - \
            (0.5*theo_int) + (0.5*ez_int)
        priv_y = Ly.dot(Ly.T) - \
            Ly.dot(c_yy).dot(Ly.T) + Ly.dot(s_yy).dot(Ly.T) - \
            (0.5*theo_int) + (0.5*ez_int)
        
        total_var_x = np.diag(covX)
        total_var_y = np.diag(covY)
        shared_x = np.diag(shared_x)
        shared_y = np.diag(shared_y)
        shared_priv_x = np.diag(priv_x)
        shared_priv_y = np.diag(priv_y)

        return {
            'psv_x' : np.mean(shared_x/total_var_x)*100,
            'psv_y' : np.mean(shared_y/total_var_y)*100,
            'psv_priv_x' : np.mean(shared_priv_x/total_var_x)*100,
            'psv_priv_y' : np.mean(shared_priv_y/total_var_y)*100,
            'psv_all' : np.mean(np.concatenate((shared_x/total_var_x,shared_y/total_var_y)))*100,
            'psv_priv_all' : np.mean(np.concatenate((shared_priv_x/total_var_x,shared_priv_y/total_var_y)))*100
        }

