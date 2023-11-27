import numpy as np
import scipy.linalg as slin

class sim_pcca_fa:

    def __init__(self,xDim,yDim,zDim,zxDim,zyDim,rand_seed=None,flat_eigs=False,equal_eigs=True,sv_goal=50):
        self.xDim = xDim
        self.yDim = yDim
        self.zDim = zDim
        self.zxDim = zxDim
        self.zyDim = zyDim

        # set random seed
        if not(rand_seed is None):
            np.random.seed(rand_seed)
        
        # generate model parameters
        mu_x = np.random.randn(xDim)
        mu_y = np.random.randn(yDim)

        if equal_eigs:
            # generate eigenspectra with equal total variances
            eig_z = np.exp(-np.arange(zDim)/2)
            var_z = np.sum(eig_z)
            if flat_eigs:
                eig_zx = np.ones(zxDim) * (var_z / zxDim)
                eig_zy = np.ones(zyDim) * (var_z / zyDim)
            else:
                eig_zx = np.exp(-np.arange(zxDim)/2)
                eig_zx = eig_zx * (var_z / np.sum(eig_zx))
                eig_zy = np.exp(-np.arange(zyDim)/2)
                eig_zy = eig_zy * (var_z / np.sum(eig_zy))
            eig_z = np.diag(np.sqrt(eig_z))
            eig_zx = np.diag(np.sqrt(eig_zx))
            eig_zy = np.diag(np.sqrt(eig_zy))
        else:
            eig_z = np.exp(-np.arange(zDim)/2)
            eig_zx = np.exp(-np.arange(zxDim)/2)
            eig_zy = np.exp(-np.arange(zyDim)/2)
            eig_z = np.diag(np.sqrt(eig_z))
            eig_zx = np.diag(np.sqrt(eig_zx))
            eig_zy = np.diag(np.sqrt(eig_zy))


        # genrate loading matrices
        W_x = np.random.randn(xDim,zDim).dot(eig_z)
        W_y = np.random.randn(yDim,zDim).dot(eig_z)
        L_x = np.random.randn(xDim,zxDim).dot(eig_zx)
        L_y = np.random.randn(yDim,zyDim).dot(eig_zy)
        
        sharex,privx = np.mean(np.diag(W_x.dot(W_x.T))), np.mean(np.diag(L_x.dot(L_x.T)))
        sharey,privy = np.mean(np.diag(W_y.dot(W_y.T))), np.mean(np.diag(L_y.dot(L_y.T)))
        # W_x = W_x * np.sqrt(2*privx/sharex)
        # W_y = W_y * np.sqrt(2*privy/sharey)
        # sharex,sharey = np.mean(np.diag(W_x.dot(W_x.T))),np.mean(np.diag(W_y.dot(W_y.T)))
        psi_x = np.random.uniform(low=0.1,high=(100/sv_goal)*(sharex+privx),size=xDim)
        psi_y = np.random.uniform(low=0.1,high=(100/sv_goal)*(sharey+privy),size=yDim)

        # compute ground truth canonical correlations
        covX = W_x.dot(W_x.T) + L_x.dot(L_x.T) + np.diag(psi_x)
        covY = W_y.dot(W_y.T) + L_y.dot(L_y.T) + np.diag(psi_y)
        covXY = W_x.dot(W_y.T)
        inv_sqrt_covX = np.linalg.inv(slin.sqrtm(covX))
        inv_sqrt_covY = np.linalg.inv(slin.sqrtm(covY))
        K = inv_sqrt_covX.dot(covXY).dot(inv_sqrt_covY)
        u,d,v = np.linalg.svd(K)
        rho = d[0:zDim]

        # store model parameters in dict
        params = {
            'mu_x':mu_x,
            'mu_y':mu_y,
            'W_x':W_x,
            'W_y':W_y,
            'L_x':L_x,
            'L_y':L_y,
            'psi_x':psi_x,
            'psi_y':psi_y,
            'zDim':zDim,
            'rho':rho
        }
        self.params = params


    def sim_data(self,N,rand_seed=None):
        # set random seed
        if not(rand_seed is None):
            np.random.seed(rand_seed)

        mu_x = self.params['mu_x'].reshape(self.xDim,1)
        mu_y = self.params['mu_y'].reshape(self.yDim,1)
        W_x,W_y = self.params['W_x'], self.params['W_y']
        L_x,L_y = self.params['L_x'], self.params['L_y']
        psi_x,psi_y = self.params['psi_x'], self.params['psi_y']

        # generate data
        z = np.random.randn(self.zDim,N)
        zx = np.random.randn(self.zxDim,N)
        zy = np.random.randn(self.zyDim,N)
        ns_x = np.diag(np.sqrt(psi_x)).dot(np.random.randn(self.xDim,N))
        ns_y = np.diag(np.sqrt(psi_y)).dot(np.random.randn(self.yDim,N))
        X = (W_x.dot(z) + L_x.dot(zx) + ns_x) + mu_x
        Y = (W_y.dot(z) + L_y.dot(zy) + ns_y) + mu_y
        
        return X.T, Y.T


    def get_params(self):
        return self.params


    def set_params(self,params):
        self.params = params

