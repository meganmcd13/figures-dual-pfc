# MODULE OF HELPER FUNCTIONS
import sys
sys.path.append('../helpers/')
sys.path.append('../helpers/pcca_fa/')

def getParams():
    # returns dictionary of helpful parameters for plotting 
    import numpy as np

    # plot colors
    color_map = {
        'across':np.array([255,76,178])/255, # pink
        'within1':np.array([111,192,255])/255, # light blue - right hemisphere
        'within2':np.array([0,87,154])/255, # dark blue - left hemisphere
        'within':np.array([0,144,255])/255, # medium blue - collapsed across both hemispheres
        'independent':np.array([200,200,200])/255 # gray
    }
    params = {
        'subjects': ('pepe','wakko','satchel'),
        'markers': ('o','s','^'), # corresponding plot markers
        'color_map': color_map
    }
    return params

def getBaseSimParams():
    # return dictionary of base simulation parameters
    params = {
        'n_trials' : 1000,
        'n1' : 30, 'n2' : 30,
        'd'  : 5,  'd1' : 3, 'd2' : 3,
        'n_boots' : 100,
        'sv_goal' : (15,10),
    }
    return params

# load dictionary using pickle
def load_dict(filename):
    import pickle
    with open(filename, 'rb') as handle:
        data = handle.read()
    # reconstructing the data as dictionary
    return pickle.loads(data)

# save dictionary using pickle
def save_dict(obj, filename):
    # obj: dictionary to be saved
    # filename: name to save file under
    import pickle
    if '.pkl' in filename: save_name = filename
    else: save_name = filename + '.pkl'

    with open(save_name, 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

# generate random jitter for plotting clarity
def jitter(length=1,spacing=0.2,rand_seed=None):
    import numpy as np
    if rand_seed is not None: np.random.seed(rand_seed)
    return np.random.uniform(low=-spacing,high=spacing,size=length)

def vector_angle(v1,v2):
    # compute angle between 2 vectors
    import numpy as np
    import scipy.linalg as slin

    v1, v2 = v1.squeeze(), v2.squeeze()
    cos_theta = np.dot(v1, v2) / (slin.norm(v1) * slin.norm(v2))
    theta = np.arccos(np.clip(cos_theta, -1.0, 1.0)) # in radians
    return theta

# principal angles:
def prinangle(A, B):
    import math
    import scipy.linalg as slin
    import numpy as np
    # check for 1d vector, ie: shape = (43,)
    if A.ndim == 1:
        A = A.reshape((len(A),1))
    if B.ndim == 1:
        B = B.reshape((len(B),1))
    A_orth = slin.orth(A)
    B_orth = slin.orth(B)
    [_, sv, _] = slin.svd(np.transpose(A_orth) @ B_orth)
    for i, val in enumerate(sv):
        if math.isclose(1, val, abs_tol=1e-5):
            sv[i] = 1 # prevent domain errors in acos
    pa = list(map(lambda x: math.acos(x) * 180 / math.pi, sv))
    return pa

# function for returning the top eigenvector of a matrix
def get_top_vec(arr,orth=True):
    # orth tells whether to first orthogonalize the matrix
    import scipy.linalg as slin
    
    if orth:
        u,_,_ = slin.svd(arr)
    else:
        u = arr / slin.norm(arr,axis=0)
    top_vec = u[:,0]
    n,_ = arr.shape
    pct_pos = (top_vec >= 0).sum() / n
    if pct_pos < 0.5:
        flip = True
        return -top_vec, (1-pct_pos), flip
    else:
        flip = False
        return top_vec, pct_pos, flip
    

# compute top angle between set of pCCA-FA params
def get_top_angle(params,across_mode='cov'):
    # params is a parameter dictionary returned by pCCA-FA model
    # across_mode ('cov' or 'corr') indicates whether to get top covariate mode or correlative mode of W
    import scipy.linalg as slin
    import pcca_fa.pcca_fa_mdl as pf

    mdl = pf.pcca_fa()
    mdl.set_params(params)
    W_1,W_2,L_1,L_2 = mdl.get_loading_matrices()

    if across_mode == 'cov':
        # orthonormalize W's to be be ordered by SV
        uw1,_,_ = slin.svd(W_1)
        uw2,_,_ = slin.svd(W_2)
    elif across_mode == 'corr':
        # order W by canon corrs
        uw1,uw2 = mdl.get_correlative_modes()
    else:
        raise ValueError('across_mode must be "cov" or "corr"')
    
    # orthogonalize L to be ordered by SV
    ul1,_,_ = slin.svd(L_1)
    ul2,_,_ = slin.svd(L_2)

    # top angles
    x1_angle = prinangle(uw1[:,0], ul1[:,0])[0]
    x2_angle = prinangle(uw2[:,0], ul2[:,0])[0]

    return x1_angle, x2_angle

# for cross-validation - determine acceptable latent dimensionalities to test
def get_dlists(X_1,X_2,across_area_dim,within_area_dim):
    # X_1 and X_2 are spike trains (n1 x N, n2 x N)
    # across_area_dim and within_area_dim are desired maximum dimensionality to test
    import numpy as np
    max_dim = np.minimum(X_1.shape[1],X_2.shape[1]) # max acceptable dim relative to number of neurons
    d = np.minimum(across_area_dim,max_dim)
    d1 = np.minimum(within_area_dim,X_1.shape[1])
    d2 = np.minimum(within_area_dim,X_2.shape[1])
    return (np.arange(d)+1).astype(int),(np.arange(d1)+1).astype(int),(np.arange(d2)+1).astype(int)

# generate Gaussian Process
def gen_GP(GP_tau, GP_len, noise_var=1e-3, seed=0, N=1):
    import numpy as np
    import scipy.linalg as slin
    from scipy.spatial.distance import pdist, squareform
    # generate GP covariance matrix
    t = np.expand_dims(np.arange(GP_len),1)
    dists = squareform(pdist(t))
    K = (1-noise_var)*np.exp( -np.square(dists) / (2*GP_tau**2)) + np.eye(GP_len)*noise_var
    sqrt_K = np.real(slin.sqrtm(K))
    
    # set random seed
    np.random.seed(seed)
    
    # generate GP
    GPs = np.zeros((GP_len,N))
    for i in range(N):
        GPs[:,i] = sqrt_K.dot(np.random.randn(GP_len))
    return GPs.squeeze()

def nansem(data,axis=0):
    # compute standard error of the mean, ignoring NaN values
    import numpy as np
    sem = np.nanstd(data,axis=axis) / np.sqrt(np.count_nonzero(~np.isnan(data),axis=axis))
    return sem

# z-score counts within condition before computing rsc
def zscWithinCond(X, conds):
    # X is n_neurons x n_trials
    import numpy as np
    assert ~np.any(np.isnan(X)), 'Spike counts cannot be NaN'

    cond_labels = np.unique(conds)
    n_cond = len(cond_labels)    
    X_zsc = np.full(X.shape,fill_value=np.nan)

    for i_cond in range(n_cond):
        cond_mask = conds == cond_labels[i_cond]
        curr_counts = X[:,cond_mask]
        cond_mean,cond_std = np.mean(curr_counts,axis=1), np.std(curr_counts,axis=1)
        X_zsc[:,cond_mask] = ((curr_counts.T - cond_mean) / cond_std).T
        if np.any((cond_std == 0) & (cond_mean == 0)):
            chan_mask = np.where((cond_std == 0) & (cond_mean == 0))[0]
            X_zsc[chan_mask,cond_mask] = 0
    return X_zsc