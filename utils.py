# imports
import pickle5 as pickle
import numpy as np

color_map = {
    'across':np.array([255,76,178])/255, # pink
    'within1':np.array([111,192,255])/255, # light blue - right hemisphere
    'within2':np.array([0,87,154])/255, # dark blue - left hemisphere
    'within':np.array([0,144,255])/255, # medium blue - collapsed across both hemispheres
    'independent':np.array([200,200,200])/255 # gray
}

def jitter(length=1):
    spacing = 0.2
    return np.random.uniform(low=-spacing,high=spacing,size=length)

# compute within area rsc for full (multiarea) covariance matrix
def compute_rsc_within_pccafa(Sigma_full, xDim):
    # Sigma is xDim+yDim x xDim+yDim
    Sigma_x = Sigma_full[:xDim,:xDim]
    Sigma_y = Sigma_full[xDim:,xDim:]
    rscs_x, rscs_y = np.array([]), np.array([])

    i_s,j_s = np.tril_indices(xDim, -1)
    for i,j in zip(i_s,j_s):
        sig_ij = Sigma_x[i,j]
        sig_ii = Sigma_x[i,i]
        sig_jj = Sigma_x[j,j]
        rsc = sig_ij / np.sqrt(sig_ii * sig_jj)
        rscs_x = np.append(rscs_x, rsc)
        
    i_s,j_s = np.tril_indices(xDim, -1)
    for i,j in zip(i_s,j_s):
        sig_ij = Sigma_y[i,j]
        sig_ii = Sigma_y[i,i]
        sig_jj = Sigma_y[j,j]
        rsc = sig_ij / np.sqrt(sig_ii * sig_jj)
        rscs_y = np.append(rscs_y, rsc)
    return rscs_x, rscs_y

# compute across area rsc for full (multiarea) covariance matrix
def compute_rsc_across_pccafa(Sigma_full,xDim):
    # Sigma is xDim+yDim x xDim+yDim
    yDim = Sigma_full.shape[0] - xDim
    Sigma = Sigma_full[xDim:,:xDim] # get neurons from opposite regions
    Sigma_x = Sigma_full[:xDim,:xDim]
    Sigma_y = Sigma_full[xDim:,xDim:]
    rscs = np.array([])
    i_s,j_s = np.indices((yDim,xDim))
    for i,j in zip(i_s,j_s):
        sig_ij = Sigma[i,j]
        sig_ii = Sigma_x[i,i]
        sig_jj = Sigma_y[j,j]
        rsc = sig_ij / np.sqrt(sig_ii * sig_jj)
        rscs = np.append(rscs, rsc)
    return rscs
