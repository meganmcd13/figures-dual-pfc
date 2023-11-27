# utilities

def jitter(length=1):
    import numpy as np
    spacing = 0.2
    return np.random.uniform(low=-spacing,high=spacing,size=length)

# compute within area rsc for full (multiarea) covariance matrix
def compute_rsc_within_pccafa(Sigma_full, xDim):
    import numpy as np
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
    import numpy as np
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

# plot a raster on given ax
def plot_raster(X,ax):
    import numpy as np
    # assumes you have already called subplot on the figure that you want
    # X: (num_neurons, num_timepoints)
    X = np.flipud(X)
    num_neurons = X.shape[0]
    num_timepoints = X.shape[1]
    for i_neuron in range(num_neurons):
        for i_time in range(num_timepoints):
            if X[i_neuron,i_time] == 1:
                ax.plot([i_time,i_time], [i_neuron,i_neuron+1],'-k',linewidth=0.3)

# load dictionary using pickle
def load_dict(filename):
    import pickle5 as pickle
    with open(filename, 'rb') as handle:
        data = handle.read()
    # reconstructing the data as dictionary
    return pickle.loads(data)

# flatten a list of lists:
def flatten(xs):
    from collections.abc import Iterable
    for x in xs:
        if isinstance(x, Iterable) and not isinstance(x, (str, bytes)):
            yield from flatten(x)
        else:
            yield x

# extract pCCA-FA model parameters
def extract_mdl_params(fit_dat):
    tmp = fit_dat['params'].item()
    param_dict={x:tmp[x].item() for x,y in tmp.dtype.fields.items()}
    param_dict['mu_x'] = param_dict['mu_x'].flatten()
    param_dict['mu_y'] = param_dict['mu_y'].flatten()
    param_dict['psi_x'] = param_dict['psi_x'].flatten()
    param_dict['psi_y'] = param_dict['psi_y'].flatten()
    param_dict['zDim'] = int(param_dict['zDim'])
    param_dict['zxDim'] = int(param_dict['zxDim'])
    param_dict['zyDim'] = int(param_dict['zyDim'])
    return param_dict

# preprocess spike counts to remove condition means and auto regressive prediction
def preprocess_counts(counts,targ_angs,binsize,ar_order):
    import counts_analysis as cp
    sc_obj1 = cp.counts_analysis(counts,targ_angs,binsize)
    cond_means = sc_obj1.compute_cond_means()
    sc_obj2 = cp.counts_analysis(sc_obj1.rm_cond_means(),targ_angs,binsize)
    _,ar_est = sc_obj2.rm_autoreg(order=ar_order,auto_type='mean',fa_remove=True,fa_dims=15)
    return sc_obj1.rm_cond_means(), ar_est, cond_means

# function for plotting figure 5
def plot_metric(dat,sub,ax,col,sym,metric='psv'):
    import numpy as np
    import pcca_fa_mdl as pf
    if sym=='o':
        jit = -0.1
    elif sym=='s':
        jit = 0
    else:
        jit = 0.1
    # loop through each session
    sess_names = [k for k,v in dat.items() if k.lower().startswith(sub)]
    psv_x,psv_y = np.zeros(len(sess_names)),np.zeros(len(sess_names))
    psv_priv_x,psv_priv_y = np.zeros(len(sess_names)),np.zeros(len(sess_names))
    for i,sess in enumerate(sess_names):
        mdl_params = extract_mdl_params(dat[sess])            
        pf_mdl = pf.pcca_fa()
        pf_mdl.set_params(mdl_params)
        fit_metrics = pf_mdl.compute_metrics()[metric]
        psv_x[i],psv_y[i] = fit_metrics[metric+'_x'],fit_metrics[metric+'_y']
        psv_priv_x[i],psv_priv_y[i] = fit_metrics[metric+'_priv_x'],fit_metrics[metric+'_priv_y']
    psv_within = np.concatenate((psv_priv_x,psv_priv_y))
    psv_across = np.concatenate((psv_x,psv_y))
    if metric=='dshared':
        count_dict = dict()
        for i in range(len(psv_within)):
            tmp = (psv_within[i],psv_across[i])
            if tmp not in count_dict:
                count_dict[tmp] = 1
            else:
                count_dict[tmp] += 1
        for t in count_dict:
            ax.plot(t[0]+jit,t[1]+jit,marker=sym,ls='',color=col,markersize=2+1.5*count_dict[t],fillstyle='none')
    else:
        ax.plot(psv_within,psv_across,marker=sym,ls='',color=col,label='subject {:s}'.format(sess_names[0][0:2]),markersize=5,fillstyle='none')
    return psv_within,psv_across