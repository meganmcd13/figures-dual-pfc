# -- Figure 2 analysis --
# compute across- and within-area spike count correlations (rsc) and signal correlations

import numpy as np
import pandas as pd
import scipy.io as sio

# helper functions
from dual_pfc_funcs import getParams, save_dict, zscWithinCond

# find pairs of neurons across areas with significant signal tuning
def getTuningCurves( X, conds ):
    import numpy as np

    n_neurons,_ = X.shape
    cond_labels = np.unique(conds)
    n_cond = len(cond_labels)
    
    tune_curves = np.zeros((n_neurons,n_cond))
    tune_curves[:] = np.nan

    for i_cond in range(n_cond):
        cond_mask = conds == cond_labels[i_cond]
        curr_counts = X[:,cond_mask]
        tune_curves[:,i_cond] = np.mean(curr_counts,axis=1)

    return tune_curves

subjects = getParams()['subjects']

n_perms = 1000
sig_level = 0.01
columns=['Subject', 'SessionName', 'WithinAreaLeftRsc', 'WithinAreaRightRsc', 'AcrossAreaRsc', 'AcrossAreaRscSimilar', 'AcrossAreaRscOpposite', 'WithinAreaLeftRscZsc', 'WithinAreaRightRscZsc', 'AcrossAreaRscZsc', 'AcrossAreaRscSimilarZsc', 'AcrossAreaRscOppositeZsc']
df = pd.DataFrame(columns=columns)

within_area_left_dists = []
within_area_right_dists = []
across_area_dists = []
for subject in subjects:
    # load all data to get counts
    mat_fname = "preprocessed_data/all_data_delay_" + subject + ".mat"
    dat = sio.loadmat(mat_fname,squeeze_me=True,struct_as_record=False)['all_data']
    fnames = dat._fieldnames
    fnames.remove('arr_spatial')
    fnames.remove('ar_order')

    # go through sessions and compute rsc
    for sess in fnames:
        print('Computing rsc for session: {}'.format(sess))
        curr_dat = getattr(dat,sess)
        df2 = {'Subject':subject, 'SessionName':sess}

        # sess info
        counts_mat = curr_dat.raw_counts
        targ_angs = curr_dat.targ_angs
        bin_size = curr_dat.binsize
        left_idx = np.where(curr_dat.arr.LH_idx > 0)[0]
        right_idx = np.where(curr_dat.arr.RH_idx > 0)[0]
        n_trials, n_neurons = counts_mat.shape
        n_left = len(left_idx)
        n_right = len(right_idx)

        # 1) compute signal correlation on raw counts (no tuning removed)
        tune_curves = getTuningCurves(counts_mat.T,targ_angs)
        sig_corr = np.corrcoef(tune_curves)
        sig_corr_acc = sig_corr[right_idx[0]:right_idx[-1]+1,left_idx[0]:left_idx[-1]+1] # get across-area pairs

        # 1a) compute chance levels for signal correlation - shuffle labels
        X = counts_mat.T.copy()
        null_delay = np.full((n_neurons,n_neurons,n_perms),fill_value=np.nan)
        for ii in range(n_perms):
            rng_delay = X.copy()
            for i_chan in range(n_neurons):
                perm = np.random.permutation(n_trials)
                rng_delay[i_chan,:] = rng_delay[i_chan,perm] # permute trials
            tmp_curves = getTuningCurves(rng_delay,targ_angs)
            null_delay[:,:,ii] = np.corrcoef(tmp_curves)
        null_acc = null_delay[right_idx[0]:right_idx[-1]+1,left_idx[0]:left_idx[-1]+1,:]

        # 1b) compute p-values for each correlation value
        rep_sig_corr_acc = np.tile(sig_corr_acc[:,:,np.newaxis],(1,1,n_perms))
        sig_diff_acc = rep_sig_corr_acc > null_acc
        p_acc = np.mean(sig_diff_acc,axis=2)
        p_acc[p_acc>0.5] = 1 - p_acc[p_acc>0.5]
        # find significant pairs
        sig_pairs_idx = (p_acc<sig_level) & (sig_corr_acc>0)
        opp_pairs_idx = (p_acc<sig_level) & (sig_corr_acc<0)

        # 2) compute rsc on preprocessed counts (with tuning and slow process removed)
        if left_idx[0] < right_idx[0]: # find out which array came first
            X = np.concatenate((curr_dat.fast_component_left,curr_dat.fast_component_right),axis=1)
        else:
            X = np.concatenate((curr_dat.fast_component_right,curr_dat.fast_component_left),axis=1)
        rsc = np.corrcoef(X.T) # n_neurons x n_neurons
        rsc_L = rsc[left_idx[0]:left_idx[-1]+1,left_idx[0]:left_idx[-1]+1]
        rsc_R = rsc[right_idx[0]:right_idx[-1]+1,right_idx[0]:right_idx[-1]+1]
        rsc_acc = rsc[right_idx[0]:right_idx[-1]+1,left_idx[0]:left_idx[-1]+1]

        # get rsc of significant sig corr pairs
        sim_corrs_acc = rsc_acc[sig_pairs_idx]
        opp_corrs_acc = rsc_acc[opp_pairs_idx]

        # only keep rscs between neurons (not auto-correlations or duplicates)
        rsc_L = rsc_L[np.triu_indices(n_left, k=1)] 
        rsc_R = rsc_R[np.triu_indices(n_right, k=1)]
        rsc_acc = rsc_acc.reshape(-1)

        within_area_left_dists.append(rsc_L)
        within_area_right_dists.append(rsc_R)
        across_area_dists.append(rsc_acc)

        df2 = {**df2, 'WithinAreaLeftRsc':np.mean(rsc_L), 'WithinAreaRightRsc':np.mean(rsc_R), 'AcrossAreaRsc':np.mean(rsc_acc),'AcrossAreaRscSimilar':np.mean(sim_corrs_acc), 'AcrossAreaRscOpposite':np.mean(opp_corrs_acc)}

        # 3) repeat rsc computation but with z-scored raw counts instead of preprocessed counts
        X = zscWithinCond(counts_mat.T, targ_angs) # z-score within condition
        rsc = np.corrcoef(X)
        rsc_L = rsc[left_idx[0]:left_idx[-1]+1,left_idx[0]:left_idx[-1]+1]
        rsc_R = rsc[right_idx[0]:right_idx[-1]+1,right_idx[0]:right_idx[-1]+1]
        rsc_acc = rsc[right_idx[0]:right_idx[-1]+1,left_idx[0]:left_idx[-1]+1]
        sim_corrs_acc = rsc_acc[sig_pairs_idx]
        opp_corrs_acc = rsc_acc[opp_pairs_idx]
        rsc_L = rsc_L[np.triu_indices(n_left, k=1)] # only keep rscs between neurons
        rsc_R = rsc_R[np.triu_indices(n_right, k=1)]
        rsc_acc = rsc_acc.reshape(-1)

        df2 = {**df2, 'WithinAreaLeftRscZsc':np.mean(rsc_L), 'WithinAreaRightRscZsc':np.mean(rsc_R), 'AcrossAreaRscZsc':np.mean(rsc_acc),'AcrossAreaRscSimilarZsc':np.mean(sim_corrs_acc), 'AcrossAreaRscOppositeZsc':np.mean(opp_corrs_acc)}
        df.loc[len(df)] = df2

# save out rsc data
dists_dict = {
    'SessionNames': df['SessionName'].to_list(),
    'WithinAreaLeftRsc': within_area_left_dists,
    'WithinAreaRightRsc': within_area_right_dists,
    'AcrossAreaRsc': across_area_dists,
}
save_name = 'preprocessed_data/within_across_rsc_distributions.pkl'
save_dict(dists_dict,save_name)

save_name = 'preprocessed_data/within_across_rsc_means.pkl'
df.to_pickle(save_name, compression='gzip')