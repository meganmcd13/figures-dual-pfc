import scipy.io as sio
import numpy as np
import sys

sys.path.append('helpers/pcca_fa/')
import helpers.pcca_fa.pcca_fa_mdl as pf
from dual_pfc_funcs import getParams, save_dict, load_dict

subjects = subjects = getParams()['subjects']
data_path = 'preprocessed_data/'
save_name = '{:s}/fast_null_fits_0_25.pkl'.format(data_path)
print(f"Will save as {save_name}")
    
warmstart = True
results = {}

for sub in subjects:
    pccafa_fits = load_dict(data_path + sub + '_pccafa_cv15dim.pkl')

    # load data
    mat_fname = '{:s}/all_data_delay_{:s}.mat'.format(data_path, sub)
    dat = sio.loadmat(mat_fname,squeeze_me=True,struct_as_record=False)['all_data']
    fnames = dat._fieldnames
    fnames.remove('ar_order')
    fnames.remove('arr_spatial')

    for i_sess,sess in enumerate(fnames,1):
        results[sess] = {}
        print('Fitting {:s} ({} of {})...'.format(sess,i_sess,len(fnames)))
        pccafa_params = pccafa_fits[sess]['params'].copy()

        # get data
        curr_dat = getattr(dat,sess)

        # first fit pCCA-FA to fast component (AR-25 removed)
        LH = getattr(curr_dat,'fast_component_left')
        RH = getattr(curr_dat,'fast_component_right')
        # crossvalidate to get canon corr - FLIP control
        mdl = pf.pcca_fa()
        mdl.train(LH,RH[::-1,:],d=pccafa_params['d'],d1=pccafa_params['d1'],d2=pccafa_params['d2'],warmstart=warmstart,rand_seed=i_sess)
        rho = mdl.compute_cv_canonical_corrs(LH,RH[::-1,:],n_folds=10,rand_seed=i_sess)
        results[sess]['fast_rho'] = rho
        results[sess]['fast_params'] = mdl.get_params()

        # now fit pCCA-FA to only mean-subtracted data (no slow removed)
        raw_counts = getattr(curr_dat,'raw_counts') # N x n1+n2
        # subtract mean by target angle
        all_conds = np.unique(curr_dat.targ_angs)
        counts_mat_nomean = np.zeros_like(raw_counts)
        for cond in all_conds:
            cond_mask = curr_dat.targ_angs==cond
            cond_counts = raw_counts[cond_mask,:]
            cond_mean = np.mean(cond_counts,axis=0)
            counts_mat_nomean[cond_mask,:] = cond_counts - cond_mean

        # separate into LH and RH
        left_idx = curr_dat.arr.LH_idx > 0
        right_idx = curr_dat.arr.RH_idx > 0
        LH = counts_mat_nomean[:,left_idx]
        RH = counts_mat_nomean[:,right_idx]

        # crossvalidate to get canon corr - FLIP control
        mdl = pf.pcca_fa()
        mdl.train(LH,RH[::-1,:],d=pccafa_params['d'],d1=pccafa_params['d1'],d2=pccafa_params['d2'],warmstart=warmstart,rand_seed=i_sess)
        rho = mdl.compute_cv_canonical_corrs(LH,RH[::-1,:],n_folds=10,rand_seed=i_sess)
        results[sess]['raw_rho'] = rho
        results[sess]['raw_params'] = mdl.get_params()

        # save the results each iter
        save_dict(results, save_name)