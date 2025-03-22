import scipy.io as sio
import numpy as np
import sys

sys.path.append('helpers/')
import helpers.pcca_fa.pcca_fa_mdl as pf
from dual_pfc_funcs import getParams, save_dict, load_dict

subjects = subjects = getParams()['subjects']
data_path = 'preprocessed_data/'
save_name = '{:s}/fast_null_fits_0_25.pkl'.format(data_path)
print(f"Will save as {save_name}")
    
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
        print('Crossvalidating {:s} ({} of {})...'.format(sess,i_sess,len(fnames)))
        pccafa_params = pccafa_fits[sess]['params'].copy()

        # get data
        curr_dat = getattr(dat,sess)
        # get the lists of dimensions
        z_list,zx_list,zy_list = np.array([pccafa_params['zDim']]), np.array([pccafa_params['zxDim']]), np.array([pccafa_params['zyDim']]),

        # first fit pCCA-FA to fast component (AR-25 removed)
        LH = getattr(curr_dat,'fast_component_left')
        RH = getattr(curr_dat,'fast_component_right')
        # crossvalidate to get canon corr - FLIP control
        mdl = pf.pcca_fa()
        mdl.crossvalidate(LH,RH[::-1,:],zDim_list=z_list,zxDim_list=zx_list,zyDim_list=zy_list,warmstart=True,parallelize=True,early_stop=False,rand_seed=i_sess)
        results[sess]['fast_rho'] = mdl.get_params()['cv_rho']
        results[sess]['fast_params'] = mdl.get_params()

        # now fit pCCA-FA to only mean-subtracted data (no slow removed)
        raw_counts = getattr(curr_dat,'raw_counts') # N x D
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
        mdl.crossvalidate(LH,RH[::-1,:],zDim_list=z_list,zxDim_list=zx_list,zyDim_list=zy_list,warmstart=True,parallelize=True,early_stop=False,rand_seed=i_sess)
        results[sess]['raw_rho'] = mdl.get_params()['cv_rho']
        results[sess]['raw_params'] = mdl.get_params()

        # save the results each iter
        save_dict(results, save_name)