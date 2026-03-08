# -- Figure S1 analysis --
# fit pCCA-FA to slow component of neural activity

import scipy.io as sio
import numpy as np
import sys

sys.path.append('../helpers/')
sys.path.append('../helpers/pcca_fa/')
import pcca_fa.pcca_fa_mdl as pf
from dual_pfc_funcs import getParams, save_dict, load_dict

subjects = subjects = getParams()['subjects']
data_path = '../preprocessed_data/'
save_name = '{:s}/slow_flip_fits.pkl'.format(data_path)
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
        d_list,d1_list,d2_list = np.array([pccafa_params['d']]), np.array([pccafa_params['d1']]), np.array([pccafa_params['d2']])

        # first fit pCCA-FA to slow component (AR-25)
        LH = getattr(curr_dat,'slow_component_left')
        RH = getattr(curr_dat,'slow_component_right')
        # crossvalidate to get canon corr - 
        mdl = pf.pcca_fa()
        mdl.crossvalidate(LH,RH,d_list=d_list,d1_list=d1_list,d2_list=d2_list,warmstart=True,parallelize=True,early_stop=False,rand_seed=i_sess)
        results[sess]['slow_rho'] = mdl.get_params()['cv_rho']
        results[sess]['slow_params'] = mdl.get_params()

        # now for pCCA-FA to flipped slow component
        # crossvalidate to get canon corr - FLIP control
        mdl = pf.pcca_fa()
        mdl.crossvalidate(LH,RH[::-1,:],d_list=d_list,d1_list=d1_list,d2_list=d2_list,warmstart=True,parallelize=True,early_stop=False,rand_seed=i_sess)
        results[sess]['slow_flip_rho'] = mdl.get_params()['cv_rho']
        results[sess]['slow_flip_params'] = mdl.get_params()

        # save the results each iter
        # save_dict(results,save_name)