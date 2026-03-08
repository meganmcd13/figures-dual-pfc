# -- Figure S4 analysis --
# fit alternative models (FA and pCCA) to same data as pCCA-FA

import numpy as np
import scipy.io as sio
import sys

sys.path.append('../helpers/')
sys.path.append('../helpers/pcca_fa/')
import pcca_fa.fa.factor_analysis as fa
import pcca_fa.cca.prob_cca as pcca
from dual_pfc_funcs import getParams, save_dict

subjects = getParams()['subjects']
max_joint_dim = 30
max_ind_dim = 20
early_stop = False

data_path = '../preprocessed_data/'
save_name = '{:s}/all_alt_mdls_cv.pkl'.format(data_path)
print(f"Will save as {save_name}")

results = {}

for sub in subjects:
    mat_fname = '{:s}/all_data_delay_{:s}.mat'.format(data_path, sub)

    # load data
    dat = sio.loadmat(mat_fname,squeeze_me=True,struct_as_record=False)['all_data']
    fnames = dat._fieldnames
    fnames.remove('ar_order')
    fnames.remove('arr_spatial')

    for i_sess,sess in enumerate(fnames,1):
        print('Crossvalidating {:s} ({} of {})...'.format(sess,i_sess,len(fnames)))

        # get data
        curr_dat = getattr(dat,sess)
        LH = getattr(curr_dat,'fast_component_left')
        RH = getattr(curr_dat,'fast_component_right')
        dmax = np.min([max_joint_dim, LH.shape[1]+RH.shape[1]])
        joint_d_list = (np.arange(dmax)+1).astype(int)
        dmax = np.min([max_ind_dim, LH.shape[1], RH.shape[1]])
        ind_d_list = (np.arange(dmax)+1).astype(int)

        # mdl 1: joint FA
        print('  Running Joint FA model...')
        joint_samples = np.concatenate((LH,RH),axis=1)
        joint_model = fa.factor_analysis(model_type='fa')
        cv_faMdl = joint_model.crossvalidate(joint_samples,zDim_list=joint_d_list,early_stop=early_stop,rand_seed=i_sess,parallelize=True)
        joint_LLs = cv_faMdl['LLs']

        # mdl 2: individual FA
        print('  Running Individual FA model...')
        model_x1 = fa.factor_analysis(model_type='fa')
        cv_faMdl_x1 = model_x1.crossvalidate(LH,zDim_list=ind_d_list,early_stop=early_stop,rand_seed=i_sess,parallelize=True)
        LLs_x1 = cv_faMdl_x1['LLs']

        model_x2 = fa.factor_analysis(model_type='fa')
        cv_faMdl_x2 = model_x2.crossvalidate(RH,zDim_list=ind_d_list,early_stop=early_stop,rand_seed=i_sess,parallelize=True)
        LLs_x2 = cv_faMdl_x2['LLs']

        # mdl 3: pCCA
        print('  Running pCCA model...')
        pcca_model = pcca.prob_cca()
        LLs_pcca = pcca_model.crossvalidate(LH,RH,zDim_list=ind_d_list,rand_seed=i_sess,parallelize=True,warmstart=False)

        # put cross-validation results into dict
        session_dict = {
            "n_trials": LH.shape[0],
            "n_neurons_left": LH.shape[1],
            "n_neurons_right": RH.shape[1],

            "joint_fa":joint_model.get_params(),
            "joint_LL":joint_LLs,

            "ind_fa_LH":model_x1.get_params(),
            "ind_fa_LH_LL":LLs_x1,

            "ind_fa_RH":model_x2.get_params(),
            "ind_fa_RH_LL":LLs_x2,

            "pcca":pcca_model.get_params(),
            "pcca_LL":LLs_pcca,
        }
        results[sess] = session_dict
        
        # save the results each iter
        save_dict(results, save_name)