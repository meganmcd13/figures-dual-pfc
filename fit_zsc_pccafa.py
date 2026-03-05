# -- Figure 4 control analysis --
# fit pCCA-FA z-scored spike counts

import scipy.io as sio
import numpy as np
import pandas as pd
import sys, argparse, time

sys.path.append('helpers/pcca_fa/')
import helpers.pcca_fa.pcca_fa_mdl as pf
from dual_pfc_funcs import get_dlists, save_dict, zscWithinCond

ar_order = 25
max_dim = 15
data_path = 'preprocessed_data/'

def main(subject):
    mat_fname = '{:s}/all_data_delay_{:s}.mat'.format(data_path, subject)

    results = {}
    save_name = '{:s}/{:s}_pccafa_cv{}dim_zsc_fast.pkl'.format(data_path, subject, max_dim)
    print(f"Will save as {save_name}")

    # load data
    dat = sio.loadmat(mat_fname,squeeze_me=True,struct_as_record=False)['all_data']
    fnames = dat._fieldnames
    fnames.remove('ar_order')
    fnames.remove('arr_spatial')

    for i_sess,sess in enumerate(fnames,1):
        print('Crossvalidating {:s} ({} of {})...'.format(sess,i_sess,len(fnames)))

        # get data
        curr_dat = getattr(dat,sess)
        raw_counts = getattr(curr_dat,'raw_counts') # N x n1+n2
        
        # z-score by target angle
        target_labels = curr_dat.targ_angs
        counts_mat_zsc = zscWithinCond(raw_counts.T, target_labels).T

        # remove slow component
        fast_counts_zsc = np.zeros_like(counts_mat_zsc)
        D = counts_mat_zsc.shape[1]
        for i in range(D):
            curr_counts = pd.Series(counts_mat_zsc[:,i])
            est_slow_comp = curr_counts.rolling(ar_order,min_periods=1,center=True).mean()
            fast_counts_zsc[:,i] = curr_counts - est_slow_comp
        
        # separate into LH and RH
        left_idx = curr_dat.arr.LH_idx > 0
        right_idx = curr_dat.arr.RH_idx > 0
        LH = fast_counts_zsc[:,left_idx]
        RH = fast_counts_zsc[:,right_idx]

        # get the lists of dimensions to cross-validate over
        d_list,d1_list,d2_list = get_dlists(LH,RH,max_dim,max_dim)

        # crossvalidate
        pcca_fa_mdl = pf.pcca_fa()
        cvLL = pcca_fa_mdl.crossvalidate(LH,RH,d_list=d_list,d1_list=d1_list,d2_list=d2_list,warmstart=True,parallelize=True,early_stop=False,rand_seed=i_sess)

        # put cross-validation results into dict
        results[sess] = {
            'cvLL': cvLL, 
            'params': pcca_fa_mdl.get_params(),
        }

        # save the results each iter
        save_dict(results, save_name)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-subject',type=str,help='name of monkey ("wakko", "satchel", or "pepe")',required=True)
    
    args = parser.parse_args()
    t = time.time()
    main(args.subject)
    print("elapsed time: ", time.time()-t)