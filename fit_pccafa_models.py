# -- MAIN analysis --
# fit pCCA-FA to neural recordings

import scipy.io as sio
import sys, argparse, time

sys.path.append('helpers/pcca_fa/')
import helpers.pcca_fa.pcca_fa_mdl as pf
from dual_pfc_funcs import get_dlists, save_dict

def main(subject):
    max_dim = 15

    data_path = 'preprocessed_data/'
    mat_fname = '{:s}/all_data_delay_{:s}.mat'.format(data_path, subject)

    results = {}
    save_name = '{:s}/{:s}_pccafa_cv{}dim.pkl'.format(data_path, subject, max_dim)
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
        LH = getattr(curr_dat,'fast_component_left')
        RH = getattr(curr_dat,'fast_component_right')

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