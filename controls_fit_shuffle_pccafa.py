import scipy.io as sio
import numpy as np
import sys, argparse, time

sys.path.append('helpers/pcca_fa/')
import helpers.pcca_fa.pcca_fa_mdl as pf
import helpers.pcca_fa.sim_pcca_fa as spf
from dual_pfc_funcs import get_dlists, save_dict, getBaseSimParams

def main(subject):
    max_dim = 15

    data_path = 'preprocessed_data/'
    mat_fname = '{:s}/all_data_delay_{:s}.mat'.format(data_path, subject)

    results = {}
    save_name = '{:s}/{:s}_pccafa_cv{}dim_shuffle.pkl'.format(data_path, subject, max_dim)
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
        tmp_LH = getattr(curr_dat,'fast_component_left') # N x n1
        tmp_RH = getattr(curr_dat,'fast_component_right') # N x n2
        n1 = curr_dat.n_arr1
        n2 = curr_dat.n_arr2

        # shuffle hemisphere labels
        all_counts = np.concatenate((tmp_LH,tmp_RH),axis=1)
        permuted_labels = np.random.permutation(n1+n2)
        LH = all_counts[:,permuted_labels[:n1]]
        RH = all_counts[:,permuted_labels[n1:]]        

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


def sim_main():
    data_path = 'preprocessed_data/'

    save_name = '{:s}/simdataset_pccafa_shuffle.pkl'.format(data_path)
    print(f"Will save as {save_name}")

    # param initialization
    warmstart = False
    flat_eigs = True

    p = getBaseSimParams()
    n1,n2 = p['n1'], p['n2']
    d,d1,d2 = p['d'], p['d1'], p['d2']
    n_boots = p['n_boots']
    sv_goal = p['sv_goal']
    N = p['n_trials']

    output_dict = {
        "n_boots": n_boots,
        "N": [],
        "sv": sv_goal,
        "sim_params": [],
        "est_params": [],
        "shuffle_params": [],
    }

    for i in range(n_boots):
        seed = n_boots + i
        print('Currently evaluating boot {} of {}'.format(i+1,n_boots))

        across_sv = np.random.uniform(low=5,high=30)
        within_sv = np.random.uniform(low=5,high=30)

        simulator = spf.sim_pcca_fa(n1,n2,d,d1,d2,flat_eigs=flat_eigs,sv_goal=[across_sv,within_sv],rand_seed=seed)
        X_1,X_2 = simulator.sim_data(N)

        # shuffle hemisphere labels
        all_counts = np.concatenate((X_1.copy(),X_2.copy()),axis=1)
        permuted_labels = np.random.permutation(n1+n2)
        X_1_shuffle = all_counts[:,permuted_labels[:n1]]
        X_2_shuffle = all_counts[:,permuted_labels[n1:]]        

        # fit pCCA-FA
        mdl = pf.pcca_fa()
        mdl.train(X_1,X_2,d,d1,d2,warmstart=warmstart)

        shuffle_mdl = pf.pcca_fa()
        shuffle_mdl.train(X_1_shuffle,X_2_shuffle,d,d1,d2,warmstart=warmstart)

        # save parameters
        output_dict["sim_params"].append(simulator.get_params())
        output_dict["est_params"].append(mdl.get_params())
        output_dict["shuffle_params"].append(shuffle_mdl.get_params())
        output_dict["N"].append(N)

        # save the results each iter
        save_dict(output_dict, save_name)

if __name__ == '__main__':
    # parser = argparse.ArgumentParser()
    # parser.add_argument('-subject',type=str,help='name of monkey ("wakko", "satchel", or "pepe")',required=True)
    
    # args = parser.parse_args()
    # t = time.time()
    # main(args.subject)

    t = time.time()
    sim_main()

    print("elapsed time: ", time.time()-t)