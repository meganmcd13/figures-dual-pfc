# -- Figure S2 simulated data --
# Structured dimensionality: 
    # across > within
    # across = within
    # across < within
# Model considerations:
    # 5-fold cross-validation, no warmstart
    
import numpy as np
import sys, time, argparse

sys.path.append('../helpers/')
sys.path.append('../helpers/pcca_fa/')
import pcca_fa.sim_pcca_fa as spf
import pcca_fa.pcca_fa_mdl as pf
from dual_pfc_funcs import getBaseSimParams, save_dict

# param initialization
warmstart = False
flat_eigs = True
cv_list = np.arange(10).astype(int) # dimensionalities to test in cross-validation

p = getBaseSimParams()
n1,n2 = p['n1'], p['n2']
n_boots = 30 # decrease since we are cross-validating
sv_goal = p['sv_goal']

def run_vary_dim(n_trials):
    config_dims = [(1,5),(3,3),(5,1)] # (across,within) dims for each config

    save_name = '../preprocessed_data/simdataset_varyDim_noWS_n{}.pkl'.format(n_trials)
    print('Will save as {}'.format(save_name))

    output_dict = {
        "dim_changed": 'both',
        "n_boots": n_boots,
        "N": n_trials,
        "sv": sv_goal,
        "dim_list": config_dims,
        "sim_params": [],
        "est_params": []
    }

    for dims in config_dims:
        print('Currently evaluating dims: {}'.format(dims))
        d = dims[0]
        d1,d2 = dims[1],dims[1]
        simulator = spf.sim_pcca_fa(n1,n2,d,d1,d2,flat_eigs=flat_eigs,sv_goal=sv_goal)
        output_dict["sim_params"].append(simulator.get_params())
        for i in range(n_boots):
            if i%10 == 0: print('  sim {} of {}'.format(i+1, n_boots))
            
            # simulate new independent dataset
            X_1,X_2 = simulator.sim_data(n_trials)

            # fit the data using pcca-fa
            model = pf.pcca_fa()
            model.crossvalidate(X_1,X_2,d_list=cv_list,d1_list=cv_list,d2_list=cv_list,n_folds=5,warmstart=warmstart,parallelize=True)

            # save parameters
            output_dict["est_params"].append(model.get_params())

            # save the results each iter
            save_dict(output_dict, save_name)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-N',type=int,help='number of trials',required=True)
    
    args = parser.parse_args()
    t = time.time()
    run_vary_dim(args.N)
    print("elapsed time: ", time.time()-t)