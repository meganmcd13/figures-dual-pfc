# -- Figure S2 simulated data --
# Structured sv:
    # across > within
    # across = within
    # across < within
# Model considerations:
    # 5-fold cross-validation, no warmstart
    
import numpy as np
import sys, argparse, time

sys.path.append('helpers/pcca_fa/')
import helpers.pcca_fa.sim_pcca_fa as spf
import helpers.pcca_fa.pcca_fa_mdl as pf
from dual_pfc_funcs import getBaseSimParams, save_dict

# param initialization
warmstart = False
flat_eigs = True
cv_list = np.arange(10).astype(int) # dimensionalities to test in cross-validation

p = getBaseSimParams()
n1,n2 = p['n1'], p['n2']
d,d1,d2 = p['d'], p['d1'], p['d2']
n_boots = 30 # decrease since we are cross-validating

def run_vary_sv(n_trials):
    config_sv = [(5,20),(15,15),(20,5)] # (across,within) sv for each config
    seeds = [224, 5, 24] # cherry picked seeds to get close sv goal

    save_name = 'preprocessed_data/simdataset_varySv_noWS_n{}.pkl'.format(n_trials)
    print('Will save as {}'.format(save_name))

    output_dict = {
        "sv_changed": 'both',
        "n_boots": n_boots,
        "N": n_trials,
        "sv_list": config_sv,
        "sim_params": [],
        "est_params": []
    }

    for sv_goal,seed in zip(config_sv,seeds):
        print('Currently evaluating sv: {}'.format(sv_goal))
        simulator = spf.sim_pcca_fa(n1,n2,d,d1,d2,flat_eigs=flat_eigs,sv_goal=sv_goal,rand_seed=seed) # one GT sim
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
    run_vary_sv(args.N)
    print("elapsed time: ", time.time()-t)