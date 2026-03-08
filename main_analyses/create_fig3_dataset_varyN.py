# -- Figure 3 simulated data --
# Structured trial counts: 100 - 1500
# Model considerations:
    # 5-fold cross-validation, no warmstart
    # also fit pCCA to compare in simulated data
    
import numpy as np
import sys, time

sys.path.append('../helpers/')
sys.path.append('../helpers/pcca_fa/')
import pcca_fa.sim_pcca_fa as spf
import pcca_fa.pcca_fa_mdl as pf
import pcca_fa.cca.prob_cca as pcca
from dual_pfc_funcs import getBaseSimParams, save_dict

# param initialization
flat_eigs = True
cv_list = np.arange(15).astype(int) # dimensionalities to test in cross-validation

p = getBaseSimParams()
n1,n2 = p['n1'], p['n2']
d,d1,d2 = p['d'], p['d1'], p['d2']
n_boots = 30 # decrease since we are cross-validating
sv_goal = p['sv_goal']
n_trials = np.array([100,150,300,600,1000,1500])
n_folds = 5

# vary number of trials
def run_vary_N():
    save_name = '../preprocessed_data/simdataset_varyN_noWS.pkl'
    print('Will save as {}'.format(save_name))

    output_dict = {
        "n_boots": n_boots,
        "N": [],
        "sv": sv_goal,
        "sim_params": [],
        "est_params": [],
        "pcca_params": []
    }
    
    for i in range(n_boots):
        for j,N in enumerate(n_trials):
            seed = j*n_boots + i
            print('Currently evaluating boot {} of {}, N = {}'.format(i+1,n_boots,N))

            simulator = spf.sim_pcca_fa(n1,n2,d,d1,d2,flat_eigs=flat_eigs,sv_goal=sv_goal)
            X_1,X_2 = simulator.sim_data(N)

            # fit the data using pcca-fa
            model = pf.pcca_fa()
            model.crossvalidate(X_1,X_2,d_list=cv_list,d1_list=cv_list,d2_list=cv_list,n_folds=n_folds,warmstart=False,parallelize=True,rand_seed=seed)

            # fit the data using pcca
            pcca_model = pcca.prob_cca()
            pcca_model.crossvalidate(X_1,X_2,zDim_list=cv_list,rand_seed=seed,n_folds=n_folds,parallelize=True,warmstart=False)

            # save parameters
            output_dict["sim_params"].append(simulator.get_params())
            output_dict["est_params"].append(model.get_params())
            output_dict["pcca_params"].append(pcca_model.get_params())
            output_dict["N"].append(N)

            # save the results each iter
            save_dict(output_dict, save_name)


if __name__ == '__main__':
    t = time.time()
    run_vary_N()
    print("elapsed time: ", time.time()-t)