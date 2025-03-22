# -- Figure S3 simulated data --
# Structured trial counts: 100 - 2000
# Model considerations:
    # 5-fold cross-validation, no warmstart
    # also fit pCCA to compare in simulated data
    
# imports
import numpy as np
import sys, time

sys.path.append('helpers/')
import helpers.pcca_fa.sim_pcca_fa as spf
import helpers.pcca_fa.pcca_fa_mdl as pf
import helpers.cca.prob_cca as pcca
from dual_pfc_funcs import getBaseSimParams, save_dict

# param initialization
flat_eigs = True
cv_list = np.arange(15) # dimensionalities to test in cross-validation

p = getBaseSimParams()
xDim,yDim = p['xDim'], p['yDim']
zDim,zxDim,zyDim = p['zDim'], p['zxDim'], p['zyDim']
n_boots = 30 # decrease since we are cross-validating
sv_goal = p['sv_goal']
n_trials = np.array([100,150,300,600,1000,1500])
n_folds = 5

# vary number of trials
def run_vary_N():
    save_name = 'preprocessed_data/simdataset_varyN_noWS.pkl'
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

            simulator = spf.sim_pcca_fa(xDim,yDim,zDim,zxDim,zyDim,flat_eigs=flat_eigs,equal_eigs=False,sv_goal=sv_goal)
            X,Y = simulator.sim_data(N)

            # fit the data using pcca-fa
            model = pf.pcca_fa()

            model.set_params(simulator.get_params())
            model.crossvalidate(X,Y,zDim_list=cv_list,zxDim_list=cv_list,zyDim_list=cv_list,n_folds=n_folds,warmstart=False,parallelize=True,rand_seed=seed)

            # fit the data using pcca
            pcca_model = pcca.prob_cca()
            pcca_model.crossvalidate(X,Y,zDim_list=cv_list,rand_seed=seed,n_folds=n_folds,parallelize=True,warmstart=False)

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