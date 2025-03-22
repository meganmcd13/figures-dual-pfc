# -- Figure S4 simulated data --
# Structured thetas: 0 - 90 deg
# Model considerations:
    # no cross-validation, no warmstart
    
# imports
from tqdm import tqdm
import sys

sys.path.append('helpers/')
from dual_pfc_funcs import save_dict, getBaseSimParams
import helpers.pcca_fa.sim_pcca_fa as spf
import helpers.pcca_fa.pcca_fa_mdl as pf

# param initialization
warmstart = False
theta_list = [0, 1, 2, 5, 10, 20, 30, 40, 50, 60, 70, 80, 90]

p = getBaseSimParams()
n_trials = p['n_trials']
xDim,yDim = p['xDim'], p['yDim']
zDim,zxDim,zyDim = p['zDim'], p['zxDim'], p['zyDim']
n_boots = p['n_boots']
sv_goal = p['sv_goal']

sim_params_list, est_params_list = [], []

for theta in theta_list:
    print('Currently evaluating theta: {}deg'.format(theta))
    for i in tqdm(range(n_boots)):
        base_simulator = spf.sim_pcca_fa(xDim,yDim,zDim,zxDim,zyDim,equal_eigs=False,sv_goal=sv_goal,theta=theta,rand_seed=i+(theta*n_boots))
        base_params = base_simulator.get_params()
        sim_params_list.append(base_params)
        X,Y = base_simulator.sim_data(n_trials)

        # fit the data using pcca-fa
        model = pf.pcca_fa()
        model.train(X,Y,zDim,zxDim,zyDim,warmstart=warmstart)
        est_params_list.append(model.get_params())

# save parameters
output_dict = {
    "thetas": theta_list,
    "n_boots": n_boots,
    "N": n_trials,
    "sv": sv_goal,
    "sim_params":sim_params_list,
    "est_params":est_params_list,
}

save_name = 'preprocessed_data/simdataset_varyTheta.pkl'
save_dict(output_dict, save_name)
