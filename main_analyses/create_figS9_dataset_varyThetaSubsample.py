# -- Figure S9 simulated data --
# Structured thetas: 
    # 0 - 90 deg
# Model considerations:
    # no cross-validation, no warmstart
    
# imports
from tqdm import tqdm
import sys
import numpy as np

sys.path.append('../helpers/')
sys.path.append('../helpers/pcca_fa/')
import pcca_fa.sim_pcca_fa as spf
import pcca_fa.pcca_fa_mdl as pf
from dual_pfc_funcs import save_dict, getBaseSimParams

# param initialization
warmstart = False

p = getBaseSimParams()
n_trials = p['n_trials']
n1,n2 = p['n1'], p['n2']
d,d1,d2 = p['d'], p['d1'], p['d2']
n_boots = p['n_boots']
sv_goal = p['sv_goal']

# resample subpopulations of neurons
theta_list = [0,30,60,90]
base_params_list, sim_params_list, est_params_list = [], [], []
n_full = 10000
n_sub = 100

rng = np.random.default_rng()
for theta in theta_list:
    print('Currently evaluating theta: {}deg'.format(theta))
    base_simulator = spf.sim_pcca_fa(n_full,n_full,d,d1,d2,sv_goal=sv_goal,theta=theta,rand_seed=theta)
    params = base_simulator.get_params().copy()
    base_params_list.append(params)

    # subsample 100 neurons that the array can pull from
    daily_population1 = rng.choice(n_full, n_sub, replace=False)
    daily_population2 = rng.choice(n_full, n_sub, replace=False)

    tmp_W_1 = params['W_1'].copy()[daily_population1,:]
    tmp_L_1 = params['L_1'].copy()[daily_population1,:]
    tmp_mu_x1 = params['mu_x1'].copy()[daily_population1]
    tmp_psi_1 = params['psi_1'].copy()[daily_population1]
    tmp_W_2 = params['W_2'].copy()[daily_population2,:]
    tmp_L_2 = params['L_2'].copy()[daily_population2,:]
    tmp_mu_x2 = params['mu_x2'].copy()[daily_population2]
    tmp_psi_2 = params['psi_2'].copy()[daily_population2]

    L_top = np.concatenate((tmp_W_1,tmp_L_1,np.zeros((n_sub,d2))),axis=1)
    L_bottom = np.concatenate((tmp_W_2,np.zeros((n_sub,d1)),tmp_L_2),axis=1)
    L_total = np.concatenate((L_top,L_bottom),axis=0)

    params['W_1'] = tmp_W_1
    params['W_2'] = tmp_W_2
    params['L_1'] = tmp_L_1
    params['L_2'] = tmp_L_2
    params['L_total'] = L_total
    params['mu_x1'] = tmp_mu_x1
    params['mu_x2'] = tmp_mu_x2
    params['psi_1'] = tmp_psi_1
    params['psi_2'] = tmp_psi_2

    for i in tqdm(range(n_boots)):
        # subsample small number for day-to-day populations
        neuron_to_keep = rng.choice(n_sub, n1, replace=False)
        tmp_W_1 = params['W_1'].copy()[neuron_to_keep,:]
        tmp_L_1 = params['L_1'].copy()[neuron_to_keep,:]
        tmp_mu_x1 = params['mu_x1'].copy()[neuron_to_keep]
        tmp_psi_1 = params['psi_1'].copy()[neuron_to_keep]

        L_top = np.concatenate((tmp_W_1,tmp_L_1,np.zeros((n1,d2))),axis=1)
        L_bottom = np.concatenate((params['W_2'],np.zeros((n_sub,d1)),params['L_2']),axis=1)
        L_total = np.concatenate((L_top,L_bottom),axis=0)

        tmp_params = params.copy()
        tmp_params['W_1'] = tmp_W_1
        tmp_params['L_1'] = tmp_L_1
        tmp_params['L_total'] = L_total
        tmp_params['mu_x1'] = tmp_mu_x1
        tmp_params['psi_1'] = tmp_psi_1

        # generate trials using the days parameters
        sim = spf.sim_pcca_fa(n1,n_sub,d,d1,d2)
        sim.set_params(tmp_params)
        sim_params_list.append(tmp_params)
        X_1,X_2 = sim.sim_data(n_trials)

        # fit the data using pcca-fa
        model = pf.pcca_fa()
        model.train(X_1,X_2,d,d1,d2,warmstart=warmstart)
        est_params_list.append(model.get_params())

# save parameters
output_dict = {
    "thetas": theta_list,
    "n_boots": n_boots,
    "N": n_trials,
    "sv": sv_goal,
    "base_params":base_params_list,
    "sim_params":sim_params_list,
    "est_params":est_params_list,
}

save_name = '../preprocessed_data/simdataset_varyThetaSubsample.pkl'
save_dict(output_dict, save_name)