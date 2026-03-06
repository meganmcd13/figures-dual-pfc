# -- Figure 5 simulated data --
# Structured thetas:
    # 30, 60, 90 deg
# Random thetas:
    # uniform (45-90 deg)
# Model considerations:
    # no cross-validation, using params from neural recordings, no warmstart
    # manipulate each area individually, i.e., two sims per session
    
import sys
import numpy as np
import scipy.io as sio

sys.path.append('helpers/pcca_fa/')
from dual_pfc_funcs import get_top_angle, getParams, save_dict, load_dict
import helpers.pcca_fa.pcca_fa_mdl as pf
import helpers.pcca_fa.sim_pcca_fa as spf

# parameters from neural recordings
subjects = getParams()['subjects']
data_path = 'preprocessed_data/'
save_name = data_path + 'simdataset_varyThetaShuffle.pkl'

# sim params
np.random.seed(4) # to get reproducible uniform thetas
NSIM = 42 # number of sessions
thetas_struct = [30, 60, 90] # degrees, use same angle for both areas
rand_low, rand_high = 45, 90
thetas_rand = np.random.uniform(low=rand_low,high=rand_high,size=(NSIM,2)) # degrees, col 1 for area 1, col 2 for area 2

results = {}
for subject in subjects:
    # load data
    pccafa_fits = load_dict(data_path + subject + '_pccafa_cv15dim.pkl')

    mat_fname = '{:s}/all_data_delay_{:s}.mat'.format(data_path, subject)
    dat = sio.loadmat(mat_fname,squeeze_me=True,struct_as_record=False)['all_data']
    fnames = dat._fieldnames
    fnames.remove('ar_order')
    fnames.remove('arr_spatial')

    for i_sess,sess in enumerate(fnames):
        results[sess] = {}
        print('Session {:s} ({} of {})...'.format(sess,i_sess+1,len(fnames)))

        # get data
        curr_dat = getattr(dat,sess)
        LH = getattr(curr_dat,'fast_component_left') # area 1
        RH = getattr(curr_dat,'fast_component_right') # area 2
        n_trials = curr_dat.n_trials
        n1,n2 = LH.shape[1],RH.shape[1]

        # get pCCA-FA params
        params = pccafa_fits[sess]['params'].copy()
        d,d1,d2 = params['d'],params['d1'],params['d2']
        x1_angle_gt, x2_angle_gt = get_top_angle(params.copy(),across_mode='cov')
        results[sess]['neural_x1'] = x1_angle_gt
        results[sess]['neural_x2'] = x2_angle_gt

        mdl = pf.pcca_fa()
        mdl.set_params(params.copy())
        psv_gt = mdl.compute_psv()
        shared_within_x1_orig = np.diag(params['L_1'] @ params['L_1'].T)
    
        # simulator
        simulator = spf.sim_pcca_fa(n1,n2,d,d1,d2)
        simulator.set_params(params.copy())
        tmp_x1,tmp_x2 = get_top_angle(simulator.get_params())
        assert(np.isclose(tmp_x1,x1_angle_gt,rtol=1e-5) and np.isclose(tmp_x2,x2_angle_gt,rtol=1e-5))

        #########################################
        # option 1: adjust L to have random theta
        print('   manipulating area 1')
        simulator.apply_rotation(thetas_rand[i_sess,0],hem='1')
        tmp,_ = get_top_angle(simulator.get_params())
        assert(np.isclose(tmp,thetas_rand[i_sess,0],rtol=1e-5))

        new_params = simulator.get_params().copy()
        mdl = pf.pcca_fa()
        mdl.set_params(new_params)
        psv_rot = mdl.compute_psv()
        shared_within_x1_new = np.diag(new_params['L_1'] @ new_params['L_1'].T)
        print('  %sv across: {:.2f}, %sv within: {:.2f}'.format(psv_gt['avg_psv_W_1'], psv_gt['avg_psv_L_1']))
        print('  ROTATED %sv across: {:.2f}, %sv within: {:.2f}'.format(psv_rot['avg_psv_W_1'], psv_rot['avg_psv_L_1']))

        X_1,X_2 = simulator.sim_data(n_trials)
    
        # fit the data using pcca-fa
        model = pf.pcca_fa()
        model.train(X_1,X_2,d,d1,d2,warmstart=False)
        fit_params = model.get_params()
    
        # find estimated angles for area 1
        x1_angle,_ = get_top_angle(fit_params)
        results[sess]['gt_rand_x1'] = thetas_rand[i_sess,0]
        results[sess]['fit_rand_x1'] = x1_angle

        print('   manipulating area 2')
        simulator.set_params(params.copy())
        tmp_x1,tmp_x2 = get_top_angle(simulator.get_params())
        assert(np.isclose(tmp_x1,x1_angle_gt,rtol=1e-5) and np.isclose(tmp_x2,x2_angle_gt,rtol=1e-5))
        simulator.apply_rotation(thetas_rand[i_sess,1],hem='2')
        _,tmp = get_top_angle(simulator.get_params())
        assert(np.isclose(tmp,thetas_rand[i_sess,1],rtol=1e-5))
        X_1,X_2 = simulator.sim_data(n_trials)
    
        # fit the data using pcca-fa
        model = pf.pcca_fa()
        model.train(X_1,X_2,d,d1,d2,warmstart=False)
        fit_params = model.get_params()
    
        # find estimated angles for area 2
        _, x2_angle = get_top_angle(fit_params)
        results[sess]['gt_rand_x2'] = thetas_rand[i_sess,1]
        results[sess]['fit_rand_x2'] = x2_angle

        #############################################
        # option 2: adjust L to have structured theta
        print('   manipulating structured thetas')
        for k,theta in enumerate(thetas_struct):
            # manipulate area 1
            simulator.set_params(params.copy())
            tmp_x1,tmp_x2 = get_top_angle(simulator.get_params())
            assert(np.isclose(tmp_x1,x1_angle_gt,rtol=1e-5) and np.isclose(tmp_x2,x2_angle_gt,rtol=1e-5))
            simulator.apply_rotation(theta,hem='1')
            tmp,_ = get_top_angle(simulator.get_params())
            assert(np.isclose(tmp,theta,rtol=1e-5))
            X_1,X_2 = simulator.sim_data(n_trials)

            # fit the data using pcca-fa
            model = pf.pcca_fa()
            model.train(X_1,X_2,d,d1,d2,warmstart=False)
            fit_params = model.get_params()

            # find estimated angles for area 1
            x1_angle, _ = get_top_angle(fit_params)
            results[sess]['fit_theta{}_x1'.format(theta)] = x1_angle

            # manipulate area 2
            simulator.set_params(params.copy())
            tmp_x1,tmp_x2 = get_top_angle(simulator.get_params())
            assert(np.isclose(tmp_x1,x1_angle_gt,rtol=1e-5) and np.isclose(tmp_x2,x2_angle_gt,rtol=1e-5))
            simulator.apply_rotation(theta,hem='2')
            _,tmp = get_top_angle(simulator.get_params())
            assert(np.isclose(tmp,theta,rtol=1e-5))
            X_1,X_2 = simulator.sim_data(n_trials)

            # fit the data using pcca-fa
            model = pf.pcca_fa()
            model.train(X_1,X_2,d,d1,d2,warmstart=False)
            fit_params = model.get_params()

            # find estimated angles
            _, x2_angle = get_top_angle(fit_params)
            results[sess]['fit_theta{}_x2'.format(theta)] = x2_angle

        # save results each session
        save_dict(results,save_name)
