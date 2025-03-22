# -- Figure 5 chance distributions --
# Structured thetas:
    # 30, 60, 90 deg
# Random thetas:
    # uniform (45-90 deg)
# Model considerations:
    # no cross-validation, using params from real data, no warmstart
    # manipulate each hemisphere individually, i.e., two sims per session
    
# imports
import sys
import numpy as np
import scipy.io as sio

sys.path.append('helpers/')
from dual_pfc_funcs import get_top_angle, getParams, save_dict, load_dict
import helpers.pcca_fa.pcca_fa_mdl as pf
import helpers.pcca_fa.sim_pcca_fa as spf

# real data
subjects = getParams()['subjects']
data_path = 'preprocessed_data/'
save_name = data_path + 'simdataset_varyThetaShuffle.pkl'

# params
np.random.seed(4) # to get good uniform thetas - somewhat arbitrary cherrypicking...
NSIM = 42 # number of sessions
thetas_struct = [30, 60, 90] # degrees, use same angle for both hemispheres
rand_low, rand_high = 45, 90
thetas_rand = np.random.uniform(low=rand_low,high=rand_high,size=(NSIM,2)) # degrees, col 1 for left hem, col 2 for right hem

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
        LH = getattr(curr_dat,'fast_component_left')
        RH = getattr(curr_dat,'fast_component_right')
        n_trials = curr_dat.n_trials
        xDim,yDim = LH.shape[1],RH.shape[1]

        # get pCCA-FA params
        params = pccafa_fits[sess]['params'].copy()
        zDim,zxDim,zyDim = params['zDim'],params['zxDim'],params['zyDim']
        x_angle_gt, y_angle_gt = get_top_angle(params.copy(),across_mode='cov')
        results[sess]['neural_x'] = x_angle_gt
        results[sess]['neural_y'] = y_angle_gt

        mdl = pf.pcca_fa()
        mdl.set_params(params.copy())
        psv_gt = mdl.compute_psv()
        shared_within_x_orig = np.diag(params['L_x'] @ params['L_x'].T)
    
        # simulator
        simulator = spf.sim_pcca_fa(xDim,yDim,zDim,zxDim,zyDim)
        simulator.set_params(params.copy())
        tmp_x,tmp_y = get_top_angle(simulator.get_params())
        assert(np.isclose(tmp_x,x_angle_gt,rtol=1e-5) and np.isclose(tmp_y,y_angle_gt,rtol=1e-5))

        #########################################
        # option 1: adjust L to have random theta
        print('   manipulating hem X')
        simulator.apply_rotation(thetas_rand[i_sess,0],hem='x')
        tmp,_ = get_top_angle(simulator.get_params())
        assert(np.isclose(tmp,thetas_rand[i_sess,0],rtol=1e-5))

        new_params = simulator.get_params().copy()
        mdl = pf.pcca_fa()
        mdl.set_params(new_params)
        psv_rot = mdl.compute_psv()
        shared_within_x_new = np.diag(new_params['L_x'] @ new_params['L_x'].T)
        print('  %sv across: {:.2f}, %sv within: {:.2f}'.format(psv_gt['psv_x'], psv_gt['psv_priv_x']))
        print('  ROTATED %sv across: {:.2f}, %sv within: {:.2f}'.format(psv_rot['psv_x'], psv_rot['psv_priv_x']))

        X,Y = simulator.sim_data(n_trials)
    
        # fit the data using pcca-fa
        model = pf.pcca_fa()
        model.train(X,Y,zDim,zxDim,zyDim,warmstart=False)
        fit_params = model.get_params()
    
        # find estimated angles for hemisphere X
        x_angle,_ = get_top_angle(fit_params)
        results[sess]['gt_rand_x'] = thetas_rand[i_sess,0]
        results[sess]['fit_rand_x'] = x_angle

        print('   manipulating hem Y')
        simulator.set_params(params.copy())
        tmp_x,tmp_y = get_top_angle(simulator.get_params())
        assert(np.isclose(tmp_x,x_angle_gt,rtol=1e-5) and np.isclose(tmp_y,y_angle_gt,rtol=1e-5))
        simulator.apply_rotation(thetas_rand[i_sess,1],hem='y')
        _,tmp = get_top_angle(simulator.get_params())
        assert(np.isclose(tmp,thetas_rand[i_sess,1],rtol=1e-5))
        X,Y = simulator.sim_data(n_trials)
    
        # fit the data using pcca-fa
        model = pf.pcca_fa()
        model.train(X,Y,zDim,zxDim,zyDim,warmstart=False)
        fit_params = model.get_params()
    
        # find estimated angles for hemisphere Y
        _, y_angle = get_top_angle(fit_params)
        results[sess]['gt_rand_y'] = thetas_rand[i_sess,1]
        results[sess]['fit_rand_y'] = y_angle

        #############################################
        # option 2: adjust L to have structured theta
        print('   manipulating structured thetas')
        for k,theta in enumerate(thetas_struct):
            # hemisphere 1: manipulate X
            simulator.set_params(params.copy())
            tmp_x,tmp_y = get_top_angle(simulator.get_params())
            assert(np.isclose(tmp_x,x_angle_gt,rtol=1e-5) and np.isclose(tmp_y,y_angle_gt,rtol=1e-5))
            simulator.apply_rotation(theta,hem='x')
            tmp,_ = get_top_angle(simulator.get_params())
            assert(np.isclose(tmp,theta,rtol=1e-5))
            X,Y = simulator.sim_data(n_trials)

            # fit the data using pcca-fa
            model = pf.pcca_fa()
            model.train(X,Y,zDim,zxDim,zyDim,warmstart=False)
            fit_params = model.get_params()

            # find estimated angles for hemisphere X
            x_angle, _ = get_top_angle(fit_params)
            results[sess]['fit_theta{}_x'.format(theta)] = x_angle

            # hemisphere 2: manipulate Y
            simulator.set_params(params.copy())
            tmp_x,tmp_y = get_top_angle(simulator.get_params())
            assert(np.isclose(tmp_x,x_angle_gt,rtol=1e-5) and np.isclose(tmp_y,y_angle_gt,rtol=1e-5))
            simulator.apply_rotation(theta,hem='y')
            _,tmp = get_top_angle(simulator.get_params())
            assert(np.isclose(tmp,theta,rtol=1e-5))
            X,Y = simulator.sim_data(n_trials)

            # fit the data using pcca-fa
            model = pf.pcca_fa()
            model.train(X,Y,zDim,zxDim,zyDim,warmstart=False)
            fit_params = model.get_params()

            # find estimated angles
            _, y_angle = get_top_angle(fit_params)
            results[sess]['fit_theta{}_y'.format(theta)] = y_angle

        # save results each session
        save_dict(results,save_name)