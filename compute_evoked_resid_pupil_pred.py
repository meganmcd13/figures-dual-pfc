# -- Figure S10 control analysis --
# perform regression onto residual event-related pupil after regressing out fast pupil component, using all co-fluctuation patterns of each type

import sys
import numpy as np
import scipy.io as sio
from sklearn.linear_model import LinearRegression

sys.path.append('helpers/pcca_fa/')
from dual_pfc_funcs import load_dict, save_dict, getParams
import helpers.pcca_fa.pcca_fa_mdl as pf

subjects = getParams()['subjects']
data_path = 'preprocessed_data/'

dat = {}
for sub in subjects:    
    # spike counts and pupil data
    mat_fname = '{:s}/all_data_delay_{:s}.mat'.format(data_path, sub)
    sub_dat = sio.loadmat(mat_fname,squeeze_me=True,struct_as_record=False)['all_data']
    fnames = sub_dat._fieldnames
    fnames.remove('ar_order')
    fnames.remove('arr_spatial')

    # get pCCA-FA fits
    model_fits = load_dict(data_path + sub + '_pccafa_cv15dim.pkl')

    # check if pupil data is good - use 2x IQR to filter out outliers   
    ddof = 0
    pupil_var = np.zeros(len(fnames))
    for i_sess,sess in enumerate(fnames):
        curr_dat = getattr(sub_dat,sess)
        pupil_vals = getattr(curr_dat,'fast_component_pupil')
        pupil_var[i_sess] = np.var(pupil_vals,ddof=ddof)

    thresh = 2
    Q1 = np.percentile(pupil_var, 25)
    Q3 = np.percentile(pupil_var, 75)
    IQR = Q3 - Q1
    pupil_mask = (pupil_var < (Q1 - thresh * IQR)) | (pupil_var > (Q3 + thresh * IQR))
    fnames = [f for i,f in enumerate(fnames) if not pupil_mask[i]]

    for i_sess,sess in enumerate(fnames):
        print('Getting pupil predictions for session: ', sess)

        # get behav dat
        curr_dat = getattr(sub_dat,sess)

        # get spike data
        LH = getattr(curr_dat,'fast_component_left')
        RH = getattr(curr_dat,'fast_component_right')

        # get latents
        params = model_fits[sess]['params']
        mdl = pf.pcca_fa()
        mdl.set_params(params)
        z,_ = mdl.estep(LH,RH)
        latents = {
                'across'      : z['z_mu'],
                'within-left' : z['zx1_mu'],
                'within-right': z['zx2_mu'],
            }

        # get pupil data
        fast_pupil = getattr(curr_dat,'fast_component_pupil')
        fast_zsc = fast_pupil.reshape(-1,1)        
        fast_zsc = (fast_zsc - np.mean(fast_zsc)) / np.std(fast_zsc)
        evoked_pupil = getattr(curr_dat.pupil,'evoked') - getattr(curr_dat.pupil,'baseline')
        evoked_zsc = evoked_pupil.reshape(-1,1)
        evoked_zsc = (evoked_zsc - np.mean(evoked_zsc)) / np.std(evoked_zsc)

        # step 1: linear regression to predict pupil evoked from fast
        lm = LinearRegression().fit(evoked_zsc,fast_zsc)
        evoked_pred = lm.predict(fast_zsc)
        evoked_resid = evoked_zsc - evoked_pred

        # step 2: predict residuals from latents
        r2 = {}
        for latent,x in latents.items():
            # predict avg pupil from latents
            lm = LinearRegression().fit(x,evoked_resid)
            r2[latent] = lm.score(x,evoked_resid)

        # save info:
        dat[sess] = {
            'fast_zsc'    : fast_zsc,
            'evoked_zsc'  : evoked_zsc,
            'latents'     : latents,
            'r2'          : r2,
            'resid_zsc'   : evoked_resid,
        }

# get null distribution for each session from other sessions latents
for sess in dat:
    print('Getting null distribution for session: ', sess)
    null_r2 = {'across':[],'within-left':[],'within-right':[]}
    for latent in ['across','within-left','within-right']:
        for j in dat:
            if sess!=j and j.startswith(sess[:2]):
                # compare session "sess" to all other sessions from the same subject
                N = min([len(dat[sess]['resid_zsc']),len(dat[j]['resid_zsc'])])
                x = dat[sess]['latents'][latent][:N,:]
                # avg pupil
                y = dat[j]['resid_zsc'][:N].reshape(-1, 1)
                # fit linear model
                lm = LinearRegression().fit(x,y)
                null_r2[latent].append(lm.score(x,y))
    dat[sess]['null_r2'] = null_r2

# save data
save_name = data_path + 'evoked_resid_pupil_prediction.pkl'
save_dict(dat, save_name)