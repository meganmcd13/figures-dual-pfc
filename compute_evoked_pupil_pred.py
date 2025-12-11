# -- Figure 6 data --
# create pupil latents
import sys
import numpy as np
import scipy.io as sio
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import r2_score

sys.path.append('helpers/pcca_fa/')
from dual_pfc_funcs import load_dict, save_dict, getParams
import helpers.pcca_fa.pcca_fa_mdl as pf

subjects = getParams()['subjects']
data_path = 'preprocessed_data/'
CROSSVAL = False

dat = {}
for sub in subjects:
    # pCCA-FA fits:
    model_fits = load_dict(data_path + sub + '_pccafa_cv15dim.pkl')

    # spike counts and pupil data
    mat_fname = '{:s}/all_data_delay_{:s}.mat'.format(data_path, sub)
    sub_dat = sio.loadmat(mat_fname,squeeze_me=True,struct_as_record=False)['all_data']
    fnames = sub_dat._fieldnames
    fnames.remove('ar_order')
    fnames.remove('arr_spatial')

    for i_sess,sess in enumerate(fnames,1):
        print('Getting pupil predictions for session: ', sess)
        # get spike data
        curr_dat = getattr(sub_dat,sess)
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

        # get evoked pupil data
        evoked_peak = getattr(curr_dat.pupil,'evoked')
        evoked_baseline = getattr(curr_dat.pupil,'baseline')
        # evoked_baseline = getattr(curr_dat,'slow_component_pupil')
        y_evoked = evoked_peak - evoked_baseline
        
        # z-score pupil per session
        y_evoked_zsc = (y_evoked - np.mean(y_evoked)) / np.std(y_evoked)

        # predict pupil from latents
        r2_evoked = {}
        pred_evoked = {}
        for latent,x in latents.items():
            if CROSSVAL:
                # evoked pupil
                lm = LinearRegression()
                pupil_hat = cross_val_predict(lm, x, y_evoked_zsc, cv=10)
                pred_evoked[latent] = pupil_hat
                r2_evoked[latent] = r2_score(y_evoked_zsc, pupil_hat)
            else:
                # predict evoked pupil from latents
                lm = LinearRegression().fit(x,y_evoked_zsc)
                pred_evoked[latent] = lm.predict(x)
                r2_evoked[latent] = lm.score(x,y_evoked_zsc)

        # save info:
        dat[sess] = {
            'pupil_zsc'   : y_evoked_zsc,
            'latents'     : latents,
            'r2'          : r2_evoked,
            'predictions' : pred_evoked,
        }

# get null distribution for each session from other sessions latents
for sess in dat:
    print('Getting null distribution for session: ', sess)
    null_r2_evoked = {'across':[],'within-left':[],'within-right':[]}
    for latent in ['across','within-left','within-right']:
        for j in dat:
            if sess!=j and j.startswith(sess[:2]):
                # compare session "sess" to all other sessions from the same subject
                N = min([len(dat[sess]['pupil_zsc']),len(dat[j]['pupil_zsc'])])
                x = dat[sess]['latents'][latent][:N,:]
                # evoked pupil
                y_evoked = dat[j]['pupil_zsc'][:N].reshape(-1, 1)
                if CROSSVAL:
                    # cross-validated
                    lm = LinearRegression()
                    pupil_hat = cross_val_predict(lm, x, y_evoked, cv=10)
                    null_r2_evoked[latent].append(r2_score(y_evoked, pupil_hat))
                else:
                    lm = LinearRegression().fit(x,y_evoked)
                    null_r2_evoked[latent].append(lm.score(x,y_evoked))
    dat[sess]['null_r2'] = null_r2_evoked

# save data
if CROSSVAL:
    save_name = data_path + 'evoked_pupil_prediction_cv.pkl'
else:
    save_name = data_path + 'evoked_pupil_prediction.pkl'
save_dict(dat, save_name)