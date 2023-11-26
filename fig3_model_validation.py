# imports
import numpy as np
import os, sys
import pandas as pd
import scipy.stats as stats
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
plt.style.use('scifigs.mplstyle')

# make figure directory
FIGURE_PATH = '../pcca_fa_figures_output/'
if not os.path.isdir(FIGURE_PATH):
    os.makedirs(FIGURE_PATH, exist_ok=True)
    print("created figure output folder : ", FIGURE_PATH)

# params and helper functions
with open("utils.py") as f:
    exec(f.read())
# plotting colors
area1 = color_map['within1']
area2 = color_map['within2']
acrossarea = color_map['across']
indep = color_map['independent']

# paths
PCCAFA_PATH = '/afs/ece.cmu.edu/project/nspg/mmcdonne/general_code/pcca_fa'
sys.path.append(PCCAFA_PATH)

import pcca_fa_mdl as pf

# load first set of simulations comparing pCCA-FA to ground truth 
zDim_pairs = ((1,5),(2,4),(3,3),(4,2),(5,1)) # first entry is global dim, second entry is local dim
dat = load_dict('data/recover_dims_pccaFa_sv25.pkl')

# compute ground truth metrics
for i,z in enumerate(zDim_pairs):
    tmp = pf.pcca_fa()
    tmp.set_params(dat[z]['sim_mdl'].get_params())
    dat[z]['gt_metrics'] = tmp.compute_metrics()
gt = pd.DataFrame(columns=['zDim','zxDim','global_sv','local_sv'])

# compute gt dshared
for i,z in enumerate(zDim_pairs):
    dshared = dat[z]['gt_metrics']['dshared']
    gt.loc[i,'zDim'] = dshared['dshared_x']
    gt.loc[i,'zxDim'] = dshared['dshared_priv_x']

# compute fit dshared
allFits_dim = pd.DataFrame(columns=['zDim','zxDim'])
for i,z in enumerate(zDim_pairs):
    tmp_z,tmp_zx = [],[]
    for m in dat[z]['fit_metrics']:
        tmp_z.append(m['dshared']['dshared_x'])
        tmp_zx.append(m['dshared']['dshared_priv_x'])
    tmp_z = np.array(tmp_z)
    tmp_zx = np.array(tmp_zx)
    allFits_dim.loc[i,'zDim'] = tmp_z
    allFits_dim.loc[i,'zxDim'] = tmp_zx
    
# compute gt psv
for i,z in enumerate(zDim_pairs):
    psv = dat[z]['gt_metrics']['psv']
    gt.loc[i,'global_sv'] = psv['psv_x']
    gt.loc[i,'local_sv'] = psv['psv_priv_x']

# compute fit psv
allFits_psv = pd.DataFrame(columns=['psv','psvx'])
for i,z in enumerate(zDim_pairs):
    tmp_z,tmp_zx = [],[]
    for m in dat[z]['fit_metrics']:
        tmp_z.append(m['psv']['psv_x'])
        tmp_zx.append(m['psv']['psv_priv_x'])
    tmp_z = np.array(tmp_z)
    tmp_zx = np.array(tmp_zx)
    allFits_psv.loc[i,'psv'] = tmp_z
    allFits_psv.loc[i,'psvx'] = tmp_zx

fig,ax = plt.subplots(2,2, figsize=(4, 4))
fig.set_figwidth(2*fig.get_figwidth())
fig.set_figheight(2*fig.get_figheight())

# vary sv
gt_glob = gt['global_sv'].to_numpy()[:,None].astype(int)
glob = np.array(allFits_psv['psv'].tolist())
glob_mean, glob_sem = np.mean(glob,axis=1), stats.sem(glob,axis=1)
glob_std = np.std(glob,axis=1)
ax[0,0].errorbar(x=gt_glob,y=glob_mean, yerr=glob_std, color=acrossarea, marker='o', label='pCCA-FA',zorder=1)
ax[0,0].scatter(x=list(flatten([[b]*n_boots for b in gt_glob])),y=list(flatten(glob)), edgecolors='black', linewidth=0.5, marker='o', s=2, facecolors='none', zorder=2)
ax[0,0].set_xlim([5,35])
ax[0,0].set_ylim([5,35])
# ground truth
ax[0,0].plot(ax[0,0].get_xlim(),ax[0,0].get_xlim(),'--',label='ground truth', color='gray',zorder=0)

gt_loc = gt['local_sv'].to_numpy()[:,None].astype(int)
loc = np.array(allFits_psv['psvx'].tolist())
loc_mean, loc_sem = np.mean(loc,axis=1), stats.sem(loc,axis=1)
loc_std = np.std(loc,axis=1)
ax[1,0].errorbar(x=gt_loc,y=loc_mean, yerr=loc_std, color=area1, marker='o', label='pCCA-FA',zorder=1)
ax[1,0].scatter(x=list(flatten([[b]*n_boots for b in gt_loc])),y=list(flatten(loc)), edgecolors='black', linewidth=0.5, marker='o', s=2, facecolors='none', zorder=2)
ax[1,0].set_xlim([5,35])
ax[1,0].set_ylim([5,35])
# ground truth
ax[1,0].plot(ax[1,0].get_xlim(),ax[1,0].get_xlim(),'--',label='ground truth', color='gray',zorder=0)

# vary d_shared
gt_glob = gt['zDim'].to_numpy()[:,None].astype(int)
glob = np.array(allFits_dim['zDim'].tolist())
glob_mean, glob_sem = np.mean(glob,axis=1), stats.sem(glob,axis=1)
glob_std = np.std(glob,axis=1)
ax[0,1].errorbar(x=gt_glob,y=glob_mean, yerr=glob_std, color=acrossarea, marker='o', label='pCCA-FA',zorder=1)
ax[0,1].scatter(x=list(flatten([[b]*n_boots for b in gt_glob])),y=list(flatten(glob)), marker='o', s=2, color=acrossarea, zorder=2)
ax[0,1].plot(ax[0,1].get_xlim(),ax[0,1].get_xlim(),'--',label='ground truth', color='gray',zorder=0)

gt_loc = gt['zxDim'].to_numpy()[:,None].astype(int)
loc = np.array(allFits_dim['zxDim'].tolist())
loc_mean, loc_sem = np.mean(loc,axis=1), stats.sem(loc,axis=1)
loc_std = np.std(loc,axis=1)
ax[1,1].errorbar(x=gt_loc,y=loc_mean, yerr=loc_std, color=area1, marker='o', label='pCCA-FA',zorder=1)
ax[1,1].scatter(x=list(flatten([[b]*n_boots for b in gt_loc])),y=list(flatten(loc)), marker='o', s=2, color=area1, zorder=2)
ax[1,1].plot(ax[1,1].get_xlim(),ax[1,1].get_xlim(),'--',label='ground truth', color='gray',zorder=0)

# formatting
ax[0,0].set_xlabel('%sv', color=acrossarea)
ax[0,0].set_ylabel('%sv', color=acrossarea)
ax[0,1].set_xlabel(r'$d_{shared}$', color=acrossarea)
ax[0,1].set_ylabel(r'$d_{shared}$', color=acrossarea)
ax[0,1].set_xticks([1,2,3,4,5])
ax[0,1].set_yticks([1,2,3,4,5])
ax[1,1].set_xlabel(r'$d_{shared}$', color=area1)
ax[1,1].set_ylabel(r'$d_{shared}$', color=area1)
ax[1,1].set_xticks([1,2,3,4,5])
ax[1,1].set_yticks([1,2,3,4,5])
ax[0,0].set_title('% shared variance')
ax[0,1].set_title(r'$d_{shared}$')
ax[1,0].set_xlabel('%sv', color=area1)
ax[1,0].set_ylabel('%sv', color=area1)
ax[0,0].set_aspect('equal', 'box')
ax[1,0].set_aspect('equal', 'box')
ax[0,1].set_aspect('equal', 'box')
ax[1,1].set_aspect('equal', 'box')

# fig.show()
pdf = PdfPages(FIGURE_PATH + 'fig3_compare_gt.pdf')
pdf.savefig(fig)
pdf.close()

# load second set of simulations comparing pCCA-FA to pCCA across a range of trial counts
n_boots = 30
n_trials = np.array([100,150,300,600,1000,1500])
dat = load_dict('data/trial_sweep_sv25.pkl')

# compute ground truth metrics
tmp = pf.pcca_fa()
tmp.set_params(dat['sim_mdl'].get_params())
gt_metrics = tmp.compute_metrics()

# get fit metrics
fits_dim = pd.DataFrame(index=n_trials,
    columns=['cca_zDim_mu','cca_fa_zDim_mu','cca_fa_zxDim_mu','cca_zDim_std','cca_fa_zDim_std','cca_fa_zxDim_std'])
for N in n_trials:
    tmpz = []
    for c in dat[N]['cca_metrics']:
        if c==0:
            tmpz.append(0)
        else:
            tmpz.append(c['dshared']['dshared_x'])
    tmpz = np.array(tmpz)
    fits_dim.loc[N,'cca_zDim_mu'] = tmpz.mean()
    fits_dim.loc[N,'cca_zDim_std'] = tmpz.std()
    
    tmpz,tmpzx = [],[]
    for c in dat[N]['cca_fa_metrics']:
        tmpz.append(c['dshared']['dshared_x'])
        tmpzx.append(c['dshared']['dshared_priv_x'])
    tmpz,tmpzx = np.array(tmpz),np.array(tmpzx)
    fits_dim.loc[N,'cca_fa_zDim_mu'] = tmpz.mean()
    fits_dim.loc[N,'cca_fa_zDim_std'] = tmpz.std()
    fits_dim.loc[N,'cca_fa_zxDim_mu'] = tmpzx.mean()
    fits_dim.loc[N,'cca_fa_zxDim_std'] = tmpzx.std()
    
fits_psv = pd.DataFrame(index=n_trials,
    columns=['cca_psv_mu','cca_fa_psv_mu','cca_fa_psvx_mu','cca_psv_std','cca_fa_psv_std','cca_fa_psvx_std'])
for N in n_trials:
    tmpz = []
    for c in dat[N]['cca_metrics']:
        if c==0:
            tmpz.append(0)
        else:
            tmpz.append(c['psv']['psv_x'])
    tmpz = np.array(tmpz)
    fits_psv.loc[N,'cca_psv_mu'] = tmpz.mean()
    fits_psv.loc[N,'cca_psv_std'] = tmpz.std()
    
    tmpz,tmpzx = [],[]
    for c in dat[N]['cca_fa_metrics']:
        tmpz.append(c['psv']['psv_x'])
        tmpzx.append(c['psv']['psv_priv_x'])
    tmpz,tmpzx = np.array(tmpz),np.array(tmpzx)
    fits_psv.loc[N,'cca_fa_psv_mu'] = tmpz.mean()
    fits_psv.loc[N,'cca_fa_psv_std'] = tmpz.std()
    fits_psv.loc[N,'cca_fa_psvx_mu'] = tmpzx.mean()
    fits_psv.loc[N,'cca_fa_psvx_std'] = tmpzx.std()

fig,ax = plt.subplots(2,2, figsize=(4, 4),sharex=True)
fig.set_figwidth(2*fig.get_figwidth())
fig.set_figheight(2*fig.get_figheight())

# PSV global
ax[0,0].errorbar(fits_psv.index,fits_psv['cca_fa_psv_mu'],yerr=fits_psv['cca_fa_psv_std'],label='pCCA-FA',fmt='o-', color=acrossarea)
ax[0,0].errorbar(fits_psv.index,fits_psv['cca_psv_mu'],yerr=fits_psv['cca_psv_std'],label='pCCA',fmt='o-', color=acrossarea, alpha=0.5)
# ground truths
gt_psv = gt_metrics['psv']['psv_x']
gt_psvx = gt_metrics['psv']['psv_priv_x']
ax[0,0].plot(ax[0,0].get_xlim(),[gt_psv,gt_psv],'--',label='ground truth', color='gray')

# D SHARED global
ax[0,1].errorbar(fits_dim.index,fits_dim['cca_fa_zDim_mu'],yerr=fits_dim['cca_fa_zDim_std'],label='pCCA-FA',fmt='o-', color=acrossarea)
ax[0,1].errorbar(fits_dim.index,fits_dim['cca_zDim_mu'],yerr=fits_dim['cca_zDim_std'],label='pCCA',fmt='o-', color=acrossarea, alpha=0.5)
# ground truths
gt_z = gt_metrics['dshared']['dshared_x']
gt_zx = gt_metrics['dshared']['dshared_priv_x']
ax[0,1].plot(ax[0,1].get_xlim(),[gt_z,gt_z],'--',label='ground truth', color='gray')

# PSV local
ax[1,0].errorbar(fits_psv.index,fits_psv['cca_fa_psvx_mu'],yerr=fits_psv['cca_fa_psvx_std'],label='pCCA-FA',fmt='o-', color=area1)
# ground truth
gt_psvx = gt_metrics['psv']['psv_priv_x']
ax[1,0].plot(ax[1,0].get_xlim(),[gt_psvx,gt_psvx],'--',label='ground truth', color='gray')

# D SHARED local
ax[1,1].errorbar(fits_dim.index,fits_dim['cca_fa_zxDim_mu'],yerr=fits_dim['cca_fa_zxDim_std'],label='pCCA-FA',fmt='o-', color=area1)
# ground truth
gt_zx = gt_metrics['dshared']['dshared_priv_x']
ax[1,1].plot(ax[1,1].get_xlim(),[gt_zx,gt_zx],'--',label='ground truth', color='gray')

# formatting
ax[0,0].legend(loc='lower right')
ax[0,0].set_ylabel('%sv', color=acrossarea)
ax[0,0].set_ylim([-2,33])
ax[0,0].set_xticks([0,500,1000,1500])
ax[0,1].set_ylabel(r'$d_{shared}$', color=acrossarea)
ax[0,1].set_ylim([-0.25,gt_z+1])
ax[0,1].set_xticks([0,500,1000,1500])
ax[1,0].legend(loc='lower right')
ax[1,0].set_ylabel('%sv', color=area1)
ax[1,0].set_ylim([-2,33])
ax[1,0].set_xticks([0,500,1000,1500])
ax[1,1].set_ylabel(r'$d_{shared}$', color=area1)
ax[1,1].set_ylim([-0.25,gt_z+1])
ax[1,1].set_xticks([0,500,1000,1500])
fig.supxlabel('number of trials used to estimate',fontsize=18)

# fig.show()
pdf = PdfPages(FIGURE_PATH + 'fig3_compare_pcca.pdf')
pdf.savefig(fig)
pdf.close()

print('finished saving Figure 3 plots')
