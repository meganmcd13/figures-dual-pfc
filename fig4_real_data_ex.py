# example data
mat_fname = 'data/pepe_nomean_noar25_delay.mat'
dat = sio.loadmat(mat_fname)
ex_sess = 'Pe180726'

mdl_params = extract_mdl_params(dat[ex_sess])
wx,wy,lx,ly,psix,psiy = mdl_params['W_x'],mdl_params['W_y'],mdl_params['L_x'],mdl_params['L_y'],mdl_params['psi_x'],mdl_params['psi_y']

across_x = np.diag(wx.dot(wx.T))
across_y = np.diag(wy.dot(wy.T))
within_x = np.diag(lx.dot(lx.T))
within_y = np.diag(ly.dot(ly.T))

total_x = across_x + within_x + psix # WWT + LLT + Psi
total_y = across_y + within_y + psiy # WWT + LLT + Psi

nx = len(total_x) # xDim
ny = len(total_y) # yDim

# calc psv
mdl = pf.pcca_fa()
mdl.set_params(mdl_params)
psv = mdl.compute_psv()
acc_sv = psv['psv_x']
within_sv = psv['psv_priv_x']
print("Example session percent shared variance:")
print(f'Across-area sv: {acc_sv:.2f}')
print(f'Within left sv: {within_sv:.2f}')

# plot
fig,ax = plt.subplots(2,1)
fig.set_figheight(2*fig.get_figheight())
xdata_left = np.arange(nx,0,-1)
xdata_right = np.arange(ny,0,-1)

ax[0].barh(xdata_left,total_x,color=indep,label='independent')
ax[0].barh(xdata_left,within_x+across_x,color=area2,label='within')
ax[0].barh(xdata_left,across_x,color=acrossarea,label='across')
ax[1].barh(xdata_right,total_y,color=indep,label='independent')
ax[1].barh(xdata_right,within_y+across_y,color=area1,label='within')
ax[1].barh(xdata_right,across_y,color=acrossarea,label='across')

# plot formatting
ax[0].set_title('Left PFC',color=area2)
ax[0].set_xticks([])
ax[0].set_yticks(np.arange(0,nx,10))
ax[0].legend(loc='best',frameon=False)
ax[0].set_ylabel('neuron index')
ax[1].set_title('Right PFC',color=area1)
ax[1].set_xticks([])
ax[1].set_yticks(np.arange(0,ny,10))
ax[1].set_ylabel('neuron index')
ax[1].legend(loc='best',frameon=False)

# fig.show()
pdf = PdfPages(FIGURE_PATH + 'fig4_partition_realdata.pdf')
pdf.savefig(fig)
pdf.close()

# noise corr
mat_fname = "data/all_data_pepe.mat"
dat = sio.loadmat(mat_fname,squeeze_me=True,struct_as_record=False)['all_data']
fnames = dat._fieldnames
fnames.remove('arr_spatial')

ex_sess = -4

counts_mat = getattr(dat,fnames[ex_sess]).counts.delay.T
targ_angs = getattr(dat,fnames[ex_sess]).beh.targ_angs
bin_size = getattr(dat,fnames[ex_sess]).binsize.delay
n_left = getattr(dat,fnames[ex_sess]).arr.LH_idx.sum()
n_right = counts_mat.shape[1]-n_left

X,_,_ = preprocess_counts(counts_mat,targ_angs,bin_size,25)

# compute rsc distributions
rsc = np.corrcoef(X.T)
rsc_L = rsc[:n_left,:n_left]
rsc_R = rsc[n_left:,n_left:]
rsc_acc = rsc[:n_left,n_left:]

rsc_L = rsc_L[np.triu_indices(n_left, k=1)]
rsc_R = rsc_R[np.triu_indices(n_right, k=1)]
rsc_acc = rsc_acc.reshape(-1)

print("rsc means (across all sessions):")
print(f"within right: {rsc_R.mean():.3f}")
print(f"within left: {rsc_L.mean():.3f}")
print(f"across: {rsc_acc.mean():.3f}")

# plot results
fig,ax = plt.subplots(3,1, constrained_layout=True)
fig.set_figheight(8)

fig.supxlabel('spike count correlation ($r_{sc}$)', fontsize=18)
fig.supylabel('proportion of pairs', fontsize=18)

bins = np.arange(-1,1.001,0.025)
ylim = 0.22
xlim = 0.4
a = 1.0
yticks = [0,0.05,0.10,0.15,0.2]
xticks = [-0.4,-0.2,0,0.2,0.4]

ax[0].hist(rsc_R,bins=bins,color=area1,alpha=a,weights=np.ones(len(rsc_R)) / len(rsc_R))
ax[0].plot(rsc_R.mean(),ylim-0.005,color=area1,alpha=a,marker='v',markersize=8)
ax[0].set_xlim([-xlim,xlim])
ax[0].set_xticks(xticks)
ax[0].set_yticks(yticks)
ax[0].set_title('within right PFC',color=area1)

ax[1].hist(rsc_L,bins=bins,color=area2,alpha=a,weights=np.ones(len(rsc_L)) / len(rsc_L))
ax[1].plot(rsc_L.mean(),ylim-0.005,color=area2,alpha=a,marker='v',markersize=8)
ax[1].set_xlim([-xlim,xlim])
ax[1].set_xticks(xticks)
ax[1].set_yticks(yticks)
ax[1].set_title('within left PFC',color=area2)

ax[2].hist(rsc_acc,bins=bins,color=acrossarea,alpha=a,weights=np.ones(len(rsc_acc)) / len(rsc_acc))
ax[2].plot(rsc_acc.mean(),ylim-0.005,color=acrossarea,alpha=a,marker='v',markersize=8)
ax[2].set_xlim([-xlim,xlim])
ax[2].set_xticks(xticks)
ax[2].set_yticks(yticks)
ax[2].set_title('across PFCs',color=acrossarea)

ax[0].axvline(x=0,color="black",linestyle="--",linewidth=1)
ax[1].axvline(x=0,color="black",linestyle="--",linewidth=1)
ax[2].axvline(x=0,color="black",linestyle="--",linewidth=1)

# fig.show()
pdf = PdfPages(FIGURE_PATH + 'fig4_rsc_realdata.pdf')
pdf.savefig(fig)
pdf.close()

# mean rsc across sessions
fig,ax = plt.subplots(nrows=3, ncols=1, sharex=True)
colors = [area1, area2, acrossarea]
labels = ['within (right)', 'within (left)', 'across']

subjects = ['wakko','satchel','pepe']
wts1, wts2, bts = np.array([]), np.array([]), np.array([])
for subject in subjects:
    mat_fname = "data/within_across_rsc_" + subject + ".mat"
    dat2 = sio.loadmat(mat_fname,squeeze_me=True,struct_as_record=False)['rsc_results']
    for sess in dat2:
        wts1 = np.append(wts1, sess.arr1_rsc_mean)
        wts2 = np.append(wts2, sess.arr2_rsc_mean)
        bts = np.append(bts, sess.bt_rsc_mean)
            
mean_wt1 = np.nanmean(wts1)
mean_wt2 = np.nanmean(wts2)
mean_bt = np.nanmean(bts)
sem_wt1 = np.nanstd(wts1) / np.sqrt(len(wts1))
sem_wt2 = np.nanstd(wts2) / np.sqrt(len(wts2))
sem_bt = np.nanstd(bts) / np.sqrt(len(bts))

ax[0].barh(0, mean_wt1, color=area1, xerr=sem_wt1, align='center', ecolor='black', edgecolor='black')
ax[1].barh(0, mean_wt2, color=area2, xerr=sem_wt2, align='center', ecolor='black', edgecolor='black')
ax[2].barh(0, mean_bt, color=acrossarea, xerr=sem_bt, align='center', ecolor='black', edgecolor='black')

j1 = 1+jitter(len(wts1))
j2 = 1+jitter(len(wts2))
j3 = 1+jitter(len(bts))
ax[0].scatter(wts1,j1, color=area1,alpha=0.5,s=2)
ax[1].scatter(wts2,j2, color=area2,alpha=0.5,s=2)
ax[2].scatter(bts,j3, color=acrossarea,alpha=0.5,s=2)

# mark example sess
ax[0].scatter(wts1[ex_sess],j1[ex_sess], color=area1,alpha=1,s=20,marker="*")
ax[1].scatter(wts2[ex_sess],j2[ex_sess], color=area2,alpha=1,s=20,marker="*")
ax[2].scatter(bts[ex_sess],j3[ex_sess], color=acrossarea,alpha=1,s=20,marker="*")

plt.xlabel('mean $r_{sc}$', fontsize=12)
for i,a in enumerate(ax):
    plt.sca(a)
    plt.tick_params(left=False,labelleft=False)
    plt.xticks(fontsize=12)
    a.set_ylabel(labels[i], color=colors[i], fontsize=12)

fig.tight_layout(pad=2)
# fig.show()
pdf = PdfPages(FIGURE_PATH + 'fig4_rsc_means_realdata.pdf')
pdf.savefig(fig)
pdf.close()

print('finished saving Figure 4 plots')