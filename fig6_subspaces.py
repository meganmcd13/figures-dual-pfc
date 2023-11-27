# load data and get params
filename = "data/preprocess_model_fits2.txt"
fits_dict = load_dict(filename)
    
all_params_list = fits_dict['model_fits']
all_subjects = fits_dict['sub_names']
x_angles_list, y_angles_list = get_top_angles(all_params_list)
nsim = len(all_params_list)
    
# choose example session
ex_sess = all_params_list[-6] # subject Sa

# panel a: weight values, example session 
top_wx,pct_wx = get_top_vec(ex_sess['W_x'])
top_wy,_ = get_top_vec(ex_sess['W_y'])
top_lx,pct_lx = get_top_vec(ex_sess['L_x'])
top_ly,_ = get_top_vec(ex_sess['L_y'])

max_y = np.amax(np.concatenate((top_wx,top_lx)))
min_y = np.amin(np.concatenate((top_wx,top_lx)))

Nx, Ny = ex_sess['W_x'].shape[0], ex_sess['W_y'].shape[0]

xdata = np.arange(Nx)
ydata = np.arange(Ny)

fig, axs = plt.subplots(1,2, constrained_layout=True)
fig.set_figwidth(2*fig.get_figwidth())

axs[0].bar(xdata, top_wx, color=acrossarea)
axs[1].bar(xdata, top_lx, color=area2)

axs[0].set_title('Across-area', fontsize=12, color=acrossarea)
axs[0].set_xlabel('neurons', fontsize=10)
axs[0].set_ylabel('weight value', fontsize=10)
axs[0].set_xlim([-0.5,Nx])
axs[0].set_ylim([min_y-0.1,max_y+0.1])
axs[0].set_xticks([])
axs[0].set_yticks([])
axs[1].set_title('Within-area', fontsize=12, color=area2)
axs[1].set_xlabel('neurons', fontsize=10)
axs[1].set_ylabel('weight value', fontsize=10)
axs[1].set_xlim([-0.5,Nx])
axs[1].set_ylim([min_y-0.1,max_y+0.1])
axs[1].set_xticks([])
axs[1].set_yticks([])
fig.suptitle('Example co-fluctuation patterns', fontsize=16)

# fig.show()
pdf = PdfPages(FIGURE_PATH + 'fig6_weight_values.pdf')
pdf.savefig(fig)
pdf.close()

# panel b: % positive weights, across sessions 
pct_pos_wx, pct_pos_wy, pct_pos_lx, pct_pos_ly = [], [], [], []
pct_pos_w, pct_pos_l = [], []
n_neurons_x_list, n_neurons_y_list = [], []
for params in all_params_list:
    _,tmp = get_top_vec(params['W_x'])
    n_neurons_x_list.append(params['W_x'].shape[0])
    pct_pos_wx.append(tmp)
    pct_pos_w.append(tmp)
    _,tmp = get_top_vec(params['W_y'])
    n_neurons_y_list.append(params['W_y'].shape[0])
    pct_pos_w.append(tmp)
    pct_pos_wy.append(tmp)
    _,tmp = get_top_vec(params['L_x'])
    pct_pos_lx.append(tmp)
    pct_pos_l.append(tmp)
    _,tmp = get_top_vec(params['L_y'])
    pct_pos_ly.append(tmp)
    pct_pos_l.append(tmp)

max_y = 15 # hardcoded

mean_acc = np.array(pct_pos_w).mean()
mean_within = np.array(pct_pos_l).mean()
print(f"Across-area avg: {mean_acc:.4f}")
print(f"Within-area avg: {mean_within:.4f}")
    
fig, axs = plt.subplots(1,2, constrained_layout=True)
fig.set_figwidth(2*fig.get_figwidth())

axs[0].hist(pct_pos_w, bins=np.arange(0.5,1,0.02), color=acrossarea)
axs[1].hist(pct_pos_l, bins=np.arange(0.5,1,0.02), color=within)
axs[0].plot(mean_acc,max_y-0.5,marker='v',color=acrossarea)
axs[1].plot(mean_within,max_y-0.5,marker='v',color=within)

axs[0].set_title('Across-area', fontsize=16, color=acrossarea)
axs[0].set_xticks(np.arange(0.25,1,0.25)), axs[0].set_yticks(np.arange(0,max_y,5))
axs[0].set_xlim([0.49,1]),axs[0].set_ylim([0,max_y])
axs[1].set_title('Within-area', fontsize=16, color=within)
axs[1].set_xticks(np.arange(0.25,1,0.25)), axs[1].set_yticks(np.arange(0,max_y,5))
axs[1].set_xlim([0.49,1]),axs[1].set_ylim([0,max_y])
fig.supxlabel('$\%$ weights > 0', fontsize=14)
fig.supylabel('count (sessions x 2 hemispheres))', fontsize=14)

# fig.show()
pdf = PdfPages(FIGURE_PATH + 'fig6_pct_pos_weights.pdf')
pdf.savefig(fig)
pdf.close()

# panel b stats:
# session counts: pe=16, wa=10, sa=16
pe_w, pe_l = pct_pos_w[:32], pct_pos_l[:32]
wa_w, wa_l = pct_pos_w[32:52], pct_pos_l[32:52]
sa_w, sa_l = pct_pos_w[52:], pct_pos_l[52:]

pe_stat = stats.wilcoxon(x=pe_w, y=pe_l)
print(f"Subject Pe p = {pe_stat.pvalue:.4f}")

wa_stat = stats.wilcoxon(x=wa_w, y=wa_l)
print(f"Subject Wa p = {wa_stat.pvalue:.4f}")

sa_stat = stats.wilcoxon(x=sa_w, y=sa_l)
print(f"Subject Sa p = {sa_stat.pvalue:.4f}")

pooled_stat = stats.wilcoxon(x=pct_pos_w, y=pct_pos_l)
print(f"pooled p = {pooled_stat.pvalue:.4f}")

# panel d: angle between top eigenvectors, across sessions 
cols = ['subject', 'real_x', 'real_y', 'rand_x_high', 'rand_y_high']
df = pd.DataFrame(columns=cols)

# chance data
filename_high = 'data/shuffleDataSimVaryTheta_rand45-90_and_align_v2.txt'
sim_dat_high = load_dict(filename_high)

# systematic alignment - 
thetas = sim_dat_high["thetas_align"] 
align_dist_x = sim_dat_high["fit_alignx"]
align_dist_y = sim_dat_high["fit_aligny"]
# random alignment
rand_dist_x_high = sim_dat_high["fit_randx"]
rand_dist_y_high = sim_dat_high["fit_randy"]

# aggregate data in dataframe
idx_for_rand = 0
for i in range(nsim):
    sub = all_subjects[i]
    real_x = x_angles_list[i]
    real_y = y_angles_list[i]
    rand_x_high = rand_dist_x_high[i,idx_for_rand]
    rand_y_high = rand_dist_y_high[i,idx_for_rand]
    df.loc[len(df.index)] = [sub, real_x, real_y, rand_x_high, rand_y_high]
    
# construct 42 element histogram for each theta:
arr_x2 = np.zeros((42,len(thetas)))
for i,sess in enumerate(align_dist_x):
    for j,t in enumerate(sess):
        arr_x2[i,j] = t
arr_y2 = np.zeros((42,len(thetas)))
for i,sess in enumerate(align_dist_y):
    for j,t in enumerate(sess):
        arr_y2[i,j] = t

for i,theta in enumerate(thetas):
    key = "align_x_"+str(theta)
    dat = arr_x2[:,i]
    df[key] = dat
    key = "align_y_"+str(theta)
    dat = arr_y2[:,i]
    df[key] = dat
    
all_angles = np.concatenate((df['real_x'].to_numpy(), df['real_y'].to_numpy()))
all_random_angles_high = np.concatenate((df['rand_x_high'].to_numpy(), df['rand_y_high'].to_numpy()))

colors = ['0', '0.4', '0.5', '0.6', '0.7', '0.8']
yaxis_ticks = [0.5, -1, -2, -3, -4, -5]
yaxis_labels = ["real data", "10 deg alignment","30 deg alignment","60 deg alignment","90 deg alignment", "45-90 deg alignment"]
# define base yvals
real = yaxis_ticks[0]
[align10, align30, align60, align90, randH] = yaxis_ticks[1:]

fig, ax = plt.subplots(constrained_layout=False, figsize=(8,5))
plt.subplots_adjust(left=0.2)

# real data
ax.scatter(x=all_angles, y=[real]*len(all_angles)+jitter(len(all_angles)), color=colors[0], s=5)

# aligned data
xdata10 = np.concatenate((df['align_x_10'].to_numpy(), df['align_y_10'].to_numpy()))
ax.scatter(x=xdata10, y=[align10]*len(xdata10)+jitter(len(xdata10)), color=colors[1], s=3)
xdata30 = np.concatenate((df['align_x_30'].to_numpy(), df['align_y_30'].to_numpy()))
ax.scatter(x=xdata30, y=[align30]*len(xdata30)+jitter(len(xdata30)), color=colors[2], s=3)
xdata60 = np.concatenate((df['align_x_60'].to_numpy(), df['align_y_60'].to_numpy()))
ax.scatter(x=xdata60, y=[align60]*len(xdata60)+jitter(len(xdata60)), color=colors[3], s=3)
xdata90 = np.concatenate((df['align_x_90'].to_numpy(), df['align_y_90'].to_numpy()))
ax.scatter(x=xdata90, y=[align90]*len(xdata90)+jitter(len(xdata90)), color=colors[4], s=3)
xdataH = all_random_angles_high
ax.scatter(x=xdataH, y=[randH]*len(xdataH)+jitter(len(xdataH)), color=colors[5], s=3)

fig.suptitle(r'$\theta_{subspaces}$', fontsize=16)
ax.set_xlabel(r'estimated angle ($^\circ$)', fontsize=10)
ax.set_xticks(np.arange(0,91,30))
ax.set_xlim([0,91])
plt.yticks(yaxis_ticks, yaxis_labels, fontsize=10)

# fig.show()
pdf = PdfPages(FIGURE_PATH + 'fig6_theta_subspaces.pdf')
pdf.savefig(fig)
pdf.close()

# panel d statistics:
# two-sample Kolmogorov-Smirnov test to compare the difference between real data and aligned dists. 
# do real data and sham data come from same distribution?
# NOTE: alternative hypothesis "less" states that the CDF of the all_angles would be less than the CDF of the xdata distribution, and that the values are greater.
alpha = 0.001

# test 1: 10 deg
p10 = stats.ks_2samp(all_angles, xdata10, alternative='less').pvalue
print(f"Real data greater than 10 deg? {p10<alpha}, p={p10:.4f}")

# test 2: 30 deg
p30 = stats.ks_2samp(all_angles, xdata30, alternative='less').pvalue
print(f"Real data greater than 30 deg? {p30<alpha}, p={p30:.4f}")

# test 3: 60 deg
p60 = stats.ks_2samp(all_angles, xdata60, alternative='less').pvalue
print(f"Real data greater than 60 deg? {p60<alpha}, p={p60:.4f}")

# test 4: 90 deg
p90 = stats.ks_2samp(all_angles, xdata90, alternative='less').pvalue
print(f"Real data greater than 90 deg? {p90<alpha}, p={p90:.4f}")

# test 5: 45-90 deg
pH = stats.ks_2samp(all_angles, xdataH, alternative='two-sided').pvalue
print(f"Real data different than 45-90 deg? {pH<alpha}, p={pH:.4f}")

print('finished saving Figure 6 plots')
print('- - - - - - - - - - - - - - - -')