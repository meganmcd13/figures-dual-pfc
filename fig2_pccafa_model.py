# random seed
np.random.seed(0)

# params
num_neurons = 20
num_timepoints = 1000
noise_var = 1e-3
GP_tau = 50
t = np.expand_dims(np.arange(num_timepoints),1)
dists = squareform(pdist(t))
K = (1-noise_var)*np.exp( -np.square(dists) / (2*GP_tau**2)) + np.eye(num_timepoints)*noise_var
sqrt_K = np.real(slin.sqrtm(K))
base_p = 1e-2
c = 5
base_add = 0.05

# generate latents
across_z = sqrt_K.dot(np.random.randn(num_timepoints))
np.random.seed(2)
area1_z = sqrt_K.dot(np.random.randn(num_timepoints))
area2_z = sqrt_K.dot(np.random.randn(num_timepoints))

# plot generated latents
fig,ax = plt.subplots(3,1)

ax[0].plot(across_z,color=acrossarea)
ax[1].plot(area1_z,color=area2)
ax[2].plot(area2_z,color=area1)

plt.tight_layout()
# fig.show()
pdf = PdfPages(FIGURE_PATH + 'fig2_latents_simdata.pdf')
pdf.savefig(fig)
pdf.close()

# generate loadings
np.random.seed(0)
across_loadings = np.random.randn(num_neurons) + 1
across_loadings /= slin.norm(across_loadings)
np.random.seed(1)
area1_loadings = np.random.randn(num_neurons) + 1
area1_loadings /= slin.norm(area1_loadings)
np.random.seed(6)
area2_loadings = np.random.randn(num_neurons) + 1
area2_loadings /= slin.norm(area2_loadings)

# generate spike trains
area1_raster = np.zeros((num_neurons,num_timepoints))
area2_raster = np.zeros((num_neurons,num_timepoints))

for ineuron in range(num_neurons):
    # area 1
    zi = across_loadings[ineuron]*across_z + area1_loadings[ineuron]*area1_z
    if np.max(zi) <= 0:
        zi[zi <= 0] = base_p
    else:
        zi = zi / np.max(zi) / c # control overall firing by d
        zi = zi + base_add
    zi[zi <= 0] = base_p
    area1_raster[ineuron,:] = np.random.binomial(np.ones((num_timepoints,)).astype('int'), zi)
    
    # area 2
    zi = across_loadings[ineuron]*across_z + area2_loadings[ineuron]*area2_z
    if np.max(zi) <= 0:
        zi[zi <= 0] = base_p
    else:
        zi = zi / np.max(zi) / c # control overall firing by d
        zi = zi + base_add
    zi[zi <= 0] = base_p
    area2_raster[ineuron,:] = np.random.binomial(np.ones((num_timepoints,)).astype('int'), zi)

# compute spike counts and across/within correlations
bin_size = 20
start_idx = bin_size-1
area1_counts = pd.DataFrame(area1_raster.T).rolling(bin_size).sum().to_numpy().T[:,start_idx::bin_size]
area2_counts = pd.DataFrame(area2_raster.T).rolling(bin_size).sum().to_numpy().T[:,start_idx::bin_size]

area1_corr = np.corrcoef(area1_counts)[np.triu_indices(num_neurons,k=1)].mean()
area2_corr = np.corrcoef(area2_counts)[np.triu_indices(num_neurons,k=1)].mean()
across_corr = np.corrcoef(area1_counts,y=area2_counts)[:num_neurons,num_neurons:].reshape(-1).mean()

# plot rasters
fig,ax = plt.subplots(2,1)
fig.set_figheight(2*fig.get_figheight())

plot_raster(area1_raster,ax[0])
plot_raster(area2_raster,ax[1])

ax[0].set_ylabel('area 1 neurons', color=area1)
ax[0].set_xlim([0,1000])
ax[0].set_ylim([0,num_neurons])
ax[0].set_xticks([])
ax[0].set_yticks([])
ax[1].set_ylabel('area 2 neurons', color=area2)
ax[1].set_xlabel('time')
ax[1].set_xlim([0,1000])
ax[1].set_ylim([0,num_neurons])
ax[1].set_xticks([])
ax[1].set_yticks([])

# fig.show()
pdf = PdfPages(FIGURE_PATH + 'fig2_rasters_simdata.pdf')
pdf.savefig(fig)
pdf.close()

# compute across and within %sv values (0.05 is independent var for each neuron)
across_loadings = across_loadings.reshape(-1,1)
area1_loadings = area1_loadings.reshape(-1,1)
area2_loadings = area2_loadings.reshape(-1,1)

across_var = np.diag(across_loadings.dot(across_loadings.T))
area1_var = np.diag(area1_loadings.dot(area1_loadings.T))
area2_var = np.diag(area2_loadings.dot(area2_loadings.T))

ind_var = base_add
area1_total = across_var + area1_var + ind_var
area2_total = across_var + area2_var + ind_var

# plot
fig,ax = plt.subplots(2,1)
fig.set_figheight(2*fig.get_figheight())

right_idx = np.arange(num_neurons)
xdata = np.arange(num_neurons,0,-1)

ax[0].barh(xdata,area1_total,color=indep,label='independent')
ax[0].barh(xdata,area1_var+across_var,color=area1,label='within')
ax[0].barh(xdata,across_var,color=acrossarea,label='across')
ax[1].barh(xdata,area2_total[right_idx],color=indep,label='independent')
ax[1].barh(xdata,area2_var+across_var[right_idx],color=area2,label='within')
ax[1].barh(xdata,across_var[right_idx],color=acrossarea,label='across')

# plot formatting
ax[0].set_title('area 1',color=area1)
ax[0].set_xticks([])
ax[0].set_yticks([1,10,20])
ax[0].set_ylabel('neuron index')
ax[0].legend(loc='best',frameon=False)
ax[1].set_title('area 2',color=area2)
ax[1].set_xticks([])
ax[1].set_yticks([1,10,20])
ax[1].set_ylabel('neuron index')
ax[1].legend(loc='best',frameon=False)

# fig.show()
pdf = PdfPages(FIGURE_PATH + 'fig2_partition_simdata.pdf')
pdf.savefig(fig)
pdf.close()

print('finished saving Figure 2 plots')
