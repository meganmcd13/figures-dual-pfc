# generate spike count covariance matrix for pitfall 1: no local
xDim,yDim = 30,30
zDim, zxDim, zyDim = 2,0,0
mu = 1
sigma_across = np.diag([1,0.05]) # affects loading similarity

# genrate loading matrices
np.random.seed(0)
W_x = np.random.randn(xDim,zDim) @ sigma_across + mu
W_y = np.random.randn(yDim,zDim) @ sigma_across + mu

Psix = np.ones((xDim,)) / .1
Psiy = np.ones((yDim,)) / .1

# large block matrices
lilde = np.block([[W_x],[W_y]])
psilde = np.diag(np.concatenate((Psix,Psiy)))
Sigma_shared = lilde @ lilde.T
Sigma = Sigma_shared + psilde

# spike count correlations
x,y = compute_rsc_within_pccafa(Sigma,xDim)
a = compute_rsc_across_pccafa(Sigma,xDim)
print('PITFALL 1: mean within-area rsc > 0 --> within-area interactions')
print(f'Within area 1 mean rsc: {x.mean():.2f}')
print(f'Within area 2 mean rsc: {y.mean():.2f}')
print(f'Across areas mean rsc: {a.mean():.2f}')
print('----------------------------------------------------------------')

# plot
fig1,ax = plt.subplots(constrained_layout=True)
fig1.supxlabel('spike count correlation ($r_{sc}$)', fontsize=18)
fig1.supylabel('proportion of neuron pairs', fontsize=18)

bins = np.arange(-1.05,1.1,0.1)
y_hat = 0.5
yrange = np.arange(0,0.7,0.1)

ax.hist(x,bins=bins,color=area2,alpha=1,weights=np.ones(len(x)) / len(x))
ax.plot(x.mean(),y_hat,color=area2,alpha=1,marker='v',markersize=10)
ax.axvline(x=0,color="black",linestyle="--",linewidth=1)
ax.set_xlim([-1,1])
ax.set_ylim([0,yrange[-1]])
ax.set_xticks(np.arange(-1,1.1,0.5))
ax.set_yticks(yrange)
ax.set_title('within (area 2)',color=area2)

# fig1.show()
pdf = PdfPages(FIGURE_PATH + 'fig1_pitfall1.pdf')
pdf.savefig(fig1)
pdf.close()

# generate spike count covariance matrix for pitfall 2: mean 0 global
xDim,yDim = 30,30
zDim, zxDim, zyDim = 2,1,1
mu = 1
sigma_across = np.diag([.8, 0.002]) # affects loading similarity
sigma_within1 = np.diag([5]) # affects loading similarity
sigma_within2 = np.diag([1]) # affects loading similarity

# genrate loading matrices
np.random.seed(0)
W_x = np.random.randn(xDim,zDim) @ sigma_across + mu
W_y = np.random.randn(yDim,zDim) @ sigma_across + mu
np.random.seed(1)
L_x = np.random.randn(xDim,zxDim) @ sigma_within1 + mu
np.random.seed(8)
L_y = np.random.randn(yDim,zyDim) @ sigma_within2 + mu

cutoff = np.floor(xDim / 2).astype(int)
W_x[:cutoff,0] = -W_x[:cutoff,0]
W_x[cutoff:,1] = -W_x[cutoff:,1]

Psix = np.ones((xDim,)) / .1
Psiy = np.ones((yDim,)) / .4

# large block matrices
lilde = np.block([[W_x,L_x,np.zeros((L_y.shape))],[W_y,np.zeros((L_x.shape)), L_y]])
psilde = np.diag(np.concatenate((Psix,Psiy)))
Sigma_shared = lilde @ lilde.T
Sigma = Sigma_shared + psilde

# spike count correlations
x,y = compute_rsc_within_pccafa(Sigma,xDim)
a = compute_rsc_across_pccafa(Sigma,xDim)
print('PITFALL 2: mean within-area rsc > 0 --> within-area interactions')
print(f'Within area 1 mean rsc: {x.mean():.2f}')
print(f'Within area 2 mean rsc: {y.mean():.2f}')
print(f'Across areas mean rsc: {a.mean():.2f}')
print('----------------------------------------------------------------')

# plot
fig2,ax = plt.subplots(constrained_layout=True)
fig2.supxlabel('spike count correlation ($r_{sc}$)', fontsize=18)
fig2.supylabel('proportion of neuron pairs', fontsize=18)

bins = np.arange(-1.05,1.1,0.1)
y_hat = 0.52
yrange = np.arange(0,0.7,0.1)

ax.hist(a,bins=bins,color=acrossarea,alpha=1,weights=np.ones(len(a)) / len(a))
ax.plot(a.mean(),y_hat,color=acrossarea,alpha=1,marker='v',markersize=10)
ax.axvline(x=0,color="black",linestyle="--",linewidth=1)
ax.set_xlim([-1,1])
ax.set_ylim([0,yrange[-1]])
ax.set_xticks(np.arange(-1,1.1,0.5))
ax.set_yticks(yrange)
ax.set_title('across',color=acrossarea)

# fig2.show()
pdf = PdfPages(FIGURE_PATH + 'fig1_pitfall2.pdf')
pdf.savefig(fig2)
pdf.close()

print('finished saving Figure 1 plots')