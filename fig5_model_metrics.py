# parameters
subjects = ('pe','wa','sa')
plot_col = ('b','k','r')
plot_sym = ('o','s','^')

# create graphics 
fig,ax = plt.subplots(1,2)
fig.set_figwidth(2*fig.get_figwidth())

# loop through each monkey
d_within,d_across = [np.zeros(0) for _ in range(2)]
psv_within,psv_across = [np.zeros(0) for _ in range(2)]
for sub,sym in zip(subjects,plot_sym):
    # load data
    mat_fname = 'data/{:s}_nomean_noar25_delay.mat'.format(sub)
    dat = sio.loadmat(mat_fname)
    
    d_wi,d_ac = plot_metric(dat,sub,ax[1],'k',sym,metric='dshared')
    d_within,d_across = np.append(d_within,d_wi),np.append(d_across,d_ac)
    
    psv_wi,psv_ac = plot_metric(dat,sub,ax[0],'k',sym,metric='psv')
    psv_within,psv_across = np.append(psv_within,psv_wi),np.append(psv_across,psv_ac)
    
    # test if across and within are different for this monkey
    _,pdim = stats.ttest_rel(d_ac,d_wi)
    _,ppsv = stats.ttest_rel(psv_ac,psv_wi)
    # print('{:s}, dim: p={:5f}'.format(sub,pdim))
    # print('{:s}, psv: p={:5f}'.format(sub,ppsv))

# test if across and within are different across all monkeys
_,p_dim = stats.ttest_rel(d_across,d_within)
_,p_psv = stats.ttest_rel(psv_across,psv_within)
sem_dim_across, sem_dim_within = stats.sem(d_across), stats.sem(d_within)
sem_psv_across, sem_psv_within = stats.sem(psv_across), stats.sem(psv_within)
print('All subjects, dim: global={:5f} +/- {:5f} s.e.m., local={:5f} +/- {:5f} s.e.m., p={:5f}'.format(d_across.mean(),sem_dim_across,d_within.mean(),sem_dim_within,p_dim))
print('All subjects, psv: global={:5f} +/- {:5f} s.e.m., local={:5f} +/- {:5f} s.e.m., p={:5f}'.format(psv_across.mean(),sem_psv_across,psv_within.mean(),sem_psv_within,p_psv))

# make plots pretty and display them
ax[0].legend(loc='best')
ax[1].plot([0,25],[0,25],'k--')
ax[1].set_title('$d_{shared}$')
ax[0].set_xlabel('within-area', color=within), ax[0].set_ylabel('across-area', color=acrossarea)
ax[1].set_xlim([0,20]),ax[1].set_ylim([0,20])
ax[1].set_xticks(np.arange(0,20.1,5)), ax[1].set_yticks(np.arange(5,20.1,5))
ax[0].plot([0,50],[0,50],'k--')
ax[0].set_title('% shared variance')
ax[1].set_xlabel('within-area', color=within), ax[1].set_ylabel('across-area', color=acrossarea)
ax[0].set_xlim([0,50]),ax[0].set_ylim([0,50])
ax[0].set_xticks(np.arange(0,50,15)), ax[0].set_yticks(np.arange(15,50,15))

ax[0].set_aspect('equal')
ax[1].set_aspect('equal')

# fig.show()
pdf = PdfPages(FIGURE_PATH + 'fig5_across_vs_within.pdf')
pdf.savefig(fig)
pdf.close()

# difference histogram plot - insets
fig,ax = plt.subplots(1,2)
fig.set_figwidth(2*fig.get_figwidth())

d_diff = d_within - d_across
psv_diff = psv_within - psv_across

ax[0].plot(d_diff.mean(),15,marker='v',color=[.5,.5,.5])
ax[0].hist(d_diff,bins=np.arange(15)-7.5,color=[.5,.5,.5])
ax[0].axvline(x=0,color="black",linestyle="--",linewidth=1)
ax[1].plot(psv_diff.mean(),12,marker='v',color=[.5,.5,.5])
ax[1].hist(psv_diff,bins=25,color=[.5,.5,.5])
ax[1].axvline(x=0,color="black",linestyle="--",linewidth=1)


# fig.show()
pdf = PdfPages(FIGURE_PATH + 'fig5_metric_diffs.pdf')
pdf.savefig(fig)
pdf.close()

print('finished saving Figure 5 plots')
print('- - - - - - - - - - - - - - - -')