# params
subjects = ['wakko', 'pepe', 'satchel']
sub_cols = ('k','k','k')
marks = ('o','s','^')
            
# load data and get params
slow_pupil_dat = load_dict("data/slow_pupil_latents.txt")
fast_pupil_dat = load_dict("data/fast_pupil_latents.txt")

# compute r2 values
cv_r2 = {}
null_r2 = {}
for sub in subjects:
    cv_r2[sub] = {'glb':[],'left':[],'right':[]}
    null_r2[sub] = {'glb':[],'left':[],'right':[]}
    for i in fast_pupil_dat[sub]:
        for latent in ['glb','left','right']:
            # true scores
            if fast_pupil_dat[sub][i][latent].shape[1]>1:
                x = fast_pupil_dat[sub][i][latent]
            else:
                x = fast_pupil_dat[sub][i][latent].reshape(-1,1)
            y = fast_pupil_dat[sub][i]['pupil'].reshape(-1, 1)
            lm = LinearRegression().fit(x,y)
            cv_r2[sub][latent].append(lm.score(x,y))
                
            for j in fast_pupil_dat[sub]:
                if i!=j:
                    N = min([len(fast_pupil_dat[sub][i]['pupil']),len(fast_pupil_dat[sub][j]['pupil'])])
                    
                    if fast_pupil_dat[sub][i][latent][0:N,:].shape[1]>1:
                        x = fast_pupil_dat[sub][i][latent][0:N,:]
                    else:
                        x = fast_pupil_dat[sub][i][latent][0:N,:].reshape(-1,1)
                    y = fast_pupil_dat[sub][j]['pupil'][0:N].reshape(-1, 1)
                    lm = LinearRegression().fit(x,y)
                    null_r2[sub][latent].append(lm.score(x,y))


fig,ax = plt.subplots()
plt.subplots_adjust(left=0.2)

for i,latent in enumerate(['glb','left','right']):
    for j,sub in enumerate(subjects):
        col = sub_cols[j]
        curr = np.array(cv_r2[sub][latent])
        if i==0:
            ax.errorbar(i*3+j*.7,curr.mean(),yerr=curr.std()/np.sqrt(len(curr)),color=col,marker=marks[j],markersize=5,label=f'subject {sub[0].upper()+sub[1]}')
        else:
            ax.errorbar(i*3+j*.7,curr.mean(),yerr=curr.std()/np.sqrt(len(curr)),color=col,marker=marks[j],markersize=5)
        null = np.array(null_r2[sub][latent])
        null_prc = np.percentile(null,[0,95])
        ax.plot(i*3+j*.7,null.mean(),color='k',alpha=0.4,linewidth=3)
        ax.plot([i*3+j*.7,i*3+j*.7],null_prc,'k-',alpha=0.4,linewidth=3)

for j,sub in enumerate(subjects):
    _,pdim = stats.ttest_rel(cv_r2[sub]['glb'],cv_r2[sub]['left'],alternative='greater')
    print(f'subject {sub[:2]}, accross > within left, p={pdim:.4f}')
    _,pdim = stats.ttest_rel(cv_r2[sub]['glb'],cv_r2[sub]['right'],alternative='greater')
    print(f'subject {sub[:2]}, across > within right, p={pdim:.4f}')

# formatting
ax.set_xticks(ticks=[0.7,3.7,6.7])
ax.set_xticklabels(['across','within left','within right'])
ax.legend()
ax.set_ylabel('$r^2$')
ax.set_title('prediction of pupil')

# fig.show()
pdf = PdfPages(FIGURE_PATH + 'fig7_pupil_pred_r2.pdf')
pdf.savefig(fig)
pdf.close()

# example session pupil prediction
ex_dat = fast_pupil_dat['satchel'][4]
y = ex_dat['pupil']

# global prediciton
lm = LinearRegression().fit(ex_dat['glb'],y)
y_hat_glb = lm.predict(ex_dat['glb'])
glb_r2 = lm.score(ex_dat['glb'],y)

# local prediction
lm = LinearRegression().fit(ex_dat['left'],y)
y_hat_left = lm.predict(ex_dat['left'])
left_r2 = lm.score(ex_dat['left'],y)

lm = LinearRegression().fit(ex_dat['right'],y)
y_hat_right = lm.predict(ex_dat['right'])
right_r2 = lm.score(ex_dat['right'],y)

fig,ax = plt.subplots(1,1)
plt.subplots_adjust(left=0.2)

start,end = 300,360
idx = np.arange(start,end,1)
ax.plot(idx,y[start:end],'k',label='Actual pupil')
ax.plot(idx,y_hat_glb[start:end],label='across-area pred', color=acrossarea)
ax.plot(idx,y_hat_left[start:end],label='within left pred', color=area2)
ax.plot(idx,y_hat_right[start:end],label='within right pred', color=area1)
ax.legend(loc='best')
ax.set_xlabel('trial number')
ax.set_ylabel('pupil size\n(normalized)')

# fig.show()
pdf = PdfPages(FIGURE_PATH + 'fig7_pupil_pred.pdf')
pdf.savefig(fig)
pdf.close()

print('finished saving Figure 6 plots')
print('- - - - - - - - - - - - - - - -')