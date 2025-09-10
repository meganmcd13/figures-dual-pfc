# Interactions across hemispheres in prefrontal cortex reflect global cognitive processing

Data processing and figure generation for the following reference:

- McDonnell, M.E.&ast;, Umakantha, A.&ast;, Williamson, R.C.&ast;, Smith, M.A.&dagger;, & Yu, B. M.&dagger; Interactions across hemispheres in prefrontal cortex reflect global cognitive processing. bioRxiv (2025). <https://www.biorxiv.org/> (&ast; and &dagger; indicate equal contribution)

## Setup

This code suite requires the following packages, which should be placed inside the `helpers/` folder.

- pCCA-FA: <https://github.com/SmithLabNeuro/pcca_fa>
- FA: <https://github.com/meganmcd13/fa>
- CCA: <https://github.com/meganmcd13/cca>

The pCCA-FA package includes an `environment.yml` file, which should be used to create a conda environment from which to run this code suite.

## Data deposit

The data used to generate these figures has been deposited in the following [repository](). The data files should be downloaded and placed in the `raw_data_forms/` folder.

## Data preprocessing

The first preprocessing steps occur in MATLAB. The two following scripts should be run:

1) `compile_behav_neural_data.m`
2) `compile_pupil_data.m`

The remaining steps take place in Python, using the conda environment created in the pCCA-FA package. `fit_pccafa_models.py` should be run first, and the remaining scripts can be run in any order. Each script is briefly described below.

- `compute_pupil_pred_1d.py` - control analysis for Figure 6 to predict pupil diameter using 1 co-fluctuation pattern
- `compute_pupil_pred.py` - main analysis for Figure 6 to predict pupil diameter using all co-fluctuation patterns
- `compute_rsc.py` - compute across- and within-area $r_{sc}$ distributions and signal tuning for Figure 2 and Figure S2
- `create_fig5_chance.py` - control analysis for Figure 5 to fit pCCA-FA models to simulated data with a specified $\theta_{sim}$
- `create_figS3_dataset_varyDim.py` - model validation simulations where dimensionality is varied 
- `create_figS3_dataset_varyN.py` - model validation simulations where number of trials is varied 
- `create_figS3_dataset_varySv.py` - model validation simulations where shared variance is varied 
- `create_figS4_dataset_varyTheta.py` - model validation simulations where $\theta_{sim}$ is varied 
- `dual_pfc_funcs.py` - file containing utilities used in various analyses
- `fit_alt_models.py` - main analysis for Figure S5 to compare pCCA-FA to alternative models
- `fit_null_pccafa_models.py` - control analysis for Figure S1 to fit pCCA-FA to flipped neural activity
- `fit_pccafa_models.py` - main script to fit pCCA-FA models to neural population activity
- `fit_slow_pccafa_models.py` - analysis for Figure S1 to fit pCCA-FA to the estimated slow components of neural activity

## Figure generation

The components of each figure can be reproduced using the respective Python notebook.

## Contact

For questions, please contact Megan McDonnell at <mmcdonnell@cmu.edu>.
