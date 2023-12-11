# imports
import sys, os
import numpy as np
import pandas as pd
import scipy.linalg as slin
import scipy.stats as stats
import scipy.io as sio
from scipy.spatial.distance import pdist, squareform
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
plt.style.use('scifigs.mplstyle')

# paths
DATA_PATH = 'data/'
PCCAFA_PATH = 'pcca_fa/'
UTIL_PATH = 'pcca_fa/utils'
sys.path.append(DATA_PATH)
sys.path.append(PCCAFA_PATH)
sys.path.append(UTIL_PATH)

# package and model imports
import pcca_fa.pcca_fa_mdl as pf
import counts_pkg.counts_analysis as cp

# output figure path
FIGURE_PATH = '../pcca_fa_figures_output/'
if not os.path.isdir(FIGURE_PATH):
    os.makedirs(FIGURE_PATH, exist_ok=True)
    print("created figure output folder : ", FIGURE_PATH)
color_map = {
    'across':np.array([255,76,178])/255, # pink
    'within1':np.array([111,192,255])/255, # light blue - right hemisphere
    'within2':np.array([0,87,154])/255, # dark blue - left hemisphere
    'within':np.array([0,144,255])/255, # medium blue - collapsed across both hemispheres
    'independent':np.array([200,200,200])/255 # gray
}
# plotting colors
within = color_map['within']
area1 = color_map['within1']
area2 = color_map['within2']
acrossarea = color_map['across']
indep = color_map['independent']

# create utils, params and helper functions
from utils import jitter, compute_rsc_within_pccafa, compute_rsc_across_pccafa, plot_raster, load_dict, flatten, extract_mdl_params, preprocess_counts, plot_metric, get_top_angles, get_top_vec

# set which figures to create:
FIGURE1 = True
FIGURE2 = True
FIGURE3 = True
FIGURE4 = True
FIGURE5 = True
FIGURE6 = True
FIGURE7 = True

# run figures
if FIGURE1:
    with open("fig1_pitfall_intro.py") as f:
        exec(f.read())
if FIGURE2:
    with open("fig2_pccafa_model.py") as f:
        exec(f.read())
if FIGURE3:
    with open("fig3_model_validation.py") as f:
        exec(f.read())
if FIGURE4:
    with open("fig4_real_data_ex.py") as f:
        exec(f.read())
if FIGURE5:
    with open("fig5_model_metrics.py") as f:
        exec(f.read())
if FIGURE6:
    with open("fig6_subspaces.py") as f:
        exec(f.read())
if FIGURE7:
    with open("fig7_pupil_prediction.py") as f:
        exec(f.read())