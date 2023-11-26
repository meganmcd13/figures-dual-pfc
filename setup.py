# imports
import sys, os

# paths
DATA_PATH = 'data/'
PCCAFA_PATH = 'core_pcca_fa/'
UTIL_PATH = 'core_utils/'
sys.path.append(DATA_PATH)
sys.path.append(PCCAFA_PATH)
sys.path.append(UTIL_PATH)

# output figure path
FIGURE_PATH = '../pcca_fa_figures_output/'
if not os.path.isdir(FIGURE_PATH):
    os.makedirs(FIGURE_PATH, exist_ok=True)
    print("created figure output folder : ", FIGURE_PATH)




# run figure
with open("fig4_real_data_ex.py") as f:
    exec(f.read())