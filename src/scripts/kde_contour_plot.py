import paths
import numpy as np
import matplotlib.pyplot as plt
from utils import plot_mean_and_90CI, load_bsplinemass_ppd, plot_o3b_res, load_subpop_ppds
from matplotlib.ticker import ScalarFormatter

idata = dd.io.load('bspline_mass_spin_1000w_10000s_thin2_independent_bspline_ratio_12-12-22.h5')

idata.posterior