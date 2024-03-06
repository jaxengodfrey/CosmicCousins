#!/usr/bin/env python

import paths
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from utils import plot_mean_and_90CI, load_bsplinemass_ppd, plot_o3b_res, load_subpop_ppds, load_macro
from matplotlib.ticker import ScalarFormatter
import matplotlib.gridspec as gridspec
import numpy as np
import seaborn as sns
from mpl_toolkits.axes_grid1.inset_locator import mark_inset
import deepdish as dd
from gwinfernodata import GWInfernoData
import matplotlib
matplotlib.rcParams['text.usetex'] = True


base_label = load_macro('base')
comp_label = load_macro('comp')
popA_label = load_macro('popA')
popB_label = load_macro('popB')

subpop_ppds = GWInfernoData.from_netcdf(paths.data / "bspline_composite_marginalized_fixtau_m1-s25-z1_msig15_qsig5_ssig5_zsig1_sigp3_NeffNobs_downsample_100k_rng6-10_ppds.h5").pdfs

figx, figy = 5, 3.5
legend_text_size = 10
label_text_size = 14
title_text_size = 12
fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(figx,figy))
i_0 = 0
i_1 = 1

fill = 0.2

q = np.random.uniform(0,1,10000)
a1 = np.random.uniform(0,1,10000)
a2 = np.random.uniform(0,1,10000)
ct1 = np.random.uniform(-1,1,10000)
ct2 = np.random.uniform(-1,1,10000)
chi = (a1*ct1 + a2*ct2*q)/(1+q)

ax = plot_mean_and_90CI(ax, subpop_ppds['chi_eff'], subpop_ppds['peak_continuum_chi_eff_pdfs'], color ='tab:purple', label=popA_label, bounds = True, lw = 2, line_style = '--', fill_alpha = fill)
ax = plot_mean_and_90CI(ax, subpop_ppds['chi_eff'], subpop_ppds['continuum_chi_eff_pdfs'], color ='tab:pink', label=popB_label, bounds = True, lw = 2, line_style = (0, (1, 1)), fill_alpha = fill)
kde1 = sns.kdeplot(x=chi, ax=ax, color='gray', common_norm = True, ls = 'dotted', lw = 1)
ax.legend(frameon=False, loc = 2, fontsize=legend_text_size);
ax.set_xlabel(r'$\chi_\mathrm{{eff}}$', fontsize=label_text_size)
ax.set_ylabel(r'$p(\chi_\mathrm{{eff}})$', fontsize=label_text_size)
ax.grid(True, which="major", ls=":")
ax.set_xlim(-0.75, 0.75)
ax.set_ylim(0, 5)     
# ax[i_1].set_yscale('log')
ax.set_title(r'{} $\chi_\mathrm{{eff}}$ Distributions'.format(comp_label))
ax.get_xaxis().set_major_formatter(ScalarFormatter())
ax.grid(True, which="major", ls=":")


# plt.title(f'GWTC-3: Spin Tilt Distribution', fontsize=title_text_size);
fig.tight_layout()
plt.savefig(paths.figures / 'chi_eff_distribution_plot.pdf', dpi=300, bbox_inches='tight');
plt.savefig(paths.figures / 'chi_eff_distribution_plot.jpeg', dpi=300, bbox_inches='tight');