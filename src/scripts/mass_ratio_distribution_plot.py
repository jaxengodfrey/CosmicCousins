#!/usr/bin/env python

import paths
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from utils import plot_mean_and_90CI, load_bsplinemass_ppd, plot_o3b_res, load_subpop_ppds, load_macro, load_gwinfernodata_ppds
from matplotlib.ticker import ScalarFormatter
import matplotlib.gridspec as gridspec
import numpy as np
import seaborn as sns
from mpl_toolkits.axes_grid1.inset_locator import mark_inset
from gwinfernodata import GWInfernoData
import matplotlib
matplotlib.rcParams['text.usetex'] = True


base_label = load_macro('base')
comp_label = load_macro('comp')
first_label = load_macro('first')
contA_label = load_macro('contA')
contB_label = load_macro('contB')
msun = load_macro('msun')

bspl_ms, bspl_mpdfs, bspl_qs, bspl_qpdfs = load_bsplinemass_ppd()
subpop_ppds = load_gwinfernodata_ppds().pdfs
tot_subpops = subpop_ppds['peak_1_mass_ratio_pdfs'].values + subpop_ppds['continuum_mass_ratio_pdfs'].values

figx, figy = 5, 7
legend_text_size = 10
label_text_size = 14
title_text_size = 12
fig, ax = plt.subplots(nrows=2, ncols=1, figsize=(figx,figy))
# fig.suptitle('Model Group 2 Spin Distributions', fontsize = 20)
i_0 = 0
i_1 = 1

fill = 0.2
ax[i_0] = plot_mean_and_90CI(ax[i_0], bspl_qs, bspl_qpdfs, color ='tab:red', label='Edelman et. al. 2022', bounds = False, lw = 2)
ax[i_0] = plot_mean_and_90CI(ax[i_0], subpop_ppds['mass_ratio'], tot_subpops, color ='k', label='Total', bounds = False, lw = 2)
ax[i_0] = plot_mean_and_90CI(ax[i_0], subpop_ppds['mass_ratio'], subpop_ppds['peak_1_mass_ratio_pdfs'], color ='tab:cyan', label=first_label, bounds = True, lw = 2, line_style = '--', fill_alpha = fill)
# ax[i_0] = plot_mean_and_90CI(ax[i_0], subpop_ppds['a1'], subpop_ppds['peak_2_a1_pdfs'], color ='tab:purple', label='High-Mass Peak', bounds = True, lw = 3, line_style = '--')
ax[i_0] = plot_mean_and_90CI(ax[i_0], subpop_ppds['mass_ratio'], subpop_ppds['continuum_mass_ratio_pdfs'], color ='tab:pink', label=contB_label, bounds = True, lw = 2, line_style = (0, (1, 1)), fill_alpha = fill)
ax[i_0].legend(frameon=False, fontsize=legend_text_size);
ax[i_0].set_xlabel('$q$', fontsize=label_text_size)
ax[i_0].set_ylabel(r'$p(q)$', fontsize=label_text_size)
ax[i_0].grid(True, which="major", ls=":")
ax[i_0].set_xlim(0, 1)
ax[i_0].set_ylim(1e-2,10)
ax[i_0].set_yscale('log')
ax[i_0].get_xaxis().set_major_formatter(ScalarFormatter())
ax[i_0].set_title('{} Mass Ratio Distributions'.format(base_label))
ax[i_0].grid(True, which="major", ls=":")

subpop_ppds = load_gwinfernodata_ppds(IP = False).pdfs
tot_subpops = subpop_ppds['peak_1_mass_ratio_pdfs'].values  + subpop_ppds['continuum_mass_ratio_pdfs'].values + subpop_ppds['continuum_1_mass_ratio_pdfs'].values

ax[i_1] = plot_mean_and_90CI(ax[i_1], bspl_qs, bspl_qpdfs, color ='tab:red', label='Edelman et al 2022', bounds = False, lw = 2)
ax[i_1] = plot_mean_and_90CI(ax[i_1], subpop_ppds['mass_ratio'], tot_subpops, color ='k', label='Total', bounds = False, lw = 2)
ax[i_1] = plot_mean_and_90CI(ax[i_1], subpop_ppds['mass_ratio'], subpop_ppds['peak_1_mass_ratio_pdfs'], color ='tab:cyan', label=first_label, bounds = True, lw = 2, line_style = '--', fill_alpha = fill)
# ax[i_1] = plot_mean_and_90CI(ax[i_1], subpop_ppds['cos_tilt_1'], subpop_ppds['peak_2_tilt1_pdfs'], color ='tab:purple', label='High-Mass Peak', bounds = True, lw = 3, line_style = '--')
ax[i_1] = plot_mean_and_90CI(ax[i_1], subpop_ppds['mass_ratio'], subpop_ppds['continuum_1_mass_ratio_pdfs'], color ='tab:purple', label=contA_label, bounds = True, lw = 2, line_style = (0, (1, 1)), fill_alpha = fill)
ax[i_1] = plot_mean_and_90CI(ax[i_1], subpop_ppds['mass_ratio'], subpop_ppds['continuum_mass_ratio_pdfs'], color ='tab:pink', label=contB_label, bounds = True, lw = 2, line_style = (0, (1, 1)), fill_alpha = fill)
ax[i_1].legend(frameon=False, fontsize=legend_text_size);
ax[i_1].set_xlabel('$q$', fontsize=label_text_size)
ax[i_1].set_ylabel(r'$p(q)$', fontsize=label_text_size)
ax[i_1].grid(True, which="major", ls=":")
ax[i_1].set_xlim(0, 1)
ax[i_1].set_ylim(1e-2, 10)
ax[i_1].set_yscale('log')
ax[i_1].set_title('{} Mass Ratio Distributions'.format(comp_label))
ax[i_1].get_xaxis().set_major_formatter(ScalarFormatter())
ax[i_1].grid(True, which="major", ls=":")



# plt.title(f'GWTC-3: Spin Tilt Distribution', fontsize=title_text_size);
fig.tight_layout()
plt.savefig(paths.figures / 'mass_ratio_distribution_plot.pdf', dpi=300, bbox_inches='tight');
plt.savefig(paths.figures / 'mass_ratio_distribution_plot.jpeg', dpi=300, bbox_inches='tight');



