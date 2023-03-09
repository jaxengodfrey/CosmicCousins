#!/usr/bin/env python

import paths
import numpy as np
import matplotlib.pyplot as plt
from utils import plot_mean_and_90CI, load_bsplinemass_ppd, plot_o3b_res, load_subpop_ppds
from matplotlib.ticker import ScalarFormatter
import matplotlib.gridspec as gridspec
import numpy as np
import seaborn as sns
from mpl_toolkits.axes_grid1.inset_locator import mark_inset
import deepdish as dd



# bspl_ms, bspl_mpdfs, bspl_qs, bspl_qpdfs = load_bsplinemass_ppd()
subpop_ppds = dd.io.load(paths.data / 'chi_eff_chi_p_ppds.h5')
# tot_subpops = subpop_ppds['peak_1_mass_ratio_pdfs'] + subpop_ppds['continuum_mass_ratio_pdfs']

figx, figy = 5, 3.5
legend_text_size = 10
label_text_size = 14
title_text_size = 12
fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(figx,figy))
# fig.suptitle('Model Group 2 Spin Distributions', fontsize = 20)
i_0 = 0
i_1 = 1

fill = 0.2

# # for i in range(samples['Base']['PeakA']['effsamples'].shape[0]):
# #     ax[i_0].hist(samples['Base']['PeakA']['effsamples'][i], alpha = 0.2, bins = 20)
# ax[i_0] = plot_mean_and_90CI(ax[i_0], subpop_ppds['Base']['PeakA']['chieffs'], subpop_ppds['Base']['PeakA']['pchieff'], color ='tab:cyan', label='Peak A', line_style = '--', bounds = True, lw = 2, fill_alpha = fill)
# # ax[i_0] = plot_mean_and_90CI(ax[i_0], subpop_ppds['mass_ratio'], subpop_ppds['peak_1_mass_ratio_pdfs'], color ='tab:cyan', label='Peak A', bounds = True, lw = 2, line_style = '--', fill_alpha = fill)
# # ax[i_0] = plot_mean_and_90CI(ax[i_0], subpop_ppds['a1'], subpop_ppds['peak_2_a1_pdfs'], color ='tab:purple', label='High-Mass Peak', bounds = True, lw = 3, line_style = '--')
# ax[i_0] = plot_mean_and_90CI(ax[i_0], subpop_ppds['Base']['ContinuumB']['chieffs'], subpop_ppds['Base']['ContinuumB']['pchieff'], color ='tab:pink', label='Continuum B', bounds = True, lw = 2, line_style = (0, (1, 1)), fill_alpha = fill)
# ax[i_0].legend(frameon=False, fontsize=legend_text_size);
# ax[i_0].set_xlabel(r'$\chi_{eff}$', fontsize=label_text_size)
# ax[i_0].set_ylabel(r'$p(\chi_{eff})$', fontsize=label_text_size)
# ax[i_0].grid(True, which="major", ls=":")
# ax[i_0].set_xlim(-0.75, 0.75)
# ax[i_0].set_ylim(0,5)
# # ax[i_0].set_yscale('log')
# ax[i_0].get_xaxis().set_major_formatter(ScalarFormatter())
# ax[i_0].set_title(r'Base Model $\chi_{eff}$ Distributions')
# ax[i_0].grid(True, which="major", ls=":")

# ax = plot_mean_and_90CI(ax, subpop_ppds['Composite']['PeakA']['chieffs'], subpop_ppds['Composite']['PeakA']['pchieff'], color ='tab:cyan', label='Peak A', bounds = True, lw = 2, line_style = '--', fill_alpha = fill)
ax = plot_mean_and_90CI(ax, subpop_ppds['Composite']['ContinuumA']['chieffs'], subpop_ppds['Composite']['ContinuumA']['pchieff'], color ='tab:purple', label='Peak A & Continuum A', bounds = True, lw = 2, line_style = '--', fill_alpha = fill)
ax = plot_mean_and_90CI(ax, subpop_ppds['Composite']['ContinuumB']['chieffs'], subpop_ppds['Composite']['ContinuumB']['pchieff'], color ='tab:pink', label='Continuum B', bounds = True, lw = 2, line_style = (0, (1, 1)), fill_alpha = fill)
ax.legend(frameon=False, loc = 2, fontsize=legend_text_size);
ax.set_xlabel(r'$\chi_{eff}$', fontsize=label_text_size)
ax.set_ylabel(r'$p(\chi_{eff})$', fontsize=label_text_size)
ax.grid(True, which="major", ls=":")
ax.set_xlim(-0.75, 0.75)
ax.set_ylim(0, 5)
# ax[i_1].set_yscale('log')
ax.set_title(r'Composite Model $\chi_{eff}$ Distributions')
ax.get_xaxis().set_major_formatter(ScalarFormatter())
ax.grid(True, which="major", ls=":")



# plt.title(f'GWTC-3: Spin Tilt Distribution', fontsize=title_text_size);
fig.tight_layout()
plt.savefig(paths.figures / 'chi_eff_distribution_plot.pdf', dpi=300);
plt.savefig(paths.figures / 'chi_eff_distribution_plot.jpeg', dpi=300);