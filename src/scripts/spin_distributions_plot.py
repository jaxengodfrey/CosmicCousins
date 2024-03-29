import paths
import matplotlib as mpl
import matplotlib.pyplot as plt
import deepdish as dd
from utils import plot_mean_and_90CI, load_subpop_ppds, load_macro, plot_90CI, load_gwinfernodata_ppds
from matplotlib.ticker import ScalarFormatter
from gwinfernodata import GWInfernoData
import numpy as np
import matplotlib
matplotlib.rcParams['text.usetex'] = True




base_label = load_macro('base')
comp_label = load_macro('comp')
popA_label = load_macro('popA')
popB_label = load_macro('popB')

figx, figy = 10, 7
legend_text_size = 12
label_text_size = 14
title_text_size = 12
fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(figx,figy))
# fig.suptitle('Model Group 2 Spin Distributions', fontsize = 20)
i_0 = 0,0
i_1 = 1,0
i_2 = 1,1
i_3 = 0,1

subpop_ppds = load_gwinfernodata_ppds().pdfs
# prior_ppds = dd.io.load(paths.data / 'bspline_1logpeak_ss_prior_marginalized_10000s_ppds.h5')

ax[i_0] = plot_mean_and_90CI(ax[i_0], subpop_ppds['a1'], subpop_ppds['peak_1_a1_pdfs'], color ='tab:cyan', label=popA_label, bounds = True, lw = 3, line_style = '--')
ax[i_0] = plot_mean_and_90CI(ax[i_0], subpop_ppds['a1'], subpop_ppds['continuum_a1_pdfs'], color ='tab:pink', label=popB_label, bounds = True, lw = 3, line_style = (0, (1, 1)))
# ax[i_0] = plot_90CI(ax[i_0], prior_ppds['a1'], prior_ppds['continuum_a1_pdfs'], color='gray', label = 'prior', lw = 1, line_style = '--')
ax[i_0].legend(frameon=False, fontsize=legend_text_size);
ax[i_0].set_xlabel(r'$a$', fontsize=label_text_size)
ax[i_0].set_ylabel(r'$p(a)$', fontsize=label_text_size)
ax[i_0].grid(True, which="major", ls=":")
ax[i_0].set_xlim(0, 1)
ax[i_0].set_ylim(0,4)
ax[i_0].get_xaxis().set_major_formatter(ScalarFormatter())
ax[i_0].set_title('{} Spin Magnitude Distributions'.format(base_label))
ax[i_0].grid(True, which="major", ls=":")

ax[i_1] = plot_mean_and_90CI(ax[i_1], subpop_ppds['cos_tilt_1'], subpop_ppds['peak_1_ct1_pdfs'], color ='tab:cyan', label=popA_label, bounds = True, lw = 3, line_style = '--')
ax[i_1] = plot_mean_and_90CI(ax[i_1], subpop_ppds['cos_tilt_1'], subpop_ppds['continuum_ct1_pdfs'], color ='tab:pink', label=popB_label, bounds = True, lw = 3, line_style = (0, (1, 1)))
# ax[i_1] = plot_90CI(ax[i_1], prior_ppds['cos_tilt_1'], prior_ppds['continuum_ct1_pdfs'], color='gray', label = 'prior', lw = 1, line_style = '--')
ax[i_1].set_xlabel(r'$\cos{t}$', fontsize=label_text_size)
ax[i_1].set_ylabel(r'$p(\cos{t})$', fontsize=label_text_size)
ax[i_1].grid(True, which="major", ls=":")
ax[i_1].set_xlim(-1, 1)
ax[i_1].set_ylim(0,1.2)
ax[i_1].set_title('{} Tilt Distributions'.format(base_label))
ax[i_1].get_xaxis().set_major_formatter(ScalarFormatter())
ax[i_1].grid(True, which="major", ls=":")

subpop_ppds_1 = load_gwinfernodata_ppds(IP = False).pdfs#dd.io.load(paths.data / 'spin_popfrac_posteriors.h5')


xmin, xmax = -1, 1
ax[i_2] = plot_mean_and_90CI(ax[i_2], subpop_ppds_1['cos_tilt_1'], subpop_ppds_1['peak_continuum_ct1_pdfs'], color ='tab:purple', label=popA_label, bounds = True, lw = 3, line_style = '--')
ax[i_2] = plot_mean_and_90CI(ax[i_2],  subpop_ppds_1['cos_tilt_1'], subpop_ppds_1['continuum_ct1_pdfs'], color ='tab:pink', label=popB_label, bounds = True, lw = 3, line_style = (0, (1, 1)))
# ax[i_2] = plot_90CI(ax[i_2], prior_ppds['cos_tilt_1'], prior_ppds['continuum_ct1_pdfs'], color='gray', label = 'prior', lw = 1, line_style = '--')
ax[i_2].set_xlabel(r'$\cos{t}$', fontsize=label_text_size)
ax[i_2].set_ylabel(r'$p(\cos{t})$', fontsize=label_text_size)
ax[i_2].set_xlim(xmin, xmax)
ax[i_2].set_title('{} Tilt Distributions'.format(comp_label))
ax[i_2].grid(True, which="major", ls=":")
ax[i_2].set_ylim(0, 1.2)

xmin, xmax = 0, 1
ax[i_3] = plot_mean_and_90CI(ax[i_3], subpop_ppds_1['a1'], subpop_ppds_1['peak_continuum_a1_pdfs'], color ='tab:purple', label=popA_label, bounds = True, lw = 3, line_style = '--')
ax[i_3] = plot_mean_and_90CI(ax[i_3], subpop_ppds_1['a1'], subpop_ppds_1['continuum_a1_pdfs'], color ='tab:pink', label=popB_label, bounds = True, lw = 3, line_style = (0, (1, 1)))
# ax[i_3] = plot_90CI(ax[i_3], prior_ppds['a1'], prior_ppds['continuum_a1_pdfs'], color='gray', label = 'prior', lw = 1, line_style = '--')
ax[i_3].legend(frameon=False, fontsize=legend_text_size);
ax[i_3].set_xlabel(r'$a$', fontsize=label_text_size)
ax[i_3].set_ylabel(r'$p(a)$', fontsize=label_text_size)
ax[i_3].set_xlim(xmin, xmax)
ax[i_3].set_title('{} Spin Magnitude Distributions'.format(comp_label))
ax[i_3].set_ylim(0, 4)
ax[i_3].grid(True, which="major", ls=":")


# plt.title(f'GWTC-3: Spin Tilt Distribution', fontsize=title_text_size);
fig.tight_layout()
plt.savefig(paths.figures / 'spin_distributions_plot.pdf', dpi=300, bbox_inches='tight');
plt.savefig(paths.figures / 'spin_distributions_plot.jpeg', dpi=300, bbox_inches='tight');
