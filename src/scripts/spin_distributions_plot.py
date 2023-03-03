import paths
import numpy as np
import matplotlib.pyplot as plt
from utils import plot_mean_and_90CI, load_subpop_ppds
import deepdish as dd
from matplotlib.ticker import ScalarFormatter


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

subpop_ppds = load_subpop_ppds(g1 = True, g1_fname = 'bspline_1logpeak_100000s_ppds.h5')

ax[i_0] = plot_mean_and_90CI(ax[i_0], subpop_ppds['a1'], subpop_ppds['peak_1_a1_pdfs']['unweighted'], color ='tab:cyan', label='Peak A', bounds = True, lw = 3, line_style = '--')
ax[i_0] = plot_mean_and_90CI(ax[i_0], subpop_ppds['a1'], subpop_ppds['continuum_a1_pdfs']['unweighted'], color ='tab:pink', label='Continuum B', bounds = True, lw = 3, line_style = (0, (1, 1)))
ax[i_0].legend(frameon=False, fontsize=legend_text_size);
ax[i_0].set_xlabel(r'$a$', fontsize=label_text_size)
ax[i_0].set_ylabel(r'$p(a)$', fontsize=label_text_size)
ax[i_0].grid(True, which="major", ls=":")
ax[i_0].set_xlim(0, 1)
ax[i_0].set_ylim(0,3)
ax[i_0].get_xaxis().set_major_formatter(ScalarFormatter())
ax[i_0].set_title('Base Model Spin Magnitude Distributions')
ax[i_0].grid(True, which="major", ls=":")

ax[i_1] = plot_mean_and_90CI(ax[i_1], subpop_ppds['cos_tilt_1'], subpop_ppds['peak_1_ct1_pdfs']['unweighted'], color ='tab:cyan', label='Peak A', bounds = True, lw = 3, line_style = '--')
ax[i_1] = plot_mean_and_90CI(ax[i_1], subpop_ppds['cos_tilt_1'], subpop_ppds['continuum_ct1_pdfs']['unweighted'], color ='tab:pink', label='Continuum B', bounds = True, lw = 3, line_style = (0, (1, 1)))
ax[i_1].set_xlabel(r'$cos(\theta)$', fontsize=label_text_size)
ax[i_1].set_ylabel(r'$p(cos(\theta))$', fontsize=label_text_size)
ax[i_1].grid(True, which="major", ls=":")
ax[i_1].set_xlim(-1, 1)
ax[i_1].set_ylim(0,1.2)
ax[i_1].set_title('Base Model Tilt Distributions')
ax[i_1].get_xaxis().set_major_formatter(ScalarFormatter())
ax[i_1].grid(True, which="major", ls=":")

subpop_ppds_1 = load_subpop_ppds(g2 = True, g2_fname = 'bspline_1logpeak_samespin_100000s_2chains.h5')#dd.io.load(paths.data / 'spin_popfrac_posteriors.h5')

xmin, xmax = -1, 1
ax[i_2] = plot_mean_and_90CI(ax[i_2], subpop_ppds_1['cos_tilt_1'], subpop_ppds_1['peak_1_continuum_ct1_pdfs']['unweighted'], color ='tab:purple', label='Peak A & Continuum A', bounds = True, lw = 3, line_style = '--')
ax[i_2] = plot_mean_and_90CI(ax[i_2],  subpop_ppds_1['cos_tilt_1'], subpop_ppds_1['continuum_ct1_pdfs']['unweighted'], color ='tab:pink', label='Continuum B', bounds = True, lw = 3, line_style = (0, (1, 1)))
ax[i_2].set_xlabel(r'$\cos{\theta}$', fontsize=label_text_size)
ax[i_2].set_ylabel(r'$p(\cos{\theta})$', fontsize=label_text_size)
ax[i_2].set_xlim(xmin, xmax)
ax[i_2].set_title('Composite Model Tilt Distribution')
ax[i_2].grid(True, which="major", ls=":")
ax[i_2].set_ylim(0, 1.2)

xmin, xmax = 0, 1
ax[i_3] = plot_mean_and_90CI(ax[i_3], subpop_ppds_1['a1'], subpop_ppds_1['peak_1_continuum_a1_pdfs']['unweighted'], color ='tab:purple', label='Peak A & Continuum A', bounds = True, lw = 3, line_style = '--')
ax[i_3] = plot_mean_and_90CI(ax[i_3], subpop_ppds_1['a1'], subpop_ppds_1['continuum_a1_pdfs']['unweighted'], color ='tab:pink', label='Continuum B', bounds = True, lw = 3, line_style = (0, (1, 1)))
ax[i_3].legend(frameon=False, fontsize=legend_text_size);
ax[i_3].set_xlabel(r'$a$', fontsize=label_text_size)
ax[i_3].set_ylabel(r'$p(a)$', fontsize=label_text_size)
ax[i_3].set_xlim(xmin, xmax)
ax[i_3].set_title('Composite Model Spin Magnitude Distribution')
ax[i_3].set_ylim(0, 3)
ax[i_3].grid(True, which="major", ls=":")

# plt.title(f'GWTC-3: Spin Tilt Distribution', fontsize=title_text_size);
fig.tight_layout()
plt.savefig(paths.figures / 'spin_distributions_plot.pdf', dpi=300);
plt.savefig(paths.figures / 'spin_distributions_plot.jpeg', dpi=300);
