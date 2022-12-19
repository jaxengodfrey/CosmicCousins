import paths
import numpy as np
import matplotlib.pyplot as plt
from utils import plot_mean_and_90CI, load_bsplinemass_ppd, plot_o3b_res, load_subpop_ppds
from matplotlib.ticker import ScalarFormatter


figx, figy = 5, 7
legend_text_size = 10
label_text_size = 14
title_text_size = 12
fig, ax = plt.subplots(nrows=2, ncols=1, figsize=(figx,figy))
i_0 = 0
i_1 = 1

subpop_ppds = load_subpop_ppds()
ax[i_0] = plot_mean_and_90CI(ax[i_0], subpop_ppds['a1'], subpop_ppds['peak_1_a1_pdfs'], color ='tab:cyan', label='Low-Mass Peak', bounds = True, lw = 3)
ax[i_0] = plot_mean_and_90CI(ax[i_0], subpop_ppds['a1'], subpop_ppds['peak_2_a1_pdfs'], color ='tab:purple', label='Mid-Mass Peak', bounds = True, lw = 3)
ax[i_0] = plot_mean_and_90CI(ax[i_0], subpop_ppds['a1'], subpop_ppds['continuum_a1_pdfs'], color ='tab:pink', label='Continuum', bounds = True, lw = 3)
ax[i_0].legend(frameon=False, fontsize=legend_text_size);
ax[i_0].set_xlabel(r'$a_1$', fontsize=label_text_size)
ax[i_0].set_ylabel(r'$p(a_1)$', fontsize=label_text_size)
ax[i_0].grid(True, which="major", ls=":")
ax[i_0].set_xlim(0, 1)
ax[i_0].get_xaxis().set_major_formatter(ScalarFormatter())
ax[i_0].set_title('Primary Spin Magnitude Distribution')
ax[i_0].grid(True, which="major", ls=":")

ax[i_1] = plot_mean_and_90CI(ax[i_1], subpop_ppds['cos_tilt_1'], subpop_ppds['peak_1_tilt1_pdfs'], color ='tab:cyan', label='Low-Mass Peak', bounds = True, lw = 3)
ax[i_1] = plot_mean_and_90CI(ax[i_1], subpop_ppds['cos_tilt_1'], subpop_ppds['peak_2_tilt1_pdfs'], color ='tab:purple', label='Mid-Mass Peak', bounds = True, lw = 3)
ax[i_1] = plot_mean_and_90CI(ax[i_1], subpop_ppds['cos_tilt_1'], subpop_ppds['continuum_tilt1_pdfs'], color ='tab:pink', label='Continuum', bounds = True, lw = 3)
ax[i_1].legend(frameon=False, fontsize=legend_text_size);
ax[i_1].set_xlabel(r'$cos(\theta_1)$', fontsize=label_text_size)
ax[i_1].set_ylabel(r'$p(cos(\theta_1))$', fontsize=label_text_size)
ax[i_1].grid(True, which="major", ls=":")
ax[i_1].set_xlim(-1, 1)
ax[i_1].set_title('Primary Tilt Distribution')
ax[i_1].get_xaxis().set_major_formatter(ScalarFormatter())
ax[i_1].grid(True, which="major", ls=":")

# plt.title(f'GWTC-3: BBH Primary Spin Magnitude Distribution', fontsize=title_text_size);
fig.tight_layout()
plt.savefig(paths.figures / 'spin_mag_distribution_plot.pdf', dpi=300);
plt.close()


