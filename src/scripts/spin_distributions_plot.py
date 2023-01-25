import paths
import numpy as np
import matplotlib.pyplot as plt
from utils import plot_mean_and_90CI, load_bsplinemass_ppd, plot_o3b_res, load_subpop_ppds
import deepdish as dd
from matplotlib.ticker import ScalarFormatter


figx, figy = 10, 7
legend_text_size = 10
label_text_size = 14
title_text_size = 12
fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(figx,figy))
i_0 = 0,0
i_1 = 1,0
i_2 = 1,1
i_3 = 0,1

subpop_ppds = load_subpop_ppds()
ax[i_0] = plot_mean_and_90CI(ax[i_0], subpop_ppds['a1'], subpop_ppds['peak_1_a1_pdfs'], color ='tab:cyan', label='Low-Mass Peak', bounds = True, lw = 3, line_style = '--')
ax[i_0] = plot_mean_and_90CI(ax[i_0], subpop_ppds['a1'], subpop_ppds['peak_2_a1_pdfs'], color ='tab:purple', label='High-Mass Peak', bounds = True, lw = 3, line_style = '--')
ax[i_0] = plot_mean_and_90CI(ax[i_0], subpop_ppds['a1'], subpop_ppds['continuum_a1_pdfs'], color ='tab:pink', label='Continuum', bounds = True, lw = 3, line_style = '--')
ax[i_0].legend(frameon=False, fontsize=legend_text_size);
ax[i_0].set_xlabel(r'$a$', fontsize=label_text_size)
ax[i_0].set_ylabel(r'$p(a)$', fontsize=label_text_size)
ax[i_0].grid(True, which="major", ls=":")
ax[i_0].set_xlim(0, 1)
ax[i_0].get_xaxis().set_major_formatter(ScalarFormatter())
ax[i_0].set_title('GWTC-3: Subpopulation Spin Magnitude Distributions')
ax[i_0].grid(True, which="major", ls=":")

ax[i_1] = plot_mean_and_90CI(ax[i_1], subpop_ppds['cos_tilt_1'], subpop_ppds['peak_1_tilt1_pdfs'], color ='tab:cyan', label='Low-Mass Peak', bounds = True, lw = 3, line_style = '--')
ax[i_1] = plot_mean_and_90CI(ax[i_1], subpop_ppds['cos_tilt_1'], subpop_ppds['peak_2_tilt1_pdfs'], color ='tab:purple', label='High-Mass Peak', bounds = True, lw = 3, line_style = '--')
ax[i_1] = plot_mean_and_90CI(ax[i_1], subpop_ppds['cos_tilt_1'], subpop_ppds['continuum_tilt1_pdfs'], color ='tab:pink', label='Continuum', bounds = True, lw = 3, line_style = '--')
# ax[i_1].legend(frameon=False, fontsize=legend_text_size);
ax[i_1].set_xlabel(r'$cos(\theta)$', fontsize=label_text_size)
ax[i_1].set_ylabel(r'$p(cos(\theta))$', fontsize=label_text_size)
ax[i_1].grid(True, which="major", ls=":")
ax[i_1].set_xlim(-1, 1)
ax[i_1].set_title('GWTC-3: Subpopulation Tilt Distributions')
ax[i_1].get_xaxis().set_major_formatter(ScalarFormatter())
ax[i_1].grid(True, which="major", ls=":")

from utils import plot_mean_and_90CI, load_iid_tilt_ppd, plot_o3b_spintilt, plot_o3b_spinmag, load_iid_mag_ppd

mag_and_tilt_pdfs = dd.io.load(paths.data / 'spin_popfrac_posteriors.h5')
tilt_tot = mag_and_tilt_pdfs['peak_1_tilt1_pdfs'] + mag_and_tilt_pdfs['peak_2_tilt1_pdfs'] + mag_and_tilt_pdfs['continuum_tilt1_pdfs']
xmin, xmax = -1, 1
xs, ct_pdfs = load_iid_tilt_ppd()
for jj in range(len(ct_pdfs)):
    ct_pdfs[jj,:] /= np.trapz(ct_pdfs[jj,:], xs)
ax[i_2] = plot_o3b_spintilt(ax[i_2],'o1o2o3_mass_c_iid_mag_iid_tilt_powerlaw_redshift_orientation_data.h5', ct1=True, lab='Abbott et. al. 2021b', col='tab:blue')
ax[i_2] = plot_mean_and_90CI(ax[i_2], xs, ct_pdfs, color='tab:red', label='Edelman et. al. 2022', bounds = False)
ax[i_2] = plot_mean_and_90CI(ax[i_2], mag_and_tilt_pdfs['cos_tilt_1'], tilt_tot, color='k', label='This Work')
high = np.percentile(ct_pdfs, 95, axis=0)
ax[i_2].set_xlabel(r'$\cos{\theta}$', fontsize=label_text_size)
ax[i_2].set_ylabel(r'$p(\cos{\theta})$', fontsize=label_text_size)
ax[i_2].set_xlim(xmin, xmax)
ax[i_2].set_title('GWTC-3: Total BBH Population Tilt Distribution')
# ax[i_2].legend(frameon=False, fontsize=legend_text_size, loc='upper left');
ax[i_2].grid(True, which="major", ls=":")
ax[i_2].set_ylim(0, 1.6)

mag_tot = mag_and_tilt_pdfs['peak_1_a1_pdfs'] + mag_and_tilt_pdfs['peak_2_a1_pdfs'] + mag_and_tilt_pdfs['continuum_a1_pdfs']
xmin, xmax = 0, 1
mags, a_pdfs = load_iid_mag_ppd()
for jj in range(len(a_pdfs)):
    a_pdfs[jj,:] /= np.trapz(a_pdfs[jj,:], mags)
ax[i_3] = plot_o3b_spinmag(ax[i_3],'o1o2o3_mass_c_iid_mag_iid_tilt_powerlaw_redshift_magnitude_data.h5', lab='Abbott et. al. 2021b', col='tab:blue')
ax[i_3] = plot_mean_and_90CI(ax[i_3], mags, a_pdfs, color='tab:red', label='Edelman et. al. 2022', bounds = False)
ax[i_3] = plot_mean_and_90CI(ax[i_3], mag_and_tilt_pdfs['a1'], mag_tot, color='k', label='This Work')
high = np.percentile(a_pdfs, 95, axis=0)
ax[i_3].legend(frameon=False, fontsize=legend_text_size);
ax[i_3].set_xlabel(r'$a$', fontsize=label_text_size)
ax[i_3].set_ylabel(r'$p(a)$', fontsize=label_text_size)
ax[i_3].set_xlim(xmin, xmax)
ax[i_3].set_title('GWTC-3: Total Population Spin Magnitude Distribution')
ax[i_3].set_ylim(0, 4.25)
ax[i_3].grid(True, which="major", ls=":")

# plt.title(f'GWTC-3: Spin Tilt Distribution', fontsize=title_text_size);
fig.tight_layout()
plt.savefig(paths.figures / 'spin_mag_distribution_plot.pdf', dpi=300);
plt.savefig(paths.figures / 'spin_mag_distribution_plot.jpeg', dpi=300);


