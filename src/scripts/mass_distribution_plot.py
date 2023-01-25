#!/usr/bin/env python

import paths
import numpy as np
import matplotlib.pyplot as plt
from utils import plot_mean_and_90CI, load_bsplinemass_ppd, plot_o3b_res, load_subpop_ppds
from matplotlib.ticker import ScalarFormatter

mmin = 5.0
mmax = 100
figx, figy = 14, 5
fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(figx,figy))
bspl_ms, bspl_mpdfs, bspl_qs, bspl_qpdfs = load_bsplinemass_ppd()
subpop_ppds = load_subpop_ppds()
tot_subpops = subpop_ppds['peak_1_mass_pdfs'] + subpop_ppds['peak_2_mass_pdfs'] + subpop_ppds['continuum_mass_pdfs']
ax = plot_o3b_res(ax,'o1o2o3_mass_c_iid_mag_iid_tilt_powerlaw_redshift_mass_data.h5', lab='Abbott et. al. 2021b', col='tab:blue', bounds=False)
ax = plot_mean_and_90CI(ax, bspl_ms, bspl_mpdfs, color='tab:red', label='Edelman et. al. 2022',bounds=False)
ax = plot_mean_and_90CI(ax, subpop_ppds['mass_1'], tot_subpops, color ='black', label='Total', bounds = True)
ax = plot_mean_and_90CI(ax, subpop_ppds['mass_1'], subpop_ppds['peak_1_mass_pdfs'], color ='tab:cyan', label='Low-Mass Peak', bounds = True, alpha = 0.75, line_style = '--', lw = 3)
ax = plot_mean_and_90CI(ax, subpop_ppds['mass_1'], subpop_ppds['peak_2_mass_pdfs'], color ='tab:purple', label='High-Mass Peak', bounds = True, alpha = 0.75, line_style = '--', lw = 3)
ax = plot_mean_and_90CI(ax, subpop_ppds['mass_1'], subpop_ppds['continuum_mass_pdfs'], color ='tab:pink', label='Continuum', bounds = True, alpha = 0.75, line_style = '--', lw = 3)
ax.legend(frameon=False, fontsize=14);
ax.set_xlabel(r'$m_1 \,\,[M_\odot]$', fontsize=18)
ax.set_ylabel(r'$p(m_1) \,\,[M_\odot^{-1}]$', fontsize=18)
ax.grid(True, which="major", ls=":")
ax.tick_params(labelsize=14)
ax.set_yscale('log')
ax.set_xscale('log')
ax.set_ylim(1e-5, 1e0)
logticks = np.array([6,8,10,20,40,70,100])
ax.set_xticks(logticks)
ax.get_xaxis().set_major_formatter(ScalarFormatter())
ax.grid(True, which="major", ls=":")
ax.set_xlim(mmin+0.5, mmax)
plt.title(f'GWTC-3: BBH Primary Mass Distribution', fontsize=18);
fig.tight_layout()
plt.savefig(paths.figures / 'mass_distribution_plot.pdf', dpi=300);
plt.savefig(paths.figures / 'mass_distribution_plot.png', dpi=300);