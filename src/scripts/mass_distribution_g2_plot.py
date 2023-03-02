#!/usr/bin/env python

import paths
import numpy as np
import matplotlib.pyplot as plt
from utils import plot_mean_and_90CI, load_bsplinemass_ppd, plot_o3b_res, load_subpop_ppds
from matplotlib.ticker import ScalarFormatter
import matplotlib.gridspec as gridspec
import seaborn as sns

mmin = 5.0
mmax = 100
figx, figy = 13, 9.5
legendfont = 12
fig = plt.figure(figsize = (figx, figy))
gs = gridspec.GridSpec(2,2,figure=fig, width_ratios=[10,1], wspace=0.05, hspace = 0.25)
ax1 = fig.add_subplot(gs[0,0])
ax = fig.add_subplot(gs[1,0])
axy = fig.add_subplot(gs[1,1])
axy1 = fig.add_subplot(gs[0,1])
fig.suptitle('Composite Model: Mass Distributions', fontsize = 18)
bspl_ms, bspl_mpdfs, bspl_qs, bspl_qpdfs = load_bsplinemass_ppd()
subpop_ppds = load_subpop_ppds(g2 = True, g2_fname = 'bspline_1logpeak_samespin_100000s_2chains.h5')
tot_subpops = subpop_ppds['peak_1_mass_pdfs'] + subpop_ppds['continuum_mass_pdfs'] + subpop_ppds['continuum_1_mass_pdfs']
fill = 0.2
sel_1 = subpop_ppds['continuum_mass_pdfs'][:,127] < 1e-3
num_1 = np.mean(sel_1)*100

# ax1 = plot_o3b_res(ax1,'o1o2o3_mass_c_iid_mag_iid_tilt_powerlaw_redshift_mass_data.h5', lab='Abbott et. al. 2021b', col='tab:blue', bounds=False, mean = False, median = True)
# ax1 = plot_mean_and_90CI(ax1, bspl_ms, bspl_mpdfs, color='tab:red', label='Edelman et. al. 2022',bounds=False, mean = False, median = True)
# ax1 = plot_mean_and_90CI(ax1, subpop_ppds['mass_1'], tot_subpops, color ='black', label='Total, this work', bounds = True, mean = False, median = True)
# ax1.legend(frameon=False, fontsize=14);
# ax1.set_xlabel(r'$m_1 \,\,[M_\odot]$', fontsize=18)
# ax1.set_ylabel(r'$p(m_1) \,\,[M_\odot^{-1}]$', fontsize=18)
# ax1.grid(True, which="major", ls=":")
# ax1.tick_params(labelsize=14)
# ax1.set_yscale('log')
# ax1.set_xscale('log')
# ax1.set_ylim(1e-5, 1e0)
# logticks = np.array([6,8,10,20,40,70,100])
# ax1.set_xticks(logticks)
# # ax1.set_xlim(np.log(5), np.log(100))
# ax1.get_xaxis().set_major_formatter(ScalarFormatter())
# ax1.grid(True, which="major", ls=":")
# ax1.set_xlim(mmin+0.5, mmax)
kde1 = sns.kdeplot(y=subpop_ppds['continuum_mass_pdfs'][:,127], ax=axy1, color='tab:pink', lw=0, common_norm = True, log_scale = True)
points1 = kde1.get_lines()[0].get_data()
kde_sel1 = points1[1] <= 1e-3
axy1.fill_betweenx(points1[1][kde_sel1], points1[0][kde_sel1], x2 = 0, color='tab:pink', alpha = 0.5)
axy1.fill_betweenx(points1[1][~kde_sel1], points1[0][~kde_sel1], x2 = 0, color='gray', alpha = 0.5)
ax1 = plot_mean_and_90CI(ax1, subpop_ppds['mass_1'], tot_subpops[sel_1], color ='black', label='Total', bounds = False, mean = False, median = True)
ax1 = plot_mean_and_90CI(ax1, subpop_ppds['mass_1'], subpop_ppds['peak_1_mass_pdfs'][sel_1], color ='tab:cyan', label='Peak A', bounds = True, alpha = 0.75, line_style = '--', lw = 3, mean = False, median = True, fill_alpha = fill)
ax1 = plot_mean_and_90CI(ax1, subpop_ppds['mass_1'], subpop_ppds['continuum_1_mass_pdfs'][sel_1], color ='tab:purple', label='Continuum A', bounds = True, alpha = 0.75, line_style = '--', lw = 3, mean = False, median = True, fill_alpha = fill)
ax1 = plot_mean_and_90CI(ax1, subpop_ppds['mass_1'], subpop_ppds['continuum_mass_pdfs'][sel_1], color ='tab:pink', label='Continuum B', bounds = True, alpha = 0.75, line_style = (0, (1, 1)), lw = 3, mean = False, median = True, fill_alpha = fill)
ax1.legend(frameon=False, fontsize=legendfont);
ax1.set_xlabel(r'$m_1 \,\,[M_\odot]$', fontsize=18)
ax1.set_ylabel(r'$p(m_1) \,\,[M_\odot^{-1}]$', fontsize=18)
ax1.text(55, 0.01, '{:.1f}% of samples'.format(num_1), fontsize = 12)
ax1.grid(True, which="major", ls=":")
ax1.tick_params(labelsize=14)
ax1.set_yscale('log')
ax1.set_xscale('log')
ax1.set_ylim(1e-5, 1e0)
logticks = np.array([6,8,10,20,40,70,100])
ax1.set_xticks(logticks)
ax1.get_xaxis().set_major_formatter(ScalarFormatter())
ax1.grid(True, which="major", ls=":")
ax1.set_xlim(mmin+0.5, mmax)
ax1.axvline(20, color = 'k', lw = 1)

axy1.set_xticks([])
axy1.set_xlabel("")
axy1.set_ylim(1e-5, 1e0)
axy1.yaxis.tick_right()
axy1.yaxis.set_label_position("right")
axy1.tick_params(labelsize=14)
axy1.grid(True, which="major", ls=":")
axy1.text(0.05, 0.5, r'$m_1 = 20$', fontsize = 12)
# axy.set_yscale('log')
# axy1.set_ylabel(r'$p(m_1 = 20) \,\,[M_\odot^{-1}]$', fontsize=18)


sel_2 = subpop_ppds['continuum_mass_pdfs'][:,127] > 1e-3
num_2 = np.mean(sel_2)*100

kde = sns.kdeplot(y=subpop_ppds['continuum_mass_pdfs'][:,127], ax=axy, color='tab:pink', lw=0, common_norm = False, log_scale = True)
points = kde.get_lines()[0].get_data()
kde_sel = points[1] >= 1e-3
axy.fill_betweenx(points[1][kde_sel], points[0][kde_sel], x2 = 0, color='tab:pink', alpha = 0.5)
axy.fill_betweenx(points[1][~kde_sel], points[0][~kde_sel], x2 = 0, color='tab:gray', alpha = 0.5)
ax = plot_mean_and_90CI(ax, subpop_ppds['mass_1'], tot_subpops[sel_2], color ='black', label='Total', bounds = False, mean = False, median = True)
ax = plot_mean_and_90CI(ax, subpop_ppds['mass_1'], subpop_ppds['peak_1_mass_pdfs'][sel_2], color ='tab:cyan', label='Peak A', bounds = True, alpha = 0.75, line_style = '--', lw = 3, mean = False, median = True, fill_alpha = fill)
ax = plot_mean_and_90CI(ax, subpop_ppds['mass_1'], subpop_ppds['continuum_1_mass_pdfs'][sel_2], color ='tab:purple', label='Continuum A', bounds = True, alpha = 0.75, line_style = '--', lw = 3, mean = False, median = True, fill_alpha = fill)
ax = plot_mean_and_90CI(ax, subpop_ppds['mass_1'], subpop_ppds['continuum_mass_pdfs'][sel_2], color ='tab:pink', label='Continuum B', bounds = True, alpha = 0.75, line_style = (0, (1, 1)), lw = 3, mean = False, median = True, fill_alpha = fill)
# ax.legend(frameon=False, fontsize=14);
ax.set_xlabel(r'$m_1 \,\,[M_\odot]$', fontsize=18)
ax.set_ylabel(r'$p(m_1) \,\,[M_\odot^{-1}]$', fontsize=18)
ax.grid(True, which="major", ls=":")
ax.tick_params(labelsize=14)
ax.text(55, 0.3, '{:.1f}% of samples'.format(num_2), fontsize = 12)
ax.set_yscale('log')
ax.set_xscale('log')
ax.set_ylim(1e-5, 1e0)
logticks = np.array([6,8,10,20,40,70,100])
ax.set_xticks(logticks)
ax.get_xaxis().set_major_formatter(ScalarFormatter())
ax.grid(True, which="major", ls=":")
ax.set_xlim(mmin+0.5, mmax)
ax.axvline(20, color = 'k', lw = 1)

axy.set_xticks([])
axy.set_xlabel("")
axy.set_ylim(1e-5, 1e0)
axy.yaxis.tick_right()
axy.yaxis.set_label_position("right")
axy.tick_params(labelsize=14)
# axy.set_yscale('log')
axy.grid(True, which="major", ls=":")
axy.text(0.05, 0.5, r'$m_1 = 20$', fontsize = 12)

# ax.set_title('Model Group 2: Gaussian Peak + 2 B-Splines')
# plt.title(f'GWTC-3: BBH Primary Mass Distribution', fontsize=18);
# fig.tight_layout()
plt.savefig(paths.figures / 'mass_distribution_g2_plot.pdf', dpi=300);
plt.savefig(paths.figures / 'mass_distribution_g2_plot.png', dpi=300);

