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


mmin = 5.0
mmax = 100
figx, figy = 13, 9.5
legendfont = 12
fig = plt.figure(figsize = (figx, figy))
gs = gridspec.GridSpec(2,1,figure=fig)
gs1 = gridspec.GridSpecFromSubplotSpec(1, 3, subplot_spec=gs[1], width_ratios = [10,1,1], wspace = 0.05)
ax = fig.add_subplot(gs[0])
ax1 = fig.add_subplot(gs1[0,0])
axy = fig.add_subplot(gs1[0,1])
axy0 = fig.add_subplot(gs1[0,2])
fig.suptitle('Base and Composite Model Mass Distributions', fontsize = 18)
bspl_ms, bspl_mpdfs, bspl_qs, bspl_qpdfs = load_bsplinemass_ppd()
subpop_ppds = load_subpop_ppds(g1 = True, g1_fname = 'bspline_1logpeak_100000s_ppds.h5')
tot_subpops = subpop_ppds['peak_1_mass_pdfs'] + subpop_ppds['continuum_mass_pdfs']

# ax1.set_title('Model Group 1: Gaussian Peak + B-Spline')

idx20 = np.sum(subpop_ppds['mass_1'] < 20)
idx35 = np.sum(subpop_ppds['mass_1'] < 35)

fill = 0.2
# sns.kdeplot(y=subpop_ppds['continuum_mass_pdfs'][:,idx20], ax=axy1, color='tab:pink', lw=3, common_norm = True, log_scale = True, linestyle = (0, (1, 1)))
# sns.kdeplot(y=subpop_ppds['continuum_1_mass_pdfs'][:,idx20], ax=axy1, color='tab:purple', lw=3, common_norm = True, log_scale = True, linestyle = (0, (1, 1)))
# sns.kdeplot(y=subpop_ppds['continuum_mass_pdfs'][:,idx35], ax=axy2, color='tab:purple', lw=3, common_norm = True, log_scale = True, linestyle = (0, (1, 1)))
# sns.kdeplot(y=subpop_ppds['continuum_1_mass_pdfs'][:,idx35], ax=axy2, color='tab:pink', lw=3, common_norm = True, log_scale = True, linestyle = (0, (1, 1)))

# for ax, color in zip([axy1, axy2], ['yellowgreen', 'darkorange']):
#     ax.spines['bottom'].set_color(color)
#     ax.spines['top'].set_color(color)
#     ax.spines['right'].set_color(color)
#     ax.spines['left'].set_color(color)

# ax.axvline(20)
# ax.axvline(35)
ax = plot_mean_and_90CI(ax, bspl_ms, bspl_mpdfs, color='tab:red', label='Edelman et. al. 2022',bounds=False, mean = False, median = True)
ax = plot_mean_and_90CI(ax, subpop_ppds['mass_1'], tot_subpops, color ='black', label='Total, Base Model', bounds = False, mean = False, median = True)
ax = plot_mean_and_90CI(ax, subpop_ppds['mass_1'], subpop_ppds['peak_1_mass_pdfs'], color ='tab:cyan', label='Peak A', bounds = True, alpha = 0.75, line_style = '--', lw = 3, mean = False, median = True, fill_alpha = fill)
ax = plot_mean_and_90CI(ax, subpop_ppds['mass_1'], subpop_ppds['continuum_mass_pdfs'], color ='tab:pink', label='Continuum B', bounds = True, alpha = 0.75, line_style = (0, (1, 1)), lw = 3, mean = False, median = True, fill_alpha = fill)
ax.legend(frameon=False, fontsize=legendfont, loc = 'upper right');
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

# axy1.set_xticks([])
# axy1.set_xlabel("")
# axy1.set_ylim(1e-8, 1e0)
# axy1.yaxis.tick_right()
# axy1.yaxis.set_label_position("right")
# axy1.tick_params(labelsize=14)
# # axy.set_yscale('log')
# axy1.set_ylabel(r'$p(m_1 = 20) \,\,[M_\odot^{-1}]$', fontsize=18)


subpop_ppds = load_subpop_ppds(g2 = True, g2_fname = 'bspline_1logpeak_samespin_100000s_2chains.h5')
tot_subpops = subpop_ppds['peak_1_mass_pdfs'] + subpop_ppds['continuum_mass_pdfs'] + subpop_ppds['continuum_1_mass_pdfs']

# axins = ax1.inset_axes([0.8, 0.8, 0.4, 0.4])
sns.kdeplot(y=subpop_ppds['continuum_mass_pdfs'][:,idx20], ax=axy, color='tab:pink', lw=0, common_norm = True, log_scale = True, fill = True, alpha = 0.5)
sns.kdeplot(y=subpop_ppds['continuum_1_mass_pdfs'][:,idx20], ax=axy, color='tab:purple', lw=0, common_norm = True, log_scale = True, fill = True, alpha = 0.5)
sns.kdeplot(y=subpop_ppds['continuum_mass_pdfs'][:,idx35], ax=axy0, color='tab:pink', lw=0, common_norm = True, log_scale = True, fill = True, alpha = 0.5)
sns.kdeplot(y=subpop_ppds['continuum_1_mass_pdfs'][:,idx35], ax=axy0, color='tab:purple', lw=0, common_norm = True, log_scale = True, fill = True, alpha = 0.5)

for axs, color in zip([axy, axy0], ['deepskyblue', 'royalblue']):
    axs.spines['bottom'].set_color(color)
    axs.spines['top'].set_color(color)
    axs.spines['right'].set_color(color)
    axs.spines['left'].set_color(color)
    axs.grid(True, which="major", ls=":", axis = 'y')
    axs.set_xlabel("")
    axs.set_ylim(1e-5, 1e0)
    axs.set_xlim(0, 0.65)
    # axs.minorticks_off()

axy.text(0.05, 0.5, r'$m_1 = 20$', fontsize = 12, color = 'deepskyblue')
axy0.text(0.05, 0.5, r'$m_1 = 35$', fontsize = 12, color = 'royalblue')
axy.tick_params(which = 'major', bottom = False, left = False, labelbottom = False, labelleft = False)
axy.minorticks_off()
axy0.tick_params(which = 'both', bottom = False, left = False, labelbottom = False, labelleft = False, right = True, labelright = True, color = 'royalblue', labelsize = 14)
# axy0.yaxis.tick_right()
# axy0.yaxis.set_label_position("right")
ax1.axvline(20, color = 'deepskyblue')
ax1.axvline(35, color = 'royalblue')
ax1 = plot_mean_and_90CI(ax1, bspl_ms, bspl_mpdfs, color='tab:red', label='Edelman et. al. 2022',bounds=False, mean = False, median = True)
ax1 = plot_mean_and_90CI(ax1, subpop_ppds['mass_1'], tot_subpops, color ='black', label='Total, Composite Model', bounds = False, mean = False, median = True)
ax1 = plot_mean_and_90CI(ax1, subpop_ppds['mass_1'], subpop_ppds['peak_1_mass_pdfs'], color ='tab:cyan', label='Peak A', bounds = True, alpha = 0.75, line_style = '--', lw = 3, mean = False, median = True, fill_alpha = fill)
ax1 = plot_mean_and_90CI(ax1, subpop_ppds['mass_1'], subpop_ppds['continuum_1_mass_pdfs'], color ='tab:purple', label='Continuum A', bounds = True, alpha = 0.75, line_style = '--', lw = 3, mean = False, median = True, fill_alpha = fill)
ax1 = plot_mean_and_90CI(ax1, subpop_ppds['mass_1'], subpop_ppds['continuum_mass_pdfs'], color ='tab:pink', label='Continuum B', bounds = True, alpha = 0.75, line_style = (0, (1, 1)), lw = 3, mean = False, median = True, fill_alpha = fill)
ax1.legend(frameon=False, fontsize=legendfont, loc = 'upper right');
ax1.set_xlabel(r'$m_1 \,\,[M_\odot]$', fontsize=18)
ax1.set_ylabel(r'$p(m_1) \,\,[M_\odot^{-1}]$', fontsize=18)
ax1.grid(True, which="major", ls=":")
ax1.tick_params(labelsize=14)
ax1.set_yscale('log')
ax1.set_xscale('log')
ax1.set_ylim(1e-5, 1e0)
logticks = np.array([6,8,10,20,40,70,100])
ax1.set_xticks(logticks)
# mark_inset(ax1, axins, loc1=3, loc2=4, fc="white", ec="0.5")
# ax1.set_xlim(np.log(5), np.log(100))
ax1.get_xaxis().set_major_formatter(ScalarFormatter())
ax1.grid(True, which="major", ls=":")
ax1.set_xlim(mmin+0.5, mmax)



# axy.yaxis.tick_right()
# axy.yaxis.set_label_position("right")
# axy.tick_params(labelsize=14)
# # axy.set_yscale('log')
# axy.set_ylabel(r'$p(m_1 = 20) \,\,[M_\odot^{-1}]$', fontsize=18)
# ax.set_title('Model Group 2: Gaussian Peak + 2 B-Splines')
# plt.title(f'GWTC-3: BBH Primary Mass Distribution', fontsize=18);
# fig.tight_layout()
plt.savefig(paths.figures / 'mass_distribution_g1_plot.pdf', dpi=300);
plt.savefig(paths.figures / 'mass_distribution_g1_plot.png', dpi=300);

