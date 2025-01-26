#!/usr/bin/env python

import paths
import numpy as np
import matplotlib.pyplot as plt
from utils import plot_mean_and_90CI, load_bsplinemass_ppd, plot_o3b_res, load_subpop_ppds, load_macro, load_gwinfernodata_ppds
from matplotlib.ticker import ScalarFormatter
import matplotlib.gridspec as gridspec
import seaborn as sns
from mpl_toolkits.axes_grid1.inset_locator import mark_inset
from gwinfernodata import GWInfernoData
import matplotlib
matplotlib.rcParams['text.usetex'] = True

import os 
print(os.environ["CONDA_PREFIX"])

base_label = load_macro('base')
comp_label = load_macro('comp')
popA_label = load_macro('popA')
popB_label = load_macro('popB')
first_label = load_macro('first')
contA_label = load_macro('contA')
contB_label = load_macro('contB')
msun = load_macro('msun')

mmin = 5.0
mmax = 100
figx, figy = 13, 9.5
legendfont = 12
fig = plt.figure(figsize = (figx, figy))
gs = gridspec.GridSpec(2,1,figure=fig)
# gs1 = gridspec.GridSpecFromSubplotSpec(1, 3, subplot_spec=gs[1], width_ratios = [10,1,1], wspace = 0.05)
ax = fig.add_subplot(gs[0])
ax1 = fig.add_subplot(gs[1])#fig.add_subplot(gs1[0,0])
# axy = fig.add_subplot(gs1[0,1])
# axy0 = fig.add_subplot(gs1[0,2])
ax.set_title(r'{} and {} Mass Distributions'.format(base_label, comp_label), fontsize = 18)
bspl_ms, bspl_mpdfs, bspl_qs, bspl_qpdfs = load_bsplinemass_ppd()
subpop_ppds = load_gwinfernodata_ppds().pdfs
tot_subpops = subpop_ppds['peak_1_mass_pdfs'].values + subpop_ppds['continuum_mass_pdfs'].values

idx20 = np.sum(subpop_ppds['mass_1'].values < 20)
idx35 = np.sum(subpop_ppds['mass_1'] < 35)

fill = 0.2
ax = plot_mean_and_90CI(ax, bspl_ms, bspl_mpdfs, color='tab:red', label='Edelman et. al. 2023',bounds=False, mean = False, median = True)
ax = plot_mean_and_90CI(ax, subpop_ppds['mass_1'], tot_subpops, color ='black', label=base_label, bounds = False, mean = False, median = True)
ax = plot_mean_and_90CI(ax, subpop_ppds['mass_1'], subpop_ppds['peak_1_mass_pdfs'], color ='tab:cyan', label=rf'{popA_label}{first_label}', bounds = True, alpha = 0.75, line_style = '--', lw = 3, mean = False, median = True, fill_alpha = fill)
ax = plot_mean_and_90CI(ax, subpop_ppds['mass_1'], subpop_ppds['continuum_mass_pdfs'], color ='tab:pink', label=rf'{popB_label}{contB_label}', bounds = True, alpha = 0.75, line_style = (0, (1, 1)), lw = 3, mean = False, median = True, fill_alpha = fill)
ax.legend(frameon=False, fontsize=legendfont, loc = 'upper right');
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
ax.set_xlabel(r'$m_1 \,\,[M_\odot]$', fontsize=18)
ax.set_ylabel(r'$p(m_1) \,\,[M_\odot^{-1}]$', fontsize=18)


subpop_ppds = load_gwinfernodata_ppds(IP = False).pdfs
tot_subpops = subpop_ppds['peak_1_mass_pdfs'].values  + subpop_ppds['continuum_mass_pdfs'].values + subpop_ppds['continuum_1_mass_pdfs'].values

# sns.kdeplot(y=subpop_ppds['continuum_mass_pdfs'][:,idx20], ax=axy, color='tab:pink', lw=0, common_norm = True, log_scale = True, fill = True, alpha = 0.5)
# sns.kdeplot(y=subpop_ppds['continuum_1_mass_pdfs'][:,idx20], ax=axy, color='tab:purple', lw=0, common_norm = True, log_scale = True, fill = True, alpha = 0.5)
# sns.kdeplot(y=subpop_ppds['continuum_mass_pdfs'][:,idx35], ax=axy0, color='tab:pink', lw=0, common_norm = True, log_scale = True, fill = True, alpha = 0.5)
# sns.kdeplot(y=subpop_ppds['continuum_1_mass_pdfs'][:,idx35], ax=axy0, color='tab:purple', lw=0, common_norm = True, log_scale = True, fill = True, alpha = 0.5)

# for axs, color in zip([axy, axy0], ['deepskyblue', 'royalblue']):
#     axs.spines['bottom'].set_color(color)
#     axs.spines['top'].set_color(color)
#     axs.spines['right'].set_color(color)
#     axs.spines['left'].set_color(color)
#     axs.grid(True, which="major", ls=":", axis = 'y')
#     axs.set_xlabel("")
#     axs.set_ylim(1e-5, 1e0)
#     axs.set_xlim(0, 0.65)

# axy.text(0.05, 0.5, r'$m_1 = 20$', fontsize = 12, color = 'deepskyblue')
# axy0.text(0.05, 0.5, r'$m_1 = 35$', fontsize = 12, color = 'royalblue')
# axy.tick_params(which = 'major', bottom = False, left = False, labelbottom = False, labelleft = False)
# axy.minorticks_off()
# # axy0.tick_params(which = 'both', bottom = False, left = False, labelbottom = False, labelleft = False, right = True, labelright = True, color = 'royalblue', labelsize = 14)
# ax1.axvline(20, color = 'deepskyblue')
# ax1.axvline(35, color = 'royalblue')
ax1 = plot_mean_and_90CI(ax1, bspl_ms, bspl_mpdfs, color='tab:red', label='Edelman et. al. 2023',bounds=False, mean = False, median = True)
ax1 = plot_mean_and_90CI(ax1, subpop_ppds['mass_1'], tot_subpops, color ='black', label=comp_label, bounds = False, mean = False, median = True)
ax1 = plot_mean_and_90CI(ax1, subpop_ppds['mass_1'], subpop_ppds['peak_1_mass_pdfs'], color ='tab:cyan', label=rf'{popA_label}{first_label}', bounds = True, alpha = 0.75, line_style = '--', lw = 3, mean = False, median = True, fill_alpha = fill)
ax1 = plot_mean_and_90CI(ax1, subpop_ppds['mass_1'], subpop_ppds['continuum_1_mass_pdfs'], color ='tab:purple', label=rf'{popA_label}{contA_label}', bounds = True, alpha = 0.75, line_style = '--', lw = 3, mean = False, median = True, fill_alpha = fill)
ax1 = plot_mean_and_90CI(ax1, subpop_ppds['mass_1'], subpop_ppds['continuum_mass_pdfs'], color ='tab:pink', label=rf'{popB_label}{contB_label}', bounds = True, alpha = 0.75, line_style = (0, (1, 1)), lw = 3, mean = False, median = True, fill_alpha = fill)
ax1.legend(frameon=False, fontsize=legendfont, loc = 'upper right');
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
ax1.set_xlabel(r'$m_1 \,\,[M_\odot]$', fontsize=18)
ax1.set_ylabel(r'$p(m_1) \,\,[M_\odot^{-1}]$', fontsize=18)

plt.savefig(paths.figures / 'mass_distribution_g1_plot.pdf', dpi=300, bbox_inches='tight');
plt.savefig(paths.figures / 'mass_distribution_g1_plot.png', dpi=300, bbox_inches='tight');

