import paths
import numpy as np
import matplotlib.pyplot as plt
from gwinfernodata import GWInfernoData
import deepdish as dd
from utils import plot_mean_and_90CI, load_macro, load_gwinfernodata_ppds
import matplotlib
matplotlib.rcParams['text.usetex'] = True



base_label = load_macro('base')
comp_label = load_macro('comp')
first_label = load_macro('first')
contA_label = load_macro('contA')
contB_label = load_macro('contB')
msun = load_macro('msun')

figx, figy = 7, 5
legendfont = 12
fig, ax = plt.subplots(1,1,figsize = (figx, figy))

cyb_ppds = dd.io.load(paths.data / "bsplines_64m1_18q_iid18mag_iid18tilt_pl18z_ppds.h5")
g1_ppds = load_gwinfernodata_ppds().pdfs
g2_ppds = load_gwinfernodata_ppds(IP = False).pdfs
 
# g1_idata = GWInfernoData.from_netcdf(paths.data / 'updated/bspline_1logpeak_marginalized_fixtau_m1-s25-z1_msig15_qsig5_ssig5_zsig1_sigp3_NeffNobs_full_200ks.h5')
# sel = g1_ppds.coords['sel']

# print(np.trapz(g1_ppds['redshift_pdfs'].values/g1_idata.posterior['rate'].values[0][sel][:,np.newaxis], g1_ppds['redshift'].values))
# norm2 = np.trapz(g2_ppds['redshift_pdfs'].values, g1_ppds['redshift'].values)
# norm2 = norm2[:, np.newaxis]

ax = plot_mean_and_90CI(ax, g1_ppds['redshift'].values, g1_ppds['redshift_pdfs'].values, color='lightseagreen', label=base_label, bounds=True, mean = False, median = True, fill_alpha = 0.1)
ax = plot_mean_and_90CI(ax, g2_ppds['redshift'].values, g2_ppds['redshift_pdfs'].values, color='mediumslateblue', label=comp_label, bounds=True, mean = False, median = True, fill_alpha = 0.2)
ax = plot_mean_and_90CI(ax, cyb_ppds['zs'], cyb_ppds['Rofz'], color='tab:red', label='Edelman et. al. 2023', bounds=False, mean = False, median = True)
ax.set_xscale('log')
ax.set_yscale('log')
ax.set_xlabel(r"$z$", fontsize=18)
ax.set_ylabel(r"$\mathcal{R}(z)\,\mathrm{Gpc}^{-3}\mathrm{yr}^{-1}$", fontsize=18)
ax.set_ylim(5,1e3)
ax.set_xlim(g1_ppds['redshift'].values[0], 1.5)
ax.legend(frameon=False, fontsize=14, loc='upper left')
ax.grid(True, which="major", ls=":")
ax.tick_params(labelsize=14)

plt.savefig(paths.figures / 'redshift_distribution_plot.pdf', dpi=300, bbox_inches='tight');
plt.savefig(paths.figures / 'redshift_distribution_plot.png', dpi=300, bbox_inches='tight');