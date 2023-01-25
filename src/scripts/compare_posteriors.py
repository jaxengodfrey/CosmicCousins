import paths
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
import arviz as az
import seaborn as sns
import matplotlib.gridspec as gridspec
from matplotlib.ticker import ScalarFormatter
from utils import load_trace
import deepdish as dd

idata = load_trace()
bspline_ps = dd.io.load(paths.data / 'bsplines_64m1_18q_iid18mag_iid18tilt_pl18z_posterior_samples.h5')


n_categories = 3
categories = ['Low-Mass\nPeak', 'High-Mass\nPeak', 'Continuum']
probs = idata.posterior['Qs'].mean(axis = 1).values[0] / (n_categories - 1)
ticks = np.linspace(0,1, n_categories)
cm = plt.cm.cool(probs)


fig, ax = plt.subplots(nrows = 2, ncols = 1, figsize = (15,8))

for i in range(69):
    # if (probs[i] < 0.6) & (probs[i] > 0.5):
    if (np.mean(idata.posterior['mass_1_obs_event_{}'.format(i)].values[0]) < 13):
        sns.kdeplot(x=idata.posterior['mass_1_obs_event_{}'.format(i)].values[0], ax=ax[0], color=cm[i], alpha=0.5, lw=1.5)
        sns.kdeplot(x=bspline_ps['mass_1_obs_event_{}'.format(i)], ax=ax[0], color='k', alpha=0.5, lw=1.5)
        sns.kdeplot(x=idata.posterior['cos_tilt_1_obs_event_{}'.format(i)].values[0], ax=ax[1], color=cm[i], alpha=0.5, lw=1.5)
        sns.kdeplot(x=bspline_ps['cos_tilt_1_obs_event_{}'.format(i)], ax=ax[1], color='k', alpha=0.5, lw=1.5)

ax[0].set_xlabel(r'$m_1$', fontsize=18)
ax[1].set_xlabel(r'$a_1$', fontsize=18)
ax[0].set_xscale('log')
logticks = np.array([5,8,10,20,40,70,100,150])
ax[0].set_xticks(logticks)
ax[0].get_xaxis().set_major_formatter(ScalarFormatter())
ax[1].tick_params(labelsize=14)
ax[0].tick_params(labelsize=14)
ax[0].set_ylim(0,0.4)
ax[1].set_ylim(0,5)
ax[0].set_xlim(5,100)
ax[1].set_xlim(-1,1)
plt.tight_layout()
cbar = fig.colorbar(plt.cm.ScalarMappable(cmap='cool'), ax = ax.ravel().tolist(), ticks=ticks, label = 'Category')
cbar.ax.set_yticklabels(categories) 
plt.show()
# plt.savefig(paths.figures / 'reweighed_catalog_a1_tilt1.pdf', dpi=300)
# plt.savefig(paths.figures / 'reweighed_catalog_a1_tilt1.png', dpi=300)

fig, ax = plt.subplots(nrows = 2, ncols = 1, figsize = (15,8))

for i in range(69):
    if (np.mean(idata.posterior['mass_1_obs_event_{}'.format(i)].values[0]) > 13) & (np.mean(idata.posterior['mass_1_obs_event_{}'.format(i)].values[0]) < 25):
        sns.kdeplot(x=idata.posterior['mass_1_obs_event_{}'.format(i)].values[0], ax=ax[0], color=cm[i], alpha=0.5, lw=1.5)
        sns.kdeplot(x=bspline_ps['mass_1_obs_event_{}'.format(i)], ax=ax[0], color='k', alpha=0.5, lw=1.5)
        sns.kdeplot(x=idata.posterior['cos_tilt_1_obs_event_{}'.format(i)].values[0], ax=ax[1], color=cm[i], alpha=0.5, lw=1.5)
        sns.kdeplot(x=bspline_ps['cos_tilt_1_obs_event_{}'.format(i)], ax=ax[1], color='k', alpha=0.5, lw=1.5)

ax[0].set_xlabel(r'$m_1$', fontsize=18)
ax[0].set_xscale('log')
ax[1].set_xlabel(r'$a_1$', fontsize=18)
logticks = np.array([5,8,10,20,40,70,100,150])
ax[0].set_xticks(logticks)
ax[0].get_xaxis().set_major_formatter(ScalarFormatter())
ax[1].tick_params(labelsize=14)
ax[0].tick_params(labelsize=14)
ax[0].set_xlim(5,100)
ax[1].set_xlim(-1,1)
ax[0].set_ylim(0,0.4)
ax[1].set_ylim(0,5)
plt.tight_layout()
cbar = fig.colorbar(plt.cm.ScalarMappable(cmap='cool'), ax = ax.ravel().tolist(), ticks=ticks, label = 'Category')
cbar.ax.set_yticklabels(categories) 
plt.show()

fig, ax = plt.subplots(nrows = 2, ncols = 1, figsize = (15,8))

for i in range(69):
    if (np.mean(idata.posterior['mass_1_obs_event_{}'.format(i)].values[0]) > 25) & (np.mean(idata.posterior['mass_1_obs_event_{}'.format(i)].values[0]) < 40):
        sns.kdeplot(x=idata.posterior['mass_1_obs_event_{}'.format(i)].values[0], ax=ax[0], color=cm[i], alpha=0.5, lw=1.5)
        sns.kdeplot(x=bspline_ps['mass_1_obs_event_{}'.format(i)], ax=ax[0], color='k', alpha=0.5, lw=1.5)
        sns.kdeplot(x=idata.posterior['cos_tilt_1_obs_event_{}'.format(i)].values[0], ax=ax[1], color=cm[i], alpha=0.5, lw=1.5)
        sns.kdeplot(x=bspline_ps['cos_tilt_1_obs_event_{}'.format(i)], ax=ax[1], color='k', alpha=0.5, lw=1.5)

ax[0].set_xlabel(r'$m_1$', fontsize=18)
ax[1].set_xlabel(r'$a_1$', fontsize=18)
ax[0].set_xscale('log')
logticks = np.array([5,8,10,20,40,70,100,150])
ax[0].set_xticks(logticks)
ax[0].get_xaxis().set_major_formatter(ScalarFormatter())
ax[1].tick_params(labelsize=14)
ax[0].tick_params(labelsize=14)
ax[0].set_xlim(5,100)
ax[1].set_xlim(-1,1)
ax[0].set_ylim(0,0.4)
ax[1].set_ylim(0,5)
plt.tight_layout()
cbar = fig.colorbar(plt.cm.ScalarMappable(cmap='cool'), ax = ax.ravel().tolist(), ticks=ticks, label = 'Category')
cbar.ax.set_yticklabels(categories) 
plt.show()

fig, ax = plt.subplots(nrows = 2, ncols = 1, figsize = (15,8))

for i in range(69):
    if probs[i] > 0.6:
        sns.kdeplot(x=idata.posterior['mass_1_obs_event_{}'.format(i)].values[0], ax=ax[0], color=cm[i], alpha=0.5, lw=1.5)
        sns.kdeplot(x=bspline_ps['mass_1_obs_event_{}'.format(i)], ax=ax[0], color='k', alpha=0.5, lw=1.5)
        sns.kdeplot(x=idata.posterior['cos_tilt_1_obs_event_{}'.format(i)].values[0], ax=ax[1], color=cm[i], alpha=0.5, lw=1.5)
        sns.kdeplot(x=bspline_ps['cos_tilt_1_obs_event_{}'.format(i)], ax=ax[1], color='k', alpha=0.5, lw=1.5)

ax[0].set_xlabel(r'$m_1$', fontsize=18)
ax[1].set_xlabel(r'$a_1$', fontsize=18)
ax[0].set_xscale('log')
logticks = np.array([5,8,10,20,40,70,100,150])
ax[0].set_xticks(logticks)
ax[0].get_xaxis().set_major_formatter(ScalarFormatter())
ax[1].tick_params(labelsize=14)
ax[0].tick_params(labelsize=14)
ax[0].set_xlim(5,100)
ax[1].set_xlim(-1,1)
ax[0].set_ylim(0,0.4)
ax[1].set_ylim(0,5)
plt.tight_layout()
cbar = fig.colorbar(plt.cm.ScalarMappable(cmap='cool'), ax = ax.ravel().tolist(), ticks=ticks, label = 'Category')
cbar.ax.set_yticklabels(categories) 
plt.show()