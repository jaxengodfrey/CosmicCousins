import paths
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
import arviz as az
import seaborn as sns
import matplotlib.gridspec as gridspec
from matplotlib.ticker import ScalarFormatter
from utils import load_trace

idata = load_trace()

n_categories = 3
categories = ['Low-Mass\nPeak', 'High-Mass\nPeak', 'Continuum']
probs = idata.posterior['Qs'].mean(axis = 1).values[0] / (n_categories - 1)
ticks = np.linspace(0,1, n_categories)
cm = plt.cm.cool(probs)

fig, ax = plt.subplots(nrows = 2, ncols = 1, figsize = (8,5))
for i in range(69):
    sns.kdeplot(x=idata.posterior['a_1_obs_event_{}'.format(i)].values[0], ax=ax[0], color=cm[i], alpha=0.5, lw=1.5)
    sns.kdeplot(x=idata.posterior['cos_tilt_1_obs_event_{}'.format(i)].values[0], ax=ax[1], color=cm[i], alpha=0.5, lw=1.5)

ax[0].set_xlabel(r'$a_1$', fontsize=18)
ax[1].set_xlabel(r'$cos(\theta_1)$', fontsize=18)
ax[0].set_ylabel(r'$p(a_1)$', fontsize=18)
ax[1].set_ylabel(r'$p(cos\theta_1)$', fontsize=18)
ax[0].tick_params(labelsize=14)
ax[1].tick_params(labelsize=14)
ax[0].set_xlim(0,1)
ax[1].set_xlim(-1,1)
plt.tight_layout()
cbar = fig.colorbar(plt.cm.ScalarMappable(cmap='cool'), ax = ax.ravel().tolist(), ticks=ticks, label = 'Category')
cbar.ax.set_yticklabels(categories) 
plt.savefig(paths.figures / 'reweighed_catalog_a1_tilt1.pdf', dpi=300)
plt.savefig(paths.figures / 'reweighed_catalog_a1_tilt1.png', dpi=300)