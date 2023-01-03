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

fig = plt.figure()
gs = gridspec.GridSpec(2,2,figure=fig,height_ratios=[1,3],width_ratios=[3,1], wspace=0, hspace=0)
ax = fig.add_subplot(gs[1,0])
axx = fig.add_subplot(gs[0,0])
axy = fig.add_subplot(gs[1,1])
for i in range(69):
    sns.kdeplot(x=idata.posterior['a_1_obs_event_{}'.format(i)].values[0], y=idata.posterior['cos_tilt_1_obs_event_{}'.format(i)].values[0], ax=ax, levels=1, color=cm[i], alpha=0.25)
    sns.kdeplot(x=idata.posterior['a_1_obs_event_{}'.format(i)].values[0], ax=axx, color=cm[i], alpha=0.5, lw=1.5)
    sns.kdeplot(y=idata.posterior['cos_tilt_1_obs_event_{}'.format(i)].values[0], ax=axy, color=cm[i], alpha=0.5, lw=1.5)

ax.set_xlabel(r'$a_1$', fontsize=18)
ax.set_ylabel(r'$cos(\theta_1)$', fontsize=18)
cbar = fig.colorbar(plt.cm.ScalarMappable(cmap='cool'), ax = axy, ticks=ticks, label = 'Category')
cbar.ax.set_yticklabels(categories) 
for axis in [axx,axy]:
    axis.set_xticks([])
    axis.set_yticks([])
    axis.set_xlabel("")
    axis.set_ylabel("")
plt.tight_layout()
plt.savefig(paths.figures / 'reweighed_catalog_a1_tilt1.pdf', dpi=300)
plt.savefig(paths.figures / 'reweighed_catalog_a1_tilt1.png', dpi=300)