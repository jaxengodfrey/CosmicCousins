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
    sns.kdeplot(x=idata.posterior['mass_1_obs_event_{}'.format(i)].values[0], y=idata.posterior['mass_ratio_obs_event_{}'.format(i)].values[0]*idata.posterior['mass_1_obs_event_{}'.format(i)].values[0], ax=ax, levels=1, color=cm[i], alpha=0.25)
    sns.kdeplot(x=idata.posterior['mass_1_obs_event_{}'.format(i)].values[0], ax=axx, color=cm[i], alpha=0.5, lw=1.5)
    sns.kdeplot(y=idata.posterior['mass_ratio_obs_event_{}'.format(i)].values[0]*idata.posterior['mass_1_obs_event_{}'.format(i)].values[0], ax=axy, color=cm[i], alpha=0.5, lw=1.5)
ax.fill_between([1,10,50,100,200], [1,10,50,100,200], [200,200,200,200,200], color='k', alpha=1, zorder=10)
# ax.legend(fontsize=12, loc='upper left').set_zorder(11)
ax.set_xscale('log')
ax.set_yscale('log')
axx.set_xscale('log')
axy.set_yscale('log')
ax.set_xlabel(r'$m_1\,[M_\odot]$', fontsize=18)
ax.set_ylabel(r'$m_2\,[M_\odot]$', fontsize=18)
logticks = np.array([5,8,10,20,40,70,100,150])
ax.set_xticks(logticks)
ax.get_xaxis().set_major_formatter(ScalarFormatter())
logticks = np.array([3,5,8,10,20,40,70,100])
ax.set_yticks(logticks)
ax.get_yaxis().set_major_formatter(ScalarFormatter())
axx.set_xlim(5,150)
axy.set_ylim(3,110)
ax.set_xlim(5,150)
ax.set_ylim(3,110)
ax.tick_params(labelsize=14)
cbar = fig.colorbar(plt.cm.ScalarMappable(cmap='cool'), ax = axy, ticks=ticks, label = 'Category')
cbar.ax.set_yticklabels(categories) 
for axis in [axx,axy]:
    axis.set_xticks([])
    axis.set_yticks([])
    axis.set_xlabel("")
    axis.set_ylabel("")
plt.tight_layout()
plt.savefig(paths.figures / 'reweighed_catalog_m1m2.pdf', dpi=300)
plt.savefig(paths.figures / 'reweighed_catalog_m1m2.png', dpi=300)