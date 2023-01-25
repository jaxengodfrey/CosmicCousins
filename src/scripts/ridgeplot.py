# https://matplotlib.org/matplotblog/posts/create-ridgeplots-in-matplotlib/

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
import matplotlib
import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.gridspec as grid_spec
from mpl_toolkits.axes_grid1 import make_axes_locatable

idata = load_trace()

event_names = ['GW150914',
 'GW151012',
 'GW151226',
 'GW170104',
 'GW170608',
 'GW170729',
 'GW170809',
 'GW170814',
 'GW170818',
 'GW170823',
 'GW190408_181802',
 'GW190412',
 'GW190413_134308',
 'GW190413_052954',
 'GW190421_213856',
 'GW190503_185404',
 'GW190512_180714',
 'GW190513_205428',
 'GW190517_055101',
 'GW190519_153544',
 'GW190521',
 'GW190521_074359',
 'GW190527_092055',
 'GW190602_175927',
 'GW190620_030421',
 'GW190630_185205',
 'GW190701_203306',
 'GW190706_222641',
 'GW190707_093326',
 'GW190708_232457',
 'GW190719_215514',
 'GW190720_000836',
 'GW190725_174728',
 'GW190727_060333',
 'GW190728_064510',
 'GW190731_140936',
 'GW190803_022701',
 'GW190805_211137',
 'GW190828_063405',
 'GW190828_065509',
 'GW190910_112807',
 'GW190915_235702',
 'GW190924_021846',
 'GW190925_232845',
 'GW190929_012149',
 'GW190930_133541',
 'GW191103_012549',
 'GW191105_143521',
 'GW191109_010717',
 'GW191127_050227',
 'GW191129_134029',
 'GW191204_171526',
 'GW191215_223052',
 'GW191216_213338',
 'GW191222_033537',
 'GW191230_180458',
 'GW200112_155838',
 'GW200128_022011',
 'GW200129_065458',
 'GW200202_154313',
 'GW200208_130117',
 'GW200209_085452',
 'GW200216_220804',
 'GW200219_094415',
 'GW200224_222234',
 'GW200225_060421',
 'GW200302_015811',
 'GW200311_115853',
 'GW200316_215756']

n_categories = 3
categories = ['Low-Mass\nPeak', 'High-Mass\nPeak', 'Continuum']
probs = idata.posterior['Qs'].mean(axis = 1).values[0] / (n_categories - 1)
ticks = np.linspace(0,1, n_categories)
cm = plt.cm.cool(probs)
sorted_probs = np.argsort(probs)


hex = []
for i in range(len(cm)):
    hex.append(matplotlib.colors.rgb2hex(cm[i]))

gs = grid_spec.GridSpec(len(cm),3)
fig = plt.figure(figsize=(7,10))

qs = np.array(idata.posterior["Qs"][0]).transpose()

n_categories = len(categories)
n_events = qs.shape[0]
n_samp = qs.shape[1]
groups = np.zeros([n_categories,n_events])
for i in range(n_categories):
    x = np.array([np.sum(qs == i, axis = 1) / n_samp])
    groups[i] = x
groups = groups.transpose()

sorted_groups = groups[np.flip(sorted_probs)]
sorted_names = np.array(event_names)[np.flip(sorted_probs)]

colors = ['cyan', 'mediumpurple', 'magenta']

ax3 = fig.add_subplot(gs[3:,0])
left = len(cm) * [0]
for idx in range(n_categories):
    ax3.barh(sorted_names, sorted_groups[:, idx], left = left, color=colors[idx])
    left = left + sorted_groups[:, idx]
ax3.margins(0,0)
ax3.set_xticks([0, 0.25, 0.5, 0.75, 1])
plt.yticks(fontsize = 6)



ax_obj1 = []
ax_obj2 = []
# ax_obj = []
for i in range(len(cm)):
    num = sorted_probs[i]
    event = event_names[num]
    x1 = idata.posterior['mass_1_obs_event_{}'.format(num)].values[0]
    x2 = idata.posterior['a_1_obs_event_{}'.format(num)].values[0]

    # creating new axes object
    ax_obj1.append(fig.add_subplot(gs[i:i+1, 1]))
    ax_obj2.append(fig.add_subplot(gs[i:i+1, 2]))
    # ax_obj.append(fig.add_subplot(gs[i:i+1, 0]))

    # plotting the distribution
    sns.kdeplot(x=x1, ax=ax_obj1[-1], color=hex[num], lw=1, fill=True, multiple = 'stack', log_scale = True)
    sns.kdeplot(x=x2, ax=ax_obj2[-1], color=hex[num], lw=1, fill=True, multiple = 'stack')
    # ax_objs[-1].plot(x_d, np.exp(logprob),color=hex[num],lw=1)
    # ax_objs[-1].fill_between(x_d, np.exp(logprob), alpha=1,color="#f0f0f0")

    # left = len(cm) * [0]
    # for idx in range(n_categories):
    #     ax_obj[-1].barh(sorted_names[i], sorted_groups[i, idx], left = left, color=colors[idx], height = 0.01)
    #     left = left + sorted_groups[i, idx]


    # setting uniform x and y lims
    # ax_obj1[-1].set_xscale('log')
    ax_obj1[-1].set_xlim(5,100)
    ax_obj1[-1].set_ylim(bottom = 0)
    ax_obj1[-1].set_ylabel('')

    ax_obj2[-1].set_xlim(0,1)
    ax_obj2[-1].set_ylim(0,7)
    ax_obj2[-1].set_ylabel('')

    # make background transparent
    rect1 = ax_obj1[-1].patch
    rect1.set_alpha(0)
    rect2 = ax_obj2[-1].patch
    rect2.set_alpha(0)
    # rect = ax_obj[-1].patch
    # rect.set_alpha(0)



    # remove borders, axis ticks, and labels
    ax_obj1[-1].set_yticklabels([])
    ax_obj2[-1].set_yticklabels([])
    # ax_obj[-1].set_yticklabels([])

    spines = ["top","right","left"]

    if i == len(hex)-1:
        ax_obj1[-1].set_xlabel(r"$m_1$", fontsize=12)
        logticks = np.array([5,10,20,40,60,100])
        ax_obj1[-1].set_xticks(logticks)
        ax_obj2[-1].set_xticks([0, 0.2, 0.5, 0.8, 1])
        ax_obj2[-1].set_xticklabels(['0', '0.2', '0.5', '0.8', '1'] , fontsize = 10)
        ax_obj1[-1].get_xaxis().set_major_formatter(ScalarFormatter())

        ax_obj2[-1].set_xlabel(r"$a_1$", fontsize=12)
        # ax_obj[-1].set_xlabel(r"event_cat", fontsize=12)

        for s in spines:
            ax_obj1[-1].spines[s].set_visible(False)
            ax_obj2[-1].spines[s].set_visible(False)
            ax3.spines[s].set_visible(False)
            # ax_obj[-1].spines[s].set_visible(False)

        # cbar = fig.colorbar(plt.cm.ScalarMappable(cmap='cool'), ax = ax_objs, ticks=ticks, label = 'Category', fraction=0.2, pad=0.4)
        # cbar.ax.set_yticklabels(categories) 
    else:
        ax_obj1[-1].tick_params(axis='x',which='both', bottom=False, top=False)
        ax_obj1[-1].set_xticklabels([])
        ax_obj1[-1].spines['bottom'].set(alpha = 0.4)

        ax_obj2[-1].tick_params(axis='x',which='both', bottom=False, top=False)
        ax_obj2[-1].set_xticklabels([])
        ax_obj2[-1].spines['bottom'].set(alpha = 0.4)

        # ax_obj[-1].tick_params(axis='x',which='both', bottom=False, top=False)
        # ax_obj[-1].set_xticklabels([])
        # ax_obj[-1].spines['bottom'].set(alpha = 0.4)

        for s in spines:
            ax_obj1[-1].spines[s].set_visible(False)
            ax_obj2[-1].spines[s].set_visible(False)
            ax3.spines[s].set_visible(False)
            # ax_obj[-1].spines[s].set_visible(False)

    
    ax_obj1[-1].tick_params(axis='y',which='both', left=False, right=False)
    ax_obj2[-1].tick_params(axis='y',which='both', left=False, right=False)
    # ax_obj[-1].tick_params(axis='y',which='both', left=False, right=False)

    # adj_event = event.replace(" ","\n")
    # ax_obj2[-1].text(1.05,0,adj_event,fontsize=6,ha="left")

gs.update(hspace=-0.7)

plt.suptitle("Primary Mass and Spin Magnitude Re-Weighed Posteriors")
fig.subplots_adjust(top=0.95, left = 0.15, right = 0.97, bottom = 0.07, wspace = 0.15)
plt.savefig(paths.figures / 'ridgeplot.png', bbox_inches = 'tight')
plt.savefig(paths.figures / 'ridgeplot.pdf')

# plt.tight_layout()
plt.show()
