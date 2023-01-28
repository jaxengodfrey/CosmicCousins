 #https://matplotlib.org/matplotblog/posts/create-ridgeplots-in-matplotlib/

import paths
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
import arviz as az
import seaborn as sns
import matplotlib.gridspec as gridspec
from matplotlib.ticker import ScalarFormatter
from utils import load_trace
from utils import load_03b_posteriors
import deepdish as dd
import matplotlib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as grid_spec
from mpl_toolkits.axes_grid1 import make_axes_locatable


data = load_03b_posteriors()

pedata = data['pedata']
param_map = data['param_map']

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

idata = load_trace()
n_categories = 3
probs = idata.posterior['Qs'].mean(axis = 1).values[0] / (n_categories - 1)
sorted_probs = np.argsort(probs)

num_events = 69

gs = grid_spec.GridSpec(num_events,3)
fig = plt.figure(figsize=(8,10))


ax_obj1 = []
ax_obj2 = []
ax_obj3 = []
for i in range(num_events):
    num = sorted_probs[i]
    event = event_names[num]
    x1 = pedata[param_map['mass_1']][num]
    x2 = pedata[param_map['a_1']][num]
    x3 = pedata[param_map['cos_tilt_1']][num]

    # creating new axes object
    ax_obj1.append(fig.add_subplot(gs[i:i+1, 0]))
    ax_obj2.append(fig.add_subplot(gs[i:i+1, 1]))
    ax_obj3.append(fig.add_subplot(gs[i:i+1, 2]))
    ax_objs = [ax_obj1, ax_obj2, ax_obj3]

    # plot the posteriors
    sns.kdeplot(x=x1, ax=ax_obj1[-1], color='k', lw=1, fill=True, multiple = 'stack', log_scale = True)
    sns.kdeplot(x=x2, ax=ax_obj2[-1], color='k', lw=1, fill=True, multiple = 'stack')
    sns.kdeplot(x=x3, ax=ax_obj3[-1], color='k', lw=1, fill=True, multiple = 'stack')


    # setting uniform x and y lims
    ax_obj1[-1].set_xlim(5,100)
    ax_obj1[-1].set_ylim(bottom = 0)

    ax_obj2[-1].set_xlim(0,1)
    ax_obj2[-1].set_ylim(0,7)

    ax_obj3[-1].set_xlim(-1,1)
    ax_obj3[-1].set_ylim(0,7)

    # make background transparent, remove borders, axis ticks, and labels
    for k in range(len(ax_objs)):
        rect = ax_objs[k][-1].patch
        rect.set_alpha(0)

        ax_objs[k][-1].set_ylabel('')
        ax_objs[k][-1].set_yticklabels([])

    
    #set subplot axes labels
    spines = ["top","right","left"]
    if i == num_events-1:
        ax_obj1[-1].set_xlabel(r"$m_1$", fontsize=12)
        logticks = np.array([5,10,20,40,60,100])
        ax_obj1[-1].set_xticks(logticks)
        ax_obj1[-1].get_xaxis().set_major_formatter(ScalarFormatter())

        ax_obj2[-1].set_xlabel(r"$a_1$", fontsize=12)
        ax_obj2[-1].set_xticks([0, 0.2, 0.5, 0.8, 1])
        ax_obj2[-1].set_xticklabels(['0', '0.2', '0.5', '0.8', '1'] , fontsize = 10)

        ax_obj3[-1].set_xlabel(r"$cos(\theta_1)}$", fontsize=12)
        ax_obj3[-1].set_xticks([-1, -0.5, 0, 0.5, 1])
        ax_obj3[-1].set_xticklabels(['-1', '-0.5', '0', '0.5', '1'] , fontsize = 10)
        

        for s in spines:
            for ax in ax_objs:
                ax[-1].spines[s].set_visible(False)

    else:
        for ax in ax_objs:
            ax[-1].tick_params(axis='x',which='both', bottom=False, top=False)
            ax[-1].set_xticklabels([])
            ax[-1].spines['bottom'].set(alpha = 0.4)


        for s in spines:
            for ax in ax_objs:
                ax[-1].spines[s].set_visible(False)

    for ax in ax_objs:
        ax[-1].tick_params(axis='y',which='both', left=False, right=False)

    # adj_event = event.replace(" ","\n")
    # ax_obj2[-1].text(1.05,0,adj_event,fontsize=6,ha="left")

gs.update(hspace=-0.7)

# plt.suptitle("Primary Mass and Spin Magnitude Re-Weighed Posteriors")
fig.subplots_adjust(left = 0.13, right = 0.97, bottom = 0.07, wspace = 0.15, top = 0.995)
plt.savefig(paths.figures / 'ridgeplot.png', bbox_inches = 'tight')
plt.savefig(paths.figures / 'ridgeplot.pdf')

# plt.tight_layout()
plt.show()
