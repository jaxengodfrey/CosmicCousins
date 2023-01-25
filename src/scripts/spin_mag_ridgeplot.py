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
from sklearn.neighbors import KernelDensity
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.gridspec as grid_spec

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

gs = grid_spec.GridSpec(len(cm),1)
fig = plt.figure(figsize=(5,10))


ax_objs = []
for i in range(len(cm)):
    num = sorted_probs[i]
    event = event_names[num]
    x = idata.posterior['a_1_obs_event_{}'.format(num)].values[0]
    x_d = np.linspace(0,1, 1000)

    kde = KernelDensity(bandwidth=0.03, kernel='gaussian')
    kde.fit(x[:, None])

    logprob = kde.score_samples(x_d[:, None])

    # creating new axes object
    ax_objs.append(fig.add_subplot(gs[i:i+1, 0]))

    # plotting the distribution
    ax_objs[-1].plot(x_d, np.exp(logprob),color="#f0f0f0",lw=1)
    ax_objs[-1].fill_between(x_d, np.exp(logprob), alpha=1,color=hex[num])


    # setting uniform x and y lims
    ax_objs[-1].set_xlim(0,1)
    ax_objs[-1].set_ylim(0,7)

    # make background transparent
    rect = ax_objs[-1].patch
    rect.set_alpha(0)

    # remove borders, axis ticks, and labels
    ax_objs[-1].set_yticklabels([])

    if i == len(hex)-1:
        ax_objs[-1].set_xlabel("spin magnitude", fontsize=16,fontweight="bold")
    else:
        ax_objs[-1].tick_params(axis='x',which='both', bottom=False, top=False)
        ax_objs[-1].set_xticklabels([])

    ax_objs[-1].tick_params(axis='y',which='both', left=False, right=False)

    spines = ["top","right","left","bottom"]
    for s in spines:
        ax_objs[-1].spines[s].set_visible(False)

    adj_event = event.replace(" ","\n")
    ax_objs[-1].text(-0.02,0,adj_event,fontsize=6,ha="right")


gs.update(hspace=-0.7)

plt.suptitle("Spin Magnitude Re-Weighed Posteriors")
fig.subplots_adjust(top=0.95, left = 0.23)

# plt.tight_layout()
plt.show()