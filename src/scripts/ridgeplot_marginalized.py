# https://matplotlib.org/matplotblog/posts/create-ridgeplots-in-matplotlib/

import paths
import numpy as np
import arviz as az
import seaborn as sns
import matplotlib.gridspec as gridspec
from matplotlib.ticker import ScalarFormatter
from utils import load_trace, load_03b_posteriors, load_gwinfernodata_idata
import deepdish as dd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as grid_spec
from matplotlib.ticker import ScalarFormatter
from gwinfernodata import GWInfernoData
import matplotlib
matplotlib.rcParams['text.usetex'] = True


# matplotlib.rcParams['text.usetex'] = True

categories = ['1', '2', '3']
# categories = ['1', '2']
idata = load_gwinfernodata_idata(IP = False) #az.from_netcdf(paths.data/'b1logpeak_marginalized_50000s_2chains.h5')
idata = idata.posterior.isel(draw=np.random.choice(500000,1000, replace=False))
# ppds = GWInfernoData.from_netcdf(paths.data/'updated/2bspline_1logpeak_30000w_20000s_ppds.h5').pdfs
# # posteriors = dd.io.load(paths.data/'composite_model_resampled_single_event_posterior.h5')
# sel = np.ones_like(ppds['continuum_mass_pdfs'][:,127].values < 1e-3)#ppds['continuum_mass_pdfs'][:,127] < 1e-3

data = load_03b_posteriors()

# thin = np.random.choice([0,1], len(idata['log_l']), p = [0.8, 0.2]) == 1

pedict = data.pedata.data.to_dict()
pedata = pedict['data']
param_map = pedict['attrs']



# idata = az.extract(dat, combined = 'True', num_samples = 3500)

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

categories = ['Peak A', 'Continuum A', 'Continuum B']

# sel = idata['Ps'].values[0][:,1] > idata['Ps'].values[0][:,2]

n_categories = len(categories)
n_events = len(event_names)
groups = np.zeros((n_events, n_categories))
probs = np.zeros(n_events)
for i in range(n_events):
    for j in range(n_categories):
        groups[i][j] = idata[f'cat_frac_subpop_{j+1}_event_{i}'].values[0].mean()
        if j == 0:
            probs[i] += groups[i][j] * j
        if j == 1:
            probs[i] += groups[i][j] * (j+1)
        if j == 2:
            probs[i] += groups[i][j] * (j-1)
    probs[i] = probs[i] / (n_categories - 1)

sorted_probs = np.argsort(probs)
means = np.array([idata[f'mass_1_obs_event_{i}'].values[0].mean() for i in range(n_events)])
sorted_probs = np.argsort(means)

##### infinity in cat frac

ticks = np.linspace(0,1, n_categories)
cm = plt.cm.cool(probs)

hex = []
for i in range(len(cm)):
    hex.append(matplotlib.colors.rgb2hex(cm[i]))

gs = grid_spec.GridSpec(len(cm),4)
fig = plt.figure(figsize=(8,10))


# qs = np.array(idata.posterior["Qs"][0]).transpose()

# n_events = qs.shape[0]
# n_samp = qs.shape[1]
# groups = np.zeros([n_categories,n_events])
# for i in range(n_categories):
#     x = np.array([np.sum(qs == i, axis = 1) / n_samp])
#     groups[i] = x
# groups = groups.transpose()

# cax = fig.add_subplot(gs[0,:])
# ticks = np.linspace(0,1, 3)
# cbar = fig.colorbar(plt.cm.ScalarMappable(cmap='cool'), ax = cax, location = 'top', ticks=ticks, label = 'Category')
# cbar.ax.set_xticklabels(categories) 

sorted_groups = groups[np.flip(sorted_probs)]
sorted_names = np.array(event_names)[np.flip(sorted_probs)]

colors = ['cyan', 'magenta', 'mediumpurple']


ax3 = fig.add_subplot(gs[3:,0])
left = len(cm) * [0]
for idx in range(n_categories):
    ax3.barh(sorted_names, sorted_groups[:, idx], left = left, color=colors[idx], align = 'edge', height = 0.7)
    left = left + sorted_groups[:, idx]
ax3.margins(0,0)
ax3.set_xticks([0, 0.2, 0.5, 0.8, 1])
ax3.set_xlabel(r"probability", fontsize=12)
ax3.set_xticklabels(['0', '0.2', '0.5', '0.8', '1'])
plt.yticks(fontsize = 6)

ax_obj1 = []
ax_obj2 = []
ax_obj3 = []
for i in range(len(cm)):
    num = sorted_probs[i]
    event = event_names[num]
    # x1 = posteriors[event]['mass_1']
    # x2 = posteriors[event]['a_1']
    # x3 = posteriors[event]['cos_tilt_1']

    # y1 = pedata[param_map['mass_1']][num]
    # y2 = pedata[param_map['a_1']][num]
    # y3 = pedata[param_map['cos_tilt_1']][num]

    x1 = idata[f'mass_1_obs_event_{num}'].values[0]
    x2 = idata[f'a_1_obs_event_{num}'].values[0]
    x3 = idata[f'cos_tilt_1_obs_event_{num}'].values[0]

    y1 = pedata[param_map['mass_1']][num]
    y2 = pedata[param_map['a_1']][num]
    y3 = pedata[param_map['cos_tilt_1']][num]

    # creating new axes object
    ax_obj1.append(fig.add_subplot(gs[i:i+1, 1]))
    ax_obj2.append(fig.add_subplot(gs[i:i+1, 2]))
    ax_obj3.append(fig.add_subplot(gs[i:i+1, 3]))
    ax_objs = [ax_obj1, ax_obj2, ax_obj3]

    # plot the posteriors

    sns.kdeplot(x=y1, ax=ax_obj1[-1], color='gray', lw=1, fill=False, multiple = 'stack', log_scale = True, ls = ':')
    sns.kdeplot(x=y2, ax=ax_obj2[-1], color='gray', lw=1, fill=False, multiple = 'stack', ls = ':')
    sns.kdeplot(x=y3, ax=ax_obj3[-1], color='gray', lw=1, fill=False, multiple = 'stack', ls = ':')

    sns.kdeplot(x=x1, ax=ax_obj1[-1], color=hex[num], lw=1, fill=True, multiple = 'stack')
    sns.kdeplot(x=x2, ax=ax_obj2[-1], color=hex[num], lw=1, fill=True, multiple = 'stack')
    sns.kdeplot(x=x3, ax=ax_obj3[-1], color=hex[num], lw=1, fill=True, multiple = 'stack')


    # ax_obj1[-1].set_box_aspect(0.2)


    # setting uniform x and y lims
    ax_obj1[-1].set_xlim(5,100)
    ax_obj1[-1].set_ylim(0, 30)

    ax_obj2[-1].set_xlim(0,1)
    ax_obj2[-1].set_ylim(0,7)

    ax_obj3[-1].set_xlim(-1,1)
    ax_obj3[-1].set_ylim(0,7)

    # make background transparent, remove borders, axis ticks, and labels
    for k in range(len(ax_objs)):
        rect = ax_objs[k][-1].patch
        rect.set_alpha(0)

        ax_objs[k][-1].set_ylabel('')
        ax_objs[k][-1].set_xlabel('')
        ax_objs[k][-1].set_yticklabels([])


    #set subplot axes labels
    spines = ["top","right","left"]
    if i == len(hex)-1:
        ax_obj1[-1].set_xlabel(r"$m_1$", fontsize=12)
        logticks = np.array([5,10,20,40,60,100])
        ax_obj1[-1].set_xticks(logticks)
        ax_obj1[-1].get_xaxis().set_major_formatter(ScalarFormatter())

        ax_obj2[-1].set_xlabel(r"$a_1$", fontsize=12)
        ax_obj2[-1].set_xticks([0, 0.2, 0.5, 0.8, 1])
        ax_obj2[-1].set_xticklabels(['0', '0.2', '0.5', '0.8', '1'] , fontsize = 10)

        ax_obj3[-1].set_xlabel(r"$\cos{\theta_1}$", fontsize=12)
        ax_obj3[-1].set_xticks([-1, -0.5, 0, 0.5, 1])
        ax_obj3[-1].set_xticklabels(['-1', '-0.5', '0', '0.5', '1'] , fontsize = 10)


        for s in spines:
            for ax in ax_objs:
                ax[-1].spines[s].set_visible(False)
            ax3.spines[s].set_visible(False)

    else:
        for ax in ax_objs:
            ax[-1].tick_params(axis='x',which='both', bottom=False, top=False)
            ax[-1].set_xticklabels([])
            ax[-1].spines['bottom'].set(alpha = 0.4)


        for s in spines:
            for ax in ax_objs:
                ax[-1].spines[s].set_visible(False)
            ax3.spines[s].set_visible(False)

    for ax in ax_objs:
        ax[-1].tick_params(axis='y',which='both', left=False, right=False)

    # adj_event = event.replace(" ","\n")
    # ax_obj2[-1].text(1.05,0,adj_event,fontsize=6,ha="left")


gs.update(hspace=-0.7)

# plt.suptitle("Primary Mass and Spin Magnitude Re-Weighed Posteriors")
fig.subplots_adjust(left = 0.13, right = 0.97, bottom = 0.07, wspace = 0.15, top = 0.995)
plt.savefig(paths.figures / 'ridgeplot_marginalized.png', bbox_inches = 'tight')
plt.savefig(paths.figures / 'ridgeplot_marginalized.pdf', bbox_inches = 'tight')

# plt.tight_layout()
# plt.show()
