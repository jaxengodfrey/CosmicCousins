from utils import radar_plot, load_trace
import arviz as az
import matplotlib.pyplot as plt
import paths
import numpy as np

idata = load_trace()
categories = ['Low-Mass\nPeak', 'High-Mass\nPeak', 'Continuum']

qs = np.array(idata.posterior["Qs"][0]).transpose()

n_categories = len(categories)
n_events = qs.shape[0]
n_samp = qs.shape[1]
groups = np.zeros([n_categories,n_events])
for i in range(n_categories):
    x = np.array([np.sum(qs == i, axis = 1) / n_samp])
    groups[i] = x
groups = groups.transpose()
probs = idata.posterior['Qs'].mean(axis = 1).values[0] / (n_categories - 1)

ticks = np.linspace(0, 1, n_categories)
cm = plt.cm.cool(probs)
fig, ax = plt.subplots(subplot_kw = {'projection': 'polar'}, figsize = (6,6))
radar_plot(categories, groups, ax, cm = cm)
cbar = fig.colorbar(plt.cm.ScalarMappable(cmap='cool'), ax = ax, ticks=ticks, orientation = 'horizontal', shrink = 1)
cbar.ax.set_xticklabels(categories) 
# cbar.set_label('Category', y=1.0)
plt.savefig(paths.figures / 'radar_plot.pdf', dpi=300);
plt.savefig(paths.figures / 'radar_plot.png', dpi=300);
plt.close()