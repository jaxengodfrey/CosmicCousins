from utils import load_trace
from tabulate import tabulate
import sys
import paths
import numpy as np

idata = load_trace()
categories = ['Low-Mass Peak', 'High-Mass Peak', 'Continuum']
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

qs = np.array(idata.posterior["Qs"][0]).transpose()

n_categories = len(categories)
n_events = qs.shape[0]
n_samp = qs.shape[1]
groups = np.zeros([n_categories,n_events])
for i in range(n_categories):
    means = np.array([np.sum(qs == i, axis = 1) / n_samp]).round(decimals = 2)
    groups[i] = means
groups = groups.transpose()

events_ls = []
for i in range(n_events):
    ls = [event_names[i]]
    for j in range(n_categories):
        ls.append(groups[i][j])
    events_ls.append(ls)

orig_stdout = sys.stdout
f = open(paths.tex / 'output/event_cat_table.tex', 'w')
sys.stdout = f

print(tabulate(events_ls, headers = ['Event'] + categories, tablefmt = 'latex_longtable'))

sys.stdout = orig_stdout
f.close()
