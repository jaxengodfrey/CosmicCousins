import json2latex
import json
import numpy as np
from scipy.integrate import cumtrapz
import paths
from utils import load_subpop_ppds, load_trace

def round_sig(f, sig=2):
    max10exp = np.floor(np.log10(abs(f))) + 1
    if max10exp < float(sig):
        return float(('%.' + str(sig) + 'g') % f)
    else:
        return int(float(('%.' + str(sig) + 'g') % f))

def save_param_cred_intervals(param_data):
    return  {'median': round_sig(np.median(param_data)), 
             'error plus':round_sig(np.percentile(param_data, 95)-np.median(param_data)), 
             'error minus': round_sig(np.median(param_data)-np.percentile(param_data, 5)),
             '5th percentile': round_sig(np.percentile(param_data, 5)), 
             '95th percentile': round_sig(np.percentile(param_data, 95)), 
             '10th percentile': round_sig(np.percentile(param_data, 10)),
             '90th percentile': round_sig(np.percentile(param_data, 90))}
    

def get_percentile(pdfs, xs, perc):
    x = []
    for m in pdfs:                                                                                                                                                                        
        i = len(m)
        cumulative_prob = cumtrapz(m, xs, initial = 0)
        init_prob = cumulative_prob[-1]
        prob = init_prob
        final_prob = init_prob * perc / 100.0                                                                                                                                              
        while prob > (final_prob):
            i -= 1
            prob = cumulative_prob[i]                                                                                                                                                         
        x.append(xs[i])
    return np.array(x)

def get_max_values(xs, pdfs):
    x = []
    for i in range(len(pdfs)):
        x.append(xs[np.argmax(pdfs[i,:])])
    return np.array(x)

def save_subpop_cred_intervals(xs, pdfs, max = True):
    c99 = get_percentile(pdfs, xs, 99)
    c1 = get_percentile(pdfs, xs, 1)
    peaks = get_max_values(xs, pdfs) 
    if max:
        return {'1percentile': save_param_cred_intervals(c1),
                '99percentile': save_param_cred_intervals(c99),
                'max': save_param_cred_intervals(peaks)}
    else:
        return {'1percentile': save_param_cred_intervals(c1), 
                '99percentile': save_param_cred_intervals(c99)}

def tilt_fracs(cts, ct_pdfs):
    gamma_fracs = []
    frac_neg_cts = []
    for i in range(len(ct_pdfs)):
        ct_pdfs[i,:] /= np.trapz(ct_pdfs[i,:], cts)
        gam = ct_pdfs[i, cts>=0.9] / ct_pdfs[i, cts<=-0.9]
        gamma_fracs.append(gam)
        neg = cts <= 0
        frac_neg_cts.append(np.trapz(ct_pdfs[i,neg], x=cts[neg]))
    gamma_fracs = np.array(gamma_fracs)
    frac_neg_cts = np.array(frac_neg_cts)
    return np.log10(gamma_fracs), frac_neg_cts

def get_branching_ratios(categories, Ps, chain_idx = 0):
    Ps = Ps.values[chain_idx]
    median = np.median(Ps, axis = 0)
    lower = median - np.percentile(Ps, 5, axis = 0)
    higher = np.percentile(Ps, 95, axis = 0) - median
    branch_dict = {}
    for i in range(len(categories)):
        branch_dict[categories[i]] = {'Frac': {'median': round_sig(median[i]), 'error plus': round_sig(higher[i]), 'error minus': round_sig(lower[i])}, 'Percent': {'median': round_sig(median[i]*100), 'error plus': round_sig(higher[i]*100), 'error minus': round_sig(lower[i]*100)}}
    return branch_dict

def get_num_constraining_events(categories, Qs, chain_idx = 0):
    Qs = Qs.values[chain_idx]
    num_dict = {}
    for i in range(len(categories)):
        sums = np.sum(Qs == i, axis = 1)
        median = np.median(sums)
        lower = median - np.percentile(sums, 5)
        higher = np.percentile(sums, 95) - median
        num_dict[categories[i]] = {'median': round_sig(median), 'error plus': round_sig(higher), 'error minus': round_sig(lower), 'low': round_sig(median - lower), 'high': round_sig(median + higher)}
    return num_dict

def DistMacros(xs, ppds, categories, param_name, tilt = False):
    print('Saving {0} Distribution Macros'.format(param_name))
    categories_dict = {}
    if tilt == False:
        for i in range(len(ppds)):
            categories_dict[categories[i]] = save_subpop_cred_intervals(xs, ppds[i])
        return categories_dict
    else:
        for i in range(len(ppds)):
            l10gf, fn = tilt_fracs(xs, ppds[i])
            x = save_subpop_cred_intervals(xs, ppds[i])
            x['log10gammafrac'] = save_param_cred_intervals(l10gf)
            x['negfrac'] = save_param_cred_intervals(fn)
            categories_dict[categories[i]] = x
        return categories_dict

def MassMacros(categories, ppds, g1 = True):
    if g1:
        ms, m_ppds = ppds['mass_1'], [ppds['peak_1_mass_pdfs'], ppds['continuum_mass_pdfs']]
        return DistMacros(ms, m_ppds, categories, 'Mass')
    else:
        ms, m_ppds = ppds['mass_1'], [ppds['peak_1_mass_pdfs'], ppds['continuum_mass_pdfs'], ppds['continuum_1_mass_pdfs']]
        return DistMacros(ms, m_ppds, categories, 'Mass')

def SpinMagMacros(categories, ppds, g1 = True):
    if g1:
        aa, a_ppds = ppds['a1'], [ppds['peak_1_a1_pdfs']['unweighted'], ppds['continuum_a1_pdfs']['unweighted']]
        return DistMacros(aa, a_ppds, categories, 'SpinMag')
    else:
        aa, a_ppds = ppds['a1'], [ppds['peak_1_continuum_a1_pdfs']['unweighted'], ppds['continuum_a1_pdfs']['unweighted']]
        categories = ['Peak+ContinuumA', 'ContinuumB']
        return DistMacros(aa, a_ppds, categories, 'SpinMag')

def TiltMacros(categories, ppds, g1 = True):
    if g1:
        cts, ct_ppds = ppds['cos_tilt_1'], [ppds['peak_1_ct1_pdfs']['unweighted'], ppds['continuum_ct1_pdfs']['unweighted']]
        return DistMacros(cts, ct_ppds, categories, 'tilt', tilt = True)
    else:
        cts, ct_ppds = ppds['cos_tilt_1'], [ppds['peak_1_continuum_ct1_pdfs']['unweighted'], ppds['continuum_ct1_pdfs']['unweighted']]
        categories = ['Peak+ContinuumA', 'ContinuumB']
        return DistMacros(cts, ct_ppds, categories, 'tilt', tilt = True)

def BranchingRatioMacros(categories, idata, g1 = True):
    return get_branching_ratios(categories, idata.posterior['Ps'])

def NumEventsMacros(categories, idata, g1 = True):
    return get_num_constraining_events(categories, idata.posterior['Qs'])

def main():
    macro_dict = {'Mass': {}, 'SpinMag': {}, 'CosTilt': {}}
    g1_ppds = load_subpop_ppds(g1 = True, g1_fname = 'bspline_1logpeak_100000s_ppds.h5')
    g2_ppds = load_subpop_ppds(g2 = True, g2_fname = 'bspline_1logpeak_samespin_100000s_ppds.h5')
    g1_idata = load_trace(g1 = True, g1_fname = 'bspline_1logpeak_100000s.h5')
    g2_idata = load_trace(g2 = True, g2_fname = 'bspline_peak_samespin_100000s.h5')
    g1_categories = ['Peak', 'Continuum']
    g2_categories = ['PeakA', 'ContinuumB', 'ContinuumA']
    macro_dict['Mass'] = {'Group 1': MassMacros(g1_categories, g1_ppds), 'Group 2': MassMacros(g2_categories, g2_ppds, g1 = False)}
    macro_dict['SpinMag'] = {'Group 1': SpinMagMacros(g1_categories, g1_ppds), 'Group 2': SpinMagMacros(g2_categories, g2_ppds, g1 = False)}
    macro_dict['CosTilt'] = {'Group 1': TiltMacros(g1_categories, g1_ppds), 'Group 2': TiltMacros(g2_categories, g2_ppds, g1 = False)}
    macro_dict['BranchingRatios'] = {'Group 1': BranchingRatioMacros(g1_categories, g1_idata), 'Group 2': BranchingRatioMacros(g2_categories, g2_idata)}
    macro_dict['NumEvents'] = {'Group 1': NumEventsMacros(g1_categories, g1_idata), 'Group 2': NumEventsMacros(g2_categories, g2_idata)}

    print("Saving macros to src/data/macros.json...")
    with open(paths.data / "macros.json", 'w') as f:
        json.dump(macro_dict, f)
    print("Creating macros in src/tex/macros.tex from data in src/data/macros.json...")
    with open(paths.tex / "macros.tex", 'w') as ff:
        json2latex.dump('macros', macro_dict, ff)

if __name__ == '__main__':
    main()