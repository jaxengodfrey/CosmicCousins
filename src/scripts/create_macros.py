import json2latex
import json
import numpy as np
from scipy.integrate import cumtrapz
from scipy.stats import gaussian_kde
import paths
from gwinfernodata import GWInfernoData
from utils import load_gwinfernodata_ppds, load_gwinfernodata_idata

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
    c90 = get_percentile(pdfs, xs, 90)
    c95 = get_percentile(pdfs, xs, 95)
    c10 = get_percentile(pdfs, xs, 10)
    c5 = get_percentile(pdfs, xs, 5)
    c1 = get_percentile(pdfs, xs, 1)
    peaks = get_max_values(xs, pdfs) 
    if max:
        return {'1percentile': save_param_cred_intervals(c1),
                '10percentile': save_param_cred_intervals(c10),
                '90percentile': save_param_cred_intervals(c90),
                '99percentile': save_param_cred_intervals(c99),
                'max': save_param_cred_intervals(peaks)}
    else:
        return {'1percentile': save_param_cred_intervals(c1),
                '10percentile': save_param_cred_intervals(c10),
                '90percentile': save_param_cred_intervals(c90),
                '99percentile': save_param_cred_intervals(c99)}

def tilt_fracs(cts, ct_pdfs):
    gamma_fracs = []
    frac_neg_cts = []
    # frac_less01 = []
    # frac_less03 = []
    for i in range(len(ct_pdfs)):
        ct_pdfs[i,:] /= np.trapz(ct_pdfs[i,:], cts)
        gam = ct_pdfs[i, cts>=0.9] / ct_pdfs[i, cts<=-0.9]
        gamma_fracs.append(gam)
        neg = cts <= 0
        # less03 = cts <= -0.3
        # less01 = cts <= 0.1
        frac_neg_cts.append(np.trapz(ct_pdfs[i,neg], x=cts[neg]))
        # frac_less01.append(np.trapz(ct_pdfs[i,less01], x=cts[less01]))
        # frac_less03.append(np.trapz(ct_pdfs[i,less03], x=cts[less03]))
    gamma_fracs = np.array(gamma_fracs)
    frac_neg_cts = np.array(frac_neg_cts)
    # frac_less01 = np.array(frac_less01)
    # frac_less03 = np.array(frac_less03)
    return np.log10(gamma_fracs), frac_neg_cts #, frac_less01, frac_less03

def get_branching_ratios(categories, Ps):
    median = np.median(Ps, axis = 0)
    lower = median - np.percentile(Ps, 5, axis = 0)
    higher = np.percentile(Ps, 95, axis = 0) - median
    branch_dict = {}
    for i in range(len(categories)):
        branch_dict[categories[i]] = {'Frac': {'median': round_sig(median[i]), 'error plus': round_sig(higher[i]), 'error minus': round_sig(lower[i])}, 'Percent': {'median': round_sig(median[i]*100), 'error plus': round_sig(higher[i]*100), 'error minus': round_sig(lower[i]*100)}}
    return branch_dict

def get_num_constraining_events(categories, posteriors):
    num_dict = {}
    # if g1:
    #     Qs = idata.posterior['Qs'].values[0]
    #     for i in range(len(categories)):
    #         sums = np.sum(Qs == i, axis = 1)
    #         median = np.median(sums)
    #         lower = median - np.percentile(sums, 5)
    #         higher = np.percentile(sums, 95) - median
    #         num_dict[categories[i]] = {'median': round_sig(median), 'error plus': round_sig(higher), 'error minus': round_sig(lower), 'low': round_sig(median - lower), 'high': round_sig(median + higher)}
    # else:
    n_categories = len(categories)
    n_events = 69
    n_samples = posteriors['logmp'].values[0].shape[0]
    groups = np.zeros((n_events, n_categories, n_samples))
    for i in range(n_events):
        for j in range(n_categories):
            groups[i][j] = posteriors[f'cat_frac_subpop_{j+1}_event_{i}'].values[0]
        
    sums = np.sum(groups, axis = 0)
    for i in range(n_categories):
        median = np.nanmedian(sums, axis = 1)[i]
        lower = median - np.nanpercentile(sums, 5, axis  = 1)[i]
        higher = np.nanpercentile(sums, 95, axis = 1)[i] - median
        num_dict[categories[i]] = {'median': round_sig(median), 'error plus': round_sig(higher), 'error minus': round_sig(lower), 'low': round_sig(median - lower), 'high': round_sig(median + higher)}

    return num_dict


def BFMacros(g1_posterior, g2_posterior):

    g1_kernel = gaussian_kde(g1_posterior['Ps'].values[0][:,0])
    g2_kernel = gaussian_kde(np.vstack([g2_posterior['Ps'].values[0][:,0],g2_posterior['Ps'].values[0][:,2]]))
    BF_CYB_PC = g2_kernel((0,0))
    BF_CYB_IP = g1_kernel(0)
    BF_IP_PC = BF_CYB_PC / BF_CYB_IP

    return round_sig(-np.log10(BF_CYB_IP), sig=3), round_sig(-np.log10(BF_CYB_PC), sig=3), round_sig(-np.log10(BF_IP_PC), sig=3)

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
            # x['less01frac'] = save_param_cred_intervals(f01)
            # x['less03frac'] = save_param_cred_intervals(f03)
            # x['fdyn'] = save_param_cred_intervals(2*fn)
            # x['fhm'] = save_param_cred_intervals(6.25*f03)
            categories_dict[categories[i]] = x
        return categories_dict

def MassMacros(categories, ppds, g1 = True):
    if g1:
        ms, m_ppds = ppds['mass_1'].values, [ppds['peak_1_mass_pdfs'].values, ppds['continuum_mass_pdfs'].values]
        return DistMacros(ms, m_ppds, categories, 'Mass')

    else:
        ms, m_ppds = ppds['mass_1'].values, [ppds['peak_1_mass_pdfs'].values, ppds['continuum_mass_pdfs'].values, ppds['continuum_1_mass_pdfs'].values]
        return DistMacros(ms, m_ppds, categories, 'Mass')

def SpinMagMacros(categories, ppds, g1 = True):
    if g1:
        aa, a_ppds = ppds['a1'].values, [ppds['peak_1_a1_pdfs'].values, ppds['continuum_a1_pdfs'].values]
        return DistMacros(aa, a_ppds, categories, 'SpinMag')
    else:
        aa, a_ppds = ppds['a1'].values, [ppds['peak_continuum_a1_pdfs'].values, ppds['continuum_a1_pdfs'].values]
        categories = ['PeakAContinuumA', 'ContinuumB']
        return DistMacros(aa, a_ppds, categories, 'SpinMag')

def TiltMacros(categories, ppds, g1 = True):
    if g1:
        cts, ct_ppds = ppds['cos_tilt_1'].values, [ppds['peak_1_ct1_pdfs'].values, ppds['continuum_ct1_pdfs'].values]
        return DistMacros(cts, ct_ppds, categories, 'tilt', tilt = True)
    else:
        cts, ct_ppds = ppds['cos_tilt_1'].values, [ppds['peak_continuum_ct1_pdfs'].values, ppds['continuum_ct1_pdfs'].values]
        categories = ['PeakAContinuumA', 'ContinuumB']
        return DistMacros(cts, ct_ppds, categories, 'tilt', tilt = True)
    
def ChiEffMacros(categories, ppds, g1 = True):
    if g1:
        cts, ct_ppds = ppds['chi_eff'].values, [ppds['peak_1_chi_eff_pdfs'].values, ppds['continuum_chi_eff_pdfs'].values]
        return DistMacros(cts, ct_ppds, categories, 'ChiEff')
    else:
        cts, ct_ppds = ppds['chi_eff'].values, [ppds['peak_continuum_chi_eff_pdfs'].values, ppds['continuum_chi_eff_pdfs'].values]
        categories = ['PeakAContinuumA', 'ContinuumB']
        return DistMacros(cts, ct_ppds, categories, 'ChiEff')

def BranchingRatioMacros(categories, posteriors):
    return get_branching_ratios(categories, posteriors['Ps'].values[0])

def NumEventsMacros(categories, posteriors):
    return get_num_constraining_events(categories, posteriors)

# def DICMacros(posteriors):
#     log_l = np.asarray(posteriors['log_l'])
#     dic = -2 * (np.mean(log_l) - np.var(log_l))
#     return round_sig(dic, sig = 4)

def main(): 
    macro_dict = {'Mass': {}, 'SpinMag': {}, 'CosTilt': {}}
    g1_ppds =  load_gwinfernodata_ppds()
    g2_ppds = load_gwinfernodata_ppds(IP = False)
    g1_idata = load_gwinfernodata_idata()
    g2_idata = load_gwinfernodata_idata(IP = False)

    g1_categories = ['PeakA', 'ContinuumB'] 
    g2_categories = ['PeakA', 'ContinuumB', 'ContinuumA']
    BFs = BFMacros(g1_idata.posterior, g2_idata.posterior)
    macro_dict['LogBayesFactors'] = {'IP_to_CYB': BFs[0], 'PC_to_CYB': BFs[1], 'PC_to_IP': BFs[2]}
    macro_dict['Mass'] = {'Base': MassMacros(g1_categories, g1_ppds.pdfs), 'Composite': MassMacros(g2_categories, g2_ppds.pdfs, g1 = False)}
    macro_dict['SpinMag'] = {'Base': SpinMagMacros(g1_categories, g1_ppds.pdfs), 'Composite': SpinMagMacros(g2_categories, g2_ppds.pdfs, g1 = False)}
    macro_dict['CosTilt'] = {'Base': TiltMacros(g1_categories, g1_ppds.pdfs), 'Composite': TiltMacros(g2_categories, g2_ppds.pdfs, g1 = False)}
    macro_dict['BranchingRatios'] = {'Base': BranchingRatioMacros(g1_categories, g1_idata.posterior), 'Composite': BranchingRatioMacros(g2_categories, g2_idata.posterior)}
    macro_dict['NumEvents'] = {'Base': NumEventsMacros(g1_categories, g1_idata.posterior), 'Composite': NumEventsMacros(g2_categories, g2_idata.posterior)}
    print('done')
    macro_dict['ChiEff'] = {'Base': ChiEffMacros(g1_categories, g1_ppds.pdfs), 'Composite': ChiEffMacros(g2_categories, g2_ppds.pdfs, g1 = False)}
    sel = g2_ppds.pdfs.coords['sel']
    sel_1 = g2_idata.posterior['Ps'].values[0][sel,1] < g2_idata.posterior['Ps'].values[0][sel,2] 
    num_1 = np.mean(sel_1)*100
    num_2 = np.mean(~sel_1)*100
    macro_dict['FracCut'] = {'BgreaterA': round_sig(num_1, sig=2), 'BlessA': round_sig(num_2, sig=2)}
    # cyb = dd.io.load(paths.data / 'updated/cover_your_basis_30000w_20000s_posterior_samples.h5')
    # peak_ss = dd.io.load(paths.data / 'updated/bspline_1logpeak_samespin_posterior_samples.h5')
    # macro_dict['DICs'] = {'Base': DICMacros(g1_idata), 'Composite': DICMacros(g2_idata), 'CYB': DICMacros(cyb), 'BaseSS': DICMacros(peak_ss)}


    print("Saving macros to src/data/macros.json...")
    with open(paths.data / "macros.json", 'w') as f:
        json.dump(macro_dict, f)
    print("Creating macros in src/tex/macros.tex from data in src/data/macros.json...")
    with open(paths.tex / "macros.tex", 'w') as ff:
        json2latex.dump('macros', macro_dict, ff)

if __name__ == '__main__':
    main()