import numpy as np
import paths
import deepdish as dd


def load_o3b_paper_run_masspdf(filename):
    """
    Generates a plot of the PPD and X% credible region for the mass distribution,
    where X=limits[1]-limits[0]
    """
    mass_1 = np.linspace(2, 100, 1000)
    mass_ratio = np.linspace(0.1, 1, 500)
        
    # load in the traces. 
    # Each entry in lines is p(m1 | Lambda_i) or p(q | Lambda_i)
    # where Lambda_i is a single draw from the hyperposterior
    # The ppd is a 2D object defined in m1 and q
    with open(filename, 'r') as _data:
        _data = dd.io.load(filename)
        marginals = _data["lines"]
    for ii in range(len(marginals['mass_1'])):
        marginals['mass_1'][ii] /= np.trapz(marginals['mass_1'][ii], mass_1)
        marginals['mass_ratio'][ii] /= np.trapz(marginals['mass_ratio'][ii], mass_ratio)
    return marginals['mass_1'], marginals['mass_ratio'], mass_1, mass_ratio

def load_mass_ppd():
    datadict = dd.io.load(paths.data / 'bsplines_64m1_18q_iid18mag_iid18tilt_pl18z_ppds.h5')
    return datadict['m1s'], datadict['dRdm1'], datadict['qs'], datadict['dRdq']

def plot_mean_and_90CI(ax, xs, ar, color, label, bounds=True, CI=90, traces=None, tracecolor='k', fill_alpha=0.08, median=False, mean=True):

    if mean:
        me = np.mean(ar, axis=0)    
        ax.plot(xs, me, color=color, label=label, lw=5, alpha=0.75)
    elif median:
        me = np.median(ar,axis=0)
        ax.plot(xs, me, color=color, label=label, lw=5, alpha=0.75)
    if bounds:
        low = np.percentile(ar, (100-CI)/2., axis=0)
        high = np.percentile(ar, 100-(100-CI)/2., axis=0)
        #ax.plot(xs, low, color='k', lw=0.05, alpha=0.05)
        #ax.plot(xs, high, color='k', lw=0.05, alpha=0.05)
        if mean | median:
            label=None
        ax.fill_between(xs, low, high, color=color, alpha=fill_alpha, label=label)
    if traces is not None:
        for _ in range(traces):
            idx = np.random.choice(ar.shape[0])
            ax.plot(xs, ar[idx], color=tracecolor, lw=0.025, alpha=0.02)  
    
    return ax

def load_bsplinemass_ppd():
    datadict = dd.io.load(paths.data / 'bsplines_64m1_18q_iid18mag_iid18tilt_pl18z_ppds.h5')
    return datadict['m1s'], datadict['dRdm1'], datadict['qs'], datadict['dRdq']


def plot_o3b_res(ax, fi, m1=True, col='tab:blue', lab='PP', bounds=False, fill_alpha=0.08):
    plpeak_mpdfs, plpeak_qpdfs, plpeak_ms, plpeak_qs = load_o3b_paper_run_masspdf(paths.data / fi)
    if m1:
        plot_mean_and_90CI(ax, plpeak_ms, plpeak_mpdfs, color=col, label=lab, bounds=bounds, fill_alpha=fill_alpha)
    else:
        plot_mean_and_90CI(ax, plpeak_qs, plpeak_qpdfs, color=col, label=lab, bounds=bounds, fill_alpha=fill_alpha)
    return ax