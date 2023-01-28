import numpy as np
import paths
import deepdish as dd
import arviz as az
import matplotlib.pyplot as plt


def load_03b_posteriors():
    data = dd.io.load( paths.data / 'posterior_samples_and_injections_spin_magnitude.h5')
    return data

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

def plot_mean_and_90CI(ax, xs, ar, color, label, bounds=True, CI=90, traces=None, lw = 5, tracecolor='k', alpha = 0.75, line_style = '-', fill_alpha=0.08, median=False, mean=True):

    if mean:
        me = np.mean(ar, axis=0)    
        ax.plot(xs, me, color=color, label=label, lw=lw, alpha=alpha, linestyle = line_style)
    elif median:
        me = np.median(ar,axis=0)
        ax.plot(xs, me, color=color, label=label, lw=lw, alpha=alpha, linestyle = line_style)
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

def load_subpop_ppds():
    datadict = dd.io.load(paths.data/ 'PPDS_bspline_mass_spin_1000w_10000s_thin2_independent_bspline_ratio_sigprior02_12-15-22.h5')
    return datadict

def load_trace():
    trace = az.from_netcdf(paths.data/'bspline_mass_spin_1000w_10000s_thin2_independent_bspline_ratio_reweighedKDEs_12-16-22.h5')
    return trace

def plot_o3b_res(ax, fi, m1=True, col='tab:blue', lab='PP', bounds=False, fill_alpha=0.08):
    plpeak_mpdfs, plpeak_qpdfs, plpeak_ms, plpeak_qs = load_o3b_paper_run_masspdf(paths.data / fi)
    if m1:
        plot_mean_and_90CI(ax, plpeak_ms, plpeak_mpdfs, color=col, label=lab, bounds=bounds, fill_alpha=fill_alpha)
    else:
        plot_mean_and_90CI(ax, plpeak_qs, plpeak_qpdfs, color=col, label=lab, bounds=bounds, fill_alpha=fill_alpha)
    return ax

def load_iid_tilt_ppd():
    datadict = dd.io.load(paths.data / 'bsplines_64m1_18q_iid18mag_iid18tilt_pl18z_ppds.h5')
    xs = datadict['tilts']
    dRdct = datadict['dRdct']
    return xs, dRdct

def plot_o3b_spintilt(ax, fi,ct1=False, col='tab:blue', lab='PP'):
    xs = np.linspace(-1, 1, 1000)
    _data = dd.io.load(paths.data / fi)
    lines = dict()
    for key in _data["lines"].keys():
        lines[key] = _data["lines"][key][()]
        for ii in range(len(lines[key])):
            lines[key][ii] /= np.trapz(lines[key][ii], xs)
    if ct1:
        ax = plot_mean_and_90CI(ax, xs, lines['cos_tilt_1'], color=col, label=lab, bounds=False)
    else:
        ax = plot_mean_and_90CI(ax, xs, lines['cos_tilt_2'], color=col, label=lab, bounds=False)
    return ax

def load_iid_mag_ppd():
    datadict = dd.io.load(paths.data / 'bsplines_64m1_18q_iid18mag_iid18tilt_pl18z_ppds.h5')
    return datadict['mags'], datadict['dRda']

def plot_o3b_spinmag(ax, fi, a1=True, col='tab:blue', lab='PP'):
    xs = np.linspace(0, 1, 1000)
    _data = dd.io.load(paths.data / fi)
    lines = dict()
    for key in _data["lines"].keys():
        lines[key] = _data["lines"][key][()]
        for ii in range(len(lines[key])):
            lines[key][ii] /= np.trapz(lines[key][ii], xs)
    if a1:
        ax = plot_mean_and_90CI(ax, xs, lines['a_1'], color=col, label=lab, bounds=False)
    else:
        ax = plot_mean_and_90CI(ax, xs, lines['a_2'], color=col, label=lab, bounds=False)
    return ax

def radar_plot(categories, events, ax, cm = None):
    categories = [*categories, categories[0]]
    events = np.column_stack((events, events[:,0]))
    label_loc = np.linspace(start=0, stop=2 * np.pi, num=len(categories))
    for i in range(len(events)):
        ax.plot(label_loc, events[i], color = cm[:][i], alpha = 0.5)
    lines, labels = plt.thetagrids(np.degrees(label_loc), labels=categories, fontsize = 10)