import numpy as np
import paths
import deepdish as dd
import arviz as az
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde as kde
import jax.numpy as jnp
from jax.scipy.special import erf
from TexSoup import TexSoup
from gwinfernodata import GWInfernoData

def load_macro(name):
    soup = TexSoup(open(paths.tex / 'ms.tex'))
    for cmd in soup.find_all('newcommand'):
        if cmd.find(name):
            return cmd.all[-1]
    raise KeyError

def load_03b_posteriors():
    data = GWInfernoData.from_netcdf( paths.data / 'xarray_posterior_samples_and_injections_spin_magnitude.h5')
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

def plot_90CI(ax, xs, ar, color, label, bounds=True, CI=90, traces=None, lw = 5, tracecolor='k', alpha = 0.75, line_style = '-'):
    low = np.percentile(ar, (100-CI)/2., axis=0)
    high = np.percentile(ar, 100-(100-CI)/2., axis=0)
        #ax.plot(xs, low, color='k', lw=0.05, alpha=0.05)
        #ax.plot(xs, high, color='k', lw=0.05, alpha=0.05)

    ax.plot(xs, low, color=color, alpha=alpha, linestyle = line_style, lw = lw)
    ax.plot(xs, high, color=color, alpha=alpha, linestyle = line_style, lw = lw)
    if traces is not None:
        for _ in range(traces):
            idx = np.random.choice(ar.shape[0])
            ax.plot(xs, ar[idx], color=tracecolor, lw=0.025, alpha=0.02)  
    return ax

def load_bsplinemass_ppd():
    datadict = dd.io.load(paths.data / 'bsplines_64m1_18q_iid18mag_iid18tilt_pl18z_ppds.h5')
    return datadict['m1s'], datadict['dRdm1'], datadict['qs'], datadict['dRdq']

# def load_subpop_ppds(g1 = False, g2 = False, g1_fname = 'bspline_1logpeak_ppds.h5', g2_fname = 'bspline_1logpeak_samespin_ppds.h5', N=2000):
#     static_keys = ['a1', 'cos_tilt_1', 'mass_1', 'mass_ratio', 'a2', 'cos_tilt_2']
#     if g1:
#         datadict = dd.io.load(paths.data/ g1_fname)
#         if N < len(datadict['peak_1_mass_pdfs']):
#             for k in datadict.keys():
#                 if k not in static_keys:
#                     if type(datadict[k]) == dict:
#                         for kk in datadict[k].keys():
#                             idxs = np.random.choice(datadict[k][kk].shape[0], N, replace=False)
#                             datadict[k][kk] = datadict[k][kk][idxs]
#                     else:
#                         idxs = np.random.choice(datadict[k].shape[0], N, replace=False)
#                         datadict[k] = datadict[k][idxs]
#         return datadict
#     elif g2:
#         datadict = dd.io.load(paths.data/ g2_fname)
#         if N < len(datadict['peak_1_mass_pdfs']):
#             for k in datadict.keys():
#                 if k not in static_keys:
#                     if type(datadict[k]) == dict:
#                         for kk in datadict[k].keys():
#                             idxs = np.random.choice(datadict[k][kk].shape[0], N, replace=False)
#                             datadict[k][kk] = datadict[k][kk][idxs]
#                     else:
#                         idxs = np.random.choice(datadict[k].shape[0], N, replace=False)
#                         datadict[k] = datadict[k][idxs]
#         return datadict

def load_subpop_ppds(g1 = False, g2 = False, g1_fname = 'bspline_1logpeak_ppds.h5', g2_fname = 'bspline_1logpeak_samespin_ppds.h5'):
    if g1:
        datadict = dd.io.load(paths.data/ g1_fname)
        return datadict
    elif g2:
        datadict = dd.io.load(paths.data/ g2_fname)
        return datadict

def load_trace(g1 = False, g2 = False, g1_fname = 'bspline_1logpeak_10000s.h5', g2_fname = 'bspline_1logpeak_samespin_10000s.h5'):
    if g1:
        trace = az.from_netcdf(paths.data/g1_fname)
        return trace
    elif g2:
        trace = az.from_netcdf(paths.data/g2_fname)
        return trace

def plot_o3b_res(ax, fi, m1=True, col='tab:blue', lab='PP', bounds=False, fill_alpha=0.08, **kwargs):
    plpeak_mpdfs, plpeak_qpdfs, plpeak_ms, plpeak_qs = load_o3b_paper_run_masspdf(paths.data / fi)
    if m1:
        plot_mean_and_90CI(ax, plpeak_ms, plpeak_mpdfs, color=col, label=lab, bounds=bounds, fill_alpha=fill_alpha, mean = kwargs['mean'], median = kwargs['median'])
    else:
        plot_mean_and_90CI(ax, plpeak_qs, plpeak_qpdfs, color=col, label=lab, bounds=bounds, fill_alpha=fill_alpha, mean = kwargs['mean'], median = kwargs['median'])
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

def truncnorm_pdf(xx, mu, sig, low, high, log = False):
    """
    $$ p(x) \propto \mathcal{N}(x | \mu, \sigma)\Theta(x-x_\mathrm{min})\Theta(x_\mathrm{max}-x) $$
    """

    if log:
        prob = jnp.exp(-jnp.power(jnp.log(xx) - mu, 2) / (2 * sig**2))
        continuous_norm = 1 / (xx * sig * (2*jnp.pi)**0.5)
        left_tail_cdf = 0.5 * ( 1 + erf((jnp.log(low) - mu) / (sig * (2**0.5))))
        right_tail_cdf = 0.5 * ( 1 + erf((jnp.log(high) - mu) / (sig * (2**0.5))))
        denom = right_tail_cdf - left_tail_cdf
    else:
        prob = jnp.exp(-jnp.power(xx - mu, 2) / (2 * sig**2))
        continuous_norm = 1 / (sig * (2*jnp.pi)**0.5)
        left_tail_cdf = 0.5 * ( 1 + erf((low - mu) / (sig * (2**0.5))))
        right_tail_cdf = 0.5 * ( 1 + erf((high - mu) / (sig * (2**0.5))))
        denom = right_tail_cdf - left_tail_cdf

    norm = continuous_norm / denom
    return jnp.where(jnp.greater(xx, high) | jnp.less(xx, low), 0, prob * norm)

class BasisSpline(object):
    def __init__(
        self,
        n_df,
        knots=None,
        interior_knots=None,
        xrange=(0, 1),
        k=4,
        proper=True,
        normalize=True,
    ):
        self.order = k
        self.N = n_df
        self.xrange = xrange
        if knots is None:
            if interior_knots is None:
                interior_knots = np.linspace(*xrange, n_df - k + 2)
            if proper:
                dx = interior_knots[1] - interior_knots[0]
                knots = np.concatenate(
                    [
                        xrange[0] - dx * np.arange(1, k)[::-1],
                        interior_knots,
                        xrange[1] + dx * np.arange(1, k),
                    ]
                )
            else:
                knots = np.append(
                    np.append(np.array([xrange[0]] * (k - 1)), interior_knots),
                    np.array([xrange[1]] * (k - 1)),
                )
        self.knots = knots
        self.interior_knots = knots
        assert len(self.knots) == self.N + self.order

        self.normalize = normalize
        self.basis_vols = np.ones(self.N)
        if normalize:
            grid = jnp.linspace(*xrange, 1000)
            grid_bases = jnp.array(self.bases(grid))
            self.basis_vols = jnp.array([jnp.trapz(grid_bases[i, :], grid) for i in range(self.N)])

    def norm(self, coefs):
        n = 1.0 / jnp.sum(self.basis_vols * coefs.flatten()) if self.normalize else 1.0
        return n

    def _basis(self, xs, i, k):
        if self.knots[i + k] - self.knots[i] < 1e-6:
            return np.zeros_like(xs)
        elif k == 1:
            v = np.zeros_like(xs)
            v[(xs >= self.knots[i]) & (xs < self.knots[i + 1])] = 1 / (self.knots[i + 1] - self.knots[i])
            return v
        else:
            v = (xs - self.knots[i]) * self._basis(xs, i, k - 1) + (self.knots[i + k] - xs) * self._basis(xs, i + 1, k - 1)
            return (v * k) / ((k - 1) * (self.knots[i + k] - self.knots[i]))

    def _bases(self, xs):
        return [self._basis(xs, i, k=self.order) for i in range(self.N)]

    def bases(self, xs):
        return jnp.concatenate(self._bases(xs)).reshape(self.N, *xs.shape)

    def project(self, bases, coefs):
        coefs /= jnp.sum(coefs)
        return jnp.einsum("i...,i->...", bases, coefs) * self.norm(coefs)

    def eval(self, xs, coefs):
        return self.project(self.bases(xs), coefs)

    def __call__(self, xs, coefs):
        return self.eval(xs, coefs)


class BSpline(BasisSpline):
    def __init__(
        self,
        n_df,
        knots=None,
        interior_knots=None,
        xrange=(0, 1),
        k=4,
        proper=True,
        normalize=False,
    ):
        super().__init__(
            n_df=n_df,
            knots=knots,
            interior_knots=interior_knots,
            xrange=xrange,
            k=k,
            proper=proper,
            normalize=normalize,
        )

    def _bases(self, xs):
        return [(self.knots[i + self.order] - self.knots[i]) / self.order * self._basis(xs, i, k=self.order) for i in range(self.N)]

    def project(self, bases, coefs):
        return jnp.einsum("i...,i->...", bases, coefs) * self.norm(coefs)


class LogXBSpline(BSpline):
    def __init__(self, n_df, knots=None, interior_knots=None, xrange=(0.01, 1), normalize=True, **kwargs):
        knots = None if knots is None else np.log(knots)
        interior_knots = None if interior_knots is None else np.log(interior_knots)
        xrange = np.log(xrange)
        super().__init__(n_df, knots=knots, interior_knots=interior_knots, xrange=xrange, **kwargs)

        self.normalize = normalize
        self.basis_vols = np.ones(self.N)
        if normalize:
            self.grid = jnp.linspace(*np.exp(xrange), 1000)
            self.grid_bases = jnp.array(self.bases(self.grid))

    def bases(self, xs):
        return super().bases(jnp.log(xs))


class LogYBSpline(BSpline):
    def __init__(self, n_df, knots=None, interior_knots=None, xrange=(0, 1), normalize=True, **kwargs):
        super().__init__(n_df, knots=knots, interior_knots=interior_knots, xrange=xrange, **kwargs)
        self.normalize = normalize
        if normalize:
            self.grid = jnp.linspace(*xrange, 1000)
            self.grid_bases = jnp.array(self.bases(self.grid))

    def _project(self, bases, coefs):
        return jnp.exp(jnp.einsum("i...,i->...", bases, coefs))

    def project(self, bases, coefs):
        return self._project(bases, coefs) * self.norm(coefs)

    def norm(self, coefs):
        n = 1.0 / jnp.trapz(self._project(self.grid_bases, coefs), self.grid) if self.normalize else 1.0
        return n


class LogXLogYBSpline(LogYBSpline):
    def __init__(self, n_df, knots=None, interior_knots=None, xrange=(0.1, 1), normalize=True, **kwargs):
        knots = None if knots is None else np.log(knots)
        interior_knots = None if interior_knots is None else np.log(interior_knots)
        xrange = np.log(xrange)
        super().__init__(n_df, knots=knots, interior_knots=interior_knots, xrange=xrange, **kwargs)

        self.normalize = normalize
        self.basis_vols = np.ones(self.N)
        if normalize:
            self.grid = jnp.linspace(*jnp.exp(xrange), 1500)
            self.grid_bases = jnp.array(self.bases(self.grid))

    def bases(self, xs):
        return super().bases(jnp.log(xs))

    def project(self, bases, coefs):
        return self._project(bases, coefs) * self.norm(coefs)



class Base1DBSplineModel(object):
    def __init__(
        self,
        nknots,
        xx,
        xx_inj,
        knots=None,
        xrange=(0, 1),
        order=3,
        prefix="c",
        domain="x",
        basis=BSpline,
        **kwargs,
    ):
        self.nknots = nknots
        self.domain = domain
        self.xmin, self.xmax = xrange
        self.order = order
        self.prefix = prefix
        self.interpolator = basis(
            nknots,
            knots=knots,
            xrange=xrange,
            k=order + 1,
            **kwargs,
        )
        self.variable_names = [f"{self.prefix}{i}" for i in range(self.nknots)]
        self.pe_design_matrix = jnp.array(self.truncate_dmat(xx, self.interpolator.bases(xx)))
        self.inj_design_matrix = jnp.array(self.truncate_dmat(xx_inj, self.interpolator.bases(xx_inj)))
        self.funcs = [self.inj_pdf, self.pe_pdf]

    def truncate_dmat(self, x, dmat):
        return jnp.where(jnp.less(x, self.xmin) | jnp.greater(x, self.xmax), 0, dmat)

    def eval_spline(self, bases, coefs):
        return self.interpolator.project(bases, coefs)

    def pe_pdf(self, coefs):
        return self.eval_spline(self.pe_design_matrix, coefs)

    def inj_pdf(self, coefs):
        return self.eval_spline(self.inj_design_matrix, coefs)

    def __call__(self, ndim, coefs):
        return self.funcs[ndim - 1](coefs)




class BSplineMass(Base1DBSplineModel):
    def __init__(
        self,
        nknots,
        m,
        m_inj,
        knots=None,
        mmin=2,
        mmax=100,
        order=3,
        prefix="f",
        domain="mass",
        **kwargs,
    ):
        super().__init__(
            nknots,
            m,
            m_inj,
            knots=knots,
            xrange=(mmin, mmax),
            order=order,
            prefix=prefix,
            domain=domain,
            **kwargs,
        )


class BSplineRatio(Base1DBSplineModel):
    def __init__(
        self,
        nknots,
        q,
        q_inj,
        qmin=0,
        knots=None,
        order=3,
        prefix="u",
        **kwargs,
    ):
        super().__init__(
            nknots,
            q,
            q_inj,
            knots=knots,
            order=order,
            prefix=prefix,
            xrange=(qmin, 1),
            domain="mass_ratio",
            **kwargs,
        )

class BSplineSpinTilt(Base1DBSplineModel):
    def __init__(
        self,
        nknots,
        ct,
        ct_inj,
        knots=None,
        order=3,
        prefix="x",
        domain="cos_tilt",
        **kwargs,
    ):
        super().__init__(
            nknots,
            ct,
            ct_inj,
            knots=knots,
            order=order,
            prefix=prefix,
            domain=domain,
            xrange=(-1, 1),
            **kwargs,
        )

class BSplineSpinMagnitude(Base1DBSplineModel):
    def __init__(
        self,
        nknots,
        a,
        a_inj,
        knots=None,
        order=3,
        prefix="c",
        domain="a",
        **kwargs,
    ):
        super().__init__(
            nknots,
            a,
            a_inj,
            knots=knots,
            order=order,
            prefix=prefix,
            domain=domain,
            **kwargs,
        )



class BSplinePrimaryBSplineRatio(object):
    def __init__(
        self,
        nknots_m,
        nknots_q,
        m1,
        m1_inj,
        q,
        q_inj,
        knots_m=None,
        knots_q=None,
        order_m=3,
        order_q=3,
        prefix_m="c",
        prefix_q="q",
        m1min=3.0,
        m2min=3.0,
        mmax=100.0,
        basis_m=BSpline,
        basis_q=BSpline,
        **kwargs,
    ):
        self.primary_model = BSplineMass(
            nknots_m,
            m1,
            m1_inj,
            knots=knots_m,
            mmin=m1min,
            mmax=mmax,
            order=order_m,
            prefix=prefix_m,
            domain="mass_1",
            basis=basis_m,
            **kwargs,
        )
        self.ratio_model = BSplineRatio(
            nknots_q,
            q,
            q_inj,
            qmin=m2min / mmax,
            knots=knots_q,
            order=order_q,
            prefix=prefix_q,
            basis=basis_q,
            **kwargs,
        )

    def __call__(self, ndim, mcoefs, qcoefs):
        return self.ratio_model(ndim, qcoefs) * self.primary_model(ndim, mcoefs)


class BSplineIIDSpinTilts(object):
    def __init__(
        self,
        nknots,
        ct1,
        ct2,
        ct1_inj,
        ct2_inj,
        knots=None,
        order=3,
        prefix="x",
        **kwargs,
    ):
        self.primary_model = BSplineSpinTilt(
            nknots=nknots,
            ct=ct1,
            ct_inj=ct1_inj,
            knots=knots,
            order=order,
            prefix=prefix,
            domain="cos_tilt_1",
            **kwargs,
        )
        self.secondary_model = BSplineSpinTilt(
            nknots=nknots,
            ct=ct2,
            ct_inj=ct2_inj,
            knots=knots,
            order=order,
            prefix=prefix,
            domain="cos_tilt_2",
            **kwargs,
        )

    def __call__(self, ndim, coefs):
        p_ct1 = self.primary_model(ndim, coefs)
        p_ct2 = self.secondary_model(ndim, coefs)
        return p_ct1 * p_ct2


class BSplineIIDSpinMagnitudes(object):
    def __init__(
        self,
        nknots,
        a1,
        a2,
        a1_inj,
        a2_inj,
        knots=None,
        order=3,
        prefix="c",
        **kwargs,
    ):
        self.primary_model = BSplineSpinMagnitude(
            nknots=nknots,
            a=a1,
            a_inj=a1_inj,
            knots=knots,
            order=order,
            prefix=prefix,
            domain="a_1",
            **kwargs,
        )
        self.secondary_model = BSplineSpinMagnitude(
            nknots=nknots,
            a=a2,
            a_inj=a2_inj,
            knots=knots,
            order=order,
            prefix=prefix,
            domain="a_2",
            **kwargs,
        )

    def __call__(self, ndim, coefs):
        p_a1 = self.primary_model(ndim, coefs)
        p_a2 = self.secondary_model(ndim, coefs)
        return p_a1 * p_a2


class BoundedKDE(kde):
    """Base class to handle the BoundedKDE
    Parameters
    ----------
    pts: np.ndarray
        The datapoints to estimate a bounded kde from
    xlow: float
        The lower bound of the distribution
    xhigh: float
        The upper bound of the distribution
    """
    def __init__(self, pts, xlow=None, xhigh=None, *args, **kwargs):
        pts = np.atleast_1d(pts)
        if pts.ndim != 1:
            raise TypeError("BoundedKDE can only be one-dimensional")
        super(BoundedKDE, self).__init__(pts.T, *args, **kwargs)
        self._xlow = xlow
        self._xhigh = xhigh

    @property
    def xlow(self):
        """The lower bound of the x domain
        """
        return self._xlow

    @property
    def xhigh(self):
        """The upper bound of the x domain
        """
        return self._xhigh


class ReflectionBoundedKDE(BoundedKDE):
    """Represents a one-dimensional Gaussian kernel density estimator
    for a probability distribution function that exists on a bounded
    domain. The bounds are treated as reflections
    Parameters
    ----------
    pts: np.ndarray
        The datapoints to estimate a bounded kde from
    xlow: float
        The lower bound of the distribution
    xhigh: float
        The upper bound of the distribution
    """
    def __init__(self, pts, xlow=None, xhigh=None, *args, **kwargs):
        super(ReflectionBoundedKDE, self).__init__(
            pts, xlow=xlow, xhigh=xhigh, *args, **kwargs
        )

    def evaluate(self, pts):
        """Return an estimate of the density evaluated at the given
        points
        """
        x = pts.T
        pdf = super(ReflectionBoundedKDE, self).evaluate(pts.T)
        if self.xlow is not None:
            pdf += super(ReflectionBoundedKDE, self).evaluate(2 * self.xlow - x)
        if self.xhigh is not None:
            pdf += super(ReflectionBoundedKDE, self).evaluate(2 * self.xhigh - x)
        return pdf

    def __call__(self, pts):
        pts = np.atleast_1d(pts)
        out_of_bounds = np.zeros(pts.shape[0], dtype='bool')

        if self.xlow is not None:
            out_of_bounds[pts < self.xlow] = True
        if self.xhigh is not None:
            out_of_bounds[pts > self.xhigh] = True

        results = self.evaluate(pts)
        results[out_of_bounds] = 0.
        return results