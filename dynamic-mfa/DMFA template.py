"""
Generic DMFA Framework
Supports flow-driven and stock-driven analysis with Monte Carlo uncertainty.
"""

import numpy as np
import pandas as pd
from scipy.stats import norm, weibull_min, lognorm
from scipy.optimize import curve_fit, brentq
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------------
# A. Lifetime / Survival module
# ---------------------------------------------------------------------------
class LifetimeModel:
    def __init__(self, dist='normal', params=None, max_age=100):
        """
        dist: 'normal', 'weibull', 'lognormal'
        params: dict, e.g. {'mu': 20, 'sigma': 6}
        """
        self.dist = dist
        self.params = params or {}
        self.max_age = max_age

    def survival(self, ages):
        ages = np.asarray(ages)
        if self.dist == 'normal':
            cdf = norm.cdf(ages, self.params['mu'], self.params['sigma'])
        elif self.dist == 'weibull':
            cdf = weibull_min.cdf(ages, self.params['shape'],
                                  scale=self.params['scale'])
        elif self.dist == 'lognormal':
            cdf = lognorm.cdf(ages, self.params['sigma'],
                              scale=np.exp(self.params['mu']))
        else:
            raise ValueError(f"Unknown distribution {self.dist}")
        sr = 1 - cdf
        sr[ages > self.max_age] = 0
        return sr


# ---------------------------------------------------------------------------
# B. Flow-driven DMFA
# ---------------------------------------------------------------------------
def flow_driven_dmfa(inflow, lifetime: LifetimeModel):
    """
    inflow: 1D array (T,) or 2D array (n_products, T)
    Returns: stock, outflow (same shape as inflow)
    """
    inflow = np.atleast_2d(inflow)
    n, T = inflow.shape
    stock = np.zeros_like(inflow, dtype=float)
    outflow = np.zeros_like(inflow, dtype=float)

    ages = np.arange(T)
    sr = lifetime.survival(ages)  # survival function S(τ)

    for p in range(n):
        for t in range(T):
            # cohort sum: contributions of inflow(i) still surviving at t
            stock[p, t] = np.sum(inflow[p, :t+1] * sr[t::-1][:t+1])
        outflow[p, 0] = inflow[p, 0] - stock[p, 0]
        outflow[p, 1:] = inflow[p, 1:] - np.diff(stock[p, :])

    return stock.squeeze(), outflow.squeeze()


# ---------------------------------------------------------------------------
# C. Stock-driven DMFA
# ---------------------------------------------------------------------------
def logistic(t, K, r, t0):
    return K / (1 + np.exp(-r * (t - t0)))


class StockDrivenDMFA:
    def __init__(self, lifetime: LifetimeModel):
        self.lifetime = lifetime
        self.logistic_params = None

    def fit_logistic(self, years, stock_per_driver):
        """Fit logistic on historical stock-per-capita (or per GDP/cap)."""
        p0 = [np.max(stock_per_driver) * 1.5, 0.1, np.median(years)]
        popt, _ = curve_fit(logistic, years, stock_per_driver, p0=p0, maxfev=10000)
        self.logistic_params = popt
        return popt

    def project_stock(self, years, driver):
        """driver: population or GDP/capita time series for `years`."""
        K, r, t0 = self.logistic_params
        s_per = logistic(np.asarray(years), K, r, t0)
        return s_per * np.asarray(driver)

    def back_calculate_inflow(self, stock):
        """Solve cohort equation for inflow given total stock series."""
        T = len(stock)
        ages = np.arange(T)
        sr = self.lifetime.survival(ages)

        inflow = np.zeros(T)
        for t in range(T):
            # Stock(t) = inflow(t)*sr[0] + Σ_{i<t} inflow(i)*sr[t-i]
            past = np.sum(inflow[:t] * sr[t - np.arange(t)]) if t > 0 else 0.0
            inflow[t] = max((stock[t] - past) / sr[0], 0.0)

        # outflow via mass balance
        outflow = np.zeros(T)
        outflow[0] = inflow[0] - stock[0]
        outflow[1:] = inflow[1:] - np.diff(stock)
        return inflow, outflow


# ---------------------------------------------------------------------------
# D. Monte Carlo engine
# ---------------------------------------------------------------------------
class MonteCarloDMFA:
    def __init__(self, n_iter=1000, seed=42):
        self.n_iter = n_iter
        self.rng = np.random.default_rng(seed)

    def sample_param(self, spec):
        """spec: dict like {'dist':'normal','loc':20,'scale':2}"""
        d = spec['dist']
        if d == 'normal':
            return self.rng.normal(spec['loc'], spec['scale'])
        if d == 'triangular':
            return self.rng.triangular(spec['low'], spec['mode'], spec['high'])
        if d == 'uniform':
            return self.rng.uniform(spec['low'], spec['high'])
        if d == 'lognormal':
            return self.rng.lognormal(spec['mean'], spec['sigma'])
        raise ValueError(d)

    def run_flow_driven(self, inflow_mean, inflow_cv,
                        lifetime_specs, dist='normal'):
        """
        inflow_mean: array of mean inflows
        inflow_cv: coefficient of variation (scalar or array)
        lifetime_specs: dict of param specs, e.g.
            {'mu':{'dist':'normal','loc':20,'scale':2},
             'sigma':{'dist':'normal','loc':6,'scale':1}}
        """
        T = len(inflow_mean)
        stocks = np.zeros((self.n_iter, T))
        outflows = np.zeros((self.n_iter, T))

        for k in range(self.n_iter):
            # sample inflow
            noise = self.rng.normal(1.0, inflow_cv, size=T)
            sampled_inflow = inflow_mean * noise
            # sample lifetime
            params = {k_: self.sample_param(v) for k_, v in lifetime_specs.items()}
            lm = LifetimeModel(dist=dist, params=params)
            s, o = flow_driven_dmfa(sampled_inflow, lm)
            stocks[k] = s
            outflows[k] = o
        return stocks, outflows

    def summarize(self, samples, percentiles=(5, 50, 95)):
        return {f'p{p}': np.percentile(samples, p, axis=0) for p in percentiles}


# ---------------------------------------------------------------------------
# E. Example usage
# ---------------------------------------------------------------------------
if __name__ == '__main__':
    # --- Flow-driven example (consumer durables, analogous to your data [2]) ---
    years = np.arange(1967, 2051)
    demand = np.array([12.43, 14.24, 15.10, 12.38, 13.47, 15.60, 18.69])  # truncated
    demand = np.pad(demand, (0, len(years)-len(demand)), constant_values=demand[-1])

    lm = LifetimeModel('normal', {'mu': 20, 'sigma': 6})
    stock, scrap = flow_driven_dmfa(demand, lm)

    # --- Monte Carlo ---
    mc = MonteCarloDMFA(n_iter=500)
    stocks_mc, scrap_mc = mc.run_flow_driven(
        inflow_mean=demand, inflow_cv=0.10,
        lifetime_specs={
            'mu':    {'dist': 'normal', 'loc': 20, 'scale': 2},
            'sigma': {'dist': 'normal', 'loc': 6,  'scale': 1},
        })
    summary = mc.summarize(stocks_mc)
    print("Median stock 2050:", summary['p50'][-1])

    # --- Stock-driven example ---
    sd = StockDrivenDMFA(LifetimeModel('normal', {'mu': 20, 'sigma': 6}))
    hist_years = np.arange(1990, 2024)
    hist_s_per_cap = 5 / (1 + np.exp(-0.15*(hist_years-2005))) + np.random.normal(0,0.05,len(hist_years))
    sd.fit_logistic(hist_years, hist_s_per_cap)
    pop = np.linspace(1.2e9, 1.4e9, len(years))
    total_stock = sd.project_stock(years, pop)
    inflow_proj, outflow_proj = sd.back_calculate_inflow(total_stock)