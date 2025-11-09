import math
from typing import Optional
import numpy as np
from scipy import stats

def _chk(cond: bool, msg: str):
    if not cond:
        raise ValueError(msg)

def binom_pmf(n: int, p: float, k: int) -> float:
    _chk(n >= 1 and isinstance(n, int), "n must be integer >= 1")
    _chk(0 <= p <= 1, "p must be in [0,1]")
    _chk(0 <= k <= n and isinstance(k, int), "k must be integer in [0,n]")
    return float(stats.binom.pmf(k, n, p))

def binom_cdf(n: int, p: float, k: int) -> float:
    _chk(n >= 1 and isinstance(n, int), "n must be integer >= 1")
    _chk(0 <= p <= 1, "p must be in [0,1]")
    _chk(0 <= k <= n and isinstance(k, int), "k must be integer in [0,n]")
    return float(stats.binom.cdf(k, n, p))

def binom_cdf_range(n: int, p: float, kmin: int, kmax: int) -> float:
    _chk(kmin <= kmax, "Require kmin <= kmax")
    return float(binom_cdf(n, p, kmax) - (binom_cdf(n, p, kmin - 1) if kmin > 0 else 0.0))

def pois_cdf(lam: float, k: int) -> float:
    _chk(lam > 0, "lambda must be > 0")
    _chk(k >= 0 and isinstance(k, int), "k must be integer >= 0")
    return float(stats.poisson.cdf(k, lam))

def geom_pmf(p: float, k: int) -> float:
    # here k = number of failures before first success, support k=0,1,2,...
    _chk(0 < p <= 1, "p must be in (0,1]")
    _chk(k >= 0 and isinstance(k, int), "k must be integer >= 0")
    # P(K=k) = (1-p)^k * p
    return float((1.0 - p)**k * p)

def norm_cdf(x: float, mu: float, sigma: float) -> float:
    _chk(sigma > 0, "sigma must be > 0")
    return float(stats.norm.cdf(x, loc=mu, scale=sigma))

def norm_inv(p: float, mu: float, sigma: float) -> float:
    _chk(0 < p < 1, "p must be in (0,1)")
    _chk(sigma > 0, "sigma must be > 0")
    return float(stats.norm.ppf(p, loc=mu, scale=sigma))

def exp_cdf(lam: float, x: float) -> float:
    _chk(lam > 0, "lambda must be > 0")
    _chk(x >= 0, "x must be >= 0")
    return float(stats.expon(scale=1.0/lam).cdf(x))

def unif_cdf(a: float, b: float, x: float) -> float:
    _chk(b > a, "require b > a")
    return float(stats.uniform(loc=a, scale=b-a).cdf(x))

def clt(mu: float, sigma: float, n: int, lower: float, upper: float) -> float:
    _chk(sigma > 0, "sigma must be > 0")
    _chk(n >= 1 and isinstance(n, int), "n must be integer >= 1")
    if lower > upper:
        lower, upper = upper, lower
    # X̄ ~ Normal(mu, sigma/sqrt(n))
    s = sigma / math.sqrt(n)
    return float(stats.norm.cdf(upper, loc=mu, scale=s) - stats.norm.cdf(lower, loc=mu, scale=s))
