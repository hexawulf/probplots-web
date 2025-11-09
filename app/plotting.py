import io
import math
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats
from scipy.stats import binom, poisson, norm

def _png_bytes(fig) -> bytes:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=140)
    plt.close(fig)
    return buf.getvalue()

def plot_normal_pdf(mu: float, sigma: float, w=960, h=480) -> bytes:
    x = np.linspace(mu - 4*sigma, mu + 4*sigma, 400)
    y = stats.norm.pdf(x, loc=mu, scale=sigma)
    fig = plt.figure(figsize=(w/96, h/96))
    ax = fig.add_subplot(111)
    ax.plot(x, y, linewidth=2)
    ax.set_title(f"N({mu}, {sigma}^2) PDF")
    ax.grid(True, alpha=0.3)
    return _png_bytes(fig)

def plot_sim_poisson(n: int, lam: float, bins=50, w=960, h=480) -> bytes:
    data = np.random.poisson(lam, size=n)
    fig = plt.figure(figsize=(w/96, h/96))
    ax = fig.add_subplot(111)
    ax.hist(data, bins=bins, density=True)
    xs = np.arange(0, max(10, int(data.max())+1))
    ax.plot(xs, stats.poisson.pmf(xs, lam), linewidth=2)
    ax.set_title(f"Poisson(λ={lam}) simulation (n={n})")
    ax.grid(True, alpha=0.3)
    return _png_bytes(fig)

def plot_sim_binom(n: int, N: int, p: float, bins=50, w=960, h=480) -> bytes:
    data = np.random.binomial(N, p, size=n)
    fig = plt.figure(figsize=(w/96, h/96))
    ax = fig.add_subplot(111)
    ax.hist(data, bins=bins, density=True)
    xs = np.arange(0, N+1)
    ax.plot(xs, stats.binom.pmf(xs, N, p), linewidth=2)
    ax.set_title(f"Binomial(N={N}, p={p}) simulation (n={n})")
    ax.grid(True, alpha=0.3)
    return _png_bytes(fig)

def plot_pmf_poisson(lam: float, kmax: int | None = None) -> bytes:
    if lam <= 0:
        lam = 1e-9
    if kmax is None:
        # cover most mass (lam + ~4*sqrt(lam)), min 10, cap 10k
        kmax = int(max(10, min(10000, math.ceil(lam + 4*math.sqrt(max(lam,1e-9))))))
    ks = np.arange(0, kmax + 1)
    pmf = poisson.pmf(ks, lam)

    fig, ax = plt.subplots(figsize=(6,3))
    ax.stem(ks, pmf, basefmt=" ")
    ax.set_title(f"Poisson PMF (λ={lam:g})")
    ax.set_xlabel("k")
    ax.set_ylabel("P(X=k)")
    ax.grid(True, alpha=0.2)
    return _png_bytes(fig)

def plot_pmf_binom(n: int, p: float) -> bytes:
    ks = np.arange(0, n + 1)
    pmf = binom.pmf(ks, n, p)
    fig, ax = plt.subplots(figsize=(6,3))
    ax.stem(ks, pmf, basefmt=" ")
    ax.set_title(f"Binomial PMF (n={n}, p={p:g})")
    ax.set_xlabel("k")
    ax.set_ylabel("P(X=k)")
    ax.grid(True, alpha=0.2)
    return _png_bytes(fig)

def plot_cdf_binom(n: int, p: float) -> bytes:
    ks = np.arange(0, n + 1)
    cdf = binom.cdf(ks, n, p)
    fig, ax = plt.subplots(figsize=(6,3))
    ax.step(ks, cdf, where="post")
    ax.set_title(f"Binomial CDF (n={n}, p={p:g})")
    ax.set_xlabel("k")
    ax.set_ylabel("P(X≤k)")
    ax.set_ylim(-0.02, 1.02)
    ax.grid(True, alpha=0.2)
    return _png_bytes(fig)

def plot_heat_joint(pairs: dict[tuple[float,float], float]) -> bytes:
    # Build sorted supports
    xs = sorted({x for (x, _) in pairs})
    ys = sorted({y for (_, y) in pairs})
    X = {x:i for i,x in enumerate(xs)}
    Y = {y:i for i,y in enumerate(ys)}
    M = np.zeros((len(ys), len(xs)))  # rows: y, cols: x (for imshow)
    for (x,y),p in pairs.items():
        if p < 0: continue
        M[Y[y], X[x]] += p

    fig, ax = plt.subplots(figsize=(5,4))
    im = ax.imshow(M, origin="lower", cmap="Blues")
    ax.set_title("Joint distribution heatmap  p(x,y)")
    ax.set_xticks(range(len(xs)), xs, rotation=0)
    ax.set_yticks(range(len(ys)), ys)
    fig.colorbar(im, ax=ax, shrink=0.8, label="p")
    return _png_bytes(fig)

def plot_clt(mu: float, sigma: float, n: int, lower: float, upper: float) -> bytes:
    if sigma <= 0 or n <= 0:
        n = max(1, n); sigma = max(1e-9, sigma)
    mu_bar = mu
    sigma_bar = sigma / math.sqrt(n)

    # x range around mean
    xs = np.linspace(mu_bar - 4*sigma_bar, mu_bar + 4*sigma_bar, 600)
    pdf = norm.pdf(xs, loc=mu_bar, scale=sigma_bar)

    fig, ax = plt.subplots(figsize=(6,3.5))
    ax.plot(xs, pdf)
    ax.set_title(f"CLT: X̄ ~ N(μ={mu_bar:g}, σ={sigma_bar:g})")
    ax.set_xlabel("x")
    ax.set_ylabel("density")
    ax.grid(True, alpha=0.2)

    # Shade [lower, upper]
    L, U = min(lower, upper), max(lower, upper)
    mask = (xs>=L) & (xs<=U)
    ax.fill_between(xs[mask], 0, pdf[mask], alpha=0.3)
    return _png_bytes(fig)

def plot_poisson_cdf(lam: float, kmax: int | None = None) -> bytes:
    if lam <= 0:
        lam = 1e-9
    if kmax is None:
        kmax = int(max(10, min(10000, math.ceil(lam + 4*math.sqrt(max(lam,1e-9))))))
    ks = np.arange(0, kmax + 1)
    cdf = poisson.cdf(ks, lam)
    
    fig, ax = plt.subplots(figsize=(6,3))
    ax.step(ks, cdf, where="post")
    ax.set_title(f"Poisson CDF (λ={lam:g})")
    ax.set_xlabel("k")
    ax.set_ylabel("P(X≤k)")
    ax.set_ylim(-0.02, 1.02)
    ax.grid(True, alpha=0.2)
    return _png_bytes(fig)

def plot_geometric_pmf(p: float, kmax: int = 20) -> bytes:
    from scipy.stats import geom
    ks = np.arange(0, kmax + 1)
    pmf = geom.pmf(ks, p)
    
    fig, ax = plt.subplots(figsize=(6,3))
    ax.stem(ks, pmf, basefmt=" ")
    ax.set_title(f"Geometric PMF (p={p:g})")
    ax.set_xlabel("k (failures before success)")
    ax.set_ylabel("P(X=k)")
    ax.grid(True, alpha=0.2)
    return _png_bytes(fig)

def plot_exponential_pdf(lam: float, xmax: float | None = None) -> bytes:
    from scipy.stats import expon
    if lam <= 0:
        lam = 1e-9
    if xmax is None:
        xmax = 5.0 / lam
    xs = np.linspace(0, xmax, 400)
    pdf = expon.pdf(xs, scale=1.0/lam)
    
    fig, ax = plt.subplots(figsize=(6,3))
    ax.plot(xs, pdf, linewidth=2)
    ax.set_title(f"Exponential PDF (λ={lam:g})")
    ax.set_xlabel("x")
    ax.set_ylabel("f(x)")
    ax.grid(True, alpha=0.2)
    return _png_bytes(fig)

def plot_exponential_cdf(lam: float, xmax: float | None = None) -> bytes:
    from scipy.stats import expon
    if lam <= 0:
        lam = 1e-9
    if xmax is None:
        xmax = 5.0 / lam
    xs = np.linspace(0, xmax, 400)
    cdf = expon.cdf(xs, scale=1.0/lam)
    
    fig, ax = plt.subplots(figsize=(6,3))
    ax.plot(xs, cdf, linewidth=2)
    ax.set_title(f"Exponential CDF (λ={lam:g})")
    ax.set_xlabel("x")
    ax.set_ylabel("F(x)")
    ax.set_ylim(-0.02, 1.02)
    ax.grid(True, alpha=0.2)
    return _png_bytes(fig)

def plot_uniform_pdf(a: float, b: float) -> bytes:
    xs = np.linspace(a - (b-a)*0.2, b + (b-a)*0.2, 400)
    ys = np.where((xs >= a) & (xs <= b), 1.0/(b-a), 0.0)
    
    fig, ax = plt.subplots(figsize=(6,3))
    ax.plot(xs, ys, linewidth=2)
    ax.set_title(f"Uniform PDF [a={a:g}, b={b:g}]")
    ax.set_xlabel("x")
    ax.set_ylabel("f(x)")
    ax.grid(True, alpha=0.2)
    return _png_bytes(fig)

def plot_uniform_cdf(a: float, b: float) -> bytes:
    xs = np.linspace(a - (b-a)*0.2, b + (b-a)*0.2, 400)
    ys = np.zeros_like(xs)
    ys[xs < a] = 0.0
    ys[xs > b] = 1.0
    mask = (xs >= a) & (xs <= b)
    ys[mask] = (xs[mask] - a) / (b - a)
    
    fig, ax = plt.subplots(figsize=(6,3))
    ax.plot(xs, ys, linewidth=2)
    ax.set_title(f"Uniform CDF [a={a:g}, b={b:g}]")
    ax.set_xlabel("x")
    ax.set_ylabel("F(x)")
    ax.set_ylim(-0.02, 1.02)
    ax.grid(True, alpha=0.2)
    return _png_bytes(fig)
