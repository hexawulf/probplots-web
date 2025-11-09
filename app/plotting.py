import io
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats

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
