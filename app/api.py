from fastapi import FastAPI, Query, HTTPException, Response
from fastapi.responses import JSONResponse, HTMLResponse

# numeric functions
from .dlbdss_core import (
    binom_pmf, binom_cdf, binom_cdf_range, pois_cdf, geom_pmf,
    norm_cdf, norm_inv, exp_cdf, unif_cdf, clt
)
# plotting
from .plotting import (
    plot_normal_pdf, plot_sim_poisson, plot_sim_binom,
    plot_pmf_poisson, plot_pmf_binom, plot_cdf_binom,
    plot_heat_joint, plot_clt, plot_poisson_cdf,
    plot_geometric_pmf, plot_exponential_pdf, plot_exponential_cdf,
    plot_uniform_pdf, plot_uniform_cdf
)

# scipy for direct PMF/CDF access
from scipy.stats import poisson, binom

# ===== Local helper import with sandbox stub =====
import os, sys
sys.path.insert(0, "/home/zk/projects/probplots")
try:
    from stats_helper import compute_joint_stats  # real implementation on the Pi
except Exception:
    # Optional sandbox stub so remote runners can start the server without your local repo
    def compute_joint_stats(joint):
        if os.environ.get("PROBPLOTS_STUB") == "1":
            xs = sorted({x for (x, _) in joint})
            ys = sorted({y for (_, y) in joint})
            return {
                "support_X": xs, "support_Y": ys,
                "joint": [{"x": x, "y": y, "p": p} for (x, y), p in sorted(joint.items())],
                "marginal_X": {}, "marginal_Y": {}, "independent": False,
                "E": {"EX": 0.0, "EY": 0.0}, "Var": {"VarX": 0.0, "VarY": 0.0},
                "Cov": 0.0, "Corr": None
            }
        raise

app = FastAPI(title="probplots-web", version="0.2")

@app.get("/about")
def about():
    return {
        "app": "probplots-web",
        "version": "0.2",
        "bind": "127.0.0.1:5009"
    }


def _bad(e: Exception):
    raise HTTPException(status_code=400, detail=str(e))

@app.get("/healthz")
def healthz():
    return {"status": "ok"}

# ---- numeric endpoints ----
@app.get("/api/norm/cdf")
def api_norm_cdf(x: float, mu: float = 0.0, sigma: float = 1.0):
    try: return {"value": norm_cdf(x, mu, sigma)}
    except Exception as e: _bad(e)

@app.get("/api/norm/inv")
def api_norm_inv(p: float, mu: float = 0.0, sigma: float = 1.0):
    try: return {"value": norm_inv(p, mu, sigma)}
    except Exception as e: _bad(e)

@app.get("/api/binom/pmf")
def api_binom_pmf(n: int, p: float, k: int):
    try: return {"value": binom_pmf(n, p, k)}
    except Exception as e: _bad(e)

@app.get("/api/binom/cdf")
def api_binom_cdf(n: int, p: float, k: int):
    try: return {"value": binom_cdf(n, p, k)}
    except Exception as e: _bad(e)

@app.get("/api/binom/cdf-range")
def api_binom_cdf_range(n: int, p: float, kmin: int, kmax: int):
    try: return {"value": binom_cdf_range(n, p, kmin, kmax)}
    except Exception as e: _bad(e)

@app.get("/api/pois/cdf")
def api_pois_cdf(lam: float, k: int):
    try: return {"value": pois_cdf(lam, k)}
    except Exception as e: _bad(e)

@app.get("/api/geom/pmf")
def api_geom_pmf(p: float, k: int):
    try: return {"value": geom_pmf(p, k)}
    except Exception as e: _bad(e)

@app.get("/api/exp/cdf")
def api_exp_cdf(lam: float, x: float):
    try: return {"value": exp_cdf(lam, x)}
    except Exception as e: _bad(e)

@app.get("/api/unif/cdf")
def api_unif_cdf(a: float, b: float, x: float):
    try: return {"value": unif_cdf(a, b, x)}
    except Exception as e: _bad(e)

@app.get("/api/clt")
def api_clt(mu: float, sigma: float, n: int, lower: float, upper: float):
    try: return {"value": clt(mu, sigma, n, lower, upper)}
    except Exception as e: _bad(e)

# ---- validation helpers ----
def _ensure_prob(p: float, name="p"):
    if not (0.0 <= p <= 1.0):
        raise HTTPException(status_code=400, detail=f"{name} must be in [0,1]")

def _ensure_nonneg_int(x: int, name="k"):
    if x < 0:
        raise HTTPException(status_code=400, detail=f"{name} must be ≥ 0")

def _ensure_pos(x: float, name="value"):
    if x <= 0:
        raise HTTPException(status_code=400, detail=f"{name} must be > 0")

# ---- poisson PMF ----
@app.get("/api/pois/pmf")
def pois_pmf(
    lam: float = Query(..., description="λ > 0"),
    k: int = Query(..., ge=0, description="k ≥ 0"),
):
    _ensure_pos(lam, "lam"); _ensure_nonneg_int(k, "k")
    val = float(poisson.pmf(k, lam))
    return {"value": val}

# ---- binomial shortcuts ----
@app.get("/api/binom/atmost")
def binom_atmost(
    n: int = Query(..., ge=0),
    p: float = Query(..., description="0≤p≤1"),
    k: int = Query(..., ge=0),
):
    _ensure_prob(p); _ensure_nonneg_int(k, "k")
    # P(X ≤ k)
    val = float(binom.cdf(k, n, p))
    return {"value": val}

@app.get("/api/binom/atleast")
def binom_atleast(
    n: int = Query(..., ge=0),
    p: float = Query(..., description="0≤p≤1"),
    k: int = Query(..., ge=0),
):
    _ensure_prob(p); _ensure_nonneg_int(k, "k")
    # P(X ≥ k) = 1 - P(X ≤ k-1)
    val = float(1.0 - binom.cdf(k - 1, n, p)) if k > 0 else 1.0
    return {"value": val}

@app.get("/api/binom/between")
def binom_between(
    n: int = Query(..., ge=0),
    p: float = Query(..., description="0≤p≤1"),
    kmin: int = Query(..., ge=0),
    kmax: int = Query(..., ge=0),
):
    _ensure_prob(p); _ensure_nonneg_int(kmin, "kmin"); _ensure_nonneg_int(kmax, "kmax")
    if kmin > kmax:
        raise HTTPException(status_code=400, detail="kmin must be ≤ kmax")
    # P(kmin ≤ X ≤ kmax) = CDF(kmax) - CDF(kmin-1)
    left = float(binom.cdf(kmax, n, p))
    right = float(binom.cdf(kmin - 1, n, p)) if kmin > 0 else 0.0
    return {"value": left - right}

# ---- joint endpoint ----
@app.get("/api/joint")
def api_joint(pairs: str = Query(..., description="semicolon-separated triples x,y,p (e.g. '0,0,0.1;1,1,0.4')")):
    try:
        items = [t for t in pairs.strip().split(";") if t]
        joint = {}
        total = 0.0
        for item in items:
            xs, ys, ps = item.split(",")
            x, y, p = float(xs), float(ys), float(ps)
            if p < 0: raise ValueError("probabilities must be >= 0")
            joint[(x, y)] = p
            total += p
        if abs(total - 1.0) > 1e-10:
            raise ValueError(f"probabilities must sum to 1 (got {total:.12g})")
        return JSONResponse(compute_joint_stats(joint))
    except Exception as e:
        _bad(e)

# ---- plotting endpoints ----
@app.get("/plot/pdf/normal")
def http_plot_normal(mu: float = 0.0, sigma: float = 1.0):
    try:
        png = plot_normal_pdf(mu, sigma)
        return Response(content=png, media_type="image/png")
    except Exception as e:
        _bad(e)

@app.get("/plot/sim/poisson")
def http_plot_sim_pois(n: int = 50000, lam: float = 3.0, bins: int = 50):
    try:
        png = plot_sim_poisson(n, lam, bins=bins)
        return Response(content=png, media_type="image/png")
    except Exception as e:
        _bad(e)

@app.get("/plot/sim/binom")
def http_plot_sim_binom(n: int = 50000, N: int = 20, p: float = 0.3, bins: int = 50):
    try:
        png = plot_sim_binom(n, N, p, bins=bins)
        return Response(content=png, media_type="image/png")
    except Exception as e:
        _bad(e)

@app.get("/plot/pmf/poisson")
def plot_pmf_poisson_ep(
    lam: float = Query(..., gt=0),
    kmax: int | None = Query(None, ge=0),
):
    try:
        png = plot_pmf_poisson(lam, kmax)
        return Response(content=png, media_type="image/png")
    except Exception as e:
        _bad(e)

@app.get("/plot/pmf/binom")
def plot_pmf_binom_ep(
    n: int = Query(..., ge=0),
    p: float = Query(..., ge=0, le=1),
):
    try:
        png = plot_pmf_binom(n, p)
        return Response(content=png, media_type="image/png")
    except Exception as e:
        _bad(e)

@app.get("/plot/cdf/binom")
def plot_cdf_binom_ep(
    n: int = Query(..., ge=0),
    p: float = Query(..., ge=0, le=1),
):
    try:
        png = plot_cdf_binom(n, p)
        return Response(content=png, media_type="image/png")
    except Exception as e:
        _bad(e)

@app.get("/plot/heat/joint")
def plot_heat_joint_ep(
    pairs: str = Query(..., description="x,y,p;..."),
):
    try:
        # Parse same as /api/joint
        def _parse_pairs(s: str):
            out = {}
            for chunk in s.split(";"):
                if not chunk.strip(): continue
                x,y,p = map(float, chunk.split(","))
                out[(x,y)] = out.get((x,y), 0.0) + p
            return out
        joint = _parse_pairs(pairs)
        png = plot_heat_joint(joint)
        return Response(content=png, media_type="image/png")
    except Exception as e:
        _bad(e)

@app.get("/plot/clt")
def plot_clt_ep(
    mu: float, sigma: float, n: int, lower: float, upper: float
):
    try:
        png = plot_clt(mu, sigma, n, lower, upper)
        return Response(content=png, media_type="image/png")
    except Exception as e:
        _bad(e)

# ---- NEW: Universal /plot/png/* endpoints ----
@app.get("/plot/png/normal")
def png_normal(mu: float = 0.0, sigma: float = 1.0):
    try:
        return Response(content=plot_normal_pdf(mu, sigma), media_type="image/png")
    except Exception as e:
        _bad(e)

@app.get("/plot/png/binomial_pmf")
def png_binomial_pmf(n: int = Query(..., ge=0), p: float = Query(..., ge=0, le=1)):
    try:
        return Response(content=plot_pmf_binom(n, p), media_type="image/png")
    except Exception as e:
        _bad(e)

@app.get("/plot/png/binomial_cdf")
def png_binomial_cdf(n: int = Query(..., ge=0), p: float = Query(..., ge=0, le=1)):
    try:
        return Response(content=plot_cdf_binom(n, p), media_type="image/png")
    except Exception as e:
        _bad(e)

@app.get("/plot/png/binomial_sim")
def png_binomial_sim(n: int = 50000, N: int = 20, p: float = 0.3, bins: int = 50):
    try:
        return Response(content=plot_sim_binom(n, N, p, bins=bins), media_type="image/png")
    except Exception as e:
        _bad(e)

@app.get("/plot/png/poisson_pmf")
def png_poisson_pmf(lam: float = Query(..., gt=0), kmax: int | None = Query(None, ge=0)):
    try:
        return Response(content=plot_pmf_poisson(lam, kmax), media_type="image/png")
    except Exception as e:
        _bad(e)

@app.get("/plot/png/poisson_cdf")
def png_poisson_cdf(lam: float = Query(..., gt=0), kmax: int | None = Query(None, ge=0)):
    try:
        return Response(content=plot_poisson_cdf(lam, kmax), media_type="image/png")
    except Exception as e:
        _bad(e)

@app.get("/plot/png/poisson_sim")
def png_poisson_sim(n: int = 50000, lam: float = 3.0, bins: int = 50):
    try:
        return Response(content=plot_sim_poisson(n, lam, bins=bins), media_type="image/png")
    except Exception as e:
        _bad(e)

@app.get("/plot/png/geometric_pmf")
def png_geometric_pmf(p: float = Query(..., ge=0, le=1), kmax: int = 20):
    try:
        return Response(content=plot_geometric_pmf(p, kmax), media_type="image/png")
    except Exception as e:
        _bad(e)

@app.get("/plot/png/exponential_pdf")
def png_exponential_pdf(lam: float = Query(..., gt=0), xmax: float | None = None):
    try:
        return Response(content=plot_exponential_pdf(lam, xmax), media_type="image/png")
    except Exception as e:
        _bad(e)

@app.get("/plot/png/exponential_cdf")
def png_exponential_cdf(lam: float = Query(..., gt=0), xmax: float | None = None):
    try:
        return Response(content=plot_exponential_cdf(lam, xmax), media_type="image/png")
    except Exception as e:
        _bad(e)

@app.get("/plot/png/uniform_pdf")
def png_uniform_pdf(a: float = 0.0, b: float = 10.0):
    try:
        return Response(content=plot_uniform_pdf(a, b), media_type="image/png")
    except Exception as e:
        _bad(e)

@app.get("/plot/png/uniform_cdf")
def png_uniform_cdf(a: float = 0.0, b: float = 10.0):
    try:
        return Response(content=plot_uniform_cdf(a, b), media_type="image/png")
    except Exception as e:
        _bad(e)

@app.get("/plot/png/clt")
def png_clt(mu: float, sigma: float, n: int, lower: float, upper: float):
    try:
        return Response(content=plot_clt(mu, sigma, n, lower, upper), media_type="image/png")
    except Exception as e:
        _bad(e)

@app.get("/plot/png/joint_discrete")
def png_joint_discrete(pairs: str = Query(..., description="x,y,p;...")):
    try:
        def _parse_pairs(s: str):
            out = {}
            for chunk in s.split(";"):
                if not chunk.strip(): continue
                x,y,p = map(float, chunk.split(","))
                out[(x,y)] = out.get((x,y), 0.0) + p
            return out
        joint = _parse_pairs(pairs)
        return Response(content=plot_heat_joint(joint), media_type="image/png")
    except Exception as e:
        _bad(e)

# ---- root serves static index ----
@app.get("/", response_class=HTMLResponse)
def root():
    with open("/home/zk/projects/probplots-web/static/index.html", "r", encoding="utf-8") as f:
        return f.read()
