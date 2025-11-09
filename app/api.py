from fastapi import FastAPI, Query, HTTPException, Response
from fastapi.responses import JSONResponse, HTMLResponse

# numeric functions
from .dlbdss_core import (
    binom_pmf, binom_cdf, binom_cdf_range, pois_cdf, geom_pmf,
    norm_cdf, norm_inv, exp_cdf, unif_cdf, clt
)
# plotting
from .plotting import plot_normal_pdf, plot_sim_poisson, plot_sim_binom

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

# ---- root serves static index ----
@app.get("/", response_class=HTMLResponse)
def root():
    with open("/home/zk/projects/probplots-web/static/index.html", "r", encoding="utf-8") as f:
        return f.read()
