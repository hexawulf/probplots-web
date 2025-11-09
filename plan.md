# probplots-web — Master Build Plan (Authoritative, Single Source of Truth)

**Target host:** Raspberry Pi 5 (Ubuntu) · **Bind:** 127.0.0.1:5009 · **Public URL (via Nginx):** https://probplots.piapps.dev  
**Back-end:** FastAPI (uvicorn) · **Front-end:** Tailwind (CDN) + vanilla JS · **Plots:** matplotlib (Agg)

This document contains **everything needed** for Claude Code (or any executor) to implement the thin web wrapper for the `probplots` project, including directory tree, file contents to generate, scripts, systemd unit, Nginx site, validation rules, and verification steps.

---

## 0) Context (read-only references)

- Core helper repo (public): `https://github.com/hexawulf/probplots`  
  Local path on host: `/home/zk/projects/probplots`
- This web wrapper repo: `https://github.com/hexawulf/probplots-web`  
  This **plan.md is authoritative** when anything conflicts.

**Do not copy code** from `probplots` into this repo. Import it at runtime from its local path.

---

## 1) Goals

1. Provide GET API endpoints for common distribution queries (Binomial, Poisson, Geometric, Normal, Exponential, Uniform, CLT helper).
2. Provide plotting endpoints that return PNGs (headless Agg backend).
3. Provide a **discrete joint-distribution** endpoint that calls the existing helper in `probplots/stats_helper` and returns marginals, conditionals, independence, and moments.
4. Serve a minimal UI (`/`) with a tiny form for quick testing (including the joint endpoint).

---

## 2) Constraints & Conventions

- **Bind only to 127.0.0.1:5009.** (Nginx will proxy HTTPS.)
- **GET-only** endpoints; validate inputs and return helpful 400 messages on invalid inputs.
- Use **matplotlib Agg**; never assume a DISPLAY.
- Keep the codebase small, dependency-light, readable.
- Mirror the Nginx style already used across piapps (site file provided below).
- Security: strict input validation; no shelling out to the CLI; only import `probplots` package for logic.

Environment assumptions:

- Python ≥ 3.10 (3.13 available).
- OS: Ubuntu aarch64 on Pi 5.
- `probplots` is present at `/home/zk/projects/probplots` and importable.

---

## 3) Deliverables (create exactly this tree and files)

```
/home/zk/projects/probplots-web
├── app/
│   ├── __init__.py
│   ├── dlbdss_core.py
│   ├── plotting.py
│   └── api.py
├── static/
│   └── index.html
├── scripts/
│   ├── install.sh
│   ├── run-dev.sh
│   └── make-systemd.sh
├── probplots-web.service
├── nginx-probplots.piapps.dev
└── README.md
```

---

## 4) File contents to generate

### 4.1 `app/__init__.py`

```python
# empty init so "app" is a package
```

### 4.2 `app/dlbdss_core.py`

Implement importable pure functions (no print/IO). Validate inputs and raise `ValueError` with clear messages.

```python
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
```

### 4.3 `app/plotting.py`

Produce PNG bytes via `BytesIO`; configure Agg.

```python
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
```

### 4.4 `app/api.py`


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


### 4.5 `static/index.html`

A tiny UI with a Joint panel and a couple of quick checks.

```html
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>probplots-web</title>
  <script src="https://cdn.tailwindcss.com"></script>
</head>
<body class="bg-slate-50 text-slate-900">
  <main class="max-w-5xl mx-auto p-6 space-y-6">
    <header class="flex items-center justify-between">
      <h1 class="text-2xl font-bold">probplots-web</h1>
      <a href="/healthz" class="text-xs underline">/healthz</a>
    </header>

    <section class="grid md:grid-cols-2 gap-6">
      <form id="f-joint" class="p-4 rounded-xl bg-white shadow">
        <h3 class="font-semibold mb-2">Joint (discrete)</h3>
        <p class="text-xs text-slate-500 mb-2">pairs: x,y,p;... (sum to 1)</p>
        <input class="border p-2 rounded w-full" name="pairs"
               value="0,0,0.1;0,1,0.2;1,0,0.3;1,1,0.4">
        <button class="mt-3 px-3 py-2 rounded bg-slate-900 text-white">Compute</button>
        <pre id="r-joint" class="mt-3 text-xs whitespace-pre-wrap break-all"></pre>
      </form>

      <div class="p-4 rounded-xl bg-white shadow">
        <h3 class="font-semibold mb-2">Quick checks</h3>
        <div class="space-x-2">
          <button id="btn-ncdf" class="px-3 py-2 rounded bg-slate-900 text-white text-sm">Normal CDF(1.96)</button>
          <button id="btn-plot" class="px-3 py-2 rounded bg-slate-900 text-white text-sm">Show Normal PDF</button>
        </div>
        <pre id="r-ncdf" class="mt-3 text-xs"></pre>
        <img id="img-pdf" class="mt-3 rounded border hidden" alt="normal pdf"/>
      </div>
    </section>
  </main>

  <script>
    (function(){
      const f = document.getElementById('f-joint');
      const out = document.getElementById('r-joint');
      f.addEventListener('submit', async (e) => {
        e.preventDefault();
        out.textContent = '...';
        const q = new URLSearchParams(new FormData(f));
        const r = await fetch('/api/joint?' + q.toString());
        out.textContent = r.ok ? JSON.stringify(await r.json(), null, 2) : ('Error: ' + await r.text());
      });

      const btnNcdf = document.getElementById('btn-ncdf');
      const btnPlot = document.getElementById('btn-plot');
      const rNcdf = document.getElementById('r-ncdf');
      const img = document.getElementById('img-pdf');

      btnNcdf.addEventListener('click', async () => {
        rNcdf.textContent = '...';
        const r = await fetch('/api/norm/cdf?x=1.96&mu=0&sigma=1');
        const j = await r.json();
        rNcdf.textContent = JSON.stringify(j);
      });

      btnPlot.addEventListener('click', async () => {
        const r = await fetch('/plot/pdf/normal?mu=0&sigma=1');
        const blob = await r.blob();
        img.src = URL.createObjectURL(blob);
        img.classList.remove('hidden');
      });
    })();
  </script>
</body>
</html>
```

### 4.6 `scripts/install.sh`

```bash
#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

/usr/bin/python3 -m venv .venv
./.venv/bin/pip install --upgrade pip
./.venv/bin/pip install fastapi "uvicorn[standard]" numpy scipy matplotlib
```

### 4.7 `scripts/run-dev.sh`

```bash
#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export OPENBLAS_NUM_THREADS=1
exec ./.venv/bin/uvicorn app.api:app --host 127.0.0.1 --port 5009
```

### 4.8 `scripts/make-systemd.sh`

```bash
#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

unit=probplots-web.service
cat > "$unit" <<'UNIT'
[Unit]
Description=probplots-web (FastAPI)
After=network.target

[Service]
Type=simple
WorkingDirectory=/home/zk/projects/probplots-web
Environment=OPENBLAS_NUM_THREADS=1
ExecStart=/home/zk/projects/probplots-web/.venv/bin/uvicorn app.api:app --host 127.0.0.1 --port 5009
Restart=always
RestartSec=2
User=zk
Group=zk

[Install]
WantedBy=multi-user.target
UNIT

sudo /usr/bin/install -m 0644 "$unit" /etc/systemd/system/$unit
sudo /usr/bin/systemctl daemon-reload
sudo /usr/bin/systemctl enable --now $unit
sudo /usr/bin/systemctl status --no-pager $unit || true
```

### 4.9 `probplots-web.service`

(Generated by `make-systemd.sh`; kept in repo for reference and diff.)

### 4.10 `nginx-probplots.piapps.dev`

Install as `/etc/nginx/sites-available/probplots.piapps.dev` and symlink to `sites-enabled`. Uses the same TLS chain as other piapps hosts.

```nginx
# =========================
# probplots.piapps.dev  (SPA served by FastAPI; same-origin calls under /api)
# =========================

# --- Strict CORS allowlist (only relevant if you ever call across origins)
# Same-origin (https://probplots.piapps.dev) needs no CORS; this is a safety net.
map $http_origin $cors_allow_probplots {
  default "";
  "~^https://probplots\.piapps\.dev$"  $http_origin;
  "~^https://(www\.)?containeryard\.org$" $http_origin;   # optional: remove if not needed
}

# --- HTTP → HTTPS
server {
  listen 80;
  server_name probplots.piapps.dev;
  return 301 https://$host$request_uri;
}

# --- HTTPS
server {
  listen 443 ssl;
  http2 on;
  server_name probplots.piapps.dev;

  # TLS (reuse the same cert bundle you use for container.piapps.dev)
  ssl_certificate     /etc/letsencrypt/live/piapps.dev-0001/fullchain.pem;
  ssl_certificate_key /etc/letsencrypt/live/piapps.dev-0001/privkey.pem;
  include /etc/letsencrypt/options-ssl-nginx.conf;
  ssl_dhparam /etc/letsencrypt/ssl-dhparams.pem;

  # Real client IPs behind Cloudflare
  include /etc/nginx/conf.d/cloudflare_real_ip.conf;

  # HSTS
  add_header Strict-Transport-Security "max-age=31536000; includeSubDomains; preload" always;

  # --- API (FIRST)
  # FastAPI listens on 127.0.0.1:5009 and serves both /api/* and the SPA index at /
  location /api/ {
    # CORS (only matters if cross-origin ever used)
    add_header Access-Control-Allow-Origin $cors_allow_probplots always;
    add_header Access-Control-Allow-Credentials "true" always;
    add_header Access-Control-Allow-Headers "Content-Type, Authorization, X-Requested-With, X-CSRF-Token" always;
    add_header Access-Control-Allow-Methods "GET, POST, PUT, PATCH, DELETE, OPTIONS" always;
    add_header Vary "Origin" always;

    # Preflight
    if ($request_method = OPTIONS) { return 204; }

    # Upstream
    proxy_pass http://127.0.0.1:5009;
    proxy_http_version 1.1;

    # WebSocket-friendly (harmless for normal HTTP)
    proxy_set_header Upgrade $http_upgrade;
    proxy_set_header Connection $connection_upgrade;

    # Forward standard headers
    proxy_set_header Host              $host;
    proxy_set_header X-Real-IP         $remote_addr;
    proxy_set_header X-Forwarded-For   $proxy_add_x_forwarded_for;
    proxy_set_header X-Forwarded-Proto $scheme;
    proxy_set_header X-Forwarded-Host  $host;
    proxy_set_header X-Forwarded-Port  443;

    # Forward Origin if backend wants it
    proxy_set_header Origin $http_origin;

    # Reasonable API timeouts
    proxy_read_timeout    75s;
    proxy_send_timeout    75s;
    proxy_connect_timeout 15s;

    client_max_body_size  10m;
  }

  # --- Health passthrough (backend JSON), exact-match
  location = /health {
    proxy_pass http://127.0.0.1:5009/healthz;
    proxy_http_version 1.1;
    proxy_set_header Connection "";

    proxy_set_header Host              $host;
    proxy_set_header X-Real-IP         $remote_addr;
    proxy_set_header X-Forwarded-For   $proxy_add_x_forwarded_for;
    proxy_set_header X-Forwarded-Proto $scheme;

    expires off;
    add_header Cache-Control "no-cache, no-store, must-revalidate" always;
    add_header Pragma "no-cache" always;
    add_header Expires "0" always;
  }

  # --- Lightweight edge health (always-200, uncacheable)
  location = /healthz {
    default_type application/json;
    add_header Cache-Control "no-cache, no-store, must-revalidate" always;
    add_header Pragma "no-cache" always;
    add_header Expires "0" always;
    return 200 '{"status":"ok","host":"$hostname","time":"$time_iso8601"}';
  }

  # --- SPA catch-all proxied to FastAPI (serves / and static/index.html)
  location / {
    add_header Cache-Control "no-cache";
    proxy_pass http://127.0.0.1:5009;
    proxy_http_version 1.1;

    proxy_set_header Upgrade $http_upgrade;
    proxy_set_header Connection $connection_upgrade;

    proxy_set_header Host              $host;
    proxy_set_header X-Real-IP         $remote_addr;
    proxy_set_header X-Forwarded-For   $proxy_add_x_forwarded_for;
    proxy_set_header X-Forwarded-Proto $scheme;
  }

  # Hardening: block dotfiles and common secrets
  location ~* (?i)(^|/)(\.env|\.git|\.htaccess|\.DS_Store|\.aws|credentials|composer\.json)$ { return 403; }
  location ~* (?i)\.(ya?ml|ini|sql|bak|old|log|config)$ { return 403; }

  access_log /var/log/nginx/probplots.piapps.dev.access.log;
  error_log  /var/log/nginx/probplots.piapps.dev.error.log warn;
}
```

### 4.11 `README.md`

```markdown
# probplots-web
Thin FastAPI + Tailwind wrapper for https://github.com/hexawulf/probplots

## Endpoints (GET)
- `/healthz`
- `/api/norm/cdf?x&mu&sigma`
- `/api/norm/inv?p&mu&sigma`
- `/api/binom/pmf?n&p&k`
- `/api/binom/cdf?n&p&k`
- `/api/binom/cdf-range?n&p&kmin&kmax`
- `/api/pois/cdf?lam&k`
- `/api/geom/pmf?p&k`  (k = failures before first success)
- `/api/exp/cdf?lam&x`
- `/api/unif/cdf?a&b&x`
- `/api/clt?mu&sigma&n&lower&upper`
- `/api/joint?pairs=0,0,0.1;0,1,0.2;1,0,0.3;1,1,0.4`

### Joint notes
- `pairs` is a semicolon-separated list of `x,y,p` triples (floats).
- p ≥ 0 and sum ≈ 1 (±1e-10) or 400 error.
- Returns: `support_X`, `support_Y`, `joint`, `marginal_X`, `marginal_Y`, `independent`, `E`, `Var`, `Cov`, `Corr`.

### Plots
- `/plot/pdf/normal?mu&sigma` → PNG
- `/plot/sim/poisson?n&lam&bins` → PNG
- `/plot/sim/binom?n&N&p&bins` → PNG

## Dev
```bash
bash scripts/install.sh
bash scripts/run-dev.sh  # binds 127.0.0.1:5009
```

```
---

## 5) Build & Install Steps (execute in order)

From `/home/zk/projects/probplots-web`:
```bash
bash scripts/install.sh
bash scripts/run-dev.sh   # run once in foreground; Ctrl+C to stop after checking /healthz
bash scripts/make-systemd.sh
```

Install Nginx site and reload:

```bash
sudo install -m 0644 /home/zk/projects/probplots-web/nginx-probplots.piapps.dev /etc/nginx/sites-available/probplots.piapps.dev
sudo ln -sf /etc/nginx/sites-available/probplots.piapps.dev /etc/nginx/sites-enabled/probplots.piapps.dev
sudo nginx -t && sudo systemctl reload nginx
```

---

## 6) Verification

```bash
# Backend health via Nginx passthrough
curl -sS https://probplots.piapps.dev/healthz

# Joint endpoint example
curl -sS 'https://probplots.piapps.dev/api/joint?pairs=0,0,0.1;0,1,0.2;1,0,0.3;1,1,0.4' | jq .

# Normal CDF smoke test
curl -sS 'https://probplots.piapps.dev/api/norm/cdf?x=1.96&mu=0&sigma=1' | jq .
```

Expected `joint` keys: `support_X`, `support_Y`, `joint`, `marginal_X`, `marginal_Y`, `independent`, `E`, `Var`, `Cov`, `Corr`.

---

## 7) Troubleshooting

- **ImportError for stats_helper** → ensure `probplots` exists at `/home/zk/projects/probplots` and that `api.py` prepends this path to `sys.path` before import.
- **Binding error / connection refused** → verify service is on `127.0.0.1:5009` and the systemd unit is active.
- **PNG responses blank** → confirm Agg backend is set in `plotting.py` and returned as `image/png`.
- **Nginx 502** → check `sudo journalctl -u probplots-web -e` and `/var/log/nginx/probplots.piapps.dev.error.log`.
- **High CPU** → set `OPENBLAS_NUM_THREADS=1` (already in unit).

---

## 8) Success Criteria

- Service responds on `https://probplots.piapps.dev/healthz` via Nginx.
- `/api/joint` returns correct JSON for the example and detects independence when factorable.
- Plot endpoints return PNGs.
- Code is small, readable, and follows this plan exactly.
