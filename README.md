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
