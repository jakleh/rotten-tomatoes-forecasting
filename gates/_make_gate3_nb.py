"""Authoring helper: assemble notebooks/gate3_tolerance.ipynb from cell sources.

Codegen only — analysis lives in the notebook (nbconvert-executed; cache-only, no
network). Gate 3 per plans/plan_gate_1_2_calibration.md: how much λ/p_fresh estimation
error can the Gate-2 edge absorb? Grid + decision rule locked in the intent cell
BEFORE execution.
"""
import os
import nbformat as nbf

MD = """# Gate 3a — λ/p_fresh error-tolerance band (directional)

Gate 2 passed with ORACLE inputs. Gate 3 asks: **how wrong can the inputs be before
the edge dies** — and is the shipped 0.2.0 estimator's error already inside the band?

**Pre-registered design (locked before execution):**
- Cells: the Gate-2 pure-oracle cells (`_cache/gate2_cells.csv`), primary snaps
  T-2d/3d/4d; headline set = the pooled unique markets (priority 3d>2d>4d; n=39/13
  movies — the most-powered Gate-2 read).
- Systematic perturbation: `λ̂ = λ_oracle × m`, m ∈ {0.25, 0.4, 0.55, 0.7, 0.85, 1.0,
  1.2, 1.5, 2.0, 3.0}; `p̂ = clip(p_fresh_oracle + δ, 0, 1)`,
  δ ∈ {−0.20 … +0.20 step 0.05}. Recompute `compute_edge` per cell per (m, δ); same
  taker-fee PnL rule and Brier-vs-market as Gate 2 (buffer = 0). The (m=1, δ=0) point
  must reproduce the cached Gate-2 `p_oracle` exactly (asserted).
- **Edge-preserving band** := (m, δ) where the movie-cluster bootstrap CI95 lower
  bound (1000 reps, shared resamples across combos) stays > 0 — separately for PnL and
  Brier-diff. Verdict band = the intersection.
- Random-noise variant (secondary): `λ̂ = λ·exp(ε)`, ε ~ N(0, σ_log) i.i.d. per cell,
  σ_log ∈ {0.25, 0.5, 1.0}, 50 draws each — does idiosyncratic error behave like
  systematic bias?
- Ridge overlay (coarse, cross-convention — CAVEAT: §1.1 LOO MAEs are phase-1 counts
  under the ET-midnight convention on the pre-April cohort; these cells use
  close−24N-hour snaps. The ratio MAE / median-cell-remaining is a relative-error
  PROXY, not a measurement; ratios computed in the overlay cell from BACKLOG §1.1
  MAEs {T-4d 21.10, T-3d 9.96, T-2d 3.42} ÷ these cells' median remaining counts).
  p_fresh: §1.4 cites estimator MAE 0.031 (T-1d, old validation) → δ ≈ ±0.03..0.05.
- Everything downstream of `gate2_cells.csv` (as_of_id=648979); n = 13 movies →
  **directional**.
"""

C_LOAD = """import os
os.environ.setdefault('MPLCONFIGDIR', os.environ.get('TMPDIR', '/tmp') + '/mpl')
import numpy as np, pandas as pd, sys
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
sys.path.insert(0, os.path.abspath('..') if os.path.basename(os.getcwd()) == 'notebooks' else os.path.abspath('.'))
from rotten_tomatoes_forecasting import compute_edge

CACHE = 'gates/_cache' if os.path.isdir('gates/_cache') else '../gates/_cache'
pr = pd.read_csv(CACHE + '/gate2_cells.csv')
pp = pr[(pr['mode'] == 'pure') & (pr['snap'] != '1d')].copy()
prio = {'3d': 0, '2d': 1, '4d': 2}
pool = (pp.assign(prio=lambda d: d['snap'].map(prio))
        .sort_values('prio').drop_duplicates('ticker')).reset_index(drop=True)
T_REM = {'4d': 96.0, '3d': 72.0, '2d': 48.0}
pool['t_rem'] = pool['snap'].map(T_REM)
pool['obs_fresh'] = (pool['obs_score'] * pool['obs_total']).round().fillna(0).astype(int)
print(f"pooled cells: {len(pool)} markets / {pool['slug'].nunique()} movies "
      f"(snaps {pool['snap'].value_counts().to_dict()})")
assert len(pool) == 39 and pool['slug'].nunique() == 13

M_GRID = np.array([0.25, 0.4, 0.55, 0.7, 0.85, 1.0, 1.2, 1.5, 2.0, 3.0])
D_GRID = np.round(np.arange(-0.20, 0.201, 0.05), 2)

def p_yes_cell(r, m, d):
    lam = r['lam'] * m
    pf = float(np.clip((r['p_fresh'] + d) if r['n_rem'] > 0 else 0.5, 0.0, 1.0))
    return compute_edge(int(r['X']), float(r['mid']) * 100, int(r['obs_fresh']),
                        int(r['obs_total']), float(r['t_rem']), lam, pf)['p_yes']

def fee_cents(price_cents):
    return 7.0 * (price_cents / 100.0) * (1 - price_cents / 100.0)

def pnl_cents(po100, y, bid, ask):
    if po100 > ask * 100:
        return (100 * y - ask * 100) - fee_cents(ask * 100)
    if po100 < bid * 100:
        no_c = 100 - bid * 100
        return (100 * (1 - y) - no_c) - fee_cents(no_c)
    return np.nan

# per-cell p_yes / pnl / brier for every grid combo
n_cells = len(pool)
PY = np.zeros((len(M_GRID), len(D_GRID), n_cells))
for i, m in enumerate(M_GRID):
    for j, d in enumerate(D_GRID):
        PY[i, j] = [p_yes_cell(r, m, d) for _, r in pool.iterrows()]
i1, j0 = M_GRID.tolist().index(1.0), list(D_GRID).index(0.0)
assert np.allclose(PY[i1, j0], pool['p_oracle'].to_numpy(), atol=1e-9), \\
    'oracle-point reconstruction mismatch (obs_fresh/t_rem/clip drift)'
y = pool['y'].to_numpy(float)
mid = pool['mid'].to_numpy(float)
bidv, askv = pool['bid'].to_numpy(float), pool['ask'].to_numpy(float)
BR_DIFF = (mid - y) ** 2 - (PY - y) ** 2                  # per-cell Brier improvement
PNL = np.full_like(PY, np.nan)
for i in range(len(M_GRID)):
    for j in range(len(D_GRID)):
        PNL[i, j] = [pnl_cents(PY[i, j, k] * 100, y[k], bidv[k], askv[k])
                     for k in range(n_cells)]
print('grid computed:', PY.shape, '| oracle point (m=1, d=0) mean PnL =',
      round(float(np.nanmean(PNL[M_GRID.tolist().index(1.0),
                                 D_GRID.tolist().index(0.0)])), 1), 'c')
"""

C_BAND = """# Edge-preserving band: shared movie-cluster bootstrap across all combos (1000 reps)
rng = np.random.default_rng(3)
movies = pool['slug'].unique()
midx = {m: np.flatnonzero((pool['slug'] == m).to_numpy()) for m in movies}
N_BOOT = 1000
resamples = [np.concatenate([midx[m] for m in rng.choice(movies, len(movies), True)])
             for _ in range(N_BOOT)]

def ci_lower(per_cell):
    stats = [np.nanmean(per_cell[idx]) if np.isfinite(per_cell[idx]).any() else np.nan
             for idx in resamples]
    return float(np.nanpercentile(stats, 2.5))

pnl_mean = np.nanmean(PNL, axis=2)
pnl_lo = np.array([[ci_lower(PNL[i, j]) for j in range(len(D_GRID))]
                   for i in range(len(M_GRID))])
br_mean = np.nanmean(BR_DIFF, axis=2)
br_lo = np.array([[ci_lower(BR_DIFF[i, j]) for j in range(len(D_GRID))]
                  for i in range(len(M_GRID))])
n_trades = np.sum(~np.isnan(PNL), axis=2)

def show(mat, title, fmt='{:+.1f}'):
    df = pd.DataFrame(mat, index=[f'm={m}' for m in M_GRID],
                      columns=[f'd={d:+.2f}' for d in D_GRID])
    print(f'\\n=== {title} ===')
    print(df.map(lambda v: fmt.format(v)).to_string())

show(pnl_mean, 'mean PnL (cents/contract), pooled n=39')
show(pnl_lo, 'PnL CI95 lower bound (cluster-boot by movie)')
show(br_lo, 'Brier-diff CI95 lower bound', fmt='{:+.3f}')
band = (pnl_lo > 0) & (br_lo > 0)
show(band.astype(int), 'EDGE-PRESERVING BAND (1 = both CI-lowers > 0)', fmt='{:d}')
show(n_trades.astype(int), 'trades taken (of 39)', fmt='{:d}')

fig, ax = plt.subplots(figsize=(8.5, 5))
im = ax.imshow(pnl_mean, cmap='RdYlGn', vmin=-30, vmax=30, aspect='auto')
for i in range(len(M_GRID)):
    for j in range(len(D_GRID)):
        ax.text(j, i, f'{pnl_mean[i, j]:+.0f}', ha='center', va='center', fontsize=7,
                fontweight='bold' if band[i, j] else 'normal')
        if band[i, j]:
            ax.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False, lw=1.8,
                                       edgecolor='black'))
ax.set_xticks(range(len(D_GRID)), [f'{d:+.2f}' for d in D_GRID], fontsize=8)
ax.set_yticks(range(len(M_GRID)), [f'{m}x' for m in M_GRID], fontsize=8)
ax.set_xlabel('p_fresh additive error δ'); ax.set_ylabel('λ multiplicative error m')
ax.set_title('Gate 3a: mean PnL (c/contract); boxed = edge-preserving band (PnL & Brier CI>0)')
plt.colorbar(im, label='cents')
plt.tight_layout(); plt.savefig(CACHE + '/gate3_band.png', dpi=110, bbox_inches='tight')
print('\\nsaved', CACHE + '/gate3_band.png')
"""

C_NOISE = """# Secondary: idiosyncratic noise vs systematic bias
print('=== random log-normal λ noise (50 draws each; δ=0) — mean PnL / share of draws with CI-lower>0 ===')
for sig in [0.25, 0.5, 1.0]:
    means, ok = [], 0
    for rep in range(50):
        eps = rng.normal(0, sig, n_cells)
        py = np.array([p_yes_cell(r, float(np.exp(eps[k])), 0.0)
                       for k, (_, r) in enumerate(pool.iterrows())])
        pnl = np.array([pnl_cents(py[k] * 100, y[k], bidv[k], askv[k])
                        for k in range(n_cells)])
        means.append(np.nanmean(pnl))
        ok += ci_lower(pnl) > 0
    print(f'  sigma_log={sig}: mean PnL {np.mean(means):+.1f}c '
          f'(draw range {np.min(means):+.1f}..{np.max(means):+.1f}), '
          f'CI-lower>0 in {ok}/50 draws')

print('\\n=== p_fresh-only noise (m=1): random sign of |δ| per cell, 50 draws ===')
for mag in [0.05, 0.10, 0.15]:
    means, ok = [], 0
    for rep in range(50):
        sgn = rng.choice([-1.0, 1.0], n_cells)
        py = np.array([p_yes_cell(r, 1.0, float(sgn[k] * mag))
                       for k, (_, r) in enumerate(pool.iterrows())])
        pnl = np.array([pnl_cents(py[k] * 100, y[k], bidv[k], askv[k])
                        for k in range(n_cells)])
        means.append(np.nanmean(pnl))
        ok += ci_lower(pnl) > 0
    print(f'  |delta|={mag:.2f}: mean PnL {np.mean(means):+.1f}c '
          f'(range {np.min(means):+.1f}..{np.max(means):+.1f}), CI-lower>0 in {ok}/50')
"""

C_RIDGE = """# Ridge overlay — COARSE cross-convention proxy (see intent-cell caveat)
print('shipped-estimator relative-error proxies vs the band:')
med_rem = pp.groupby('snap')['n_rem'].median()
ridge_mae = {'4d': 21.10, '3d': 9.96, '2d': 3.42}     # BACKLOG §1.1 LOO MAE (counts)
for lab in ['4d', '3d', '2d']:
    rel = ridge_mae[lab] / med_rem[lab]
    print(f"  T-{lab}: LOO MAE {ridge_mae[lab]} / median cell remaining "
          f"{med_rem[lab]:.0f} -> relative λ error ~{rel:.2f} -> m in "
          f"[{1 - rel:.2f}, {1 + rel:.2f}]")
print('  p_fresh: estimator MAE 0.031 (BACKLOG §1.4, T-1d, old validation) -> '
      'delta ~ +/-0.03..0.05')

def band_at(m_target, d_target):
    i = int(np.argmin(np.abs(M_GRID - m_target)))
    j = int(np.argmin(np.abs(D_GRID - d_target)))
    return (f'(m={M_GRID[i]}, d={D_GRID[j]:+.2f}): mean PnL {pnl_mean[i, j]:+.1f}c, '
            f'PnL CI-lo {pnl_lo[i, j]:+.1f}, Brier CI-lo {br_lo[i, j]:+.3f}, '
            f'in-band={bool(band[i, j])}')

print('\\nband readout at the Ridge-proxy corners (nearest grid point):')
for m_t in [0.7, 0.75, 1.3, 1.5]:
    for d_t in [-0.05, 0.05]:
        print('  ' + band_at(m_t, d_t))
print('\\nNOTE: proxies are relative-MAE point estimates under a DIFFERENT window '
      'convention and an older cohort — Gate 3b (run the actual 0.2.0 estimator on '
      'these cells, ET-midnight aligned, pool cache required) is the real overlay.')
"""

MD_TAIL = """## Reading guide

- The **band table/heatmap** is the deliverable: the (m, δ) region where the Gate-2
  edge survives estimation error with movie-clustered CI support. Asymmetries matter:
  under-estimating λ (m<1) vs over-estimating (m>1) need not be symmetric — frozen-ish
  inputs degrade toward the current-state forecast, which Gate 1b showed has no edge.
- Random-noise draws (secondary) test whether idiosyncratic error is gentler than
  systematic bias (averaging across cells) or harsher (occasionally flipping trades
  at boundaries).
- The Ridge overlay is a PROXY ONLY (cross-convention, older cohort). Gate 3b = run
  `estimate_lambda`/`estimate_p_fresh` for real on these cells.
- Everything is downstream of `gate2_cells.csv` (as_of_id=648979); n = 13 movies →
  directional.
"""

nb = nbf.v4.new_notebook()
nb.cells = [nbf.v4.new_markdown_cell(MD), nbf.v4.new_code_cell(C_LOAD),
            nbf.v4.new_code_cell(C_BAND), nbf.v4.new_code_cell(C_NOISE),
            nbf.v4.new_code_cell(C_RIDGE), nbf.v4.new_markdown_cell(MD_TAIL)]
nb.metadata.kernelspec = {"name": "python3", "display_name": "Python 3", "language": "python"}
os.makedirs("notebooks", exist_ok=True)
nbf.write(nb, "notebooks/gate3_tolerance.ipynb")
print("wrote notebooks/gate3_tolerance.ipynb")
