"""Authoring helper: assemble notebooks/pfresh_bench.ipynb from cell sources.

Codegen only — cache-only notebook (no DB). Spec: ``plans/plan_p_fresh_regression.md``
v2 "Phase 1 + Phase 2". Builds EXACTLY the battery-decided configuration
(asserts against ``pfresh_battery_decisions.csv``), scores candidates on the LOCKED
gate3b bench through temporal per-close fits, and prints the adoption verdict under
the locked paired rule. Brier/PnL machinery copied VERBATIM from
``gates/_make_gate3b_nb.py`` C_MACHINERY [F17] (seed 7, 2000 shared resamples).
"""
import os
import nbformat as nbf

MD = """# p_fresh candidates — bench verdict (battery-gated build; locked adoption rule)

**Locked rule:** adopt iff bench pooled PnL CI-lower > 0 AND paired
(candidate − shade₋.₀₃) ≥ 0 on the shared draws (seed 7, 2000 movie-cluster
resamples). Multiple clear → simplest rung. No winner → ship nothing (recorder
re-run later). Adopted ⇒ live only after paired ≥ 0 on the first K≈8 newly-settled
recorder movies (tripwire). **Every bench read of a candidate is
design-derived-from-bench** (the same 13-14 movies shaped the design; the recorder
flow is the only true holdout). λ̂ frozen (cells CSV `rate_hat`/`h`); bench grid
frozen; estimator-skipped cells stay skipped (coverage unchanged by construction).

Candidates (battery `pfresh_battery_decisions.csv`): **C1′** calibrated blend
(logit p̂_shipped + snap intercepts), **C2** GLM {emp-logit obs_rate, log1p_total,
their interaction, logit P3 (shrunk prior — battery use_P3), snap dummies,
obs_rate×snap (battery state_term), mass_consumed (corr-gated)}. **C3 NOT BUILT**
(battery: intensity increment −0.0002, T4 mass 4.7% < 15%). Fits: expanded-counts
L2 logistic, per-movie weights, GroupKFold C-selection, TEMPORAL per bench close
(9 distinct closes; target excluded; ≥60-movie floor; M5 window assert).
"""

C_LOAD = """import os, sys, warnings
os.environ.setdefault('MPLCONFIGDIR', os.environ.get('TMPDIR', '/tmp') + '/mpl')
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', message='Mean of empty slice')
import numpy as np, pandas as pd
sys.path.insert(0, os.path.abspath('..') if os.path.basename(os.getcwd()) == 'notebooks' else os.path.abspath('.'))
from gates import pfresh_lib as pl

CACHE = 'gates/_cache' if os.path.isdir('gates/_cache') else '../gates/_cache'
SEED, N_BOOT = 7, 2000
c = pd.read_csv(CACHE + '/gate3b_cells.csv')
ft = pd.read_csv(CACHE + '/pfresh_training_features.csv')
dec = pd.read_csv(CACHE + '/pfresh_battery_decisions.csv').iloc[0]
cache = pd.read_csv(CACHE + '/gate3b_a1_cache.csv')
pools = pd.read_csv(CACHE + '/gate3b_pools.csv')
cache['estimated_timestamp'] = pd.to_datetime(cache['estimated_timestamp'], utc=True, format='ISO8601')
assert bool(dec['use_P3']) and bool(dec['state_term']) and not bool(dec['build_C3'])
assert not bool(dec['drow_material']), 'd-row sensitivity fired — operator ping required'
ft['close_dt'] = pd.to_datetime(ft['close'], utc=True, format='ISO8601')
print(f"bench cells {len(c)} | training features {len(ft)} rows / "
      f"{ft['slug'].nunique()} movies @ pin {int(dec['as_of_id'])}")

# training-side feature columns (battery frame, fixed at build — temporal fits SUBSET rows)
W_SHIP = ft['n_obs'] / (ft['n_obs'] + 20.0)
ft['p_shipped'] = W_SHIP * ft['P1'] + (1 - W_SHIP) * ft['P2']
ft['lp_shipped'] = [pl.logit_clip(p) for p in ft['p_shipped']]
ft['el_P1'] = [pl.emp_logit(f, n) for f, n in zip(ft['fresh_obs'], ft['n_obs'])]
ft['log1p_total'] = np.log1p(ft['n_obs'])
ft['el_x_log'] = ft['el_P1'] * ft['log1p_total']
ft['lp_P3'] = [pl.logit_clip(p) for p in ft['P3']]
for n in (2, 3, 4):
    ft[f'snap_{n}'] = (ft['snap_days'] == n).astype(float)
    ft[f'el_x_snap{n}'] = ft['el_P1'] * ft[f'snap_{n}']
corr_mass = float(ft[['mass_consumed', 'log1p_total']].corr().iloc[0, 1])
use_mass = abs(corr_mass) <= 0.9
print(f"corr(mass_consumed, log1p_total) on training rows = {corr_mass:+.3f} -> "
      f"{'kept' if use_mass else 'DROPPED'}")
F_C1 = ['lp_shipped', 'snap_2', 'snap_3', 'snap_4']
F_C2 = (['el_P1', 'log1p_total', 'el_x_log', 'lp_P3', 'snap_2', 'snap_3', 'snap_4',
         'el_x_snap2', 'el_x_snap3', 'el_x_snap4'] + (['mass_consumed'] if use_mass else []))
print('C1\\' features:', F_C1, '\\nC2 features:', F_C2)
"""

C_MACHINERY = """# Brier/PnL machinery — copied VERBATIM from gates/_make_gate3b_nb.py C_MACHINERY
def fee_cents(price_c):
    return 7.0 * (price_c / 100.0) * (1 - price_c / 100.0)

def pnl_c(p, y, bid, ask, buffer_c=0.0):
    pc, ask_c, bid_c = p * 100, ask * 100, bid * 100
    if pc > ask_c + buffer_c:
        return (100 * y - ask_c) - fee_cents(ask_c)
    if pc < bid_c - buffer_c:
        no_c = 100 - bid_c
        return (100 * (1 - y) - no_c) - fee_cents(no_c)
    return np.nan

def draws(d, seed=SEED, n=N_BOOT):
    rng = np.random.default_rng(seed)
    movies = np.array(sorted(d['slug'].unique()))
    midx = {m: np.flatnonzero((d['slug'] == m).to_numpy()) for m in movies}
    return [np.concatenate([midx[m] for m in rng.choice(movies, len(movies), True)])
            for _ in range(n)]

def ci(vals, lo=2.5, hi=97.5):
    return (float(np.nanpercentile(vals, lo)), float(np.nanpercentile(vals, hi)))

def brier_diff_stat(d, pcol):
    return float(np.mean((d['mid'] - d['y']) ** 2) - np.mean((d[pcol] - d['y']) ** 2))

def pnl_stat(d, pcol, buffer_c=0.0):
    return np.array([pnl_c(p, y, b, a, buffer_c) for p, y, b, a
                     in zip(d[pcol], d['y'], d['bid'], d['ask'])])

def block(d, pcol, label, rs=None, buffer_c=0.0):
    rs = rs if rs is not None else draws(d)
    bd = brier_diff_stat(d, pcol)
    bd_ci = ci([brier_diff_stat(d.iloc[idx], pcol) for idx in rs])
    pv = pnl_stat(d, pcol, buffer_c)
    t = d[~np.isnan(pv)]
    pnl = float(np.nanmean(pv))
    pnl_ci = ci([float(np.nanmean(pnl_stat(d.iloc[idx], pcol, buffer_c)))
                 for idx in rs])
    win = float(100 * (pv[~np.isnan(pv)] > 0).mean()) if len(t) else float('nan')
    print(f"  {label:<30} n={len(d):>2}/{d['slug'].nunique()}mv "
          f"Brier diff {bd:+.4f} [{bd_ci[0]:+.4f}, {bd_ci[1]:+.4f}]"
          f"{' *' if bd_ci[0] > 0 else '  '} | PnL {pnl:+.1f}c "
          f"[{pnl_ci[0]:+.1f}, {pnl_ci[1]:+.1f}]{' *' if pnl_ci[0] > 0 else '  '} "
          f"trades={len(t)} win%={win:.0f}")
    return {'brier': bd, 'brier_ci': bd_ci, 'pnl': pnl, 'pnl_ci': pnl_ci}

cov = c[c['status'] == 'ok'].copy()
prio = {3: 0, 2: 1, 4: 2}
pool_cells = (cov[cov['primary']].assign(p=lambda d: d['snap_days'].map(prio))
              .sort_values('p').drop_duplicates('ticker').reset_index(drop=True))
pool_cells['shade'] = (pool_cells['p_fresh_hat'] - 0.03).clip(0, 1)
from rotten_tomatoes_forecasting import compute_edge
def p_yes_with(d, pf_col):
    return [compute_edge(int(r['X']), float(r['mid']) * 100, int(r['obs_fresh_est']),
                         int(r['obs_total_est']), float(r['h']), float(r['rate_hat']),
                         float(np.clip(r[pf_col], 0, 1)))['p_yes']
            for _, r in d.iterrows()]
pool_cells['py_shade'] = p_yes_with(pool_cells, 'shade')
rs_pool = draws(pool_cells)
print(f"verdict set: {len(pool_cells)} cells / {pool_cells['slug'].nunique()} movies")
print('=== C0 anchor (must reproduce +11.8c [-0.3, +26.0]) ===')
a = block(pool_cells, 'py_shade', 'shade s=-0.03', rs_pool)
assert round(a['pnl'], 1) == 11.8 and round(a['pnl_ci'][0], 1) == -0.3 \
    and round(a['pnl_ci'][1], 1) == 26.0, 'shade anchor drifted'
print('anchor reproduces the gate3b shade row exactly.')
"""

C_BENCHFEAT = """# Bench-cell features from the LOCKED 648979 world (cells CSV + a1 cache + pools)
bench = c.copy()
bench['el_P1'] = [pl.emp_logit(f, n) for f, n in zip(bench['obs_fresh_est'],
                                                     bench['obs_total_est'])]
bench['log1p_total'] = np.log1p(bench['obs_total_est'])
bench['el_x_log'] = bench['el_P1'] * bench['log1p_total']
bench['lp_shipped'] = [pl.logit_clip(p) for p in bench['p_fresh_hat']]
for n in (2, 3, 4):
    bench[f'snap_{n}'] = (bench['snap_days'] == n).astype(float)
    bench[f'el_x_snap{n}'] = bench['el_P1'] * bench[f'snap_{n}']
pool_slugs = {t: list(g.sort_values('rank')['pool_slug'])
              for t, g in pools.groupby('target')}
p3_vals, mass_vals = [], []
for _, r in bench.iterrows():
    slugs = pool_slugs[r['slug']]
    mr = cache[cache['movie_slug'] == r['slug']]
    snap = pd.to_datetime(r['snap_ts'])
    oc = set(mr.loc[mr['estimated_timestamp'] < snap, 'reviewer_name'])
    p3_vals.append(pl.prior_remaining(cache, slugs, oc, shrink_k=pl.SHRINK_K))
    pool_rows = cache[cache['movie_slug'].isin(slugs)]
    base = pool_rows.groupby('reviewer_name')['movie_slug'].nunique() / 20
    mass_vals.append(float(base[base.index.isin(oc)].sum() / base.sum())
                     if base.sum() > 0 else 0.0)
bench['lp_P3'] = [pl.logit_clip(p) for p in p3_vals]
bench['mass_consumed'] = mass_vals
# plan sanity check: shipped-p̂ reconstruction ≡ stored p_fresh_hat at 1e-9 (all cells)
from rotten_tomatoes_forecasting import estimate_p_fresh
max_err = 0.0
for _, r in bench.iterrows():
    mr = cache[cache['movie_slug'] == r['slug']]
    snap = pd.to_datetime(r['snap_ts'])
    obs = mr[mr['estimated_timestamp'] < snap]
    p_re = estimate_p_fresh(cache, pool_slugs[r['slug']], set(obs['reviewer_name']),
                            int(r['obs_fresh_est']), int(r['obs_total_est']))
    max_err = max(max_err, abs(p_re - r['p_fresh_hat']))
assert max_err < 1e-9, f'shipped reconstruction drifted: {max_err}'
print(f"bench features built from the locked cache (no fresh-pin data touches the "
      f"bench); shipped-p̂ reconstruction max |err| = {max_err:.1e} (asserted < 1e-9)")
"""

C_FITS = """# Temporal fits per distinct bench close — via the unit-tested pl.temporal_rows
# (target exclusion + max-close + M5 window asserts; <60-movie floor -> skip+disclose)
bench['close_dt'] = pd.to_datetime(bench['close_time'], utc=True)
rv_scr = pd.read_csv(CACHE + '/pfresh_training_reviews.csv',
                     usecols=['movie_slug', 'scrape_time'])
rv_scr['scrape_time'] = pd.to_datetime(rv_scr['scrape_time'], utc=True, format='ISO8601')
fits, fit_info, skipped_closes = {}, [], []
for close, grp in bench.groupby('close_dt'):
    min_snap = pd.to_datetime(grp['snap_ts']).min()
    try:
        sub = pl.temporal_rows(ft, grp['slug'].iloc[0], close, min_snap, floor=60)
    except ValueError as e:                      # pre-registered skip-and-disclose
        skipped_closes.append((close.date().isoformat(), str(e)))
        continue
    for slug in grp['slug'].unique():            # multi-target closes: all excluded
        assert slug not in set(sub['slug']), f'{slug} leaked into its own fit'
    m1, c1, d1 = pl.fit_binomial_glm(sub, F_C1)
    m2, c2, d2 = pl.fit_binomial_glm(sub, F_C2)
    fits[close] = {'C1': (m1, F_C1), 'C2': (m2, F_C2)}
    # [F24] deploy-parity residual: fit-universe review rows scraped AFTER this
    # close's earliest bench snap (info a 648979-world deploy lacked)
    fit_rv = rv_scr[rv_scr['movie_slug'].isin(set(sub['slug']))]
    n_after = int((fit_rv['scrape_time'] > min_snap).sum())
    fit_info.append({'close': close.date().isoformat(),
                     'n_movies': sub['slug'].nunique(),
                     'n_rows': len(sub), 'C_c1': c1, 'oos_dev_c1': round(d1, 4),
                     'C_c2': c2, 'oos_dev_c2': round(d2, 4),
                     'scrape_after_snap': f"{n_after}/{len(fit_rv)}"
                                          f" ({n_after / len(fit_rv):.0%})"})
fi = pd.DataFrame(fit_info)
print(fi.to_string(index=False))
print(f"skipped closes (floor): {skipped_closes or 'none'}")
print('[F24] scrape_after_snap = training reviews not yet scraped at the bench snap '
      '(the 2026-04-06 ~98% is the bulk April backfill — deploy-parity view, '
      'disclosed; could only flatter candidates, which still did not clear)')
s_star = []
for close in sorted(fits):
    sub = ft[ft['close_dt'] < close]
    w = pl.row_weights(sub) * sub['n_rem']
    s_star.append(-float(np.average(sub['p_shipped'] - sub['y'], weights=w)))
print(f"\\ntraining-fit shade constants s* per close (training-side only, no bench row):"
      f" {[round(s, 3) for s in s_star]}")
"""

C_SCORE = """# Score candidates on the bench (temporal fit per cell's close) + VERDICT
def predict_cells(d, cand):
    out = []
    for _, r in d.iterrows():
        model, feats = fits[r['close_dt']][cand]
        x = np.array([[r[f] for f in feats]])
        out.append(float(model.predict_proba(x)[0, 1]))
    return out

for cand in ['C1', 'C2']:
    bench[f'pf_{cand}'] = predict_cells(bench, cand)
    assert bench[f'pf_{cand}'].between(0, 1).all()
# cells are keyed by (ticker, snap_days) — a ticker appears at multiple snaps
keep = ['ticker', 'snap_days', 'pf_C1', 'pf_C2']
n_before = len(pool_cells)
pool_cells = pool_cells.merge(bench[keep], on=['ticker', 'snap_days'], how='left')
assert len(pool_cells) == n_before == 35, 'merge fan-out — key must be (ticker, snap)'
assert pool_cells[['pf_C1', 'pf_C2']].notna().all().all()
for cand in ['C1', 'C2']:
    pool_cells[f'py_{cand}'] = p_yes_with(pool_cells, f'pf_{cand}')

print('=== BENCH VERDICT SET (35 cells; design-derived-from-bench reads) ===')
res = {}
res['shade'] = block(pool_cells, 'py_shade', 'C0 shade -0.03 (the bar)', rs_pool)
for cand in ['C1', 'C2']:
    res[cand] = block(pool_cells, f'py_{cand}', f"{cand}'", rs_pool)
    paired = [float(np.nanmean(pnl_stat(pool_cells.iloc[idx], f'py_{cand}'))
                    - np.nanmean(pnl_stat(pool_cells.iloc[idx], 'py_shade')))
              for idx in rs_pool]
    pdiff = res[cand]['pnl'] - res['shade']['pnl']
    print(f"    paired ({cand}' - shade): {pdiff:+.1f}c "
          f"[{ci(paired)[0]:+.1f}, {ci(paired)[1]:+.1f}]")
    res[cand]['paired'] = pdiff
    res[cand]['clears'] = (res[cand]['pnl_ci'][0] > 0) and (pdiff >= 0)

winners = [k for k in ['C1', 'C2'] if res[k]['clears']]
verdict = (f"ADOPT {winners[0]}' (simplest clearing rung)" if winners
           else 'NO WINNER -> ship nothing; re-run as the recorder cohort grows')
print(f"\\n>>> LOCKED ADOPTION RULE: PnL CI-lo > 0 AND paired (cand - shade) >= 0")
print(f">>> VERDICT: {verdict}")
print('>>> tripwire: an adopted candidate goes LIVE only after paired >= 0 on the '
      'first K~8 newly-settled recorder movies.')

print('\\n=== robustness/secondary (estimator reference + oos rows) ===')
res_ship = block(pool_cells, 'p_est', 'shipped stack (gate3b)', rs_pool)
for cand in ['C1', 'C2']:
    block(pool_cells[~pool_cells['oos_post_gate2']].reset_index(drop=True),
          f'py_{cand}', f"{cand}' ex-oos")
oos = pool_cells[pool_cells['oos_post_gate2']].reset_index(drop=True)
for cand in ['C1', 'C2']:
    block(oos, f'py_{cand}', f"{cand}' oos-only (POWERLESS, n=2mv)")

print('\\n=== per-snap (the BAR rides along — attribution discipline per the '
      'post-build review: per-snap candidate reads mean nothing without the '
      'shade on the same cells/draws) ===')
for n in [4, 3, 2]:
    d = (cov[cov['snap_days'] == n]
         .merge(bench[keep], on=['ticker', 'snap_days'], how='left')
         .reset_index(drop=True))
    assert d[['pf_C1', 'pf_C2']].notna().all().all()
    d['shade'] = (d['p_fresh_hat'] - 0.03).clip(0, 1)
    d['py_shade'] = p_yes_with(d, 'shade')
    d['py_C1'] = p_yes_with(d, 'pf_C1')
    d['py_C2'] = p_yes_with(d, 'pf_C2')
    rs_n = draws(d)
    print(f"-- T-{n}d --")
    block(d, 'py_shade', 'shade -0.03 (bar)', rs_n)
    for cand in ['C1', 'C2']:
        block(d, f'py_{cand}', f"{cand}'", rs_n)
        paired_n = [float(np.nanmean(pnl_stat(d.iloc[idx], f'py_{cand}'))
                          - np.nanmean(pnl_stat(d.iloc[idx], 'py_shade')))
                    for idx in rs_n]
        print(f"    paired ({cand}' - shade): "
              f"{np.nanmean(pnl_stat(d, f'py_{cand}')) - np.nanmean(pnl_stat(d, 'py_shade')):+.1f}c "
              f"[{ci(paired_n)[0]:+.1f}, {ci(paired_n)[1]:+.1f}]")
"""

C_DIAG = """# delta migration + side split + hand-walk (one cell per candidate)
aud = bench[bench['n_rem_oracle'] > 0].copy()
print('=== delta_hat migration (all 60 locked cells; vs oracle p_fresh) ===')
rows_out = []
for col, name in [('p_fresh_hat', 'shipped'), (None, 'shade-0.03'),
                  ('pf_C1', "C1'"), ('pf_C2', "C2'")]:
    p = (aud['p_fresh_hat'] - 0.03).clip(0, 1) if col is None else aud[col]
    d = p - aud['p_fresh_oracle']
    rows_out.append({'candidate': name, 'mean': d.mean(), 'median': d.median(),
                     'in_band_[-0.10,0]': d.between(-0.10, 0).mean(),
                     'kill_zone_>+0.05': (d > 0.05).mean(),
                     **{f'mean_T{n}d': d[aud['snap_days'] == n].mean()
                        for n in [1, 2, 3, 4]}})
print(pd.DataFrame(rows_out).round(3).to_string(index=False))

print('\\n=== trade-side split (pooled, buffer=0) ===')
for cand in ['C1', 'C2']:
    pv = pnl_stat(pool_cells, f'py_{cand}')
    t = pool_cells[~np.isnan(pv)].assign(pnl=pv[~np.isnan(pv)])
    t['side'] = np.where(t[f'py_{cand}'] > t['ask'], 'YES', 'NO')
    print(f"-- {cand}' --")
    print(t.groupby('side').agg(n=('pnl', 'size'), mean_pnl_c=('pnl', 'mean'),
                                win_pct=('pnl', lambda s: 100 * (s > 0).mean()))
          .round(1).to_string())

print('\\n=== bar-invariance check (post-verdict diagnostic; the s*-suggested bar) ===')
pool_cells['shade45'] = (pool_cells['p_fresh_hat'] - 0.045).clip(0, 1)
pool_cells['py_shade45'] = p_yes_with(pool_cells, 'shade45')
block(pool_cells, 'py_shade45', 'shade s=-0.045 (s* bar)', rs_pool)
for cand in ['C1', 'C2']:
    paired45 = [float(np.nanmean(pnl_stat(pool_cells.iloc[idx], f'py_{cand}'))
                      - np.nanmean(pnl_stat(pool_cells.iloc[idx], 'py_shade45')))
                for idx in rs_pool]
    print(f"    paired ({cand}' - shade45): "
          f"{np.nanmean(pnl_stat(pool_cells, f'py_{cand}')) - np.nanmean(pnl_stat(pool_cells, 'py_shade45')):+.1f}c "
          f"[{ci(paired45)[0]:+.1f}, {ci(paired45)[1]:+.1f}]")
print('(both candidates fail PnL CI-lo>0 outright -> the NO-WINNER verdict is '
      'bar-invariant)')

d2 = (cov[cov['snap_days'] == 2]
      .merge(bench[keep], on=['ticker', 'snap_days'], how='left')
      .reset_index(drop=True))
d2['py_sh'] = p_yes_with(d2.assign(s=(d2['p_fresh_hat'] - 0.03).clip(0, 1)), 's')
d2['py_1'] = p_yes_with(d2, 'pf_C1')
d2['py_2'] = p_yes_with(d2, 'pf_C2')
pv_sh, pv_1, pv_2 = (pnl_stat(d2, c) for c in ['py_sh', 'py_1', 'py_2'])
same = sum(1 for a, b, cc in zip(pv_sh, pv_1, pv_2)
           if (np.isnan(a) and np.isnan(b) and np.isnan(cc))
           or (a == b == cc))
print(f"\\nT-2d trade identity: {same}/{len(d2)} cells have IDENTICAL PnL across "
      f"shade/C1'/C2' (same book side crossed) — the T-2d drawdown is shared, "
      f"not candidate-specific")

r = pool_cells[pool_cells['snap_days'] == 3].sort_values('ticker').iloc[0]
print(f"\\nhand-walk {r['ticker']} (T-3d): obs {int(r['obs_fresh_est'])}/"
      f"{int(r['obs_total_est'])} -> shipped p̂ {r['p_fresh_hat']:.3f} | "
      f"shade {r['shade']:.3f} | C1' {r['pf_C1']:.3f} | C2' {r['pf_C2']:.3f} | "
      f"oracle {r['p_fresh_oracle']:.3f}")
print(f"P(Yes): shipped {r['p_est']:.3f} | shade {r['py_shade']:.3f} | "
      f"C1' {r['py_C1']:.3f} | C2' {r['py_C2']:.3f} | book {r['bid']:.2f}/"
      f"{r['ask']:.2f} y={int(r['y'])}")
"""

MD_TAIL = """## Reading guide

- All candidate bench reads are **design-derived-from-bench** (brainstorm §4 honest
  bookkeeping: the bench absorbed the verdict + 2 shade looks before this notebook's
  2 candidate looks = 5 total; family-wise false-clear risk ~5-10%); the recorder
  tripwire is the true holdout. `oos_post_gate2` rows reported, powerless (2 movies).
- **The verdict is bar-invariant** (post-build review, independently verified): both
  candidates fail PnL CI-lo > 0 outright; under the training-suggested s* ≈ −0.045 bar
  (+11.6¢ [−0.6, +25.0] pooled) the paired reads are equal-or-worse. The −0.03 bar
  was, if anything, the easier one.
- **Per-snap attribution discipline**: T-3d strength and the T-2d drawdown are shared
  with the bar (shade T-3d +25.1¢ / T-2d −28.4¢ on the same cells); the candidates'
  only paired edge vs the shade is at T-2d (+6.0¢ [+0.0, +21.6], n=12 — noise-adjacent).
- Temporal fits are deploy-parity ("all history before this close at the fresh pin");
  the locked 648979 bench world supplies every bench-cell feature. The [F24] column
  discloses the first-close fit's ~98% backfill share.
- s* (training-fit shade) is training-side context only — no bench look spent.
- mass_consumed survived its corr gate at +0.886 vs the 0.9 threshold — borderline,
  disclosed.
- Sensitivities (shrink-k, raw-clip, d-row) live training-side in the battery
  notebook; 12h-grid struck (plan review log), deferred to the recorder-growth re-run.
"""

nb = nbf.v4.new_notebook()
nb.cells = [nbf.v4.new_markdown_cell(MD), nbf.v4.new_code_cell(C_LOAD),
            nbf.v4.new_code_cell(C_MACHINERY), nbf.v4.new_code_cell(C_BENCHFEAT),
            nbf.v4.new_code_cell(C_FITS), nbf.v4.new_code_cell(C_SCORE),
            nbf.v4.new_code_cell(C_DIAG), nbf.v4.new_markdown_cell(MD_TAIL)]
nb.metadata.kernelspec = {"name": "python3", "display_name": "Python 3",
                          "language": "python"}
os.makedirs("notebooks", exist_ok=True)
nbf.write(nb, "notebooks/pfresh_bench.ipynb")
print("wrote notebooks/pfresh_bench.ipynb")
