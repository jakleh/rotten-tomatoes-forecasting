"""Authoring helper: assemble notebooks/gate3b_deployable.ipynb from cell sources.

Codegen only — the analysis lives in the notebook (nbconvert-executed; cache-only +
the library + gates modules, NO network/DB). Spec: ``plans/plan_gate3b.md`` (twice
adversarially reviewed; decision rule pre-registered there and restated verbatim in
the intent cell BEFORE any result is computed). Inputs are the driver's locked caches
(``gates/build_gate3b.py`` — the lock chain ran there; this notebook only SCORES).
"""
import os
import nbformat as nbf

MD = """# Gate 3b — the deployable-stack verdict (real 0.2.0 estimator vs the book)

**Question:** does the ACTUAL shipped stack (`extract_lambda_features` →
`estimate_lambda` (shipped artifact) → `estimate_p_fresh` → `compute_edge`), run
honestly at its own midnight-ET decision time, beat the state-at-snap book — and what
fraction of the in-grid oracle ceiling does it capture? Companion audit: per-cell
λ/p_fresh error (m̂, δ̂) against the revised Gate-3a tolerance bands.

**Pre-registered decision rule (locked in `plans/plan_gate3b.md` BEFORE any estimator
P(Yes) was computed; the driver enforced the lock chain grid → guard/floor → oracle →
estimator):**
- **CLEARS** = pooled-unique-market (snap priority 3d>2d>4d; **T-1d excluded**)
  **PnL CI-lower > 0 AND Brier-diff CI-lower > 0** vs the state-at-snap book —
  movie-cluster bootstrap, **2000 resamples, shared across all statistics, seed 7**.
- Headline tables apply the 15d deployment gap-cap (`trimmed` ≠ `skip:*`); trades
  cross the spread (Yes at ask / No at 100−bid) net of taker fee `7·p·(1−p)`¢.
- **Covered-cell intersection rule:** every estimator-vs-oracle statistic (capture
  fraction, Brier/PnL gaps) runs on the cells the estimator PRICES — same cells, same
  resamples, both stacks. The all-clean-cell in-grid oracle is context only.
- **Coverage guard:** pooled coverage < 0.5 ⇒ headline becomes "the stack
  under-covers the arena" regardless of covered-cell metrics.
- PnL-clears-but-Brier-doesn't ⇒ "trades profitably, distorts probabilities" —
  deployable with a flag. Neither clears ⇒ the gap to the in-grid ceiling + the m̂/δ̂
  decomposition IS the improvement target (never "abandon"; the ceiling passed).
- Conservative-shade rows are a **post-verdict bench re-score** inside the audit
  section — they never enter CLEARS.

Grid: midnight-ET snaps (Option B, locked), 19-movie cohort − `data_not_ready`
(animal_farm_2025 / backrooms / in_the_grey fail the settlement-consistency check;
power_ballad by operator call) = **17 effective movies**. Reviews/guard pinned at
`as_of_id=648979`; books from the committed `gates/recorded/` store. Labels: Kalshi
`result`.
"""

C_LOAD = """import os, sys, warnings
os.environ.setdefault('MPLCONFIGDIR', os.environ.get('TMPDIR', '/tmp') + '/mpl')
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', message='Mean of empty slice')  # zero-trade boot draw
import numpy as np, pandas as pd
sys.path.insert(0, os.path.abspath('..') if os.path.basename(os.getcwd()) == 'notebooks' else os.path.abspath('.'))

CACHE = 'gates/_cache' if os.path.isdir('gates/_cache') else '../gates/_cache'
STORE = 'gates/recorded' if os.path.isdir('gates/recorded') else '../gates/recorded'
c = pd.read_csv(CACHE + '/gate3b_cells.csv')
meta = pd.read_csv(CACHE + '/gate3b_meta.csv').iloc[0]
rd = pd.read_csv(CACHE + '/gate3b_readiness.csv')

PIN = int(meta['as_of_id'])
assert PIN == 648979 and (c['as_of_id'] == PIN).all()
assert set(c['status'].unique()) <= {'ok', 'skip:features', 'skip:lambda', 'trimmed'}
assert (c['primary'] == c['snap_days'].isin([2, 3, 4])).all()
PRIMARY = [int(s) for s in str(meta['primary_snaps']).split(';')]
SEED, N_BOOT = 7, 2000

print(f"locked grid @ as_of_id={PIN} | data_not_ready = {meta['data_not_ready']}")
print(f"T-3d retention vs Gate-2 grid: {int(meta['retention_t3_num'])}/"
      f"{int(meta['retention_t3_den'])} = "
      f"{meta['retention_t3_num']/meta['retention_t3_den']:.0%} "
      f"(readiness-applied {int(meta['retention_t3_num_ready'])}/"
      f"{int(meta['retention_t3_den'])})")
g = c.groupby('snap_days')
print(g.agg(markets=('ticker', 'size'), movies=('slug', 'nunique')).to_string())
print('\\nstatus by snap:')
print(c.groupby(['snap_days', 'status']).size().unstack(fill_value=0).to_string())

# coverage (precise rule): numerator = cells the stack CANNOT price (skip:*),
# denominator = the locked oracle-clean AND ct cells; trimmed counts as priceable
prio = {3: 0, 2: 1, 4: 2}
pooled_all = (c[c['primary']].assign(p=lambda d: d['snap_days'].map(prio))
              .sort_values('p').drop_duplicates('ticker'))
cov_pooled = 1 - pooled_all['status'].str.startswith('skip').mean()
print('\\nper-snap coverage (1 - skips/cells):')
print((1 - c.groupby('snap_days')['status'].apply(lambda s: s.str.startswith('skip').mean())).round(3).to_string())
print(f"POOLED coverage (unique markets, 3d>2d>4d): {cov_pooled:.1%} "
      f"({int((~pooled_all['status'].str.startswith('skip')).sum())}/{len(pooled_all)})"
      + ('' if cov_pooled >= 0.5 else '  ** UNDER-COVERAGE GUARD TRIPPED **'))
assert cov_pooled >= 0.5, 'coverage guard: headline reframes to under-coverage'
n_trim = int((c['status'] == 'trimmed').sum())
print(f"trimmed (15d gap-cap): {n_trim} cells"
      + (' -> capped and uncapped tables are identical' if n_trim == 0 else ''))
print(f"\\nreadiness at pin {PIN} (per movie):")
print(rd[rd['pin'] == PIN][['slug', 'score_at_pin', 'implied_lo', 'implied_hi',
                            'lastday_d_at_pin', 'ledger_lastday', 'pass']]
      .to_string(index=False))
"""

C_MACHINERY = """# Shared Brier/PnL machinery + the Gate-2 regression anchor.
# The anchor: pointed back at the untouched 648979 close-24N caches, THIS notebook's
# functions must reproduce the published REVISED Gate-2 numbers (point estimates; CIs
# are rng-stream-dependent). Anchor pin is decoupled from the build pin by design.
def fee_cents(price_c):
    return 7.0 * (price_c / 100.0) * (1 - price_c / 100.0)

def pnl_c(p, y, bid, ask, buffer_c=0.0):
    \"\"\"Taker PnL (cents) crossing the resting book at probability p, else NaN.\"\"\"
    pc, ask_c, bid_c = p * 100, ask * 100, bid * 100
    if pc > ask_c + buffer_c:
        return (100 * y - ask_c) - fee_cents(ask_c)
    if pc < bid_c - buffer_c:
        no_c = 100 - bid_c
        return (100 * (1 - y) - no_c) - fee_cents(no_c)
    return np.nan

def draws(d, seed=SEED, n=N_BOOT):
    \"\"\"Pre-drawn movie-cluster resamples (row-index arrays) — drawn ONCE per cell
    set and reused by every statistic on that set (paired comparisons).\"\"\"
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
    v = np.array([pnl_c(p, y, b, a, buffer_c) for p, y, b, a
                  in zip(d[pcol], d['y'], d['bid'], d['ask'])])
    return v   # per-cell PnL (NaN = no trade)

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
    return {'brier': bd, 'brier_ci': bd_ci, 'pnl': pnl, 'pnl_ci': pnl_ci,
            'n_trades': len(t)}

g2 = pd.read_csv(CACHE + '/gate2_cells.csv')
assert len(g2) == 124, 'expected the ex-animal_farm revised gate2_cells.csv'
g2p = (g2[(g2['mode'] == 'pure') & (g2['snap'] != '1d')]
       .assign(p=lambda d: d['snap'].map({'3d': 0, '2d': 1, '4d': 2}))
       .sort_values('p').drop_duplicates('ticker'))
print('=== Gate-2 regression anchor (untouched 648979 close-24N caches) ===')
a_pool = block(g2p, 'p_oracle', 'pooled (3d>2d>4d)')
a_3d = block(g2[(g2['mode'] == 'pure') & (g2['snap'] == '3d')], 'p_oracle', 'T-3d')
assert round(a_pool['brier'], 4) == 0.1247 and round(a_pool['pnl'], 1) == 32.1, \\
    'pooled anchor drifted from the published revised Gate-2 numbers'
assert round(a_3d['brier'], 4) == 0.0883 and round(a_3d['pnl'], 1) == 27.1, \\
    'T-3d anchor drifted from the published revised Gate-2 numbers'
print('anchor reproduces the published REVISED Gate-2 point estimates exactly '
      '(+0.1247/+32.1c pooled, +0.0883/+27.1c T-3d); CIs are rng-stream-dependent.')
"""

C_HANDWALK = """# Hand-walk ONE covered cell end-to-end from the caches (features -> lambda ->
# p_fresh -> P(Yes) -> trade), asserting equality with the driver's CSV row.
from rotten_tomatoes_forecasting import (compute_edge, estimate_lambda,
                                         estimate_p_fresh, extract_lambda_features,
                                         load_default_regressor)
from rotten_tomatoes_forecasting.pool import build_a1_pool_context

cache = pd.read_csv(CACHE + '/gate3b_a1_cache.csv')
# estimator view: estimated_timestamp is ALREADY noon-shifted at ingest (never re-shift)
cache['estimated_timestamp'] = pd.to_datetime(cache['estimated_timestamp'], utc=True, format='ISO8601')
act = dict(pd.read_csv(CACHE + '/gate3b_activity.csv').itertuples(index=False, name=None))
mi = pd.read_csv(('movies_index.csv' if os.path.exists('movies_index.csv')
                  else '../movies_index.csv'))
mk = pd.read_csv(STORE + '/markets.csv')
cdm = {**dict(zip(mi['Slug'], pd.to_datetime(mi['Bet Close Date'], utc=True))),
       **{s: pd.to_datetime(t, utc=True)
          for s, t in mk.groupby('slug')['close_time'].first().items()}}

r = (c[(c['snap_days'] == 3) & (c['status'] == 'ok')]
     .sort_values('ticker').iloc[0])
slug, n, close = r['slug'], int(r['snap_days']), pd.to_datetime(r['close_time'], utc=True)
snap_ts = pd.to_datetime(r['snap_ts'])
ctx = build_a1_pool_context(slug, cdm, cache)
feats = extract_lambda_features(slug, snap_days=n, close_ts=close, reviews_df=cache,
                                close_date_map=cdm, a1_context=ctx, activity_lookup=act)
print(f"cell: {r['ticker']} (X={int(r['X'])}) at T-{n}d snap {r['snap_ts']} "
      f"(h={r['h']:.0f})\\nfeatures:")
for k, v in feats.items():
    print(f"  {k:<26} {v:.4f}")
pred = estimate_lambda(load_default_regressor(), feats, snap_days=n, close_ts=close,
                       hours_to_close=float(r['h']))
tr = cache[cache['movie_slug'] == slug]
obs = tr[tr['estimated_timestamp'] < snap_ts]
fresh, total = int((obs['tomatometer_sentiment'] == 'positive').sum()), len(obs)
p_hat = estimate_p_fresh(cache, ctx.training_slugs, set(obs['reviewer_name']), fresh, total)
e = compute_edge(int(r['X']), float(r['mid']) * 100, fresh, total, float(r['h']),
                 pred.rate_per_hour, p_hat)
print(f"\\nlambda: phase1 {pred.phase1_pred:.2f} + phase2 {pred.phase2_pred:.2f} "
      f"= total {pred.total_pred:.2f} ({pred.rate_per_hour:.4f}/h)"
      f"\\np_fresh_hat: {p_hat:.4f} (obs {fresh}/{total})"
      f"\\nP(Yes): {e['p_yes']:.4f} | book bid/ask {r['bid']:.2f}/{r['ask']:.2f} "
      f"(mid {r['mid']:.3f})")
assert abs(pred.total_pred - r['total_pred']) < 1e-9
assert abs(pred.rate_per_hour - r['rate_hat']) < 1e-12
assert abs(p_hat - r['p_fresh_hat']) < 1e-12
assert abs(e['p_yes'] - r['p_est']) < 1e-9
assert total == int(r['obs_total_est']) and fresh == int(r['obs_fresh_est'])
side = ('buy YES at ask' if e['p_yes'] * 100 > r['ask'] * 100
        else 'buy NO at 100-bid' if e['p_yes'] * 100 < r['bid'] * 100 else 'no trade')
pnl = pnl_c(e['p_yes'], int(r['y']), r['bid'], r['ask'])
print(f"trade: {side} -> y={int(r['y'])} -> PnL {pnl:+.1f}c "
      f"(oracle P={r['p_oracle']:.3f}; m_hat={r['m_hat']:.2f}, "
      f"delta_hat={r['delta_hat']:+.3f})")
print('hand-walk reproduces the driver row exactly (asserted at 1e-9).')
"""

C_HEADLINE = """# HEADLINE: the shipped stack vs the state-at-snap book.
# Covered = status 'ok' (gap-cap applied; trimmed=0 here so capped==uncapped).
cov = c[c['status'] == 'ok'].copy()
print('=== per-snap (covered cells; estimator vs book) ===')
for n in [4, 3, 2, 1]:
    d = cov[cov['snap_days'] == n].reset_index(drop=True)
    if not len(d):
        continue
    tag = ' (secondary)' if n == 1 else ''
    rs = draws(d)
    print(f"-- T-{n}d{tag} --")
    block(d, 'p_est', 'estimator', rs)
    block(d, 'p_est', 'estimator buffer=5c', rs, buffer_c=5.0)

pool = (cov[cov['primary']].assign(p=lambda d: d['snap_days'].map(prio))
        .sort_values('p').drop_duplicates('ticker').reset_index(drop=True))
rs_pool = draws(pool)
n_unpriceable = pooled_all['ticker'].nunique() - pool['ticker'].nunique()
print(f"\\n=== POOLED unique markets (3d>2d>4d; T-1d excluded) — THE VERDICT SET "
      f"(n={len(pool)}/{pool['slug'].nunique()}mv, seed {SEED}, {N_BOOT} shared "
      f"resamples) ===")
print(f"(deploy semantics: a market skipped at its priority snap enters at its next "
      f"PRICED primary snap; {n_unpriceable} of {pooled_all['ticker'].nunique()} "
      f"markets priceable at NO primary snap are out entirely)")
v = block(pool, 'p_est', 'estimator vs book', rs_pool)
block(pool, 'p_est', 'estimator buffer=5c', rs_pool, buffer_c=5.0)

clears_pnl, clears_brier = v['pnl_ci'][0] > 0, v['brier_ci'][0] > 0
verdict = ('CLEARS (PnL AND Brier CI-lower > 0): the deployable stack beats the book'
           if clears_pnl and clears_brier else
           'TRADES PROFITABLY, DISTORTS PROBABILITIES (PnL clears, Brier does not): '
           'deployable with a flag'
           if clears_pnl else
           'BRIER CLEARS, PnL DOES NOT: probability-better but not tradeable through '
           'the spread at this n'
           if clears_brier else
           'DOES NOT CLEAR: the gap to the in-grid ceiling + the m/delta decomposition '
           'is the improvement target (ceiling already passed; never abandon)')
print(f"\\n>>> PRE-REGISTERED VERDICT: {verdict}")

print('\\n=== robustness rows (pooled covered set) ===')
billie = [s for s in c['slug'].unique() if 'billie' in s]
block(pool[~pool['slug'].isin(billie)].reset_index(drop=True), 'p_est', 'ex-billie')
block(pool[~pool['oos_post_gate2']].reset_index(drop=True), 'p_est',
      'ex-oos (16-movie Gate-2 universe)')
block(pool[~pool['in_ridge_fit']].reset_index(drop=True), 'p_est',
      'ex-Ridge-fit-overlap movies')
pv = pnl_stat(pool, 'p_est')
t = pool[~np.isnan(pv)].assign(pnl=pv[~np.isnan(pv)])
t['side'] = np.where(t['p_est'] > t['ask'], 'YES', 'NO')
print('\\ntrade-side split (pooled, buffer=0):')
print(t.groupby('side').agg(n=('pnl', 'size'), movies=('slug', 'nunique'),
                            mean_pnl_c=('pnl', 'mean'),
                            win_pct=('pnl', lambda s: 100 * (s > 0).mean()),
                            mean_entry_mid=('mid', 'mean')).round(1).to_string())
"""

C_CEILING = """# Ceiling comparison — covered-cell intersection rule: same cells, same resamples,
# both stacks (the all-clean-cell oracle is reported separately as context, never as
# the denominator under a selected numerator).
print(f"=== estimator vs IN-GRID oracle on the covered pooled cells (n={len(pool)}, "
      f"paired draws) ===")
b_est = block(pool, 'p_est', 'estimator vs book', rs_pool)
b_orc = block(pool, 'p_oracle', 'in-grid oracle vs book', rs_pool)
pnl_e = [float(np.nanmean(pnl_stat(pool.iloc[idx], 'p_est'))) for idx in rs_pool]
pnl_o = [float(np.nanmean(pnl_stat(pool.iloc[idx], 'p_oracle'))) for idx in rs_pool]
gap = [o - e for e, o in zip(pnl_e, pnl_o)]
print(f"  PnL gap (oracle - estimator): {b_orc['pnl'] - b_est['pnl']:+.1f}c "
      f"[{ci(gap)[0]:+.1f}, {ci(gap)[1]:+.1f}] (paired)")
cap = [e / o for e, o in zip(pnl_e, pnl_o) if o > 0]
if len(cap) >= 0.975 * N_BOOT:
    print(f"  capture fraction (PnL_est/PnL_oracle): "
          f"{b_est['pnl'] / b_orc['pnl']:.0%} [{ci(cap)[0]:.0%}, {ci(cap)[1]:.0%}] "
          f"(paired; ratio defined in {len(cap)}/{N_BOOT} draws)")
else:
    print(f"  capture fraction: point {b_est['pnl'] / b_orc['pnl']:.0%}; ratio CI "
          f"suppressed (oracle PnL <= 0 in {N_BOOT - len(cap)}/{N_BOOT} draws)")

print('\\n=== context: in-grid oracle on ALL locked cells (incl. estimator-skipped) ===')
all_pool = (c[c['primary']].assign(p=lambda d: d['snap_days'].map(prio))
            .sort_values('p').drop_duplicates('ticker').reset_index(drop=True))
rs_all = draws(all_pool)
block(all_pool, 'p_oracle', 'pure oracle (pooled)', rs_all)
block(all_pool, 'p_oracle_lagged', 'lagged oracle (context)', rs_all)
print('NOTE: this in-grid ceiling is the like-for-like reference; Gate-2 pooled '
      '+32.1c lives on the close-24N grid (out-of-grid reference only).')
"""

C_AUDIT = """# Error decomposition + p_fresh audit (BACKLOG 1.4) — runs on ALL locked cells
# (p_fresh audit is decoupled from lambda-skip selection; delta excludes the inert
# n_rem==0 sentinel, of which this grid has zero).
print(f"n_rem_oracle==0 sentinel cells (delta undefined): "
      f"{int((c['n_rem_oracle'] == 0).sum())}")
aud = c[c['n_rem_oracle'] > 0].copy()
LAM_BAND, PF_BAND = (0.55, 3.0), (-0.10, 0.0)   # REVISED Gate-3a bands (2026-06-10)
aud['lam_in'] = aud['m_hat'].between(*LAM_BAND)
aud['pf_in'] = aud['delta_hat'].between(*PF_BAND)

print('\\n=== m_hat = total_pred / n_remaining_oracle (3a band axis) ===')
q = aud[aud['m_hat'].notna()].groupby('snap_days')['m_hat'].describe(
    percentiles=[0.25, 0.5, 0.75])[['count', '25%', '50%', '75%', 'min', 'max']]
print(q.round(2).to_string())
print(f"fraction inside lambda band m in [{LAM_BAND[0]}, {LAM_BAND[1]}]: "
      f"{aud['lam_in'].mean():.0%} (n={aud['m_hat'].notna().sum()})")
m_conv = aud['total_pred'] / aud['realized_est_conv'].replace(0, np.nan)
print(f"secondary (estimator-convention realized): median m_conv "
      f"{m_conv.median():.2f}; realized==0 cells excluded: "
      f"{int((aud['realized_est_conv'] == 0).sum())}")

print('\\n=== delta_hat = p_fresh_hat - p_fresh_oracle (3a band axis) ===')
t = aud.groupby('snap_days')['delta_hat'].agg(['count', 'mean', 'median', 'std'])
print(t.round(3).to_string())
print(f"pooled: mean {aud['delta_hat'].mean():+.3f} median "
      f"{aud['delta_hat'].median():+.3f} | sign +/-: "
      f"{int((aud['delta_hat'] > 0).sum())}/{int((aud['delta_hat'] < 0).sum())}")
print(f"fraction inside p_fresh band delta in [{PF_BAND[0]}, {PF_BAND[1]}]: "
      f"{aud['pf_in'].mean():.0%} | fraction in the kill zone (delta > +0.05): "
      f"{(aud['delta_hat'] > 0.05).mean():.0%}")
print(f"strict joint band (lambda IN and p_fresh IN): "
      f"{(aud['lam_in'] & aud['pf_in']).mean():.0%}")

print('\\n=== 2x2: band membership vs per-cell PnL (covered traded cells) ===')
av = aud[aud['status'] == 'ok'].copy()
av['pnl'] = pnl_stat(av, 'p_est')
tr2 = av[av['pnl'].notna()]
print(tr2.groupby(['lam_in', 'pf_in']).agg(
    n=('pnl', 'size'), mean_pnl_c=('pnl', 'mean'),
    win_pct=('pnl', lambda s: 100 * (s > 0).mean())).round(1).to_string())

print('\\n=== p_fresh audit extras ===')
print(f"|p_fresh_hat - p_fresh_hat_le_snap| (pool post-snap-tail sensitivity): "
      f"mean {np.abs(aud['p_fresh_hat'] - aud['p_fresh_hat_le_snap']).mean():.4f}, "
      f"max {np.abs(aud['p_fresh_hat'] - aud['p_fresh_hat_le_snap']).max():.4f}")
print(f"pool rows with est > snap (deploy-unseen tail): median "
      f"{aud['pool_tail_frac'].median():.2%}, max {aud['pool_tail_frac'].max():.2%}")
terc = pd.qcut(aud['obs_total_est'].rank(method='first'), 3,
               labels=['low', 'mid', 'high'])
print('\\nEXPLORATORY (hypothesis-only, feeds the p_fresh-regression riff): '
      'delta_hat by observed-count tercile x snap')
print(aud.groupby([terc, 'snap_days'], observed=True)['delta_hat']
      .agg(['count', 'mean']).round(3).to_string())
"""

C_SHADE = """# CONSERVATIVE-SHADE re-score — POST-VERDICT BENCH ROW (plan review I7: never
# enters CLEARS, never revises the 3b verdict; the asymmetric 3a band makes a constant
# negative shade on p_fresh_hat a candidate one-line fix — quantified here).
from rotten_tomatoes_forecasting import compute_edge as _ce

def p_yes_shaded(row, s):
    pf = float(np.clip(row['p_fresh_hat'] + s, 0.0, 1.0))
    return _ce(int(row['X']), float(row['mid']) * 100, int(row['obs_fresh_est']),
               int(row['obs_total_est']), float(row['h']), float(row['rate_hat']),
               pf)['p_yes']

chk = pool.apply(lambda r: p_yes_shaded(r, 0.0), axis=1)
assert np.allclose(chk, pool['p_est'], atol=1e-9), 'shade reconstruction drifted'
print('reconstruction at s=0 reproduces p_est (1e-9). Shade rows (same pooled cells, '
      'same resamples):')
res = {}
for s in [0.0, -0.03, -0.05]:
    d = pool.assign(p_shade=pool.apply(lambda r: p_yes_shaded(r, s), axis=1))
    res[s] = block(d, 'p_shade', f'shade s={s:+.2f}', rs_pool)
"""

MD_TAIL = """## Reading guide / caveats

- **Directional**: ≤13 covered movies per snap; the bench (cells + books + in-grid
  oracle + this machinery) re-scores any future p_fresh/λ variant by re-running only
  the estimator pass over the locked `gate3b_cells.csv`.
- The estimator observes raw `est < snap` while the oracle defers m/h rows by +1min —
  a ≤1-minute information sliver kept deliberately (fit-convention parity; adding
  +1min to the estimator would be train/serve skew).
- Books are the honest at-snap state with their staleness (carried quotes at midnight
  are ~1-2 min old per the pre-lock measurement); taker-crossing PnL, no depth/refill
  assumption. This gate measures the ESTIMATOR stack, not the full strategy layer
  (which adds further filters on top of the 15d gap-cap).
- Oracle-conditioned reads are the forecaster's PRIZE; only observable-conditioned
  gaps are tradeable inefficiency (axis-language discipline).
- `in_ridge_fit` movies (closes before the 2026-04-19 artifact fit) are IN the λ
  model's training set — membership re-derived from `movies_index.csv`, cross-checked
  against `metadata.cohort_size=144`; residual movies_index-drift risk disclosed.
- The A1 pool (20 most recent resolved before target close; `gate3b_pools.csv`) is a
  deploy decision being measured, not tuned; pool members' full histories are kept
  (fit parity), with the deploy-unseen post-snap tail quantified above.
"""

nb = nbf.v4.new_notebook()
nb.cells = [nbf.v4.new_markdown_cell(MD), nbf.v4.new_code_cell(C_LOAD),
            nbf.v4.new_code_cell(C_MACHINERY), nbf.v4.new_code_cell(C_HANDWALK),
            nbf.v4.new_code_cell(C_HEADLINE), nbf.v4.new_code_cell(C_CEILING),
            nbf.v4.new_code_cell(C_AUDIT), nbf.v4.new_code_cell(C_SHADE),
            nbf.v4.new_markdown_cell(MD_TAIL)]
nb.metadata.kernelspec = {"name": "python3", "display_name": "Python 3",
                          "language": "python"}
os.makedirs("notebooks", exist_ok=True)
nbf.write(nb, "notebooks/gate3b_deployable.ipynb")
print("wrote notebooks/gate3b_deployable.ipynb")
