"""Authoring helper: assemble notebooks/gate2_oracle.ipynb from cell sources.

Codegen only — the analysis lives in the notebook (nbconvert-executed; reads only
gates/_cache/ + the library + gates.oracle). Spec: plans/plan_gate_1_2_calibration.md
— "Pass criteria" (Brier+PnL 2×2), "Verification + review amendments" (two oracles),
"Gate-2 stratification layer" (prize / encompassing / terciles), "Pre-registered
dense-cohort floor" (cell set = oracle-clean ∧ contested ∧ ≤10c at T-2d/3d/4d,
T-1d secondary).
"""
import os
import nbformat as nbf

MD = """# Gate 2 — oracle λ/p_fresh through `compute_edge` vs the market (directional)

**Question:** with the BEST inputs that exist (oracle λ = realized remaining review
rate, oracle p_fresh = realized remaining fresh-rate — the MLE), does the
Poisson×Binomial architecture beat the state-at-snap book on the tradeable arena?
Dispersion stays Poisson×Binomial — this is the **real-time-forecaster ceiling**, not
perfect foresight.

- **Cells:** oracle-clean movies (pre-registered guard, `gate2_density.ipynb`) ×
  contested (0.2<mid<0.8) ∧ spread≤10¢ state-at-snap books, snaps **T-2d/T-3d/T-4d**
  (primary; T-3d central) + T-1d (secondary). One obs per market within a snap.
- **Two oracles:** `pure` (publication-time +1min — architecture ceiling, headline) and
  `lagged` (scrape-time visibility — current-pipeline reality). Gap = value of faster
  scraping.
- **Benchmark:** the actual at-snap book with its staleness; PnL entries CROSS the
  spread (Yes at ask / No at 100−bid) net of the taker fee `ceil(0.07·P·(1−P))`
  (`kalshi-trading/src/kalshi/fees.py`, fee-schedule PDF eff. 2026-02-05; KXRT series
  `fee_multiplier=1`, `fee_type=quadratic` per the public API 2026-06-09 → standard
  rate; per-contract estimate `7·p·(1−p)` cents).
- **Verdict frame:** Brier × PnL 2×2, cluster-boot by movie; **asymmetric fork** — only
  a clear fail (both CIs clearly < 0 AND a clean form-diagnostic) abandons;
  inconclusive NEVER does. Power context: at n≈13 movies and a true 25% Brier
  improvement with moderate oracle noise, ~half of runs read inconclusive
  (`gate2_density.ipynb`).
- Labels: Kalshi `result`. Reviews/density pinned at `as_of_id=648979`.
"""

C_CELLS = """import os, sys, warnings
os.environ.setdefault('MPLCONFIGDIR', os.environ.get('TMPDIR', '/tmp') + '/mpl')
warnings.filterwarnings('ignore', category=FutureWarning)
import numpy as np, pandas as pd
sys.path.insert(0, os.path.abspath('..') if os.path.basename(os.getcwd()) == 'notebooks' else os.path.abspath('.'))
from gates.oracle import oracle_inputs
from rotten_tomatoes_forecasting import compute_edge

CACHE = 'gates/_cache' if os.path.isdir('gates/_cache') else '../gates/_cache'
mk = pd.read_csv(CACHE + '/cohort_markets.csv')
cd = pd.read_csv(CACHE + '/candles.csv')
de = pd.read_csv(CACHE + '/density.csv')
rv = pd.read_csv(CACHE + '/reviews_cohort.csv')
assert rv['as_of_id'].nunique() == 1 and int(rv['as_of_id'].iloc[0]) == 648979
assert de['as_of_id'].nunique() == 1 and int(de['as_of_id'].iloc[0]) == 648979
rv['estimated_timestamp'] = pd.to_datetime(rv['estimated_timestamp'], utc=True, format='ISO8601')
rv['scrape_time'] = pd.to_datetime(rv['scrape_time'], utc=True, format='ISO8601')
mk['close_dt'] = pd.to_datetime(mk['close_time'], utc=True)
mk['y'] = (mk['result'] == 'yes').astype(int)
cd = cd[cd['secs_to_close'] >= 0]

SNAPS = {'4d': 96, '3d': 72, '2d': 48, '1d': 24}        # 1d = secondary
clean = de.set_index(['slug', 'snap_days'])['oracle_clean'] \\
    if 'oracle_clean' in de.columns else None
if clean is None:                                        # density.csv stores raw rules
    de['oracle_clean'] = (de['live_tracked_through_snap'].astype(bool)
                          & (de['n_d_near_snap'] <= np.maximum(2, 0.10 * de['n_remaining']))
                          & (de['n_last_day_d'] <= 2))
    clean = de.set_index(['slug', 'snap_days'])['oracle_clean']

# state-at-snap book per (ticker, snap): last candle (ANY kind) at/before the snap
books = []
for tk, g in cd.groupby('ticker'):
    for lab, h in SNAPS.items():
        gg = g[g['secs_to_close'] >= h * 3600]
        if not len(gg):
            continue
        r = gg.loc[gg['secs_to_close'].idxmin()]
        books.append({'ticker': tk, 'snap': lab, 'snap_h': h,
                      'bid': r['yes_bid'], 'ask': r['yes_ask'], 'mid': r['mid'],
                      'stale_min': (r['secs_to_close'] - h * 3600) / 60.0})
bk = pd.DataFrame(books).merge(
    mk[['ticker', 'slug', 'floor_strike', 'close_dt', 'y']], on='ticker')
bk['spread'] = bk['ask'] - bk['bid']
bk['live'] = bk['mid'].notna()
bk['ct'] = bk['live'] & bk['mid'].gt(0.2) & bk['mid'].lt(0.8) & bk['spread'].le(0.10)
bk['fresh'] = bk['stale_min'] <= 60
bk['snap_days'] = (bk['snap_h'] // 24).astype(int)
bk['movie_clean'] = [bool(clean.get((s, d), False))
                     for s, d in zip(bk['slug'], bk['snap_days'])]
cells = bk[bk['ct'] & bk['movie_clean']].copy()
print('Gate-2 cells (oracle-clean ∧ contested ∧ spread<=10c), one row per market×snap:')
print(cells.groupby('snap').agg(markets=('ticker', 'size'), movies=('slug', 'nunique'),
                                fresh_books=('fresh', 'sum')).to_string())
dirty = bk[bk['ct'] & ~bk['movie_clean']]
print('\\nct cells EXCLUDED by the cohort guard (dirty movie at snap):',
      dirty.groupby('snap').size().to_dict())
"""

C_ORACLE = """# Oracle inputs per (movie, snap) x mode -> P(Yes) per market via compute_edge
preds = []
for (slug, lab), grp in cells.groupby(['slug', 'snap']):
    close = grp['close_dt'].iloc[0]
    h = SNAPS[lab]
    snap_ts = close - pd.Timedelta(hours=h)
    mrv = rv[rv['slug'] == slug]
    for mode in ('pure', 'lagged'):
        o = oracle_inputs(mrv, close, snap_ts, mode=mode)
        for _, r in grp.iterrows():
            X = int(r['floor_strike'])
            e = compute_edge(X, float(r['mid']) * 100, o.fresh_obs, o.total_obs,
                             o.t_rem_hours, o.lambda_rate, o.p_fresh)
            f = compute_edge(X, float(r['mid']) * 100, o.fresh_obs, o.total_obs,
                             o.t_rem_hours, 0.0, o.p_fresh)
            preds.append({'ticker': r['ticker'], 'slug': slug, 'snap': lab, 'mode': mode,
                          'mid': float(r['mid']), 'bid': float(r['bid']),
                          'ask': float(r['ask']), 'spread': float(r['spread']),
                          'stale_min': float(r['stale_min']), 'fresh_book': bool(r['fresh']),
                          'y': int(r['y']), 'X': X,
                          'p_oracle': float(e['p_yes']), 'p_frozen': float(f['p_yes']),
                          'prize': abs(float(e['p_yes']) - float(f['p_yes'])),
                          'lam': o.lambda_rate, 'p_fresh': o.p_fresh,
                          'n_rem': o.n_remaining, 'obs_total': o.total_obs,
                          'obs_score': (o.fresh_obs / o.total_obs) if o.total_obs else np.nan})
pr = pd.DataFrame(preds)
assert pr['p_oracle'].between(0, 1).all()
pr.to_csv(CACHE + '/gate2_cells.csv', index=False)
print('predictions:', len(pr), 'rows ->', 'gate2_cells.csv')
for (lab, mode), d in pr.groupby(['snap', 'mode']):
    print(f"  {lab} {mode:>6}: n={len(d):>2} ({d['slug'].nunique()} movies) "
          f"median lam/h={d['lam'].median():.2f} p_fresh={d['p_fresh'].median():.2f} "
          f"corr(p_oracle, mid)={d[['p_oracle', 'mid']].corr().iloc[0, 1]:.2f}")
"""

C_HEADLINE = """# HEADLINE 2x2: Brier + spread-crossing PnL vs the state-at-snap book
rng = np.random.default_rng(11)

def fee_cents(price_cents):
    return 7.0 * (price_cents / 100.0) * (1 - price_cents / 100.0)

def pnl_row(r, buffer_c=0.0):
    po, y = r['p_oracle'] * 100, r['y']
    ask_c, bid_c = r['ask'] * 100, r['bid'] * 100
    if po > ask_c + buffer_c:                      # buy Yes at ask
        return (100 * y - ask_c) - fee_cents(ask_c)
    if po < bid_c - buffer_c:                      # buy No at 100-bid
        no_c = 100 - bid_c
        return (100 * (1 - y) - no_c) - fee_cents(no_c)
    return np.nan                                  # no trade

def cboot(d, col_fn, n=2000):
    movies = d['slug'].unique()
    gr = {m: d[d['slug'] == m] for m in movies}
    out = []
    for _ in range(n):
        dd = pd.concat([gr[m] for m in rng.choice(movies, len(movies), replace=True)])
        out.append(col_fn(dd))
    return np.nanpercentile(out, [2.5, 97.5])

def brier_block(d, label):
    bs_m = float(np.mean((d['mid'] - d['y']) ** 2))
    bs_o = float(np.mean((d['p_oracle'] - d['y']) ** 2))
    ci = cboot(d, lambda dd: float(np.mean((dd['mid'] - dd['y']) ** 2)
                                   - np.mean((dd['p_oracle'] - dd['y']) ** 2)))
    print(f"  {label:<26} n={len(d):>2}/{d['slug'].nunique()}mv  "
          f"BS_mkt={bs_m:.4f} BS_orc={bs_o:.4f} diff={bs_m - bs_o:+.4f} "
          f"CI95=[{ci[0]:+.4f}, {ci[1]:+.4f}]{'  *clears 0*' if ci[0] > 0 else ''}")
    return bs_m - bs_o, ci

def pnl_block(d, label, buffer_c=0.0):
    p = d.apply(lambda r: pnl_row(r, buffer_c), axis=1)
    t = d[p.notna()].assign(pnl=p.dropna())
    if not len(t):
        print(f"  {label:<26} 0 trades")
        return
    ci = cboot(t, lambda dd: float(dd['pnl'].mean()))
    print(f"  {label:<26} trades={len(t):>2}/{len(d)} ({t['slug'].nunique()}mv) "
          f"mean={t['pnl'].mean():+.1f}c win%={100 * (t['pnl'] > 0).mean():.0f} "
          f"CI95=[{ci[0]:+.1f}, {ci[1]:+.1f}]c{'  *clears 0*' if ci[0] > 0 else ''}")

billie = [s for s in mk['slug'].unique() if 'billie' in s]
for mode in ('pure', 'lagged'):
    print(f"\\n=== {mode.upper()} oracle ===")
    for lab in ['4d', '3d', '2d', '1d']:
        d = pr[(pr['snap'] == lab) & (pr['mode'] == mode)]
        if not len(d):
            continue
        tag = ' (secondary)' if lab == '1d' else ''
        print(f"-- T-{lab}{tag} --")
        brier_block(d, 'Brier')
        if mode == 'pure':
            brier_block(d[~d['slug'].isin(billie)], 'Brier ex-billie')
        pnl_block(d, 'PnL taker buffer=0')
        pnl_block(d, 'PnL taker buffer=5c', 5.0)

# pooled one-obs-per-market (snap priority 3d > 2d > 4d; excludes secondary 1d)
prio = {'3d': 0, '2d': 1, '4d': 2}
pool = (pr[(pr['mode'] == 'pure') & (pr['snap'] != '1d')]
        .assign(prio=lambda d: d['snap'].map(prio))
        .sort_values('prio').drop_duplicates('ticker'))
print('\\n-- pooled unique markets (pure; snap priority 3d>2d>4d) --')
brier_block(pool, 'Brier'); pnl_block(pool, 'PnL taker buffer=0')

# trade-side decomposition (too-good-edge check: how much is the contested Yes-tilt?)
print('\\n-- trade-side decomposition (pooled, pure, buffer=0) --')
p = pool.apply(pnl_row, axis=1)
t = pool[p.notna()].assign(pnl=p.dropna())
t['side'] = np.where(t['p_oracle'] * 100 > t['ask'] * 100, 'YES', 'NO')
print(t.groupby('side').agg(n=('pnl', 'size'), movies=('slug', 'nunique'),
                            mean_pnl_c=('pnl', 'mean'),
                            win_pct=('pnl', lambda s: 100 * (s > 0).mean()),
                            mean_entry_mid=('mid', 'mean')).round(1).to_string())
print('\\n2x2 read: Brier-better x PnL>=0 per the plan; inconclusive (CI straddles 0) '
      'is EXPECTED ~half the time at this n even for a real 25% effect -> asymmetric fork.')
"""

C_STRAT = """# Stratification layer (operator-confirmed 2026-06-09): prize / encompassing / terciles
pp = pr[pr['mode'] == 'pure'].copy()

print('=== prize-sensitivity (oracle-conditioned gap = forecaster PRIZE, not inefficiency) ===')
for lab in ['4d', '3d', '2d']:
    d = pp[pp['snap'] == lab]
    if len(d) < 6:
        continue
    terc = pd.qcut(d['prize'], 3, labels=['low', 'mid', 'high'], duplicates='drop')
    t = d.groupby(terc, observed=True).agg(
        n=('y', 'size'),
        mkt_abs_err=('mid', lambda s: float(np.mean(np.abs(d.loc[s.index, 'y'] - s)))),
        orc_abs_err=('p_oracle', lambda s: float(np.mean(np.abs(d.loc[s.index, 'y'] - s)))),
        prize_med=('prize', 'median'))
    rho = float(np.corrcoef(d['prize'], np.abs(d['y'] - d['mid']))[0, 1])
    print(f"T-{lab}: corr(prize, |y-mid|) = {rho:+.2f}")
    print(t.round(3).to_string())

print('\\n=== encompassing LOMO logistic (ridge-regularized C=1): y ~ logit(mid) [+ logit(p_oracle)] ===')
print('NOTE: the UNregularized version is unstable at this n (near-separable 13-movie '
      'folds -> exploding coefficients, degenerate OOS logloss) — regularization is a '
      'small-sample necessity, disclosed; the fit-free Brier comparison above is the '
      'reliable headline either way.')
from sklearn.linear_model import LogisticRegression

def lomo_logloss(d, cols):
    eps = 1e-3
    X = np.column_stack([np.log(np.clip(d[c], eps, 1 - eps)
                                ) - np.log(1 - np.clip(d[c], eps, 1 - eps)) for c in cols])
    y, sl = d['y'].to_numpy(), d['slug'].to_numpy()
    ll = np.full(len(d), np.nan)
    for m in np.unique(sl):
        tr, te = sl != m, sl == m
        if y[tr].min() == y[tr].max():
            return np.nan                       # degenerate training fold
        f = LogisticRegression(C=1.0, max_iter=2000).fit(X[tr], y[tr])
        p = np.clip(f.predict_proba(X[te])[:, 1], 1e-6, 1 - 1e-6)
        ll[te] = -(y[te] * np.log(p) + (1 - y[te]) * np.log(1 - p))
    return float(np.nanmean(ll))

for lab in ['4d', '3d', '2d']:
    d = pp[pp['snap'] == lab]
    if d['slug'].nunique() < 6 or d['y'].nunique() < 2:
        continue
    a = lomo_logloss(d, ['mid'])
    b = lomo_logloss(d, ['mid', 'p_oracle'])
    print(f"T-{lab}: LOMO logloss mid-only={a:.4f}  +oracle={b:.4f}  "
          f"delta={a - b:+.4f} (positive = oracle adds OOS info beyond the price)")

print('\\n=== exploratory terciles (hypothesis generators ONLY; n tiny) ===')
for col in ['lam', 'p_fresh']:
    d = pp[pp['snap'].isin(['2d', '3d', '4d'])]
    terc = pd.qcut(d[col].rank(method='first'), 3, labels=['low', 'mid', 'high'])
    t = d.groupby([terc, 'snap'], observed=True).apply(
        lambda g: pd.Series({'n': len(g), 'mean_y_minus_mid': float((g['y'] - g['mid']).mean())}),
        include_groups=False)
    print(f"-- oracle {col} terciles x snap --")
    print(t.round(3).to_string())
"""

MD_TAIL = """## Reading guide / caveats

- 16-movie cohort, ~13 clean movies per snap → **directional**; the asymmetric fork is
  load-bearing (inconclusive ≠ fail; only clear-fail + clean form-diagnostic abandons —
  the form-diagnostic [PIT/rank-histogram] is only built if the headline reads fail).
- The **pure** oracle is the architecture-ceiling headline; **lagged** prices the
  current 50-min scraper cadence. Their gap = value of faster scraping.
- Oracle-conditioned reads are the forecaster's **PRIZE**; only observable-conditioned
  gaps are tradeable inefficiency (axis-language discipline, plan §stratification).
- PnL is taker-crossing at the resting book with its staleness — a stale book is
  hittable once; no refill/size assumption (depth unavailable historically).
- `_cache/gate2_cells.csv` holds every cell + oracle inputs/outputs for independent
  recompute.
"""

nb = nbf.v4.new_notebook()
nb.cells = [nbf.v4.new_markdown_cell(MD), nbf.v4.new_code_cell(C_CELLS),
            nbf.v4.new_code_cell(C_ORACLE), nbf.v4.new_code_cell(C_HEADLINE),
            nbf.v4.new_code_cell(C_STRAT), nbf.v4.new_markdown_cell(MD_TAIL)]
nb.metadata.kernelspec = {"name": "python3", "display_name": "Python 3", "language": "python"}
os.makedirs("notebooks", exist_ok=True)
nbf.write(nb, "notebooks/gate2_oracle.ipynb")
print("wrote notebooks/gate2_oracle.ipynb")
