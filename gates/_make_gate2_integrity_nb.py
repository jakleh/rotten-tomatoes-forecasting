"""Authoring helper: assemble notebooks/gate2_integrity_recheck.ipynb.

Codegen only. Origin (2026-06-10): the recorder's settlement-consistency check exposed
(a) a sentiment-CASE switch in rows scraped >= ~2026-06-02 (raw table preserved per
operator; processing layer made case-insensitive) and (b) a coverage-thin movie
(animal_farm_2025, self-label 20 vs settlement-implied [23,25]) that sat INSIDE the
Gate-2 clean set. Operator directive: re-run gates + fix docs wherever the integrity
issue may have corrupted findings — no numerical-claim drift.

This notebook measures, against the PUBLISHED gate2_cells.csv (pinned as_of_id=648979):
  1. the uppercase-sentiment census of the pinned review cache (the contamination channel);
  2. per-cell oracle-input drift under the case-insensitive recompute (tolerance 1e-9);
  3. an anchor reproduction of the published headline POINT estimates (draw-independent);
  4. the ex-animal_farm robustness re-read (new citable disclosure numbers).
Decision rule (pre-registered in the intent cell): zero drift -> published Gate-2/3a
numbers stand verbatim, docs gain the disclosure + ex-ANI row; any nonzero drift ->
re-execute gate2_oracle.ipynb + gate3_tolerance.ipynb and correct every downstream doc.
"""
import os

import nbformat as nbf

MD = """# Gate-2 integrity recheck (2026-06-10) — sentiment-case switch + animal_farm coverage

**Origin.** The §1.7 recorder's settlement-consistency check (self-label must land inside
the score interval implied by the event's own strike results) exposed two data-integrity
issues: rows scraped ≥ ~2026-06-02 carry UPPERCASE `tomatometer_sentiment` (every
case-sensitive `== 'positive'` silently undercounts fresh), and `animal_farm_2025` is
coverage-thin (self-label 20 vs settlement-implied [23,25] on 45 captured pre-close rows)
while having been INSIDE the Gate-2 oracle-clean set. The processing layer is now
case-insensitive (`gates/oracle.py`, `gates/db_facts.py`, `p_fresh.py`).

**Questions (pre-registered):**
1. How many rows of the pinned Gate-2 review cache (`reviews_cohort.csv`, as_of_id=648979)
   are uppercase, per movie, and how many sit at est_ts ≤ close (the only rows that could
   have fed the oracle)?
2. Recomputing every published cell's oracle inputs with the case-insensitive oracle:
   does ANY of (`lam`, `p_fresh`, `n_rem`, `obs_total`, `p_oracle`, `p_frozen`) drift
   beyond 1e-9 vs the published `gate2_cells.csv`?
3. Anchor: do the published headline POINT estimates (pooled Brier diff +0.0966, pooled
   PnL +27.5¢; T-3d +0.0775 / +24.1¢) reproduce exactly from the recomputed cells?
   (Bootstrap CIs are rng-sequence-dependent and not independently reproducible — point
   estimates are the anchor.)
4. Ex-`animal_farm_2025` robustness: the pooled + T-3d/T-4d Brier and PnL re-reads with
   the coverage-thin movie removed (its oracle consumed whatever the DB had; the DB
   demonstrably misses reviews RT counted).

**Decision rule:** zero drift in (2) ⇒ published Gate-2 (and the Gate-3a sweep anchored
on these cells) stands verbatim; docs gain the animal_farm disclosure + the ex-ANI
robustness row. Any nonzero drift ⇒ re-execute `gate2_oracle.ipynb` +
`gate3_tolerance.ipynb` and correct every downstream number.
"""

C_LOAD = """import os, sys, warnings
os.environ.setdefault('MPLCONFIGDIR', os.environ.get('TMPDIR', '/tmp') + '/mpl')
warnings.filterwarnings('ignore', category=FutureWarning)
import numpy as np, pandas as pd
sys.path.insert(0, os.path.abspath('..') if os.path.basename(os.getcwd()) == 'notebooks' else os.path.abspath('.'))
from gates.oracle import oracle_inputs
from rotten_tomatoes_forecasting import compute_edge

CACHE = 'gates/_cache' if os.path.isdir('gates/_cache') else '../gates/_cache'
rv = pd.read_csv(CACHE + '/reviews_cohort.csv')
pub = pd.read_csv(CACHE + '/gate2_cells.csv')          # the PUBLISHED cells (read-only here)
mk = pd.read_csv(CACHE + '/cohort_markets.csv')
assert rv['as_of_id'].nunique() == 1 and int(rv['as_of_id'].iloc[0]) == 648979
rv['estimated_timestamp'] = pd.to_datetime(rv['estimated_timestamp'], utc=True, format='ISO8601')
rv['scrape_time'] = pd.to_datetime(rv['scrape_time'], utc=True, format='ISO8601')
mk['close_dt'] = pd.to_datetime(mk['close_time'], utc=True)
close_by_slug = mk.drop_duplicates('slug').set_index('slug')['close_dt']

# Q1 — uppercase census of the pinned cache
sv = rv['tomatometer_sentiment'].value_counts(dropna=False)
print('sentiment values in reviews_cohort.csv (as_of 648979):')
print(sv.to_string())
up = rv[rv['tomatometer_sentiment'].str.fullmatch(r'[A-Z]+', na=False)].copy()
up['close'] = up['slug'].map(close_by_slug)
up['pre_close'] = up['estimated_timestamp'] <= up['close']
print('\\nuppercase rows per movie (pre-close = could have fed the oracle):')
if len(up):
    print(up.groupby('slug').agg(n=('slug', 'size'), pre_close=('pre_close', 'sum'),
                                 first_scrape=('scrape_time', 'min')).to_string())
else:
    print('  none')
cell_movies = set(pub['slug'].unique())
up_in_cells = up[up['slug'].isin(cell_movies) & up['pre_close']]
print(f"\\nuppercase PRE-CLOSE rows belonging to Gate-2 CELL movies: {len(up_in_cells)}"
      f" (movies: {sorted(up_in_cells['slug'].unique()) if len(up_in_cells) else '—'})")
"""

C_DRIFT = """# Q2 — per-cell recompute with the case-insensitive oracle vs the published cells
SNAPS = {'4d': 96, '3d': 72, '2d': 48, '1d': 24}
TOL = 1e-9
CHECK = ['lam', 'p_fresh', 'n_rem', 'obs_total', 'p_oracle', 'p_frozen']
rows = []
for (slug, lab, mode), grp in pub.groupby(['slug', 'snap', 'mode']):
    close = close_by_slug[slug]
    snap_ts = close - pd.Timedelta(hours=SNAPS[lab])
    o = oracle_inputs(rv[rv['slug'] == slug], close, snap_ts, mode=mode)
    for _, r in grp.iterrows():
        X = int(r['X'])
        e = compute_edge(X, float(r['mid']) * 100, o.fresh_obs, o.total_obs,
                         o.t_rem_hours, o.lambda_rate, o.p_fresh)
        f = compute_edge(X, float(r['mid']) * 100, o.fresh_obs, o.total_obs,
                         o.t_rem_hours, 0.0, o.p_fresh)
        new = {'lam': o.lambda_rate, 'p_fresh': o.p_fresh, 'n_rem': o.n_remaining,
               'obs_total': o.total_obs, 'p_oracle': float(e['p_yes']),
               'p_frozen': float(f['p_yes'])}
        d = {c: abs(new[c] - float(r[c])) for c in CHECK}
        rows.append({'ticker': r['ticker'], 'slug': slug, 'snap': lab, 'mode': mode,
                     **{f'new_{c}': new[c] for c in CHECK},
                     **{f'drift_{c}': d[c] for c in CHECK},
                     'any_drift': max(d.values()) > TOL})
dr = pd.DataFrame(rows)
assert len(dr) == len(pub), 'recheck must cover every published cell row'
n_drift = int(dr['any_drift'].sum())
print(f'published cell rows: {len(pub)} | rows with ANY input/output drift > {TOL}: {n_drift}')
print('max |drift| per column:')
print(dr[[f'drift_{c}' for c in CHECK]].max().to_string())
if n_drift:
    out = CACHE + '/gate2_integrity_drift.csv'
    dr[dr['any_drift']].to_csv(out, index=False)
    print(f'NONZERO DRIFT -> wrote {out}; decision rule: re-execute gate2/gate3a + correct docs')
else:
    print('ZERO drift: the published Gate-2 cells (and the Gate-3a sweep anchored on them) '
          'stand verbatim under the case-insensitive recompute.')
"""

C_ANCHOR = """# Q3 anchor (point estimates, draw-independent) + Q4 ex-animal_farm robustness
rng = np.random.default_rng(11)

def fee_cents(price_cents):
    return 7.0 * (price_cents / 100.0) * (1 - price_cents / 100.0)

def pnl_row(r, buffer_c=0.0):
    po, y = r['p_oracle'] * 100, r['y']
    ask_c, bid_c = r['ask'] * 100, r['bid'] * 100
    if po > ask_c + buffer_c:
        return (100 * y - ask_c) - fee_cents(ask_c)
    if po < bid_c - buffer_c:
        no_c = 100 - bid_c
        return (100 * (1 - y) - no_c) - fee_cents(no_c)
    return np.nan

def cboot(d, col_fn, n=2000):
    movies = d['slug'].unique()
    gr = {m: d[d['slug'] == m] for m in movies}
    out = []
    for _ in range(n):
        dd = pd.concat([gr[m] for m in rng.choice(movies, len(movies), replace=True)])
        out.append(col_fn(dd))
    return np.nanpercentile(out, [2.5, 97.5])

def blocks(d, tag):
    bs = float(np.mean((d['mid'] - d['y']) ** 2) - np.mean((d['p_oracle'] - d['y']) ** 2))
    bci = cboot(d, lambda dd: float(np.mean((dd['mid'] - dd['y']) ** 2)
                                    - np.mean((dd['p_oracle'] - dd['y']) ** 2)))
    p = d.apply(pnl_row, axis=1)
    t = d[p.notna()].assign(pnl=p.dropna())
    pci = cboot(t, lambda dd: float(dd['pnl'].mean()))
    print(f"  {tag:<28} n={len(d)}/{d['slug'].nunique()}mv  Brier diff={bs:+.4f} "
          f"CI[{bci[0]:+.4f},{bci[1]:+.4f}]  PnL trades={len(t)} mean={t['pnl'].mean():+.1f}c "
          f"CI[{pci[0]:+.1f},{pci[1]:+.1f}] win%={100 * (t['pnl'] > 0).mean():.0f}")
    return bs, float(t['pnl'].mean())

prio = {'3d': 0, '2d': 1, '4d': 2}
pure = pub[(pub['mode'] == 'pure') & (pub['snap'] != '1d')]
pool = (pure.assign(prio=lambda d: d['snap'].map(prio))
        .sort_values('prio').drop_duplicates('ticker'))

print('=== anchor: published headline POINT estimates from the published cells ===')
bs_pool, pnl_pool = blocks(pool, 'pooled (all movies)')
d3 = pub[(pub['mode'] == 'pure') & (pub['snap'] == '3d')]
bs_3d, pnl_3d = blocks(d3, 'T-3d (all movies)')
assert abs(bs_pool - 0.0966) < 5e-4 and abs(pnl_pool - 27.5) < 0.05, 'pooled anchor drifted'
assert abs(bs_3d - 0.0775) < 5e-4 and abs(pnl_3d - 24.1) < 0.05, 'T-3d anchor drifted'
print('  -> anchors reproduce the published +0.0966/+27.5c pooled and +0.0775/+24.1c T-3d')

print('\\n=== Q4: ex-animal_farm_2025 robustness (the coverage-thin movie removed) ===')
ANI = 'animal_farm_2025'
print(f"ANI cells: pooled {int((pool['slug'] == ANI).sum())} of {len(pool)} markets; "
      f"T-3d {int((d3['slug'] == ANI).sum())} of {len(d3)}; "
      f"T-4d {int(((pub['mode'] == 'pure') & (pub['snap'] == '4d') & (pub['slug'] == ANI)).sum())}"
      f" of {int(((pub['mode'] == 'pure') & (pub['snap'] == '4d')).sum())}")
blocks(pool[pool['slug'] != ANI], 'pooled ex-ANI')
blocks(d3[d3['slug'] != ANI], 'T-3d ex-ANI')
d4 = pub[(pub['mode'] == 'pure') & (pub['snap'] == '4d')]
blocks(d4[d4['slug'] != ANI], 'T-4d ex-ANI')
d2 = pub[(pub['mode'] == 'pure') & (pub['snap'] == '2d')]
blocks(d2[d2['slug'] != ANI], 'T-2d ex-ANI')
"""

MD_TAIL = """## Reading guide

- The drift table (Q2) is the verdict: zero rows over 1e-9 means the sentiment-case
  switch never touched the published Gate-2 inputs — the only uppercase rows at
  est ≤ close belong to movies the cohort guard had already excluded from cells.
- The ex-ANI rows (Q4) are NEW citable numbers (this notebook is their source); CIs here
  use this notebook's own rng stream — point estimates are the comparable quantity vs
  the published tables.
- animal_farm's defect is COVERAGE (DB missed reviews RT counted; settlement-implied
  score [23,25] vs self-label 20 on 45 pre-close rows), not case: its rows are all
  lowercase (first-scraped 2026-04-24). Its oracle was internally consistent
  (observed + remaining == terminal on what the DB had) but the DB itself was short.
- Gate 1 / arena: not re-run — reasoning recorded in the plan addendum (the uppercase
  channel post-dates every Gate-1 observation window for the affected movies, and the
  ANI coverage defect only perturbs observed-state features for 1/16 movies in analyses
  whose conclusion was a null / a map, not a marginal effect)."""

nb = nbf.v4.new_notebook()
nb.cells = [nbf.v4.new_markdown_cell(MD), nbf.v4.new_code_cell(C_LOAD),
            nbf.v4.new_code_cell(C_DRIFT), nbf.v4.new_code_cell(C_ANCHOR),
            nbf.v4.new_markdown_cell(MD_TAIL)]
nb.metadata.kernelspec = {"name": "python3", "display_name": "Python 3", "language": "python"}
os.makedirs("notebooks", exist_ok=True)
nbf.write(nb, "notebooks/gate2_integrity_recheck.ipynb")
print("wrote notebooks/gate2_integrity_recheck.ipynb")
