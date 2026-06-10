"""Authoring helper: assemble notebooks/gate2_density.ipynb from cell sources.

Codegen only — the analysis lives in the notebook (nbconvert-executed, reads only
gates/_cache/). Applies the PRE-REGISTERED oracle-clean cell definition + dense-n floor
from plans/plan_gate_1_2_calibration.md ("Pre-registered dense-cohort floor",
2026-06-09, locked before the density driver's first run).
"""
import os
import nbformat as nbf

MD = """# Gate-2 dense-cohort guard (D1 STOP-gate)

Applies the **pre-registered** oracle-clean cell definition + floor
(`plans/plan_gate_1_2_calibration.md` § "Pre-registered dense-cohort floor", locked
2026-06-09 before `gates/build_density.py` first ran) to the cached density facts.

- **(a) live-tracked-through-snap:** first `scrape_time` ≤ snap.
- **(b) snap-boundary clean:** `n_d_near_snap ≤ max(2, 0.10 × n_remaining)`.
- **(c) close-boundary clean (M2):** movie `n_last_day_d ≤ 2`.
- **Floor:** a snap is runnable iff ≥ 8 oracle-clean movies; else "underpowered /
  inconclusive" (never abandon).

Also closes the remaining Phase-0 checks (`'s'`-confidence rows; scraper-timing) and
runs the pre-registered **power simulation** — market-side data only (gate1b mids +
labels), ASSUMED oracle effect = 25% contested-region Brier improvement.

All DB facts are `as_of_id`-pinned (printed below) via `gates/db_facts.py`.
"""

C_LOAD = """import os
os.environ.setdefault('MPLCONFIGDIR', os.environ.get('TMPDIR', '/tmp') + '/mpl')
import numpy as np, pandas as pd

CACHE = 'gates/_cache' if os.path.isdir('gates/_cache') else '../gates/_cache'
de = pd.read_csv(CACHE + '/density.csv')
meta = pd.read_csv(CACHE + '/density_meta.csv')
g1b = pd.read_csv(CACHE + '/gate1b_input.csv')
mk = pd.read_csv(CACHE + '/cohort_markets.csv')

assert de['as_of_id'].nunique() == 1
AS_OF = int(de['as_of_id'].iloc[0])
print('as_of_id =', AS_OF, '(all claims below pinned to id <=', AS_OF, ')')
print('rows:', len(de), '| movies:', de['slug'].nunique(), '| snaps:', sorted(de['snap_days'].unique()))

# Phase-0: 's'-confidence rows
conf_cols = [c for c in meta.columns if c.startswith('conf_')]
print('\\ncohort timestamp_confidence counts:', {c[5:]: int(meta[c].iloc[0]) for c in conf_cols})
assert 'conf_s' not in meta.columns, "'s'-confidence rows exist — revisit sub-minute handling"
print("Phase-0 check: zero 's'-confidence rows -> +1min lag design stands")

# Phase-0: scraper-timing — does coverage reach each movie's 10am close?
mv = de[de['snap_days'] == 1][['slug', 'close_time', 'movie_total', 'movie_max_est_ts',
                               'n_last_day', 'n_last_day_d', 'n_after_close',
                               'first_scrape']].set_index('slug')
mv['close_dt'] = pd.to_datetime(mv['close_time'], utc=True)
mv['max_est'] = pd.to_datetime(mv['movie_max_est_ts'], utc=True, format='ISO8601')
mv['covers_close'] = (mv['max_est'] >= mv['close_dt']) | (mv['n_after_close'] > 0)
print('\\nscraper-timing: movies whose review coverage reaches close:',
      int(mv['covers_close'].sum()), '/', len(mv))
if not mv['covers_close'].all():
    short = mv[~mv['covers_close']]
    print(short[['close_time', 'movie_max_est_ts', 'n_after_close']].to_string())
    print('NOTE: max_est < close with n_after_close==0 -> verify the movie simply had no '
          'reviews between max_est and close (benign) vs scraper stopped early (gap).')
"""

C_GATE = """# Pre-registered rules (verbatim from the plan; locked before data)
de['rule_a_live_tracked'] = de['live_tracked_through_snap'].astype(bool)
# consistency re-derive of (a) from raw timestamps — guards the driver's logic
fs = pd.to_datetime(de['first_scrape'], utc=True, format='ISO8601')
snap_dt = pd.to_datetime(de['snap_ts'], utc=True, format='ISO8601')
assert ((fs <= snap_dt) == de['rule_a_live_tracked']).all(), 'driver/notebook (a) mismatch'

de['rule_b_boundary'] = de['n_d_near_snap'] <= np.maximum(2, 0.10 * de['n_remaining'])
de['rule_c_close_m2'] = de['n_last_day_d'] <= 2
de['oracle_clean'] = de['rule_a_live_tracked'] & de['rule_b_boundary'] & de['rule_c_close_m2']

print('=== per-snap oracle-clean counts (floor: >=8 movies -> runnable) ===')
rows = []
for sd, d in de.groupby('snap_days'):
    rows.append({'snap_days': sd, 'movies': len(d),
                 'a_live_tracked': int(d['rule_a_live_tracked'].sum()),
                 'b_boundary': int(d['rule_b_boundary'].sum()),
                 'c_close_m2': int(d['rule_c_close_m2'].sum()),
                 'ORACLE_CLEAN': int(d['oracle_clean'].sum()),
                 'runnable(>=8)': bool(d['oracle_clean'].sum() >= 8)})
print(pd.DataFrame(rows).set_index('snap_days').to_string())

print('\\n=== exclusions, by movie x snap (why a cell is dirty) ===')
bad = de[~de['oracle_clean']]
for _, r in bad.iterrows():
    why = []
    if not r['rule_a_live_tracked']:
        why.append(f"a: first_scrape {str(r['first_scrape'])[:10]} > snap")
    if not r['rule_b_boundary']:
        why.append(f"b: n_d_near_snap={int(r['n_d_near_snap'])} > "
                   f"max(2, 0.1x{int(r['n_remaining'])})")
    if not r['rule_c_close_m2']:
        why.append(f"c: n_last_day_d={int(r['n_last_day_d'])}")
    print(f"  {r['slug'][:40]:<42} T-{int(r['snap_days'])}d  " + '; '.join(why))

print('\\n=== oracle-input scale on clean cells (context for oracle precision) ===')
cl = de[de['oracle_clean']]
print(cl.groupby('snap_days')[['n_remaining', 'n_remaining_mh', 'n_remaining_d']]
        .median().round(1).to_string())
print('\\nscrape-lag on clean cells, m/h remaining reviews (minutes; scrape-lagged-oracle context):')
lag = cl.assign(lag_p50_min=cl['lag_p50_s'] / 60, lag_p90_min=cl['lag_p90_s'] / 60)
print(lag.groupby('snap_days')[['lag_p50_min', 'lag_p90_min']].median().round(1).to_string())
"""

C_POWER = """# Pre-registered power simulation — market-side data ONLY (no oracle values exist yet).
# Assumed NET effect: oracle Brier = 75% of market Brier on the contested set.
#
# Mechanism (and a disclosed implementation refinement): the simplest model
# p_o = mid + kappa*(y-mid) is DEGENERATE for power — it improves every row
# deterministically ((mid-y)^2 - ((1-kappa)(mid-y))^2 > 0 whenever mid != y), so any
# bootstrap CI excludes 0 and power = 1.0 mechanically. A real oracle improves on
# average but with idiosyncratic error. So: p_o = clip(mid + kappa'*(y-mid) + eps),
# eps ~ N(0, sigma_o), with the noise share omega = sigma_o^2 / BS_market and
# (1-kappa')^2 = 0.75 - omega, so the NET improvement stays ~25% (pre-clipping).
# omega = 0 reproduces the degenerate case (shown for transparency). This refinement
# is forced by visible math, decided BEFORE any oracle values were computed.
rng = np.random.default_rng(7)

snap2 = g1b[(g1b['snap'] == '2d') & (g1b['mid'] > 0.2) & (g1b['mid'] < 0.8)].copy()
print(f'noise-scale set: gate1b 2d contested rows n={len(snap2)} '
      f'({snap2["slug"].nunique()} movies)')
movies = snap2['slug'].unique()
gr = {m: snap2[snap2['slug'] == m][['mid', 'y']].to_numpy() for m in movies}
bs_mkt = float(np.mean((snap2['mid'] - snap2['y']) ** 2))
print(f'market Brier on this set: {bs_mkt:.4f}')

def power_at(n_movies, omega, n_draws=200, n_boot=400):
    kappa = 1 - np.sqrt(0.75 - omega)
    sigma = np.sqrt(omega * bs_mkt)
    hits, realized_impr = 0, []
    for _ in range(n_draws):
        cohort = [gr[m] for m in rng.choice(movies, n_movies, replace=True)]
        rows = np.vstack(cohort)
        mid, y = rows[:, 0], rows[:, 1]
        po = np.clip(mid + kappa * (y - mid) + rng.normal(0, sigma, len(mid)), 0.01, 0.99)
        d = (mid - y) ** 2 - (po - y) ** 2          # per-row Brier improvement
        realized_impr.append(np.mean(d) / np.mean((mid - y) ** 2))
        idx = rng.integers(0, len(d), (n_boot, len(d)))
        lo = np.percentile(d[idx].mean(axis=1), 2.5)
        hits += lo > 0
    return hits / n_draws, float(np.mean(realized_impr))

print('\\npower to detect (CI95 lower > 0) an assumed ~25% net contested Brier improvement:')
print('  omega = oracle idiosyncratic-noise share of market Brier')
for omega in [0.0, 0.10, 0.20]:
    line = f'  omega={omega:.2f}: '
    for n in [8, 11, 13, 14, 16]:
        p, ri = power_at(n, omega)
        line += f'n={n}: {p:.2f}  '
    print(line + f'(realized net improvement ~{ri:.0%})')
print('\\nNOTE: omega=0 is the degenerate deterministic-improvement case (power trivially '
      '~1). Locked decision rule: if power < ~0.5 even at n=16 under moderate noise '
      '(omega=0.10), Gate 2 is explicitly directional regardless of the clean-n floor. '
      'Bootstrap here is by-row within draw (movies enter via cohort redraw).')
"""

MD_TAIL = """## Reading guide

- The verdict table is the STOP-gate: snaps with ≥8 oracle-clean movies are runnable
  for Gate 2; the exclusions list says exactly which (movie, snap) cells are out and
  why (a = backfilled-through-snap, b = snap-boundary d-ambiguity, c = close-day
  d-ambiguity / M2).
- `n_remaining` medians on clean cells size the oracle's per-cell sample (λ is a
  realized count over the window; p_fresh a realized fraction of `n_remaining`).
- Scrape-lag quantiles are context for the **scrape-lagged oracle (#2)** only; the
  headline pure-publication-time oracle does not gate on them.
- The power simulation uses ONLY market-side data (gate1b mids + labels) + an assumed
  effect, per the pre-registration — no oracle values were computed before this gate.
"""

nb = nbf.v4.new_notebook()
nb.cells = [nbf.v4.new_markdown_cell(MD), nbf.v4.new_code_cell(C_LOAD),
            nbf.v4.new_code_cell(C_GATE), nbf.v4.new_code_cell(C_POWER),
            nbf.v4.new_markdown_cell(MD_TAIL)]
nb.metadata.kernelspec = {"name": "python3", "display_name": "Python 3", "language": "python"}
os.makedirs("notebooks", exist_ok=True)
nbf.write(nb, "notebooks/gate2_density.ipynb")
print("wrote notebooks/gate2_density.ipynb")
