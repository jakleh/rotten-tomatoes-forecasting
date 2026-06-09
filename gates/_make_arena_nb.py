"""Authoring helper: assemble notebooks/arena_map.ipynb from cell sources.

Codegen only (no analysis here) — the analysis lives in the notebook, executed via
nbconvert. Reads only cached CSVs (gates/_cache/), so the notebook runs sandboxed.
"""
import os
import nbformat as nbf

MD_INTENT = """# Tradeable-edge arena map (Gate-2 precondition)

**Question (operator-confirmed, 2026-06-07 handoff):** over time-to-close, which
(market × minute) cells are BOTH **contested** (`0.2 < mid < 0.8`) AND have a **live
two-sided quote**? Gate 1 found the market calibrated-but-stale → tradeability, not
forecasting, is the binding constraint. If the live∧contested arena is ~empty even
early, edge isn't capturable on this cohort (near-decisive abandon signal); if a
window exists, that's where Gate 2 runs.

**Candle semantics (settled empirically below).** Kalshi 1-min candles are
**activity-gated**: candle rows cover only a small fraction of each market's listing
window (listing windows run 2.6–40 *days*). The probe cell tests whether the book
persists unchanged through silent gaps (`open` of the candle after a gap == `close` of
the candle before it). If yes, **carrying the last candle's bid/ask forward (LOCF)
reconstructs the true book at every minute** — that's the primary read; candle-emitted
minutes ("strict") are reported as the no-assumption lower bound.

**Definitions.**
- *live (two-sided)*: book state has `0 < bid ≤ ask < 1` at that minute.
- *contested*: `0.2 < mid < 0.8` (handoff definition; band sensitivity shown).
- *spread bands*: a 0.07/0.93 book has mid 0.50 — "contested" but untradeable.
  Operative read = contested ∧ spread ≤ 10¢ (`ct`); ≤5¢ and ≤20¢ shown.
- *fresh*: book state last refreshed by a candle ≤ 60 min ago (`stale_min ≤ 60`) — an
  actively-maintained book, vs a resting one (still hittable once, but no refill
  signal).
- *listed(τ)*: markets whose open→close window covers τ (true denominator,
  independent of candle emission).

**Methodology fixes folded in from the Gate-1 review:** market-level claims read once
per market (no per-snap pseudo-replication); snap reads carry explicit **staleness**;
everything liquidity-stratified (spread bands). State-at-snap = the last candle of ANY
kind at/before the snap (a one-sided candle kills the carried quote — no reaching past
it to an older live quote, except in the explicitly-labeled "any staleness" legacy row).

**Data:** `gates/_cache/{cohort_markets,candles,gate1b_input}.csv` (built 2026-06-07,
settled 16-movie / 280-market KXRT cohort; candle history immutable post-settlement) +
`candle_open_probe.csv` (2026-06-09, 6 markets spanning the liquidity range, bid/ask
open+close). Review-formation overlay uses gate1b's publication-time observed state
(day-level reviews step at midnight UTC — approximate by construction).
n = 16 movies → **directional, not certifying**.
"""

C_LOAD = """import os
os.environ.setdefault('MPLCONFIGDIR', os.environ.get('TMPDIR', '/tmp') + '/mpl')
import numpy as np, pandas as pd
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt

CACHE = 'gates/_cache' if os.path.isdir('gates/_cache') else '../gates/_cache'
mk = pd.read_csv(CACHE + '/cohort_markets.csv')
cd = pd.read_csv(CACHE + '/candles.csv')
g1b = pd.read_csv(CACHE + '/gate1b_input.csv')
mk['open_dt'] = pd.to_datetime(mk['open_time'], utc=True)
mk['close_dt'] = pd.to_datetime(mk['close_time'], utc=True)
mk['window_min'] = (mk['close_dt'] - mk['open_dt']).dt.total_seconds() / 60
mk['window_h'] = mk['window_min'] / 60.0
mk['y'] = (mk['result'] == 'yes').astype(int)

assert len(mk) == 280 and mk['slug'].nunique() == 16, (len(mk), mk['slug'].nunique())
assert mk['slug'].notna().all(), 'unmapped slugs in cohort'
assert not cd.duplicated(['ticker', 'ts']).any(), 'duplicate (ticker, ts) candle rows'

no_candles = sorted(set(mk['ticker']) - set(cd['ticker']))
print('markets', len(mk), '| movies', mk['slug'].nunique(), '| candle rows', len(cd))
print('markets with zero candles:', len(no_candles), no_candles[:8])

post = int((cd['secs_to_close'] < 0).sum())
cd = cd[cd['secs_to_close'] >= 0].copy()
print(f'dropped {post} post-close rows; min/max secs_to_close now',
      int(cd['secs_to_close'].min()), '/', int(cd['secs_to_close'].max()))

# mid-rule recompute (guard against cache definition drift)
b, a = cd['yes_bid'], cd['yes_ask']
mid_re = np.where((b > 0) & (b <= a) & (a < 1), (b + a) / 2.0, np.nan)
bad = int((~np.isclose(mid_re, cd['mid'].to_numpy(), equal_nan=True)).sum())
assert bad == 0, f'{bad} cached mids disagree with the two-sided rule'

# emission density: candles are activity-gated, NOT one-per-minute
dens = cd.groupby('ticker').size().rename('n_candles').to_frame().join(
    mk.set_index('ticker')['window_min'])
dens['emission_density'] = dens['n_candles'] / (dens['window_min'] + 1)
print('\\nemission density (candle rows / listing-window minutes):')
print(dens['emission_density'].describe().round(4).to_string())

print('\\nlisting-window length (days), per market:')
print((mk['window_min'] / 1440).describe().round(2).to_string())
slugmap = dict(zip(mk['ticker'], mk['slug']))
cd['slug'] = cd['ticker'].map(slugmap)
print('\\nmovies:', sorted(mk['slug'].unique()))
"""

C_PROBE = """# Does the book persist through silent candle gaps? (LOCF validity test)
pr = pd.read_csv(CACHE + '/candle_open_probe.csv').sort_values(['ticker', 'ts'])
print('probe markets:', pr['ticker'].nunique(), '| rows', len(pr))

res = []
for tk, g in pr.groupby('ticker'):
    g = g.sort_values('ts')
    dt = g['ts'].diff()
    for cond, lab in [(dt == 60, 'adjacent'), (dt > 60, 'after-gap')]:
        for side in ('bid', 'ask'):
            prev_c = g[f'{side}_close'].shift()[cond]
            cur_o = g[f'{side}_open'][cond]
            same = ((prev_c.isna() & cur_o.isna())
                    | (prev_c.notna() & cur_o.notna() & np.isclose(prev_c, cur_o)))
            res.append({'ticker': tk, 'pair': lab, 'side': side, 'n': int(cond.sum()),
                        'state_identical': float(same.mean()) if cond.any() else np.nan})
res = pd.DataFrame(res)
print(res.pivot_table(index='ticker', columns=['pair', 'side'],
                      values=['n', 'state_identical']).round(4).to_string())
for lab in ['adjacent', 'after-gap']:
    d = res[res['pair'] == lab]
    pooled = float(np.average(d['state_identical'].fillna(0), weights=d['n']))
    print(f'pooled P(book state identical across boundary | {lab}): {pooled:.5f} '
          f'(n={int(d["n"].sum())})')
gap_d = res[res['pair'] == 'after-gap']
LOCF_OK = float(np.average(gap_d['state_identical'].fillna(0), weights=gap_d['n'])) > 0.99
print('\\nLOCF verdict:', 'VALID — book persists through silent gaps; LOCF is the '
      'primary read' if LOCF_OK else 'NOT validated — treat strict as primary!')

# probe close fields must agree with the main cache on overlapping (ticker, ts)
j = pr.merge(cd[['ticker', 'ts', 'yes_bid', 'yes_ask']], on=['ticker', 'ts'], how='inner')
mismatch = int((~np.isclose(j['bid_close'], j['yes_bid'], equal_nan=True)).sum()
               + (~np.isclose(j['ask_close'], j['yes_ask'], equal_nan=True)).sum())
print(f'probe-vs-cache overlap rows: {len(j)}, bid/ask close mismatches: {mismatch}')
assert mismatch == 0
"""

C_GRID = """# LOCF minute grid: expand each candle by its carry (until next candle / close)
cdm = cd.sort_values(['ticker', 'ts']).reset_index(drop=True)
close_ts = cdm['ts'] + cdm['secs_to_close']
nxt = cdm.groupby('ticker')['ts'].shift(-1)
end_ts = nxt.fillna(close_ts + 60).astype('int64')   # last candle carries through τ=0
carry = ((end_ts - cdm['ts']) // 60).astype('int64')
assert (carry >= 1).all()
cdm['carry_min'] = carry

n_total = int(carry.sum())
reps = np.repeat(cdm.index.to_numpy(), carry.to_numpy())
starts = np.concatenate([[0], np.cumsum(carry.to_numpy())[:-1]])
off = (np.arange(n_total) - np.repeat(starts, carry.to_numpy())).astype('int64')
codes, uniq = pd.factorize(cdm['ticker'])
slug_by_code = pd.Series([slugmap[t] for t in uniq])      # ticker code -> slug
scode_by_code, mslug = pd.factorize(slug_by_code)         # ticker code -> movie code; movie code -> slug

g = pd.DataFrame({
    'code': codes[reps].astype('int16'),
    'tau_min': ((close_ts.to_numpy()[reps] - (cdm['ts'].to_numpy()[reps] + off * 60))
                // 60).astype('int32'),
    'bid': cdm['yes_bid'].to_numpy()[reps],          # f64: keep cache precision —
    'ask': cdm['yes_ask'].to_numpy()[reps],          # f32 flips exact 0.2/0.8 boundaries
    'mid': cdm['mid'].to_numpy()[reps],              # the validated cached mid, carried
    'stale_min': off.astype('int32'),
})
g['mcode'] = pd.Series(scode_by_code[g['code'].to_numpy()], dtype='int8')
assert int((g['tau_min'] < 0).sum()) == 0
assert not (g['code'].astype('int64') * 100000 + g['tau_min']).duplicated().any()

g['spread'] = np.where(g['mid'].notna(), g['ask'] - g['bid'], np.nan)
g['live'] = g['mid'].notna()
g['contested'] = g['live'] & (g['mid'] > 0.2) & (g['mid'] < 0.8)
g['ct'] = g['contested'] & (g['spread'] <= 0.10)
g['fresh'] = g['stale_min'] <= 60
g['tau_h'] = g['tau_min'].astype('float32') / 60.0

# strict (stale_min==0) must reproduce the raw candle-based counts exactly
s0 = g[g['stale_min'] == 0]
raw_live = int(cd['mid'].notna().sum())
assert len(s0) == len(cd) and int(s0['live'].sum()) == raw_live, (len(s0), int(s0['live'].sum()), raw_live)

print(f'grid: {len(g):,} market-minutes (book-known, first candle -> close) '
      f'across {g["code"].nunique()} markets')
print('listed market-minutes (sum of full listing windows):',
      f"{int(mk['window_min'].sum() + len(mk)):,}",
      '| pre-first-candle (no book yet):',
      f"{int(mk['window_min'].sum() + len(mk)) - len(g):,}")

def totline(d, label):
    n = len(d)
    print(f'  {label:>34}: live {int(d["live"].sum()):>9,} | contested '
          f'{int(d["contested"].sum()):>9,} | ct(<=10c) {int(d["ct"].sum()):>9,} '
          f'| ct&fresh {int((d["ct"] & d["fresh"]).sum()):>8,}   (of {n:,} minutes)')
print('\\nminute-state totals (market-minutes):')
totline(g, 'LOCF (book-known minutes)')
totline(s0, 'strict (candle-emitted only)')

print('\\nspread distribution on LOCF contested minutes (cents):')
print((g.loc[g['contested'], 'spread'] * 100).astype(float).describe(
    percentiles=[0.1, 0.25, 0.5, 0.75, 0.9]).round(1).to_string())

# one-sided-but-hittable extras (no mid, one side resting at a contested price)
os_ = g[~g['live']]
print(f'\\none-sided extras (LOCF minutes): yes-buyable contested ask: '
      f'{int(((os_["ask"] > 0.2) & (os_["ask"] < 0.8)).sum()):,}; '
      f'no-buyable contested bid: {int(((os_["bid"] > 0.2) & (os_["bid"] < 0.8)).sum()):,}')

# Gate-2 cell inventory: compact span form (per candle row + carry duration)
spans = cdm.loc[(cdm['mid'] > 0.2) & (cdm['mid'] < 0.8),
                ['ticker', 'slug', 'ts', 'secs_to_close', 'mid', 'yes_bid', 'yes_ask',
                 'volume', 'carry_min']].copy()
spans['spread'] = spans['yes_ask'] - spans['yes_bid']
spans['ct'] = spans['spread'] <= 0.10
spans.to_csv(CACHE + '/arena_spans.csv', index=False)
old = CACHE + '/arena_cells.csv'
if os.path.exists(old):
    os.remove(old)
print(f"\\nsaved {len(spans):,} contested book-state spans -> _cache/arena_spans.csv "
      f"(each row carries carry_min minutes; tight subset: {int(spans['ct'].sum()):,} rows)")
"""

C_MAP = """# Occupancy vs time-to-close (LOCF primary, strict dashed; listed from true windows)
BIN_H = 0.5
g['bin'] = (g['tau_min'] // int(BIN_H * 60)).astype('int32')
edges = np.arange(0, int(336 / BIN_H) + 1)            # to 14 days for curves
listed_curve = (mk['window_h'].to_numpy()[:, None] > (edges[None, :] * BIN_H)).sum(axis=0)

def occ_curve(d, flag):
    dd = d[d[flag]]
    return (dd.groupby('bin')['code'].nunique(), dd.groupby('bin')['mcode'].nunique())

wmax_by_movie = mk.groupby('slug')['window_h'].max().to_numpy()
listed_movies_curve = (wmax_by_movie[:, None] > (edges[None, :] * BIN_H)).sum(axis=0)

fig, axes = plt.subplots(1, 2, figsize=(13.5, 4.6))
for ax, unit, idx in [(axes[0], 'markets', 0), (axes[1], 'movies', 1)]:
    ax.plot(edges * BIN_H, listed_curve if unit == 'markets'
            else listed_movies_curve, 'k:', lw=1.2, label='listed')
    for flag, lab, color in [('live', 'live two-sided', 'C0'),
                             ('contested', 'contested (any spread)', 'C1'),
                             ('ct', 'contested & <=10c', 'C3')]:
        c = occ_curve(g, flag)[idx]
        ax.plot(c.index * BIN_H, c.values, '-', color=color, lw=1.5, label=lab + ' (LOCF)')
        cs = occ_curve(g[g['stale_min'] == 0], flag)[idx]
        ax.plot(cs.index * BIN_H, cs.values, '--', color=color, lw=0.8,
                label=lab + ' (strict)' if flag == 'ct' else None)
    ax.set_xlabel('hours to close'); ax.set_ylabel(unit)
    ax.set_xlim(0, 240); ax.invert_xaxis(); ax.grid(alpha=0.3)
axes[0].legend(loc='upper left', fontsize=7)
axes[0].set_title('arena occupancy per 30-min bin (>=1 qualifying minute)')
plt.tight_layout(); plt.savefig(CACHE + '/arena_occupancy.png', dpi=110, bbox_inches='tight')
print('saved', CACHE + '/arena_occupancy.png')

EDGES_H = [0, 1, 3, 6, 12, 24, 48, 72, 96, 120, 168, 336, 1e9]
LABELS = ['0-1h', '1-3h', '3-6h', '6-12h', '12-24h', '24-48h', '48-72h',
          '72-96h', '96-120h', '120-168h', '7-14d', '>14d']
g['bucket'] = pd.cut(g['tau_h'], EDGES_H, labels=LABELS, right=True, include_lowest=True)

wh = mk['window_h'].to_numpy()
rows = []
for lab, lo, hi in zip(LABELS, EDGES_H[:-1], EDGES_H[1:]):
    listed_min = int((np.clip(np.minimum(wh, hi) - lo, 0, None) * 60).sum())
    listed_mkts = int((wh > lo).sum())
    d = g[g['bucket'] == lab]
    r = {'bucket': lab, 'listed_mkt_min': listed_min, 'listed_mkts': listed_mkts,
         'book_known_min': len(d)}
    for f in ['live', 'contested', 'ct']:
        dd = d[d[f]]
        r[f + '_mkt_min'] = len(dd)
        if f == 'ct':
            r['ctf_mkt_min'] = int((dd['fresh']).sum())
            r['ct_mkts'] = dd['code'].nunique()
            r['ct_movies'] = dd['mcode'].nunique()
    r['ct_occ_%'] = round(r['ct_mkt_min'] / listed_min * 100, 1) if listed_min else np.nan
    rows.append(r)
bt = pd.DataFrame(rows).set_index('bucket')
print('\\n=== arena bucket table (LOCF; listed = true windows; ctf = ct & fresh<=60min) ===')
print(bt.to_string())

print('\\nheadlines (LOCF):')
for win, lab in [(6, '<=6h'), (24, '<=24h'), (72, '<=72h'), (120, '<=120h')]:
    d = g[g['tau_h'] <= win]
    dd = d[d['ct']]
    print(f'  within {lab:>6}: contested {int(d["contested"].sum()):>9,} mkt-min | '
          f'ct(<=10c) {int(dd.shape[0]):>9,} mkt-min, {dd["code"].nunique():>3} mkts, '
          f'{dd["mcode"].nunique():>2} movies | ct&fresh {int(dd["fresh"].sum()):>7,} mkt-min')

print('\\ncontested-band sensitivity (LOCF):')
for lo, hi in [(0.1, 0.9), (0.2, 0.8), (0.35, 0.65)]:
    c = g['live'] & (g['mid'] > lo) & (g['mid'] < hi)
    ctx = c & (g['spread'] <= 0.10)
    print(f'  contested=({lo},{hi}): any-spread {int(c.sum()):>9,} mkt-min, '
          f'<=10c {int(ctx.sum()):>9,} mkt-min, <=10c movies {g[ctx]["mcode"].nunique()}')
"""

C_DEATH = """# Where the arena dies (LOCF) + review-formation overlay
tick_by_code = pd.Series(uniq)
per_mkt = mk.set_index('ticker')[['slug', 'floor_strike', 'window_h', 'y']].copy()
for f in ['live', 'contested', 'ct']:
    last = g[g[f]].groupby('code')['tau_h'].min().astype(float)
    last.index = tick_by_code[last.index].values
    per_mkt['last_' + f + '_h'] = last
    tot_min = g[g[f]].groupby('code').size()
    tot_min.index = tick_by_code[tot_min.index].values
    per_mkt[f + '_min_total'] = tot_min
per_mkt['ct_min_total'] = per_mkt['ct_min_total'].fillna(0).astype(int)

print('per-market last contested&<=10c minute (LOCF, hours to close; NaN = never):')
print(per_mkt['last_ct_h'].describe().round(2).to_string())
print('markets EVER contested&<=10c:', int(per_mkt['last_ct_h'].notna().sum()), '/', len(per_mkt))

mv = per_mkt.groupby('slug').agg(
    n_markets=('floor_strike', 'size'),
    n_ct_markets=('last_ct_h', lambda s: int(s.notna().sum())),
    ct_mkt_min=('ct_min_total', 'sum'),
    movie_last_ct_h=('last_ct_h', 'min'),
    movie_last_contested_h=('last_contested_h', 'min'),
    movie_last_live_h=('last_live_h', 'min'),
)

SNAP_H = {'5d': 120.0, '3d': 72.0, '2d': 48.0, '1d': 24.0,
          '12h': 12.0, '6h': 6.0, '3h': 3.0, '1h': 1.0}
chk = g1b.groupby(['slug', 'snap'])['obs_total'].nunique()
assert (chk == 1).all(), 'obs_total inconsistent within (slug, snap)'
obs = g1b.groupby(['slug', 'snap'])['obs_total'].first().unstack()
tot = mk.groupby('slug')['total_at_close'].first()

def frac_curve(slug):
    xs, ys = [0.0], [1.0]            # τ=0: all reviews in
    if slug in obs.index and tot.get(slug, 0) > 0:   # NaN-safe (NaN > 0 is False)
        for sn, h in SNAP_H.items():
            if sn in obs.columns:
                v = obs.at[slug, sn]
                if not np.isnan(v):
                    xs.append(h); ys.append(v / tot[slug])
    o = np.argsort(xs)
    return np.array(xs)[o], np.array(ys)[o]

mv['pct_reviews_still_to_come_at_last_ct'] = np.nan
for slug in mv.index:
    t = mv.at[slug, 'movie_last_ct_h']
    if np.isnan(t):
        continue
    xs, ys = frac_curve(slug)
    if len(xs) < 2:
        continue
    mv.at[slug, 'pct_reviews_still_to_come_at_last_ct'] = float((1 - np.interp(t, xs, ys)) * 100)

print('\\n=== per movie: arena death vs review formation (LOCF) ===')
print(mv.sort_values('movie_last_ct_h').round(2).to_string())
print('\\nmedian movie-level last contested&<=10c (h to close):',
      round(float(mv['movie_last_ct_h'].median()), 1))
print('median % of final reviews still to come at that minute:',
      round(float(mv['pct_reviews_still_to_come_at_last_ct'].median()), 1))

ok = mv.dropna(subset=['movie_last_ct_h', 'pct_reviews_still_to_come_at_last_ct'])
plt.figure(figsize=(7, 5))
plt.scatter(ok['movie_last_ct_h'], ok['pct_reviews_still_to_come_at_last_ct'], s=40)
for s, r in ok.iterrows():
    plt.annotate(s[:14], (r['movie_last_ct_h'], r['pct_reviews_still_to_come_at_last_ct']),
                 fontsize=7, xytext=(4, 3), textcoords='offset points')
plt.xlabel('movie last contested&<=10c minute (hours to close)')
plt.ylabel('% of final reviews still to come')
plt.gca().invert_xaxis(); plt.grid(alpha=0.3)
plt.title('does the live-contested window overlap score formation?')
plt.tight_layout(); plt.savefig(CACHE + '/arena_death_vs_formation.png', dpi=110, bbox_inches='tight')
print('saved', CACHE + '/arena_death_vs_formation.png')

# state AT snap (exact minute on the LOCF grid) + formation + embargo check
print('\\n=== state AT snap (LOCF exact minute) ===')
for sn, h in SNAP_H.items():
    w = g[g['tau_min'] == int(h * 60)]
    n_listed = int((mk['window_h'] > h).sum())
    m_ct = sorted(mslug[w.loc[w['ct'], 'mcode'].unique()]) if w['ct'].any() else []
    n_ctf = w.loc[w['ct'] & w['fresh'], 'mcode'].nunique()
    if sn in obs.columns:
        stc = [(1 - obs.at[s, sn] / tot[s]) * 100 for s in m_ct
               if s in obs.index and not np.isnan(obs.at[s, sn]) and tot[s] > 0]
        emb = int((obs[sn] == 0).sum())
    else:
        stc, emb = [], -1
    med = round(float(np.median(stc)), 1) if stc else float('nan')
    print(f'  {sn:>3}: listed mkts {n_listed:>3} | ct mkts {int(w["ct"].sum()):>3} '
          f'({w.loc[w["ct"], "mcode"].nunique():>2} movies, {n_ctf:>2} fresh) '
          f'| median % reviews still to come (ct movies): {str(med):>5} '
          f'| movies with 0 obs reviews: {emb}')
"""

C_1D = """# Snap reads with staleness (one obs per market): state-at-snap + legacy any-staleness
ymap = dict(zip(mk['ticker'], mk['y']))

def state_at(tau_h_target):
    w = g[g['tau_min'] == int(tau_h_target * 60)].copy()
    w['ticker'] = tick_by_code[w['code'].to_numpy()].values
    w['slug'] = slug_by_code[w['code'].to_numpy()].values
    w['y'] = w['ticker'].map(ymap)
    return w[['ticker', 'slug', 'mid', 'spread', 'stale_min', 'live', 'y']]

def last_live_quote(tau_h_target):   # legacy Gate-1-style: last LIVE candle <= snap
    rows = []
    for tk, gg in cd[cd['mid'].notna()].groupby('ticker'):
        ggg = gg[gg['secs_to_close'] >= tau_h_target * 3600]
        if not len(ggg):
            continue
        r = ggg.loc[ggg['secs_to_close'].idxmin()]
        rows.append({'ticker': tk, 'slug': slugmap[tk], 'mid': float(r['mid']),
                     'spread': float(r['yes_ask'] - r['yes_bid']),
                     'staleness_h': float(r['secs_to_close'] / 3600 - tau_h_target),
                     'y': int(ymap[tk])})
    return pd.DataFrame(rows)

rng = np.random.default_rng(0)
def cboot(d, n=2000):
    movies = d['slug'].unique()
    gr = {m: d[d['slug'] == m] for m in movies}
    out = []
    for _ in range(n):
        samp = rng.choice(movies, len(movies), replace=True)
        dd = pd.concat([gr[m] for m in samp])
        out.append(float(dd['y'].mean() - dd['mid'].mean()))
    return np.percentile(out, [2.5, 97.5])

def report(name, dd):
    if not len(dd):
        print(f'  {name:<40}: n=0')
        return
    ci = cboot(dd) if dd['slug'].nunique() > 1 else (np.nan, np.nan)
    print(f'  {name:<40}: n={len(dd):>3} ({dd["slug"].nunique()} movies) '
          f'mean_mid={dd["mid"].mean():.3f} realized={dd["y"].mean():.3f} '
          f'diff={dd["y"].mean() - dd["mid"].mean():+.3f} '
          f'CI95(diff, cluster-boot)=[{ci[0]:+.3f}, {ci[1]:+.3f}]')

billie = [s for s in mk['slug'].unique() if 'billie' in s]
print('billie slug match:', billie)

for h, lab in [(48, '2d'), (24, '1d'), (12, '12h')]:
    st = state_at(h)
    live = st[st['live']]
    c = live[(live['mid'] > 0.2) & (live['mid'] < 0.8)]
    print(f'\\n=== snap {lab}: {len(st)} markets with a book state at snap '
          f'({int(st["live"].sum())} live two-sided) ===')
    print('  book-state staleness at snap (min):',
          st['stale_min'].describe(percentiles=[0.5, 0.9]).round(0).loc[
          ['mean', '50%', '90%', 'max']].astype(int).to_dict())
    report('contested (state at snap)', c)
    report('contested fresh (stale<=60min)', c[c['stale_min'] <= 60])
    report('contested fresh+tight (<=10c)',
           c[(c['stale_min'] <= 60) & (c['spread'] <= 0.10)])
    report('contested fresh ex-billie',
           c[(c['stale_min'] <= 60) & (~c['slug'].isin(billie))])
    lq = last_live_quote(h)
    cl = lq[(lq['mid'] > 0.2) & (lq['mid'] < 0.8)]
    report('legacy: last live quote, any staleness', cl)
print('\\nNOTE: n is tiny and movie-clustered -> directional only; '
      'diff = realized - mid (positive = market under-priced Yes).')
"""

MD_TAIL = """## Reading guide / caveats

- **LOCF validity** is established in the probe cell (book state identical across
  silent gaps); if that had failed, the strict (candle-emitted) numbers would be the
  primary read — both are printed throughout.
- The bucket table + occupancy plot are **inventory** (market-minutes; each market
  counts once per minute); calibration-style reads (snap cells) are one-obs-per-market
  with explicit staleness. Cluster unit for uncertainty is the **movie** (n=16).
- A resting (stale) book is still hittable once — but only the `fresh` (≤60 min) tier
  signals an actively-maintained market with refill capacity.
- The review-formation overlay uses gate1b's publication-time observed state at the
  snap grid (linear interpolation between snaps; day-level reviews step at midnight
  UTC) — approximate, for overlap orientation only.
- `_cache/arena_spans.csv` is the Gate-2 cell inventory in compact span form: one row
  per contested book state (candle) with `carry_min` = how many minutes that state
  persisted; expand spans to minutes to enumerate cells.
"""

nb = nbf.v4.new_notebook()
nb.cells = [nbf.v4.new_markdown_cell(MD_INTENT), nbf.v4.new_code_cell(C_LOAD),
            nbf.v4.new_code_cell(C_PROBE), nbf.v4.new_code_cell(C_GRID),
            nbf.v4.new_code_cell(C_MAP), nbf.v4.new_code_cell(C_DEATH),
            nbf.v4.new_code_cell(C_1D), nbf.v4.new_markdown_cell(MD_TAIL)]
nb.metadata.kernelspec = {"name": "python3", "display_name": "Python 3", "language": "python"}
os.makedirs("notebooks", exist_ok=True)
nbf.write(nb, "notebooks/arena_map.ipynb")
print("wrote notebooks/arena_map.ipynb")
