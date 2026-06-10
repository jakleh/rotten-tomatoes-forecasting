"""Authoring helper: assemble notebooks/pfresh_battery.ipynb from cell sources.

Codegen only — analysis lives in the notebook (cache-only, no DB). Spec:
``plans/plan_p_fresh_regression.md`` v2 "Phase 0 — the battery" + "Pinned
conventions". The intent cell restates conventions + readings BEFORE any result;
the notebook ends by WRITING the machine-readable battery decisions the bench
notebook builds against (``gates/_cache/pfresh_battery_decisions.csv``).
"""
import os
import nbformat as nbf

MD = """# p_fresh Phase 0 — the falsification battery (pre-registered)

Decides WHAT GETS BUILT (plan v2; brainstorm v3 §0). All on the pinned training pull;
the bench world (648979) is touched ONLY for (a) the §1.1 probe re-derivation and
(b) the C3 coverage gate's anchorable-share (observed-review critics — no outcomes).

**Pinned conventions (restated from the plan BEFORE results):** row weight
`1/(n_snaps_m · n_rem)` (movie totals 1, equal per snap); battery score = weighted
mean per-review binomial deviance (p clipped 1e-3) pooled + per-snap; Spearman
unweighted on the common set; GLM = expanded-counts L2 logistic, C ∈ {0.01…1000},
GroupKFold(5) by movie; added-value increments on locked feature-complete subsets
with the base arm re-scored there; curves/anchors fit on ALL training reviews
(≤~1%/movie contamination of grouped-CV increments — disclosed; bench scoring uses
strictly temporal refits).

**Readings (pre-registered):**
- T1 (per snap, SHRUNK prior_actual decides): |bias(actual)| < ⅓·|bias(P2)| at BOTH
  T-3d and T-4d → composition-dominant; within ⅔ → behavior-dominant; else mixed.
- T2: standalone deviance ladder P0→P3 on the common set; P4/P5 OOS deviance
  improvement ≤ 0 → intensity channel dead; P4-vs-P5 picks the C3 encoding;
  P3 beats P2 → C2 uses the shrunk prior.
- T3: median within-critic OOS boundary-accuracy lift ≤ +2pp → per-critic anchors
  dead (C3 falls back to global curves).
- T4: scored-mass share at curve p ∈ [0.2, 0.8] < 15% → intensity ceiling low (C3
  demoted regardless of T2).
- T5: OOS deviance improvement > 0 AND positive visible-score coefficient → the
  obs_rate×snap state term earns its C2 slot (+ behavior interpretation for T1).
- T6/T7: informational (subtraction value; thin-critic ceiling).
- C3 build gate: (T2 intensity increment > 0) AND T4 passes AND bench anchorable
  coverage ≥ 50%; encoding anchored iff T3 passes AND P5 ≥ P4, else global.
- d-row placement sensitivity: per-snap |Δ mean y| > 0.02 under boundary placements →
  pause for operator ping before the bench pass.
"""

C_LOAD = """import os, sys, warnings
os.environ.setdefault('MPLCONFIGDIR', os.environ.get('TMPDIR', '/tmp') + '/mpl')
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)
import numpy as np, pandas as pd
sys.path.insert(0, os.path.abspath('..') if os.path.basename(os.getcwd()) == 'notebooks' else os.path.abspath('.'))
from gates import pfresh_lib as pl
from rotten_tomatoes_forecasting.pool import _most_recent_resolved_slugs

CACHE = 'gates/_cache' if os.path.isdir('gates/_cache') else '../gates/_cache'
STORE = 'gates/recorded' if os.path.isdir('gates/recorded') else '../gates/recorded'
rows = pd.read_csv(CACHE + '/pfresh_training_rows.csv')
rv = pd.read_csv(CACHE + '/pfresh_training_reviews.csv')
meta = pd.read_csv(CACHE + '/pfresh_meta.csv').iloc[0]
PIN = int(meta['as_of_id'])
assert (rv['as_of_id'] == PIN).all() and len(rows) == int(meta['n_rows'])
rv['estimated_timestamp'] = pd.to_datetime(rv['estimated_timestamp'], utc=True, format='ISO8601')
rv['est_raw'] = pd.to_datetime(rv['est_raw'], utc=True, format='ISO8601')
rows['close_dt'] = pd.to_datetime(rows['close'], utc=True, format='ISO8601')
rows['snap_dt'] = pd.to_datetime(rows['snap_ts'], utc=True, format='ISO8601')
print(f"training: {len(rows)} rows / {rows['slug'].nunique()} movies @ pin {PIN}")
assert (rows['n_obs'] > 0).all() and (rows['n_rem'] > 0).all()
print('common row set = ALL rows (n_obs>0 by construction) ->', len(rows))

mi = pd.read_csv('movies_index.csv' if os.path.exists('movies_index.csv') else '../movies_index.csv')
mk = pd.read_csv(STORE + '/markets.csv')
cdm = {**dict(zip(mi['Slug'], pd.to_datetime(mi['Bet Close Date'], utc=True))),
       **{s: pd.to_datetime(t, utc=True) for s, t in mk.groupby('slug')['close_time'].first().items()}}

rv = pl.add_parsed_scores(rv)
curves = pl.global_curves(rv)
anchored_rows = rv[rv['anchored_scored']]
per_critic_n = anchored_rows.groupby('reviewer_name').size()
anchors = {c: a for c, g in anchored_rows.groupby('reviewer_name')
           if (a := pl.fit_anchor(g)) is not None}
print(f"parsed: {rv['score_family'].notna().mean():.1%} of rows scored | "
      f"anchored rows {len(anchored_rows)} | critics with anchors {len(anchors)} | "
      f"curve points {len(curves)}")
"""

C_PREDICTORS = """# Per-row predictors P0..P5 + T1/T6 variants (pool objects per movie, snap-indep)
mrv_by = {s: g for s, g in rv.groupby('movie_slug')}
recs = []
for slug, g in rows.groupby('slug'):
    close = g['close_dt'].iloc[0]
    pool = _most_recent_resolved_slugs(cdm, before=close, n=20, exclude_slug=slug)
    pool_rows = rv[rv['movie_slug'].isin(pool)]
    fresh_pool = pool_rows['tomatometer_sentiment'].eq('positive')
    p0 = float(fresh_pool.mean())
    rates_raw, g_rate = pl.critic_rates(pool_rows)
    rates_shr, _ = pl.critic_rates(pool_rows, shrink_k=pl.SHRINK_K)
    base = (pool_rows.groupby('reviewer_name')['movie_slug'].nunique() / 20)
    mr = mrv_by[slug]
    for _, r in g.iterrows():
        snap = r['snap_dt']
        obs = mr[mr['estimated_timestamp'] < snap]
        rem = mr[(mr['estimated_timestamp'] >= snap) & (mr['estimated_timestamp'] <= close)]
        assert len(obs) == r['n_obs'] and len(rem) == r['n_rem']
        oc = set(obs['reviewer_name'])
        p2 = pl.prior_remaining(rv, pool, oc)
        p3 = pl.prior_remaining(rv, pool, oc, shrink_k=pl.SHRINK_K)
        p6_full = pl.prior_remaining(rv, pool, set())          # T6: no subtraction
        pa_shr = pl.prior_actual(rem, rates_shr, g_rate)
        pa_raw = pl.prior_actual(rem, rates_raw, g_rate, raw_default=0.5)
        exc_g, n_sc = pl.intensity_excess(obs, curves)
        exc_a, _ = pl.intensity_excess(obs, curves, anchors)
        rem_hist = [len(rv[(rv['reviewer_name'] == c)
                           & (rv['movie_slug'] != slug)]) for c in rem['reviewer_name']]
        rem_anch = [c in anchors and anchors[c].n >= 10 for c in rem['reviewer_name']]
        recs.append({**{k: r[k] for k in ('slug', 'snap_days', 'n_obs', 'fresh_obs',
                                          'n_rem', 'fresh_rem', 'close')},
                     'P0': p0, 'P1': r['fresh_obs'] / r['n_obs'],
                     'P2': p2, 'P3': p3, 'P6_full': p6_full,
                     'PA_shr': pa_shr, 'PA_raw': pa_raw,
                     'exc_global': exc_g, 'exc_anchored': exc_a, 'n_scored_obs': n_sc,
                     'mass_consumed': float(base[base.index.isin(oc)].sum() / base.sum())
                                      if base.sum() > 0 else 0.0,
                     'rem_zero_hist': float(np.mean([h == 0 for h in rem_hist])),
                     'rem_thin_scored': float(np.mean([not a for a in rem_anch]))})
ft = pd.DataFrame(recs)
ft['y'] = ft['fresh_rem'] / ft['n_rem']
ft.to_csv(CACHE + '/pfresh_training_features.csv', index=False)  # the bench fits on
print('predictor frame:', ft.shape, '-> pfresh_training_features.csv')
"""

C_T1 = """# T1 — composition vs behavior (SHRUNK prior_actual decides; raw = context)
print('=== T1: mean(p - y) per snap (weighted by row weight x n_rem = movie-equal) ===')
w = pl.row_weights(ft) * ft['n_rem']
def wmean(col, mask):
    return float(np.average(ft.loc[mask, col] - ft.loc[mask, 'y'],
                            weights=w[mask]))
t1 = pd.DataFrame({
    f'T-{n}d': {'bias_P2_shipped': wmean('P2', ft['snap_days'] == n),
                'bias_actual_shrunk': wmean('PA_shr', ft['snap_days'] == n),
                'bias_actual_raw': wmean('PA_raw', ft['snap_days'] == n)}
    for n in [1, 2, 3, 4]}).T
print(t1.round(4).to_string())
verdicts = {}
for lab in ['T-3d', 'T-4d']:
    b2, ba = abs(t1.loc[lab, 'bias_P2_shipped']), abs(t1.loc[lab, 'bias_actual_shrunk'])
    verdicts[lab] = ('composition' if ba < b2 / 3 else
                     'behavior' if ba > 2 * b2 / 3 else 'mixed')
t1_verdict = (verdicts['T-3d'] if verdicts['T-3d'] == verdicts['T-4d']
              else f"split({verdicts['T-3d']}/{verdicts['T-4d']})")
print(f"T1 verdict: {verdicts} -> {t1_verdict}")
"""

C_T2 = """# T2 — channel ladder (standalone deviances on the common set; locked-subset
# added-value fits for the intensity channel)
print('=== T2: standalone weighted deviance (lower better; pooled + per-snap) ===')
for p in ['P0', 'P1', 'P2', 'P3', 'P6_full', 'PA_shr']:
    dv = pl.deviance_of_predictor(ft.assign(**{p: ft[p].clip(0.001, 0.999)}), p)
    rho = ft[[p, 'y']].corr(method='spearman').iloc[0, 1]
    per_snap = ' '.join(
        f"T{n}d={pl.deviance_of_predictor(ft[ft['snap_days'] == n], p):.3f}"
        for n in [1, 2, 3, 4])
    print(f"  {p:<8} deviance {dv:.4f}  spearman {rho:+.3f}  | {per_snap}")
use_P3 = (pl.deviance_of_predictor(ft, 'P3') < pl.deviance_of_predictor(ft, 'P2'))
print(f"-> C2 prior choice: {'P3 (shrunk)' if use_P3 else 'P2 (shipped)'}")

print('\\n=== T2 added-value: intensity over emp-logit(P1), locked subset ===')
sub = ft[ft['n_scored_obs'] >= 1].copy()
sub['el_P1'] = [pl.emp_logit(f, n) for f, n in zip(sub['fresh_obs'], sub['n_obs'])]
print(f"locked subset: {len(sub)}/{len(ft)} rows ({sub['slug'].nunique()} movies)")
base_m, base_c, base_dev = pl.fit_binomial_glm(sub, ['el_P1'])
g_m, g_c, g_dev = pl.fit_binomial_glm(sub, ['el_P1', 'exc_global'])
a_m, a_c, a_dev = pl.fit_binomial_glm(sub, ['el_P1', 'exc_anchored'])
print(f"  base OOS deviance {base_dev:.4f} (C={base_c})")
print(f"  +exc_global   {g_dev:.4f} (C={g_c})  improvement {base_dev - g_dev:+.4f}")
print(f"  +exc_anchored {a_dev:.4f} (C={a_c})  improvement {base_dev - a_dev:+.4f}")
p4_gain, p5_gain = base_dev - g_dev, base_dev - a_dev
intensity_alive = max(p4_gain, p5_gain) > 0
"""

C_T3T4 = """# T3 — anchor transferability | T4 — curve steepness
lifts, skipped = [], 0
for critic, g in anchored_rows.groupby('reviewer_name'):
    if len(g) < pl.ANCHOR_MIN_SCORED:
        continue
    g = g.sort_values('estimated_timestamp')
    h1, h2 = g.iloc[:len(g) // 2], g.iloc[len(g) // 2:]
    y1 = h1['tomatometer_sentiment'].eq('positive').to_numpy(int)
    y2 = h2['tomatometer_sentiment'].eq('positive').to_numpy(int)
    if y1.min() == y1.max():
        skipped += 1
        continue
    from sklearn.linear_model import LogisticRegression
    m = LogisticRegression(C=1.0, max_iter=1000).fit(h1[['score_level']], y1)
    acc = float((m.predict(h2[['score_level']]) == y2).mean())
    maj = float((y2 == int(y1.mean() >= 0.5)).mean())
    lifts.append(acc - maj)
t3_lift = float(np.median(lifts)) if lifts else float('nan')
anchors_alive = bool(lifts) and t3_lift > 0.02
print(f"T3: {len(lifts)} critics tested ({skipped} one-class skipped) | "
      f"median OOS accuracy lift {t3_lift:+.3f} -> per-critic anchors "
      f"{'ALIVE' if anchors_alive else 'DEAD (fall back to global curves)'}")

lv = anchored_rows.assign(p_curve=[curves.get((f, float(l)), np.nan)
                                   for f, l in zip(anchored_rows['score_family'],
                                                   anchored_rows['score_level'])])
mass_informative = float(lv['p_curve'].between(0.2, 0.8).mean())
t4_pass = mass_informative >= 0.15
print(f"\\nT4: scored-mass share at curve p in [0.2, 0.8]: {mass_informative:.1%} "
      f"-> intensity ceiling {'OK' if t4_pass else 'LOW (C3 demoted)'}")
for fam, g in lv.groupby('score_family'):
    t = g.groupby('score_level')['tomatometer_sentiment'].agg(
        n='size', p=lambda s: s.eq('positive').mean())
    print(f"-- {fam} --\\n{t.round(3).to_string()}")
"""

C_T5 = """# T5 — herding (state terms beyond critic identity; eligible-movie reviews)
from scipy import sparse
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import OneHotEncoder

el_movies = set(rows['slug'])
sub5 = rv[rv['movie_slug'].isin(el_movies)].copy()
vs = pd.concat([pl.visible_state(g) for _, g in sub5.groupby('movie_slug')])
sub5 = sub5.join(vs)
counts = sub5.groupby('reviewer_name').size()
sub5 = sub5[sub5['reviewer_name'].isin(counts[counts >= 10].index)]
sub5 = sub5[sub5['visible_total'] > 0]
sub5['el_vis'] = [pl.emp_logit(f, t) for f, t in zip(sub5['visible_fresh'],
                                                     sub5['visible_total'])]
sub5['log_vt'] = np.log1p(sub5['visible_total'])
y5 = sub5['tomatometer_sentiment'].eq('positive').to_numpy(int)
g5 = sub5['movie_slug'].to_numpy()
enc = OneHotEncoder(handle_unknown='ignore')
FE = enc.fit_transform(sub5[['reviewer_name']])
Xb = FE
Xs = sparse.hstack([FE, sparse.csr_matrix(sub5[['el_vis', 'log_vt']].to_numpy())]).tocsr()
gkf = GroupKFold(5)
def t5_oos(X, C):
    devs = []
    for tr, te in gkf.split(X, y5, groups=g5):
        m = LogisticRegression(C=C, max_iter=2000).fit(X[tr], y5[tr])
        p = np.clip(m.predict_proba(X[te])[:, 1], 1e-3, 1 - 1e-3)
        devs.append(float(np.mean(-2 * (y5[te] * np.log(p)
                                        + (1 - y5[te]) * np.log(1 - p)))))
    return float(np.mean(devs))
base_c5, base_d5 = min(((c, t5_oos(Xb, c)) for c in pl.C_GRID), key=lambda t: t[1])
state_d5 = t5_oos(Xs, base_c5)   # C selected on the base model (conservative:
#                                   handicaps the state arm, which must win anyway)
m_full = LogisticRegression(C=base_c5, max_iter=2000).fit(Xs, y5)
coef_score = float(m_full.coef_[0][-2])
t5_fires = (base_d5 - state_d5 > 0) and (coef_score > 0)
print(f"T5: n={len(sub5)} reviews / {sub5['movie_slug'].nunique()} movies / "
      f"{sub5['reviewer_name'].nunique()} critic FE | base OOS dev {base_d5:.4f} "
      f"(C={base_c5}) -> +state {state_d5:.4f} (improvement {base_d5 - state_d5:+.4f}) "
      f"| visible-score coef {coef_score:+.3f}")
print(f"-> state term {'EARNS its C2 slot' if t5_fires else 'does NOT fire'} "
      f"(unseen-critic FE=0 at predict via handle_unknown=ignore)")
"""

C_T6T7 = """# T6 — subtraction value | T7 — thin-critic ceiling (informational)
d_p2 = pl.deviance_of_predictor(ft, 'P2')
d_full = pl.deviance_of_predictor(ft, 'P6_full')
print(f"T6: shipped prior {d_p2:.4f} vs no-subtraction full-pool {d_full:.4f} "
      f"(subtraction improvement {d_full - d_p2:+.4f})")
print('\\nT7: share of realized REMAINING reviews by critic-history class (per snap):')
t7 = ft.groupby('snap_days')[['rem_zero_hist', 'rem_thin_scored']].mean()
print(t7.rename(columns={'rem_zero_hist': 'zero_history',
                         'rem_thin_scored': 'not_anchorable(>=10)'}).round(3).to_string())
"""

C_PROBE_DROW = """# §1.1 probe re-derivation (bench caches; must reproduce) + d-row sensitivity
c3b = pd.read_csv(CACHE + '/gate3b_cells.csv')
wb = c3b['obs_total_est'] / (c3b['obs_total_est'] + 20.0)
obs_rate_b = c3b['obs_fresh_est'] / c3b['obs_total_est']
prior_b = (c3b['p_fresh_hat'] - wb * obs_rate_b) / (1 - wb)
lo, hi = np.minimum(obs_rate_b, prior_b), np.maximum(obs_rate_b, prior_b)
outside = int(((c3b['p_fresh_oracle'] < lo) | (c3b['p_fresh_oracle'] > hi)).sum())
t3m = c3b['snap_days'] == 3
m_obs = float((obs_rate_b[t3m] - c3b.loc[t3m, 'p_fresh_oracle']).mean())
m_pri = float((prior_b[t3m] - c3b.loc[t3m, 'p_fresh_oracle']).mean())
print(f"probe re-derivation: hull outside {outside}/60 | T-3d obs-bias {m_obs:+.3f} "
      f"prior-bias {m_pri:+.3f}")
assert outside == 35 and round(m_obs, 3) == 0.078 and round(m_pri, 3) == 0.062

print('\\nd-row placement sensitivity on training y (earliest=raw est_ts; latest=+24h):')
deltas = {}
for n in [1, 2, 3, 4]:
    sub_n = rows[rows['snap_days'] == n]
    dy = []
    for _, r in sub_n.iterrows():
        mr = mrv_by[r['slug']]
        close, snap = r['close_dt'], r['snap_dt']
        for shift in [pd.Timedelta(0), pd.Timedelta(hours=24)]:
            eff = mr['estimated_timestamp'].where(
                mr['timestamp_confidence'] != 'd', mr['est_raw'] + shift)
            rem = mr[(eff >= snap) & (eff <= close)]
            y_v = (rem['tomatometer_sentiment'].eq('positive').mean()
                   if len(rem) else np.nan)
            dy.append(y_v)
    a = np.array(dy).reshape(-1, 2)
    base_y = (rows.loc[rows['snap_days'] == n, 'fresh_rem']
              / rows.loc[rows['snap_days'] == n, 'n_rem']).to_numpy()
    d_lo = np.nanmean(a[:, 0]) - np.nanmean(base_y)
    d_hi = np.nanmean(a[:, 1]) - np.nanmean(base_y)
    deltas[n] = max(abs(d_lo), abs(d_hi))
    print(f"  T-{n}d: mean-y shift earliest {d_lo:+.4f} / latest {d_hi:+.4f}")
drow_material = max(deltas.values()) > 0.02
print(f"-> d-row placement {'MATERIAL — operator ping before bench' if drow_material else 'immaterial'}")
"""

C_SENS = """# Training-side sensitivities [F12] (never bench rows; 12h-grid struck per the
# post-build review — deferred to the recorder-growth re-run, plan review log)
print('shrink-k sensitivity on the P3 prior (deviance; pinned k=10):')
for k in [5, 10, 20]:
    col = f'P3_k{k}'
    if k == 10:
        ft[col] = ft['P3']
    else:
        vals = []
        for slug, g in rows.groupby('slug'):
            close = g['close_dt'].iloc[0]
            pool = _most_recent_resolved_slugs(cdm, before=close, n=20, exclude_slug=slug)
            mr = mrv_by[slug]
            for _, r in g.iterrows():
                oc = set(mr.loc[mr['estimated_timestamp'] < r['snap_dt'],
                                'reviewer_name'])
                vals.append(pl.prior_remaining(rv, pool, oc, shrink_k=float(k)))
        ft[col] = vals
    print(f"  k={k:>2}: {pl.deviance_of_predictor(ft, col):.4f}")

print('\\nraw-clip vs empirical-logit encoding (added-value base fit, OOS deviance):')
sub_s = ft[ft['n_scored_obs'] >= 0].copy()
sub_s['el_P1'] = [pl.emp_logit(f, n) for f, n in zip(sub_s['fresh_obs'], sub_s['n_obs'])]
sub_s['lc_P1'] = [pl.logit_clip(p, 0.02, 0.98) for p in sub_s['P1']]
_, _, d_el = pl.fit_binomial_glm(sub_s, ['el_P1'])
_, _, d_lc = pl.fit_binomial_glm(sub_s, ['lc_P1'])
print(f"  emp-logit {d_el:.4f} vs raw-clip {d_lc:.4f} (delta {d_lc - d_el:+.4f})")
print('12h-grid row sensitivity: STRUCK (post-build review; deferred to the '
      'recorder-growth re-run — plan review log).')
"""

C_DECIDE = """# Battery decisions -> the bench notebook builds EXACTLY this (machine-readable)
cells3b = pd.read_csv(CACHE + '/gate3b_a1_cache.csv')
cells3b['estimated_timestamp'] = pd.to_datetime(cells3b['estimated_timestamp'], utc=True, format='ISO8601')
# anchorable per plan F11: >=2 scored fresh AND >=2 scored rotten in dominant family
two_sided = set()
for c, g in anchored_rows.groupby('reviewer_name'):
    fr = g['tomatometer_sentiment'].eq('positive').sum()
    if fr >= 2 and (len(g) - fr) >= 2:
        two_sided.add(c)
cells_grid = pd.read_csv(CACHE + '/gate3b_cells.csv')
cov_num = cov_den = 0
for (slug, snap_ts), _ in cells_grid.groupby(['slug', 'snap_ts']):
    mr = cells3b[cells3b['movie_slug'] == slug]
    obs = mr[mr['estimated_timestamp'] < pd.to_datetime(snap_ts)]
    cov_den += len(obs)
    cov_num += int(obs['reviewer_name'].isin(two_sided).sum())
coverage = cov_num / cov_den if cov_den else 0.0
print(f"C3 coverage gate: {coverage:.1%} of bench observed reviews carry a two-sided "
      f"anchorable critic (gate >= 50%)")

build_C3 = bool(intensity_alive and t4_pass and coverage >= 0.5)
c3_mode = ('anchored' if (anchors_alive and p5_gain >= p4_gain) else 'global') if build_C3 else ''
dec = {'use_P3': bool(use_P3), 'state_term': bool(t5_fires),
       'build_C3': build_C3, 'c3_mode': c3_mode,
       'p4_gain': round(p4_gain, 4), 'p5_gain': round(p5_gain, 4),
       't3_lift': round(t3_lift, 4), 't4_mass': round(mass_informative, 4),
       'coverage': round(coverage, 4), 't1_verdict': t1_verdict,
       'drow_material': bool(drow_material), 'as_of_id': PIN}
pd.DataFrame([dec]).to_csv(CACHE + '/pfresh_battery_decisions.csv', index=False)
print('\\nBATTERY DECISIONS ->', dec)
"""

MD_TAIL = """## Reading guide

- The decisions CSV is the build contract: the bench notebook asserts it builds
  exactly this configuration. Channels killed here are killed with their measured
  increment on record — that measurement is deliverable #1.
- T1's verdict steers INTERPRETATION + next-cycle direction (a composition-dominant
  read has no deployable hook this cycle — `prior_actual` is oracle-composition).
- Effective n ≈ movie count (~135); per-movie weights + grouped CV throughout;
  expectations modest by design.
"""

nb = nbf.v4.new_notebook()
nb.cells = [nbf.v4.new_markdown_cell(MD), nbf.v4.new_code_cell(C_LOAD),
            nbf.v4.new_code_cell(C_PREDICTORS), nbf.v4.new_code_cell(C_T1),
            nbf.v4.new_code_cell(C_T2), nbf.v4.new_code_cell(C_T3T4),
            nbf.v4.new_code_cell(C_T5), nbf.v4.new_code_cell(C_T6T7),
            nbf.v4.new_code_cell(C_PROBE_DROW), nbf.v4.new_code_cell(C_SENS),
            nbf.v4.new_code_cell(C_DECIDE), nbf.v4.new_markdown_cell(MD_TAIL)]
nb.metadata.kernelspec = {"name": "python3", "display_name": "Python 3",
                          "language": "python"}
os.makedirs("notebooks", exist_ok=True)
nbf.write(nb, "notebooks/pfresh_battery.ipynb")
print("wrote notebooks/pfresh_battery.ipynb")
