"""Authoring helper: assemble notebooks/gate1_calibration.ipynb from cell sources.

Codegen only (no analysis here) — the analysis lives in the notebook, executed via
nbconvert. Reads only the cached CSVs (gates/_cache/), so the notebook runs sandboxed.
"""
import os
import nbformat as nbf

MD = """# Gate 1 — market-mid calibration (directional)

Is the Kalshi market calibrated on settlement, measured on the **order-book mid** as a
function of **time-to-close**, over the cached settled cohort? Uses only minutes with a
**real two-sided quote** (calibration *conditional on tradeable*). Cluster-bootstrap by
**movie** (the correlated unit). n is small (16 movies) → **directional, not certifying**.

Out of scope here (Gate 1b, needs observed-at-snap review state): the incremental-info-
over-price test. This notebook is Gate 1a: is the price itself calibrated?
"""

C_LOAD = """import os
os.environ.setdefault('MPLCONFIGDIR', os.environ.get('TMPDIR', '/tmp') + '/mpl')
import numpy as np, pandas as pd
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt

CACHE = 'gates/_cache' if os.path.isdir('gates/_cache') else '../gates/_cache'
mk = pd.read_csv(CACHE + '/cohort_markets.csv')
cd = pd.read_csv(CACHE + '/candles.csv')
mk['y'] = (mk['result'] == 'yes').astype(int)
outy = dict(zip(mk['ticker'], mk['y']))
slugmap = dict(zip(mk['ticker'], mk['slug']))
print('markets', len(mk), '| movies', mk['slug'].nunique(), '| candle rows', len(cd))

SNAPS = {'5d':5*86400,'3d':3*86400,'2d':2*86400,'1d':86400,'12h':12*3600,'6h':6*3600,'3h':3*3600,'1h':3600}
cd = cd[cd['mid'].notna()].copy()  # tradeable (real two-sided quote) minutes only
rows = []
for tk, g in cd.groupby('ticker'):
    for name, s in SNAPS.items():
        gg = g[g['secs_to_close'] >= s]
        if len(gg):
            r = gg.loc[gg['secs_to_close'].idxmin()]   # most recent quote as of the snap
            rows.append({'ticker':tk,'slug':slugmap[tk],'snap':name,'mid':float(r['mid']),'y':outy[tk]})
obs = pd.DataFrame(rows)
print('snap observations', len(obs))
print(obs['snap'].value_counts().reindex(list(SNAPS)).to_dict())
"""

C_CALIB = """def brier(p, y):
    p = np.asarray(p, float); y = np.asarray(y, float)
    return float(np.mean((p - y)**2))

base = float(obs['y'].mean())
print('cohort base rate (yes):', round(base, 3))
edges = np.linspace(0, 1, 11)
def calib(df):
    b = pd.cut(df['mid'], edges, include_lowest=True)
    return df.groupby(b, observed=True).agg(n=('y','size'), p_mkt=('mid','mean'), p_emp=('y','mean'))

print('\\n=== overall reliability (snaps pooled) ===')
print(calib(obs).round(3).to_string())
bs_m = brier(obs['mid'], obs['y']); bs_b = brier([base]*len(obs), obs['y'])
print('\\nBrier(market)=', round(bs_m,4), '| Brier(base-rate)=', round(bs_b,4),
      '| skill=', round(1 - bs_m/bs_b, 3))

print('\\n=== by time-to-close ===')
for snap in SNAPS:
    d = obs[obs['snap'] == snap]
    if len(d) < 5:
        continue
    mm = float(d['mid'].mean()); em = float(d['y'].mean())
    print(snap.rjust(4), 'n=', str(len(d)).rjust(4), 'mean_mid=', round(mm,3),
          'emp=', round(em,3), 'Brier=', round(brier(d['mid'], d['y']),4), 'gap=', round(mm-em,3))

t = calib(obs)
plt.figure(figsize=(5,5))
plt.plot([0,1],[0,1],'k--',lw=1)
plt.scatter(t['p_mkt'], t['p_emp'], s=(t['n']*2).clip(upper=400))
plt.xlabel('market mid'); plt.ylabel('empirical P(yes)')
plt.title('Gate 1: market-mid reliability (pooled snaps)')
plt.savefig(CACHE + '/gate1_reliability.png', dpi=110, bbox_inches='tight')
print('\\nsaved', CACHE + '/gate1_reliability.png')
"""

C_BOOT = """rng = np.random.default_rng(0)
movies = obs['slug'].unique()
groups = {m: obs[obs['slug'] == m] for m in movies}
def boot(fn, n=1000):
    out = []
    for _ in range(n):
        samp = rng.choice(movies, len(movies), replace=True)
        out.append(fn(pd.concat([groups[m] for m in samp])))
    return np.array(out)

bm = boot(lambda d: brier(d['mid'], d['y']))
print('market Brier (cluster-boot by movie, n_movies=' + str(len(movies)) + '):',
      round(brier(obs['mid'], obs['y']), 4),
      'CI95=[', round(np.percentile(bm,2.5),4), ',', round(np.percentile(bm,97.5),4), ']')
bg = boot(lambda d: abs(float(d['mid'].mean()) - float(d['y'].mean())))
print('aggregate calibration gap |mean_mid - emp|:',
      round(abs(float(obs['mid'].mean()) - float(obs['y'].mean())), 4),
      'CI95=[', round(np.percentile(bg,2.5),4), ',', round(np.percentile(bg,97.5),4), ']')
print('\\nNOTE: 16 movies -> wide CIs by design; directional read only (M5 asymmetric-fork stance).')
"""

nb = nbf.v4.new_notebook()
nb.cells = [nbf.v4.new_markdown_cell(MD), nbf.v4.new_code_cell(C_LOAD),
            nbf.v4.new_code_cell(C_CALIB), nbf.v4.new_code_cell(C_BOOT)]
nb.metadata.kernelspec = {"name": "python3", "display_name": "Python 3", "language": "python"}
os.makedirs("notebooks", exist_ok=True)
nbf.write(nb, "notebooks/gate1_calibration.ipynb")
print("wrote notebooks/gate1_calibration.ipynb")
