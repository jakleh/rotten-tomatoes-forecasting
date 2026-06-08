"""Authoring helper: assemble notebooks/gate1b_incremental_info.ipynb.

Codegen only. The notebook does the Gate-1b incremental-info-over-price test
(leave-one-movie-out logistic: price-only vs price+review-signal) on the cached
gate1b_input.csv. Run via nbconvert (sandbox off — Jupyter kernel needs local sockets).
"""
import os
import nbformat as nbf

MD = """# Gate 1b — incremental info over price (directional)

Does a review-derived signal available **at snap** add **out-of-sample** predictive
power **over the market price**? Leave-one-movie-out logistic regression:
- **A (price-only):** `result ~ logit(mid)`
- **B (price + review):** `+ obs_margin` (observed score - threshold) `+ obs_total`

B beats A iff OOS Δlog-loss > 0 **and** its cluster-bootstrap-by-movie CI excludes 0.
If B ~ A, the market already prices the observed review state (no edge from raw reviews;
the hope then rests on better *forecasting the final score* — Gate 2). n=16 movies →
directional.
"""

CODE = """import os, numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, brier_score_loss

CACHE = 'gates/_cache' if os.path.isdir('gates/_cache') else '../gates/_cache'
df = pd.read_csv(CACHE + '/gate1b_input.csv').reset_index(drop=True)
print('rows', len(df), '| movies', df['slug'].nunique(), '| obs_total==0:', int((df['obs_total']==0).sum()))

p = np.clip(df['mid'].values, 0.02, 0.98)
df['logit_mid'] = np.log(p / (1 - p))
for c in ['obs_margin', 'obs_total']:
    sd = df[c].std()
    df[c + '_z'] = (df[c] - df[c].mean()) / (sd if sd else 1.0)

FA = ['logit_mid']
FB = ['logit_mid', 'obs_margin_z', 'obs_total_z']
y = df['y'].values
movies = df['slug'].unique()
predA = np.full(len(df), np.nan); predB = np.full(len(df), np.nan)
for m in movies:
    te = df.index[df['slug'] == m].to_numpy()
    tr = df.index[df['slug'] != m].to_numpy()
    for F, pred in ((FA, predA), (FB, predB)):
        lr = LogisticRegression(max_iter=2000).fit(df.loc[tr, F].values, y[tr])
        pred[te] = lr.predict_proba(df.loc[te, F].values)[:, 1]

llA = log_loss(y, predA, labels=[0,1]); llB = log_loss(y, predB, labels=[0,1])
bsA = brier_score_loss(y, predA); bsB = brier_score_loss(y, predB)
print('\\nLOMO out-of-sample:')
print('  A price-only   : logloss', round(llA,4), '| brier', round(bsA,4))
print('  B price+review : logloss', round(llB,4), '| brier', round(bsB,4))
print('  delta logloss (A-B, >0 = review ADDS):', round(llA-llB,4), '| delta brier:', round(bsA-bsB,4))

rng = np.random.default_rng(0)
dd = pd.DataFrame({'slug': df['slug'].values, 'y': y, 'pA': predA, 'pB': predB})
gb = {m: dd[dd['slug'] == m] for m in movies}
boot = []
for _ in range(1000):
    z = pd.concat([gb[m] for m in rng.choice(movies, len(movies), replace=True)])
    boot.append(log_loss(z['y'], z['pA'], labels=[0,1]) - log_loss(z['y'], z['pB'], labels=[0,1]))
boot = np.array(boot)
print('  delta-logloss cluster-boot CI95 by movie: [', round(np.percentile(boot,2.5),4), ',', round(np.percentile(boot,97.5),4), ']')

lrF = LogisticRegression(max_iter=2000).fit(df[FB].values, y)
print('\\nfull-fit B coefs:', dict(zip(FB, np.round(lrF.coef_[0], 3))))
print('\\nVerdict: review signal adds over price iff delta>0 AND CI lower bound>0.')
"""

nb = nbf.v4.new_notebook()
nb.cells = [nbf.v4.new_markdown_cell(MD), nbf.v4.new_code_cell(CODE)]
nb.metadata.kernelspec = {"name": "python3", "display_name": "Python 3", "language": "python"}
os.makedirs("notebooks", exist_ok=True)
nbf.write(nb, "notebooks/gate1b_incremental_info.ipynb")
print("wrote notebooks/gate1b_incremental_info.ipynb")
