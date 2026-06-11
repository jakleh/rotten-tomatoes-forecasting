"""Pure helpers for the p_fresh regression program (battery + candidates + bench).

Spec: ``plans/plan_p_fresh_regression.md`` v2 ("Pinned conventions" + "Shared pure
module") — every constant and convention here is pinned THERE; this module only
implements. No I/O, no network: notebooks and the driver import these.

Conventions implemented (plan tags in brackets):
- subjective-score parser with the F14 family/normalization set;
- global per-(family, level) fresh curves with n>=25 support + neighbor interpolation;
- penalized per-critic anchors, shrunk in PROBABILITY space by n/(n+10) [F8];
- intensity-excess encoding (curve-probability; movie mean signal minus binary mean on
  the same scored rows) [F11];
- shipped-form and shrunk remaining-critic priors + T1's oracle-composition prior;
- visible_state with the tied-timestamp exclusion rule [F7];
- expanded-counts binomial GLM with per-movie weight normalization (equal per snap)
  and GroupKFold C-selection over the lambda-precedent grid [F3];
- temporal-fit row filter with the target/close/M5-window asserts [F4][F5].
"""
from __future__ import annotations

import re
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold

C_GRID = (0.01, 0.1, 1.0, 10.0, 100.0, 1000.0)   # lambda-model precedent [F3a]
SHRINK_K = 10.0                                   # per-critic rate/anchor pseudo-obs
CURVE_MIN_N = 25                                  # per-(family, level) support floor
ANCHOR_MIN_SCORED = 10                            # T3 census threshold
EPS_P = 1e-3                                      # deviance probability clip [F3c]

LETTER_ORD = {"a+": 12, "a": 11, "a-": 10, "b+": 9, "b": 8, "b-": 7, "c+": 6,
              "c": 5, "c-": 4, "d+": 3, "d": 2, "d-": 1, "f+": 0, "f": 0, "f-": 0}
FRAC_DENOMS = {4, 5, 10, 20, 100}

_MIXED_RE = re.compile(r"^(\d+)\s+1/2$")          # "3 1/2" -> 3.5
_FRAC_RE = re.compile(r"^(\.?\d+(?:\.\d+)?)\s*/\s*(\d+(?:\.\d+)?)$")
_NUM_RE = re.compile(r"^\d+(?:\.\d+)?$")


def parse_subjective(s) -> tuple[str, float] | None:
    """(family, level) per the plan's parser spec [F14]; None = unparseable/ambiguous.

    Fraction/star levels are binned to the nearest 0.5; letters map to the 0..12
    ordinal. Bare numerals (no denominator, no star word) are ambiguous -> None.
    """
    if s is None or (isinstance(s, float) and np.isnan(s)):
        return None
    t = str(s).strip().lower()
    if not t:
        return None
    t = t.replace("-plus", "+").replace("-minus", "-").replace(" plus", "+")
    t = re.sub(r"\s+out\s+of\s+", "/", t)
    if t.endswith("%"):
        body = t[:-1].strip()
        if _NUM_RE.match(body):
            return ("frac_100", round(float(body) * 2) / 2)
        return None
    star = False
    if t.endswith("stars") or t.endswith("star"):
        star = True
        t = re.sub(r"\s*stars?$", "", t).strip()
    m = _MIXED_RE.match(t)
    if m:                                          # "3 1/2 stars" (star word required)
        return ("stars", float(m.group(1)) + 0.5) if star else None
    m = _FRAC_RE.match(t)
    if m:                                          # "x/y" (incl. "5/5 stars", ".5/4")
        x, y = float(m.group(1)), float(m.group(2))
        y_int = int(round(y))
        if y_int in FRAC_DENOMS:                   # decimal denominators normalize
            return (f"frac_{y_int}", round(x * 2) / 2)
        return None
    if t in LETTER_ORD:
        return ("letter", float(LETTER_ORD[t]))
    if star and _NUM_RE.match(t):                  # "4 stars", "2.5 stars"
        return ("stars", round(float(t) * 2) / 2)
    return None                                    # bare numerals & residue


def add_parsed_scores(reviews: pd.DataFrame) -> pd.DataFrame:
    """Attach family/level columns + each critic's DOMINANT family (most reviews,
    tie -> most recent review's family); non-dominant-family rows get family=None."""
    out = reviews.copy()
    parsed = out["subjective_score"].map(parse_subjective)
    out["score_family"] = parsed.map(lambda p: p[0] if p else None)
    out["score_level"] = parsed.map(lambda p: p[1] if p else np.nan)
    scored = out[out["score_family"].notna()]
    dom = {}
    for critic, g in scored.groupby("reviewer_name"):
        counts = g["score_family"].value_counts()
        top = counts[counts == counts.max()].index
        if len(top) == 1:
            dom[critic] = top[0]
        else:                                      # tie -> most recent review's family
            dom[critic] = g.sort_values("estimated_timestamp")["score_family"].iloc[-1]
    out["dominant_family"] = out["reviewer_name"].map(dom)
    keep = out["score_family"].notna() & (out["score_family"] == out["dominant_family"])
    out["anchored_scored"] = keep                  # rows usable by curves/anchors
    return out


def global_curves(scored: pd.DataFrame) -> dict[tuple[str, float], float]:
    """{(family, level): P(fresh)} on anchored_scored rows; levels with n<CURVE_MIN_N
    get the inverse-distance blend of the two nearest supported levels in-family
    (fallback: family rate)."""
    rows = scored[scored["anchored_scored"]]
    fresh = rows["tomatometer_sentiment"].eq("positive")
    out: dict[tuple[str, float], float] = {}
    for fam, g in rows.groupby("score_family"):
        fam_rate = float(fresh[g.index].mean())
        stats = (pd.DataFrame({"fresh": fresh[g.index], "level": g["score_level"]})
                 .groupby("level")["fresh"].agg(["mean", "size"]))
        good = stats[stats["size"] >= CURVE_MIN_N]
        for lvl, row in stats.iterrows():
            if row["size"] >= CURVE_MIN_N:
                out[(fam, float(lvl))] = float(row["mean"])
            elif len(good):
                d = np.abs(good.index.to_numpy(dtype=float) - float(lvl))
                w = 1.0 / np.maximum(d, 0.5)
                near = np.argsort(d)[:2]
                out[(fam, float(lvl))] = float(
                    np.average(good["mean"].to_numpy()[near], weights=w[near]))
            else:
                out[(fam, float(lvl))] = fam_rate
    return out


@dataclass
class Anchor:
    """Per-critic shrunk score->P(fresh) curve [F8]: probability-space blend of a
    penalized personal logistic (C=1; constant-rate fallback for one-class critics)
    with the family global curve, w = n/(n+SHRINK_K)."""
    family: str
    n: int
    w: float
    personal: object | None     # fitted LogisticRegression or None
    const_rate: float

    def p(self, level: float, curves: dict) -> float:
        g = curves.get((self.family, float(level)), self.const_rate)
        if self.personal is not None:
            pers = float(self.personal.predict_proba([[level]])[0, 1])
        else:
            pers = self.const_rate
        return self.w * pers + (1 - self.w) * g


def fit_anchor(critic_rows: pd.DataFrame) -> Anchor | None:
    """Anchor from one critic's anchored_scored rows (their dominant family)."""
    rows = critic_rows[critic_rows["anchored_scored"]]
    if not len(rows):
        return None
    fam = rows["score_family"].iloc[0]
    y = rows["tomatometer_sentiment"].eq("positive").to_numpy(int)
    X = rows[["score_level"]].to_numpy(float)
    n = len(rows)
    model = None
    if y.min() != y.max():
        model = LogisticRegression(C=1.0, max_iter=1000).fit(X, y)
    return Anchor(family=fam, n=n, w=n / (n + SHRINK_K),
                  personal=model, const_rate=float(y.mean()))


def intensity_signal(rows: pd.DataFrame, curves: dict,
                     anchors: dict | None = None) -> pd.Series:
    """Per-review curve-probability signal [F11]: global-curve mode (anchors=None,
    the P4 encoding) or shrunk-personal mode (P5). NaN for non-anchored rows."""
    out = pd.Series(np.nan, index=rows.index)
    sel = rows[rows["anchored_scored"]]
    for idx, r in sel.iterrows():
        key = (r["score_family"], float(r["score_level"]))
        if anchors is not None and r["reviewer_name"] in anchors:
            out[idx] = anchors[r["reviewer_name"]].p(r["score_level"], curves)
        else:
            out[idx] = curves.get(key, np.nan)
    return out


def intensity_excess(obs_rows: pd.DataFrame, curves: dict,
                     anchors: dict | None = None) -> tuple[float, int]:
    """Movie-level intensity excess on the SAME scored observed rows [F11]:
    mean(signal) - mean(binary fresh); (0.0, 0) when no scored rows (deploy rule)."""
    sig = intensity_signal(obs_rows, curves, anchors).dropna()
    if not len(sig):
        return 0.0, 0
    binary = obs_rows.loc[sig.index, "tomatometer_sentiment"].eq("positive").mean()
    return float(sig.mean() - binary), int(len(sig))


def critic_rates(pool_rows: pd.DataFrame, *, shrink_k: float | None = None
                 ) -> tuple[dict[str, float], float]:
    """Per-critic fresh rates over the pool rows + the pool-global rate.
    shrink_k=None -> raw rates (shipped semantics); else (fresh + k*g)/(n + k)."""
    fresh = pool_rows["tomatometer_sentiment"].eq("positive")
    g = float(fresh.mean()) if len(pool_rows) else 0.5
    tot = pool_rows.groupby("reviewer_name").size()
    fr = fresh.groupby(pool_rows["reviewer_name"]).sum()
    if shrink_k is None:
        rates = (fr / tot).to_dict()
    else:
        rates = ((fr + shrink_k * g) / (tot + shrink_k)).to_dict()
    return rates, g


def prior_remaining(reviews: pd.DataFrame, pool_slugs: list[str],
                    observed_critics: set[str], *, shrink_k: float | None = None
                    ) -> float:
    """The shipped-form remaining-critic prior (base_rate-weighted fresh rate over
    NOT-yet-observed pool critics; 0.65 fallback) — raw rates reproduce
    estimate_p_fresh's prior exactly; shrink_k gives the P3 variant."""
    pool_rows = reviews[reviews["movie_slug"].isin(pool_slugs)]
    if not len(pool_rows):
        return 0.65
    n_pool = len(pool_slugs)
    base = (pool_rows.groupby("reviewer_name")["movie_slug"].nunique() / n_pool)
    rates, g = critic_rates(pool_rows, shrink_k=shrink_k)
    default = 0.5 if shrink_k is None else g
    rem = base[~base.index.isin(observed_critics)]
    if rem.sum() <= 0:
        return 0.65
    num = sum(b * rates.get(c, default) for c, b in rem.items())
    return float(num / rem.sum())


def prior_actual(remaining_rows: pd.DataFrame, rates: dict[str, float],
                 global_rate: float, *, raw_default: float | None = None) -> float:
    """T1's oracle-composition prior: mean over the ACTUAL remaining reviews of their
    critic's pool rate (missing critic -> global_rate, or raw_default if given)."""
    d = global_rate if raw_default is None else raw_default
    vals = [rates.get(c, d) for c in remaining_rows["reviewer_name"]]
    return float(np.mean(vals)) if vals else np.nan


def visible_state(movie_rows: pd.DataFrame) -> pd.DataFrame:
    """Running (fresh, total) BEFORE each review, estimator view; tied-timestamp rows
    see the state EXCLUDING all tied rows [F7]."""
    g = movie_rows.sort_values("estimated_timestamp")
    fresh = g["tomatometer_sentiment"].eq("positive").to_numpy(int)
    ts = g["estimated_timestamp"].to_numpy()
    cum_f = np.cumsum(fresh) - fresh
    cum_t = np.arange(len(g))
    vis_f, vis_t = cum_f.copy(), cum_t.copy()
    # ties: state at the first row of the tied block applies to the whole block
    first_of_block = np.ones(len(g), dtype=bool)
    first_of_block[1:] = ts[1:] != ts[:-1]
    block_start = np.maximum.accumulate(np.where(first_of_block, np.arange(len(g)), 0))
    vis_f = cum_f[block_start]
    vis_t = cum_t[block_start]
    return pd.DataFrame({"visible_fresh": vis_f, "visible_total": vis_t}, index=g.index)


def emp_logit(fresh, total) -> float:
    """Empirical logit logit((fresh+0.5)/(total+1)) [F15]."""
    p = (fresh + 0.5) / (total + 1.0)
    return float(np.log(p / (1 - p)))


def logit_clip(p, lo: float = 0.01, hi: float = 0.99) -> float:
    """Logit of a model probability, clipped to [0.01, 0.99] [F15]."""
    q = min(max(float(p), lo), hi)
    return float(np.log(q / (1 - q)))


def row_weights(rows: pd.DataFrame) -> pd.Series:
    """Per-EXPANDED-REVIEW weight 1/(n_snaps_m * n_rem) [F3b]: each movie totals 1,
    split equally across its rows (snaps), then across that row's remaining reviews."""
    n_snaps = rows.groupby("slug")["slug"].transform("size")
    return 1.0 / (n_snaps * rows["n_rem"])


def expand_counts(rows: pd.DataFrame, feature_cols: list[str]
                  ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """(X, y, w, groups) — two weighted pseudo-rows per (movie, snap)."""
    w_row = row_weights(rows)
    X, y, w, g = [], [], [], []
    for (_, r), wr in zip(rows.iterrows(), w_row):
        x = [r[c] for c in feature_cols]
        if r["fresh_rem"] > 0:
            X.append(x); y.append(1); w.append(wr * r["fresh_rem"]); g.append(r["slug"])
        if r["n_rem"] - r["fresh_rem"] > 0:
            X.append(x); y.append(0); w.append(wr * (r["n_rem"] - r["fresh_rem"]))
            g.append(r["slug"])
    return (np.array(X, float), np.array(y, int), np.array(w, float),
            np.array(g, object))


def weighted_deviance(p: np.ndarray, y: np.ndarray, w: np.ndarray) -> float:
    """Weighted mean per-review binomial deviance, p clipped to [EPS_P, 1-EPS_P]."""
    pc = np.clip(p, EPS_P, 1 - EPS_P)
    d = -2 * (y * np.log(pc) + (1 - y) * np.log(1 - pc))
    return float(np.sum(w * d) / np.sum(w))


def deviance_of_predictor(rows: pd.DataFrame, pcol: str) -> float:
    """Battery score [F3c] for a per-row probability predictor column."""
    w_row = row_weights(rows)
    p = rows[pcol].to_numpy(float)
    pc = np.clip(p, EPS_P, 1 - EPS_P)
    d1, d0 = -2 * np.log(pc), -2 * np.log(1 - pc)
    num = np.sum(w_row * (rows["fresh_rem"] * d1
                          + (rows["n_rem"] - rows["fresh_rem"]) * d0))
    return float(num / np.sum(w_row * rows["n_rem"]))


def fit_binomial_glm(rows: pd.DataFrame, feature_cols: list[str],
                     c_grid: tuple[float, ...] = C_GRID, n_splits: int = 5):
    """Pinned-convention GLM [F3]: expanded counts, per-movie-normalized weights,
    GroupKFold C-selection on weighted OOS deviance. Returns (model, best_C, oos)."""
    X, y, w, g = expand_counts(rows, feature_cols)
    gkf = GroupKFold(n_splits=min(n_splits, len(np.unique(g))))
    best_c, best_dev = None, np.inf
    for c in c_grid:
        devs = []
        for tr, te in gkf.split(X, y, groups=g):
            m = LogisticRegression(C=c, max_iter=2000)
            m.fit(X[tr], y[tr], sample_weight=w[tr])
            devs.append(weighted_deviance(m.predict_proba(X[te])[:, 1], y[te], w[te]))
        dev = float(np.mean(devs))
        if dev < best_dev:
            best_c, best_dev = c, dev
    model = LogisticRegression(C=best_c, max_iter=2000)
    model.fit(X, y, sample_weight=w)
    return model, best_c, best_dev


F_C1 = ["lp_shipped", "snap_2", "snap_3", "snap_4"]
F_C2 = ["el_P1", "log1p_total", "el_x_log", "lp_P3", "snap_2", "snap_3", "snap_4",
        "el_x_snap2", "el_x_snap3", "el_x_snap4", "mass_consumed"]


def prepare_training_frame(ft: pd.DataFrame) -> pd.DataFrame:
    """Derived candidate-feature columns on the battery feature frame — the SAME
    formulas the bench notebook used (one implementation for scorer + re-scores)."""
    out = ft.copy()
    w = out["n_obs"] / (out["n_obs"] + 20.0)
    out["p_shipped"] = w * out["P1"] + (1 - w) * out["P2"]
    out["lp_shipped"] = [logit_clip(p) for p in out["p_shipped"]]
    out["el_P1"] = [emp_logit(f, n) for f, n in zip(out["fresh_obs"], out["n_obs"])]
    out["log1p_total"] = np.log1p(out["n_obs"])
    out["el_x_log"] = out["el_P1"] * out["log1p_total"]
    out["lp_P3"] = [logit_clip(p) for p in out["P3"]]
    for n in (2, 3, 4):
        out[f"snap_{n}"] = (out["snap_days"] == n).astype(float)
        out[f"el_x_snap{n}"] = out["el_P1"] * out[f"snap_{n}"]
    out["close_dt"] = pd.to_datetime(out["close"], utc=True, format="ISO8601")
    return out


def c2_feature_row(cache: pd.DataFrame, target_slug: str, pool_slugs: list[str],
                   snap_days: int, snap_ts: pd.Timestamp,
                   *, shrink_k: float = SHRINK_K) -> dict | None:
    """One C2′ feature row for a LIVE/bench target at a snap — the bench notebook's
    cell construction, factored (plan_live_scorer pin 3). ``cache`` = estimator-view
    reviews for target ∪ pool. Returns the row dict (F_C2 keys + diagnostics), or
    None when n_obs == 0 (pre-registered deploy rule: raw-prior passthrough outside
    the GLM)."""
    tr = cache[cache["movie_slug"] == target_slug]
    obs = tr[tr["estimated_timestamp"] < snap_ts]
    n_obs = len(obs)
    fresh_obs = int(obs["tomatometer_sentiment"].eq("positive").sum())
    oc = set(obs["reviewer_name"])
    p2 = prior_remaining(cache, pool_slugs, oc)
    p3 = prior_remaining(cache, pool_slugs, oc, shrink_k=shrink_k)
    if n_obs == 0:
        return None
    pool_rows = cache[cache["movie_slug"].isin(pool_slugs)]
    base = pool_rows.groupby("reviewer_name")["movie_slug"].nunique() / len(pool_slugs)
    mass = (float(base[base.index.isin(oc)].sum() / base.sum())
            if base.sum() > 0 else 0.0)
    w = n_obs / (n_obs + 20.0)
    el = emp_logit(fresh_obs, n_obs)
    row = {
        "el_P1": el, "log1p_total": float(np.log1p(n_obs)),
        "el_x_log": el * float(np.log1p(n_obs)), "lp_P3": logit_clip(p3),
        "mass_consumed": mass,
    }
    for n in (2, 3, 4):
        row[f"snap_{n}"] = 1.0 if snap_days == n else 0.0
        row[f"el_x_snap{n}"] = el * row[f"snap_{n}"]
    row.update({"n_obs": n_obs, "fresh_obs": fresh_obs, "P2": p2, "P3": p3,
                "p_shipped": w * (fresh_obs / n_obs) + (1 - w) * p2})
    return row


def temporal_rows(rows: pd.DataFrame, target_slug: str, target_close: pd.Timestamp,
                  min_snap_ts: pd.Timestamp, *, floor: int = 60) -> pd.DataFrame:
    """Training rows for scoring one bench target [F4][F5]: movies closing strictly
    before target_close; asserts target exclusion, the M5 window non-collision, and
    the >=floor movie count (raises on floor breach — the notebook catches + skips)."""
    closes = pd.to_datetime(rows["close"], utc=True)
    sub = rows[closes < target_close]
    assert target_slug not in set(sub["slug"]), f"{target_slug} leaked into its own fit"
    sub_closes = pd.to_datetime(sub["close"], utc=True)
    assert (sub_closes < target_close).all()
    in_window = (sub_closes > min_snap_ts) & (sub_closes <= target_close)
    assert not in_window.any(), (
        f"{target_slug}: training close inside (snap, close] — M5 collision")
    n_movies = sub["slug"].nunique()
    if n_movies < floor:
        raise ValueError(f"{target_slug}: temporal fit has {n_movies} movies (<{floor})")
    return sub
