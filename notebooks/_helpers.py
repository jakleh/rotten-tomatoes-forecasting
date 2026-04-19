"""Shared helpers extracted from stratified_training_validation.ipynb.

Factored so pre_ship_tuning.ipynb (and future notebooks) can import the
validated functions without re-running the 74-cell source notebook.

Source cells in stratified_training_validation.ipynb (for cross-reference):
  - Cell 1: data loading (reviews, movies, close_date_map, first_review_ts)
  - Cell 3: cohort gaps
  - Cell 5: matched_training_slugs
  - Cell 7: snapshot_state, actual_remaining
  - Cell 19: close_day_count, F estimation
  - Cell 31: predict_window, actual_in_window
  - Cell 46: _fit_critic_kde_capped, build_kde_lambda_model_capped
  - Cell 51: critics_in_window, jaccard
  - Cell 52: gap_overlap_ranked_selector, combined_score_selector
  - Cell 54: bootstrap_mae_delta
  - Cell 61: passes_skip_rules_for_snap
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rotten_tomatoes_forecasting import (
    build_critic_profiles,
    build_kde_lambda_model,
    default_training_slugs,
    estimate_lambda,
)
from rotten_tomatoes_forecasting.critic_model import (
    KDELambdaModel,
    _blended_integral,
    _compute_scaling,
)

CACHE_DIR = ROOT / 'notebooks' / '.cache'
CACHE_DIR.mkdir(exist_ok=True)


# --- Data loading ---------------------------------------------------------

def _load_data():
    reviews = pd.read_csv(ROOT / 'reviews.csv')
    reviews['estimated_timestamp'] = pd.to_datetime(
        reviews['estimated_timestamp'], format='ISO8601', utc=True,
    )
    assert reviews['estimated_timestamp'].notna().all(), 'NaT timestamps in reviews'

    movies = pd.read_csv(ROOT / 'movies_index.csv')
    movies['Bet Close Date'] = pd.to_datetime(
        movies['Bet Close Date'], utc=True, errors='coerce',
    )

    now = pd.Timestamp.now(tz='UTC')
    resolved = movies.dropna(subset=['Bet Close Date'])
    resolved = resolved[resolved['Bet Close Date'] < now]

    close_date_map = resolved.set_index('Slug')['Bet Close Date'].to_dict()

    first_review_ts = (
        reviews[reviews['movie_slug'].isin(close_date_map)]
        .groupby('movie_slug')['estimated_timestamp'].min()
    )

    gaps = (
        first_review_ts.rename('first_review_ts')
        .reset_index()
        .rename(columns={'movie_slug': 'slug'})
    )
    gaps['close_ts'] = gaps['slug'].map(close_date_map)
    gaps['gap_days'] = (gaps['close_ts'] - gaps['first_review_ts']).dt.total_seconds() / 86400
    gaps = gaps.dropna(subset=['close_ts', 'gap_days'])
    gaps = gaps[gaps['gap_days'] > 0].reset_index(drop=True)

    gap_lookup = dict(zip(gaps['slug'], gaps['gap_days']))

    return reviews, movies, resolved, close_date_map, first_review_ts, gaps, gap_lookup


reviews, movies, resolved, close_date_map, first_review_ts, gaps, gap_lookup = _load_data()


def gap_for_slug(slug):
    return gap_lookup.get(slug)


# --- Training-set selectors ----------------------------------------------

def matched_training_slugs(target_slug, target_gap, band, n=20):
    """Return (slugs, effective_band). Expands band in 0.5d steps until n found."""
    target_close = close_date_map[target_slug]
    candidates = gaps[
        (gaps['close_ts'] < target_close)
        & (gaps['slug'] != target_slug)
    ].copy()
    current = band
    while True:
        matched = candidates[(candidates['gap_days'] - target_gap).abs() <= current]
        if len(matched) >= n or current > 1000:
            break
        current += 0.5
    selected = matched.sort_values('close_ts', ascending=False).head(n)
    return selected['slug'].tolist(), current


def critics_in_window(slug, window_start, window_days):
    window_end = window_start + pd.Timedelta(days=window_days)
    movie_reviews = reviews[
        (reviews['movie_slug'] == slug)
        & (reviews['estimated_timestamp'] >= window_start)
        & (reviews['estimated_timestamp'] <= window_end)
    ]
    return set(movie_reviews['reviewer_name'])


def jaccard(set_a, set_b):
    union = set_a | set_b
    if not union:
        return 0.0
    return len(set_a & set_b) / len(union)


def gap_overlap_ranked_selector(
    target, target_gap, target_critics, target_window_days, k=20, gap_band=5.0,
):
    """Filter to |gap_diff| ≤ gap_band, rank by Jaccard, top k."""
    target_close = close_date_map[target]
    candidates = gaps[
        (gaps['close_ts'] < target_close)
        & (gaps['slug'] != target)
        & ((gaps['gap_days'] - target_gap).abs() <= gap_band)
    ]
    if len(candidates) == 0:
        return [], 0.0
    rows = []
    for _, row in candidates.iterrows():
        slug = row['slug']
        train_first = first_review_ts.loc[slug]
        train_critics = critics_in_window(slug, train_first, target_window_days)
        score = jaccard(target_critics, train_critics)
        rows.append((slug, score))
    rows.sort(key=lambda x: x[1], reverse=True)
    selected = [r[0] for r in rows[:k]]
    median_score = float(np.median([r[1] for r in rows[:k]])) if rows else 0.0
    return selected, median_score


def rate_matched_selector(
    target, target_gap, target_critics, target_rate, target_window_days,
    k=20, w_gap=0.33, w_jac=0.33, w_vol=0.33,
    sigma_gap=8.0, sigma_vol=5.0,
):
    """Three-feature combined score: gap + Jaccard + volume.

    volume_score = exp(−|target_rate − candidate_rate| / sigma_vol), where rate = critics/day
    in the aligned post-first-review window of duration `target_window_days`.

    Falls back to combined_score_selector when w_vol=0 (no volume weight).
    """
    target_close = close_date_map[target]
    candidates = gaps[
        (gaps['close_ts'] < target_close)
        & (gaps['slug'] != target)
    ]
    if len(candidates) == 0:
        return [], 0.0
    rows = []
    for _, row in candidates.iterrows():
        slug = row['slug']
        gap_diff = abs(row['gap_days'] - target_gap)
        gap_score = 1.0 if np.isinf(sigma_gap) else float(np.exp(-gap_diff / sigma_gap))
        train_first = first_review_ts.loc[slug]
        train_critics = critics_in_window(slug, train_first, target_window_days)
        j = jaccard(target_critics, train_critics)
        cand_rate = len(train_critics) / target_window_days if target_window_days > 0 else 0.0
        rate_diff = abs(target_rate - cand_rate)
        vol_score = float(np.exp(-rate_diff / sigma_vol))
        combined = w_gap * gap_score + w_jac * j + w_vol * vol_score
        rows.append((slug, combined))
    rows.sort(key=lambda x: x[1], reverse=True)
    selected = [r[0] for r in rows[:k]]
    median_score = float(np.median([r[1] for r in rows[:k]])) if rows else 0.0
    return selected, median_score


def combined_score_selector(
    target, target_gap, target_critics, target_window_days,
    k=20, alpha=0.5, sigma_gap=8.0,
):
    """Weighted score alpha*gap_score + (1-alpha)*jaccard over all candidates, top k.

    When sigma_gap is np.inf, gap_score is flat (equivalent to pure Jaccard ranking).
    """
    target_close = close_date_map[target]
    candidates = gaps[
        (gaps['close_ts'] < target_close)
        & (gaps['slug'] != target)
    ]
    if len(candidates) == 0:
        return [], 0.0
    rows = []
    for _, row in candidates.iterrows():
        slug = row['slug']
        gap_diff = abs(row['gap_days'] - target_gap)
        if np.isinf(sigma_gap):
            gap_score = 1.0
        else:
            gap_score = float(np.exp(-gap_diff / sigma_gap))
        train_first = first_review_ts.loc[slug]
        train_critics = critics_in_window(slug, train_first, target_window_days)
        j = jaccard(target_critics, train_critics)
        combined = alpha * gap_score + (1 - alpha) * j
        rows.append((slug, combined, gap_score, j))
    rows.sort(key=lambda x: x[1], reverse=True)
    selected = [r[0] for r in rows[:k]]
    median_score = float(np.median([r[1] for r in rows[:k]])) if rows else 0.0
    return selected, median_score


# --- Snapshot state, actuals ---------------------------------------------

def snapshot_state(target_slug, snap_time):
    target_close = close_date_map[target_slug]
    obs = reviews[
        (reviews['movie_slug'] == target_slug)
        & (reviews['estimated_timestamp'] < snap_time)
        & (reviews['estimated_timestamp'] < target_close)
    ]
    if obs.empty:
        return None
    return {
        'observed_critics': set(obs['reviewer_name']),
        'observed_count': len(obs),
        'first_review_dbc': float(
            (target_close - obs['estimated_timestamp'].min()).total_seconds() / 86400
        ),
    }


def actual_remaining(target_slug, snap_dbc):
    """Count of reviews with 0 < dbc <= snap_dbc. Matches library's internal filter."""
    target_close = close_date_map[target_slug]
    movie_reviews = reviews[reviews['movie_slug'] == target_slug].copy()
    movie_reviews['dbc'] = (
        target_close - movie_reviews['estimated_timestamp']
    ).dt.total_seconds() / 86400
    return int(((movie_reviews['dbc'] > 0) & (movie_reviews['dbc'] <= snap_dbc)).sum())


def actual_in_window(target, dbc_from, dbc_to):
    target_close = close_date_map[target]
    movie_reviews = reviews[reviews['movie_slug'] == target].copy()
    movie_reviews['dbc'] = (
        target_close - movie_reviews['estimated_timestamp']
    ).dt.total_seconds() / 86400
    return int(((movie_reviews['dbc'] > dbc_to) & (movie_reviews['dbc'] <= dbc_from)).sum())


def close_day_count(slug):
    """Reviews timestamped to the close day (any confidence)."""
    target_close = close_date_map[slug]
    movie_reviews = reviews[reviews['movie_slug'] == slug]
    same_day = movie_reviews['estimated_timestamp'].dt.floor('D') == target_close.floor('D')
    return int(same_day.sum())


# --- Skip rule ------------------------------------------------------------

def passes_skip_rules_for_snap(state, snap_dbc, min_first_review_dbc=None, min_critics=3):
    """Snap-adaptive skip rule. Default: first_review_dbc >= snap_dbc + 1."""
    if state is None:
        return False, 'no observations'
    if min_first_review_dbc is None:
        min_first_review_dbc = snap_dbc + 1.0
    if state['first_review_dbc'] < min_first_review_dbc:
        return False, f'first_review_dbc={state["first_review_dbc"]:.2f} < {min_first_review_dbc}'
    if len(state['observed_critics']) < min_critics:
        return False, f'critics={len(state["observed_critics"])} < {min_critics}'
    return True, None


# --- Bandwidth-capped KDE model ------------------------------------------

def _fit_critic_kde_capped(
    timing_data, population_prior, shrinkage_k, bandwidth_floor, bandwidth_ceiling,
):
    n = len(timing_data)
    result = {'empirical': None, 'n': n, 'k': shrinkage_k}
    if n < 2 or timing_data.std() == 0:
        return result
    try:
        kde = gaussian_kde(timing_data)
        effective_bw = kde.factor * timing_data.std()
        if effective_bw < bandwidth_floor:
            kde.set_bandwidth(bandwidth_floor / timing_data.std())
        elif effective_bw > bandwidth_ceiling:
            kde.set_bandwidth(bandwidth_ceiling / timing_data.std())
        result['empirical'] = kde
    except (np.linalg.LinAlgError, ValueError):
        pass
    return result


def build_kde_lambda_model_capped(
    profiles, shrinkage_k=3.0, bandwidth_floor=0.5, bandwidth_ceiling=1.0,
):
    all_timing = np.concatenate(profiles.df['timing_data'].values)
    population_prior = gaussian_kde(all_timing)
    pop_bw = population_prior.factor * all_timing.std()
    if pop_bw > bandwidth_ceiling:
        population_prior.set_bandwidth(bandwidth_ceiling / all_timing.std())

    critic_kdes = {}
    for _, row in profiles.df.iterrows():
        timing = np.array(row['timing_data'])
        entry = _fit_critic_kde_capped(
            timing, population_prior, shrinkage_k, bandwidth_floor, bandwidth_ceiling,
        )
        critic_kdes[row['reviewer_name']] = entry

    return KDELambdaModel(
        profiles=profiles,
        population_prior=population_prior,
        critic_kdes=critic_kdes,
        shrinkage_k=shrinkage_k,
        bandwidth_floor=bandwidth_floor,
    )


# --- Window-integrated prediction -----------------------------------------

def predict_window(
    model, dbc_from, dbc_to, observed_critics,
    observed_count=None, first_review_dbc=None,
):
    """Predict reviews in window (dbc_to, dbc_from]. Applies scaling if obs given."""
    pop_integral = model.population_prior.integrate_box_1d(dbc_to, dbc_from)
    expected = 0.0
    for _, row in model.profiles.df.iterrows():
        name = row['reviewer_name']
        if name in observed_critics:
            continue
        w = row['base_rate']
        entry = model.critic_kdes.get(name)
        if entry is None:
            continue
        integral = _blended_integral(
            entry, model.population_prior, dbc_to, dbc_from, pop_integral=pop_integral,
        )
        expected += w * integral
    if observed_count is not None and first_review_dbc is not None:
        scaling = _compute_scaling(model, dbc_from, observed_count, first_review_dbc)
        expected *= scaling
    return expected


# --- Weighted KDE (similarity-weighted training contributions) ----------

def combined_score_with_scores(
    target, target_gap, target_critics, target_window_days,
    k=20, alpha=0.5, sigma_gap=8.0,
):
    """Like combined_score_selector but returns {slug: score} for the top-k.

    Scores are the raw combined_score values in [0, 1], usable as KDE weights.
    """
    target_close = close_date_map[target]
    candidates = gaps[
        (gaps['close_ts'] < target_close)
        & (gaps['slug'] != target)
    ]
    if len(candidates) == 0:
        return {}
    rows = []
    for _, row in candidates.iterrows():
        slug = row['slug']
        gap_diff = abs(row['gap_days'] - target_gap)
        gap_score = 1.0 if np.isinf(sigma_gap) else float(np.exp(-gap_diff / sigma_gap))
        train_first = first_review_ts.loc[slug]
        train_critics = critics_in_window(slug, train_first, target_window_days)
        j = jaccard(target_critics, train_critics)
        combined = alpha * gap_score + (1 - alpha) * j
        rows.append((slug, combined))
    rows.sort(key=lambda x: x[1], reverse=True)
    return {slug: score for slug, score in rows[:k]}


def build_weighted_critic_profiles(
    reviews_df, close_date_map_local, training_scores, verbose=False,
):
    """Like build_critic_profiles but with per-movie similarity weights.

    training_scores: dict {slug: weight}. Higher weight = more similar to target.
    Weights are L1-normalized so they sum to 1 across the training set.

    Each critic's base_rate becomes `sum_of_weights_for_movies_this_critic_reviewed`.
    Each review's timing_data point is tagged with its source movie's weight, so
    downstream KDE fitting can use per-point weights.
    """
    training_slugs = list(training_scores.keys())
    n_movies = len(training_slugs)
    raw_weights = np.array([training_scores[s] for s in training_slugs], dtype=float)
    # Normalize so the weights sum to n_movies (so base_rate range is comparable to unweighted).
    # Under equal weighting, each movie's weight = 1.0, summing to n_movies.
    # With similarity weights, normalize total to n_movies so scale matches.
    total_w = raw_weights.sum()
    if total_w <= 0:
        # Fallback: uniform weights when all similarity scores are zero
        # (e.g., target has no critic overlap with any candidate).
        norm_weights = np.ones_like(raw_weights)
    else:
        norm_weights = raw_weights * (n_movies / total_w)
    slug_weight = dict(zip(training_slugs, norm_weights))

    train = reviews_df[reviews_df['movie_slug'].isin(training_slugs)].copy()
    close_map = pd.Series(close_date_map_local)
    train['bet_close'] = train['movie_slug'].map(close_map)
    train['days_before_close'] = (
        train['bet_close'] - train['estimated_timestamp']
    ).dt.total_seconds() / 86400
    train = train[train['days_before_close'] > 0].copy()
    train['movie_weight'] = train['movie_slug'].map(slug_weight)

    rows = []
    for name, group in train.groupby('reviewer_name'):
        # base_rate: weighted sum of distinct movies this critic reviewed (/n_movies since normalized)
        movies_seen = group['movie_slug'].unique()
        base_rate = float(sum(slug_weight[s] for s in movies_seen) / n_movies)
        fresh = (group['tomatometer_sentiment'] == 'positive').sum()
        total = len(group)
        timing = group['days_before_close'].values.tolist()
        weights = group['movie_weight'].values.tolist()
        rows.append({
            'reviewer_name': name,
            'base_rate': base_rate,
            'fresh_rate': fresh / total if total > 0 else 0.5,
            'timing_data': timing,
            'timing_weights': weights,
            'n_reviews': total,
        })

    df = pd.DataFrame(
        rows,
        columns=['reviewer_name', 'base_rate', 'fresh_rate', 'timing_data',
                 'timing_weights', 'n_reviews'],
    )
    if verbose:
        print(f'[weighted profiles] {len(df)} critics, {n_movies} training movies')
        if len(df):
            print(f'[weighted profiles] sum(base_rate)={df["base_rate"].sum():.2f}  '
                  f'(unweighted would be {train.groupby("reviewer_name")["movie_slug"].nunique().sum()/n_movies:.2f})')

    from rotten_tomatoes_forecasting.critic_model import CriticProfiles
    return CriticProfiles(df=df, training_slug_count=n_movies)


def _fit_weighted_critic_kde(
    timing_data, weights, population_prior, shrinkage_k, bandwidth_floor, bandwidth_ceiling,
):
    """Weighted gaussian_kde with bandwidth floor+ceiling, same fallback policy as unweighted."""
    n = len(timing_data)
    result = {'empirical': None, 'n': n, 'k': shrinkage_k}
    if n < 2 or np.std(timing_data) == 0:
        return result
    w = np.asarray(weights, dtype=float)
    if w.sum() <= 1e-12:
        return result
    try:
        kde = gaussian_kde(timing_data, weights=w)
        effective_bw = kde.factor * np.std(timing_data)
        if effective_bw < bandwidth_floor:
            kde.set_bandwidth(bandwidth_floor / np.std(timing_data))
        elif effective_bw > bandwidth_ceiling:
            kde.set_bandwidth(bandwidth_ceiling / np.std(timing_data))
        result['empirical'] = kde
    except (np.linalg.LinAlgError, ValueError):
        pass
    return result


def build_weighted_kde_lambda_model(
    profiles, shrinkage_k=3.0, bandwidth_floor=0.5, bandwidth_ceiling=0.7,
):
    """Weighted KDE model. Per-critic AND population KDEs use per-point weights
    drawn from CriticProfiles.df['timing_weights'].

    profiles.df MUST have a 'timing_weights' column (produced by build_weighted_critic_profiles).
    """
    all_timing = np.concatenate(profiles.df['timing_data'].values)
    all_weights = np.concatenate([np.asarray(w) for w in profiles.df['timing_weights'].values])
    population_prior = gaussian_kde(all_timing, weights=all_weights)
    pop_bw = population_prior.factor * np.std(all_timing)
    if pop_bw > bandwidth_ceiling:
        population_prior.set_bandwidth(bandwidth_ceiling / np.std(all_timing))

    critic_kdes = {}
    for _, row in profiles.df.iterrows():
        timing = np.array(row['timing_data'])
        weights = np.array(row['timing_weights'])
        entry = _fit_weighted_critic_kde(
            timing, weights, population_prior, shrinkage_k, bandwidth_floor, bandwidth_ceiling,
        )
        critic_kdes[row['reviewer_name']] = entry

    return KDELambdaModel(
        profiles=profiles,
        population_prior=population_prior,
        critic_kdes=critic_kdes,
        shrinkage_k=shrinkage_k,
        bandwidth_floor=bandwidth_floor,
    )


# --- Path B-lite feature extraction --------------------------------------

def critic_activity_counts(training_slugs=None):
    """Return dict: critic_name -> count of movies the critic has reviewed.

    If training_slugs is None, counts across the whole cohort (useful for
    feature computation at target time). Else restricts to the given slugs
    (useful for building a feature before the target's close date).
    """
    sub = reviews if training_slugs is None else reviews[reviews['movie_slug'].isin(training_slugs)]
    return sub.groupby('reviewer_name')['movie_slug'].nunique().to_dict()


def observed_review_stats(slug, window_start, window_days, activity_lookup,
                          low_activity_threshold=5):
    """Compute per-movie observed-window statistics used as Path B-lite features.

    - n_critics                total unique critics in window
    - rate                     n_critics / window_days (critics per day)
    - top_critic_frac          fraction of reviews with top_critic=True
    - low_activity_frac        fraction of observed critics whose activity_lookup < threshold
    - pub_diversity            unique publication_name count in window
    - pub_entropy              Shannon entropy over publication_name frequencies
    """
    if window_days <= 0:
        return {'n_critics': 0, 'rate': 0.0, 'top_critic_frac': 0.0,
                'low_activity_frac': 0.0, 'pub_diversity': 0, 'pub_entropy': 0.0}
    window_end = window_start + pd.Timedelta(days=window_days)
    sub = reviews[
        (reviews['movie_slug'] == slug)
        & (reviews['estimated_timestamp'] >= window_start)
        & (reviews['estimated_timestamp'] <= window_end)
    ]
    n_reviews = len(sub)
    if n_reviews == 0:
        return {'n_critics': 0, 'rate': 0.0, 'top_critic_frac': 0.0,
                'low_activity_frac': 0.0, 'pub_diversity': 0, 'pub_entropy': 0.0}

    critics = set(sub['reviewer_name'])
    n_critics = len(critics)
    rate = n_critics / window_days
    # top_critic stored as 't' / 'f' in this CSV (verified)
    is_top = sub['top_critic'].astype(str).str.lower().isin(['t', 'true'])
    top_frac = float(is_top.mean())
    # low-activity critic fraction — activity_lookup is the pre-computed dict
    low_activity = sum(1 for c in critics if activity_lookup.get(c, 0) < low_activity_threshold)
    low_activity_frac = low_activity / n_critics
    # publication diversity
    pubs = sub['publication_name'].fillna('unknown')
    pub_diversity = pubs.nunique()
    pub_counts = pubs.value_counts(normalize=True).values
    pub_entropy = float(-(pub_counts * np.log(pub_counts + 1e-12)).sum())
    return {
        'n_critics': n_critics,
        'rate': float(rate),
        'top_critic_frac': top_frac,
        'low_activity_frac': float(low_activity_frac),
        'pub_diversity': int(pub_diversity),
        'pub_entropy': pub_entropy,
    }


# --- Per-target base_rate adjustment (Option C tiered lookup) ----------

def _assign_tier(value, cutoffs):
    """Returns 0..len(cutoffs) based on which interval `value` falls into."""
    for i, c in enumerate(cutoffs):
        if value <= c:
            return i
    return len(cutoffs)


def compute_base_rate_multiplier_matrix(
    target_tier_map,         # dict: slug -> target_tier (0..3)
    critic_tier_map,         # dict: critic_name -> critic_tier (0..2)
    n_critic_tiers=3,
    n_target_tiers=4,
):
    """Compute empirical multiplier matrix from cohort.

    multiplier[c_tier, t_tier] = P(review | c_tier, t_tier) / P(review | c_tier)

    Where P(review | c_tier) is the average rate at which critics in c_tier
    review any target in the cohort, and P(review | c_tier, t_tier) is the
    average rate for targets in t_tier specifically.
    """
    # Gather per-pair review indicator
    critic_targets = reviews.groupby('reviewer_name')['movie_slug'].apply(set).to_dict()

    # Count (critic_tier, target_tier) cells
    cell_reviewed = np.zeros((n_critic_tiers, n_target_tiers))
    cell_possible = np.zeros((n_critic_tiers, n_target_tiers))

    target_slugs = [s for s in target_tier_map if s in critic_targets or True]  # iterate over all

    for critic, c_tier in critic_tier_map.items():
        reviewed = critic_targets.get(critic, set())
        for slug, t_tier in target_tier_map.items():
            cell_possible[c_tier, t_tier] += 1
            if slug in reviewed:
                cell_reviewed[c_tier, t_tier] += 1

    # Compute rates
    with np.errstate(divide='ignore', invalid='ignore'):
        cell_rate = cell_reviewed / cell_possible
        row_rate = cell_reviewed.sum(axis=1) / cell_possible.sum(axis=1)
        multiplier = cell_rate / row_rate.reshape(-1, 1)
    multiplier = np.nan_to_num(multiplier, nan=1.0, posinf=1.0, neginf=1.0)
    return multiplier, cell_rate, row_rate


def adjusted_predict_window(
    model, dbc_from, dbc_to, observed_critics,
    target_tier, critic_tier_map, multiplier_matrix,
    observed_count=None, first_review_dbc=None,
    scaling_threshold=40.0, scaling_clamp=(0.5, 2.0),
):
    """Like predict_window_custom but with per-critic base_rate multipliers keyed by tier.

    target_tier: int in [0, n_target_tiers-1] for the target being predicted.
    critic_tier_map: dict reviewer_name -> tier.
    multiplier_matrix: [n_critic_tiers, n_target_tiers] from compute_base_rate_multiplier_matrix.
    """
    pop_integral = model.population_prior.integrate_box_1d(dbc_to, dbc_from)
    expected = 0.0
    for _, row in model.profiles.df.iterrows():
        name = row['reviewer_name']
        if name in observed_critics:
            continue
        w = row['base_rate']
        c_tier = critic_tier_map.get(name, 0)  # default to low tier if unknown
        mult = multiplier_matrix[c_tier, target_tier]
        w_adjusted = w * mult
        entry = model.critic_kdes.get(name)
        if entry is None:
            continue
        integral = _blended_integral(
            entry, model.population_prior, dbc_to, dbc_from, pop_integral=pop_integral,
        )
        expected += w_adjusted * integral
    if observed_count is not None and first_review_dbc is not None:
        scaling = _compute_scaling_custom(
            model, dbc_from, observed_count, first_review_dbc,
            threshold=scaling_threshold, clamp=scaling_clamp,
        )
        expected *= scaling
    return expected


# --- Custom scaling (for BACKLOG §1.8 clamp sweep) -----------------------

def _compute_scaling_custom(
    model, days_before_close, observed_count, first_review_dbc,
    threshold=40.0, clamp=(0.5, 2.0),
):
    """Like `rotten_tomatoes_forecasting.critic_model._compute_scaling`
    but with tunable threshold and clamp tuple. Defaults match library.
    """
    pop_integral = model.population_prior.integrate_box_1d(
        days_before_close, first_review_dbc,
    )
    expected_so_far = 0.0
    for _, row in model.profiles.df.iterrows():
        w = row['base_rate']
        entry = model.critic_kdes.get(row['reviewer_name'])
        if entry is None:
            continue
        integral = _blended_integral(
            entry, model.population_prior, days_before_close, first_review_dbc,
            pop_integral=pop_integral,
        )
        expected_so_far += w * integral
    if expected_so_far < threshold:
        return 1.0
    scaling = observed_count / expected_so_far
    return max(clamp[0], min(clamp[1], scaling))


def predict_window_custom(
    model, dbc_from, dbc_to, observed_critics,
    observed_count=None, first_review_dbc=None,
    scaling_threshold=40.0, scaling_clamp=(0.5, 2.0),
):
    """Window-integrated KDE prediction with tunable scaling threshold and clamp.

    Mirrors the library's internal scaling logic but exposes clamp for sweep tests.
    """
    pop_integral = model.population_prior.integrate_box_1d(dbc_to, dbc_from)
    expected = 0.0
    for _, row in model.profiles.df.iterrows():
        name = row['reviewer_name']
        if name in observed_critics:
            continue
        w = row['base_rate']
        entry = model.critic_kdes.get(name)
        if entry is None:
            continue
        integral = _blended_integral(
            entry, model.population_prior, dbc_to, dbc_from, pop_integral=pop_integral,
        )
        expected += w * integral
    if observed_count is not None and first_review_dbc is not None:
        scaling = _compute_scaling_custom(
            model, dbc_from, observed_count, first_review_dbc,
            threshold=scaling_threshold, clamp=scaling_clamp,
        )
        expected *= scaling
    return expected


# --- Bootstrap paired CI --------------------------------------------------

def bootstrap_mae_delta(deltas, n_boot=1000, seed=42):
    """Point estimate + paired bootstrap 95% CI on mean of deltas.

    Positive delta = method has smaller MAE than control (method is better).
    """
    rng = np.random.default_rng(seed)
    n = len(deltas)
    if n == 0:
        return np.nan, np.nan, np.nan
    boot_means = np.array([
        rng.choice(deltas, size=n, replace=True).mean()
        for _ in range(n_boot)
    ])
    return (
        float(np.mean(deltas)),
        float(np.quantile(boot_means, 0.025)),
        float(np.quantile(boot_means, 0.975)),
    )
