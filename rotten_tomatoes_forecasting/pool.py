"""
Critic pool primitives for Ridge lambda features and p_fresh estimation.

Two callers consume `compute_critic_base_rates` over different training sets:
- `extract_lambda_features` uses the A1 pool (20 most recent resolved before target close).
- `estimate_p_fresh` uses the caller's training slugs.

They do NOT share a pool definition — the slug lists differ. This module provides
the shared primitive without coupling the two callers.
"""

from dataclasses import dataclass

import pandas as pd


@dataclass
class A1Context:
    """LOO-clean A1 pool aggregates used by `extract_lambda_features`.

    Built once per target via `build_a1_pool_context`. Describes:
      - Which critics appear in the 20 most recent resolved movies before target close.
      - Their per-critic base_rate (movies_reviewed_in_A1 / 20).
      - The total sum (≈ mean reviews per movie in the pool).
      - The top-30 critics by base_rate (the A1 "workhorses").

    The three finite-pool Ridge features are computed from this context by comparing
    it to the target's observed critic set at snap time.
    """

    training_slugs: list[str]
    base_rate: dict[str, float]
    total_sum: float
    top_tier: set[str]


def _most_recent_resolved_slugs(
    close_date_map: dict[str, pd.Timestamp],
    before: pd.Timestamp,
    n: int,
    exclude_slug: str | None = None,
) -> list[str]:
    """Return the `n` slugs with the largest close date strictly before `before`.

    LOO-clean: `exclude_slug` is always dropped. Ties broken by slug name (stable).
    """
    items = [(slug, ts) for slug, ts in close_date_map.items() if ts < before]
    if exclude_slug is not None:
        items = [(s, t) for s, t in items if s != exclude_slug]
    items.sort(key=lambda st: (st[1], st[0]), reverse=True)
    return [s for s, _ in items[:n]]


def compute_critic_base_rates(
    reviews_df: pd.DataFrame,
    training_slugs: list[str],
) -> dict[str, float]:
    """Per-critic rate of reviewing any movie in the given training set.

    `base_rate[critic] = distinct_movies_reviewed_in_training / len(training_slugs)`.

    Shared primitive. Callers choose the training_slugs (A1 pool for lambda, caller-owned
    pool for p_fresh).
    """
    n = len(training_slugs)
    if n == 0:
        return {}
    sub = reviews_df[reviews_df["movie_slug"].isin(training_slugs)]
    counts = sub.groupby("reviewer_name")["movie_slug"].nunique()
    return (counts / n).to_dict()


def build_a1_pool_context(
    target_slug: str,
    close_date_map: dict[str, pd.Timestamp],
    reviews_df: pd.DataFrame,
    n: int = 20,
    top_tier_n: int = 30,
) -> A1Context | None:
    """Build the LOO-clean A1 pool context for one target.

    Picks the `n` most recent resolved movies with close date strictly before the
    target's close, target always excluded via `exclude_slug`. Computes per-critic
    base_rate from that pool, identifies the top `top_tier_n` critics by count,
    and caches the total sum.

    `close_date_map` should contain the target (its close is needed as the cutoff).
    LOO exclusion happens inside the function.

    Returns None when the target is absent from `close_date_map`, or when fewer
    than 5 eligible training movies exist (insufficient pool).
    """
    target_close = close_date_map.get(target_slug)
    if target_close is None:
        return None
    training_slugs = _most_recent_resolved_slugs(
        close_date_map, before=target_close, n=n, exclude_slug=target_slug
    )
    if len(training_slugs) < 5:
        return None

    sub = reviews_df[reviews_df["movie_slug"].isin(training_slugs)]
    counts = sub.groupby("reviewer_name")["movie_slug"].nunique()
    base_rate = (counts / n).to_dict()
    total_sum = float(sum(base_rate.values()))
    top_tier = set(counts.nlargest(top_tier_n).index)

    return A1Context(
        training_slugs=training_slugs,
        base_rate=base_rate,
        total_sum=total_sum,
        top_tier=top_tier,
    )
