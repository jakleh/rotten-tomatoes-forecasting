"""Movie-title -> DB-slug mapping shared by the gates drivers.

Extracted verbatim from ``build_cohort.py`` (2026-06-09) so the cohort builder and the
recorder share one source of truth. ``NAME_RE`` parses the movie display name out of a
Kalshi market's ``rules_primary``; ``map_slug`` resolves that name against the DB's slug
universe (exact normalized match, then unique prefix containment, then difflib fuzzy).
"""
from __future__ import annotations

import difflib
import re

NAME_RE = re.compile(r"If (.+?) has a Tomatometer score", re.I)


def norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", s.lower()).strip("_")


def map_slug(name: str | None, db_norm: dict[str, str]) -> str | None:
    """Resolve a display name to a slug; ``db_norm`` maps ``norm(slug) -> slug``."""
    if not name:
        return None
    key = norm(name)
    if key in db_norm:
        return db_norm[key]
    starts = [s for k, s in db_norm.items() if k.startswith(key) or key.startswith(k)]
    if len(starts) == 1:
        return starts[0]
    close = difflib.get_close_matches(key, list(db_norm), n=1, cutoff=0.85)
    return db_norm[close[0]] if close else None
