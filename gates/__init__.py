"""Gate 1 / Gate 2 calibration support code (read-only data layer).

Not part of the shipped ``rotten_tomatoes_forecasting`` package — this is the
analysis/verification layer for ``plans/plan_gate_1_2_calibration.md``:

- ``kalshi_data``: public Kalshi market-data fetcher (no auth, no credentials)
- ``db_facts``:    read-only, ``as_of_id``-pinned queries against the reviews replica
"""
