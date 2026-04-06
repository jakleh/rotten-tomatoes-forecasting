#!/usr/bin/env python3
"""Populate empty fields in movies_index.csv from price histories and reviews data."""

import pandas as pd
from pathlib import Path

ROOT = Path(__file__).parent
PRICE_DIR = ROOT / "rt-price-histories"
REVIEWS_PATH = ROOT / "reviews.csv"
INDEX_PATH = ROOT / "movies_index.csv"

# Final-minute price thresholds for classifying resolution
HIT_PRICE = 90   # price >= this → resolved Yes
MISS_PRICE = 10  # price <= this → resolved No


def parse_minute_csv(slug):
    """Extract bet dates and score range from minute-level price CSV."""
    folder = PRICE_DIR / slug
    minute_files = list(folder.glob("*-minute.csv"))
    if not minute_files:
        print(f"  WARNING: No minute CSV for {slug}")
        return None

    df = pd.read_csv(minute_files[0], parse_dates=["timestamp"])
    if df.empty:
        print(f"  WARNING: Empty minute CSV for {slug}")
        return None

    bet_open_date = df["timestamp"].iloc[0].strftime("%Y-%m-%d")
    bet_close_date = df["timestamp"].iloc[-1].strftime("%Y-%m-%d")

    # Parse "Above X" columns from last row, skip NaN
    last_row = df.iloc[-1]
    threshold_prices = []
    for col in df.columns:
        if col.startswith("Above "):
            price = last_row[col]
            if not pd.isna(price):
                threshold_prices.append((int(col.replace("Above ", "")), price))

    threshold_prices.sort(key=lambda x: x[0])

    if len(threshold_prices) < 2:
        print(f"  WARNING: {slug} has fewer than 2 valid threshold prices")
        return None

    # Find boundary via biggest price drop between adjacent sorted thresholds.
    # Stale prices create *increases* when scanning upward, so the biggest *drop*
    # reliably lands at the true hit→miss transition.
    max_drop = 0
    highest_hit = None
    lowest_miss = None

    for i in range(len(threshold_prices) - 1):
        t1, p1 = threshold_prices[i]
        t2, p2 = threshold_prices[i + 1]
        drop = p1 - p2
        if drop > max_drop:
            max_drop = drop
            highest_hit = t1
            lowest_miss = t2

    if highest_hit is None:
        # No drop found — all prices non-decreasing. Check if all hit or all miss.
        if all(p >= HIT_PRICE for _, p in threshold_prices):
            highest_hit = max(t for t, _ in threshold_prices)
        elif all(p <= MISS_PRICE for _, p in threshold_prices):
            lowest_miss = min(t for t, _ in threshold_prices)
        else:
            print(f"  WARNING: {slug} has no clear boundary (max drop = {max_drop:.1f})")

    # Warn on non-monotonic prices (stale data on wrong side of boundary)
    if highest_hit is not None and lowest_miss is not None:
        for t, p in threshold_prices:
            if t <= highest_hit and p <= MISS_PRICE:
                print(f"  WARNING: {slug} stale low price on hit side: Above {t} = {p}")
            elif t >= lowest_miss and p >= HIT_PRICE:
                print(f"  WARNING: {slug} stale high price on miss side: Above {t} = {p}")

    # Compute decimal score range (assuming standard rounding)
    # "Above X" hit → displayed >= X+1 → underlying fraction >= (X+0.5)/100
    # "Above X" miss → displayed <= X → underlying fraction < (X+0.5)/100
    score_low = (highest_hit + 0.5) / 100 if highest_hit is not None else 0.0
    score_high = (lowest_miss + 0.5) / 100 if lowest_miss is not None else 1.0

    if score_low >= score_high:
        print(f"  WARNING: {slug} has inverted score range: {score_low}-{score_high}")

    return {
        "bet_open_date": bet_open_date,
        "bet_close_date": bet_close_date,
        "score_range": f"{score_low:.4f}-{score_high:.4f}",
    }


def load_reviews():
    """Load reviews.csv with parsed timestamps."""
    reviews = pd.read_csv(REVIEWS_PATH)
    reviews["estimated_timestamp"] = pd.to_datetime(reviews["estimated_timestamp"], format="ISO8601", utc=True)
    # Normalize to date for day-level comparisons
    reviews["review_date"] = reviews["estimated_timestamp"].dt.date
    return reviews


def compute_reviews_data(reviews, slug, bet_close_date_str):
    """Compute embargo lift date and review count range for a movie."""
    movie = reviews[reviews["movie_slug"] == slug]

    if movie.empty:
        print(f"  WARNING: No reviews found for {slug}")
        return None

    # Embargo lift: earliest review timestamp
    earliest = movie["estimated_timestamp"].min()
    embargo_lift_date = pd.Timestamp(earliest).strftime("%Y-%m-%d") if pd.notna(earliest) else ""

    # Total reviews at bet close (range due to day-level timestamp ambiguity)
    if bet_close_date_str:
        close_date = pd.Timestamp(bet_close_date_str).date()
        lower = int((movie["review_date"] < close_date).sum())
        upper = int((movie["review_date"] <= close_date).sum())
        total_reviews = str(lower) if lower == upper else f"{lower}-{upper}"
    else:
        total_reviews = ""

    return {
        "embargo_lift_date": embargo_lift_date,
        "total_reviews": total_reviews,
    }


def main():
    index = pd.read_csv(INDEX_PATH, dtype=str)
    row_count_in = len(index)
    reviews = load_reviews()

    print(f"Processing {row_count_in} movies...\n")

    filled = {"Bet Open Date": 0, "Tomatometer Score Range Bet Close": 0,
              "Embargo Lift Date": 0, "Total Reviews Bet Close": 0}
    warnings = 0

    for i, row in index.iterrows():
        slug = row["Slug"]

        # --- Price history data ---
        price_data = parse_minute_csv(slug)

        if price_data:
            # Verify bet close date matches
            existing_close = row.get("Bet Close Date", "")
            if existing_close and existing_close != price_data["bet_close_date"]:
                print(f"  NOTE: {slug} close date mismatch: CSV has {existing_close}, price data has {price_data['bet_close_date']}")
                warnings += 1

            if not row.get("Bet Open Date") or pd.isna(row.get("Bet Open Date")):
                index.at[i, "Bet Open Date"] = price_data["bet_open_date"]
                filled["Bet Open Date"] += 1

            if not row.get("Tomatometer Score Range Bet Close") or pd.isna(row.get("Tomatometer Score Range Bet Close")):
                index.at[i, "Tomatometer Score Range Bet Close"] = price_data["score_range"]
                filled["Tomatometer Score Range Bet Close"] += 1
        else:
            warnings += 1

        # --- Reviews data ---
        bet_close = row.get("Bet Close Date", "")
        reviews_data = compute_reviews_data(reviews, slug, bet_close if pd.notna(bet_close) else "")

        if reviews_data:
            if not row.get("Embargo Lift Date") or pd.isna(row.get("Embargo Lift Date")):
                index.at[i, "Embargo Lift Date"] = reviews_data["embargo_lift_date"]
                filled["Embargo Lift Date"] += 1

            if not row.get("Total Reviews Bet Close") or pd.isna(row.get("Total Reviews Bet Close")):
                index.at[i, "Total Reviews Bet Close"] = reviews_data["total_reviews"]
                filled["Total Reviews Bet Close"] += 1
        else:
            warnings += 1

    # --- Sanity checks ---
    print("\n--- Sanity checks ---")
    errors = []

    # Row count preserved
    assert len(index) == row_count_in, f"Row count changed: {row_count_in} -> {len(index)}"
    print(f"  Row count: {len(index)} (unchanged)")

    # Score bounds in [0, 1] and not inverted
    for _, row in index.iterrows():
        sr = row.get("Tomatometer Score Range Bet Close")
        if pd.notna(sr) and sr:
            low, high = sr.split("-")
            low, high = float(low), float(high)
            if not (0 <= low <= 1 and 0 <= high <= 1):
                errors.append(f"  {row['Slug']}: score bounds out of [0,1]: {sr}")
            if low >= high:
                errors.append(f"  {row['Slug']}: inverted score range: {sr}")

    # Bet open < bet close
    for _, row in index.iterrows():
        bo, bc = row.get("Bet Open Date"), row.get("Bet Close Date")
        if pd.notna(bo) and pd.notna(bc) and bo and bc:
            if bo >= bc:
                errors.append(f"  {row['Slug']}: bet open ({bo}) >= bet close ({bc})")

    # Embargo lift <= bet close
    for _, row in index.iterrows():
        el, bc = row.get("Embargo Lift Date"), row.get("Bet Close Date")
        if pd.notna(el) and pd.notna(bc) and el and bc:
            if el > bc:
                errors.append(f"  {row['Slug']}: embargo lift ({el}) > bet close ({bc})")

    # Review count lower <= upper
    for _, row in index.iterrows():
        tr = row.get("Total Reviews Bet Close")
        if pd.notna(tr) and tr and "-" in tr:
            parts = tr.split("-")
            if int(parts[0]) > int(parts[1]):
                errors.append(f"  {row['Slug']}: review count lower > upper: {tr}")

    if errors:
        print("  FAILURES:")
        for e in errors:
            print(e)
    else:
        print("  Score bounds, date ordering, review counts: all passed")

    # --- Summary ---
    print(f"\n--- Summary ---")
    for col, count in filled.items():
        print(f"  {col}: filled {count} rows")
    print(f"  Warnings: {warnings}")

    index.to_csv(INDEX_PATH, index=False)
    print(f"\nWrote {len(index)} rows to {INDEX_PATH}")


if __name__ == "__main__":
    main()
