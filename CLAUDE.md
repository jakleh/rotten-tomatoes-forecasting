# CLAUDE.md

## Project Overview

Finding profitable edges in Kalshi prediction markets for Rotten Tomatoes Tomatometer scores. Combines a scraped review database with Kalshi market price histories to identify where and when the market misprices movie scores.

This project connects to the same Neon PostgreSQL database used by the RT scraper (separate repo at `~/Desktop/rotten-tomatoes-analysis/`), but is fully independent -- no shared code, no shared config.

## Development Philosophy

The goal is `data -> max profit`. Premature formalization (investing in specific models before understanding where the edge actually is) is the main risk. The workflow:

1. **Build dataset** — scrape, dump, enrich. The richer and more unique the dataset, the better the eventual strategies.
2. **Observe informally** — stare at charts, brainstorm, look for patterns. This is a creative exercise. Brainstorm notes go in `brainstorm/` (gitignored).
3. **Cheap quantitative gut-check** — not modeling, just counting. "How many of 141 movies show this?" / "What's the median edge in cents?" / "Does this scatter plot have any structure?" Kills bad ideas in 20 minutes.
4. **Iterate 1-3** until concepts survive the gut-checks.
5. **Formalize** the survivors into modular functions. By this point the ideas are crystal clear, so the math is just implementation.

**Current phase:** Steps 1-2. Building the dataset (scraping ~141 movies into the reviews DB, collecting Kalshi price histories). Brainstorming strategies from informal observation of the price data. Not yet formalizing.

## File Structure

```
├── pyproject.toml              # Dependencies (uv managed)
├── .gitignore
├── CLAUDE.md                   # This file
├── BACKLOG.md                  # Ideas to explore, infrastructure, platform mechanics
├── SOURCES.md                  # Literature, data, and hard numbers to gather
├── PROMPTS.md                  # Handoff prompts for new conversations
├── notebooks/                  # One notebook per idea/exploration
│   └── poisson_binomial_threshold.ipynb
├── rt-price-histories/         # Kalshi market price CSVs (~141 movies, minute/hour/day)
├── plans/                      # Implementation plans (gitignored)
└── brainstorm/                 # Strategy brainstorms (gitignored)
```

## Tech Stack

- **Language**: Python >= 3.13
- **Package manager**: uv
- **Database**: Neon (serverless PostgreSQL) via SQLAlchemy + psycopg2-binary
- **Analysis**: pandas, numpy, scipy, matplotlib
- **Notebooks**: Jupyter via ipykernel

## Database Connection

The notebook connects to the same Neon PostgreSQL instance used by the scraper. Connection is via `DATABASE_URL` environment variable.

```bash
# Launch notebook with database access
DATABASE_URL="postgresql://..." uv run jupyter notebook
```

- Neon cold-starts take ~1-3s on first connection
- `sslmode=require` is needed for Neon connections
- Connection strings starting with `postgres://` must be rewritten to `postgresql://` for SQLAlchemy

## Database Schema

### `reviews` table (Neon PostgreSQL)

This table is populated by the RT scraper (separate project). It is **insert-only** -- reviews are never deleted or updated. The scraper runs every 50 minutes via Cloud Run Jobs.

| Field | Type | Description |
|---|---|---|
| id | SERIAL | Auto-increment primary key |
| unique_review_id | TEXT (UNIQUE) | MD5 hash of (movie_slug + name + publication + rating). Dedup key. |
| movie_slug | TEXT | Movie identifier (e.g., "project_hail_mary") |
| reviewer_name | TEXT | Critic's name |
| publication_name | TEXT | Publication (e.g., "The Guardian") |
| top_critic | BOOLEAN | True if scraped from top-critics filter |
| tomatometer_sentiment | TEXT | **"positive" or "negative"** -- this is the fresh/rotten signal |
| subjective_score | TEXT | Critic's rating in their own scale (e.g., "3/5", "A-", "8/10") |
| written_review | TEXT | Review snippet text |
| site_timestamp_text | TEXT | Raw RT relative timestamp as scraped (e.g., "5m", "3h", "2d", "Mar 20") |
| scrape_time | TIMESTAMPTZ | UTC datetime when the scrape ran |
| estimated_timestamp | TIMESTAMPTZ | Computed absolute time: scrape_time minus the offset in site_timestamp_text |
| timestamp_confidence | TEXT | Timestamp granularity: "m" (minute-level), "h" (hour-level), "d" (day-level or date-format) |
| page_position | INTEGER | 0-indexed position in scrape result (0 = newest review on page) |

**Indexes**: `idx_reviews_movie_slug` on `(movie_slug)`, `idx_reviews_movie_timestamp` on `(movie_slug, estimated_timestamp)`

### Key data characteristics

- **Tomatometer score** = count of "positive" sentiment / total reviews for a movie
- **Deduplication**: Each review appears exactly once (enforced by UNIQUE on `unique_review_id`)
- **Two-pass scraping**: The scraper runs `top-critics` then `all-critics` filters. A review scraped as a top critic has `top_critic = True`. The same review is NOT duplicated when scraped again in the all-critics pass (dedup catches it).
- **Timestamp precision varies**: "m"-confidence timestamps are accurate to ~1 minute, "h" to ~1 hour, "d" to ~1 day. KDE-based analysis should be aware of this noise.
- **Scrape frequency**: Every 50 minutes. Reviews with relative timestamps ("5m", "3h") get more precise `estimated_timestamp` values than date-format ones ("Mar 20").
- **Movies tracked**: Configured in the scraper's `movies.json`. Being expanded to cover all ~141 movies that had Kalshi RT markets. Originally tracked: `project_hail_mary`, `ready_or_not_2_here_i_come`, `forbidden_fruits_2026`, `they_will_kill_you`.

## Approaches & Ideas

No single model is settled. Specific approaches live in `brainstorm/` (gitignored) and in individual notebooks under `notebooks/`. The first approach explored was a Poisson/binomial threshold-breaking model (see `brainstorm/brainstorm_poisson_binomial_threshold.md` and `notebooks/poisson_binomial_threshold.ipynb`).

The pattern for new ideas: brainstorm -> gut-check -> notebook -> formalize (if it survives). See `brainstorm/` for all strategy brainstorms and `BACKLOG.md` for the full index of ideas to explore.

## Backlog & Strategy

`BACKLOG.md` is the index of ideas to explore, data infrastructure to build, platform mechanics to look up, and operational items for when we're ready to bet. Read it before starting new work.

**Current state:** One notebook exists (`notebooks/poisson_binomial_threshold.ipynb`) implementing the first model approach. Kalshi price histories for ~141 resolved markets are in `rt-price-histories/`. The RT scraper is being expanded to cover all ~141 movies (in progress). We're in the observe-and-brainstorm phase — no model is finalized.

`SOURCES.md` lists specific data, hard numbers, and literature needed to resolve open questions. `PROMPTS.md` has ready-to-use handoff prompts for starting new conversations.

**Recommended next steps (in order):**
1. **Finish dataset build.** Complete the ~141 movie scrape. Dump the reviews table locally so price histories and review data can be joined by `movie_slug` (folder names in `rt-price-histories/` match DB slugs).
2. **Informal observation + gut-checks on the joint dataset.** See `brainstorm/` for strategy ideas in progress. Key questions: How often does the market misprice relative to actual score? Does that concentrate in low-activity markets? Does trading activity correlate with review count?
3. **Formalize surviving strategies** into modular functions once the patterns are validated.

## How to Run

```bash
# Install dependencies
cd ~/Desktop/rt-analysis
uv sync

# Launch notebook
DATABASE_URL="postgresql://..." uv run jupyter notebook
```

## Dependencies

`pyproject.toml`: sqlalchemy, psycopg2-binary, pandas, numpy, scipy, matplotlib, ipykernel.
