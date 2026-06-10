# Prompts

Handoff prompts for starting new conversations on the rotten-tomatoes-forecasting forecasting library. Read CLAUDE.md and BACKLOG.md for full context before using any prompt.

---

## Handoff convention — READ THIS FIRST

Every prompt below is a task payload. Before acting on one, run the session-start ritual:
1. **Foundation:** read `CLAUDE.md` "Current Conventions" (authoritative) + the relevant `BACKLOG.md` section + any findings doc the task names.
2. **Plans:** read any referenced `plans/*.md` in full (gitignored, local).
3. **Sanity-check:** `uv sync`, then `.venv/bin/python -m pytest tests/ -q` (expect 699 green as of 2026-06-10 late PM) before touching code.
4. **Recorder staleness:** `.venv/bin/python -m gates.recorder --check` — exit 1 means run `python -m gates.recorder` (sandbox-off for the DB join; `--no-db` works sandboxed and tops up later). The session ritual IS the §1.7 recorder's scheduler; `gates/recorded/` is the committed system of record.
5. **Then** start the task.

A good NEW handoff prompt gives: the task in one line; the foundation docs to read; a prerequisite/STOP check if the task is data-gated; concrete numbered steps; an explicit decision rule + out-of-scope list; and the deliverable.

## Context for all prompts

This repo is the focused RT-modeling workspace (v0.2.0, Ridge lambda model); the math is pure DataFrame-in / numbers-out, ending at `compute_edge()`. Per the scripts direction (CLAUDE.md "Project Overview"), eventual deployment is a script that imports the edge-calc + kalshi execution helpers — not the orchestrator.

0.2.0 public API (in `rotten_tomatoes_forecasting/`): `compute_edge`, `estimate_lambda`, `estimate_p_fresh`, `fit_lambda_regressor`, `load_default_regressor`, `extract_lambda_features`, `compute_close_day_phase2`, `LambdaRegressor`, `LambdaPrediction`, `naive_lambda`, `naive_p_fresh`. (The 0.1.x KDE API — `build_critic_profiles`, `KDELambdaModel`, `CriticProfiles`, `default_training_slugs`, etc. — was removed.)

---

## Next session — START HERE (handoff 2026-06-10 late PM — p_fresh REGRESSION PROGRAM RAN END-TO-END: BATTERY DECISIVE, BENCH NO WINNER (BAR-INVARIANT); SHADE-AS-INTERIM = THE OPEN OPERATOR DECISION)

**Shipped this session (after `d5f60f1` Gate 3b; operator: "go ahead" through results, weigh-in at the readout):**
1. **Brainstorm v3 signed** (battery decides BUILT / bench rule decides SHIPS / recorder tripwire decides LIVE) → **`plans/plan_p_fresh_regression.md` v2** (pre-build Explore review: 3 BLOCKING spec holes fixed pre-code — v1's C1 rung was hull-bound dead [oracle outside the [obs_rate, prior] interval in 35/60 bench cells, BOTH ingredients over-fresh at T-3d +0.078/+0.062 → every candidate carries an additive calibration escape]; the cache-only training universe was the bench wearing a hat → full-universe pull; decision-protocol contradictions → paired rule + fallback + tripwire).
2. **BUILT + RAN:** `gates/pfresh_lib.py` (+37 tests, suite 662 → **699**) + `gates/build_pfresh_training.py` (pin **653572**: 26,188 rows / 160 slugs → **135 eligible movies / 529 snap-rows**) + `notebooks/pfresh_battery.ipynb` + `notebooks/pfresh_bench.ipynb` (+ codegens). **BATTERY:** remaining-critic priors ≈ NO ranking signal (Spearman −0.05 vs obs_rate +0.874; P3 marginally > P2); **intensity channel DEAD** (increment −0.0002; score→fresh curves are STEP FUNCTIONS — 4.7% of scored mass in p ∈ [0.2, 0.8]; anchors transfer +0.167 but there's no gradient) → **C3 never built, kill measured**; **state-dependence huge** (visible-score coef +0.99 beyond critic FE, OOS +0.1975); bias is **behavior not composition** (oracle-composition keeps +0.054/+0.074); d-row/shrink-k immaterial. **BENCH (locked paired rule, 35 cells, temporal per-close fits):** C1′ +7.8¢ [−5.2, +23.0] (paired vs shade −4.1¢ [−9.1, +0.5]), C2′ +10.0¢ [−4.4, +25.2] (paired −1.9¢ [−7.2, +1.9]) vs the −0.03 shade bar +11.8¢ [−0.3, +26.0] → **NO WINNER → ship nothing** (pre-registered fallback). Both candidates CRUSH the shipped +2.4¢ and fix δ̂ (kill-zone 42% → 27/30%, T-3d hump +0.078 → +0.024/+0.036) — the calibration works, a constant just matches it at n=35.
3. **Post-build Explore review — verdict "trustworthy and over-determined (BAR-INVARIANT)"** (independent full re-execution, every number exact; s*≈−0.045 bar counterfactual +11.6¢ [−0.6, +25.0] — both candidates fail CI-lo>0 under ANY bar). Its attribution catch is folded in: **per-snap "T-3d strength (+25.1¢*)/T-2d crater (−22..−28¢)" are the BAR's numbers too** (shade T-3d +25.1¢*, T-2d −28.4¢; 11/12 T-2d trades identical); candidates' only paired edge = T-2d +6.0¢ [+0.0, +21.6] (n=12). Also folded: [F24] print (earliest-close fit ≈98% April-backfill rows — deploy-parity, disclosed), shade-per-snap rows, reconstruction assert (1.1e-16), oos-only rows, per-snap T2 deviances, shrink-k/raw-clip sensitivities (12h-grid struck → next cycle), floor→skip-and-disclose via the tested `pl.temporal_rows`. Dev-catches on record: a ticker-only merge fanned the 35-cell verdict set to 56 rows on first execution (caught by a point estimate outside its own CI; fixed with len==35 + no-NaN asserts).

**NEXT TASK (operator decision + then mechanical):**
1. **OPERATOR DECISION at the readout:** ship the **shade as a flagged interim** `estimate_p_fresh` change (−0.03 measured +11.8¢ [−0.3, +26.0]; training data independently says s* ≈ −0.04; it IS the current frontier and nearly clears) — or **wait** for the recorder-growth re-run. Either way: the **K≈8 tripwire** governs anything going live, and the re-run (battery + bench re-score on the grown cohort) is one estimator pass over the standing machinery (`pfresh_training_*.csv`, `pfresh_battery_decisions.csv`, the locked bench).
2. If shade ships: small PROTOCOL plan for the library change (constant + provenance comment + tests; the s* refit cadence question rides along).
3. Standing: recorder ritual (step 4); BACKLOG §1.9 backfill + `kalshi-trading/src/series/KXRT/db.py:89` fix (operator-side); next bench re-lock (e.g., +8 movies) restarts ladder bookkeeping per the frozen-bench policy.
4. Next-cycle research leads (measured, not vibes): T5's state signal (visible-score path) as a remaining-rate model input; T1's behavior verdict (late-review rotten-skew) as the structural target; intensity stays dead unless a finer-grained score source appears.

**Read at session start:** CLAUDE.md "Current Conventions" + the Gate-3b and p_fresh-program status paragraphs; `plans/plan_p_fresh_regression.md` — "RESULT" + "Review log"; memories `project_pfresh_regression_result`, `project_gate3b_result`, `reference_db_access`, `reference_notebook_execution`.

**Conventions established this session:** the battery/bench/tripwire split (BUILT/SHIPS/LIVE) is the standing evaluation pattern for estimator work; per-snap candidate reads always ride WITH the bar on the same cells/draws (attribution discipline); paired-on-shared-draws is the comparison form; bar-invariance checks before "no winner" readouts; pre-registered hard-asserts on expected outcomes catch implementation bugs (twice this session).

**Sanity-check on arrival:** `uv sync && .venv/bin/python -m pytest tests/ -q` (expect **699** green); `.venv/bin/python -m gates.recorder --check`; `git log --oneline -3`; `gates/_cache/pfresh_{training_rows,training_features,battery_decisions}.csv` + `gate3b_cells.csv` exist (training rebuild: `python -m gates.build_pfresh_training`, DB sandbox-off, ~2min; notebooks re-execute cache-only). Kalshi sandboxed OK; Neon + nbconvert need `dangerouslyDisableSandbox`.

**Push status:** Gate-3b commit `d5f60f1` pushed; this session's p_fresh-program commit local on `main` — push per operator.

---

## Prior handoff (2026-06-10 PM) — SUPERSEDED same day (p_fresh program ran; results at the readout)

**Shipped this session (one close-session commit on top of `ec50adf`):**
1. **GATE 3b EXECUTED per the locked plan** — driver `gates/build_gate3b.py` (lock chain in stage order: recorded-store book LOCF at midnight-ET snaps → provisional guard checkpoint [reproduced the review's 21/9, 22/12, 11/9, 7/7 exactly] → readiness pin selection [fail set {ANI,BAC,INT} at ALL 3 candidate pins → smallest pin **648979** locked; +POW operator-excluded] → true `snap_density` guard + ≥8 floor [T-4d exactly 8] + T-3d retention 22/25=88% → in-grid oracle → estimator pass) + `gates/db_facts.py::{critic_activity, fetch_reviews_full}` + `gates/_make_gate3b_nb.py` → **`notebooks/gate3b_deployable.ipynb`** + `tests/test_gate3b.py` (suite 651 → **662**). Caches: `_cache/gate3b_{cells,grid,a1_cache,activity,pools,readiness,meta}.csv`.
2. **VERDICT (pre-registered rule): the shipped 0.2.0 stack DOES NOT CLEAR** — pooled (35 unique mkts/13 movies, T-1d excluded, 2000 shared resamples seed 7) Brier diff **+0.0015 [−0.0564, +0.0709]**, taker PnL **+2.4¢ [−12.6, +16.8]**. Decomposition (the prize): **λ fine — 93% of cells inside m ∈ [0.55, 3.0]**; **p_fresh fails — 33% in δ-band, 42% in the kill zone δ̂ > +0.05**, horizon sign-flip (T-3d mean +0.078/median +0.123 over-fresh; T-1d −0.063/−0.136 under); 2×2: p_fresh-in-band trades **+20.8¢ (75% win)** vs out-of-band **−10.7¢**. **In-grid oracle on the same 35 cells +31.4¢ [+22.0, +40.4]** → paired gap +29.0¢, capture **8%**; lagged≈pure; coverage 91.9% (3 `skip:features`, 0 `skip:lambda`, 0 `trimmed`). **Conservative −0.03 shade (post-verdict bench re-score, fenced): +11.8¢ [−0.3, +26.0]** — one line recovers ~⅓ of the ceiling, CI-lo a hair under 0. Full result: `plans/plan_gate3b.md` § "GATE 3b RESULT"; memory `project_gate3b_result`. Post-build adversarial Explore review: faithful, no blocking findings. Dev-catch worth knowing: the readiness check originally used `movie_coverage` ALL-TIME counts (post-close rows poisoned every label) — the pre-registered hard-assert on the expected fail set caught it before anything locked; fixed to `observed_state`.
3. **`brainstorm/brainstorm_p_fresh_regression.md` WRITTEN (gitignored) — STOPPED for operator sign-off.** Candidate ladder C0 shade → C1 learned blend w(total, t_rem) → C2 small GLM (obs_rate, log1p_total, prior, t_rem, pool-consumed) → C3 + generosity-anchored subjective channel (§2.3 shrinkage folds in; subjective_score ~30% NaN → locked-row ablation); pre-registered decision rule: adopt iff bench PnL CI-lo > 0 AND point ≥ the shade's +11.8¢; eval = re-run the estimator pass on the LOCKED `gate3b_cells.csv` (the shade rows demonstrate the pattern); fits never see gate-cell outcomes (temporal LOO); ladder walked ONCE. Four open questions for the operator at the doc's end.

**NEXT TASK (operator-gated):**
1. **Get sign-off on `brainstorm/brainstorm_p_fresh_regression.md`** (answer its 4 open questions: ladder/stop rule; decision-rule strictness; training-row density; is C1-standalone shippable) → then PROTOCOL plan doc → build per the locked eval protocol. The bench makes each candidate cheap: p̂ swap + re-score, no grid/oracle rebuild.
2. **Operator-side reminders (not agent tasks):** BACKLOG §1.9 scraper-repo backfill (ANI/POW + BAC/INT; acceptance = `validate_recorded` consistency table green → movies auto-re-admit to future gate re-runs); `kalshi-trading/src/series/KXRT/db.py:89` case fix; **push** (`git push origin main`).
3. Standing: recorder cadence via ritual step 4 (seeded + fresh as of 2026-06-10).

**Read at session start:** CLAUDE.md "Current Conventions" + status paragraphs (Gate-3b paragraph is new); `plans/plan_gate3b.md` — the "GATE 3b RESULT" section + "Scope resolution"; `brainstorm/brainstorm_p_fresh_regression.md` IN FULL (the sign-off object); memories `project_gate3b_result`, `project_gate2_result`, `reference_db_access`, `reference_notebook_execution`.

**Conventions established this session:** the locked `gate3b_cells.csv` is the standing p_fresh/λ re-scoring bench (estimator-pass-only re-runs; nothing upstream rebuilds); the a1 cache carries TWO views (`estimated_timestamp` = noon-shifted estimator view, `est_raw` = oracle view — "each stack self-consistent"); readiness self-labels ALWAYS via `observed_state` (est ≤ close), never `movie_coverage`'s all-time counts; spread quantized to integer cents in the 3b grid (disclosed deviation from Gate-2's raw-float ≤).

**Sanity-check on arrival:** `uv sync && .venv/bin/python -m pytest tests/ -q` (expect **662** green); `.venv/bin/python -m gates.recorder --check`; `git log --oneline -3`; `.venv/bin/python -m gates.validate_recorded` (still exactly 3 coverage-class MISMATCHes — ANI/BAC/INT — until §1.9 lands); `gates/_cache/gate3b_cells.csv` exists (60 rows; rebuild = `python -m gates.build_gate3b`, needs DB sandbox-off, ~3min). Kalshi reads sandboxed OK; Neon + nbconvert need `dangerouslyDisableSandbox`.

**Push status:** this session's close commit local on `main`; push per operator.

---

## Prior handoff (2026-06-10 AM) — SUPERSEDED 2026-06-10 PM (Gate 3b ran; verdict + bench landed)

**Shipped this session (4 commits, 9339c64 → 8c6b886):**
1. `9339c64` — **BACKLOG §1.7 recorder BUILT** (infra riff operator-signed first: `brainstorm/brainstorm_recorder_infra.md` — local-first, zero-secret Kalshi core, GHA = deferred phase-2): `gates/recorder.py` (incremental/idempotent → COMMITTED `gates/recorded/`; candles-before-ledger crash safety; per-market close epochs; Int64 columns; DB join = aggregates only with `db_joined` top-up; coverage watch; `n_aged_out` retention canary; `--check` staleness wired into ritual step 4 below) + `gates/slug_map.py` + `gates/validate_recorded.py` (rerunnable Phase-3 cross-check). **Seeded @ as_of 649484: 338 markets / 19 events / 1,210,586 candle rows (~9MB gz)** incl. 3 movies settled since the 06-07 cache; adversarially reviewed pre-commit (dup-ticker wedge / close-drift / Int64 drift fixed + regression-tested). Recon byproducts: **BACKLOG §1.8** (kalshi-trading git-tracks 24MB of 2024-26 KXRT website price histories — no bid/ask, calibration-class backfill) + coverage-watch's first reading (8 pre-embargo open events — informational per `b4e1b19`).
2. `ad8260a` — **DATA-INTEGRITY INCIDENT** (found by the recorder's new settlement-consistency check: self-label must land in the score interval implied by the event's own strike results): **rows scraped ≥ ~2026-06-02 carry UPPERCASE sentiment** (raw table preserved per operator; `gates/db_facts.py`+`gates/oracle.py`+`p_fresh.py` now case-insensitive — CLAUDE.md schema note) + coverage-thin movies (DB missing reviews RT counted). Recorder auto-rejoin healed 77 ledger rows (MAS/POW/SCA/backrooms were case-poisoned — scary_movie read 0/99 fresh vs true ~24). `notebooks/gate2_integrity_recheck.ipynb`: **ZERO sentiment-case drift in all 134 original Gate-2 cells** (no uppercase pre-close row touched a cell movie); Gate 1/arena not re-run by recorded reasoning (plan addendum).
3. `8c6b886` — **GATE 2 + 3a REVISED CANONICAL (operator directive: readiness criterion promoted into the cell definition; ex-`animal_farm_2025`, whose DB history is provably short):** re-executed `gate2_oracle.ipynb` → **T-3d Brier diff +0.0883 [+0.0367,+0.1377] / PnL +27.1¢ [+16.5,+36.3] (86% win); pooled +0.1247 [+0.0655,+0.1954] / +32.1¢ [+22.4,+41.7] (33 trades/12 movies, 91% win)**; both sides win; lagged≈pure. Re-swept `gate3_tolerance.ipynb`: λ band **m ∈ [0.55,3.0]** at δ=0 (±170% noise 48/50); p_fresh PnL band **δ ∈ [−0.10, 0]** at m=1, δ=+0.05 kills the CI (random ±0.05 → 34/50) → **p_fresh priority inversion STANDS, asymmetry unchanged, conservative-shade option strengthened**. 2026-06-09 originals kept as labeled history in the plan. Suite **628 → 651**.

**Operator decisions (load-bearing):** raw DB rows are NEVER normalized in place (handle case at processing, always `lower()`); the data-readiness criterion is canonical (`data_not_ready` ≠ oracle-dirty); **power_ballad is data_not_ready by operator call** (label matches settlement post-fix but history starts 2026-05-01 with no deep backfill — suspected months-earlier accumulation) → **Gate-3b cohort = 17 movies** (15 original + MAS/SCA); p_fresh scope resolution: 3b runs FIRST as the bench, the regression-model riff (learned blend + t_rem + generosity-anchored subjective channel) comes AFTER its measured δ̂; recorder cadence = the session ritual (step 4).

**NEXT TASK (operator-confirmed menu):**
1. **GATE 3b BUILD** — execute `plans/plan_gate3b.md` (READ IN FULL; twice adversarially reviewed, every pre-registration locked). Lock chain enforced in the driver: book-only LOCF measurement at the midnight-ET snaps → `snap_density` guard + ≥8 floor per snap (data-ready 17-movie cohort) → in-grid oracle → A1-pool review cache (sentiment lowercased at ingest; `subjective_score` column included) + full-universe `activity_lookup` pull → estimator pass (`skip:features`/`skip:lambda` reason codes; 15d gap-cap as `trimmed`) → `gates/_make_gate3b_nb.py` → `notebooks/gate3b_deployable.ipynb` → **deployable-stack verdict vs the revised +32.1¢ ceiling** + the per-cell δ̂/m̂ audit (band axes per the plan). Gate-2 regression anchor runs on the untouched 648979 caches (`gate2_cells_20260609_preexclusion.csv` snapshot also in `_cache/`).
2. **POST-BUILD adversarial Explore review (fable 5) — operator-slated:** gap-check the build + notebook against the plan BEFORE finalizing docs/verdict; verify quoted numbers/lines against source per `feedback_agent_review_skepticism`.
3. **If room:** write `brainstorm/brainstorm_p_fresh_regression.md` from the audit's measured per-cell δ̂ (design captured in the plan's "Scope resolution" section incl. the conservative-shade quantification) → STOP for operator sign-off.
4. **Operator-side reminders (not agent tasks):** BACKLOG §1.9 backfill in the scraper repo (acceptance = `validate_recorded`'s consistency table goes green; ANI/POW then re-admit to future gate re-runs automatically); `kalshi-trading/src/series/KXRT/db.py:89` one-line case fix; push this session's commits if not already.

**Read at session start:** CLAUDE.md "Current Conventions" + status paragraphs (incl. the 2026-06-10 integrity paragraph + schema sentiment note); `plans/plan_gate3b.md` IN FULL; `plans/plan_gate_1_2_calibration.md` — the two REVISED result sections + the integrity addendum; memories `project_gate2_result` (revised), `reference_db_access` (sentiment-case gotcha), `reference_kalshi_access`, `reference_notebook_execution`.

**Conventions established this session:** settlement-consistency check = the standing data-integrity detector (recorder warns + auto-rejoins; `validate_recorded` prints the table); sentiment comparisons ALWAYS case-insensitive; one pin per build, never mixed (anchors pin separately on untouched caches); per-unit adversarial Explore reviews (pre-build on plans, post-build on implementations) with specifics verified against source — this session they caught a wedge-class recorder bug, two blocking plan holes, AND (via the recorder's own check) the DB incident; `gates/recorded/` is the committed system of record (`_cache/` stays the rebuildable working set).

**Sanity-check on arrival:** `uv sync && .venv/bin/python -m pytest tests/ -q` (expect **651** green); `.venv/bin/python -m gates.recorder --check` (exit 0 if <10d); `git log --oneline -5` (HEAD ≥ `8c6b886`); `.venv/bin/python -m gates.validate_recorded` (expect exactly 3 consistency MISMATCHes — ANI/BAC/INT coverage-class — until §1.9 backfill lands; POW reads consistent but stays data_not_ready by operator call); `gates/_cache/gate2_cells.csv` is the ex-ANI version (124 rows; pre-exclusion snapshot beside it). Kalshi reads work sandboxed; Neon needs `dangerouslyDisableSandbox`; nbconvert needs sandbox-off.

**Push status:** 4 session commits + the close-session commit local on `main`; push per operator (`git push origin main`).

---

## Prior handoff (2026-06-09 PM) — SUPERSEDED 2026-06-10 (recorder shipped; integrity incident resolved; Gate 2/3a revised)

**Shipped this session (one session, two phases):** (1) the **arena map** (see the superseded AM handoff below — arena = T-2d..T-5d, LOCF book reconstruction validated) and (2) **GATE 2, end-to-end**: pre-registered dense-cohort STOP-gate (`gates/build_density.py` + `notebooks/gate2_density.ipynb` — T-2d/3d/4d runnable at 13/13/11 oracle-clean movies, T-5d underpowered; zero 's'-rows; power sim honest-fixed), the oracle module (`gates/oracle.py`: m/h at est+1min, d crowd-forward to next UTC midnight, pure vs scrape-lagged boundaries) test-validated via `tests/test_oracle.py` + `tests/test_edge_battery.py` (500-case grid vs an exact-integer reference; **suite 98 → 628**), pinned review cache (`gates/build_reviews_cache.py`, as_of_id=648979), and the Gate-2 run (`gates/_make_gate2_nb.py` → `notebooks/gate2_oracle.ipynb`).

**GATE-2 RESULT — STRONG PASS (directional; read `project_gate2_result` + the plan's "GATE 2 RESULT" section IN FULL):** oracle λ/p_fresh → `compute_edge` beats the state-at-snap book on **Brier AND spread-crossing taker-fee PnL**: T-3d +0.0775 Brier diff [+0.0237, +0.1271] / +24.1¢ [+11.9, +34.4] (83% win, 23 trades/13 movies); T-4d clears both; pooled +0.0966 / +27.5¢; robust ex-billie; **both trade sides win** (22 NO +20.3¢ / 14 YES +38.8¢); **lagged ≈ pure** (scrape cadence not binding); encompassing LOMO Δlogloss +0.25 at 3d/4d (mirror of Gate-1b's ≈0) → **the market prices the current state but not the flow**. Adversarially verified by independent recompute. Caveats: ≤13 movies/snap → directional; the oracle is the architecture CEILING, not a deployable edge.

**GATE 3a DONE same session (`notebooks/gate3_tolerance.ipynb`; plan § "GATE 3a RESULT"):** λ-error tolerance is HUGE (PnL CI>0 for m ∈ [0.55, 3.0]; ±170% random λ noise survives 44/50), **p_fresh tolerance is TIGHT and binding** (PnL band δ ∈ [−0.05, 0]; random ±0.05 clears only 26/50; over-estimating worse than under-). Shipped-0.2.0 proxies: Ridge λ error comfortably inside; `estimate_p_fresh` (~±0.03–0.05) right AT the band edge. **Strategic inversion: the edge hinges on p_fresh, which nobody has touched since 0.1.x — not λ, where all the effort went.**

**NEXT TASK (operator-confirmed order at 2026-06-09 close): infra riff → RECORDER → GATE 3b.**
0. **PRIORITY #0 (operator-added): riff on recorder infra BEFORE building** — provisioning/config (cloud vs VM), secrets, monitoring/observability; lightweight + secure, no ocean-boiling. Start with an **Explore agent (fable 5)** recon of **`~/Desktop/rotten-tomatoes-analysis`** as the PRIMARY analog (operator's call, agent concurs: it runs the review scraper as a **Cloud Run Job on a Scheduler** — a periodic batch pull, exactly the recorder's shape; map how it provisions/deploys, injects `DATABASE_URL`, logs, and alerts) + a lighter skim of `~/Desktop/kalshi-trading` (the VM/scripts deployment direction + any deploy helpers; its sub-minute trading loop is the WRONG shape for this, but its conventions may still bind). Two scope-shrinking facts: the recorder needs **NO Kalshi credentials** (all endpoints public — `reference_kalshi_access`) and its only secret is the **read-only** Neon URL (`agent_neon_read_only` — `reference_db_access`; note BACKLOG §1.6's grant-test intersects if a cloud role is provisioned). Output: a short `brainstorm/brainstorm_recorder_infra.md` with a recommendation, operator sign-off, THEN build.
1. **BACKLOG §1.7 weekly settled-market recorder** (after #0): the API retention window rolls and every un-snapshotted settling week is cohort permanently lost; Gate-3b also wants fresh out-of-sample movies. Small build reusing `gates/kalshi_data.py` + `gates/db_facts.py` + the `build_cohort.py` shape: snapshot newly-settled KXRT markets (metadata + full 1-min candles + self-labeled 10am score + pinned review rows), APPEND-don't-overwrite into `gates/_cache/` (or a versioned sibling), idempotent on re-run; **locally-runnable first, the cloud trigger as a thin wrapper** per #0's design.
2. **Gate 3b (same session if room):** build the A1-pool review cache (the 20 most recent resolved movies before each target close — new `db_facts`/driver pull, as_of-pinned) + align the **ET-midnight snap convention** (CLAUDE.md "Current Conventions") vs the gate cells' close−24N snaps (compare only P(Yes), per the plan's windowing note). Run `extract_lambda_features` → `estimate_lambda` (shipped artifact) + `estimate_p_fresh` per cell → same Brier/PnL machinery vs the state-at-snap book → **the deployable-stack verdict** (the oracle ceiling is +27.5¢ pooled; how much survives real inputs?).
3. **p_fresh audit first-class** (§1.4, now priority): per-cell `estimate_p_fresh` error on these 16 movies vs the oracle p_fresh — distribution, bias sign (the band is asymmetric), and whether §2.2 time-varying / §2.3 hierarchical variants close the gap. (Natural companion to Gate 3b — its per-cell p_fresh predictions feed both.)
4. Park: form-diagnostic/PIT (Gate-2 pass made it moot); contested-Yes-tilt hint (logged exploratory).

**Conventions (all still apply):** `db_facts` as_of_id pinning (648979 this session); one-obs-per-market + staleness + spread stratification; notebooks = citable numbers (`/audit-numbers`); nbconvert needs `dangerouslyDisableSandbox` (kernel sockets); Kalshi API reachable sandboxed, **Neon DB needs sandbox-off** (DNS blocked in-sandbox — see session note); axis-language discipline (oracle-conditioned = PRIZE, observable-conditioned = inefficiency).

**Sanity-check on arrival:** `uv sync && .venv/bin/python -m pytest tests/ -q` (expect **628** green); `git log --oneline -3`; confirm `gates/_cache/{cohort_markets,candles,gate1b_input,arena_spans,density,reviews_cohort,gate2_cells}.csv` exist (gate3 band artifacts: `gate3_band.png` + `notebooks/gate3_tolerance.ipynb`).

**Push status:** committed at this session close; push per operator (`/ship` or `git push origin main`).

---

## Prior handoff (2026-06-09 AM) — SUPERSEDED same day (arena map done; Gate 2 RAN and PASSED)

**Shipped this session:** the **tradeable-edge arena map** (the Gate-1 handoff's binding-constraint gate). New: `gates/probe_candle_open.py` (Kalshi candle bid/ask open+close probe), `gates/_make_arena_nb.py` (codegen), `notebooks/arena_map.ipynb` (executed; the citable numbers), cache artifacts `gates/_cache/{candle_open_probe.csv, arena_spans.csv, arena_occupancy.png, arena_death_vs_formation.png}` (gitignored). Docs: "Arena map result (2026-06-09)" section in `plans/plan_gate_1_2_calibration.md` (gitignored), CLAUDE.md Gate-status update, BACKLOG 1.5 update, memory `project_arena_map` (+ `reference_kalshi_access` candle-semantics fix).

**Arena result (read `project_arena_map` + the plan section IN FULL):** the arena EXISTS and is EARLY. Kalshi candles are activity-gated but the book persists through silent gaps (P=1.00000, n=21,078 gap-pairs) → LOCF book-state reconstruction is valid. Contested∧≤10¢-spread occupancy ramps 0% (last hour) → 22% (5-7d) → 39% (>14d); at **T-3d all 16/16 movies** have a contested tight (mostly fresh) book with median **28% of reviews still to come**; T-5d: 14 movies / 65% to come but 5 of the 16 movies pre-embargo (0 obs reviews); ≤12h ~dead (confirms Gate 1). Movie books die at median 12.3h pre-close with only 0.7% of reviews left → **the tradeable-edge window is T-2d..T-5d, centered T-3d**. Contested under-pricing hint persists honestly measured (+0.08 at 2d → +0.24 at 1d, realized−mid) but all staleness-honest CIs straddle 0 — hint, not result.

**NEXT TASK — GATE 2 (unblocked, now scoped to the arena):**
1. **Dense-cohort guard first (STOP-gate):** via `db_facts` (read-only, `as_of_id`-pinned), count the 16 cohort movies live-tracked-through-snap (m/h timestamps near T-2d..T-5d snaps, dense through close). Below the (pre-registered, power-calc'd) floor → "underpowered/inconclusive", NOT fail.
2. **Oracle math + tests** per the plan ("Math under test"): oracle λ = realized remaining count / t_rem (+1min lag, two-oracle framing), oracle p_fresh; `compute_edge` battery vs an independent reference implementation (exact-integer boundary: Yes iff `200·fresh ≥ (2X+1)·total`).
3. **Gate 2 run:** oracle → `compute_edge` → P(Yes) at T-2d/T-3d/T-5d on the ct cells in `gates/_cache/arena_spans.csv`, benchmarked against the **state-at-snap book with its staleness** (entry crosses the spread: Yes at ask, No at 1−bid) — NOT an idealized mid. Brier + PnL 2×2, cluster-boot by movie; asymmetric fork (only a clear fail + clean form-diagnostic abandons). **Include the 2026-06-09 stratification layer** (plan section "Gate-2 stratification layer", operator-confirmed): prize-sensitivity test (prize := |P_oracle − P_frozen|, P_frozen = `compute_edge(..., lambda_rate=0)`; error-vs-prize regression), encompassing LOMO logistic `y ~ logit(p_mkt) + logit(P_oracle)`, oracle-λ/p_fresh tercile diagnostics (exploratory). Keep the axis-language discipline: oracle-conditioned gap = forecaster's PRIZE, observable-conditioned gap = tradeable inefficiency — never conflate.
4. Opportunistic: **BACKLOG 1.7 recorder is getting urgent** — the API retention window rolls; every un-snapshotted settling week is cohort permanently lost (the contested-hint needs n).

**Conventions this session (all still apply):** `db_facts` discipline; one-obs-per-market + staleness columns + spread stratification for any market-level claim; notebooks are the citable-number source (`/audit-numbers`); notebook execution needs `dangerouslyDisableSandbox` (kernel sockets — `reference_notebook_execution`); Kalshi/Neon hosts are sandbox-allowlisted and verified live (Kalshi probe ran sandboxed this session).

**Sanity-check on arrival:** `uv sync && .venv/bin/python -m pytest tests/ -q` (98 green); `git log --oneline -3`; confirm `gates/_cache/{cohort_markets,candles,gate1b_input,arena_spans}.csv` exist (cohort/candles rebuild: `python -m gates.build_cohort`, ~23 min; arena_spans: re-execute `notebooks/arena_map.ipynb`).

**Push status:** committed locally at this session close; `git push origin main` left to the operator (agent push to main was permission-blocked).

---

## Prior handoff (2026-06-07) — SUPERSEDED 2026-06-09 (arena map completed; Gate 2 unblocked)

**Shipped this session (one close-session commit):** new `gates/` package — `kalshi_data.py` (public Kalshi/KXRT fetcher, no auth), `db_facts.py` (read-only `as_of_id`-pinned reviews queries), `build_cohort.py` + `build_snap_state.py` (drivers → `gates/_cache/`, gitignored), `_make_gate1*.py` (nbformat notebook builders); `notebooks/gate1_calibration.ipynb` + `gate1b_incremental_info.ipynb`; `BACKLOG.md` §1.6 (DB write-perm security test) + §1.7 (weekly settled-market recorder); `.gitignore` += `gates/_cache/`; CLAUDE.md Gate-1 status. Gitignored/out-of-repo: heavy `plans/plan_gate_1_2_calibration.md` edits + new/updated memories.

**Gate-1 result (directional, 16-movie settled KXRT cohort) — read `project_gate1_findings` IN FULL:** the market PRICES the observed review state (no current-state edge — Gate-1b LOMO logistic: review signal adds ~0 OOS, CI straddles 0, early-snap NEGATIVE); it's calibrated where it quotes but contested-region skill is only ~0.30 (the 0.92 headline was extreme-inflated); and it's STALE/THIN — median last real two-sided quote ~4d before close; the 28 markets quoted within 6h of close are all decided (0 contested). **Tradeability, not forecasting, is the binding constraint.** All numbers were adversarially reviewed + independently recomputed.

**NEXT TASK (operator-confirmed): MAP THE TRADEABLE-EDGE ARENA.** Over time-to-close, characterize the (market × minute) cells that are BOTH contested (`0.2 < mid < 0.8`) AND have a live two-sided quote (`mid` not NaN): does a live-contested window exist (likely *earlier*, while scores still form), and how large (movies, market-minutes)? This is the cheap binding-constraint gate before the Gate-2 oracle — if ~empty even early → edge isn't capturable on this cohort (near-decisive abandon signal); if a window exists → that's where Gate 2 runs, benchmarked against the **actual at/before-T quote with its staleness**, not an idealized price. Work from cached `gates/_cache/{cohort_markets,candles,gate1b_input}.csv` (rebuild via `python -m gates.build_cohort` only if absent, ~23 min). Fold in the review's methodology fixes (read once per market, not per snap; staleness column; liquidity stratification). Conditional follow-on: chase the faint contested-at-1d under-pricing hint (mean mid 0.56 → realized 0.77, n=17 — de-`billie_eilish`, one-obs-per-market).

**Read in full at start:** `plans/plan_gate_1_2_calibration.md` (esp. "Verification + review amendments (2026-06-07)"); memories `project_gate1_findings`, `project_gate_calibration_design`, `reference_kalshi_access`, `reference_db_access`, `reference_notebook_execution`.

**Conventions this session:** `db_facts` read-only `as_of_id`-pinned queries (DB analog of `/audit-numbers`); two-oracle framing (pure publication-time = architecture ceiling; scrape-lagged = current-pipeline value); Gate-1 reads are one-obs-per-market + conditional-on-tradeable + contested-region-stratified; asymmetric Gate-2 fork (only a clear fail abandons; "inconclusive" never does); **notebook execution needs `dangerouslyDisableSandbox`** (Jupyter kernel binds local sockets — see `reference_notebook_execution`).

**Sanity-check on arrival:** `uv sync && .venv/bin/python -m pytest tests/ -q` (expect 98 green); `git log --oneline -3`; confirm `gates/_cache/cohort_markets.csv` exists (else rebuild). The Kalshi/Neon hosts are allowlisted in `.claude/settings.local.json` and should be **live this fresh session** → try DB/Kalshi reads sandboxed first; fall back to `dangerouslyDisableSandbox` if blocked.

**Push status:** committed + pushed to `origin/main` at this session close.

---

## Prior handoff (2026-06-04) — SUPERSEDED 2026-06-07 (Gate Phase 0 + Gate 1 completed)

**Shipped this session:** the prune (`bf8892e` — deleted `notebooks/`, archived 4 KDE-era findings) + docs refocus (`a024923` — CLAUDE.md / PROTOCOL `/audit-*` line / this PROMPTS template). Persisted but gitignored/out-of-repo: `plans/plan_gate_1_2_calibration.md` and the gate-design + `db_facts` memories.

**Read first:** CLAUDE.md "Current Conventions" + "Project Overview"; `plans/plan_gate_1_2_calibration.md` IN FULL; memories `project_gate_calibration_design`, `feedback_db_facts_verification`, `project_minute_level_movies`.

**Task — start Gate Phase 0** (the plan is the spec):
1. Verify Kalshi serves historical **minute price** (candlesticks) for the *closed* RT cohort — kalshi-trading side / operator's API access. Hard prerequisite.
2. First **`db_facts`** read-only query functions (pinned by `WHERE id <= N`): settle (a) scraper-timing — does each movie have reviews through its 10am close; (b) the **dense-near-close movie count** (Gate-2 STOP-gate `n`); (c) any `timestamp_confidence='s'` rows.
3. **D1 coverage notebook** (per-movie m/h-vs-day fraction near close) + pull **Kalshi-`result` labels**.
4. **Gate 1** (market calibration + incremental-info, full 144 cohort) — runs regardless of dense-n.

**DECIDE FIRST (operator flagged 2026-06-04) — mixed-granularity snapping:** with `d`/`h`/`m` reviews interleaved, how do we place the coarser ones? (A) each review at its own honest bound (`d`→midnight, `h`→top-of-hour, `m`→minute); (B) the d/h/m *ablation* = uniformly coarsen ALL reviews to the tested level (measures granularity's marginal value); (C) interpolate a coarse review's position from neighboring fine reviews (refinement of A). Distinguish the **real best-data calibration (A/C, mixed)** from the **ablation (B, uniformly coarsened)**. Likely run A vs C, keep B as the value-of-granularity measurement. Resolve before building the oracle.

**Locked (don't re-litigate):** oracle = realized λ/p_fresh (best inputs; dispersion = genuine real-time uncertainty → real-time-forecaster ceiling, NOT perfect foresight); +1 min look-ahead lag; Kalshi `result` as label; orderbook-mid (not last-trade) as "the market"; pass = Brier + PnL 2×2. See `project_gate_calibration_design`.

**Sanity-check on arrival:** `uv sync && .venv/bin/python -m pytest tests/ -q` (expect 98 green); `git log --oneline -3`.

**Push status:** all work pushed to `origin/main` at close.

---

### Prompt 1: Kalshi-independent lambda validation (Backlog 1.1)

```
Read CLAUDE.md and BACKLOG.md 1.1. Design and implement a validation notebook that tests estimate_lambda() accuracy using minute-level review data, without any reference to Kalshi market resolution.

The idea: for movies with minute-level timestamps, we know the exact review count at any point in time. At snapshot time T, we can call estimate_lambda() to predict remaining reviews, then compare to the actual count that arrived between T and close. This is a pure model accuracy test.

Currently only the_drama and the_super_mario_galaxy_movie have minute-level data. Check if that's enough for meaningful validation, or if we need to wait for more data. If the sample is too small, document what we'd need and defer.

Work in notebooks/lambda_validation.ipynb. Write findings to findings/lambda_validation.md if results are meaningful.
```

### Prompt 2: Close-day lambda bias patch (Backlog 1.3)

```
Read CLAUDE.md, BACKLOG.md 1.3, and brainstorm/brainstorm_close_day_lambda_bias.md. The KDE model drops reviews on the close day because of a UTC midnight vs actual close time mismatch. The partial fix (full UTC datetimes in movies_index.csv) was applied but 98% of reviews still have day-level timestamps.

Evaluate the patch approaches in the brainstorm doc and implement the best one. The fix should be in rotten_tomatoes_forecasting/critic_model.py (the build_critic_profiles or estimate_lambda path). Validate by checking whether lambda estimates change for movies near close.
```

### Prompt 3: p_fresh calibration audit (Backlog 1.2)

```
Read CLAUDE.md and BACKLOG.md 1.2. Break down the existing p_fresh calibration (from findings/critic_kde_model_validation.md) by movie characteristics. Are there systematic biases for certain types of movies?

Work in notebooks/p_fresh_calibration.ipynb.
```

### Prompt 4: Pre-ship tuning for validated deployment stack (COMPLETED 2026-04-18)

```
Read, in order: CLAUDE.md, BACKLOG.md (especially §1.5-§1.7), findings/stratified_training_investigation.md (especially TL;DR + §10-15), and brainstorm/brainstorm_pre_ship_tuning.md. The brainstorm has been reviewed by an independent agent and incorporates those fixes.

Context: A long investigation (summarized in the findings doc) validated a deployment-recommended stack: `combined_score (α=0.5) + bandwidth_cap (0.7d) + piecewise (F=1.0)`. Before integrating this into library code, a small tuning pass is warranted to re-validate or tighten a few parameters. The Path B (learned multi-feature model) alternative has been scoped and deferred — see plans/plan_learned_similarity_model.md.

Your task is to execute the pre-ship tuning pass as scoped in the brainstorm:

1. Start a successor notebook: `notebooks/pre_ship_tuning.ipynb`. Cell 1 imports helpers from `notebooks/stratified_training_validation.ipynb` (predict_window, combined_score_selector, gap_overlap_ranked_selector, build_kde_lambda_model_capped, passes_skip_rules_for_snap, critics_in_window, jaccard, actual_remaining, etc.). Do not rebuild these. Consider importing via `%run` or factoring into a small `.py` module under `notebooks/` if import hygiene matters.

2. Cell 2: compute the anchor baseline — current deployment stack MAE at T-3d, T-5d, T-7d full window (snap to midnight UTC of close). This is the comparison point for everything below. Use F=0.7 as the ship-time piecewise value (not F=1.0 — see brainstorm's G1 discussion).

3. Cell 3: gap-distribution diagnostic. For each of 143 targets, count candidates with |gap_diff| ≤ 0.5d, 1d, 2d, 5d. Print a summary table stratified by gap quantile (Q1-Q4). This decides whether tight-σ_gap is feasible.

4. Execute T1 (σ_gap sweep ∈ {2, 4, 8, 16, ∞}) per brainstorm. Use bootstrap CI (1000 resamples, paired per target) for decision rule: ≥3% T-3d MAE improvement with CI95 lower > 0 → replace σ_gap=8. Otherwise keep current.

5. Execute T2 (n_training sweep ∈ {5, 10, 15, 20, 25, 30, 50}) using T1's σ_gap winner. Same decision rule.

6. Report results. If neither T1 nor T2 produce a winner, the stack ships unchanged — but the re-validation itself is valuable.

7. Update `findings/stratified_training_investigation.md` with a new section summarizing pre-ship tuning. Update `BACKLOG.md` §1.5 if any parameter value changes. Update the memory pointer at `/Users/jakelehner/.claude/projects/-Users-jakelehner-Desktop-rotten-tomatoes-forecasting/memory/project_stratified_training_results.md` to reflect final ship values.

After this pass, the next conversation will cover library integration (a separate plan doc covering the four interventions from BACKLOG §1.5: bandwidth ceiling parameter, selector function, piecewise helper, snapshot-state helper).

Items explicitly OUT OF SCOPE for this pass (documented in brainstorm's "Out of scope" section): α tuning (at noise floor, keep 0.5), F varying by movie type (Path B), target-scope narrowing (strategy concern), new features, learned-weights model, retiring piecewise, library integration (next pass).

Gated on fresh reviews.csv from Neon (NOT blocking for this pass): G1 (F re-estimation), G2a (re-pick frequency curve on h/m targets). These can run as a follow-up after Jake pulls fresh data.

Follow PROTOCOL.md: the successor notebook is an analysis notebook tier, so a brief intent section at top suffices — no separate plan doc needed unless T1 or T2 reveals something that changes the approach. Do use a TodoWrite-style task breakdown for tracking.

Do NOT add new features during this pass. The user's stance: no more feature hunting, we're at the local optimum given current feature set + cohort size. If something looks like Path B territory, defer it.

Session-level goal: complete T1+T2, ship-decide, and hand off to the integration conversation.
```

### Prompt 5: Library integration for validated stack (SUPERSEDED by Prompt 8)

> **2026-04-19:** the Ridge investigation (`findings/ridge_lambda_investigation.md`) changed the ship candidate. Use **Prompt 8** below for library integration. This Prompt 5 is kept as historical context for the original KDE-vs-Ridge dual-model question that led to the Ridge investigation.

```
Read, in order: CLAUDE.md, BACKLOG.md §1.5 and §1.8, findings/stratified_training_investigation.md §16-17, and findings/path_b_lite_investigation.md (especially TL;DR, §9, §15-17).

Context: extensive investigation converged on two viable ship candidates, each with known tradeoffs (per path_b_lite_investigation.md §15):

  Weighted-KDE stack (per-target stability, known h/m bias):
    combined_score(α=0.5, σ_gap=8) top-20 selector
    + weighted KDE build (per-data-point weights = combined_score values) [new]
    + bandwidth (floor=0.5d, ceiling=0.7d)
    + midnight+noon snap convention: snap_time = midnight UTC on close-N, day-level reviews shifted to 12:00 UTC [new]
    + phase_1 = KDE integral over (midnight_utc_dbc, snap_dbc_effective]
    + phase_2 = constant C=2
    → cohort MAE 13.26, h/m MAE 18.23, h/m me −6.78

  Ridge(α=10) regression (cohort-calibrated, per-target variance):
    Features: observed_count, first_review_dbc, target_gap, observed_rate,
              rate_last_day, rate_first_day, top_critic_frac, pub_diversity,
              pub_entropy, low_activity_frac
    → cohort MAE 8.85, h/m MAE 26.91, h/m me +0.69

Neither dominates. Ridge wins cohort + calibration; KDE wins h/m MAE with systematic under-bias. Choice depends on deployment priorities.

Known architectural ceiling: 14+ interventions tested in Path B-lite investigation couldn't break through the h/m under-prediction limit. Root cause is that features observable at T-3d don't predict late-surge behavior on long-gap targets. See findings/path_b_lite_investigation.md §13 for full ruled-out list.

Task: write a library integration plan in plans/plan_library_integration.md covering:

1. API surface. Which new symbols to export from __init__.py, which live as internal helpers:
   - `build_kde_lambda_model(bandwidth_ceiling=0.7)` — add parameter with new default.
   - `build_kde_lambda_model_weighted(profiles, scores)` — NEW public function.
   - `combined_score_selector` + `combined_score_with_scores` — NEW public functions.
   - `compute_close_day_phase2(C=2.0)` — trivial, returns C.
   - `snapshot_state` helper — promote from notebook.
   - Consider: `fit_phase1_regressor(features, targets) -> regressor` + `predict_phase1_regression(regressor, features)` for the Ridge alternative. Ships alongside KDE.
   - Convention switch: rename defaults or document? Backward compat with existing consumers matters.

2. Dual-model policy: ship BOTH KDE and Ridge? Or just KDE with Ridge as experimental?
   Pros of both: deployment flexibility, orchestrator picks per strategy.
   Cons: doubles API surface, test matrix.

3. Default-value policy for new parameters: opt-in or on-by-default?
   Migration notes for the orchestrator consumer at ~/Desktop/kalshi-trading/.

4. Test additions for each new function + integration tests.

5. CHANGELOG entry and version bump (0.1.0 → 0.2.0? Breaking-ish given snap semantics change).

After the plan is signed off, the implementation pass is a separate conversation.

BLOCKERS / open questions for Jake before writing the plan:
- Ship single model (KDE only) or dual (KDE + Ridge)?
- Is midnight+noon snap convention a hard migration or does library support both?
- Does orchestrator currently expect the old snap convention? Check kalshi-trading/src/ before writing the plan.

OUT OF SCOPE (deferred):
- Finite-pool model (brainstorm exists; architectural rework).
- Hier-Bayes with different shape family (Gamma was null).
- TMDb metadata path (declined by Jake).
- More h/m cohort data (wait).
```

### Prompt 6: Wait-and-revisit pass (future, after h/m cohort grows)

```
Read findings/path_b_lite_investigation.md §13-14. Several interventions that were ruled out may become viable as the cohort's h/m fraction grows over time.

Trigger: when ≥30% of resolved cohort movies have h/m timestamps (currently ~5%).

Re-test with fresh cohort:
1. Weighted KDE + midnight+noon on h/m-representative cohort. Does the architectural ceiling still hold?
2. Hierarchical Bayes with non-parametric partial-pooling (per-movie empirical CDFs rather than Gamma parametric).
3. Shape-similarity selector with expanded cohort coverage (more shape-similar neighbors available).
4. Finite-pool model — if cohort shifts toward h/m, per-target P(review | movie) becomes tractable from more-granular data.

Evaluation: hold out last N h/m-resolved movies (not the original 5 — those are now permanent cohort members). Decision rule: h/m MAE materially below current ship candidates.
```

Optional follow-up (NOT in this plan, but worth logging as a BACKLOG item): time-series regression as alternative or blend. See findings §9 — Ridge beats KDE on cohort MAE (9.71 vs 13.45) and is near-calibrated on h/m (me ≈ 0), but higher h/m variance. Worth exploring as a blended predictor, but not prerequisite for initial integration.
```

### Prompt 7: Continue phase-1 KDE menu mix-and-match

```
Read, in order: CLAUDE.md, brainstorm/brainstorm_phase1_kde_menu.md (the menu reference — primary doc for this work), findings/path_b_lite_investigation.md (full investigation context), findings/stratified_training_investigation.md §16-17 (pre-ship tuning + G1 piecewise audit).

Context: Jake is running a mix-and-match iteration on the phase-1 KDE pipeline. Despite 14+ ruled-out interventions documented in path_b_lite_investigation.md (concluded 2026-04-18), he wants to keep exploring combinations of the tested pipeline components — particularly those not yet tested in combination. The component menu lives at brainstorm/brainstorm_phase1_kde_menu.md, organized by pipeline stage A-K plus alternative architectures.

Current ship candidate stack (per menu): A3 + B1 + C2 + D2 + D3 + E2 + F1 + G2 + H2 + I2 + J2 + K1.
  - cohort MAE 13.26, h/m MAE 18.23, h/m mean_err −6.78
  - Midnight+noon convention adopted per Jake 2026-04-18
  - Known architectural ceiling on h/m under-prediction (oracle test confirmed)

Primary untested options (per menu):
  - **E5 (transformed base_rate):** square-root / log / cap / rank compression to reduce workhorse-critic dominance without per-target tiering (which Option C / E3 mis-classified late-surge targets)
  - **C5 (hier-Bayes with log-normal or mixture-Gamma)**
  - **C6 (non-parametric empirical-CDF partial pooling)**
  - **D5 (custom bandwidth ceiling by target type)**
  - **E6 / K3 (per-critic base_rate conditioned on target features)** — requires metadata

Reference notebooks (in notebooks/):
  - _helpers.py — factored helpers (combined_score, weighted KDE build, etc.)
  - path_b_lite_weighted_kde.ipynb — C2 weighted KDE win
  - option_c_base_rate_adjustment.ipynb — E3 Option C (Q1 win, h/m regression)
  - scaling_clamp_sweep.ipynb + scaling_diagnostic.ipynb — F2/F4 ruled out
  - shape_scale_model.ipynb — C3 ruled out
  - hier_bayes_shape.ipynb — C4 ruled out
  - time_series_regression.ipynb — R1 Ridge alternative
  - critic_magnet.ipynb — K2 falsified

Workflow for each test:
  1. Jake proposes a combination: "A3 + B1 + C2 + D3 + E5a + F1 + G2 + H2 + I2 + J2 + K1"
  2. Build a notebook that runs the combo under midnight+noon convention (G2 + H2).
  3. Evaluate on cohort CV (no h/m) + h/m subset (5 movies: the_drama, super_mario_galaxy, forbidden_fruits, they_will_kill_you, you_me_and_tuscany).
  4. Compare to ship stack baseline.
  5. Decision rule: h/m MAE improvement AND cohort MAE non-regression (>−5%).

Per PROTOCOL.md, notebook tier: brief intent section at top, no formal plan doc required for each combo. If a combo produces a meaningful result (positive or negative), add it to brainstorm_phase1_kde_menu.md under the relevant category, updating "ruled out" or "adopted" status as appropriate.

Jake's stated preference: "in the face of all logic and reason, I still think the KDE approach can be fixed fully." Engage seriously — the architectural ceiling finding is real but not proven absolute across all untested combinations. A promising untested combo could still move things. The oracle test (findings §8.2) established the limit is in the architecture, not selection — but components like E5 (base_rate transform) haven't been tested in isolation.

Do NOT push for library integration without Jake's explicit go-ahead. Prior session included a course-correction where Jake pushed back on "eager to integrate" — the current model is good enough to document but not necessarily to ship.

Start by asking Jake what combo he wants to test first, or propose one based on the menu if he asks.
```

### Prompt 8: Ridge lambda integration (replaces Prompt 5)

```
BEFORE writing any timestamp / snap / phase logic, re-read the "Current Conventions"
section at the top of CLAUDE.md. It defines the ET-midnight snap convention, 10h
phase-2 window, C=1 constant, opt-in noon-shift, JSON artifact format, and snap_days
routing. These supersede ALL older conventions in findings/, notebooks/, brainstorm/,
and the current live library code (which is being replaced).

If you grep historical docs for "snap_time" or "phase_2" or "apply_noon_shift", you'll
land on notebooks that use the older UTC-midnight convention, C=2 phase-2, or silent
noon-shift — those are intentionally-preserved historical validation artifacts with
"CONVENTION WARNING" banners at the top. The banners say "do not copy into new code."
Heed them. Current conventions in CLAUDE.md win.

Read, in order: CLAUDE.md (conventions first, then overview), plans/plan_ridge_integration.md, findings/ridge_lambda_investigation.md (TL;DR + §2 + §4 for the spec), brainstorm/brainstorm_ridge_optimization.md (for the investigation context).

Context: the 2026-04-19 Ridge investigation established ridge_t2 as the ship lambda estimator, superseding the KDE-based architecture. Plan doc is signed-off and covers: module layout, API surface, fit artifact strategy, test additions, version bump, orchestrator migration, phased implementation.

Task: execute the plan's Phases A → E (library-side implementation). That's:

  Phase A (scaffolding):   create lambda_model.py, features.py, pool.py, p_fresh.py module stubs; define LambdaRegressor + LambdaPrediction dataclasses; migrate estimate_p_fresh unchanged.
  Phase B (features):      implement pool.py (A1 pool, base_rates, top-tier) + extract_lambda_features (17 features).
  Phase C (fit/predict):   implement fit_lambda_regressor + estimate_lambda returning LambdaPrediction.
  Phase D (artifact):      fit on current cohort, pickle to _artifacts/default_regressor.pkl, implement load_default_regressor.
  Phase E (cleanup):       delete critic_model.py contents, update __init__.py exports, update CLAUDE.md / BACKLOG.md §1.5a / PARAMETERS.md, version bump + changelog.

Per PROTOCOL.md: the plan doc exists; implementation can proceed without another planning pass. Test coverage target: 80%+ on new modules.

Decision points flagged in plan §13:
  - Default artifact naming: recommend `default_regressor.pkl` (simple).
  - p_fresh consolidation with pool.py: recommend YES (avoid duplicate base_rate paths).
  - snap_dbc routing: recommend explicit snap_days arg (no interpolation).
  - Out-of-range snaps: raise (don't interpolate or fallback).
  - Version sequencing: full replace at 0.2.0 (not a 0.2→0.3 deprecation).

If Jake has different preferences on any of these decision points, confirm before implementing.

Out of scope (per plan §12): tier 3 stacking, TMDb metadata, orchestrator-side changes (covered by findings/trading_strategy_from_ridge_errors.md separately). Phase F is an orchestrator-repo effort.

Deliverable: a PR-ready working library at version 0.2.0 with all Phase A-E work complete, tests passing, docs updated.
```

### Prompt 9: Multi-anchor instantaneous-rate Ridge features (notebook only, data-gated)

```
Read, in order: CLAUDE.md (especially "Current Conventions"), BACKLOG.md §2.4, findings/ridge_lambda_investigation.md (§4.1 for current 17-feature set, §4.3 for LOO baseline).

PREREQUISITE CHECK FIRST. Before any modeling, count cohort movies with m/h-confidence pre-close timestamps spanning enough of the first-review-to-close window to compute smoothed instantaneous rates at multiple anchor points. Memory notes only `the_drama` and `super_mario_galaxy` had useful pre-close minute-level data at the 0.2.0 ship (2026-04-19). If the current count is ≤5, STOP and tell Jake this is still data-gated — do not force the experiment on insufficient data.

Context: 2026-04-20 brainstorm with Jake. The committed proposal (BACKLOG §2.4) is rate-at-anchor features for Ridge, NOT a standalone MVT-style predictor — that framing was explicitly ruled out in the brainstorm because MVT doesn't guarantee a shared t* across movies.

Task (notebook tier only — NO library changes):

1. Work in notebooks/anchor_rate_features.ipynb. Brief intent section at top per PROTOCOL.md analysis-notebook tier.

2. Report the h/m-qualified cohort subset size. Apply the stop rule above.

3. Define fractional-time anchors (e.g., f ∈ {0.2, 0.4, 0.6, 0.8} of the first-review-to-close gap). For each anchor f, compute a smoothed instantaneous review rate using a small time window. The window width is a knob — experiment and document the choice. This is the regressor version of the KDE bandwidth problem; be explicit about it.

4. Add anchor-rate features to the existing 17-feature Ridge stack. Rerun cohort LOO using fit_lambda_regressor-compatible infrastructure. Compare MAE snap-by-snap against the shipped 0.2.0 baseline (numbers in findings/ridge_lambda_investigation.md §4 and BACKLOG.md §1.1).

5. Decision rule: materially better cohort LOO MAE at early snaps (especially T-5d / T-4d) AND no regression at late snaps. Not shipping regardless of outcome — this pass is notebook-only exploration.

6. Write findings summary in the notebook (or findings/anchor_rate_features.md if results are meaningful). Update BACKLOG.md §2.4 with the outcome.

Known risks to document in the notebook:
- Smoothing window choice reintroduces a bandwidth parameter in miniature.
- Local burst/lull near an anchor biases the rate estimate. Mitigate by averaging over several nearby anchors or short windows.
- h/m cohort may not be representative of the full 144-movie cohort; caveat any generalization claim.

OUT OF SCOPE: library integration (separate backlog item if this wins), standalone MVT-style predictor (ruled out in brainstorm), shipping to any 0.2.x.

Deliverable to Jake: sample-size check result, MAE comparison table if the experiment ran, short findings blurb.
```
