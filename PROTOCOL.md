# Build Protocol

Every non-trivial piece of work follows this sequence. Do not skip steps or reorder them.

## Relationship to brainstorming

Brainstorms (`brainstorm/`) and plans (`plans/`) serve different purposes:

- **Brainstorm**: Open-ended thinking. "What if we did X?" No commitment to build. Ideas live and die here cheaply. Part of the project's observe-and-iterate workflow (see CLAUDE.md, Development Philosophy steps 1-3).
- **Plan**: Commitment to build. "We're doing X. Here's exactly how, what it assumes, and what it produces." A plan exists because a brainstorm (or a backlog item) survived the gut-check phase.

A brainstorm is a *prerequisite* to a plan, not part of this protocol. This protocol starts when the decision to build has been made.

---

## Phase 1: Plan

1. Write a plan doc in `plans/` (e.g., `plans/populate_movies_index.md`).
   - **Context**: Why this exists. Link to the brainstorm or backlog item that spawned it.
   - **Data inputs and outputs**: What files/tables go in, what comes out, what format.
   - **Implementation steps**: Concrete steps with enough detail to code from.
   - **Assumptions**: Every non-obvious assumption, with justification. These become the checklist for Phase 3.
   - **Sanity checks**: What assertions or spot-checks should the code include? (Row counts, range checks, known-value comparisons.)
   - **Scope boundary**: What this does NOT cover.
2. Present the plan to the user for alignment before writing any code.

## Phase 2: Implement

1. Write the code following the plan.
2. Include inline sanity checks from the plan (assertions, row counts, range checks, warning prints for edge cases).
3. Run the code and confirm it completes without errors or unexpected warnings.

## Phase 3: Validate

1. Walk through the assumptions list from the plan and check each against reality:
   - Did any warnings fire? What do they mean?
   - Spot-check 3-5 rows against manually verifiable data.
   - Did any edge cases surface that the plan didn't anticipate?
2. Fix any gaps discovered during validation.
3. Update docs as needed: CLAUDE.md, brainstorm docs, backlog, and the plan doc itself if the implementation diverged.

---

## Scaling rigor to impact

One protocol, but the depth scales with how much downstream work depends on the output:

| Work type | Plan depth | Sanity checks | Validation |
|---|---|---|---|
| Data pipeline script (writes CSVs others depend on) | Full plan doc | Assertions + warnings + row counts | Spot-check + assumption walkthrough |
| Analysis notebook (gut-check, exploration) | Brief intent section at top of notebook | Basic assertions (data loaded, expected shape) | Eyeball results, note surprises |
| One-off query or plot | None | None | None |

The question to ask: "If this produces wrong numbers silently, what breaks?" If the answer is "other analyses," full protocol. If the answer is "nothing, I'll see it immediately," lighter touch.

---

## When to apply

- Any script that produces data other work depends on
- Any analysis with non-obvious assumptions
- Any work the user explicitly asks to be planned
- Any change to shared infrastructure (database queries, data formats, project config)

## When NOT to apply

- Exploratory plotting and one-off queries
- Documentation-only changes
- Trivial fixes (typos, formatting, single-line corrections)
