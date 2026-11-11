---
name: contest-calibration
description: Enter skill contests and hackathons (Hackster, Devpost, element14, PCBWay, Kaggle-style, sweepstakes) with honest odds and checkable evidence. Use when choosing which contests to enter, estimating win probability, judging an entry before submission, logging submissions/results, or reporting contest status. Replaces hand-set probabilities, self-graded persona judges, and unreceipted "submitted" claims with the evolution.contests tools.
---

# Contest Calibration

No tool ensures a win. Contests have human judges and other entrants. This skill
does what can be done: spend hours where the odds are best, ship complete
entries, test them the way judges will, and learn from every result, losses
included.

The tools live in `hermes-agent-self-evolution` (`evolution.contests`). Ledger:
`~/.hermes/contests/ledger.jsonl` unless the user names another.

## Hard rules

1. **Never type a probability.** Every P(win) comes from
   `python -m evolution.contests.cli predict`. If the field size is unknown, the
   answer is "unknown". Go observe it (platform counter, applicant count, last
   edition's gallery); never use a default.
2. **Never call self-graded scores independent.** Personas you wrote, or judges
   that saw this conversation, are `self_graded`. Report them as drafts, not as
   evidence.
3. **No receipt, no submission.** Record a submission only with a confirmation
   email, confirmation page, platform entry URL or API record. A screenshot of a
   filled form is `form_screenshot` and stays UNCONFIRMED.
4. **Record every result**, including losses, with the announcement URL and time.
5. **No win guarantees in any outward text.** Run
   `python -m evolution.contests.cli lint <file>` on status reports and pitches.
6. Probabilities are always reported with their interval and evidence class,
   e.g. `P(win) 1.9% (90% 0.4%-6.8%), evidence: base_rate_only`.

## Workflow

### 1. Select (before any work)

- Verify eligibility and rules from the official page (region, age, team size,
  originality, license, video length, deadline with timezone).
- Count the field. `--entrants` includes us. Use `--entrants-range LOW HIGH`
  when only an estimate exists, and cite it with `--entrants-source`.
- Model multi-stage programs (apply, then selected challengers, then prizes) as
  one prediction per stage. The application stage has its own odds.
- Sweepstakes / random draws: add `--sweepstakes`. Skill does not apply.
- Predict and record:
  `python -m evolution.contests.cli predict --ledger L --contest-id ID --prizes N --entrants M --entrants-source URL --deadline ISO`
- Rank candidates by EV per hour (`expected_value_per_hour`). Show the
  pessimistic `ev_per_hour_low` alongside the mean.
- A judgment shift (`--shift`, log-odds) needs `--reason` with concrete,
  checkable facts (e.g. "gallery shows 4 of 31 entries meet the video rule").
  It is labelled `+judgment_shift` and is not evidence.

### 2. Build to the rubric

- Write the official rubric down verbatim. Map each criterion to a concrete
  artifact (repo, BOM, schematic, video, license file, limits section).
- Reuse assets across contests only where rules allow it (originality clauses).

### 3. Judge it blind (before submitting)

Use `evolution.contests.judges`:

- Collect >=3 **real past winners** and >=3 **typical entrants** from the same or
  a comparable contest. Scrub placement words ("winner", "1st place",
  "finalist"); `make_blind_batch` refuses leaky text.
- Add >=1 **control**: a copy of our entry with a planted defect (license
  removed, video over the limit, BOM missing). This catches judges that do not
  read carefully.
- Judges: >=3, isolated from this conversation (fresh subagents, separate API
  calls, or humans), preferably from different model families. Each gets only
  the packet prompt.
- `evaluate_panel` then decides whether the panel's opinion counts:
  real winners must outrank typical entrants (AUC >= 0.75), judges must agree
  (ICC >= 0.4), planted defects must be caught (>= 0.8), and the panel must not
  be self-graded or reused from the optimisation loop.
- If the panel is NOT VALID, say so and do not cite its scores.
- Report: where the entry ranks among real winners and entrants, with the
  bootstrap interval. Never report "95/100".

### 4. Iterate (bounded)

- Fix against the rubric and the panel feedback. Iterate with an optimisation
  judge set; score the final version only with `split_judges` holdout judges.
- Stop after 2 consecutive iterations with no gain on the holdout panel, or when
  marginal hours beat another contest's EV per hour.
- Robustness: run >=20 fresh adversarial inputs a judge might try and report
  `summarize_trials(...).claim("fresh judge-style prompts")` in the entry.

### 5. Submit

- Final compliance checklist against the rules page, then submit.
- Record the receipt:
  `python -m evolution.contests.cli submit --ledger L --contest-id ID --receipt-kind confirmation_email --receipt-ref MSG_ID --receipt-file saved.eml`

### 6. Close the loop

- When results post:
  `python -m evolution.contests.cli result --ledger L --contest-id ID --outcome won|placed|lost|disqualified --source URL --announced-at ISO`
- Monthly: `python -m evolution.contests.cli report --ledger L`. The forecaster has
  learned something only if its Brier skill score against the naive base rate is
  above 0 over >=20 resolved contests. Until then every estimate is mostly
  prior. Say so.

## Status report format

```
Contest | stage | P(win) mean (90%) | evidence | EV/h | receipt | result
```

Counts come from `report --json`. Never from memory.
