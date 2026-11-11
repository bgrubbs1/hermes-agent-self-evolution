---
name: contest-calibration
description: Find, enter and learn from skill contests, hackathons, hardware applications and sweepstakes (Devpost, Hackster, element14, HeroX, Innocentive, lablab, Kaggle-style) with honest odds and checkable evidence. Use when discovering contests, choosing which to enter, estimating win probability, judging an entry before submission, logging receipts and results, or reporting contest status. Replaces hand-set probabilities, self-graded persona judges, HTML scrapers that report "0 contests", and unreceipted "submitted" claims with the evolution.contests tools.
---

# Contest Calibration

No tool ensures a win. Contests have human judges and other entrants. This skill
does what can be done: find contests, spend hours where the odds are best, ship
complete entries, test them the way judges will, and learn from every result,
losses included.

Setup (once per machine): `pip install -e <path>/hermes-agent-self-evolution`.
Ledger: `~/.hermes/contests/ledger.jsonl`. Registry: `~/.hermes/contests/registry.json`.
Below, `C` means `python -m evolution.contests.cli` and `L` the ledger path.

## Hard rules

1. **Never type a probability.** Every P(win) comes from `C predict`. If the
   field size is unknown the answer is "unknown": observe it (platform counter,
   gallery count, applicant count, last edition); never use a default.
2. **Never call self-graded scores independent.** Personas you wrote, or judges
   that saw this conversation, are `self_graded`: drafts, not evidence.
3. **No receipt, no submission.** `C submit` only with a confirmation email,
   confirmation page, platform entry URL or API record. A filled-form screenshot
   is `form_screenshot` (unconfirmed). Nothing at all: `C attempt`.
4. **Record every result**, losses included, with the announcement source and
   time, plus final `--entrants` and `--prizes` (galleries show both).
5. **Never bypass bot protection.** If a site shows a bot challenge or refuses
   automated clients, stop and take the value from the page by hand into the
   registry. Do not spoof browsers or change user agents to get through.
6. **No win guarantees in outward text.** Run `C lint <file>` on reports and pitches.
7. Report probabilities with interval and evidence class, e.g.
   `P(win) 1.3% (90% 0.06%-5.1%), evidence: base_rate+history(n=6,wins=0)`.

## Workflow

### 1. Discover

- `C discover` reads the Devpost API and Hackster's contests data. A source that
  returns nothing raises an error with diagnostics: report it, never "no contests".
- `C discover --prizes 10` also reads up to 10 Devpost hackathon pages (1 per
  second) to count prize slots. A bot challenge there is reported, not bypassed.
- Sites without usable data (element14, HeroX, Innocentive, lablab, organizer
  forms): add entries to the registry. A registry entry whose `contest_id`
  matches a discovered contest overrides fields such as `prize_slots`.
- Organizer emails (confirmations, reminders, "winners announced") are a
  discovery and receipt source: search mail for them.

### 2. Select (before any work)

- Verify eligibility and rules on the official page (region, age, team,
  originality, license, video length, deadline with timezone).
- Count prize slots you can actually win (exclude sponsor-tool tracks you will
  not use). `--entrants` includes us; `--entrants-range LOW HIGH` when only
  registrations are known; cite `--entrants-source`.
- `--kind skill` (default), `--kind application` for selection stages (RoadTests,
  idea/hardware applications, proposal rounds), `--kind sweepstakes` for draws.
  Each kind learns its own offset; sweepstakes never use skill.
- Predict and record before the deadline:
  `C predict --ledger L --contest-id ID --prizes N --entrants-range LO HI --entrants-source SRC --deadline ISO`
- Rank by EV per hour (`expected_value_per_hour`); show `ev_per_hour_low` too.
- `--shift` (log-odds) needs `--reason` with checkable facts; it is labelled
  `+judgment_shift` and is not evidence.

### 3. Build to the rubric

- Copy the official rubric verbatim; map each criterion to a concrete artifact
  (repo, BOM, schematic, video, license file, limits section).
- Reuse assets across contests only where originality rules allow.

### 4. Judge it blind (before submitting)

- First check the panel itself: `backtest_panel` on >=5 past contests with
  published winners (winners + typical entrants, placement words scrubbed).
  If its AUC interval includes 0.5, the panel is no better than chance: stop
  using its scores.
- Then `make_blind_batch` with our entry, >=3 real winners, >=3 typical entrants
  and >=1 control (our entry with a planted defect: license removed, video over
  the limit, BOM missing).
- Judges: `make_llm_judge_fn({...})` with >=3 models from different families
  (e.g. anthropic/..., openai/..., gemini/...); each call is isolated. Humans
  count too. Never this conversation.
- `evaluate_panel` decides whether the verdict counts (winner-vs-entrant AUC >=
  0.75, ICC >= 0.4, defect detection >= 0.8, not self-graded or reused). If NOT
  VALID, say so and do not cite the scores. Report rank among real winners and
  entrants with its interval, never "95/100".
- Respect robots.txt when collecting references (Hackster disallows `/projects/*`).

### 5. Iterate (bounded)

- Iterate against `split_judges` optimisation judges; score the final version
  only with the holdout judges.
- Stop after 2 iterations without holdout gain, or when the hours would earn
  more EV in another contest.
- Robustness: >=20 fresh adversarial inputs a judge might try; put
  `summarize_trials(...).claim("fresh judge-style prompts")` in the entry.

### 6. Submit

- Final compliance pass against the rules page, then submit and wait for the
  organizer's confirmation.
  `C submit --ledger L --contest-id ID --receipt-kind confirmation_email --receipt-ref gmail:MSG_ID`
- "Submission started" emails are not confirmations: finish the submission.

### 7. Close the loop

- On results: `C result --ledger L --contest-id ID --outcome won|placed|lost|disqualified|selected|not_selected --source URL --announced-at ISO --entrants N --prizes K`
  Add `--milestone finalist:20:yes` for wider stages or `--rank R` for full
  leaderboards: each makes one result count for more.
- Check the gallery yourself for whether our project is a winner. Similar names
  are not proof: open the winning project and check its team or repo.
- Monthly `C report --ledger L`. The forecaster has learned something only if
  its Brier skill score against the naive base rate is above 0 over >=20
  forecast contests. Until then every estimate is mostly prior. Say so.

## Status report format

```
Contest | kind | stage | P(win) mean (90%) | evidence | EV/h | receipt | result
```

Counts come from `C report --json`. Never from memory.
