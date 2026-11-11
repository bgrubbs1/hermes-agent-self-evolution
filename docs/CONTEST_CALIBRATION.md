# Contest calibration (`evolution.contests`)

Tools that make a contest-entering agent's numbers checkable. They do not make
wins certain. Skill contests have human judges and other entrants, so no system
can promise a win. What a system *can* do is spend hours where the odds are
best, ship complete and robust entries, and learn from every result.

## Why this exists

A self-review of the user's contest bot (Nov 2026) found four defects. Each
module below fixes one or more of them.

| Defect found in the bot | Consequence | Fix here |
|---|---|---|
| `calculate_historical_score()` uses hard-coded 2-5% defaults, and synthetic "winner" records feed the analyser | "100% win probability" was an artifact of mock input | `base_rates.py`: P(win) comes from prizes / entrants plus a skill offset learned only from resolved, receipted contests. With no field count it refuses to answer. |
| 10 "independent judges" were personas written in the build session | The 79 -> 92.5 -> 95.1 trajectory was the author grading their own work | `judges.py`: blind batches mixed with real winners, typical entrants and planted-defect controls, plus a validity gate the panel must pass before its opinion counts |
| Form-submit screenshots treated as entries | The ledger could not separate an entry from an attempted one | `ledger.py`: strong receipts only. Screenshots stay "unconfirmed" and are excluded from learning. |
| Predictions never scored against results | Every score was static opinion | `ledger.py` + `calibration.py`: predicted-vs-actual Brier/log-loss against the naive base-rate comparator, with reliability bins and Wilson intervals |
| "It works" claims about entries | Judges probe entries, and adjectives do not survive probing | `robustness.py`: `k/N passed (95% Wilson interval)` claims, plus a linter that flags win guarantees |

## Design choices worth knowing

- **Skill offset, not a quality score.** `P(win) = sigmoid(logit(prizes/entrants) + delta)`.
  `delta` is our log-odds advantage over an average entry, fitted by grid posterior from
  resolved contests. The prior is `Normal(-sd^2/2, sd)`. That centring keeps the
  no-history mean equal to the naive base rate for small base rates. A symmetric prior
  on log-odds would quietly inflate every forecast by about 65%. For large base rates the
  centring errs conservative instead: 3 prizes among 10 entrants gives a 24% mean against a
  naive 30%.
- **No defaults for the field.** `ContestField` without `entrants` or `entrants_range` raises
  `UnknownFieldError`. That is the specific failure that produced the mock 2-5% priors.
- **Sweepstakes are chance-only.** Skill history and judgment shifts are ignored, and
  judgment shifts are refused.
- **Judgment is allowed but labelled.** `manual_shift` needs a written reason and tags the
  estimate `+judgment_shift`.
- **Hindsight is excluded.** Only predictions recorded before the result's
  `announced_at` count, and the CLI makes `announced_at` mandatory.
- **Comparator is explicit.** Calibration reports Brier skill vs. the base rate recorded
  with each prediction. A positive score over >=20 resolved contests is the first point at
  which the forecaster has demonstrably learned something.

### Judge panel validity gate (`evaluate_panel`)

A panel's verdict on our entry is reported as VALID only when:

1. no judge shares the author's context (`self_graded`) and none steered the iterations
   (`optimization_contaminated`; use `split_judges` for a held-out panel);
2. >=3 judges, plus >=3 real winners and >=3 typical entrants in the batch;
3. the panel ranks real winners above typical entrants (AUC >= 0.75). A panel that cannot
   recognise past winners cannot recognise ours;
4. inter-rater agreement ICC(2,1) >= 0.4;
5. planted-defect controls are scored below the entry by >= 80% of judge-control pairs.

Additional safeguards:

- **Leak scrubbing.** Artifacts containing "winner", "1st place", "finalist", etc. are
  refused. A winner's page usually carries a badge, and that badge gives the answer away.
- **Correlated judges.** N instances of one model family are flagged
  `single_model_family`, because their errors are correlated and their agreement is
  inflated. Mixed families or humans are better.
- **Survivorship bias.** Published "typical" entries skew toward complete projects, so
  winner-vs-typical gaps measured from public galleries understate how often real
  entries are incomplete. Treat the gallery as a lower bound on field quality.

## Discovery that fails loudly (`discovery.py`)

The bot's scrapers parsed HTML and reported "Found 0 contests". Discovery now reads structured data:

- **Devpost**: the public JSON API (`/api/hackathons`). It gives title, URL, dates, registrations and total
  prize value. Its `prizes_counts` field is **not** a slot count: on 2026-11-11 it read 14+3 and later 1+0 for
  the same hackathon. Slot counts come from the hackathon page's prize list (`--prizes N`, one request per
  second). Those pages sit behind a Cloudflare bot challenge for scripted clients. When it appears, it is
  reported, not bypassed.
- **Hackster**: the `ContestsPage` JSON block embedded in `/contests`. robots.txt allows `/contests` and
  disallows `/projects/*`.
- **Registry**: a hand-kept JSON list for element14 (refuses automated clients), HeroX, Innocentive, lablab and
  organizer forms. Entries matching a discovered `contest_id` override its fields.

A source that returns nothing raises `DiscoveryError` with HTTP status, byte counts and parser hits. It never
returns `[]`. The tests run against trimmed copies of the live responses captured on 2026-11-11.

## Learning faster than one bit per contest

- **Staged results.** `Observation` generalises won/lost to ordered stages. A result can record finalists
  (`--milestone finalist:20:yes`) or a full-leaderboard `--rank`. Under the same model,
  P(reach stage k) = sigmoid(logit(slots_k/N) + delta), so a finalist that lost tells the fit far more than a
  plain loss.
- **Final field sizes.** Results carry the final `entrants` and `prizes` from the gallery. Each receipted, sourced
  result then trains the skill offset even without a pre-announcement forecast. Calibration of forecasts
  still needs one.
- **Kinds.** `skill`, `application` (selection rounds such as RoadTests, idea/hardware applications and proposal
  stages) and `sweepstakes` are fitted separately. Applications resolve in weeks, so they add data fastest.
- **Judge backtests.** `backtest_panel` scores a judge panel on past contests whose winners are published. It
  reports per-contest winner-vs-entrant AUC, a bootstrap interval over contests, and top-pick hit rate against
  chance. This is calibration available today; it does not wait for our own results.
- **Runnable isolated judges.** `make_llm_judge_fn({"j1": "anthropic/...", "j2": "openai/...", "j3": "gemini/..."})`
  makes one stateless call per judge with only the blind packet. It replaces the persona-judge loop.

## Using it from the contest bot

```python
from evolution.contests import ContestField, Ledger, estimate_win_probability, fit_skill_offset

ledger = Ledger("~/.hermes/contests/ledger.jsonl")
skill = fit_skill_offset(ledger.skill_history())   # replaces calculate_historical_score() defaults
est = estimate_win_probability(
    ContestField("hackster-foo-2026", prizes=3, entrants=1200, entrants_source="contest page counter"),
    skill,
)
ledger.record_prediction(est, deadline="2026-12-01T23:59:00-05:00")
print(est.summary())
```

Replace the persona-judge loop with `make_blind_batch` -> `run_panel(batch, judge_fn)` ->
`evaluate_panel`. `judge_fn` receives a `JudgePacket` and returns JSON text or a dict. Run each
call in a fresh context (separate API call or subagent), never in the authoring session.

CLI equivalents: `python -m evolution.contests.cli {predict,submit,result,report,lint} --help`.

The Hermes skill `skills/contest-calibration/SKILL.md` encodes the workflow. It passes this
repo's skill constraints, so GEPA can evolve it later. Its fitness should come from
holdout-panel validity and ledger calibration, not from a judge that shares its context.

## On r/SATNA_PROJECT as a source

The requested thread could not be read: Reddit and its mirrors are blocked from this
environment. Reconstruction from third-party citations (Nov 2026) suggests the subreddit
mostly reposts AI/dev-tool roundups. No contest-specific methods were found. Patterns adopted
here, all unverified secondhand:

- **Planted-defect evaluation corpora** (Library-First Engineering, Claude-LFE): adopted as
  `control` artifacts in judge panels.
- **Gated promotion and bounded fix loops** (Library-First Engineering stops after 2 failed
  cycles; agentic-stack graduates lessons only through a gate): adopted as "learn only from
  receipted, resolved contests" and the skill's stop rule.
- **Reflexion / Self-Refine loops** (ai-engineering-from-scratch): useful for iteration, but
  self-refinement scored by the same model is exactly the self-grading failure, so the final
  score comes from a held-out panel.

Signals of star inflation and marketing bots were observed around some repos promoted there.
Treat its benchmark claims as unverified until reproduced.

## Limitations

- Contests resolve slowly. With a few results per month, the skill offset will be weakly
  identified for a long time, and the report says so. Until n >= 20 the honest summary is
  "base rate plus prior".
- Base rates assume exchangeable entries and one prize per entrant. Tiered prizes, category
  prizes and per-person limits should be modelled as separate fields.
- A valid panel is evidence about rubric quality, not about any particular human judge's
  taste. Real outcomes in the ledger remain the ground truth.
