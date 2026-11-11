"""Bayesian win-probability estimates grounded in field size, not hand-set defaults.

Model
-----
For a contest with ``prizes`` winning slots and ``N`` competing entries
(including ours), an exchangeable "average" entry wins a slot with base rate
``b = prizes / N``.

Our entries may be better or worse than average.  That is captured by a single
log-odds offset ``delta``:

    P(win) = sigmoid(logit(b) + delta)

``delta`` is *learned* from resolved contests in the outcome ledger (grid
posterior, normal prior).  The prior is centred at ``-sd**2 / 2`` so that, with
no history, the mean forecast equals the naive base rate for small ``b`` rather
than being inflated by the prior's spread.

Sweepstakes / random draws are chance-only: the skill offset is ignored.

There is deliberately no default entrant count.  If nobody has observed or
estimated the field, the honest answer is "unknown", and a hard-coded 2-5 %
default is exactly how mock priors end up looking like measurements.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from typing import Iterable, Optional

from evolution.contests.stats import logit, logsumexp, sigmoid, weighted_quantile

MODEL_VERSION = "base-rate-logit-offset/1"


class UnknownFieldError(ValueError):
    """Raised when a contest has no observed or estimated entrant count."""


@dataclass(frozen=True)
class ContestField:
    """What is known about the competition for one contest (or one stage of it).

    ``entrants`` is the total number of competing entries *including ours*.
    Use ``entrants_range`` (low, high) when only an estimate is available.
    Model multi-stage programs (apply -> selected challenger -> prize) as one
    field per stage.
    """

    contest_id: str
    prizes: int
    entrants: Optional[int] = None
    entrants_range: Optional[tuple[int, int]] = None
    skill_based: bool = True
    entrants_source: str = ""

    def __post_init__(self) -> None:
        if not self.contest_id.strip():
            raise ValueError("contest_id is required")
        if self.prizes < 1:
            raise ValueError("prizes must be >= 1 (count only prizes you are eligible for)")
        if self.entrants is not None and self.entrants < 1:
            raise ValueError("entrants must be >= 1")
        if self.entrants_range is not None:
            low, high = self.entrants_range
            if low < 1 or high < low:
                raise ValueError("entrants_range must be (low, high) with 1 <= low <= high")
        if self.entrants is None and self.entrants_range is None:
            raise UnknownFieldError(
                f"{self.contest_id}: no entrant count or range. Observe the field "
                "(platform counter, applicant count, past editions) before estimating; "
                "do not substitute a default."
            )

    def entrant_grid(self, points: int = 9) -> list[float]:
        """Entrant counts to integrate over (log-uniform across a range)."""
        if self.entrants is not None:
            return [float(self.entrants)]
        low, high = self.entrants_range  # type: ignore[misc]
        if low == high or points < 2:
            return [float(low)]
        log_low, log_high = math.log(low), math.log(high)
        return [math.exp(log_low + (log_high - log_low) * i / (points - 1)) for i in range(points)]

    def base_rate(self) -> float:
        """Mean naive win rate prizes / entrants over the entrant grid."""
        grid = self.entrant_grid()
        return sum(min(1.0, self.prizes / n) for n in grid) / len(grid)


@dataclass(frozen=True)
class SkillPosterior:
    """Discrete posterior over the log-odds skill offset ``delta``."""

    grid: tuple[float, ...]
    weights: tuple[float, ...]
    n_obs: int
    wins: int
    prior_mean: float
    prior_sd: float

    @property
    def mean(self) -> float:
        return sum(g * w for g, w in zip(self.grid, self.weights))

    def interval(self, low: float = 0.05, high: float = 0.95) -> tuple[float, float]:
        return (
            weighted_quantile(self.grid, self.weights, low),
            weighted_quantile(self.grid, self.weights, high),
        )

    @property
    def odds_multiplier(self) -> float:
        """exp(mean delta): how our odds compare with an average entry's."""
        return math.exp(self.mean)


def fit_skill_offset(
    history: Iterable[tuple[float, bool]],
    *,
    prior_sd: float = 1.0,
    prior_mean: Optional[float] = None,
    grid_size: int = 241,
    span_sd: float = 6.0,
) -> SkillPosterior:
    """Posterior over ``delta`` from resolved skill contests.

    ``history`` holds ``(base_rate_at_prediction_time, won)`` pairs taken only
    from contests with a verified submission and a published result.
    """
    if prior_sd <= 0:
        raise ValueError("prior_sd must be positive")
    if grid_size < 3:
        raise ValueError("grid_size must be >= 3")
    if prior_mean is None:
        prior_mean = -(prior_sd**2) / 2.0
    history = [(float(b), bool(y)) for b, y in history]
    for b, _ in history:
        if not 0.0 < b <= 1.0:
            raise ValueError(f"base rate {b} outside (0, 1]")

    lo = prior_mean - span_sd * prior_sd
    hi = prior_mean + span_sd * prior_sd
    grid = [lo + (hi - lo) * i / (grid_size - 1) for i in range(grid_size)]
    base_logits = [(logit(b), y) for b, y in history]

    log_w = []
    for d in grid:
        lp = -0.5 * ((d - prior_mean) / prior_sd) ** 2
        for lb, y in base_logits:
            p = sigmoid(lb + d)
            lp += math.log(max(p, 1e-300)) if y else math.log(max(1.0 - p, 1e-300))
        log_w.append(lp)
    norm = logsumexp(log_w)
    weights = tuple(math.exp(x - norm) for x in log_w)
    return SkillPosterior(
        grid=tuple(grid),
        weights=weights,
        n_obs=len(history),
        wins=sum(1 for _, y in history if y),
        prior_mean=prior_mean,
        prior_sd=prior_sd,
    )


@dataclass(frozen=True)
class WinEstimate:
    contest_id: str
    p_mean: float
    p_low: float
    p_high: float
    base_rate: float
    skill_based: bool
    skill_offset_mean: float
    n_history: int
    evidence_class: str
    model_version: str = MODEL_VERSION
    notes: tuple[str, ...] = field(default_factory=tuple)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["notes"] = list(self.notes)
        return d

    def summary(self) -> str:
        return (
            f"{self.contest_id}: P(win) {self.p_mean:.2%} "
            f"(90% interval {self.p_low:.2%}-{self.p_high:.2%}); "
            f"naive base rate {self.base_rate:.2%}; evidence: {self.evidence_class}"
        )


def estimate_win_probability(
    contest: ContestField,
    skill: Optional[SkillPosterior] = None,
    *,
    manual_shift: float = 0.0,
    manual_reason: str = "",
    interval: tuple[float, float] = (0.05, 0.95),
) -> WinEstimate:
    """Posterior predictive P(win) for one contest.

    ``manual_shift`` is a log-odds adjustment from human or agent judgment.  It
    needs a written reason, is labelled in ``evidence_class``, and is refused
    for chance-only contests.
    """
    if manual_shift and not manual_reason.strip():
        raise ValueError("manual_shift requires a written manual_reason")
    if manual_shift and not contest.skill_based:
        raise ValueError("chance-only contests cannot be shifted by judgment")
    if skill is None:
        skill = fit_skill_offset([])

    grid_n = contest.entrant_grid()
    values: list[float] = []
    weights: list[float] = []
    for n in grid_n:
        b = min(1.0, contest.prizes / n)
        if not contest.skill_based:
            values.append(b)
            weights.append(1.0 / len(grid_n))
            continue
        lb = logit(b)
        for d, w in zip(skill.grid, skill.weights):
            if w < 1e-15:
                continue
            values.append(sigmoid(lb + d + manual_shift))
            weights.append(w / len(grid_n))

    total = sum(weights)
    p_mean = sum(v * w for v, w in zip(values, weights)) / total
    p_low = weighted_quantile(values, weights, interval[0])
    p_high = weighted_quantile(values, weights, interval[1])

    notes: list[str] = []
    if not contest.skill_based:
        evidence = "chance_only"
        notes.append("random draw: skill history and judgment do not apply")
    elif skill.n_obs == 0:
        evidence = "base_rate_only"
        notes.append("no resolved contests yet: interval is the prior on skill, not a measurement")
    else:
        evidence = f"base_rate+history(n={skill.n_obs},wins={skill.wins})"
        if skill.n_obs < 20:
            notes.append(f"skill offset weakly identified (n={skill.n_obs} < 20)")
    if manual_shift:
        evidence += "+judgment_shift"
        notes.append(f"judgment shift {manual_shift:+.2f} log-odds: {manual_reason.strip()}")
    if contest.entrants is None:
        low, high = contest.entrants_range  # type: ignore[misc]
        notes.append(f"entrant count is an estimate ({low}-{high})")
    if contest.entrants_source:
        notes.append(f"field source: {contest.entrants_source}")

    return WinEstimate(
        contest_id=contest.contest_id,
        p_mean=p_mean,
        p_low=p_low,
        p_high=p_high,
        base_rate=contest.base_rate(),
        skill_based=contest.skill_based,
        skill_offset_mean=skill.mean if contest.skill_based else 0.0,
        n_history=skill.n_obs,
        evidence_class=evidence,
        notes=tuple(notes),
    )


def expected_value_per_hour(estimate: WinEstimate, prize_value: float, hours: float) -> dict:
    """EV per hour using the posterior mean, with the pessimistic bound alongside."""
    if hours <= 0:
        raise ValueError("hours must be positive")
    return {
        "contest_id": estimate.contest_id,
        "ev_per_hour": prize_value * estimate.p_mean / hours,
        "ev_per_hour_low": prize_value * estimate.p_low / hours,
        "evidence_class": estimate.evidence_class,
    }
