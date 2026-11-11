"""Predicted-vs-actual scoring for resolved contests.

The comparator is the naive base rate (prizes / entrants) recorded with each
prediction.  A forecaster that cannot beat that comparator on Brier score has
not learned anything about our odds, however confident its numbers look.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from typing import Optional, Sequence

from evolution.contests.ledger import ResolvedContest
from evolution.contests.stats import brier_score, log_loss, wilson_interval

MIN_RESOLVED_FOR_SIGNAL = 20


@dataclass(frozen=True)
class ReliabilityBin:
    n: int
    p_min: float
    p_max: float
    mean_predicted: float
    observed_rate: float
    observed_ci: tuple[float, float]


@dataclass
class CalibrationReport:
    n: int
    wins: int
    expected_wins: Optional[float] = None
    expected_wins_sd: Optional[float] = None
    brier_model: Optional[float] = None
    brier_base_rate: Optional[float] = None
    brier_skill_score: Optional[float] = None
    log_loss_model: Optional[float] = None
    log_loss_base_rate: Optional[float] = None
    bins: list[ReliabilityBin] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)

    def to_markdown(self) -> str:
        lines = ["## Contest calibration", ""]
        lines.append(f"- Resolved contests: **{self.n}** (verified wins: **{self.wins}**)")
        if self.n:
            lines.append(
                f"- Expected wins from forecasts: {self.expected_wins:.2f} "
                f"(sd {self.expected_wins_sd:.2f}); actual: {self.wins}"
            )
            lines.append(
                f"- Brier: model {self.brier_model:.4f} vs base rate {self.brier_base_rate:.4f}"
                + (
                    f" (skill score {self.brier_skill_score:+.3f}; >0 beats prizes/entrants)"
                    if self.brier_skill_score is not None
                    else ""
                )
            )
            lines.append(
                f"- Log loss: model {self.log_loss_model:.4f} vs base rate {self.log_loss_base_rate:.4f}"
            )
            lines += ["", "| bin p-range | n | mean predicted | observed (95% CI) |", "|---|---|---|---|"]
            for b in self.bins:
                lines.append(
                    f"| {b.p_min:.3f}-{b.p_max:.3f} | {b.n} | {b.mean_predicted:.3f} | "
                    f"{b.observed_rate:.3f} ({b.observed_ci[0]:.3f}-{b.observed_ci[1]:.3f}) |"
                )
        if self.warnings:
            lines += ["", "**Warnings**", ""] + [f"- {w}" for w in self.warnings]
        return "\n".join(lines) + "\n"


def calibration_report(
    resolved: Sequence[ResolvedContest],
    *,
    n_bins: int = 5,
    min_n: int = MIN_RESOLVED_FOR_SIGNAL,
) -> CalibrationReport:
    resolved = list(resolved)
    wins = sum(1 for r in resolved if r.won)
    report = CalibrationReport(n=len(resolved), wins=wins)
    if not resolved:
        report.warnings.append(
            "no resolved contests with a strong receipt and a pre-announcement prediction: "
            "calibration is unmeasured, so every probability is still a prior"
        )
        return report

    model_pairs = [(r.p, int(r.won)) for r in resolved]
    base_pairs = [(r.base_rate, int(r.won)) for r in resolved]
    report.expected_wins = sum(r.p for r in resolved)
    report.expected_wins_sd = math.sqrt(sum(r.p * (1 - r.p) for r in resolved))
    report.brier_model = brier_score(model_pairs)
    report.brier_base_rate = brier_score(base_pairs)
    if report.brier_base_rate > 0:
        report.brier_skill_score = 1.0 - report.brier_model / report.brier_base_rate
    report.log_loss_model = log_loss(model_pairs)
    report.log_loss_base_rate = log_loss(base_pairs)

    # Equal-count bins: contest probabilities cluster near zero, so equal-width
    # bins would put nearly everything in the first one.
    ordered = sorted(resolved, key=lambda r: r.p)
    n_bins = max(1, min(n_bins, len(ordered)))
    size, extra = divmod(len(ordered), n_bins)
    start = 0
    for i in range(n_bins):
        stop = start + size + (1 if i < extra else 0)
        chunk = ordered[start:stop]
        start = stop
        k = sum(1 for r in chunk if r.won)
        report.bins.append(
            ReliabilityBin(
                n=len(chunk),
                p_min=chunk[0].p,
                p_max=chunk[-1].p,
                mean_predicted=sum(r.p for r in chunk) / len(chunk),
                observed_rate=k / len(chunk),
                observed_ci=wilson_interval(k, len(chunk)),
            )
        )

    if len(resolved) < min_n:
        report.warnings.append(
            f"only {len(resolved)} resolved contests (< {min_n}): scores are dominated by noise; "
            "do not tune the model on them"
        )
    if report.expected_wins_sd and report.expected_wins_sd > 0:
        z = (wins - report.expected_wins) / report.expected_wins_sd
        if abs(z) > 2:
            direction = "fewer" if z < 0 else "more"
            report.warnings.append(
                f"actual wins are {abs(z):.1f} sd {direction} than forecast: "
                f"forecasts look {'over' if z < 0 else 'under'}confident"
            )
    return report
