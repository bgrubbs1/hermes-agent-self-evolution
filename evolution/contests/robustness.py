"""Measured reliability claims and an overclaim linter for outward-facing text."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Sequence

from evolution.contests.stats import wilson_interval


@dataclass(frozen=True)
class TrialSummary:
    passed: int
    trials: int
    rate: float
    ci_low: float
    ci_high: float

    def claim(self, what: str) -> str:
        """Sentence safe to paste into an entry, e.g. a README robustness section."""
        text = (
            f"Passed {self.passed}/{self.trials} {what} "
            f"({self.rate:.0%}; 95% Wilson interval {self.ci_low:.0%}-{self.ci_high:.0%})."
        )
        if self.trials < 10:
            text += " Small sample: treat as indicative only."
        return text


def summarize_trials(outcomes: Sequence[bool]) -> TrialSummary:
    """Pass rate over fresh, held-out trials (e.g. adversarial prompts a judge might try)."""
    if not outcomes:
        raise ValueError("no trials")
    passed = sum(1 for o in outcomes if o)
    low, high = wilson_interval(passed, len(outcomes))
    return TrialSummary(passed, len(outcomes), passed / len(outcomes), low, high)


_OVERCLAIMS = [
    (r"\b(guarantee[sd]?|ensures?|assures?)\b[^.\n]{0,40}\b(wins?|winning|victory|first place|prize)\b",
     "promises a win; contests with human judges and other entrants cannot be guaranteed"),
    (r"\b100\s*%\s*(win|chance|probability|success)", "100% win probability"),
    (r"\b(win|success) (rate|probability|chance)\s*(of|:|=)?\s*(9[5-9]|100)\s*%",
     "near-certain win rate"),
    (r"\b(certain|sure|bound) to win\b", "asserts a certain win"),
    (r"\bcan(?:no|')t lose\b", "asserts the entry cannot lose"),
]


_NEGATION_BEFORE = re.compile(
    r"\b(no|not|nothing|never|cannot|can't|doesn't|does not|won't|will not|nobody|neither)\b\W+(\w+\W+){0,2}$",
    re.IGNORECASE,
)


def _negated(text: str, start: int) -> bool:
    sentence_start = max(text.rfind(".", 0, start), text.rfind("\n", 0, start)) + 1
    return bool(_NEGATION_BEFORE.search(text[sentence_start:start]))


def _quoted(text: str, start: int) -> bool:
    """True when the match sits inside double quotes on its line (citing a claim, not making it)."""
    line_start = text.rfind("\n", 0, start) + 1
    before = text[line_start:start]
    return (before.count('"') % 2 == 1) or (before.count("“") > before.count("”"))


def lint_claims(text: str) -> list[dict]:
    """Flag win guarantees in reports, pitches and agent status messages.

    Negated ("cannot guarantee a win") and double-quoted claims pass.
    """
    issues = []
    for pattern, why in _OVERCLAIMS:
        for match in re.finditer(pattern, text, re.IGNORECASE):
            if _negated(text, match.start()) or _quoted(text, match.start()):
                continue
            line = text.count("\n", 0, match.start()) + 1
            issues.append({"line": line, "text": match.group(0), "why": why})
    return sorted(issues, key=lambda i: i["line"])
