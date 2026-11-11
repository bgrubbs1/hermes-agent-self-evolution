"""Contest calibration: honest odds, receipts, blind judges, closed-loop learning.

Pure Python (no DSPy import) so an external contest bot can depend on it.

    from evolution.contests import (
        ContestField, Ledger, estimate_win_probability, fit_skill_offset, calibration_report,
    )
    ledger = Ledger("contests/ledger.jsonl")
    skill = fit_skill_offset(ledger.skill_history())
    est = estimate_win_probability(ContestField("hackster-x", prizes=3, entrants=1200), skill)
    ledger.record_prediction(est, deadline="2026-12-01T23:59:00-05:00")

See docs/CONTEST_CALIBRATION.md.
"""

from evolution.contests.base_rates import (
    MODEL_VERSION,
    ContestField,
    SkillPosterior,
    UnknownFieldError,
    WinEstimate,
    estimate_win_probability,
    expected_value_per_hour,
    fit_skill_offset,
)
from evolution.contests.calibration import CalibrationReport, calibration_report
from evolution.contests.judges import (
    Artifact,
    BlindBatch,
    JudgePacket,
    JudgeSpec,
    LeakError,
    PanelResult,
    evaluate_panel,
    make_blind_batch,
    run_panel,
    split_judges,
)
from evolution.contests.ledger import Ledger, LedgerError, ResolvedContest
from evolution.contests.robustness import TrialSummary, lint_claims, summarize_trials

__all__ = [
    "MODEL_VERSION",
    "Artifact",
    "BlindBatch",
    "CalibrationReport",
    "ContestField",
    "JudgePacket",
    "JudgeSpec",
    "LeakError",
    "Ledger",
    "LedgerError",
    "PanelResult",
    "ResolvedContest",
    "SkillPosterior",
    "TrialSummary",
    "UnknownFieldError",
    "WinEstimate",
    "calibration_report",
    "estimate_win_probability",
    "evaluate_panel",
    "expected_value_per_hour",
    "fit_skill_offset",
    "lint_claims",
    "make_blind_batch",
    "run_panel",
    "split_judges",
    "summarize_trials",
]
