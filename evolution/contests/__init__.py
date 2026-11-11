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
    KINDS,
    MODEL_VERSION,
    ContestField,
    Observation,
    SkillPosterior,
    UnknownFieldError,
    WinEstimate,
    estimate_win_probability,
    expected_value_per_hour,
    fit_skill_offset,
)
from evolution.contests.calibration import CalibrationReport, calibration_report
from evolution.contests.discovery import ContestLead, DiscoveryError, discover, load_registry
from evolution.contests.judges import (
    Artifact,
    BacktestResult,
    BlindBatch,
    JudgePacket,
    JudgeSpec,
    LeakError,
    PanelResult,
    PastContest,
    backtest_panel,
    evaluate_panel,
    make_blind_batch,
    run_panel,
    split_judges,
)
from evolution.contests.ledger import Ledger, LedgerError, ResolvedContest
from evolution.contests.llm_judges import judge_specs, make_llm_judge_fn
from evolution.contests.robustness import TrialSummary, lint_claims, summarize_trials

__all__ = [
    "KINDS",
    "MODEL_VERSION",
    "Artifact",
    "BacktestResult",
    "BlindBatch",
    "CalibrationReport",
    "ContestField",
    "ContestLead",
    "DiscoveryError",
    "Observation",
    "PastContest",
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
    "backtest_panel",
    "calibration_report",
    "discover",
    "estimate_win_probability",
    "evaluate_panel",
    "expected_value_per_hour",
    "fit_skill_offset",
    "judge_specs",
    "lint_claims",
    "load_registry",
    "make_blind_batch",
    "make_llm_judge_fn",
    "run_panel",
    "split_judges",
    "summarize_trials",
]
