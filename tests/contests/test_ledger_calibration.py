"""Tests for the receipt-gated ledger and calibration scoring."""

import pytest

from evolution.contests.base_rates import ContestField, Observation, estimate_win_probability
from evolution.contests.calibration import calibration_report
from evolution.contests.ledger import Ledger, LedgerError, ResolvedContest, parse_time

T0 = "2026-10-01T12:00:00+00:00"
T1 = "2026-10-20T12:00:00+00:00"
T2 = "2026-11-05T12:00:00Z"


def _est(cid, prizes=1, entrants=20):
    return estimate_win_probability(ContestField(cid, prizes=prizes, entrants=entrants))


@pytest.fixture
def ledger(tmp_path):
    return Ledger(tmp_path / "ledger.jsonl")


class TestRecording:
    def test_naive_timestamps_rejected(self):
        with pytest.raises(LedgerError):
            parse_time("2026-10-01T12:00:00")

    def test_submission_requires_known_receipt(self, ledger):
        with pytest.raises(LedgerError):
            ledger.record_submission("c", "none", "n/a")
        with pytest.raises(LedgerError):
            ledger.record_submission("c", "confirmation_email", "  ")

    def test_result_requires_source(self, ledger):
        with pytest.raises(LedgerError):
            ledger.record_result("c", "won", "", announced_at=T2)

    def test_append_only_jsonl(self, ledger):
        ledger.record_prediction(_est("c"), made_at=T0)
        ledger.record_submission("c", "confirmation_email", "msg-1", submitted_at=T1)
        assert [e["type"] for e in ledger.events()] == ["prediction", "submission"]


class TestAudit:
    def test_full_chain_resolves(self, ledger):
        ledger.record_prediction(_est("c"), made_at=T0)
        ledger.record_submission("c", "confirmation_page", "https://x/entry/1", submitted_at=T1)
        ledger.record_result("c", "lost", "https://x/winners", announced_at=T2)
        audit = ledger.audit()
        assert len(audit.resolved) == 1 and not audit.resolved[0].won
        (obs,) = ledger.skill_history()
        assert obs.thresholds == (pytest.approx(0.05),) and obs.reached == 0

    def test_screenshot_receipt_is_unconfirmed_and_excluded(self, ledger):
        ledger.record_prediction(_est("vr"), made_at=T0)
        ledger.record_submission("vr", "form_screenshot", "shots/form.png", submitted_at=T1)
        ledger.record_result("vr", "won", "https://x/winners", announced_at=T2)
        audit = ledger.audit()
        assert audit.resolved == []
        assert "vr" in audit.unconfirmed_submissions
        assert "form screenshot" in audit.excluded["vr"]
        assert audit.verified_wins == 0

    def test_hindsight_prediction_not_used(self, ledger):
        ledger.record_submission("c", "confirmation_email", "m", submitted_at=T1)
        ledger.record_result("c", "won", "https://x/w", announced_at=T1)
        ledger.record_prediction(_est("c"), made_at=T2)
        assert "no field size" in ledger.audit().excluded["c"]

    def test_hindsight_prediction_not_scored_but_result_still_learns(self, ledger):
        ledger.record_submission("c", "confirmation_email", "m", submitted_at=T1)
        ledger.record_result("c", "won", "https://x/w", announced_at=T1, entrants=40, prizes=2)
        ledger.record_prediction(_est("c"), made_at=T2)
        audit = ledger.audit()
        (r,) = audit.resolved
        assert r.p is None and audit.forecasted == []
        assert r.observation.thresholds == (pytest.approx(0.05),) and r.observation.reached == 1

    def test_latest_pre_announcement_prediction_used(self, ledger):
        ledger.record_prediction(_est("c", entrants=100), made_at=T0)
        ledger.record_prediction(_est("c", entrants=10), made_at=T1)
        ledger.record_submission("c", "platform_entry_url", "https://x/e", submitted_at=T1)
        ledger.record_result("c", "won", "https://x/w", announced_at=T2)
        (r,) = ledger.resolved()
        assert r.base_rate == pytest.approx(0.1)
        assert r.p is not None

    def test_conflicting_results_excluded(self, ledger):
        ledger.record_prediction(_est("c"), made_at=T0)
        ledger.record_submission("c", "confirmation_email", "m", submitted_at=T1)
        ledger.record_result("c", "won", "https://x/w", announced_at=T2)
        ledger.record_result("c", "lost", "https://x/w", announced_at=T2)
        assert "conflicting" in ledger.audit().excluded["c"]

    def test_pending_and_malformed(self, ledger, tmp_path):
        ledger.record_submission("p", "confirmation_email", "m", submitted_at=T1)
        assert ledger.audit().pending == ["p"]
        with open(ledger.path, "a") as fh:
            fh.write("{not json\n")
        with pytest.raises(LedgerError):
            ledger.events()


def _resolved(p, base, won):
    return ResolvedContest(
        "c", "skill", won, "won" if won else "lost", Observation.binary(base, won), p=p, base_rate=base,
        predicted_at=T0,
    )


class TestCalibration:
    def test_empty_report_warns(self):
        rep = calibration_report([])
        assert rep.n == 0 and "unmeasured" in rep.warnings[0]
        assert "Resolved contests: **0**" in rep.to_markdown()

    def test_skill_score_positive_when_model_beats_base_rate(self):
        rows = [_resolved(0.4, 0.05, True)] * 4 + [_resolved(0.05, 0.05, False)] * 16
        rep = calibration_report(rows)
        assert rep.brier_skill_score > 0
        assert rep.wins == 4

    def test_overconfidence_flagged(self):
        rows = [_resolved(0.9, 0.05, False) for _ in range(25)]
        rep = calibration_report(rows)
        assert rep.brier_skill_score < 0
        assert any("overconfident" in w for w in rep.warnings)

    def test_small_sample_warning_and_bins(self):
        rows = [_resolved(i / 10, 0.1, i % 3 == 0) for i in range(1, 8)]
        rep = calibration_report(rows, n_bins=3)
        assert sum(b.n for b in rep.bins) == 7
        assert any("noise" in w for w in rep.warnings)
        for b in rep.bins:
            assert b.observed_ci[0] <= b.observed_rate <= b.observed_ci[1]


class TestStagedResultsAndAttempts:
    def test_milestones_build_staged_observation(self, ledger):
        ledger.record_submission("c", "confirmation_email", "m", submitted_at=T1)
        ledger.record_result("c", "lost", "https://x/w", announced_at=T2, entrants=200, prizes=3,
                             milestones=[{"name": "finalist", "slots": 20, "reached": True}])
        (r,) = ledger.resolved()
        assert r.observation.thresholds == (pytest.approx(0.1), pytest.approx(0.015))
        assert r.observation.reached == 1

    def test_rank_result(self, ledger):
        ledger.record_submission("k", "platform_api", "kaggle:sub:1", submitted_at=T1)
        ledger.record_result("k", "lost", "https://x/lb", announced_at=T2, entrants=1000, rank=57)
        (r,) = ledger.resolved()
        assert r.observation.thresholds == (pytest.approx(0.057), pytest.approx(0.056))

    def test_inconsistent_milestone_excluded(self, ledger):
        ledger.record_submission("c", "confirmation_email", "m", submitted_at=T1)
        ledger.record_result("c", "won", "https://x/w", announced_at=T2, entrants=200, prizes=3,
                             milestones=[{"name": "finalist", "slots": 20, "reached": False}])
        assert "inconsistent" in ledger.audit().excluded["c"]

    def test_attempts_are_listed_not_counted(self, ledger):
        ledger.record_attempt("sw1", "entry form submitted; no confirmation email", kind="sweepstakes",
                              attempted_at="unknown")
        ledger.record_result("sw1", "lost", "https://x/w", announced_at=T2)
        audit = ledger.audit()
        assert audit.unreceipted_attempts == ["sw1"]
        assert "without any receipt" in audit.excluded["sw1"]
        assert audit.resolved == []

    def test_unknown_submission_time_allowed(self, ledger):
        rec = ledger.record_submission("old", "form_screenshot", "shots/", submitted_at="unknown")
        assert rec["submitted_at"] == "unknown"
        with pytest.raises(LedgerError):
            ledger.record_submission("old", "form_screenshot", "shots/", submitted_at="last week")

    def test_kinds_learn_separately(self, ledger):
        ledger.record_submission("app", "confirmation_email", "m", kind="application", submitted_at=T1)
        ledger.record_result("app", "selected", "msg-2", announced_at=T2, entrants=50, prizes=10)
        assert ledger.skill_history("skill") == []
        assert len(ledger.skill_history("application")) == 1
        assert ledger.skill_history("sweepstakes") == []

    def test_selection_without_field_size_still_counts_as_verified(self, ledger):
        ledger.record_submission("app", "confirmation_email", "m", kind="application", submitted_at=T1)
        ledger.record_result("app", "selected", "msg-2", announced_at=T2)
        audit = ledger.audit()
        assert audit.verified_wins == 1 and "no field size" in audit.excluded["app"]

    def test_v1_ledger_records_still_read(self, ledger):
        import json
        with open(ledger.path, "w") as fh:
            fh.write(json.dumps({"type": "prediction", "contest_id": "c", "p_mean": 0.05, "p_low": 0.01,
                                 "p_high": 0.1, "base_rate": 0.05, "skill_based": False,
                                 "evidence_class": "chance_only", "made_at": T0}) + "\n")
        ledger.record_submission("c", "confirmation_email", "m", submitted_at=T1)
        ledger.record_result("c", "lost", "https://x/w", announced_at=T2)
        (r,) = ledger.resolved()
        assert r.kind == "sweepstakes"
