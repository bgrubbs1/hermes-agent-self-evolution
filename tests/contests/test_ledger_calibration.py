"""Tests for the receipt-gated ledger and calibration scoring."""

import pytest

from evolution.contests.base_rates import ContestField, estimate_win_probability
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
        assert ledger.skill_history() == [(pytest.approx(0.05), False)]

    def test_screenshot_receipt_is_unconfirmed_and_excluded(self, ledger):
        ledger.record_prediction(_est("vr"), made_at=T0)
        ledger.record_submission("vr", "form_screenshot", "shots/form.png", submitted_at=T1)
        ledger.record_result("vr", "won", "https://x/winners", announced_at=T2)
        audit = ledger.audit()
        assert audit.resolved == []
        assert "vr" in audit.unconfirmed_submissions
        assert "form screenshot" in audit.excluded["vr"]
        assert audit.verified_wins == 0

    def test_hindsight_prediction_excluded(self, ledger):
        ledger.record_submission("c", "confirmation_email", "m", submitted_at=T1)
        ledger.record_result("c", "won", "https://x/w", announced_at=T1)
        ledger.record_prediction(_est("c"), made_at=T2)
        assert "before the result" in ledger.audit().excluded["c"]

    def test_latest_pre_announcement_prediction_used(self, ledger):
        ledger.record_prediction(_est("c", entrants=100), made_at=T0)
        ledger.record_prediction(_est("c", entrants=10), made_at=T1)
        ledger.record_submission("c", "platform_entry_url", "https://x/e", submitted_at=T1)
        ledger.record_result("c", "won", "https://x/w", announced_at=T2)
        (r,) = ledger.resolved()
        assert r.base_rate == pytest.approx(0.1)

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
    return ResolvedContest("c", p, base, True, won, "won" if won else "lost", T0)


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
