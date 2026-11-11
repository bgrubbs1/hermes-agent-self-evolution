"""Tests for field-size base rates and the learned skill offset."""

import math

import pytest

from evolution.contests.base_rates import (
    ContestField,
    UnknownFieldError,
    estimate_win_probability,
    expected_value_per_hour,
    fit_skill_offset,
)


class TestContestField:
    def test_unknown_field_is_refused(self):
        with pytest.raises(UnknownFieldError):
            ContestField("x", prizes=3)

    def test_base_rate_is_prizes_over_entrants(self):
        assert ContestField("x", prizes=3, entrants=1200).base_rate() == pytest.approx(3 / 1200)

    def test_base_rate_caps_at_one(self):
        assert ContestField("x", prizes=10, entrants=4).base_rate() == 1.0

    def test_range_integrates_log_uniform(self):
        f = ContestField("x", prizes=1, entrants_range=(10, 1000))
        grid = f.entrant_grid()
        assert grid[0] == pytest.approx(10) and grid[-1] == pytest.approx(1000)
        assert grid[len(grid) // 2] == pytest.approx(100)  # geometric midpoint

    @pytest.mark.parametrize("kwargs", [
        {"prizes": 0, "entrants": 10},
        {"prizes": 1, "entrants": 0},
        {"prizes": 1, "entrants_range": (50, 10)},
    ])
    def test_invalid_inputs(self, kwargs):
        with pytest.raises(ValueError):
            ContestField("x", **kwargs)


class TestSkillOffset:
    def test_empty_history_is_the_prior(self):
        post = fit_skill_offset([])
        assert post.n_obs == 0
        assert post.mean == pytest.approx(-0.5, abs=1e-3)  # centred at -sd^2/2

    def test_wins_above_base_rate_raise_offset(self):
        post = fit_skill_offset([(0.05, True)] * 6 + [(0.05, False)] * 4)
        assert post.mean > 1.5
        assert post.odds_multiplier > 4

    def test_losses_at_base_rate_lower_offset(self):
        post = fit_skill_offset([(0.2, False)] * 15)
        assert post.mean < -0.5

    def test_interval_narrows_with_data(self):
        wide = fit_skill_offset([(0.1, True), (0.1, False)])
        narrow = fit_skill_offset([(0.1, True)] * 10 + [(0.1, False)] * 90)
        w_lo, w_hi = wide.interval()
        n_lo, n_hi = narrow.interval()
        assert (n_hi - n_lo) < (w_hi - w_lo)
        # 10% observed at a 10% base rate: an average entrant
        assert abs(narrow.mean) < 0.4

    def test_weights_normalised(self):
        assert sum(fit_skill_offset([(0.3, True)]).weights) == pytest.approx(1.0)


class TestEstimate:
    def test_no_history_mean_matches_base_rate_for_small_b(self):
        est = estimate_win_probability(ContestField("x", prizes=3, entrants=1200))
        assert est.p_mean == pytest.approx(3 / 1200, rel=0.05)
        assert est.p_low < est.base_rate < est.p_high
        assert est.evidence_class == "base_rate_only"

    def test_no_history_never_reports_certainty(self):
        est = estimate_win_probability(ContestField("pcbway", prizes=5, entrants=200))
        assert est.p_high < 0.2

    def test_history_moves_estimate(self):
        skill = fit_skill_offset([(0.02, True)] * 5 + [(0.02, False)] * 5)
        est = estimate_win_probability(ContestField("x", prizes=1, entrants=50), skill)
        assert est.p_mean > 0.1
        assert est.evidence_class.startswith("base_rate+history(n=10,wins=5)")

    def test_sweepstakes_ignore_skill(self):
        skill = fit_skill_offset([(0.02, True)] * 9)
        est = estimate_win_probability(ContestField("sw", prizes=1, entrants=10_000, kind="sweepstakes"), skill)
        assert est.p_mean == pytest.approx(1e-4)
        assert est.evidence_class == "chance_only"

    def test_manual_shift_needs_reason_and_is_labelled(self):
        f = ContestField("x", prizes=1, entrants=100)
        with pytest.raises(ValueError):
            estimate_win_probability(f, manual_shift=1.0)
        est = estimate_win_probability(f, manual_shift=1.0, manual_reason="only 4 complete entries visible")
        assert est.evidence_class.endswith("+judgment_shift")
        assert est.p_mean > estimate_win_probability(f).p_mean

    def test_manual_shift_refused_for_chance(self):
        with pytest.raises(ValueError):
            estimate_win_probability(
                ContestField("sw", prizes=1, entrants=100, kind="sweepstakes"), manual_shift=1, manual_reason="x"
            )

    def test_ev_per_hour(self):
        est = estimate_win_probability(ContestField("x", prizes=1, entrants=100))
        ev = expected_value_per_hour(est, prize_value=1000, hours=10)
        assert ev["ev_per_hour"] == pytest.approx(100 * est.p_mean)
        assert ev["ev_per_hour_low"] <= ev["ev_per_hour"]
        assert math.isfinite(ev["ev_per_hour"])
