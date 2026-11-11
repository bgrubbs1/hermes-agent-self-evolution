"""Tests for blind judge panels and their validity gate."""

import json

import pytest

from evolution.contests.judges import (
    Artifact,
    JudgeSpec,
    LeakError,
    evaluate_panel,
    evidence_class,
    icc_2_1,
    make_blind_batch,
    parse_judge_response,
    run_panel,
    split_judges,
)

CRITERIA = ("functionality", "documentation")

# Hidden quality: what an ideal judge would see.
QUALITY = {
    "ours": 7.5,
    "w1": 9.0, "w2": 8.5, "w3": 8.0,
    "e1": 4.0, "e2": 5.0, "e3": 3.5,
    "ctl": 5.5,
}


def _artifacts():
    entry = Artifact("ours", "Sensor glove with firmware, BOM and docs.", "entry")
    refs = [Artifact(a, f"Project description number {i}.", r) for i, (a, r) in enumerate([
        ("w1", "winner"), ("w2", "winner"), ("w3", "winner"),
        ("e1", "entrant"), ("e2", "entrant"), ("e3", "entrant"),
        ("ctl", "control"),
    ])]
    return entry, refs


def _judges(families=("claude", "gpt", "gemini")):
    return [JudgeSpec(f"j{i}", fam) for i, fam in enumerate(families)]


def _batch(judges=None, seed=1):
    entry, refs = _artifacts()
    return make_blind_batch(
        entry, refs, judges or _judges(), rubric="Score like the contest judges.", criteria=CRITERIA, seed=seed
    )


def _oracle(noise_by_judge=None, invert=False):
    """Fake judge that scores hidden quality; looks items up by text, never by role."""
    text_to_id = {a.text: a.artifact_id for a in [_artifacts()[0], *_artifacts()[1]]}

    def judge(packet):
        out = {}
        bump = (noise_by_judge or {}).get(packet.judge_id, 0.0)
        for bid, text in packet.items:
            q = QUALITY[text_to_id[text]]
            q = 10 - q if invert else q
            out[bid] = {c: min(10.0, max(0.0, q + bump)) for c in packet.criteria}
        return json.dumps(out)

    return judge


class TestBlinding:
    def test_packets_hide_roles_and_ids(self):
        batch = _batch()
        for packet in batch.packets:
            prompt = packet.prompt()
            items_section = prompt.split("## Items")[1].lower()
            for word in ("ours", "winner", "entrant", "control", "w1", "e1", "ctl"):
                assert word not in items_section
            assert len({bid for bid, _ in packet.items}) == 8

    def test_each_judge_gets_independent_order(self):
        batch = _batch()
        orders = [tuple(text for _, text in p.items) for p in batch.packets]
        assert len(set(orders)) > 1

    def test_leaky_reference_refused(self):
        entry, refs = _artifacts()
        refs[0] = Artifact("w1", "Grand Prize winner! Our robot arm.", "winner")
        with pytest.raises(LeakError, match="w1"):
            make_blind_batch(entry, refs, _judges(), rubric="r", criteria=CRITERIA)

    def test_wont_is_not_a_leak(self):
        entry, refs = _artifacts()
        refs[3] = Artifact("e1", "It won't crash on bad input.", "entrant")
        make_blind_batch(entry, refs, _judges(), rubric="r", criteria=CRITERIA)

    def test_parse_rejects_missing_items_and_bad_scores(self):
        packet = _batch().packets[0]
        with pytest.raises(ValueError):
            parse_judge_response("{}", packet)
        bad = {bid: {c: 11 for c in CRITERIA} for bid, _ in packet.items}
        with pytest.raises(ValueError):
            parse_judge_response(json.dumps(bad), packet)


class TestPanelValidity:
    def test_good_multi_family_panel_is_valid(self):
        batch = _batch()
        scores = run_panel(batch, _oracle({"j0": 0.3, "j1": -0.2, "j2": 0.0}))
        res = evaluate_panel(batch, scores)
        assert res.valid, res.reasons
        assert res.evidence_class == "multi_family_panel"
        assert res.winner_vs_entrant_auc == 1.0
        assert res.control_detection_rate == 1.0
        assert res.beats_winners_fraction == 0.0  # 7.5 < every winner
        assert res.entry_percentile == pytest.approx(0.5)
        assert "VALID" in res.to_markdown()

    def test_panel_that_cannot_spot_winners_is_invalid(self):
        batch = _batch()
        res = evaluate_panel(batch, run_panel(batch, _oracle(invert=True)))
        assert not res.valid
        assert any("AUC" in r for r in res.reasons)

    def test_self_graded_panel_is_invalid_regardless_of_scores(self):
        judges = [JudgeSpec("j0", "claude", shares_author_context=True), *_judges()[1:]]
        batch = _batch(judges)
        res = evaluate_panel(batch, run_panel(batch, _oracle()))
        assert res.evidence_class == "self_graded"
        assert not res.valid
        assert "NOT VALID" in res.to_markdown()

    def test_too_few_references_invalid(self):
        entry, refs = _artifacts()
        batch = make_blind_batch(entry, refs[:1] + refs[3:4], _judges(), rubric="r", criteria=CRITERIA)
        text_quality = {a.text: QUALITY[a.artifact_id] for a in [entry, *refs]}
        scores = run_panel(batch, lambda p: {b: {c: text_quality[t] for c in CRITERIA} for b, t in p.items})
        res = evaluate_panel(batch, scores)
        assert not res.valid and any("need >= 3" in r for r in res.reasons)

    def test_single_family_warns(self):
        batch = _batch(_judges(("claude", "claude", "claude")))
        res = evaluate_panel(batch, run_panel(batch, _oracle({"j0": 0.2, "j1": -0.1})))
        assert res.evidence_class == "single_model_family"
        assert any("correlated" in w for w in res.warnings)


class TestHelpers:
    def test_icc_perfect_and_none(self):
        assert icc_2_1([[1, 1], [5, 5], [9, 9]]) == pytest.approx(1.0)
        assert icc_2_1([[1, 2]]) is None

    def test_icc_matches_shrout_fleiss_1979(self):
        data = [[9, 2, 5, 8], [6, 1, 3, 2], [8, 4, 6, 8], [7, 1, 2, 6], [10, 5, 6, 9], [6, 2, 4, 7]]
        assert icc_2_1(data) == pytest.approx(0.29, abs=0.005)

    def test_icc_noise_is_low(self):
        assert icc_2_1([[1, 9], [9, 1], [5, 5], [1, 9]]) < 0.2

    def test_evidence_class_order(self):
        assert evidence_class([JudgeSpec("h", "human"), JudgeSpec("h2", "human")]) == "human_panel"
        assert evidence_class([JudgeSpec("a", "x", used_in_optimization=True)]) == "optimization_contaminated"

    def test_split_judges_marks_optimisation_side(self):
        opt, hold = split_judges(_judges(("a", "b", "c", "d")), 2, seed=0)
        assert len(opt) == 2 and len(hold) == 2
        assert all(j.used_in_optimization for j in opt)
        assert not any(j.used_in_optimization for j in hold)
        assert {j.judge_id for j in opt}.isdisjoint({j.judge_id for j in hold})
