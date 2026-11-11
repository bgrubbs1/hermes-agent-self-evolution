"""Tests for discovery (against saved live responses), staged outcomes, backtests and LLM judges."""

import json
from pathlib import Path

import pytest
from click.testing import CliRunner

from evolution.contests.base_rates import Observation, UnknownFieldError, fit_skill_offset
from evolution.contests.cli import main
from evolution.contests.discovery import (
    ContestLead,
    DiscoveryError,
    discover,
    fetch_devpost,
    fetch_devpost_prizes,
    load_registry,
    parse_devpost,
    parse_devpost_prizes,
    parse_hackster,
)
from evolution.contests.judges import Artifact, JudgeSpec, PastContest, backtest_panel
from evolution.contests.llm_judges import judge_specs, make_llm_judge_fn, model_family

FIX = Path(__file__).parent / "fixtures"
DEVPOST = json.loads((FIX / "devpost_open_2026-11-11.json").read_text())
HACKSTER = (FIX / "hackster_contests_2026-11-11.html").read_text()
PAYPAL_PRIZES = (FIX / "devpost_paypal_prizes_2026-11-11.html").read_text()


class TestDevpost:
    def test_parses_live_payload(self):
        leads = parse_devpost(DEVPOST)
        assert [l.name for l in leads] == ["PayPal AI Hackathon", "Kiroween 2026",
                                          "Meta VR Start Developer Competition 2026"]
        paypal = leads[0]
        assert paypal.contest_id == "devpost-paypalaihackathon"
        assert paypal.prize_value_usd == 67500
        assert paypal.prize_slots is None  # prizes_counts is unreliable; slots come from the page
        assert paypal.registrations == 34603
        assert paypal.open and not paypal.invite_only

    def test_prize_page_gives_slots(self):
        prizes = parse_devpost_prizes(PAYPAL_PRIZES)
        assert len(prizes) == 17
        assert prizes[0] == {"title": "1st Place", "winners": 1, "value_usd": 12000.0}
        lead = parse_devpost(DEVPOST)[0]
        errors = fetch_devpost_prizes([lead], get=lambda url: (200, PAYPAL_PRIZES), delay=0)
        assert errors == {} and lead.prize_slots == sum(p["winners"] for p in prizes)
        assert "upper bound" in lead.notes[0]

    def test_prize_page_failures_are_reported(self):
        lead = parse_devpost(DEVPOST)[0]
        errors = fetch_devpost_prizes([lead], get=lambda url: (200, "<html></html>"), delay=0)
        assert "no prizes section" in errors[lead.contest_id] and lead.prize_slots is None

    def test_field_from_registrations_is_a_wide_range(self):
        lead = parse_devpost(DEVPOST)[0]
        lead.prize_slots = 21
        f = lead.contest_field()
        low, high = f.entrants_range
        assert high == 34603 and low == 1731  # 5% assumption only sets the floor
        assert "assumed" in f.entrants_source

    def test_submission_count_beats_registrations(self):
        lead = parse_devpost(DEVPOST)[0]
        lead.prize_slots = 21
        lead.submissions = 900
        assert lead.contest_field().entrants == 900

    def test_schema_drift_raises(self):
        with pytest.raises(DiscoveryError, match="schema changed"):
            parse_devpost({"challenges": []})
        with pytest.raises(DiscoveryError, match="missing keys"):
            parse_devpost({"hackathons": [{"title": "x"}]})

    def test_zero_results_is_an_error_not_an_answer(self):
        empty = json.dumps({"hackathons": [], "meta": {"total_count": 0}})
        with pytest.raises(DiscoveryError, match="0 open"):
            fetch_devpost(pages=1, get=lambda url: (200, empty))

    def test_http_failure_reports_status(self):
        with pytest.raises(DiscoveryError) as exc:
            fetch_devpost(pages=1, get=lambda url: (403, "denied"))
        assert exc.value.diagnostics["pages"][0]["status"] == 403

    def test_pagination_stops_at_total(self):
        calls = []

        def get(url):
            calls.append(url)
            return 200, json.dumps(DEVPOST | {"meta": {"total_count": 3}})

        assert len(fetch_devpost(pages=5, get=get)) == 3 and len(calls) == 1


class TestHackster:
    def test_parses_live_contests_block(self):
        leads = parse_hackster(HACKSTER)
        comps = [l for l in leads if not l.contest_id.startswith("hackster-giveaway")]
        assert len(comps) == 4
        au = next(l for l in comps if l.contest_id == "hackster-autodesk-university-2027-product")
        assert au.deadline == "2026-12-20T23:59:00-08:00"
        assert au.url == "https://www.hackster.io/contests/autodesk-university-2027-product"
        assert au.prize_slots is None  # listing shows only the headline prize
        with pytest.raises(ValueError, match="prize slot count unknown"):
            au.contest_field()

    def test_merit_giveaway_is_skill_not_sweepstakes(self):
        give = [l for l in parse_hackster(HACKSTER) if l.contest_id.startswith("hackster-giveaway")]
        assert give and give[0].kind == "skill"

    def test_missing_block_raises(self):
        with pytest.raises(DiscoveryError, match="not found"):
            parse_hackster("<html>redesigned page</html>")


class TestRegistryAndDiscover:
    def test_registry_and_partial_failure(self, tmp_path):
        reg = tmp_path / "registry.json"
        reg.write_text(json.dumps([{"name": "element14 Design Challenge", "url": "https://community.element14.com/x",
                                    "kind": "application", "prize_slots": 10, "registrations": 120}]))
        leads, errors = discover(["devpost"], registry=reg, get=lambda url: (403, ""))
        assert "devpost" in errors and len(leads) == 1
        assert leads[0].contest_field().kind == "application"

    def test_registry_overrides_discovered_fields(self, tmp_path):
        reg = tmp_path / "registry.json"
        reg.write_text(json.dumps([{"contest_id": "devpost-paypalaihackathon", "prize_slots": 21}]))
        leads, errors = discover(["devpost"], registry=reg,
                                 get=lambda url: (200, json.dumps(DEVPOST | {"meta": {"total_count": 3}})))
        paypal = next(l for l in leads if l.contest_id == "devpost-paypalaihackathon")
        assert paypal.prize_slots == 21 and paypal.registrations == 34603 and errors == {}
        assert "overridden from registry" in paypal.notes[-1]

    def test_registry_unmatched_entry_needs_name(self, tmp_path):
        reg = tmp_path / "registry.json"
        reg.write_text(json.dumps([{"contest_id": "devpost-unknown", "prize_slots": 3}]))
        with pytest.raises(ValueError, match="name and url"):
            discover(["devpost"], registry=reg, get=lambda url: (403, ""))

    def test_bot_challenge_is_reported_not_bypassed(self, monkeypatch):
        import io
        import urllib.error
        import urllib.request
        from email.message import Message


        headers = Message()
        headers["Cf-Mitigated"] = "challenge"

        def deny(req, timeout=0):
            raise urllib.error.HTTPError(req.full_url, 403, "Forbidden", headers, io.BytesIO(b"Just a moment"))

        monkeypatch.setattr(urllib.request, "urlopen", deny)
        lead = parse_devpost(DEVPOST)[0]
        errors = fetch_devpost_prizes([lead], delay=0)
        assert "bot challenge (not bypassed)" in errors[lead.contest_id]

    def test_registry_rejects_unknown_keys(self, tmp_path):
        reg = tmp_path / "r.json"
        reg.write_text(json.dumps([{"name": "x", "url": "u", "win_probability": 0.95}]))
        with pytest.raises(ValueError, match="unknown keys"):
            load_registry(reg)

    def test_nothing_anywhere_raises(self):
        with pytest.raises(DiscoveryError):
            discover(["devpost", "hackster"], get=lambda url: (500, ""))

    def test_lead_without_any_field_raises(self):
        with pytest.raises(UnknownFieldError):
            ContestLead("s", "c", "n", "u", prize_slots=3).contest_field()

    def test_cli_discover_offline(self, tmp_path, monkeypatch):
        import evolution.contests.discovery as d

        monkeypatch.setattr(d, "http_get", lambda url, timeout=30.0: (200, json.dumps(DEVPOST)))
        monkeypatch.setitem(d.SOURCES, "devpost", lambda get=None: d.fetch_devpost(pages=1, get=d.http_get))
        r = CliRunner().invoke(main, ["discover", "--source", "devpost"])
        assert r.exit_code == 0, r.output
        assert "devpost-paypalaihackathon" in r.output and "slots=?" in r.output


class TestObservations:
    def test_binary_equals_single_threshold(self):
        a = fit_skill_offset([(0.1, True), (0.1, False)])
        b = fit_skill_offset([Observation.binary(0.1, True), Observation.binary(0.1, False)])
        assert a.mean == pytest.approx(b.mean)

    def test_staged_result_is_more_informative(self):
        lost_only = fit_skill_offset([Observation.binary(3 / 200, False)] * 3)
        lost_as_finalist = fit_skill_offset([Observation.staged(200, [(20, True), (3, False)])] * 3)
        assert lost_as_finalist.mean > lost_only.mean
        lo1, hi1 = lost_only.interval()
        lo2, hi2 = lost_as_finalist.interval()
        assert (hi2 - lo2) < (hi1 - lo1)

    def test_rank_observation(self):
        top = fit_skill_offset([Observation.from_rank(5, 1000)] * 3)
        mid = fit_skill_offset([Observation.from_rank(500, 1000)] * 3)
        assert top.mean > mid.mean

    def test_invalid_observations(self):
        with pytest.raises(ValueError):
            Observation((0.1, 0.2), 1)  # not decreasing
        with pytest.raises(ValueError):
            Observation.staged(100, [(20, False), (3, True)])  # narrower without wider
        with pytest.raises(ValueError):
            Observation.from_rank(0, 10)

    def test_staged_collapses_capped_stages(self):
        obs = Observation.staged(10, [(50, True), (20, True), (2, False)])
        assert obs.thresholds == (1.0, 0.2) and obs.reached == 1


QUALITY = {"w": 8.5, "e": 4.0}


def _past(n):
    out = []
    for i in range(n):
        arts = [Artifact(f"c{i}-w{j}", f"Contest {i} project alpha {j}", "winner") for j in range(2)]
        arts += [Artifact(f"c{i}-e{j}", f"Contest {i} project beta {j}", "entrant") for j in range(4)]
        out.append(PastContest(f"c{i}", "Judge like the sponsor.", tuple(arts)))
    return out


def _oracle(invert=False):
    def judge(packet):
        res = {}
        for bid, text in packet.items:
            q = QUALITY["w"] if "alpha" in text else QUALITY["e"]
            q = 10 - q if invert else q
            res[bid] = {c: q for c in packet.criteria}
        return res
    return judge


class TestBacktest:
    JUDGES = [JudgeSpec("a", "claude"), JudgeSpec("b", "gpt"), JudgeSpec("c", "gemini")]

    def test_good_panel_validates(self):
        res = backtest_panel(_past(6), self.JUDGES, _oracle(), criteria=("overall",))
        assert res.valid and res.mean_auc == 1.0
        assert res.top1_hit_rate == 1.0 and res.top1_chance == pytest.approx(2 / 6)
        assert "VALID" in res.to_markdown()

    def test_bad_panel_fails(self):
        res = backtest_panel(_past(6), self.JUDGES, _oracle(invert=True), criteria=("overall",))
        assert not res.valid and res.mean_auc == 0.0

    def test_too_few_contests(self):
        res = backtest_panel(_past(2), self.JUDGES, _oracle(), criteria=("overall",))
        assert not res.valid and any("past contests" in r for r in res.reasons)

    def test_past_contest_requires_both_roles(self):
        with pytest.raises(ValueError):
            PastContest("x", "r", (Artifact("w", "t", "winner"),))


class TestLLMJudges:
    def test_families(self):
        assert model_family("anthropic/claude-sonnet-5-5") == "claude"
        assert model_family("openai/gpt-5") == "gpt"
        assert model_family("gemini/gemini-3-pro") == "gemini"
        specs = judge_specs({"j1": "anthropic/claude-sonnet-5-5", "j2": "openai/gpt-5"})
        assert {s.model_family for s in specs} == {"claude", "gpt"}
        assert not any(s.shares_author_context for s in specs)

    def test_each_judge_gets_only_its_packet(self):
        seen = {}

        def factory(model):
            def call(prompt):
                seen.setdefault(model, []).append(prompt)
                return "{}"
            return call

        fn = make_llm_judge_fn({"j1": "m1", "j2": "m2"}, lm_factory=factory)
        from evolution.contests.judges import JudgePacket

        fn(JudgePacket("j1", "rubric", ("x",), (("id1", "text one"),)))
        fn(JudgePacket("j2", "rubric", ("x",), (("id2", "text two"),)))
        assert len(seen["m1"]) == 1 and "text one" in seen["m1"][0] and "text two" not in seen["m1"][0]
        with pytest.raises(KeyError):
            fn(JudgePacket("j3", "rubric", ("x",), ()))
