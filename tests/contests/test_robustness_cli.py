"""Tests for reliability claims, the overclaim linter, and the CLI."""

import json

import pytest
from click.testing import CliRunner

from evolution.contests.cli import main
from evolution.contests.robustness import lint_claims, summarize_trials


class TestTrials:
    def test_claim_has_interval(self):
        s = summarize_trials([True] * 27 + [False] * 3)
        assert s.passed == 27 and s.trials == 30
        assert s.ci_low < 0.9 < s.ci_high
        assert "27/30" in s.claim("fresh adversarial prompts")

    def test_small_sample_flagged(self):
        assert "Small sample" in summarize_trials([True, True, True]).claim("prompts")

    def test_empty_rejected(self):
        with pytest.raises(ValueError):
            summarize_trials([])


class TestLint:
    @pytest.mark.parametrize("text", [
        "This pipeline ensures we win every contest.",
        "PCBWay: 100% win probability",
        "target win rate: 95%",
        "We are certain to win.",
        "Guaranteed first place.",
    ])
    def test_overclaims_flagged(self, text):
        assert lint_claims(text)

    @pytest.mark.parametrize("text", [
        "Nothing ensures wins.",
        "This bot cannot guarantee a win.",
        'The old scorer printed "PCBWay: 100% win probability" from mock data.',
        "P(win) 2.4% (90% interval 0.6%-7.9%).",
        "The checklist ensures every entry includes a license file.",
    ])
    def test_honest_text_passes(self, text):
        assert lint_claims(text) == []


class TestCli:
    def test_full_cycle(self, tmp_path):
        runner = CliRunner()
        led = str(tmp_path / "l.jsonl")
        r = runner.invoke(main, ["predict", "--ledger", led, "--contest-id", "hk", "--prizes", "3",
                                 "--entrants", "1200", "--entrants-source", "platform counter"])
        assert r.exit_code == 0, r.output
        assert "base_rate_only" in r.output

        receipt = tmp_path / "receipt.eml"
        receipt.write_text("Thanks, your entry was received.")
        r = runner.invoke(main, ["submit", "--ledger", led, "--contest-id", "hk", "--receipt-kind",
                                 "confirmation_email", "--receipt-ref", "msg-123", "--receipt-file", str(receipt)])
        assert r.exit_code == 0 and "verified" in r.output

        r = runner.invoke(main, ["result", "--ledger", led, "--contest-id", "hk", "--outcome", "lost",
                                 "--source", "https://example.com/winners", "--announced-at", "2099-01-01T00:00:00Z"])
        assert r.exit_code == 0, r.output

        r = runner.invoke(main, ["report", "--ledger", led, "--json"])
        data = json.loads(r.output)
        assert data["audit"]["resolved"] == 1
        assert data["calibration"]["n"] == 1
        sub = [json.loads(l) for l in open(led)][1]
        assert len(sub["receipt_sha256"]) == 64

    def test_predict_without_field_fails_cleanly(self, tmp_path):
        r = CliRunner().invoke(main, ["predict", "--ledger", str(tmp_path / "l"), "--contest-id", "x",
                                      "--prizes", "3"])
        assert r.exit_code != 0 and "no entrant count" in r.output

    def test_lint_exit_code(self, tmp_path):
        bad = tmp_path / "status.md"
        bad.write_text("ok\nThis bot ensures a 100% win rate.\n")
        r = CliRunner().invoke(main, ["lint", str(bad)])
        assert r.exit_code == 1 and "status.md:2" in r.output
