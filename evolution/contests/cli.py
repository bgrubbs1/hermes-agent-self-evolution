"""Command line for the contest ledger, so agents use tools instead of typing numbers.

    python -m evolution.contests.cli predict  --ledger L --contest-id X --prizes 3 --entrants 1200
    python -m evolution.contests.cli submit   --ledger L --contest-id X --receipt-kind confirmation_email --receipt-ref <msg-id>
    python -m evolution.contests.cli result   --ledger L --contest-id X --outcome lost --source <url> --announced-at <iso>
    python -m evolution.contests.cli report   --ledger L
    python -m evolution.contests.cli lint     FILE...
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import click

from evolution.contests.base_rates import (
    ContestField,
    UnknownFieldError,
    estimate_win_probability,
    fit_skill_offset,
)
from evolution.contests.calibration import calibration_report
from evolution.contests.ledger import RECEIPT_KINDS, OUTCOMES, Ledger, LedgerError, sha256_file
from evolution.contests.robustness import lint_claims

LEDGER_OPTION = click.option(
    "--ledger", "ledger_path", required=True, type=click.Path(dir_okay=False), help="JSONL ledger file"
)


@click.group()
def main() -> None:
    """Contest calibration ledger."""


@main.command()
@LEDGER_OPTION
@click.option("--contest-id", required=True)
@click.option("--prizes", required=True, type=int, help="Prize slots you are eligible for")
@click.option("--entrants", type=int, default=None, help="Total competing entries incl. yours")
@click.option("--entrants-range", type=(int, int), default=None, help="LOW HIGH estimate if count unknown")
@click.option("--entrants-source", default="", help="Where the count came from (URL, counter, past edition)")
@click.option("--sweepstakes", is_flag=True, help="Random draw: skill history ignored")
@click.option("--deadline", default=None, help="ISO-8601 with timezone")
@click.option("--shift", type=float, default=0.0, help="Judgment log-odds shift (needs --reason)")
@click.option("--reason", default="", help="Written reason for --shift")
@click.option("--dry-run", is_flag=True, help="Print the estimate without recording it")
def predict(ledger_path, contest_id, prizes, entrants, entrants_range, entrants_source, sweepstakes,
            deadline, shift, reason, dry_run):
    """Estimate P(win) from the field size and our resolved history, and record it."""
    ledger = Ledger(ledger_path)
    try:
        contest = ContestField(
            contest_id=contest_id,
            prizes=prizes,
            entrants=entrants,
            entrants_range=entrants_range,
            skill_based=not sweepstakes,
            entrants_source=entrants_source,
        )
        skill = fit_skill_offset(ledger.skill_history())
        est = estimate_win_probability(contest, skill, manual_shift=shift, manual_reason=reason)
    except (UnknownFieldError, ValueError) as exc:
        raise click.ClickException(str(exc)) from exc
    click.echo(est.summary())
    for note in est.notes:
        click.echo(f"  - {note}")
    if not dry_run:
        try:
            ledger.record_prediction(est, deadline=deadline)
        except LedgerError as exc:
            raise click.ClickException(str(exc)) from exc
        click.echo(f"recorded prediction in {ledger_path}")


@main.command()
@LEDGER_OPTION
@click.option("--contest-id", required=True)
@click.option("--receipt-kind", required=True, type=click.Choice(sorted(RECEIPT_KINDS)))
@click.option("--receipt-ref", required=True, help="Message id, confirmation URL, or file path")
@click.option("--receipt-file", type=click.Path(exists=True, dir_okay=False), default=None,
              help="Saved receipt (email .eml, PDF, PNG); its sha256 is stored")
@click.option("--submitted-at", default=None, help="ISO-8601 with timezone (default now)")
@click.option("--notes", default="")
def submit(ledger_path, contest_id, receipt_kind, receipt_ref, receipt_file, submitted_at, notes):
    """Record a submission. A receipt is mandatory; screenshots count as unconfirmed."""
    digest = sha256_file(Path(receipt_file)) if receipt_file else None
    try:
        rec = Ledger(ledger_path).record_submission(
            contest_id, receipt_kind, receipt_ref, receipt_sha256=digest, submitted_at=submitted_at, notes=notes
        )
    except LedgerError as exc:
        raise click.ClickException(str(exc)) from exc
    status = "verified" if rec["strong_receipt"] else "UNCONFIRMED (weak receipt)"
    click.echo(f"recorded submission for {contest_id}: {status}")


@main.command()
@LEDGER_OPTION
@click.option("--contest-id", required=True)
@click.option("--outcome", required=True, type=click.Choice(sorted(OUTCOMES)))
@click.option("--source", default="", help="Winners announcement URL")
@click.option("--announced-at", required=True, help="When results became public (ISO-8601 with timezone)")
@click.option("--notes", default="")
def result(ledger_path, contest_id, outcome, source, announced_at, notes):
    """Record a published result. Record losses too."""
    try:
        Ledger(ledger_path).record_result(contest_id, outcome, source, announced_at=announced_at, notes=notes)
    except LedgerError as exc:
        raise click.ClickException(str(exc)) from exc
    click.echo(f"recorded {outcome} for {contest_id}")


@main.command()
@LEDGER_OPTION
@click.option("--json", "as_json", is_flag=True)
def report(ledger_path, as_json):
    """Ledger audit plus predicted-vs-actual calibration against the base-rate comparator."""
    ledger = Ledger(ledger_path)
    try:
        audit = ledger.audit()
    except LedgerError as exc:
        raise click.ClickException(str(exc)) from exc
    cal = calibration_report(audit.resolved)
    skill = fit_skill_offset([(r.base_rate, r.won) for r in audit.resolved if r.skill_based])
    lo, hi = skill.interval()
    if as_json:
        click.echo(json.dumps({
            "audit": audit.to_dict(),
            "calibration": cal.to_dict(),
            "skill_offset": {"mean": skill.mean, "interval_90": [lo, hi], "n": skill.n_obs},
        }, indent=2))
        return
    a = audit.to_dict()
    click.echo("## Ledger audit\n")
    click.echo(f"- Verified submissions: {a['verified_submissions']}; verified wins: {a['verified_wins']}")
    click.echo(f"- Pending results: {', '.join(a['pending']) or 'none'}")
    click.echo(f"- Unconfirmed submissions (weak receipt): {', '.join(a['unconfirmed_submissions']) or 'none'}")
    for cid, why in a["excluded"].items():
        click.echo(f"- Excluded {cid}: {why}")
    click.echo(
        f"- Skill offset: {skill.mean:+.2f} log-odds (90% {lo:+.2f} to {hi:+.2f}; n={skill.n_obs}); "
        "0 = average entrant\n"
    )
    click.echo(cal.to_markdown())


@main.command()
@click.argument("files", nargs=-1, type=click.Path(exists=True, dir_okay=False))
def lint(files):
    """Flag win guarantees ("ensures a win", "100% win probability") in text files."""
    sources = [(f, Path(f).read_text(encoding="utf-8")) for f in files] or [("<stdin>", sys.stdin.read())]
    found = 0
    for name, text in sources:
        for issue in lint_claims(text):
            found += 1
            click.echo(f"{name}:{issue['line']}: {issue['text']!r} -> {issue['why']}")
    if found:
        sys.exit(1)


if __name__ == "__main__":
    main()
