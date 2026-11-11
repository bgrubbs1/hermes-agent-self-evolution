"""Command line for contest discovery and the ledger, so agents use tools instead of typing numbers.

    python -m evolution.contests.cli discover --registry contests/registry.json
    python -m evolution.contests.cli predict  --ledger L --contest-id X --prizes 3 --entrants 1200
    python -m evolution.contests.cli submit   --ledger L --contest-id X --receipt-kind confirmation_email --receipt-ref <msg-id>
    python -m evolution.contests.cli attempt  --ledger L --contest-id X --evidence "form filled, no confirmation"
    python -m evolution.contests.cli result   --ledger L --contest-id X --outcome lost --source <url> --announced-at <iso>
    python -m evolution.contests.cli import   --ledger L history.json
    python -m evolution.contests.cli report   --ledger L
    python -m evolution.contests.cli lint     FILE...
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import click

from evolution.contests.base_rates import (
    KINDS,
    ContestField,
    UnknownFieldError,
    estimate_win_probability,
    fit_skill_offset,
)
from evolution.contests.calibration import calibration_report
from evolution.contests.discovery import DiscoveryError, discover as run_discovery, fetch_devpost_prizes
from evolution.contests.ledger import OUTCOMES, RECEIPT_KINDS, Ledger, LedgerError, sha256_file
from evolution.contests.robustness import lint_claims

LEDGER_OPTION = click.option(
    "--ledger", "ledger_path", required=True, type=click.Path(dir_okay=False), help="JSONL ledger file"
)
KIND_OPTION = click.option("--kind", type=click.Choice(KINDS), default="skill", show_default=True)


@click.group()
def main() -> None:
    """Contest discovery, odds, receipts and calibration."""


@main.command()
@click.option("--source", "sources", multiple=True, default=("devpost", "hackster"), show_default=True,
              type=click.Choice(["devpost", "hackster"]))
@click.option("--registry", type=click.Path(exists=True, dir_okay=False), default=None,
              help="Hand-maintained JSON list for sites that cannot be scraped")
@click.option("--prizes", "prize_pages", type=int, default=0, show_default=True,
              help="Read up to N Devpost hackathon pages (1 request/second) to count prize slots")
@click.option("--json", "as_json", is_flag=True)
def discover(sources, registry, prize_pages, as_json):
    """List open contests. Zero results is reported as an error, never as 'no contests'."""
    try:
        leads, errors = run_discovery(sources, registry=registry)
    except DiscoveryError as exc:
        raise click.ClickException(str(exc)) from exc
    if prize_pages:
        for cid, err in fetch_devpost_prizes(leads, limit=prize_pages).items():
            errors[f"prizes:{cid}"] = err
    if as_json:
        click.echo(json.dumps({"leads": [lead.to_dict() for lead in leads], "errors": errors}, indent=2))
    else:
        for lead in leads:
            slots = lead.prize_slots if lead.prize_slots is not None else "?"
            reg = lead.registrations if lead.registrations is not None else "?"
            value = f"${lead.prize_value_usd:,.0f}" if lead.prize_value_usd else "?"
            when = lead.deadline or lead.deadline_text or "?"
            click.echo(f"{lead.contest_id}\t{lead.kind}\tslots={slots}\tregistered={reg}\tprize={value}\t"
                       f"deadline={when}\t{lead.url}")
        for name, err in errors.items():
            click.echo(f"SOURCE FAILED {name}: {err}", err=True)
    if errors:
        sys.exit(2)


@main.command()
@LEDGER_OPTION
@click.option("--contest-id", required=True)
@click.option("--prizes", required=True, type=int, help="Prize (or selection) slots you are eligible for")
@click.option("--entrants", type=int, default=None, help="Total competing entries incl. yours")
@click.option("--entrants-range", type=(int, int), default=None, help="LOW HIGH estimate if count unknown")
@click.option("--entrants-source", default="", help="Where the count came from (URL, counter, past edition)")
@KIND_OPTION
@click.option("--sweepstakes", is_flag=True, help="Shorthand for --kind sweepstakes")
@click.option("--deadline", default=None, help="ISO-8601 with timezone")
@click.option("--shift", type=float, default=0.0, help="Judgment log-odds shift (needs --reason)")
@click.option("--reason", default="", help="Written reason for --shift")
@click.option("--dry-run", is_flag=True, help="Print the estimate without recording it")
def predict(ledger_path, contest_id, prizes, entrants, entrants_range, entrants_source, kind, sweepstakes,
            deadline, shift, reason, dry_run):
    """Estimate P(win) from the field size and our resolved history, and record it."""
    ledger = Ledger(ledger_path)
    kind = "sweepstakes" if sweepstakes else kind
    try:
        contest = ContestField(
            contest_id=contest_id,
            prizes=prizes,
            entrants=entrants,
            entrants_range=entrants_range,
            kind=kind,
            entrants_source=entrants_source,
        )
        skill = fit_skill_offset(ledger.skill_history(kind))
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
@KIND_OPTION
@click.option("--submitted-at", default=None, help="ISO-8601 with timezone, or 'unknown' for old entries")
@click.option("--notes", default="")
def submit(ledger_path, contest_id, receipt_kind, receipt_ref, receipt_file, kind, submitted_at, notes):
    """Record a submission. A receipt is mandatory; screenshots count as unconfirmed."""
    digest = sha256_file(Path(receipt_file)) if receipt_file else None
    try:
        rec = Ledger(ledger_path).record_submission(
            contest_id, receipt_kind, receipt_ref, kind=kind, receipt_sha256=digest,
            submitted_at=submitted_at, notes=notes,
        )
    except LedgerError as exc:
        raise click.ClickException(str(exc)) from exc
    status = "verified" if rec["strong_receipt"] else "UNCONFIRMED (weak receipt)"
    click.echo(f"recorded submission for {contest_id}: {status}")


@main.command()
@LEDGER_OPTION
@click.option("--contest-id", required=True)
@click.option("--evidence", required=True, help="What shows the attempt happened (no receipt exists)")
@KIND_OPTION
@click.option("--attempted-at", default=None, help="ISO-8601 with timezone, or 'unknown'")
def attempt(ledger_path, contest_id, evidence, kind, attempted_at):
    """Log an entry that has no receipt. It is shown in audits and never counted."""
    try:
        Ledger(ledger_path).record_attempt(contest_id, evidence, kind=kind, attempted_at=attempted_at)
    except LedgerError as exc:
        raise click.ClickException(str(exc)) from exc
    click.echo(f"recorded unreceipted attempt for {contest_id}")


def _parse_milestone(text: str) -> dict:
    try:
        name, slots, reached = text.split(":")
        return {"name": name, "slots": int(slots), "reached": reached.lower() in ("yes", "true", "1")}
    except ValueError as exc:
        raise click.BadParameter(f"milestone {text!r} must be NAME:SLOTS:yes|no") from exc


@main.command()
@LEDGER_OPTION
@click.option("--contest-id", required=True)
@click.option("--outcome", required=True, type=click.Choice(sorted(OUTCOMES)))
@click.option("--source", default="", help="Winners announcement URL or result message id")
@click.option("--announced-at", required=True, help="When results became public (ISO-8601 with timezone)")
@click.option("--entrants", type=int, default=None, help="Final number of competing entries incl. ours")
@click.option("--prizes", type=int, default=None, help="Final number of prize (or selection) slots")
@click.option("--milestone", "milestones", multiple=True, help="Wider stage NAME:SLOTS:yes|no, e.g. finalist:20:yes")
@click.option("--rank", type=int, default=None, help="Full-leaderboard rank (needs --entrants)")
@click.option("--notes", default="")
def result(ledger_path, contest_id, outcome, source, announced_at, entrants, prizes, milestones, rank, notes):
    """Record a published result. Record losses too; field size and stages make each one count for more."""
    try:
        Ledger(ledger_path).record_result(
            contest_id, outcome, source, announced_at=announced_at, entrants=entrants, prizes=prizes,
            milestones=[_parse_milestone(m) for m in milestones], rank=rank, notes=notes,
        )
    except LedgerError as exc:
        raise click.ClickException(str(exc)) from exc
    click.echo(f"recorded {outcome} for {contest_id}")


@main.command(name="import")
@LEDGER_OPTION
@click.argument("history", type=click.Path(exists=True, dir_okay=False))
def import_history(ledger_path, history):
    """Append events from a JSON list of {"type": submission|attempt|result|prediction-less events}."""
    ledger = Ledger(ledger_path)
    records = json.loads(Path(history).read_text(encoding="utf-8"))
    handlers = {
        "submission": lambda r: ledger.record_submission(
            r["contest_id"], r["receipt_kind"], r["receipt_ref"], kind=r.get("kind", "skill"),
            submitted_at=r.get("submitted_at"), notes=r.get("notes", "")),
        "attempt": lambda r: ledger.record_attempt(
            r["contest_id"], r["evidence"], kind=r.get("kind", "skill"), attempted_at=r.get("attempted_at")),
        "result": lambda r: ledger.record_result(
            r["contest_id"], r["outcome"], r.get("source", ""), announced_at=r["announced_at"],
            entrants=r.get("entrants"), prizes=r.get("prizes"), milestones=r.get("milestones"),
            rank=r.get("rank"), notes=r.get("notes", "")),
    }
    count = 0
    for r in records:
        if r.get("type") not in handlers:
            raise click.ClickException(f"unsupported record type {r.get('type')!r} (predictions must be live)")
        try:
            handlers[r["type"]](r)
        except (KeyError, LedgerError) as exc:
            raise click.ClickException(f"{r.get('contest_id')}: {exc}") from exc
        count += 1
    click.echo(f"imported {count} events into {ledger_path}")


@main.command()
@LEDGER_OPTION
@click.option("--json", "as_json", is_flag=True)
def report(ledger_path, as_json):
    """Ledger audit, learned skill offsets, and forecast calibration against the base-rate comparator."""
    ledger = Ledger(ledger_path)
    try:
        audit = ledger.audit()
    except LedgerError as exc:
        raise click.ClickException(str(exc)) from exc
    cal = calibration_report(audit.resolved)
    skills = {}
    for kind in ("skill", "application"):
        post = fit_skill_offset([r.observation for r in audit.resolved if r.kind == kind])
        lo, hi = post.interval()
        skills[kind] = {"mean": post.mean, "interval_90": [lo, hi], "n": post.n_obs}
    if as_json:
        click.echo(json.dumps({"audit": audit.to_dict(), "calibration": cal.to_dict(), "skill_offset": skills},
                              indent=2))
        return
    a = audit.to_dict()
    click.echo("## Ledger audit\n")
    click.echo(f"- Verified submissions: {a['verified_submissions']}; verified wins/selections: {a['verified_wins']}")
    click.echo(f"- Resolved: {a['resolved']} (with a pre-announcement forecast: {a['resolved_with_forecast']})")
    click.echo(f"- Pending results: {', '.join(a['pending']) or 'none'}")
    click.echo(f"- Unconfirmed submissions (weak receipt): {', '.join(a['unconfirmed_submissions']) or 'none'}")
    click.echo(f"- Attempts without any receipt: {', '.join(a['unreceipted_attempts']) or 'none'}")
    for cid, why in a["excluded"].items():
        click.echo(f"- Excluded {cid}: {why}")
    for kind, s in skills.items():
        click.echo(
            f"- Skill offset ({kind}): {s['mean']:+.2f} log-odds "
            f"(90% {s['interval_90'][0]:+.2f} to {s['interval_90'][1]:+.2f}; n={s['n']}); 0 = average entrant"
        )
    click.echo("")
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
