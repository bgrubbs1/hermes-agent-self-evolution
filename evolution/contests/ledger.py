"""Append-only, receipt-gated contest ledger: predictions, submissions, results.

A contest counts toward calibration only if all of these hold:

1. a prediction was recorded *before* the result was announced (no hindsight);
2. the submission has a strong receipt (confirmation email/page, platform
   entry URL or API record). A screenshot of a filled form shows that a form
   was filled, not that the entry was accepted, so it stays "unconfirmed";
3. a result was recorded with a public source, and results do not conflict.

Losses are recorded the same way as wins.  A ledger that only hears about wins
can never be calibrated.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from evolution.contests.base_rates import WinEstimate

STRONG_RECEIPTS = frozenset({"confirmation_email", "confirmation_page", "platform_entry_url", "platform_api"})
WEAK_RECEIPTS = frozenset({"form_screenshot"})
RECEIPT_KINDS = STRONG_RECEIPTS | WEAK_RECEIPTS

# outcome -> 1 (took a prize), 0 (did not), None (excluded)
OUTCOMES: dict[str, Optional[int]] = {
    "won": 1,
    "placed": 1,
    "lost": 0,
    "disqualified": 0,
    "withdrawn": None,
}


class LedgerError(ValueError):
    pass


def parse_time(value: str) -> datetime:
    """Parse an ISO-8601 timestamp; timezone is mandatory."""
    text = value.strip()
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError as exc:
        raise LedgerError(f"bad timestamp {value!r}: {exc}") from exc
    if parsed.tzinfo is None:
        raise LedgerError(f"timestamp {value!r} has no timezone")
    return parsed


def now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 16), b""):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass(frozen=True)
class ResolvedContest:
    contest_id: str
    p: float
    base_rate: float
    skill_based: bool
    won: bool
    outcome: str
    predicted_at: str


@dataclass
class LedgerAudit:
    resolved: list[ResolvedContest] = field(default_factory=list)
    excluded: dict[str, str] = field(default_factory=dict)
    pending: list[str] = field(default_factory=list)
    unconfirmed_submissions: list[str] = field(default_factory=list)
    verified_submissions: int = 0
    verified_wins: int = 0

    def to_dict(self) -> dict:
        return {
            "resolved": len(self.resolved),
            "verified_submissions": self.verified_submissions,
            "verified_wins": self.verified_wins,
            "pending": sorted(self.pending),
            "unconfirmed_submissions": sorted(self.unconfirmed_submissions),
            "excluded": dict(sorted(self.excluded.items())),
        }


class Ledger:
    """JSONL ledger. Events are appended, never edited."""

    def __init__(self, path: Path | str):
        self.path = Path(path).expanduser()

    # -- writing ---------------------------------------------------------

    def _append(self, record: dict) -> dict:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(record, sort_keys=True) + "\n")
        return record

    def record_prediction(
        self,
        estimate: WinEstimate,
        *,
        made_at: Optional[str] = None,
        deadline: Optional[str] = None,
    ) -> dict:
        made_at = made_at or now_iso()
        parse_time(made_at)
        if deadline:
            parse_time(deadline)
        return self._append(
            {
                "type": "prediction",
                "contest_id": estimate.contest_id,
                "p_mean": estimate.p_mean,
                "p_low": estimate.p_low,
                "p_high": estimate.p_high,
                "base_rate": estimate.base_rate,
                "skill_based": estimate.skill_based,
                "evidence_class": estimate.evidence_class,
                "model_version": estimate.model_version,
                "notes": list(estimate.notes),
                "made_at": made_at,
                "deadline": deadline,
            }
        )

    def record_submission(
        self,
        contest_id: str,
        receipt_kind: str,
        receipt_ref: str,
        *,
        receipt_sha256: Optional[str] = None,
        submitted_at: Optional[str] = None,
        notes: str = "",
    ) -> dict:
        if receipt_kind not in RECEIPT_KINDS:
            raise LedgerError(
                f"receipt_kind {receipt_kind!r} not in {sorted(RECEIPT_KINDS)}; "
                "an entry without a receipt is not a submission"
            )
        if not receipt_ref.strip():
            raise LedgerError("receipt_ref is required (message id, URL, or file path)")
        submitted_at = submitted_at or now_iso()
        parse_time(submitted_at)
        return self._append(
            {
                "type": "submission",
                "contest_id": contest_id,
                "receipt_kind": receipt_kind,
                "receipt_ref": receipt_ref,
                "receipt_sha256": receipt_sha256,
                "strong_receipt": receipt_kind in STRONG_RECEIPTS,
                "submitted_at": submitted_at,
                "notes": notes,
            }
        )

    def record_result(
        self,
        contest_id: str,
        outcome: str,
        source: str,
        *,
        announced_at: str,
        notes: str = "",
    ) -> dict:
        if outcome not in OUTCOMES:
            raise LedgerError(f"outcome {outcome!r} not in {sorted(OUTCOMES)}")
        if outcome != "withdrawn" and not source.strip():
            raise LedgerError("results need a public source (winners announcement URL)")
        parse_time(announced_at)
        return self._append(
            {
                "type": "result",
                "contest_id": contest_id,
                "outcome": outcome,
                "source": source,
                "announced_at": announced_at,
                "notes": notes,
            }
        )

    # -- reading ---------------------------------------------------------

    def events(self) -> list[dict]:
        if not self.path.exists():
            return []
        events = []
        with open(self.path, encoding="utf-8") as fh:
            for lineno, line in enumerate(fh, 1):
                if not line.strip():
                    continue
                try:
                    events.append(json.loads(line))
                except json.JSONDecodeError as exc:
                    raise LedgerError(f"{self.path}:{lineno}: malformed JSON ({exc})") from exc
        return events

    def audit(self) -> LedgerAudit:
        by_contest: dict[str, dict[str, list[dict]]] = {}
        for ev in self.events():
            slot = by_contest.setdefault(ev["contest_id"], {"prediction": [], "submission": [], "result": []})
            slot.setdefault(ev["type"], []).append(ev)

        audit = LedgerAudit()
        for cid, evs in sorted(by_contest.items()):
            subs = evs["submission"]
            strong = [s for s in subs if s.get("receipt_kind") in STRONG_RECEIPTS]
            if strong:
                audit.verified_submissions += 1
            elif subs:
                audit.unconfirmed_submissions.append(cid)

            results = evs["result"]
            if not results:
                if subs:
                    audit.pending.append(cid)
                continue

            outcomes = {r["outcome"] for r in results}
            if len(outcomes) > 1:
                audit.excluded[cid] = f"conflicting results {sorted(outcomes)}"
                continue
            outcome = outcomes.pop()
            if OUTCOMES[outcome] is None:
                audit.excluded[cid] = "withdrawn"
                continue
            if not strong:
                audit.excluded[cid] = (
                    "no strong submission receipt (form screenshot only)" if subs else "no submission receipt"
                )
                continue

            announced = min(parse_time(r["announced_at"]) for r in results)
            preds = [p for p in evs["prediction"] if parse_time(p["made_at"]) < announced]
            if not preds:
                audit.excluded[cid] = "no prediction recorded before the result was announced"
                continue
            pred = max(preds, key=lambda p: parse_time(p["made_at"]))

            won = OUTCOMES[outcome] == 1
            if won:
                audit.verified_wins += 1
            audit.resolved.append(
                ResolvedContest(
                    contest_id=cid,
                    p=float(pred["p_mean"]),
                    base_rate=float(pred["base_rate"]),
                    skill_based=bool(pred["skill_based"]),
                    won=won,
                    outcome=outcome,
                    predicted_at=pred["made_at"],
                )
            )
        return audit

    def resolved(self) -> list[ResolvedContest]:
        return self.audit().resolved

    def skill_history(self) -> list[tuple[float, bool]]:
        """(base_rate, won) pairs for resolved skill-based contests: input to fit_skill_offset."""
        return [(r.base_rate, r.won) for r in self.resolved() if r.skill_based]
