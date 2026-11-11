"""Append-only, receipt-gated contest ledger: predictions, submissions, attempts, results.

A contest counts toward calibration only if all of these hold:

1. a prediction was recorded *before* the result was announced (no hindsight);
2. the submission has a strong receipt (confirmation email/page, platform
   entry URL or API record). A screenshot of a filled form shows that a form
   was filled, not that the entry was accepted, so it stays "unconfirmed";
3. a result was recorded with a public source, and results do not conflict.

The skill offset is learned from a wider set: any contest with a strong
receipt and a sourced result, using the final field size when the result
records it.  Learning needs no prediction, but calibration (scoring forecasts)
does.

Entries with no receipt at all are logged as *attempts*: visible in the audit,
never counted.  Losses are recorded the same way as wins.  A ledger that only
hears about wins can never be calibrated.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from evolution.contests.base_rates import KINDS, Observation, WinEstimate

STRONG_RECEIPTS = frozenset({"confirmation_email", "confirmation_page", "platform_entry_url", "platform_api"})
WEAK_RECEIPTS = frozenset({"form_screenshot"})
RECEIPT_KINDS = STRONG_RECEIPTS | WEAK_RECEIPTS

# outcome -> 1 (cleared the final stage), 0 (did not), None (excluded)
OUTCOMES: dict[str, Optional[int]] = {
    "won": 1,
    "placed": 1,
    "selected": 1,
    "lost": 0,
    "not_selected": 0,
    "disqualified": 0,
    "withdrawn": None,
}
UNKNOWN_TIME = "unknown"


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
    """A contest with a strong receipt and a sourced result."""

    contest_id: str
    kind: str
    won: bool
    outcome: str
    observation: Observation
    p: Optional[float] = None  # forecast made before the announcement, if any
    base_rate: Optional[float] = None  # naive base rate recorded with that forecast
    predicted_at: Optional[str] = None

    @property
    def skill_based(self) -> bool:
        return self.kind != "sweepstakes"

    @property
    def forecast(self) -> bool:
        return self.p is not None


@dataclass
class LedgerAudit:
    resolved: list[ResolvedContest] = field(default_factory=list)
    excluded: dict[str, str] = field(default_factory=dict)
    pending: list[str] = field(default_factory=list)
    unconfirmed_submissions: list[str] = field(default_factory=list)
    unreceipted_attempts: list[str] = field(default_factory=list)
    verified_submissions: int = 0
    verified_wins: int = 0

    @property
    def forecasted(self) -> list[ResolvedContest]:
        """Resolved contests that also had a pre-announcement forecast: input to calibration."""
        return [r for r in self.resolved if r.forecast]

    def to_dict(self) -> dict:
        return {
            "resolved": len(self.resolved),
            "resolved_with_forecast": len(self.forecasted),
            "verified_submissions": self.verified_submissions,
            "verified_wins": self.verified_wins,
            "pending": sorted(self.pending),
            "unconfirmed_submissions": sorted(self.unconfirmed_submissions),
            "unreceipted_attempts": sorted(self.unreceipted_attempts),
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

    @staticmethod
    def _time(value: Optional[str], *, allow_unknown: bool = False) -> str:
        if value is None:
            return now_iso()
        if allow_unknown and value == UNKNOWN_TIME:
            return value
        parse_time(value)
        return value

    def record_prediction(
        self,
        estimate: WinEstimate,
        *,
        made_at: Optional[str] = None,
        deadline: Optional[str] = None,
    ) -> dict:
        if deadline:
            parse_time(deadline)
        return self._append(
            {
                "type": "prediction",
                "contest_id": estimate.contest_id,
                "kind": estimate.kind,
                "p_mean": estimate.p_mean,
                "p_low": estimate.p_low,
                "p_high": estimate.p_high,
                "base_rate": estimate.base_rate,
                "evidence_class": estimate.evidence_class,
                "model_version": estimate.model_version,
                "notes": list(estimate.notes),
                "made_at": self._time(made_at),
                "deadline": deadline,
            }
        )

    def record_submission(
        self,
        contest_id: str,
        receipt_kind: str,
        receipt_ref: str,
        *,
        kind: str = "skill",
        receipt_sha256: Optional[str] = None,
        submitted_at: Optional[str] = None,
        notes: str = "",
    ) -> dict:
        """``submitted_at=None`` means now; ``"unknown"`` is allowed for imported history."""
        if receipt_kind not in RECEIPT_KINDS:
            raise LedgerError(
                f"receipt_kind {receipt_kind!r} not in {sorted(RECEIPT_KINDS)}; "
                "an entry without a receipt is an attempt (record_attempt), not a submission"
            )
        if not receipt_ref.strip():
            raise LedgerError("receipt_ref is required (message id, URL, or file path)")
        if kind not in KINDS:
            raise LedgerError(f"kind must be one of {KINDS}")
        return self._append(
            {
                "type": "submission",
                "contest_id": contest_id,
                "kind": kind,
                "receipt_kind": receipt_kind,
                "receipt_ref": receipt_ref,
                "receipt_sha256": receipt_sha256,
                "strong_receipt": receipt_kind in STRONG_RECEIPTS,
                "submitted_at": self._time(submitted_at, allow_unknown=True),
                "notes": notes,
            }
        )

    def record_attempt(
        self, contest_id: str, evidence: str, *, kind: str = "skill", attempted_at: Optional[str] = None
    ) -> dict:
        """An entry with no receipt at all. Shown in audits; never counted."""
        if not evidence.strip():
            raise LedgerError("describe what evidence of the attempt exists")
        if kind not in KINDS:
            raise LedgerError(f"kind must be one of {KINDS}")
        return self._append(
            {
                "type": "attempt",
                "contest_id": contest_id,
                "kind": kind,
                "evidence": evidence,
                "attempted_at": self._time(attempted_at, allow_unknown=True),
            }
        )

    def record_result(
        self,
        contest_id: str,
        outcome: str,
        source: str,
        *,
        announced_at: str,
        entrants: Optional[int] = None,
        prizes: Optional[int] = None,
        milestones: Optional[list[dict]] = None,
        rank: Optional[int] = None,
        notes: str = "",
    ) -> dict:
        """Record a published result.

        ``entrants`` (final competing entries incl. ours) and ``prizes`` (final
        slots) sharpen learning; ``milestones`` adds wider stages such as
        ``{"name": "finalist", "slots": 20, "reached": True}``; ``rank`` records
        a full-leaderboard position.
        """
        if outcome not in OUTCOMES:
            raise LedgerError(f"outcome {outcome!r} not in {sorted(OUTCOMES)}")
        if outcome != "withdrawn" and not source.strip():
            raise LedgerError("results need a public source (announcement URL or message id)")
        parse_time(announced_at)
        for m in milestones or []:
            if set(m) - {"name", "slots", "reached"} or not {"slots", "reached"} <= set(m):
                raise LedgerError("milestones need slots and reached (and optionally name)")
        if rank is not None and not entrants:
            raise LedgerError("rank needs entrants")
        return self._append(
            {
                "type": "result",
                "contest_id": contest_id,
                "outcome": outcome,
                "source": source,
                "announced_at": announced_at,
                "entrants": entrants,
                "prizes": prizes,
                "milestones": milestones or [],
                "rank": rank,
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

    @staticmethod
    def _kind(evs: dict[str, list[dict]]) -> str:
        # The prediction is where kind is chosen deliberately; submissions default to "skill".
        for key in ("prediction", "submission", "attempt"):
            for ev in evs.get(key, []):
                if "kind" in ev:
                    return ev["kind"]
                if "skill_based" in ev:  # ledgers written by model version 1
                    return "skill" if ev["skill_based"] else "sweepstakes"
        return "skill"

    @staticmethod
    def _observation(result: dict, won: bool, pred: Optional[dict]) -> Optional[Observation]:
        entrants, prizes = result.get("entrants"), result.get("prizes")
        if result.get("rank") and entrants:
            return Observation.from_rank(int(result["rank"]), int(entrants))
        if entrants and prizes:
            stages = [(int(m["slots"]), bool(m["reached"])) for m in result.get("milestones") or []]
            stages.append((int(prizes), won))
            return Observation.staged(int(entrants), stages)
        if pred is not None:
            return Observation.binary(float(pred["base_rate"]), won)
        return None

    def audit(self) -> LedgerAudit:
        by_contest: dict[str, dict[str, list[dict]]] = {}
        for ev in self.events():
            slot = by_contest.setdefault(
                ev["contest_id"], {"prediction": [], "submission": [], "result": [], "attempt": []}
            )
            slot.setdefault(ev["type"], []).append(ev)

        audit = LedgerAudit()
        for cid, evs in sorted(by_contest.items()):
            kind = self._kind(evs)
            subs = evs["submission"]
            strong = [s for s in subs if s.get("receipt_kind") in STRONG_RECEIPTS]
            if strong:
                audit.verified_submissions += 1
            elif subs:
                audit.unconfirmed_submissions.append(cid)
            elif evs["attempt"]:
                audit.unreceipted_attempts.append(cid)

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
                if subs:
                    audit.excluded[cid] = "no strong submission receipt (form screenshot only)"
                elif evs["attempt"]:
                    audit.excluded[cid] = "attempt without any receipt"
                else:
                    audit.excluded[cid] = "no submission receipt"
                continue

            won = OUTCOMES[outcome] == 1
            if won:
                audit.verified_wins += 1  # counted even when the field size is unknown
            announced = min(parse_time(r["announced_at"]) for r in results)
            preds = [p for p in evs["prediction"] if parse_time(p["made_at"]) < announced]
            pred = max(preds, key=lambda p: parse_time(p["made_at"])) if preds else None
            result = max(results, key=lambda r: (r.get("entrants") is not None, r.get("prizes") is not None))
            try:
                obs = self._observation(result, won, pred)
            except ValueError as exc:
                audit.excluded[cid] = f"inconsistent result detail: {exc}"
                continue
            if obs is None:
                audit.excluded[cid] = (
                    "no field size: record entrants and prizes with the result, "
                    "or a prediction before the announcement"
                )
                continue
            audit.resolved.append(
                ResolvedContest(
                    contest_id=cid,
                    kind=kind,
                    won=won,
                    outcome=outcome,
                    observation=obs,
                    p=float(pred["p_mean"]) if pred else None,
                    base_rate=float(pred["base_rate"]) if pred else None,
                    predicted_at=pred["made_at"] if pred else None,
                )
            )
        return audit

    def resolved(self) -> list[ResolvedContest]:
        return self.audit().resolved

    def skill_history(self, kind: str = "skill") -> list[Observation]:
        """Observations for fit_skill_offset, for one contest kind (sweepstakes never)."""
        if kind == "sweepstakes":
            return []
        return [r.observation for r in self.resolved() if r.kind == kind]
