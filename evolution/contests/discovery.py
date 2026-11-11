"""Contest discovery from structured sources, where zero results is an error.

The old scrapers parsed HTML and reported "Found 0 contests" as if that were
an answer.  Here:

* sources read structured data: the Devpost JSON API, and the JSON block
  Hackster embeds in its contests page (``data-hypernova-key="ContestsPage"``);
* every fetch identifies itself honestly and only touches paths the site's
  robots.txt allows (checked 2026-11-11: Devpost allows all; Hackster allows
  ``/contests`` and disallows ``/projects/*``);
* a source that returns nothing, or whose schema changed, raises
  :class:`DiscoveryError` with diagnostics instead of returning ``[]``;
* sites that refuse automated clients (element14 at the time of writing) are
  not scraped around: add them to a hand-maintained registry file instead.
"""

from __future__ import annotations

import html as html_lib
import json
import math
import re
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable, Iterable, Optional

from evolution.contests.base_rates import ContestField

USER_AGENT = "hermes-contest-discovery/0.1 (+https://github.com/bgrubbs1/hermes-agent-self-evolution)"
DEVPOST_API = "https://devpost.com/api/hackathons"
HACKSTER_CONTESTS = "https://www.hackster.io/contests"

# Fraction of registrants assumed, at the low end, to actually submit.  This is
# an assumption, not a measurement: it only sets the lower end of a wide range,
# and the gallery count replaces it once published.
DEFAULT_MIN_SUBMIT_FRACTION = 0.05


class DiscoveryError(RuntimeError):
    """A source returned nothing usable. ``diagnostics`` says why."""

    def __init__(self, message: str, diagnostics: dict):
        super().__init__(f"{message}: {json.dumps(diagnostics, sort_keys=True)}")
        self.diagnostics = diagnostics


@dataclass
class ContestLead:
    source: str
    contest_id: str
    name: str
    url: str
    kind: str = "skill"  # skill | application | sweepstakes
    deadline: Optional[str] = None  # ISO-8601 with timezone when the source gives one
    deadline_text: Optional[str] = None
    prize_value_usd: Optional[float] = None
    prize_slots: Optional[int] = None
    registrations: Optional[int] = None
    submissions: Optional[int] = None
    open: bool = True
    invite_only: bool = False
    tags: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)

    def contest_field(self, *, min_submit_fraction: float = DEFAULT_MIN_SUBMIT_FRACTION) -> ContestField:
        """Field estimate for this lead. Raises UnknownFieldError when nothing is known."""
        if not self.prize_slots:
            raise ValueError(f"{self.contest_id}: prize slot count unknown; read the rules page")
        if self.submissions:
            return ContestField(
                self.contest_id, prizes=self.prize_slots, entrants=self.submissions, kind=self.kind,
                entrants_source=f"{self.source} submission count",
            )
        if self.registrations:
            high = max(self.registrations, self.prize_slots + 1)
            low = max(self.prize_slots + 1, math.ceil(self.registrations * min_submit_fraction))
            return ContestField(
                self.contest_id, prizes=self.prize_slots, entrants_range=(min(low, high), high), kind=self.kind,
                entrants_source=(
                    f"{self.source} registrations={self.registrations}; submissions assumed "
                    f">= {min_submit_fraction:.0%} of registrations until the gallery count is published"
                ),
            )
        return ContestField(self.contest_id, prizes=self.prize_slots, kind=self.kind)  # raises UnknownFieldError


HttpGet = Callable[[str], tuple[int, str]]


class BotChallengeError(DiscoveryError):
    """The site answered with a bot challenge. It is reported, never bypassed."""


def http_get(url: str, timeout: float = 30.0) -> tuple[int, str]:
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT, "Accept": "application/json, text/html"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:  # noqa: S310 - fixed https URLs
            return resp.status, resp.read().decode("utf-8", errors="replace")
    except urllib.error.HTTPError as exc:
        if exc.headers.get("Cf-Mitigated") == "challenge":
            raise BotChallengeError(
                "site presented a bot challenge (not bypassed); take this value from the rules page "
                "and put it in the registry",
                {"url": url, "status": exc.code},
            ) from exc
        return exc.code, exc.read().decode("utf-8", errors="replace")


def _slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")


def _money(text) -> Optional[float]:
    if text is None:
        return None
    if isinstance(text, (int, float)):
        return float(text)
    plain = re.sub(r"<[^>]+>", "", str(text))
    digits = re.sub(r"[^0-9.]", "", plain)
    try:
        return float(digits) if digits else None
    except ValueError:
        return None


# -- Devpost -----------------------------------------------------------------

DEVPOST_REQUIRED = ("id", "title", "url")


def parse_devpost(payload: dict) -> list[ContestLead]:
    items = payload.get("hackathons")
    if not isinstance(items, list):
        raise DiscoveryError("devpost schema changed", {"top_level_keys": sorted(payload)})
    leads = []
    for h in items:
        missing = [k for k in DEVPOST_REQUIRED if k not in h]
        if missing:
            raise DiscoveryError("devpost item missing keys", {"missing": missing, "keys": sorted(h)})
        subdomain = re.sub(r"^https?://", "", h["url"]).split(".")[0]
        # prizes_counts is NOT a slot count: for the same hackathon it read 14+3 and
        # later 1+0 on 2026-11-11.  Slots come from the hackathon page instead.
        notes = ["prize slots unknown until the hackathon page is read (fetch_devpost_prizes)"]
        leads.append(
            ContestLead(
                source="devpost",
                contest_id=f"devpost-{subdomain}",
                name=h["title"],
                url=h["url"],
                deadline_text=h.get("submission_period_dates"),
                prize_value_usd=_money(h.get("prize_amount")),
                registrations=h.get("registrations_count"),
                open=h.get("open_state") == "open",
                invite_only=bool(h.get("invite_only")),
                tags=[t.get("name", "") for t in h.get("themes") or []],
                notes=notes,
            )
        )
    return leads


def fetch_devpost(*, pages: int = 3, per_page: int = 50, get: HttpGet = http_get) -> list[ContestLead]:
    leads, diag = [], {"source": "devpost", "pages": []}
    for page in range(1, pages + 1):
        url = f"{DEVPOST_API}?status[]=open&page={page}&per_page={per_page}"
        status, body = get(url)
        page_diag = {"url": url, "status": status, "bytes": len(body)}
        diag["pages"].append(page_diag)
        if status != 200:
            raise DiscoveryError("devpost request failed", diag)
        try:
            payload = json.loads(body)
        except json.JSONDecodeError as exc:
            page_diag["error"] = str(exc)
            raise DiscoveryError("devpost returned non-JSON", diag) from exc
        batch = parse_devpost(payload)
        page_diag["items"] = len(batch)
        leads.extend(batch)
        total = (payload.get("meta") or {}).get("total_count")
        if not batch or (total is not None and len(leads) >= total):
            break
    if not leads:
        raise DiscoveryError("devpost returned 0 open hackathons", diag)
    return leads


_PRIZE_BLOCK = re.compile(r'<div class="[^"]*\bprize\b[^"]*" id="prize_\d+">(.*?)(?=<div class="[^"]*\bprize\b[^"]*" id="prize_\d+">|\Z)', re.S)


def _text(fragment: str) -> str:
    return re.sub(r"\s+", " ", html_lib.unescape(re.sub(r"<[^>]+>", " ", fragment))).strip()


def parse_devpost_prizes(page_html: str) -> list[dict]:
    """Prize list from a hackathon page: [{"title", "winners", "value_usd"}]."""
    start = page_html.find('id="prizes"')
    if start < 0:
        raise DiscoveryError("devpost page has no prizes section", {"bytes": len(page_html)})
    prizes = []
    for block in _PRIZE_BLOCK.findall(page_html[start:]):
        title = re.search(r'class="prize-title"[^>]*>(.*?)</h6>', block, re.S)
        winners = re.search(r'class="prize-winners"[^>]*>\s*(\d+)\s+winners?', block)
        value = re.search(r'class="prize-value"[^>]*>(.*?)</div>', block, re.S)
        if not winners:
            raise DiscoveryError("devpost prize block without a winner count", {"title": _text(title.group(1)) if title else None})
        prizes.append({
            "title": _text(title.group(1)) if title else "",
            "winners": int(winners.group(1)),
            "value_usd": _money(value.group(1)) if value else None,
        })
    if not prizes:
        raise DiscoveryError("devpost prizes section has no prize blocks", {"bytes": len(page_html)})
    return prizes


def fetch_devpost_prizes(
    leads: Iterable[ContestLead], *, get: HttpGet = http_get, delay: float = 1.0, limit: Optional[int] = None
) -> dict[str, str]:
    """Fill ``prize_slots`` for Devpost leads from their hackathon pages (one polite request each).

    Slots = total winners across prizes: an upper bound on distinct winning
    entries, since some prizes are track-restricted or not stackable.
    Returns {contest_id: error} for pages that could not be read.
    """
    import time

    errors: dict[str, str] = {}
    done = 0
    for lead in leads:
        if lead.source != "devpost" or (limit is not None and done >= limit):
            continue
        if done:
            time.sleep(delay)
        done += 1
        try:
            status, body = get(lead.url)
            if status != 200:
                raise DiscoveryError("devpost page request failed", {"status": status})
            prizes = parse_devpost_prizes(body)
        except Exception as exc:
            errors[lead.contest_id] = str(exc)
            continue
        lead.prize_slots = sum(p["winners"] for p in prizes)
        lead.notes = [
            f"prize_slots = {lead.prize_slots} winners across {len(prizes)} prizes (upper bound on distinct "
            "winning entries; check which tracks you are eligible for)"
        ]
    return errors


# -- Hackster ----------------------------------------------------------------

_HYPERNOVA = re.compile(
    r'<script type="application/json" data-hypernova-key="ContestsPage"[^>]*><!--(.*?)--></script>', re.S
)


def parse_hackster(page_html: str) -> list[ContestLead]:
    match = _HYPERNOVA.search(page_html)
    if not match:
        raise DiscoveryError("hackster ContestsPage data block not found", {"bytes": len(page_html)})
    data = json.loads(html_lib.unescape(match.group(1)))
    leads = []
    for c in data.get("competition") or []:
        if "name" not in c or "url" not in c:
            raise DiscoveryError("hackster contest missing keys", {"keys": sorted(c)})
        prize = c.get("prize") or {}
        leads.append(
            ContestLead(
                source="hackster",
                contest_id="hackster-" + c["url"].rstrip("/").split("/")[-1],
                name=c["name"],
                url="https://www.hackster.io" + c["url"] if c["url"].startswith("/") else c["url"],
                deadline=c.get("end_date_in_pdt"),
                prize_value_usd=_money(prize.get("cash_value")),
                open=bool(c.get("is_registration_open")) and c.get("status") == "IN_PROGRESS",
                notes=["listing shows the headline prize only; read the rules page for slot count"],
            )
        )
    for g in data.get("giveaways") or []:
        title = g.get("title") or g.get("name")
        if not title or "url" not in g:
            continue
        leads.append(
            ContestLead(
                source="hackster",
                contest_id="hackster-giveaway-" + g["url"].rstrip("/").split("/")[-1],
                name=title,
                url=g["url"],
                kind="sweepstakes" if g.get("entry_type") != "merit" else "skill",
                deadline=g.get("end_date_in_pdt"),
                prize_value_usd=_money(g.get("total_prize_value")),
                open=bool(g.get("is_registration_open")),
                notes=[f"giveaway entry_type={g.get('entry_type')}"],
            )
        )
    return leads


def fetch_hackster(*, get: HttpGet = http_get) -> list[ContestLead]:
    status, body = get(HACKSTER_CONTESTS)
    diag = {"source": "hackster", "url": HACKSTER_CONTESTS, "status": status, "bytes": len(body)}
    if status != 200:
        raise DiscoveryError("hackster request failed", diag)
    leads = parse_hackster(body)
    if not leads:
        raise DiscoveryError("hackster returned 0 contests", diag)
    return leads


# -- Registry ----------------------------------------------------------------

def load_registry(path: Path | str) -> list[dict]:
    """Hand-maintained entries (JSON list) for sites that cannot or should not be scraped.

    An entry whose ``contest_id`` matches a discovered contest overrides its
    fields (e.g. ``prize_slots`` read from the rules page); any other entry is a
    new lead and needs ``name`` and ``url``.
    """
    raw = json.loads(Path(path).expanduser().read_text(encoding="utf-8"))
    if not isinstance(raw, list):
        raise ValueError("registry must be a JSON list of contest objects")
    allowed = set(ContestLead.__dataclass_fields__)
    for item in raw:
        unknown = set(item) - allowed
        if unknown:
            raise ValueError(f"registry entry {item.get('contest_id')}: unknown keys {sorted(unknown)}")
        if "contest_id" not in item and not (item.get("name") and item.get("url")):
            raise ValueError("registry entries need contest_id, or name and url")
    return raw


def _apply_registry(leads: list[ContestLead], entries: list[dict]) -> list[ContestLead]:
    by_id = {lead.contest_id: lead for lead in leads}
    out = list(leads)
    for item in entries:
        cid = item.get("contest_id")
        if cid in by_id:
            lead = by_id[cid]
            for key, value in item.items():
                if key != "contest_id" and value is not None:
                    setattr(lead, key, value)
            lead.notes = lead.notes + ["fields overridden from registry"]
            continue
        if not item.get("name") or not item.get("url"):
            raise ValueError(f"registry entry {cid}: no discovered contest with this id, so name and url are needed")
        fields = dict(item)
        fields.setdefault("source", "registry")
        fields.setdefault("contest_id", f"registry-{_slug(fields['name'])}")
        lead = ContestLead(**fields)
        by_id[lead.contest_id] = lead
        out.append(lead)
    return out


SOURCES = {"devpost": fetch_devpost, "hackster": fetch_hackster}


def discover(
    sources: Iterable[str] = ("devpost", "hackster"),
    *,
    registry: Optional[Path | str] = None,
    get: HttpGet = http_get,
) -> tuple[list[ContestLead], dict[str, str]]:
    """Run every source; collect leads and per-source errors. Raise only if nothing at all came back."""
    leads: list[ContestLead] = []
    errors: dict[str, str] = {}
    for name in sources:
        try:
            leads.extend(SOURCES[name](get=get))
        except Exception as exc:  # report every source's failure, keep the others
            errors[name] = str(exc)
    seen, unique = set(), []
    for lead in leads:
        if lead.contest_id not in seen:
            seen.add(lead.contest_id)
            unique.append(lead)
    if registry:
        unique = _apply_registry(unique, load_registry(registry))
    if not unique:
        raise DiscoveryError("no source produced any contest", {"errors": errors})
    return unique, errors
