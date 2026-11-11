"""Blind judge panels whose verdicts can be checked before they are believed.

Persona judges written in the same session as the entry are the author
grading their own homework.  This module replaces that with a protocol:

* **Blind batch.** Each judge gets only the rubric and a shuffled set of
  artifacts with random IDs: our entry, real past *winners*, typical
  *entrants*, and optional *controls* (copies of our entry with a planted
  defect, e.g. missing license or over-length video).  Judges never see roles,
  other judges' scores, or the build conversation.
* **Leak check.** Artifacts mentioning placement ("winner", "1st place",
  "finalist" ...) are refused, because a badge on a winner's page tells the
  judge the answer.
* **Validity gate.** A panel's opinion of our entry counts only if the same
  panel (a) ranks real winners above typical entrants (AUC), (b) agrees with
  itself (ICC(2,1)), (c) catches planted defects, and (d) is not self-graded
  or reused from the optimisation loop (Goodhart guard).
* **Correlated judges.** N copies of one model family are not N independent
  judges. That is reported as its own evidence class.

The LLM call is a caller-supplied ``judge_fn(packet) -> str | dict`` so any
runtime (Claude subagents, DSPy, Hermes delegate_task, humans) can plug in.
"""

from __future__ import annotations

import json
import random
import re
from dataclasses import asdict, dataclass, field
from typing import Callable, Optional, Sequence, Union

ROLES = ("entry", "winner", "entrant", "control")
SCORE_MIN, SCORE_MAX = 0.0, 10.0

LEAK_PATTERN = re.compile(
    r"\b(winner|winners|winning entry|1st place|first place|2nd place|second place|"
    r"3rd place|third place|grand prize|honou?rable mention|finalist|runner[- ]up|"
    r"judges'? choice|prize recipient)\b",
    re.IGNORECASE,
)

PROMPT_TEMPLATE = """You are an independent judge for a skill contest.

Score every item below against the rubric. Judge each item on its own merits.
You are not told who made any item or how any item placed, and you must not guess.
Use the full range {lo:g}-{hi:g} for each criterion.

## Rubric
{rubric}

## Criteria
{criteria}

## Items
{items}

Return only JSON of the form:
{{"<item id>": {{"<criterion>": <score>, ...}}, ...}}
covering every item and every criterion.
"""


class LeakError(ValueError):
    """An artifact reveals its placement or other information judges must not see."""


@dataclass(frozen=True)
class JudgeSpec:
    judge_id: str
    model_family: str  # e.g. "claude", "gpt", "gemini", "llama", "human"
    shares_author_context: bool = False  # saw the build conversation / is the authoring agent
    used_in_optimization: bool = False  # its scores steered iterations of this entry


@dataclass(frozen=True)
class Artifact:
    artifact_id: str
    text: str
    role: str
    source: str = ""

    def __post_init__(self) -> None:
        if self.role not in ROLES:
            raise ValueError(f"role {self.role!r} not in {ROLES}")


@dataclass(frozen=True)
class JudgePacket:
    judge_id: str
    rubric: str
    criteria: tuple[str, ...]
    items: tuple[tuple[str, str], ...]  # (blind_id, text)

    def prompt(self) -> str:
        items = "\n\n".join(f"### Item {bid}\n{text}" for bid, text in self.items)
        return PROMPT_TEMPLATE.format(
            lo=SCORE_MIN,
            hi=SCORE_MAX,
            rubric=self.rubric.strip(),
            criteria="\n".join(f"- {c}" for c in self.criteria),
            items=items,
        )


@dataclass(frozen=True)
class BlindBatch:
    packets: tuple[JudgePacket, ...]
    judges: tuple[JudgeSpec, ...]
    key: dict[str, dict[str, str]]  # judge_id -> blind_id -> artifact_id
    roles: dict[str, str]  # artifact_id -> role
    criteria: tuple[str, ...]


def find_leaks(artifacts: Sequence[Artifact], forbidden_terms: Sequence[str] = ()) -> dict[str, list[str]]:
    leaks: dict[str, list[str]] = {}
    for art in artifacts:
        hits = sorted({m.group(0).lower() for m in LEAK_PATTERN.finditer(art.text)})
        lowered = art.text.lower()
        hits += [t for t in forbidden_terms if t.lower() in lowered]
        if hits:
            leaks[art.artifact_id] = hits
    return leaks


def make_blind_batch(
    entry: Artifact,
    references: Sequence[Artifact],
    judges: Sequence[JudgeSpec],
    *,
    rubric: str,
    criteria: Sequence[str],
    seed: Optional[int] = None,
    forbidden_terms: Sequence[str] = (),
) -> BlindBatch:
    if entry.role != "entry":
        raise ValueError("entry must have role 'entry'")
    if any(r.role == "entry" for r in references):
        raise ValueError("references must be winners, entrants or controls")
    if not judges:
        raise ValueError("at least one judge is required")
    if not criteria:
        raise ValueError("at least one criterion is required")
    ids = [entry.artifact_id] + [r.artifact_id for r in references]
    if len(set(ids)) != len(ids):
        raise ValueError("artifact ids must be unique")
    judge_ids = [j.judge_id for j in judges]
    if len(set(judge_ids)) != len(judge_ids):
        raise ValueError("judge ids must be unique")

    artifacts = [entry, *references]
    leaks = find_leaks(artifacts, forbidden_terms)
    if leaks:
        detail = "; ".join(f"{aid}: {', '.join(hits)}" for aid, hits in sorted(leaks.items()))
        raise LeakError(f"scrub placement/identity hints before judging ({detail})")

    master = random.Random(seed)
    packets, key = [], {}
    for judge in judges:
        rng = random.Random(master.getrandbits(64))
        order = list(artifacts)
        rng.shuffle(order)
        mapping, items, used = {}, [], set()
        for art in order:
            bid = f"{rng.getrandbits(32):08x}"
            while bid in used:
                bid = f"{rng.getrandbits(32):08x}"
            used.add(bid)
            mapping[bid] = art.artifact_id
            items.append((bid, art.text))
        key[judge.judge_id] = mapping
        packets.append(
            JudgePacket(judge_id=judge.judge_id, rubric=rubric, criteria=tuple(criteria), items=tuple(items))
        )
    return BlindBatch(
        packets=tuple(packets),
        judges=tuple(judges),
        key=key,
        roles={a.artifact_id: a.role for a in artifacts},
        criteria=tuple(criteria),
    )


def _extract_json(text: str) -> dict:
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end <= start:
        raise ValueError("judge response contains no JSON object")
    return json.loads(text[start : end + 1])


def parse_judge_response(response: Union[str, dict], packet: JudgePacket) -> dict[str, dict[str, float]]:
    data = _extract_json(response) if isinstance(response, str) else response
    expected = {bid for bid, _ in packet.items}
    missing = expected - set(data)
    if missing:
        raise ValueError(f"judge {packet.judge_id} skipped items {sorted(missing)}")
    scores: dict[str, dict[str, float]] = {}
    for bid in expected:
        row = data[bid]
        scores[bid] = {}
        for crit in packet.criteria:
            if crit not in row:
                raise ValueError(f"judge {packet.judge_id} item {bid} missing criterion {crit!r}")
            value = float(row[crit])
            if not SCORE_MIN <= value <= SCORE_MAX:
                raise ValueError(f"score {value} outside [{SCORE_MIN}, {SCORE_MAX}]")
            scores[bid][crit] = value
    return scores


JudgeFn = Callable[[JudgePacket], Union[str, dict]]


def run_panel(batch: BlindBatch, judge_fn: JudgeFn) -> dict[str, dict[str, dict[str, float]]]:
    """Call each judge once, in isolation. Returns judge_id -> artifact_id -> criterion -> score."""
    results = {}
    for packet in batch.packets:
        blind_scores = parse_judge_response(judge_fn(packet), packet)
        mapping = batch.key[packet.judge_id]
        results[packet.judge_id] = {mapping[bid]: crit for bid, crit in blind_scores.items()}
    return results


def evidence_class(judges: Sequence[JudgeSpec]) -> str:
    if any(j.shares_author_context for j in judges):
        return "self_graded"
    if any(j.used_in_optimization for j in judges):
        return "optimization_contaminated"
    families = {j.model_family.lower() for j in judges}
    if families == {"human"}:
        return "human_panel"
    model_families = families - {"human"}
    if len(model_families) >= 2 or "human" in families:
        return "multi_family_panel"
    return "single_model_family"


def icc_2_1(matrix: Sequence[Sequence[float]]) -> Optional[float]:
    """ICC(2,1): two-way random effects, absolute agreement, single rater. Rows=items, cols=judges."""
    n = len(matrix)
    if n < 2:
        return None
    k = len(matrix[0])
    if k < 2 or any(len(row) != k for row in matrix):
        return None
    grand = sum(sum(r) for r in matrix) / (n * k)
    row_means = [sum(r) / k for r in matrix]
    col_means = [sum(matrix[i][j] for i in range(n)) / n for j in range(k)]
    ss_rows = k * sum((m - grand) ** 2 for m in row_means)
    ss_cols = n * sum((m - grand) ** 2 for m in col_means)
    ss_total = sum((x - grand) ** 2 for r in matrix for x in r)
    ss_err = ss_total - ss_rows - ss_cols
    ms_rows = ss_rows / (n - 1)
    ms_cols = ss_cols / (k - 1)
    ms_err = ss_err / ((n - 1) * (k - 1))
    denom = ms_rows + (k - 1) * ms_err + k * (ms_cols - ms_err) / n
    if denom <= 0:
        return None
    return (ms_rows - ms_err) / denom


def _auc(pos: Sequence[float], neg: Sequence[float]) -> Optional[float]:
    if not pos or not neg:
        return None
    wins = sum(1.0 if p > q else 0.5 if p == q else 0.0 for p in pos for q in neg)
    return wins / (len(pos) * len(neg))


@dataclass
class PanelResult:
    evidence_class: str
    valid: bool
    reasons: list[str]
    warnings: list[str]
    n_judges: int
    n_winners: int
    n_entrants: int
    n_controls: int
    entry_mean: float
    entry_ci: tuple[float, float]
    entry_percentile: float  # share of winners+entrants the entry outscores (panel means)
    beats_winners_fraction: Optional[float]  # over judge x winner pairs
    winner_vs_entrant_auc: Optional[float]
    control_detection_rate: Optional[float]  # share of judge x control pairs scored below the entry
    icc: Optional[float]
    artifact_means: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)

    def to_markdown(self) -> str:
        status = "VALID" if self.valid else "NOT VALID: do not cite"
        fmt = lambda v: "n/a" if v is None else f"{v:.3f}"  # noqa: E731
        lines = [
            f"## Judge panel ({status})",
            "",
            f"- Evidence class: **{self.evidence_class}** ({self.n_judges} judges)",
            f"- References: {self.n_winners} real winners, {self.n_entrants} typical entrants, "
            f"{self.n_controls} planted-defect controls",
            f"- Entry composite: {self.entry_mean:.2f} (95% bootstrap over judges "
            f"{self.entry_ci[0]:.2f}-{self.entry_ci[1]:.2f})",
            f"- Entry outscores {self.entry_percentile:.0%} of reference entries; "
            f"beats real winners in {fmt(self.beats_winners_fraction)} of judge-winner pairs",
            f"- Panel checks: winner-vs-entrant AUC {fmt(self.winner_vs_entrant_auc)}, "
            f"ICC(2,1) {fmt(self.icc)}, planted-defect detection {fmt(self.control_detection_rate)}",
        ]
        if self.reasons:
            lines += ["", "**Why not valid**", ""] + [f"- {r}" for r in self.reasons]
        if self.warnings:
            lines += ["", "**Warnings**", ""] + [f"- {w}" for w in self.warnings]
        return "\n".join(lines) + "\n"


def evaluate_panel(
    batch: BlindBatch,
    scores: dict[str, dict[str, dict[str, float]]],
    *,
    weights: Optional[dict[str, float]] = None,
    min_judges: int = 3,
    min_references_per_role: int = 3,
    min_auc: float = 0.75,
    min_icc: float = 0.4,
    min_control_detection: float = 0.8,
    bootstrap: int = 2000,
    seed: int = 0,
) -> PanelResult:
    weights = weights or {c: 1.0 for c in batch.criteria}
    wsum = sum(weights[c] for c in batch.criteria)
    judge_ids = [j.judge_id for j in batch.judges]
    if set(scores) != set(judge_ids):
        raise ValueError("scores must cover exactly the batch's judges")
    artifact_ids = list(batch.roles)

    def composite(judge_id: str, aid: str) -> float:
        row = scores[judge_id][aid]
        return sum(weights[c] * row[c] for c in batch.criteria) / wsum

    matrix = {aid: [composite(j, aid) for j in judge_ids] for aid in artifact_ids}
    means = {aid: sum(v) / len(v) for aid, v in matrix.items()}
    entry_id = next(a for a, r in batch.roles.items() if r == "entry")
    winners = [a for a, r in batch.roles.items() if r == "winner"]
    entrants = [a for a, r in batch.roles.items() if r == "entrant"]
    controls = [a for a, r in batch.roles.items() if r == "control"]

    rng = random.Random(seed)
    k = len(judge_ids)
    boots = sorted(
        sum(matrix[entry_id][rng.randrange(k)] for _ in range(k)) / k for _ in range(max(1, bootstrap))
    )
    ci = (boots[int(0.025 * (len(boots) - 1))], boots[int(0.975 * (len(boots) - 1))])

    refs = winners + entrants
    percentile = (
        sum(1.0 if means[entry_id] > means[a] else 0.5 if means[entry_id] == means[a] else 0.0 for a in refs)
        / len(refs)
        if refs
        else 0.0
    )
    beats = None
    if winners:
        pairs = [(j, w) for j in range(k) for w in winners]
        beats = sum(1 for j, w in pairs if matrix[entry_id][j] > matrix[w][j]) / len(pairs)
    auc = _auc([means[a] for a in winners], [means[a] for a in entrants])
    detection = None
    if controls:
        pairs = [(j, c) for j in range(k) for c in controls]
        detection = sum(1 for j, c in pairs if matrix[entry_id][j] > matrix[c][j]) / len(pairs)
    icc = icc_2_1([matrix[a] for a in artifact_ids])

    ev = evidence_class(batch.judges)
    reasons: list[str] = []
    warnings: list[str] = []
    if ev == "self_graded":
        reasons.append("a judge shares the author's context: this is self-grading")
    if ev == "optimization_contaminated":
        reasons.append("a judge was used to steer iterations; hold out fresh judges for the final score")
    if k < min_judges:
        reasons.append(f"only {k} judges (< {min_judges})")
    if len(winners) < min_references_per_role or len(entrants) < min_references_per_role:
        reasons.append(
            f"need >= {min_references_per_role} real winners and typical entrants to test the panel "
            f"(have {len(winners)} and {len(entrants)})"
        )
    elif auc is not None and auc < min_auc:
        reasons.append(f"panel cannot tell real winners from typical entrants (AUC {auc:.2f} < {min_auc})")
    if icc is None:
        reasons.append("inter-rater agreement could not be computed")
    elif icc < min_icc:
        reasons.append(f"judges disagree too much (ICC {icc:.2f} < {min_icc})")
    if detection is not None and detection < min_control_detection:
        reasons.append(
            f"panel missed planted defects (detection {detection:.2f} < {min_control_detection})"
        )
    if not controls:
        warnings.append("no planted-defect controls: the panel's sensitivity to rule breaks is untested")
    if ev == "single_model_family":
        warnings.append("all judges share one model family: errors are correlated and agreement is inflated")

    return PanelResult(
        evidence_class=ev,
        valid=not reasons,
        reasons=reasons,
        warnings=warnings,
        n_judges=k,
        n_winners=len(winners),
        n_entrants=len(entrants),
        n_controls=len(controls),
        entry_mean=means[entry_id],
        entry_ci=ci,
        entry_percentile=percentile,
        beats_winners_fraction=beats,
        winner_vs_entrant_auc=auc,
        control_detection_rate=detection,
        icc=icc,
        artifact_means=means,
    )


def split_judges(
    judges: Sequence[JudgeSpec], n_holdout: int, *, seed: Optional[int] = None
) -> tuple[list[JudgeSpec], list[JudgeSpec]]:
    """Split a pool into optimisation judges (marked used) and untouched holdout judges.

    Iterate against the first list; report only the second.  Scoring the final
    entry with the judges you optimised against is Goodhart's law in action.
    """
    if not 0 < n_holdout < len(judges):
        raise ValueError("n_holdout must leave at least one judge on each side")
    pool = list(judges)
    random.Random(seed).shuffle(pool)
    holdout = pool[:n_holdout]
    optimise = [
        JudgeSpec(j.judge_id, j.model_family, j.shares_author_context, used_in_optimization=True)
        for j in pool[n_holdout:]
    ]
    return optimise, holdout
