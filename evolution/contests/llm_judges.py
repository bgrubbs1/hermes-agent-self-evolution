"""Isolated LLM judges: one stateless API call per judge, nothing shared.

This replaces persona judges written inside the authoring conversation.  Each
judge is a separate model call that receives only the blind packet prompt, so
it cannot see the build discussion, our self-assessment, or other judges.

Models are named the way DSPy / LiteLLM name them, e.g.
``anthropic/claude-sonnet-5-5``, ``openai/gpt-5``, ``gemini/gemini-3-pro``.
Mixing families avoids the correlated errors of N copies of one model.
"""

from __future__ import annotations

from typing import Callable, Optional

from evolution.contests.judges import JudgeFn, JudgePacket, JudgeSpec

LMCall = Callable[[str], str]

_FAMILIES = {
    "anthropic": "claude",
    "claude": "claude",
    "openai": "gpt",
    "gpt": "gpt",
    "azure": "gpt",
    "gemini": "gemini",
    "vertex_ai": "gemini",
    "google": "gemini",
    "mistral": "mistral",
    "groq": "llama",
    "ollama": "local",
    "deepseek": "deepseek",
    "xai": "grok",
}


def model_family(model: str) -> str:
    prefix = model.split("/", 1)[0].lower()
    if prefix in _FAMILIES:
        return _FAMILIES[prefix]
    lowered = model.lower()
    for key, fam in _FAMILIES.items():
        if key in lowered:
            return fam
    return prefix


def judge_specs(models: dict[str, str]) -> list[JudgeSpec]:
    """JudgeSpecs for fresh, isolated model judges (never the authoring session)."""
    return [JudgeSpec(judge_id=jid, model_family=model_family(m)) for jid, m in models.items()]


def _dspy_lm_factory(temperature: float, max_tokens: int) -> Callable[[str], LMCall]:
    def factory(model: str) -> LMCall:
        import dspy  # imported lazily so the contests package stays dependency-free

        lm = dspy.LM(model, temperature=temperature, max_tokens=max_tokens, cache=False)

        def call(prompt: str) -> str:
            out = lm(prompt)
            first = out[0] if isinstance(out, list) else out
            return first.get("text", "") if isinstance(first, dict) else str(first)

        return call

    return factory


def make_llm_judge_fn(
    models: dict[str, str],
    *,
    lm_factory: Optional[Callable[[str], LMCall]] = None,
    temperature: float = 0.0,
    max_tokens: int = 4000,
) -> JudgeFn:
    """Build a ``judge_fn`` for run_panel / backtest_panel from {judge_id: model}."""
    factory = lm_factory or _dspy_lm_factory(temperature, max_tokens)
    calls = {jid: factory(model) for jid, model in models.items()}

    def judge_fn(packet: JudgePacket) -> str:
        if packet.judge_id not in calls:
            raise KeyError(f"no model configured for judge {packet.judge_id}")
        return calls[packet.judge_id](packet.prompt())

    return judge_fn
