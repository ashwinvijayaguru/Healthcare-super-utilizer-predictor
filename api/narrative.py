"""Patient-facing plain-language summary.

Two providers behind one interface:

* ``TemplateNarrator`` - deterministic, always available, no network, no cost.
* ``ClaudeNarrator`` - calls the Anthropic API for warmer, better-organised prose.

The template provider is the floor, not a stub. If the API key is absent, the
call times out, the response fails to parse, or the model returns something that
fails validation, the service falls back and the user still gets a complete,
correct summary. A patient-facing health feature that can go blank because a
third-party API had a bad minute is not a production feature.

Safety posture: the LLM is given ONLY the structured facts already computed, and
is explicitly constrained to rephrase them. It never sees free text, is never
asked to diagnose, and its output is validated field-by-field before use.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any

import httpx

from api.config import get_settings
from ml.config import TIER_META

log = logging.getLogger(__name__)

SYSTEM_PROMPT = """You write short health summaries that are read by patients, not clinicians.

You will receive a JSON object of facts that have already been calculated. Your job is \
to express those facts warmly and clearly. You must not add any medical fact, number, \
diagnosis, prognosis or recommendation that is not present in the input.

Rules:
- Write at roughly a 6th-grade reading level. Short sentences. Everyday words.
- Never use the words "super utilizer", "risk score", "model", "algorithm" or "prediction".
- Never state or imply that something WILL happen. Use "could", "may", "more likely".
- Never diagnose, never suggest starting/stopping/changing any medication or treatment.
- Be encouraging and non-alarming. The reader may already be frightened or tired.
- Frame every risk as something the care team will help with, not as the reader's fault.
- Address the reader as "you".

Return ONLY a JSON object, with no markdown fences and no preamble, shaped exactly like:
{
  "headline": "one short sentence, max 90 characters",
  "body": "2-3 short paragraphs separated by \\n\\n, max 900 characters total",
  "what_this_means": ["3-4 bullet points, each one short sentence"],
  "next_steps": [{"text": "one concrete step", "owner": "care_team" or "patient"}],
  "reassurance": "one warm closing sentence"
}"""


class TemplateNarrator:
    """Deterministic summary assembled from the same structured facts."""

    source = "template"

    def generate(self, intake: dict[str, Any], result: dict[str, Any]) -> dict[str, Any]:
        tier = result["risk_tier"]
        meta = TIER_META[tier]

        phrases = [d["patient_phrase"] for d in result["drivers"] if d.get("patient_phrase")][:3]
        strengths = [d["patient_phrase"] for d in result["protective_factors"] if d.get("patient_phrase")][:2]

        if tier in {"very_high", "high"}:
            headline = "Your care team would like to check in with you more often"
        elif tier == "rising":
            headline = "A few things are worth keeping an eye on"
        else:
            headline = "Things look steady right now"

        body = [meta["patient_line"]]
        if phrases:
            joined = phrases[0] if len(phrases) == 1 else (
                ", ".join(phrases[:-1]) + " and " + phrases[-1])
            body.append(f"The main things we looked at were {joined}.")
        if strengths:
            body.append(f"It also helps that you have {strengths[0]}.")
        body.append(
            "This is not a prediction about you as a person. It is a way for your care team "
            "to make sure the right support reaches you before a small problem becomes a big one."
        )

        what_this_means: list[str] = []
        if result["expected_in_100"] >= 1:
            what_this_means.append(
                f"Out of 100 people with a health history like yours, about "
                f"{result['expected_in_100']} may need urgent hospital care in the next year. "
                "That also means most do not."
            )
        for flag in result["red_flags"][:3]:
            what_this_means.append(flag["patient_detail"])
        if not what_this_means:
            what_this_means.append("Nothing in your history stands out as needing urgent attention.")

        next_steps = [{"text": meta["action"], "owner": "care_team"}]
        if intake.get("has_primary_care") == 0:
            next_steps.append({"text": "We will help you find a regular doctor.", "owner": "care_team"})
        if intake.get("transportation_barrier") == 1:
            next_steps.append({"text": "Tell us if getting to an appointment is a problem, "
                                       "and we can arrange a ride.", "owner": "patient"})
        if intake.get("medication_adherence_pdc", 1.0) < 0.8:
            next_steps.append({"text": "Bring all of your medicines to your next visit, "
                                       "including the ones you have stopped.", "owner": "patient"})
        if intake.get("food_insecurity") == 1:
            next_steps.append({"text": "We can connect you with food support in your area.",
                               "owner": "care_team"})
        next_steps.append({"text": "Ask any question you have. Nothing is too small.", "owner": "patient"})

        return {
            "headline": headline,
            "body": "\n\n".join(body),
            "what_this_means": what_this_means[:4],
            "next_steps": next_steps[:5],
            "reassurance": "Your care team is on your side, and you do not have to manage this alone.",
            "source": self.source,
        }


class ClaudeNarrator:
    """LLM-written summary, constrained to the facts and validated on return."""

    source = "llm"

    def __init__(self) -> None:
        self.settings = get_settings()

    def _facts(self, intake: dict[str, Any], result: dict[str, Any]) -> dict[str, Any]:
        """Only the already-computed facts. No free text, no raw record."""
        return {
            "age": int(intake["age"]),
            "risk_level": result["risk_tier_label"],
            "people_out_of_100_with_similar_history": result["expected_in_100"],
            "what_the_care_team_plans_to_do": TIER_META[result["risk_tier"]]["action"],
            "main_factors": [
                {"factor": d["patient_phrase"] or d["label"], "makes_risk": d["direction"]}
                for d in result["drivers"][:5] if d.get("patient_phrase")
            ],
            "things_working_in_their_favour": [
                d["patient_phrase"] or d["label"] for d in result["protective_factors"][:3]
            ],
            "safety_concerns": [
                {"concern": f["patient_detail"], "severity": f["severity"]}
                for f in result["red_flags"][:5]
            ],
        }

    def generate(self, intake: dict[str, Any], result: dict[str, Any]) -> dict[str, Any]:
        payload = {
            "model": self.settings.anthropic_model,
            "max_tokens": self.settings.llm_max_tokens,
            "system": SYSTEM_PROMPT,
            "messages": [{
                "role": "user",
                "content": json.dumps(self._facts(intake, result), indent=2),
            }],
        }
        response = httpx.post(
            "https://api.anthropic.com/v1/messages",
            headers={
                "x-api-key": self.settings.anthropic_api_key or "",
                "anthropic-version": "2023-06-01",
                "content-type": "application/json",
            },
            json=payload,
            timeout=self.settings.llm_timeout_seconds,
        )
        response.raise_for_status()
        data = response.json()
        text = "".join(block.get("text", "") for block in data.get("content", [])
                       if block.get("type") == "text").strip()
        parsed = _parse_json_object(text)
        _validate_narrative(parsed)
        parsed["source"] = self.source
        return parsed


def _parse_json_object(text: str) -> dict[str, Any]:
    """Tolerate a model that wraps its JSON in prose or code fences."""
    text = re.sub(r"^```(?:json)?|```$", "", text.strip(), flags=re.MULTILINE).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, re.S)
        if not match:
            raise
        return json.loads(match.group(0))


BANNED_TERMS = ("super utilizer", "super-utilizer", "risk score", "algorithm", "the model predicts")


def _validate_narrative(parsed: dict[str, Any]) -> None:
    """Reject anything structurally wrong or off-tone. Failure means fallback."""
    required = {"headline", "body", "what_this_means", "next_steps", "reassurance"}
    missing = required - set(parsed)
    if missing:
        raise ValueError(f"narrative missing fields: {sorted(missing)}")
    if not isinstance(parsed["what_this_means"], list) or not parsed["what_this_means"]:
        raise ValueError("what_this_means must be a non-empty list")
    if not isinstance(parsed["next_steps"], list) or not parsed["next_steps"]:
        raise ValueError("next_steps must be a non-empty list")

    normalised: list[dict[str, str]] = []
    for step in parsed["next_steps"]:
        if isinstance(step, str):
            normalised.append({"text": step, "owner": "care_team"})
        elif isinstance(step, dict) and "text" in step:
            owner = step.get("owner")
            normalised.append({"text": str(step["text"]),
                               "owner": owner if owner in {"care_team", "patient"} else "care_team"})
        else:
            raise ValueError("malformed next_steps entry")
    parsed["next_steps"] = normalised

    blob = " ".join([
        str(parsed["headline"]), str(parsed["body"]), str(parsed["reassurance"]),
        " ".join(map(str, parsed["what_this_means"])),
    ]).lower()
    for term in BANNED_TERMS:
        if term in blob:
            raise ValueError(f"narrative used prohibited term: {term!r}")
    if len(str(parsed["headline"])) > 160:
        raise ValueError("headline too long")


def build_patient_summary(intake: dict[str, Any], result: dict[str, Any]) -> dict[str, Any]:
    """Try the LLM, fall back to the template. Never raises."""
    settings = get_settings()
    if settings.anthropic_api_key:
        try:
            return ClaudeNarrator().generate(intake, result)
        except Exception as exc:  # noqa: BLE001 - degradation must be total
            log.warning("LLM narrative unavailable (%s: %s); using template",
                        type(exc).__name__, exc)
    return TemplateNarrator().generate(intake, result)


def narrative_provider_name() -> str:
    return "claude" if get_settings().anthropic_api_key else "template"
