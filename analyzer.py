import json
from typing import Any, Optional

from google import genai

from config import GOOGLE_API_KEY, GEMINI_MODEL

from utils import _close_open_json_delimiters, _json_candidates, _strip_json_fence, _string_or_none, _string_list, _string_or_empty

client = genai.Client(api_key=GOOGLE_API_KEY)

DISCLAIMER = (
    "This output was generated automatically by a language model based on the "
    "consultation transcript. It is NOT a medical diagnosis and may contain errors "
    "or omissions. It must be reviewed and validated by a licensed healthcare "
    "professional before any clinical decision is made."
)

ANALYSIS_PROMPT = """
You are a clinical documentation support assistant helping a physician review a consultation.

Based ONLY on the content of the medical consultation transcript below, produce a structured analysis in JSON.

Rules:

- Base your analysis exclusively on what was actually said in the transcript.
- Do not invent symptoms, history, medications or diagnoses.
- Do NOT provide a definitive diagnosis.
- List only possible conditions/hypotheses for a licensed physician to consider.
- If there is not enough information, return an empty list for "possible_conditions".
- If urgent symptoms are mentioned, include them in "red_flags".
- Respond ONLY with a valid JSON object.
- Do not include markdown.
- Do not include explanations outside the JSON.

Schema:

{
  "chief_complaint": "patient's main complaint, or null if unclear",
  "symptoms_reported": [],
  "relevant_history": [],
  "possible_conditions": [
    {
      "condition": "",
      "rationale": "",
      "confidence": "low | medium | high"
    }
  ],
  "red_flags": [],
  "recommendation": "",
  "notes": ""
}

Consultation transcript:

{{ transcript }}
"""

ANALYSIS_RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {
        "chief_complaint": {"type": "string", "nullable": True},
        "symptoms_reported": {"type": "array", "items": {"type": "string"}},
        "relevant_history": {"type": "array", "items": {"type": "string"}},
        "possible_conditions": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "condition": {"type": "string"},
                    "rationale": {"type": "string"},
                    "confidence": {
                        "type": "string",
                        "enum": ["low", "medium", "high"],
                    },
                },
                "required": ["condition", "rationale", "confidence"],
            },
        },
        "red_flags": {"type": "array", "items": {"type": "string"}},
        "recommendation": {"type": "string", "nullable": True},
        "notes": {"type": "string"},
    },
    "required": [
        "chief_complaint",
        "symptoms_reported",
        "relevant_history",
        "possible_conditions",
        "red_flags",
        "recommendation",
        "notes",
    ],
}


def analyze_consultation(transcript_text: str) -> dict:
    prompt = ANALYSIS_PROMPT.replace(
        "{{ transcript }}",
        transcript_text,
    )

    response = client.models.generate_content(
        model=GEMINI_MODEL,
        contents=prompt,
        config={
            "temperature": 0,
            "max_output_tokens": 4096,
            "response_mime_type": "application/json",
            "response_schema": ANALYSIS_RESPONSE_SCHEMA,
        },
    )

    analysis = _parse_json_response(response.text)
    analysis["disclaimer"] = DISCLAIMER

    return analysis


def _parse_json_response(content: str) -> dict:
    cleaned = _strip_json_fence(content)
    last_error = None

    for candidate in _json_candidates(cleaned):
        try:
            return _normalize_analysis(json.loads(candidate))
        except json.JSONDecodeError as exc:
            last_error = exc

        try:
            parsed, _ = json.JSONDecoder().raw_decode(candidate)
            return _normalize_analysis(parsed)
        except json.JSONDecodeError as exc:
            last_error = exc

        repaired = _close_open_json_delimiters(candidate)
        if repaired == candidate:
            continue

        try:
            return _normalize_analysis(json.loads(repaired))
        except json.JSONDecodeError as exc:
            last_error = exc

    return _fallback_analysis(content, last_error)


def _normalize_analysis(value: Any) -> dict:
    if not isinstance(value, dict):
        return _fallback_analysis(str(value), None)

    return {
        "chief_complaint": _string_or_none(value.get("chief_complaint")),
        "symptoms_reported": _string_list(value.get("symptoms_reported")),
        "relevant_history": _string_list(value.get("relevant_history")),
        "possible_conditions": _condition_list(value.get("possible_conditions")),
        "red_flags": _string_list(value.get("red_flags")),
        "recommendation": _string_or_none(value.get("recommendation")),
        "notes": _string_or_empty(value.get("notes")),
    }


def _condition_list(value: Any) -> list[dict]:
    if not isinstance(value, list):
        return []

    conditions = []
    for item in value:
        if isinstance(item, str):
            item = {"condition": item, "rationale": "", "confidence": "low"}
        if not isinstance(item, dict):
            continue

        confidence = str(item.get("confidence") or "low").strip().lower()
        if confidence not in {"low", "medium", "high"}:
            confidence = "low"

        condition = _string_or_empty(item.get("condition"))
        rationale = _string_or_empty(item.get("rationale"))
        if condition or rationale:
            conditions.append(
                {
                    "condition": condition,
                    "rationale": rationale,
                    "confidence": confidence,
                }
            )

    return conditions


def _fallback_analysis(content: str, error: Optional[json.JSONDecodeError]) -> dict:
    notes = (
        "Nao foi possivel interpretar a resposta do modelo como JSON valido. "
        "O texto original foi preservado em 'raw_response'."
    )

    if error is not None:
        notes = f"{notes} Erro: {error.msg} na posicao {error.pos}."

    return {
        "chief_complaint": None,
        "symptoms_reported": [],
        "relevant_history": [],
        "possible_conditions": [],
        "red_flags": [],
        "recommendation": None,
        "notes": notes,
        "raw_response": content,
    }
