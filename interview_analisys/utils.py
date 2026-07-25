import json
from typing import Any, Optional


def _strip_json_fence(content: str) -> str:
    cleaned = (content or "").strip()

    if not cleaned.startswith("```"):
        return cleaned

    lines = cleaned.splitlines()
    if lines and lines[0].strip().lower() in {"```", "```json"}:
        lines = lines[1:]
    if lines and lines[-1].strip() == "```":
        lines = lines[:-1]

    return "\n".join(lines).strip()


def _json_candidates(content: str) -> list[str]:
    candidates = [content]
    first_brace = content.find("{")

    if first_brace > 0:
        candidates.append(content[first_brace:])

    last_brace = content.rfind("}")
    if first_brace >= 0 and last_brace > first_brace:
        candidates.append(content[first_brace : last_brace + 1])

    return [candidate.strip() for candidate in candidates if candidate.strip()]


def _close_open_json_delimiters(content: str) -> str:
    stack = []
    in_string = False
    escaped = False

    for char in content:
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue

        if char == '"':
            in_string = True
        elif char == "{":
            stack.append("}")
        elif char == "[":
            stack.append("]")
        elif char in "}]":
            if not stack or stack[-1] != char:
                return content
            stack.pop()

    if in_string or not stack:
        return content

    return content + "".join(reversed(stack))


def _string_or_empty(value: Any) -> str:
    if value is None:
        return ""

    return str(value).strip()


def _string_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []

    return [str(item).strip() for item in value if str(item).strip()]


def _string_or_none(value: Any) -> Optional[str]:
    if value is None:
        return None

    text = str(value).strip()
    return text or None