"""
Utilities for working with OpenAI API for structured outputs.

This module provides helpers to:
- Create chat completions with message-style input
- Parse structured outputs into Pydantic models using JSON mode
- Handle both standard and beta structured output APIs
"""

from typing import Any, Dict, List, Optional, Sequence, Type, TypeVar, cast
import json

from pydantic import BaseModel


T = TypeVar("T", bound=BaseModel)


def _normalize_messages_input(messages: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Convert chat-style messages into standard format (list of role/content dicts).

    Accepts already-correct structures and passes them through unchanged.
    """
    normalized: List[Dict[str, Any]] = []
    for msg in messages:
        content = msg.get("content")
        if isinstance(content, list):
            normalized.append({"role": msg.get("role", "user"), "content": content})
        else:
            normalized.append({"role": msg.get("role", "user"), "content": str(content) if content is not None else ""})
    return normalized


def extract_message_text(response: Any) -> str:
    """
    Extract plain text from a chat completion response.

    Handles both SDK objects and dict responses.
    Returns an empty string if nothing is found.
    """
    try:
        # SDK response object
        if hasattr(response, 'choices') and len(response.choices) > 0:
            message = response.choices[0].message
            if hasattr(message, 'content') and message.content:
                return message.content.strip()
        # Dict response
        elif isinstance(response, dict):
            choices = response.get('choices', [])
            if choices and len(choices) > 0:
                message = choices[0].get('message', {})
                content = message.get('content', '')
                if content:
                    return content.strip()
    except Exception:
        pass
    return ""


async def create_text_completion(
    client: Any,
    *,
    model: str,
    messages: Sequence[Dict[str, Any]],
    temperature: float = 0.3,
    max_tokens: int = 4096,
) -> str:
    """
    Create a chat completion and extract the assistant text.

    Falls back to an empty string if no content is present.
    """
    normalized = _normalize_messages_input(messages)

    response = await client.chat.completions.create(
        model=model,
        messages=normalized,
        temperature=temperature,
        max_tokens=max_tokens,
    )

    return extract_message_text(response)


async def parse_pydantic_response(
    client: Any,
    *,
    model: str,
    messages: Sequence[Dict[str, Any]],
    response_format: Type[T],
    temperature: float = 0.3,
    max_tokens: int = 4096,
) -> T:
    """
    Parse structured output using OpenAI API into the provided Pydantic model type.

    Uses JSON mode with schema instructions to ensure valid JSON output.
    Falls back to text parsing if structured output is not available.
    """
    normalized = _normalize_messages_input(messages)

    # Build JSON schema from the Pydantic model
    try:
        schema = response_format.model_json_schema()
    except Exception:
        # Pydantic v1 fallback
        schema = response_format.schema()  # type: ignore[attr-defined]

    # Inject a strict schema instruction to ensure JSON-only output
    schema_str = json.dumps(schema, indent=2)
    schema_instruction = {
        "role": "system",
        "content": (
            "You must respond with ONLY a single valid JSON object that matches the following JSON Schema. "
            "Do not include any prose, code fences, markdown, or additional text. "
            "Output pure JSON only.\n\n"
            f"JSON Schema:\n{schema_str}"
        ),
    }

    # Prepend schema instruction
    messages_with_schema = [schema_instruction] + list(normalized)

    # Try using JSON mode for more reliable output
    try:
        response = await client.chat.completions.create(
            model=model,
            messages=messages_with_schema,
            temperature=temperature,
            max_tokens=max_tokens,
            response_format={"type": "json_object"},
        )
    except Exception:
        # Fall back to regular completion if JSON mode not supported
        response = await client.chat.completions.create(
            model=model,
            messages=messages_with_schema,
            temperature=temperature,
            max_tokens=max_tokens,
        )

    # Parse JSON from the response
    text_value = extract_message_text(response)
    if text_value:
        # Clean up any markdown code fences
        text_value = text_value.strip()
        if text_value.startswith("```json"):
            text_value = text_value[7:]
        elif text_value.startswith("```"):
            text_value = text_value[3:]
        if text_value.endswith("```"):
            text_value = text_value[:-3]
        text_value = text_value.strip()

        try:
            data = json.loads(text_value)
            try:
                return cast(T, response_format.model_validate(data))
            except Exception:
                return cast(T, response_format.parse_obj(data))  # type: ignore[attr-defined]
        except json.JSONDecodeError as exc:
            raise RuntimeError(f"Structured output parsing failed: invalid JSON in model output: {exc}\nOutput: {text_value[:500]}")

    raise RuntimeError("Structured output parsing failed: no content found in API response.")


# Backwards compatibility aliases
async def responses_create_text(
    client: Any,
    *,
    model: str,
    messages: Sequence[Dict[str, Any]],
    reasoning_effort: str = "low",
    text_verbosity: str = "medium",
) -> str:
    """
    Backwards compatible wrapper - uses standard chat completions.

    The reasoning_effort and text_verbosity parameters are ignored
    as they are not supported by standard OpenAI models.
    """
    return await create_text_completion(
        client,
        model=model,
        messages=messages,
    )


async def responses_parse_pydantic(
    client: Any,
    *,
    model: str,
    messages: Sequence[Dict[str, Any]],
    response_format: Type[T],
    reasoning_effort: str = "low",
    text_verbosity: str = "medium",
) -> T:
    """
    Backwards compatible wrapper - uses standard chat completions with JSON mode.

    The reasoning_effort and text_verbosity parameters are ignored
    as they are not supported by standard OpenAI models.
    """
    return await parse_pydantic_response(
        client,
        model=model,
        messages=messages,
        response_format=response_format,
    )
