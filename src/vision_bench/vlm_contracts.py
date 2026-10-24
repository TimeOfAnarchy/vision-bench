"""Versioned, application-independent image-chat tasks and response validation."""

from __future__ import annotations

import json
import math

from jsonschema import Draft202012Validator, SchemaError
from referencing import Registry
from referencing.exceptions import NoSuchResource

from .errors import BenchmarkError
from .io import canonical_json, identifier, positive_int

JSON_SCHEMA_DRAFT = "https://json-schema.org/draft/2020-12/schema"


def strict_json(text: str | bytes):
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError("Duplicate JSON key.")
            result[key] = value
        return result

    def invalid(value):
        raise ValueError("Non-finite JSON constant.")

    result = json.loads(text, object_pairs_hook=pairs, parse_constant=invalid)
    canonical_json(result)  # Also rejects overflowing floats such as 1e999.
    return result


def _no_resource(uri):
    raise NoSuchResource(ref=uri)


def task_validator(task: dict) -> Draft202012Validator:
    if task.get("schema_version") != 1:
        raise BenchmarkError("VLM task schema_version must be 1.")
    identifier(task.get("id"))
    identifier(task.get("revision"))
    for key in ("system_prompt", "user_prompt"):
        if not isinstance(task.get(key), str) or not task[key].strip():
            raise BenchmarkError(f"Task needs a nonempty {key}.")
    schema = task.get("output_schema")
    if not isinstance(schema, dict) or schema.get("type") != "object":
        raise BenchmarkError("Task output_schema must describe an object.")
    if schema.get("$schema", JSON_SCHEMA_DRAFT) != JSON_SCHEMA_DRAFT:
        raise BenchmarkError("Only JSON Schema draft 2020-12 is supported.")

    # All resources are embedded. No implicit network access during validation.
    def check_refs(value):
        if isinstance(value, dict):
            for key, child in value.items():
                if key in {"$ref", "$dynamicRef"} and (
                    not isinstance(child, str) or not child.startswith("#")
                ):
                    raise BenchmarkError("Only document-local schema references are supported.")
                check_refs(child)
        elif isinstance(value, list):
            for child in value:
                check_refs(child)

    check_refs(schema)
    try:
        Draft202012Validator.check_schema(schema)
    except SchemaError as exc:
        raise BenchmarkError("Invalid task output schema.") from exc
    fields = task.get("comparison_fields")
    if (
        not isinstance(fields, list)
        or not fields
        or not all(isinstance(key, str) for key in fields)
        or len(set(fields)) != len(fields)
    ):
        raise BenchmarkError("Task needs unique comparison_fields.")
    for key in fields:
        if (
            not isinstance(key, str)
            or key not in schema.get("properties", {})
            or key not in schema.get("required", [])
        ):
            raise BenchmarkError("Comparison fields must be required top-level schema properties.")
    return Draft202012Validator(schema, registry=Registry(retrieve=_no_resource))


def validate_profile(profile: dict) -> None:
    if profile.keys() - {
        "schema_version",
        "id",
        "model",
        "model_revision",
        "response_format",
        "image",
        "generation",
    }:
        raise BenchmarkError("Unknown VLM profile fields; connection credentials belong in the environment.")
    if profile.get("schema_version") != 1:
        raise BenchmarkError("VLM profile schema_version must be 1.")
    identifier(profile.get("id"))
    for key in ("model", "model_revision"):
        if not isinstance(profile.get(key), str) or not profile[key].strip():
            raise BenchmarkError(f"Profile needs an explicit {key}.")
    if profile.get("response_format") not in {"none", "json_object", "json_schema"}:
        raise BenchmarkError("Profile response_format must be none, json_object or json_schema.")
    image = profile.get("image", {})
    if not isinstance(image, dict) or not 64 <= positive_int(image.get("max_side"), name="max_side") <= 2048:
        raise BenchmarkError("Image max_side must be between 64 and 2048.")
    if not 1 <= positive_int(image.get("jpeg_quality"), name="jpeg_quality") <= 100:
        raise BenchmarkError("JPEG quality must be between 1 and 100.")
    generation = profile.get("generation")
    allowed = {"max_tokens", "temperature", "top_p", "seed", "chat_template_kwargs"}
    if (
        not isinstance(generation, dict)
        or not {"max_tokens", "temperature"} <= generation.keys()
        or generation.keys() - allowed
    ):
        raise BenchmarkError("Profile generation needs max_tokens/temperature and only supported options.")
    if positive_int(generation["max_tokens"], name="max_tokens") > 16384:
        raise BenchmarkError("max_tokens must not exceed 16384.")
    for key, low, high in (("temperature", 0, 2), ("top_p", 0, 1)):
        if key in generation:
            value = generation[key]
            if (
                isinstance(value, bool)
                or not isinstance(value, (float, int))
                or not math.isfinite(value)
                or not low <= value <= high
            ):
                raise BenchmarkError(f"Invalid generation {key}.")
    if "seed" in generation and (
        isinstance(generation["seed"], bool) or not isinstance(generation["seed"], int)
    ):
        raise BenchmarkError("Generation seed must be an integer.")
    if "chat_template_kwargs" in generation and not isinstance(generation["chat_template_kwargs"], dict):
        raise BenchmarkError("chat_template_kwargs must be an object.")
    canonical_json(profile)


def parse_response(body: bytes, task: dict, model: str, validator=None) -> dict:
    """Strict output parsing: no fence stripping, substring rescue or label coercion."""

    def failure(kind):
        return {"outcome": kind, "output": None, "reported_model": None, "usage": None}

    try:
        payload = strict_json(body)
    except (ValueError, UnicodeError, RecursionError):
        return failure("response_json_error")
    if not isinstance(payload, dict):
        return failure("response_shape_error")
    if payload.get("model") != model:
        reported = payload.get("model")
        return {
            **failure("model_mismatch"),
            "reported_model": reported if isinstance(reported, str) else None,
        }
    result = {**failure("response_shape_error"), "reported_model": model}
    choices = payload.get("choices")
    if not isinstance(choices, list) or len(choices) != 1 or not isinstance(choices[0], dict):
        return result
    choice = choices[0]
    message = choice.get("message")
    if not isinstance(message, dict) or message.get("role") != "assistant":
        return result
    usage = payload.get("usage")
    if isinstance(usage, dict):
        result["usage"] = {
            key: value
            for key, value in usage.items()
            if key in {"prompt_tokens", "completion_tokens", "total_tokens"}
            and isinstance(value, int)
            and not isinstance(value, bool)
            and value >= 0
        }
    if message.get("refusal") or choice.get("finish_reason") == "content_filter":
        return {**result, "outcome": "refusal"}
    if choice.get("finish_reason") == "length":
        return {**result, "outcome": "truncated"}
    if choice.get("finish_reason") != "stop" or message.get("tool_calls"):
        return result
    if not isinstance(message.get("content"), str):
        return result
    try:
        output = strict_json(message["content"])
    except (ValueError, UnicodeError, RecursionError):
        return {**result, "outcome": "output_json_error"}
    validator = validator if validator is not None else task_validator(task)
    try:
        valid = validator.is_valid(output)
    except Exception as exc:
        raise BenchmarkError("Task schema cannot resolve an embedded reference.") from exc
    if not valid:
        return {**result, "outcome": "schema_error"}
    return {**result, "outcome": "ok", "output": output}


def projection(output: dict, task: dict) -> str:
    return canonical_json({key: output[key] for key in task["comparison_fields"]})
