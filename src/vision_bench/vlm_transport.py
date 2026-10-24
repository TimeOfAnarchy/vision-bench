"""Non-streaming image-chat HTTP transport; credentials are never persisted."""

from __future__ import annotations

import base64
import hashlib
import http.client
import io
import urllib.error
import urllib.parse
import urllib.request
import warnings
from dataclasses import dataclass
from pathlib import Path

from PIL import Image

from .errors import BenchmarkError
from .io import canonical_json

MAX_RESPONSE_BYTES = 4 * 1024 * 1024
IMAGE_PREPROCESS_VERSION = "rgb-white-lanczos-jpeg-no-exif-v1"


@dataclass
class TransportError(Exception):
    kind: str
    status: int | None = None
    body: bytes = b""


def validate_endpoint(endpoint: str) -> str:
    value = urllib.parse.urlsplit(endpoint)
    if (
        value.scheme not in {"http", "https"}
        or not value.hostname
        or value.username is not None
        or value.password is not None
        or value.query
        or value.fragment
    ):
        raise BenchmarkError("Endpoint must be an HTTP(S) URL without credentials, query or fragment.")
    _ = value.port
    return endpoint


def prepare_image(path: Path, expected_sha256: str, config: dict) -> tuple[bytes, dict]:
    with path.open("rb") as handle:
        source = handle.read(64 * 1024 * 1024 + 1)
    if len(source) > 64 * 1024 * 1024 or hashlib.sha256(source).hexdigest() != expected_sha256:
        raise BenchmarkError("Image size/hash validation failed.")
    with warnings.catch_warnings():
        warnings.simplefilter("error", Image.DecompressionBombWarning)
        with Image.open(io.BytesIO(source)) as image:
            if (
                max(image.size) > 8192
                or image.width * image.height > 12_000_000
                or getattr(image, "n_frames", 1) != 1
            ):
                raise BenchmarkError("VLM input must be a single-frame image within 8192 pixels / 12MP.")
            rgba = image.convert("RGBA")
            canvas = Image.new("RGBA", image.size, "white")
            canvas.alpha_composite(rgba)
            rgb = canvas.convert("RGB")
            rgb.thumbnail((config["max_side"], config["max_side"]), Image.Resampling.LANCZOS)
            output = io.BytesIO()
            rgb.save(output, format="JPEG", quality=config["jpeg_quality"])
            encoded = output.getvalue()
            return encoded, {
                "sha256": hashlib.sha256(encoded).hexdigest(),
                "width": rgb.width,
                "height": rgb.height,
                "bytes": len(encoded),
            }


def chat_request(task: dict, profile: dict, image: bytes) -> dict:
    payload = {
        "model": profile["model"],
        "stream": False,
        **profile["generation"],
        "messages": [
            {"role": "system", "content": task["system_prompt"]},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": task["user_prompt"]},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": "data:image/jpeg;base64," + base64.b64encode(image).decode("ascii")
                        },
                    },
                ],
            },
        ],
    }
    if profile["response_format"] == "json_object":
        payload["response_format"] = {"type": "json_object"}
    elif profile["response_format"] == "json_schema":
        payload["response_format"] = {
            "type": "json_schema",
            "json_schema": {"name": "vision_output", "strict": True, "schema": task["output_schema"]},
        }
    return payload


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


def post_chat(
    endpoint: str, payload: dict, *, timeout: float, api_key: str | None = None
) -> tuple[int, bytes]:
    headers = {"Content-Type": "application/json", "Accept": "application/json"}
    if api_key:
        headers["Authorization"] = "Bearer " + api_key
    request = urllib.request.Request(
        validate_endpoint(endpoint),
        data=canonical_json(payload).encode("utf-8"),
        headers=headers,
        method="POST",
    )
    # Do not leak local-image requests to ambient HTTP proxies or follow redirects.
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}), _NoRedirect())
    try:
        try:
            response = opener.open(request, timeout=timeout)
        except urllib.error.HTTPError as exc:
            response = exc
        with response:
            body = response.read(MAX_RESPONSE_BYTES + 1)
            if len(body) > MAX_RESPONSE_BYTES:
                raise TransportError("response_too_large", response.code, body[:MAX_RESPONSE_BYTES])
            return response.code, body
    except TimeoutError as exc:
        raise TransportError("timeout") from exc
    except urllib.error.URLError as exc:
        kind = "timeout" if isinstance(exc.reason, TimeoutError) else "network_error"
        raise TransportError(kind) from exc
    except (OSError, http.client.HTTPException) as exc:
        raise TransportError("network_error") from exc
