#!/usr/bin/env python3

from __future__ import annotations

import argparse
import base64
import json
import os
import sys
from pathlib import Path
from typing import Any
from urllib import error, request


DEFAULT_WORKER_URL = os.environ.get("CHATLLM_OMNIVOICE_WORKER_URL", "http://127.0.0.1:8021")
TRUE_VALUES = {"1", "true", "yes", "on"}
FALSE_VALUES = {"0", "false", "no", "off"}


def _parse_bool(name: str, value: str) -> bool:
    lowered = value.strip().lower()
    if lowered in TRUE_VALUES:
        return True
    if lowered in FALSE_VALUES:
        return False
    raise ValueError(f"{name} must be a boolean value")


def _parse_int(name: str, value: str) -> int:
    try:
        return int(value)
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer") from exc


def _parse_float(name: str, value: str) -> float:
    try:
        return float(value)
    except ValueError as exc:
        raise ValueError(f"{name} must be a number") from exc


def _load_prompt(prompt: str | None, prompt_file: str | None) -> str:
    if prompt_file:
        return Path(prompt_file).read_text(encoding="utf-8")
    if prompt is not None:
        return prompt
    raise ValueError("one of --prompt/--prompt_file or request_json.prompt is required")


def _normalize_worker_url(value: str | None) -> str:
    raw = (value or "").strip() or DEFAULT_WORKER_URL
    return raw.rstrip("/")


def _build_generation_payload(additional: dict[str, str]) -> dict[str, Any]:
    generation: dict[str, Any] = {}

    float_keys = {
        "guidance_scale",
        "t_shift",
        "speed",
        "duration",
        "layer_penalty_factor",
        "position_temperature",
        "class_temperature",
        "audio_chunk_duration",
        "audio_chunk_threshold",
    }
    int_keys = {"num_step"}
    bool_keys = {"denoise", "preprocess_prompt", "postprocess_output"}

    for key, value in additional.items():
        if value is None:
            continue
        raw = value.strip()
        if raw == "":
            continue
        if key in float_keys:
            generation[key] = _parse_float(key, raw)
        elif key in int_keys:
            generation[key] = _parse_int(key, raw)
        elif key in bool_keys:
            generation[key] = _parse_bool(key, raw)

    return generation


def _build_worker_payload(prompt: str, additional: dict[str, str]) -> dict[str, Any]:
    instruct = (additional.get("instruct") or additional.get("voice") or "").strip()
    language = (additional.get("language") or "").strip()
    if language.lower() == "auto":
        language = ""

    ref_audio_path = (additional.get("ref_audio_file") or additional.get("ref_audio_path") or "").strip()
    ref_audio_url = (additional.get("ref_audio_url") or "").strip()
    ref_audio_wav_base64 = (additional.get("ref_audio_wav_base64") or "").strip()
    ref_audio_filename = (additional.get("ref_audio_filename") or "").strip()
    ref_text = (additional.get("ref_text") or "").strip()

    ref_sources = [value for value in (ref_audio_path, ref_audio_url, ref_audio_wav_base64) if value]
    if ref_sources and not ref_text:
        raise ValueError("ref_text is required when reference audio is provided")
    if ref_text and not ref_sources:
        raise ValueError("ref_text requires reference audio")

    payload: dict[str, Any] = {
        "text": prompt,
        "generation": _build_generation_payload(additional),
        "output": {
            "format": "wav",
            "codec": "pcm_s16le",
            "bitrate_kbps": 48,
            "return_base64": False,
            "include_wav_base64": True,
        },
    }

    if language:
        payload["language"] = language
    if instruct:
        payload["instruct"] = instruct
    if ref_text:
        payload["ref_text"] = ref_text
    if ref_audio_path:
        payload["ref_audio_path"] = ref_audio_path
    if ref_audio_url:
        payload["ref_audio_url"] = ref_audio_url
    if ref_audio_wav_base64:
        payload["ref_audio_wav_base64"] = ref_audio_wav_base64
    if ref_audio_filename:
        payload["ref_audio_filename"] = ref_audio_filename

    return payload


def _load_request_spec(args: argparse.Namespace) -> dict[str, Any]:
    request_spec: dict[str, Any] = {}
    if args.request_json:
        request_spec = json.loads(Path(args.request_json).read_text(encoding="utf-8"))

    additional = dict(request_spec.get("additional") or {})
    for key, value in args.set or []:
        additional[key] = value

    prompt = _load_prompt(
        args.prompt if args.prompt_file is None else None,
        args.prompt_file,
    ) if not args.request_json else _load_prompt(
        request_spec.get("prompt"),
        request_spec.get("prompt_file"),
    )

    if not prompt.strip():
        raise ValueError("prompt must not be empty")

    tts_export = (
        ((args.tts_export or "") if args.tts_export is not None else (request_spec.get("tts_export") or "")).strip()
        if args.request_json
        else (args.tts_export or "").strip()
    )
    if not tts_export:
        raise ValueError("tts_export is required")

    worker_url = _normalize_worker_url(
        ((args.worker_url if args.worker_url is not None else request_spec.get("worker_url")) if args.request_json else args.worker_url)
        or additional.get("worker_url")
        or additional.get("omnivoice_worker_url")
    )

    timeout_seconds = (
        args.timeout_seconds if args.timeout_seconds is not None else request_spec.get("timeout_seconds")
    ) if args.request_json else args.timeout_seconds
    if timeout_seconds is None:
        raw_timeout = additional.get("timeout_seconds") or additional.get("worker_timeout_seconds")
        timeout_seconds = _parse_int("timeout_seconds", raw_timeout) if raw_timeout else 300
    else:
        timeout_seconds = int(timeout_seconds)

    return {
        "worker_url": worker_url,
        "timeout_seconds": timeout_seconds,
        "prompt": prompt,
        "tts_export": tts_export,
        "additional": additional,
    }


def _post_json(url: str, payload: dict[str, Any], timeout_seconds: int) -> dict[str, Any]:
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    req = request.Request(
        url,
        data=body,
        headers={"content-type": "application/json"},
        method="POST",
    )
    try:
        with request.urlopen(req, timeout=timeout_seconds) as resp:
            raw = resp.read()
    except error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"worker HTTP {exc.code}: {detail}") from exc
    except error.URLError as exc:
        raise RuntimeError(f"worker request failed: {exc.reason}") from exc

    try:
        return json.loads(raw.decode("utf-8"))
    except json.JSONDecodeError as exc:
        raise RuntimeError("worker returned invalid JSON") from exc


def _save_wav(output: dict[str, Any], target: Path) -> None:
    encoded = output.get("wav_base64") or output.get("audio_base64")
    if not encoded:
        raise RuntimeError("worker response is missing wav_base64/audio_base64")
    raw = base64.b64decode(encoded)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(raw)


def run_bridge(spec: dict[str, Any]) -> dict[str, Any]:
    worker_input = _build_worker_payload(spec["prompt"], spec["additional"])
    envelope = {
        "run": {
            "mode": "sync",
            "timeout_seconds": spec["timeout_seconds"],
        },
        "input": worker_input,
    }
    response = _post_json(
        f"{spec['worker_url']}/worker-sdk/run",
        envelope,
        spec["timeout_seconds"],
    )

    error_code = response.get("error_code")
    if error_code:
        raise RuntimeError(f"{error_code}: {response.get('error_message') or 'worker request failed'}")

    output = response.get("output") or {}
    _save_wav(output, Path(spec["tts_export"]))
    return output


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Bridge ChatLLM.cpp TTS requests to a local OmniVoice worker-sdk service.",
    )
    parser.add_argument("--request_json", help="JSON request file written by chatllm.cpp main")
    parser.add_argument("--worker_url", default=None, help=f"worker base URL (default: {DEFAULT_WORKER_URL})")
    parser.add_argument("--timeout_seconds", type=int, default=None, help="worker request timeout in seconds")
    parser.add_argument("-p", "--prompt", default=None, help="text to synthesize")
    parser.add_argument("--prompt_file", default=None, help="path to a UTF-8 prompt file")
    parser.add_argument("--tts_export", default=None, help="output WAV path")
    parser.add_argument("--set", nargs=2, action="append", default=[], metavar=("KEY", "VALUE"))
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    try:
        spec = _load_request_spec(args)
        output = run_bridge(spec)
    except Exception as exc:  # noqa: BLE001
        print(str(exc), file=sys.stderr)
        return 1

    summary = {
        "tool": output.get("tool"),
        "voice_mode": output.get("voice_mode"),
        "generated_sample_rate_hz": output.get("generated_sample_rate_hz"),
        "duration_seconds": output.get("duration_seconds"),
        "tts_export": spec["tts_export"],
    }
    print(json.dumps(summary, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
