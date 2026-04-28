"""Optional OpenAI-compatible chat completion backend."""

from __future__ import annotations

import http.client
import json
import os
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

from src.llm.backends.base import BaseLLMBackend


CLIENT_ERROR_STATUSES = {400, 401, 403, 404}
RETRYABLE_HTTP_STATUSES = {429, 500, 502, 503, 504}
MINIMAL_PROFILES = {"opencode-zen", "minimax"}
BACKOFF_SECONDS = [1, 3, 8]


class OpenAICompatibleBackend(BaseLLMBackend):
    def __init__(
        self,
        base_url: str | None = None,
        api_key: str | None = None,
        model: str | None = None,
        provider_profile: str | None = None,
        max_retries: int | None = None,
        debug_dir: str | Path | None = None,
    ) -> None:
        self.base_url = base_url or os.environ.get("OPENAI_BASE_URL")
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        self.model = model or os.environ.get("OPENAI_MODEL")
        self.provider_profile = (
            provider_profile or os.environ.get("OPENAI_PROVIDER_PROFILE") or "default"
        )
        self.max_retries = (
            max_retries
            if max_retries is not None
            else int(os.environ.get("OPENAI_MAX_RETRIES", "3"))
        )
        self.debug_dir = Path(debug_dir or "results/debug")

    def configured(self) -> bool:
        return bool(self.base_url and self.api_key and self.model)

    def generate(self, prompt: str) -> str:
        if not self.configured():
            raise RuntimeError(
                "OPENAI_BASE_URL, OPENAI_API_KEY, and OPENAI_MODEL are required for this backend"
            )

        attempted_models: list[str] = []
        last_error: str = ""
        for model in self.model_candidates():
            attempted_models.append(model)
            result, error, should_try_next_model = self._generate_with_model(prompt, model)
            if result is not None:
                return result
            last_error = error or "unknown error"
            if not should_try_next_model:
                break

        raise RuntimeError(
            "OpenAI-compatible request failed. "
            f"attempted_models={attempted_models}; last_error={last_error}"
        )

    def _generate_with_model(self, prompt: str, model: str) -> tuple[str | None, str, bool]:
        transient_error = ""
        for attempt in range(self.max_retries + 1):
            try:
                body = self._request(prompt, model)
                try:
                    parsed = json.loads(body)
                except json.JSONDecodeError as exc:
                    transient_error = self.format_error(
                        model=model,
                        error_type=type(exc).__name__,
                        message=str(exc),
                        response_body=body,
                    )
                    self.write_debug_error(transient_error, model, type(exc).__name__)
                    if attempt < self.max_retries:
                        self.sleep_before_retry(attempt)
                        continue
                    return None, transient_error, False
                return self.extract_content(parsed), "", False
            except urllib.error.HTTPError as exc:
                body = read_error_body(exc)
                formatted = self.format_error(
                    model=model,
                    error_type=type(exc).__name__,
                    message=str(exc),
                    status_code=exc.code,
                    response_body=body,
                )
                self.write_debug_error(formatted, model, type(exc).__name__)
                if exc.code in CLIENT_ERROR_STATUSES:
                    return None, formatted, True
                if exc.code in RETRYABLE_HTTP_STATUSES and attempt < self.max_retries:
                    self.sleep_before_retry(attempt)
                    continue
                return None, formatted, False
            except RETRYABLE_EXCEPTIONS as exc:
                transient_error = self.format_error(
                    model=model,
                    error_type=type(exc).__name__,
                    message=str(exc),
                )
                self.write_debug_error(transient_error, model, type(exc).__name__)
                if attempt < self.max_retries:
                    self.sleep_before_retry(attempt)
                    continue
                return None, transient_error, False
        return None, transient_error, False

    def _request(self, prompt: str, model: str) -> str:
        url = self.endpoint_url()
        payload = self.build_payload(prompt, model)
        headers = self.build_headers()
        self.write_debug_request(payload)
        request = urllib.request.Request(
            url,
            data=json.dumps(payload).encode("utf-8"),
            headers=headers,
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=120) as response:
            body = response.read().decode("utf-8")
        self.write_debug_response(body)
        return body

    def build_payload(self, prompt: str, model: str | None = None) -> dict[str, Any]:
        model_id = model or self.model
        payload: dict[str, Any] = {
            "model": model_id,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0,
        }
        if self.provider_profile in MINIMAL_PROFILES:
            payload["max_tokens"] = 4096
            payload["stream"] = False
        return payload

    def build_headers(self) -> dict[str, str]:
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        if self.provider_profile in MINIMAL_PROFILES:
            headers.update(
                {
                    "Accept": "application/json",
                    "Accept-Encoding": "identity",
                    "Connection": "close",
                }
            )
        return headers

    def model_candidates(self) -> list[str]:
        candidates = [self.model] if self.model else []
        fallback_text = os.environ.get("OPENAI_MODEL_FALLBACKS", "")
        candidates.extend(
            item.strip() for item in fallback_text.split(",") if item.strip()
        )
        deduped: list[str] = []
        for candidate in candidates:
            if candidate not in deduped:
                deduped.append(candidate)
        return deduped

    def endpoint_path(self) -> str:
        return "/chat/completions"

    def endpoint_url(self) -> str:
        return self.base_url.rstrip("/") + self.endpoint_path()

    def extract_content(self, parsed: dict[str, Any]) -> str:
        return parsed["choices"][0]["message"]["content"]

    def sleep_before_retry(self, attempt: int) -> None:
        time.sleep(BACKOFF_SECONDS[min(attempt, len(BACKOFF_SECONDS) - 1)])

    def format_error(
        self,
        model: str,
        error_type: str,
        message: str,
        status_code: int | None = None,
        response_body: str | None = None,
    ) -> str:
        parts = [
            f"status_code={status_code}" if status_code is not None else "status_code=None",
            f"OPENAI_BASE_URL={self.base_url}",
            f"OPENAI_MODEL={model}",
            f"provider_profile={self.provider_profile}",
            f"endpoint_path={self.endpoint_path()}",
            f"error_type={error_type}",
            f"message={message}",
        ]
        if response_body:
            parts.append(f"response_body={response_body[:1000]}")
        return self.sanitize("; ".join(parts))

    def sanitize(self, text: str) -> str:
        if self.api_key:
            return text.replace(self.api_key, "***")
        return text

    def debug_enabled(self) -> bool:
        return os.environ.get("OPENAI_DEBUG_REQUEST") == "1"

    def write_debug_request(self, payload: dict[str, Any]) -> None:
        if not self.debug_enabled():
            return
        self.debug_dir.mkdir(parents=True, exist_ok=True)
        debug_payload = {
            "endpoint": self.endpoint_url(),
            "endpoint_path": self.endpoint_path(),
            "provider_profile": self.provider_profile,
            "model": payload.get("model"),
            "payload": payload,
        }
        text = json.dumps(debug_payload, indent=2, ensure_ascii=False)
        (self.debug_dir / "openai_request_last.json").write_text(
            self.sanitize(text) + "\n",
            encoding="utf-8",
        )

    def write_debug_response(self, body: str) -> None:
        if not self.debug_enabled():
            return
        self.debug_dir.mkdir(parents=True, exist_ok=True)
        (self.debug_dir / "openai_response_last.txt").write_text(
            self.sanitize(body[:5000]), encoding="utf-8"
        )

    def write_debug_error(self, message: str, model: str, error_type: str) -> None:
        if not self.debug_enabled():
            return
        self.debug_dir.mkdir(parents=True, exist_ok=True)
        payload = {
            "error_type": error_type,
            "message": message,
            "model": model,
            "base_url": self.base_url,
            "endpoint": self.endpoint_url(),
            "endpoint_path": self.endpoint_path(),
            "provider_profile": self.provider_profile,
        }
        text = json.dumps(payload, indent=2, ensure_ascii=False)
        (self.debug_dir / "openai_error_last.txt").write_text(
            self.sanitize(text) + "\n", encoding="utf-8"
        )


RETRYABLE_EXCEPTIONS = (
    urllib.error.URLError,
    TimeoutError,
    http.client.IncompleteRead,
    ConnectionResetError,
)


def read_error_body(exc: urllib.error.HTTPError) -> str:
    try:
        body = exc.read()
    except Exception:
        return ""
    if isinstance(body, bytes):
        return body.decode("utf-8", errors="replace")
    return str(body)
