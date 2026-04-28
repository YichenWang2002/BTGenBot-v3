"""Base interface for LLM generation backends."""

from __future__ import annotations


class BaseLLMBackend:
    def generate(self, prompt: str) -> str:
        raise NotImplementedError

