"""
Factory wrapper for LLM providers via LiteLLM.

LiteLLM is a unified interface for many LLM backends
(Ollama, OpenAI, Anthropic, etc.). By pointing MODEL_NAME at
"ollama/mistral:latest", all requests are forwarded to the local
Ollama server without any cloud API keys.

Usage:
    model = LLMFactory()
    answer = model.generate(system_prompt="...", user_prompt="...")
"""

import os

import litellm

from src.config import settings


class LLMFactory:
    """Singleton wrapper that forwards prompts to any LiteLLM-supported model.

    Implemented as a singleton so the class is only instantiated once,
    avoiding redundant configuration lookups on every graph node call.
    """

    _instance = None

    def __new__(cls, *args, **kwargs):
        # Singleton: reuse the same instance across all callers
        if not cls._instance:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        # Guard against re-initialisation on repeated LLMFactory() calls
        if getattr(self, "_initialised", False):
            return
        self._initialised = True

    def generate(self, system_prompt: str, user_prompt: str) -> str:
        """Perform a single round-trip LLM call and return the response text.

        Args:
            system_prompt: Instructions for the model (role, constraints, format).
            user_prompt:   The user's actual question or task.

        Returns:
            The model's text response as a plain string.
        """
        # Read model config from settings; fall back gracefully to Ollama defaults
        model = getattr(settings, "MODEL_NAME", "ollama/mistral:latest")
        temperature = getattr(settings, "TEMPERATURE", 0.0)
        max_tokens = getattr(settings, "MAX_TOKENS", 2048)

        # LiteLLM reads OLLAMA_API_BASE for ollama/ models
        # Inside Docker: use service name 'ollama-service'
        # Outside Docker: use 'localhost'
        ollama_url = os.environ.get("OLLAMA_BASE_URL", "http://ollama-service:11434")
        os.environ["OLLAMA_API_BASE"] = ollama_url

        response = litellm.completion(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=temperature,
            max_tokens=max_tokens,
            timeout=120,
        )
        return response.choices[0].message.content
