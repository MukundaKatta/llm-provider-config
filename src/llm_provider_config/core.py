"""Structured configuration for LLM providers with env-var loading.

Define provider settings once (API key, base URL, timeouts, retry policy)
and load them from environment variables or supply them directly.
Validate before use so you catch misconfiguration early.

Example::

    import os
    from llm_provider_config import ProviderConfig

    os.environ["ANTHROPIC_API_KEY"] = "sk-ant-placeholder"
    config = ProviderConfig.from_env("anthropic")

    issues = config.validate()
    if not issues:
        print(f"Ready: {config.provider_name} at {config.base_url}")
    else:
        for issue in issues:
            print(issue.message)
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any

# ---------------------------------------------------------------------------
# Defaults per provider
# ---------------------------------------------------------------------------

_PROVIDER_DEFAULTS: dict[str, dict[str, Any]] = {
    "anthropic": {
        "base_url": "https://api.anthropic.com",
        "timeout_seconds": 60.0,
        "max_retries": 3,
        "api_key_env": "ANTHROPIC_API_KEY",
    },
    "openai": {
        "base_url": "https://api.openai.com/v1",
        "timeout_seconds": 60.0,
        "max_retries": 3,
        "api_key_env": "OPENAI_API_KEY",
    },
    "gemini": {
        "base_url": "https://generativelanguage.googleapis.com",
        "timeout_seconds": 60.0,
        "max_retries": 3,
        "api_key_env": "GEMINI_API_KEY",
    },
    "groq": {
        "base_url": "https://api.groq.com/openai/v1",
        "timeout_seconds": 30.0,
        "max_retries": 2,
        "api_key_env": "GROQ_API_KEY",
    },
    "bedrock": {
        "base_url": "",  # resolved by boto3 per region
        "timeout_seconds": 120.0,
        "max_retries": 3,
        "api_key_env": "",  # uses AWS credentials
    },
}


# ---------------------------------------------------------------------------
# ConfigIssue
# ---------------------------------------------------------------------------


@dataclass
class ConfigIssue:
    """A single validation problem in a :class:`ProviderConfig`.

    Attributes:
        field:   Name of the config field that has a problem.
        message: Human-readable description.
        level:   ``"error"`` (must fix) or ``"warning"`` (advisory).
    """

    field: str
    message: str
    level: str = "error"

    def to_dict(self) -> dict[str, str]:
        """Return a JSON-serialisable dict."""
        return {"field": self.field, "message": self.message, "level": self.level}

    def __repr__(self) -> str:
        return (
            f"ConfigIssue(field={self.field!r},"
            f" level={self.level!r},"
            f" message={self.message!r})"
        )


# ---------------------------------------------------------------------------
# ProviderConfig
# ---------------------------------------------------------------------------


@dataclass
class ProviderConfig:
    """Configuration for a single LLM provider.

    Attributes:
        provider_name:    Canonical provider id (e.g. ``"anthropic"``).
        api_key:          API key or token.  Empty string = not set.
        base_url:         Base URL for the provider's API.
        timeout_seconds:  HTTP request timeout in seconds.
        max_retries:      Maximum number of retry attempts on transient errors.
        extra_headers:    Additional HTTP headers to include in every request.
        notes:            Optional free-text notes.
    """

    provider_name: str
    api_key: str = ""
    base_url: str = ""
    timeout_seconds: float = 60.0
    max_retries: int = 3
    extra_headers: dict[str, str] = field(default_factory=dict)
    notes: str = ""

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def validate(self) -> list[ConfigIssue]:
        """Check for configuration problems.

        Returns:
            List of :class:`ConfigIssue` objects.  Empty means all good.
        """
        issues: list[ConfigIssue] = []

        if not self.provider_name.strip():
            issues.append(
                ConfigIssue(
                    field="provider_name",
                    message="provider_name must not be empty",
                )
            )

        # API key required for all providers except bedrock (uses AWS creds)
        if self.provider_name.lower() != "bedrock" and not self.api_key.strip():
            issues.append(
                ConfigIssue(
                    field="api_key",
                    message=f"api_key is required for provider {self.provider_name!r}",
                )
            )

        if self.timeout_seconds <= 0:
            issues.append(
                ConfigIssue(
                    field="timeout_seconds",
                    message=f"timeout_seconds must be > 0, got {self.timeout_seconds}",
                )
            )

        if self.max_retries < 0:
            issues.append(
                ConfigIssue(
                    field="max_retries",
                    message=f"max_retries must be >= 0, got {self.max_retries}",
                )
            )

        if self.timeout_seconds > 300:
            issues.append(
                ConfigIssue(
                    field="timeout_seconds",
                    message=(
                        f"timeout_seconds={self.timeout_seconds} is unusually large;"
                        " consider 60–120 s"
                    ),
                    level="warning",
                )
            )

        return issues

    @property
    def is_valid(self) -> bool:
        """``True`` when :meth:`validate` finds no errors (warnings ok)."""
        return all(i.level != "error" for i in self.validate())

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable dict.  The api_key is redacted."""
        redacted = "***" if self.api_key else ""
        return {
            "provider_name": self.provider_name,
            "api_key": redacted,
            "base_url": self.base_url,
            "timeout_seconds": self.timeout_seconds,
            "max_retries": self.max_retries,
            "extra_headers": dict(self.extra_headers),
            "notes": self.notes,
        }

    def __repr__(self) -> str:
        masked = "***" if self.api_key else "(not set)"
        return (
            f"ProviderConfig(provider={self.provider_name!r},"
            f" api_key={masked},"
            f" base_url={self.base_url!r})"
        )

    # ------------------------------------------------------------------
    # Factory methods
    # ------------------------------------------------------------------

    @classmethod
    def from_env(
        cls,
        provider_name: str,
        *,
        env: dict[str, str] | None = None,
    ) -> ProviderConfig:
        """Load a :class:`ProviderConfig` from environment variables.

        Uses the provider's default ``api_key_env`` variable name plus
        optional ``<PROVIDER>_BASE_URL`` and ``<PROVIDER>_TIMEOUT``
        overrides.

        Args:
            provider_name: Provider id (case-insensitive).
            env:           Optional mapping to use instead of :data:`os.environ`.
                           Useful in tests.

        Returns:
            A populated :class:`ProviderConfig`.
        """
        _env = env if env is not None else dict(os.environ)
        key = provider_name.lower()
        defaults = dict(_PROVIDER_DEFAULTS.get(key, {}))

        # Resolve api_key
        api_key_env = defaults.pop("api_key_env", "")
        api_key = _env.get(api_key_env, "") if api_key_env else ""

        # Allow per-provider overrides via env
        base_url_override = _env.get(f"{key.upper()}_BASE_URL", "")
        base_url = base_url_override or defaults.get("base_url", "")

        timeout_str = _env.get(f"{key.upper()}_TIMEOUT", "")
        try:
            timeout = (
                float(timeout_str)
                if timeout_str
                else defaults.get("timeout_seconds", 60.0)
            )
        except ValueError:
            timeout = defaults.get("timeout_seconds", 60.0)

        return cls(
            provider_name=provider_name.lower(),
            api_key=api_key,
            base_url=base_url,
            timeout_seconds=timeout,
            max_retries=defaults.get("max_retries", 3),
        )

    @classmethod
    def for_anthropic(cls, *, api_key: str = "") -> ProviderConfig:
        """Return a :class:`ProviderConfig` pre-configured for Anthropic."""
        d = _PROVIDER_DEFAULTS["anthropic"]
        return cls(
            provider_name="anthropic",
            api_key=api_key,
            base_url=d["base_url"],
            timeout_seconds=d["timeout_seconds"],
            max_retries=d["max_retries"],
        )

    @classmethod
    def for_openai(cls, *, api_key: str = "") -> ProviderConfig:
        """Return a :class:`ProviderConfig` pre-configured for OpenAI."""
        d = _PROVIDER_DEFAULTS["openai"]
        return cls(
            provider_name="openai",
            api_key=api_key,
            base_url=d["base_url"],
            timeout_seconds=d["timeout_seconds"],
            max_retries=d["max_retries"],
        )

    @classmethod
    def for_gemini(cls, *, api_key: str = "") -> ProviderConfig:
        """Return a :class:`ProviderConfig` pre-configured for Gemini."""
        d = _PROVIDER_DEFAULTS["gemini"]
        return cls(
            provider_name="gemini",
            api_key=api_key,
            base_url=d["base_url"],
            timeout_seconds=d["timeout_seconds"],
            max_retries=d["max_retries"],
        )


# ---------------------------------------------------------------------------
# ConfigRegistry
# ---------------------------------------------------------------------------


class ConfigRegistry:
    """Hold multiple :class:`ProviderConfig` objects keyed by provider name.

    Args:
        configs: Optional initial list of configs.

    Raises:
        ValueError: If two configs share the same ``provider_name``.
    """

    def __init__(self, configs: list[ProviderConfig] | None = None) -> None:
        self._configs: dict[str, ProviderConfig] = {}
        for c in configs or []:
            self.register(c)

    def register(self, config: ProviderConfig) -> None:
        """Add *config* to the registry.

        Raises:
            ValueError: If ``config.provider_name`` is already registered.
        """
        key = config.provider_name.lower()
        if key in self._configs:
            raise ValueError(f"Provider {config.provider_name!r} already registered")
        self._configs[key] = config

    def get(self, provider_name: str) -> ProviderConfig | None:
        """Return the config for *provider_name*, or ``None``."""
        return self._configs.get(provider_name.lower())

    def remove(self, provider_name: str) -> None:
        """Remove a provider from the registry.

        Raises:
            KeyError: If the provider is not registered.
        """
        key = provider_name.lower()
        if key not in self._configs:
            raise KeyError(f"Provider {provider_name!r} not found")
        del self._configs[key]

    @property
    def provider_names(self) -> list[str]:
        """All registered provider names."""
        return list(self._configs.keys())

    def validate_all(self) -> dict[str, list[ConfigIssue]]:
        """Run :meth:`~ProviderConfig.validate` on every registered config.

        Returns:
            Dict mapping provider name → list of issues.  Empty list = ok.
        """
        return {name: cfg.validate() for name, cfg in self._configs.items()}

    def __len__(self) -> int:
        return len(self._configs)

    def __repr__(self) -> str:
        return f"ConfigRegistry(providers={self.provider_names})"
