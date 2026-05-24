# llm-provider-config

Structured configuration for LLM providers with env-var loading and validation. Zero dependencies.

## Install

```bash
pip install llm-provider-config
```

## Quick start

```python
import os
from llm_provider_config import ProviderConfig

os.environ["ANTHROPIC_API_KEY"] = "sk-ant-..."

config = ProviderConfig.from_env("anthropic")
issues = config.validate()
if not issues:
    print(f"Ready: {config.provider_name} → {config.base_url}")
```

## API

### `ProviderConfig`

| Field | Type | Default | Description |
|---|---|---|---|
| `provider_name` | `str` | — | Provider id (e.g. `"anthropic"`) |
| `api_key` | `str` | `""` | API key (redacted in `repr` and `to_dict`) |
| `base_url` | `str` | `""` | API base URL |
| `timeout_seconds` | `float` | `60.0` | HTTP timeout |
| `max_retries` | `int` | `3` | Retry attempts |
| `extra_headers` | `dict` | `{}` | Additional request headers |

| Method | Description |
|---|---|
| `validate()` | Return list of `ConfigIssue` objects |
| `is_valid` | `True` when no errors (warnings OK) |
| `to_dict()` | JSON-serialisable dict (api_key redacted) |
| `from_env(provider, *, env)` | Load from environment variables |
| `for_anthropic(*, api_key)` | Pre-configured Anthropic preset |
| `for_openai(*, api_key)` | Pre-configured OpenAI preset |
| `for_gemini(*, api_key)` | Pre-configured Gemini preset |

### Environment variables

| Variable | Effect |
|---|---|
| `ANTHROPIC_API_KEY` | API key for Anthropic |
| `OPENAI_API_KEY` | API key for OpenAI |
| `GEMINI_API_KEY` | API key for Gemini |
| `GROQ_API_KEY` | API key for Groq |
| `<PROVIDER>_BASE_URL` | Override base URL |
| `<PROVIDER>_TIMEOUT` | Override timeout in seconds |

### `ConfigRegistry`

```python
from llm_provider_config import ConfigRegistry, ProviderConfig

registry = ConfigRegistry([
    ProviderConfig.for_anthropic(api_key="sk-ant-..."),
    ProviderConfig.for_openai(api_key="sk-..."),
])

config = registry.get("anthropic")
issues = registry.validate_all()  # {"anthropic": [], "openai": []}
```

## License

MIT
