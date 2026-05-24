"""Tests for llm-provider-config."""

from __future__ import annotations

import pytest

from llm_provider_config import ConfigIssue, ConfigRegistry, ProviderConfig

# ---------------------------------------------------------------------------
# ConfigIssue
# ---------------------------------------------------------------------------


def test_config_issue_defaults():
    issue = ConfigIssue(field="api_key", message="missing")
    assert issue.level == "error"


def test_config_issue_warning():
    issue = ConfigIssue(field="timeout_seconds", message="large", level="warning")
    assert issue.level == "warning"


def test_config_issue_to_dict():
    issue = ConfigIssue(field="api_key", message="missing", level="error")
    d = issue.to_dict()
    assert d["field"] == "api_key"
    assert d["message"] == "missing"
    assert d["level"] == "error"


def test_config_issue_repr():
    issue = ConfigIssue(field="x", message="bad", level="error")
    r = repr(issue)
    assert "x" in r
    assert "bad" in r


# ---------------------------------------------------------------------------
# ProviderConfig — defaults
# ---------------------------------------------------------------------------


def test_provider_config_defaults():
    c = ProviderConfig(provider_name="anthropic")
    assert c.api_key == ""
    assert c.max_retries == 3
    assert c.timeout_seconds == 60.0
    assert c.extra_headers == {}


def test_provider_config_to_dict_redacts_key():
    c = ProviderConfig(provider_name="anthropic", api_key="sk-real-key")
    d = c.to_dict()
    assert d["api_key"] == "***"


def test_provider_config_to_dict_empty_key():
    c = ProviderConfig(provider_name="anthropic")
    d = c.to_dict()
    assert d["api_key"] == ""


def test_provider_config_repr_masks_key():
    c = ProviderConfig(provider_name="anthropic", api_key="sk-real-key")
    r = repr(c)
    assert "***" in r
    assert "sk-real-key" not in r


def test_provider_config_repr_not_set():
    c = ProviderConfig(provider_name="anthropic")
    r = repr(c)
    assert "(not set)" in r


# ---------------------------------------------------------------------------
# ProviderConfig.validate()
# ---------------------------------------------------------------------------


def test_valid_config():
    c = ProviderConfig(
        provider_name="anthropic",
        api_key="sk-ant-placeholder",
        base_url="https://api.anthropic.com",
        timeout_seconds=60.0,
    )
    assert c.is_valid
    assert c.validate() == []


def test_missing_api_key_error():
    c = ProviderConfig(provider_name="anthropic", api_key="")
    issues = c.validate()
    errors = [i for i in issues if i.level == "error"]
    assert any(i.field == "api_key" for i in errors)


def test_bedrock_no_api_key_ok():
    c = ProviderConfig(
        provider_name="bedrock",
        api_key="",
        timeout_seconds=120.0,
    )
    errors = [i for i in c.validate() if i.level == "error"]
    assert not any(i.field == "api_key" for i in errors)


def test_empty_provider_name_error():
    c = ProviderConfig(provider_name="", api_key="key")
    issues = c.validate()
    assert any(i.field == "provider_name" for i in issues)


def test_negative_timeout_error():
    c = ProviderConfig(provider_name="openai", api_key="key", timeout_seconds=-1.0)
    issues = c.validate()
    assert any(i.field == "timeout_seconds" and i.level == "error" for i in issues)


def test_zero_timeout_error():
    c = ProviderConfig(provider_name="openai", api_key="key", timeout_seconds=0.0)
    issues = c.validate()
    assert any(i.field == "timeout_seconds" and i.level == "error" for i in issues)


def test_negative_max_retries_error():
    c = ProviderConfig(provider_name="openai", api_key="key", max_retries=-1)
    issues = c.validate()
    assert any(i.field == "max_retries" for i in issues)


def test_large_timeout_warning():
    c = ProviderConfig(provider_name="anthropic", api_key="key", timeout_seconds=600.0)
    issues = c.validate()
    warnings = [i for i in issues if i.level == "warning"]
    assert any(i.field == "timeout_seconds" for i in warnings)


def test_is_valid_ignores_warnings():
    c = ProviderConfig(provider_name="anthropic", api_key="key", timeout_seconds=600.0)
    # Has a warning but no errors
    assert c.is_valid


# ---------------------------------------------------------------------------
# ProviderConfig.from_env()
# ---------------------------------------------------------------------------


def test_from_env_anthropic():
    env = {"ANTHROPIC_API_KEY": "sk-ant-test"}
    c = ProviderConfig.from_env("anthropic", env=env)
    assert c.api_key == "sk-ant-test"
    assert c.provider_name == "anthropic"
    assert "anthropic" in c.base_url


def test_from_env_openai():
    env = {"OPENAI_API_KEY": "sk-oai-test"}
    c = ProviderConfig.from_env("openai", env=env)
    assert c.api_key == "sk-oai-test"
    assert "openai" in c.base_url


def test_from_env_gemini():
    env = {"GEMINI_API_KEY": "ai-test-key"}
    c = ProviderConfig.from_env("gemini", env=env)
    assert c.api_key == "ai-test-key"


def test_from_env_missing_key_empty_string():
    env: dict[str, str] = {}
    c = ProviderConfig.from_env("anthropic", env=env)
    assert c.api_key == ""


def test_from_env_base_url_override():
    env = {
        "ANTHROPIC_API_KEY": "sk-ant-test",
        "ANTHROPIC_BASE_URL": "https://custom.endpoint.io",
    }
    c = ProviderConfig.from_env("anthropic", env=env)
    assert c.base_url == "https://custom.endpoint.io"


def test_from_env_timeout_override():
    env = {"ANTHROPIC_API_KEY": "key", "ANTHROPIC_TIMEOUT": "120"}
    c = ProviderConfig.from_env("anthropic", env=env)
    assert c.timeout_seconds == 120.0


def test_from_env_invalid_timeout_uses_default():
    env = {"ANTHROPIC_API_KEY": "key", "ANTHROPIC_TIMEOUT": "not-a-number"}
    c = ProviderConfig.from_env("anthropic", env=env)
    assert c.timeout_seconds > 0


def test_from_env_unknown_provider():
    env = {}
    c = ProviderConfig.from_env("custom-provider", env=env)
    assert c.provider_name == "custom-provider"


# ---------------------------------------------------------------------------
# ProviderConfig factory class methods
# ---------------------------------------------------------------------------


def test_for_anthropic_defaults():
    c = ProviderConfig.for_anthropic(api_key="key")
    assert c.provider_name == "anthropic"
    assert c.api_key == "key"
    assert "anthropic.com" in c.base_url


def test_for_openai_defaults():
    c = ProviderConfig.for_openai(api_key="key")
    assert c.provider_name == "openai"
    assert "openai.com" in c.base_url


def test_for_gemini_defaults():
    c = ProviderConfig.for_gemini(api_key="key")
    assert c.provider_name == "gemini"
    assert "googleapis" in c.base_url


def test_for_anthropic_no_key():
    c = ProviderConfig.for_anthropic()
    assert c.api_key == ""


# ---------------------------------------------------------------------------
# ConfigRegistry
# ---------------------------------------------------------------------------


def test_registry_empty():
    r = ConfigRegistry()
    assert len(r) == 0
    assert r.provider_names == []


def test_registry_register():
    r = ConfigRegistry()
    r.register(ProviderConfig.for_anthropic(api_key="key"))
    assert len(r) == 1
    assert "anthropic" in r.provider_names


def test_registry_duplicate_raises():
    r = ConfigRegistry()
    r.register(ProviderConfig.for_anthropic(api_key="key"))
    with pytest.raises(ValueError, match="already registered"):
        r.register(ProviderConfig.for_anthropic(api_key="key2"))


def test_registry_get_existing():
    c = ProviderConfig.for_anthropic(api_key="key")
    r = ConfigRegistry([c])
    got = r.get("anthropic")
    assert got is c


def test_registry_get_missing():
    r = ConfigRegistry()
    assert r.get("openai") is None


def test_registry_get_case_insensitive():
    c = ProviderConfig.for_anthropic(api_key="key")
    r = ConfigRegistry([c])
    assert r.get("ANTHROPIC") is not None


def test_registry_remove():
    r = ConfigRegistry([ProviderConfig.for_anthropic(api_key="key")])
    r.remove("anthropic")
    assert len(r) == 0


def test_registry_remove_missing_raises():
    r = ConfigRegistry()
    with pytest.raises(KeyError):
        r.remove("nonexistent")


def test_registry_validate_all():
    r = ConfigRegistry(
        [
            ProviderConfig.for_anthropic(api_key="key"),
            ProviderConfig.for_openai(api_key="key"),
        ]
    )
    results = r.validate_all()
    assert "anthropic" in results
    assert "openai" in results
    assert results["anthropic"] == []
    assert results["openai"] == []


def test_registry_repr():
    r = ConfigRegistry()
    assert "ConfigRegistry" in repr(r)


def test_registry_from_list():
    configs = [
        ProviderConfig.for_anthropic(api_key="k1"),
        ProviderConfig.for_openai(api_key="k2"),
    ]
    r = ConfigRegistry(configs)
    assert len(r) == 2
