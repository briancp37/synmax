"""Tests for LLM client functionality."""

from unittest.mock import Mock, patch

import pytest

from data_agent.core.llm_client import LLMClient, get_default_llm_client


class TestLLMClient:
    """Test cases for LLMClient class."""

    def test_openai_model_initialization(self):
        """Test OpenAI model initialization with API key."""
        with patch("data_agent.core.llm_client.config") as mock_config:
            mock_config.OPENAI_API_KEY = "test-key"
            with patch("data_agent.core.llm_client.openai.OpenAI") as mock_openai:
                client = LLMClient(model="gpt-5")
                assert client.model == "gpt-5"
                assert client.provider == "openai"
                mock_openai.assert_called_once_with(api_key="test-key")

    def test_anthropic_model_initialization(self):
        """Test Anthropic model initialization with API key."""
        with patch("data_agent.core.llm_client.config") as mock_config:
            mock_config.ANTHROPIC_API_KEY = "test-key"
            with patch("data_agent.core.llm_client.Anthropic") as mock_anthropic:
                client = LLMClient(model="claude-sonnet")
                assert client.model == "claude-sonnet"
                assert client.provider == "anthropic"
                mock_anthropic.assert_called_once_with(api_key="test-key")

    def test_openai_missing_api_key(self):
        """Test error when OpenAI API key is missing."""
        with patch("data_agent.core.llm_client.config") as mock_config:
            mock_config.OPENAI_API_KEY = None
            with pytest.raises(ValueError, match="OpenAI API key is required"):
                LLMClient(model="gpt-5")

    def test_anthropic_missing_api_key(self):
        """Test error when Anthropic API key is missing."""
        with patch("data_agent.core.llm_client.config") as mock_config:
            mock_config.ANTHROPIC_API_KEY = None
            with pytest.raises(ValueError, match="Anthropic API key is required"):
                LLMClient(model="claude-sonnet")

    def test_unsupported_model(self):
        """Test error for unsupported model."""
        with pytest.raises(ValueError, match="Unsupported model"):
            LLMClient(model="invalid-model")  # type: ignore[arg-type]

    def test_openai_call_success(self):
        """Test successful OpenAI API call."""
        with patch("data_agent.core.llm_client.config") as mock_config:
            mock_config.OPENAI_API_KEY = "test-key"
            with patch("data_agent.core.llm_client.openai.OpenAI") as mock_openai:
                # Mock the response structure
                mock_response = Mock()
                mock_response.choices = [Mock()]
                mock_response.choices[0].message.content = "Test response"
                mock_response.choices[0].message.role = "assistant"
                mock_response.choices[0].message.tool_calls = None
                mock_response.usage.prompt_tokens = 10
                mock_response.usage.completion_tokens = 20
                mock_response.usage.total_tokens = 30
                mock_response.model = "gpt-5"

                mock_client = Mock()
                mock_client.chat.completions.create.return_value = mock_response
                mock_openai.return_value = mock_client

                client = LLMClient(model="gpt-5")
                messages = [{"role": "user", "content": "Test message"}]

                result = client.call(messages)

                assert result["content"] == "Test response"
                assert result["role"] == "assistant"
                assert result["provider"] == "openai"
                assert result["usage"]["prompt_tokens"] == 10
                assert result["usage"]["completion_tokens"] == 20
                assert result["usage"]["total_tokens"] == 30

    def test_openai_call_with_tools(self):
        """Test OpenAI API call with tools."""
        with patch("data_agent.core.llm_client.config") as mock_config:
            mock_config.OPENAI_API_KEY = "test-key"
            with patch("data_agent.core.llm_client.openai.OpenAI") as mock_openai:
                mock_response = Mock()
                mock_response.choices = [Mock()]
                mock_response.choices[0].message.content = "Test response"
                mock_response.choices[0].message.role = "assistant"
                mock_response.choices[0].message.tool_calls = [Mock()]
                mock_response.usage.prompt_tokens = 10
                mock_response.usage.completion_tokens = 20
                mock_response.usage.total_tokens = 30
                mock_response.model = "gpt-5"

                mock_client = Mock()
                mock_client.chat.completions.create.return_value = mock_response
                mock_openai.return_value = mock_client

                client = LLMClient(model="gpt-5")
                messages = [{"role": "user", "content": "Test message"}]
                tools = [{"type": "function", "function": {"name": "test_tool"}}]

                client.call(messages, tools=tools, tool_choice="auto")

                # Verify tools were passed correctly
                call_args = mock_client.chat.completions.create.call_args
                assert "tools" in call_args.kwargs
                assert "tool_choice" in call_args.kwargs
                assert call_args.kwargs["tools"] == tools
                assert call_args.kwargs["tool_choice"] == "auto"

    def test_anthropic_call_success(self):
        """Test successful Anthropic API call."""
        with patch("data_agent.core.llm_client.config") as mock_config:
            mock_config.ANTHROPIC_API_KEY = "test-key"
            with patch("data_agent.core.llm_client.Anthropic") as mock_anthropic:
                # Mock the response structure
                mock_response = Mock()
                mock_response.content = [Mock()]
                mock_response.content[0].text = "Test response"
                mock_response.usage.input_tokens = 10
                mock_response.usage.output_tokens = 20

                mock_client = Mock()
                mock_client.messages.create.return_value = mock_response
                mock_anthropic.return_value = mock_client

                client = LLMClient(model="claude-sonnet")
                messages = [{"role": "user", "content": "Test message"}]

                result = client.call(messages)

                assert result["content"] == "Test response"
                assert result["role"] == "assistant"
                assert result["provider"] == "anthropic"
                assert result["usage"]["prompt_tokens"] == 10
                assert result["usage"]["completion_tokens"] == 20
                assert result["usage"]["total_tokens"] == 30

    def test_anthropic_model_name_mapping(self):
        """Test Anthropic model name mapping."""
        with patch("data_agent.core.llm_client.config") as mock_config:
            mock_config.ANTHROPIC_API_KEY = "test-key"
            with patch("data_agent.core.llm_client.Anthropic") as mock_anthropic:
                mock_response = Mock()
                mock_response.content = [Mock()]
                mock_response.content[0].text = "Test response"
                mock_response.usage.input_tokens = 10
                mock_response.usage.output_tokens = 20

                mock_client = Mock()
                mock_client.messages.create.return_value = mock_response
                mock_anthropic.return_value = mock_client

                client = LLMClient(model="claude-sonnet")
                messages = [{"role": "user", "content": "Test message"}]

                client.call(messages)

                # Verify the correct Anthropic model name was used
                call_args = mock_client.messages.create.call_args
                assert call_args.kwargs["model"] == "claude-3-5-sonnet-20241022"

    def test_openai_api_error(self):
        """Test handling of OpenAI API errors."""
        with patch("data_agent.core.llm_client.config") as mock_config:
            mock_config.OPENAI_API_KEY = "test-key"
            with patch("data_agent.core.llm_client.openai.OpenAI") as mock_openai:
                mock_client = Mock()
                mock_client.chat.completions.create.side_effect = Exception("API Error")
                mock_openai.return_value = mock_client

                client = LLMClient(model="gpt-5")
                messages = [{"role": "user", "content": "Test message"}]

                with pytest.raises(Exception, match="OpenAI API call failed"):
                    client.call(messages)

    def test_anthropic_api_error(self):
        """Test handling of Anthropic API errors."""
        with patch("data_agent.core.llm_client.config") as mock_config:
            mock_config.ANTHROPIC_API_KEY = "test-key"
            with patch("data_agent.core.llm_client.Anthropic") as mock_anthropic:
                mock_client = Mock()
                mock_client.messages.create.side_effect = Exception("API Error")
                mock_anthropic.return_value = mock_client

                client = LLMClient(model="claude-sonnet")
                messages = [{"role": "user", "content": "Test message"}]

                with pytest.raises(Exception, match="Anthropic API call failed"):
                    client.call(messages)


class TestGetDefaultLLMClient:
    """Test cases for get_default_llm_client function."""

    def test_get_default_client_gpt5(self):
        """Test getting default client with GPT-5."""
        with patch("data_agent.core.llm_client.config") as mock_config:
            mock_config.DATA_AGENT_MODEL = "gpt-5"
            mock_config.OPENAI_API_KEY = "test-key"
            with patch("data_agent.core.llm_client.openai.OpenAI"):
                client = get_default_llm_client()
                assert client.model == "gpt-5"
                assert client.provider == "openai"

    def test_get_default_client_claude_sonnet(self):
        """Test getting default client with Claude Sonnet."""
        with patch("data_agent.core.llm_client.config") as mock_config:
            mock_config.DATA_AGENT_MODEL = "claude-sonnet"
            mock_config.ANTHROPIC_API_KEY = "test-key"
            with patch("data_agent.core.llm_client.Anthropic"):
                client = get_default_llm_client()
                assert client.model == "claude-sonnet"
                assert client.provider == "anthropic"

    def test_get_default_client_invalid_model(self):
        """Test error when default model is invalid."""
        with patch("data_agent.core.llm_client.config") as mock_config:
            mock_config.DATA_AGENT_MODEL = "invalid-model"
            with pytest.raises(ValueError, match="Invalid model 'invalid-model'"):
                get_default_llm_client()


class TestModelValidation:
    """Test model validation and type checking."""

    def test_all_supported_models(self):
        """Test that all supported models can be initialized."""
        models_and_configs = [
            ("gpt-4.1", "OPENAI_API_KEY", "data_agent.core.llm_client.openai.OpenAI"),
            ("gpt-5", "OPENAI_API_KEY", "data_agent.core.llm_client.openai.OpenAI"),
            ("claude-sonnet", "ANTHROPIC_API_KEY", "data_agent.core.llm_client.Anthropic"),
            ("claude-opus", "ANTHROPIC_API_KEY", "data_agent.core.llm_client.Anthropic"),
        ]

        for model, key_attr, mock_path in models_and_configs:
            with patch("data_agent.core.llm_client.config") as mock_config:
                setattr(mock_config, key_attr, "test-key")
                with patch(mock_path):
                    client = LLMClient(model=model)  # type: ignore[arg-type]
                    assert client.model == model

    def test_provider_detection(self):
        """Test that provider is correctly detected from model name."""
        # GPT models
        with patch("data_agent.core.llm_client.config") as mock_config:
            mock_config.OPENAI_API_KEY = "test-key"
            with patch("data_agent.core.llm_client.openai.OpenAI"):
                client = LLMClient(model="gpt-4.1")
                assert client.provider == "openai"

                client = LLMClient(model="gpt-5")
                assert client.provider == "openai"

        # Claude models
        with patch("data_agent.core.llm_client.config") as mock_config:
            mock_config.ANTHROPIC_API_KEY = "test-key"
            with patch("data_agent.core.llm_client.Anthropic"):
                client = LLMClient(model="claude-sonnet")
                assert client.provider == "anthropic"

                client = LLMClient(model="claude-opus")
                assert client.provider == "anthropic"
