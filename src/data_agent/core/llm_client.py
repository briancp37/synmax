"""LLM client for interfacing with OpenAI and Anthropic models."""

from dataclasses import dataclass
from typing import Any, Literal

import openai
from anthropic import Anthropic

from data_agent import config

ModelName = Literal["gpt-4.1", "gpt-5", "claude-sonnet", "claude-opus"]


@dataclass
class LLMClient:
    """Client for making calls to LLM providers (OpenAI and Anthropic)."""

    model: ModelName

    def __post_init__(self) -> None:
        """Initialize the appropriate client based on model prefix."""
        if self.model.startswith("gpt-"):
            if not config.OPENAI_API_KEY:
                raise ValueError(
                    "OpenAI API key is required for GPT models. "
                    "Please set OPENAI_API_KEY environment variable or add it to your .env file."
                )
            self.openai_client = openai.OpenAI(api_key=config.OPENAI_API_KEY)
            self.provider = "openai"
        elif self.model.startswith("claude-"):
            if not config.ANTHROPIC_API_KEY:
                raise ValueError(
                    "Anthropic API key is required for Claude models. "
                    "Please set ANTHROPIC_API_KEY environment variable or add it to your .env file."
                )
            self.anthropic_client = Anthropic(api_key=config.ANTHROPIC_API_KEY)
            self.provider = "anthropic"
        else:
            raise ValueError(f"Unsupported model: {self.model}")

    def call(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Make a call to the LLM provider.

        Args:
            messages: List of message dictionaries with 'role' and 'content' keys
            tools: Optional list of tool definitions for function calling
            **kwargs: Additional arguments passed to the provider

        Returns:
            Response dictionary from the provider

        Raises:
            ValueError: If model is not supported or API key is missing
            Exception: If API call fails
        """
        if self.provider == "openai":
            return self._call_openai(messages, tools, **kwargs)
        elif self.provider == "anthropic":
            return self._call_anthropic(messages, tools, **kwargs)
        else:
            raise ValueError(f"Unsupported provider: {self.provider}")

    def _call_openai(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Make a call to OpenAI API."""
        try:
            # Convert our model names to OpenAI's model names
            openai_model = self._get_openai_model_name(self.model)
            call_args = {"model": openai_model, "messages": messages, **kwargs}

            if tools:
                call_args["tools"] = tools
                call_args["tool_choice"] = kwargs.get("tool_choice", "auto")

            # Some models don't support custom temperature, so try without it first
            response = None
            try:
                response = self.openai_client.chat.completions.create(**call_args)
            except Exception as e:
                if "temperature" in str(e) and "temperature" in call_args:
                    # Retry without temperature
                    call_args_no_temp = {k: v for k, v in call_args.items() if k != "temperature"}
                    response = self.openai_client.chat.completions.create(**call_args_no_temp)
                else:
                    raise

            return {
                "content": response.choices[0].message.content,
                "role": response.choices[0].message.role,
                "tool_calls": getattr(response.choices[0].message, "tool_calls", None),
                "usage": {
                    "prompt_tokens": response.usage.prompt_tokens if response.usage else 0,
                    "completion_tokens": response.usage.completion_tokens if response.usage else 0,
                    "total_tokens": response.usage.total_tokens if response.usage else 0,
                },
                "model": response.model,
                "provider": "openai",
            }
        except Exception as e:
            raise Exception(f"OpenAI API call failed: {e}") from e

    def _call_anthropic(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Make a call to Anthropic API."""
        try:
            # Convert model name to Anthropic's naming convention
            anthropic_model = self._get_anthropic_model_name(self.model)

            call_args = {
                "model": anthropic_model,
                "messages": messages,
                "max_tokens": kwargs.get("max_tokens", 4000),
                **{k: v for k, v in kwargs.items() if k != "max_tokens"},
            }

            if tools:
                call_args["tools"] = tools

            response = self.anthropic_client.messages.create(**call_args)

            return {
                "content": response.content[0].text if response.content else "",
                "role": "assistant",
                "tool_calls": getattr(response, "tool_calls", None),
                "usage": {
                    "prompt_tokens": response.usage.input_tokens if response.usage else 0,
                    "completion_tokens": response.usage.output_tokens if response.usage else 0,
                    "total_tokens": (
                        (response.usage.input_tokens + response.usage.output_tokens)
                        if response.usage
                        else 0
                    ),
                },
                "model": anthropic_model,
                "provider": "anthropic",
            }
        except Exception as e:
            raise Exception(f"Anthropic API call failed: {e}") from e

    def _get_openai_model_name(self, model: ModelName) -> str:
        """Convert our model names to OpenAI's model names."""
        model_mapping = {
            "gpt-4.1": "gpt-4",
            "gpt-5": "gpt-4",  # Use gpt-4 as fallback since gpt-5 doesn't exist yet
        }
        return model_mapping.get(model, model)

    def _get_anthropic_model_name(self, model: ModelName) -> str:
        """Convert our model names to Anthropic's model names."""
        model_mapping = {
            "claude-sonnet": "claude-3-5-sonnet-20241022",
            "claude-opus": "claude-3-opus-20240229",
        }
        return model_mapping.get(model, model)


def get_default_llm_client() -> LLMClient:
    """Get the default LLM client from configuration."""
    model = config.DATA_AGENT_MODEL
    if model not in ["gpt-4.1", "gpt-5", "claude-sonnet", "claude-opus"]:
        raise ValueError(
            f"Invalid model '{model}' in DATA_AGENT_MODEL. "
            "Supported models: gpt-4.1, gpt-5, claude-sonnet, claude-opus"
        )
    return LLMClient(model=model)  # type: ignore[arg-type]
