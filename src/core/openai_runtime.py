"""
OpenAI SDK runtime — dùng cho:

  Blue Team → OpenRouter liquid/lfm-2.5-2.6b (create_blue_pair)
  Red Team  → OpenAI gpt-4o-mini (create_openai_pair) khi RED_TEAM_PROVIDER=openai

Gemini Red Team dùng Google ADK trong agents/*.py — không đi qua file này.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

from core.config import (
    get_red_model,
    get_red_provider,
    get_blue_model,
    get_blue_provider,
    blue_client_kwargs,
    red_openai_client_kwargs,
)


@dataclass
class OpenAIAgent:
    name: str
    instruction: str
    provider: str = "openai"


@dataclass
class _MockInvocationContext:
    user_id: str = "student"


@dataclass
class OpenAIRunner:
    """Optional ADK-style plugins + Chat Completions."""

    app_name: str
    model: str
    plugins: list = field(default_factory=list)
    provider: str = "openai"
    temperature: float = 0.4
    client_kwargs: dict = field(default_factory=dict)
    input_hooks: list[Callable[[str], str | None]] = field(default_factory=list)
    output_hooks: list[Callable[[str], str]] = field(default_factory=list)

    fallback_models: list[str] = field(default_factory=list)
    last_model_used: str | None = None

    def _client(self):
        from openai import OpenAI

        return OpenAI(**(self.client_kwargs or {}))

    def _complete(self, client, messages):
        """Call ``self.model``; on "no endpoints" 404 try ``fallback_models``
        (same model on another OpenRouter tier). Rate-limit 429s are retried
        with backoff. ``last_model_used`` records which slug answered."""
        import time
        from openai import NotFoundError, RateLimitError

        candidates = [self.model, *self.fallback_models]
        for i, model in enumerate(candidates):
            for attempt in range(4):
                try:
                    completion = client.chat.completions.create(
                        model=model, messages=messages, temperature=self.temperature
                    )
                    self.last_model_used = model
                    return completion
                except NotFoundError:
                    if i == len(candidates) - 1:
                        raise
                    break  # try next candidate
                except RateLimitError:
                    if attempt == 3:
                        raise
                    time.sleep(5 * (attempt + 1))

    async def chat(self, agent: OpenAIAgent, user_message: str) -> str:
        for hook in self.input_hooks:
            blocked = hook(user_message)
            if blocked:
                return blocked

        block_msg = await self._run_input_plugins(user_message)
        if block_msg is not None:
            return block_msg

        client = self._client()
        messages = [
            {"role": "system", "content": agent.instruction},
            {"role": "user", "content": user_message},
        ]
        completion = self._complete(client, messages)
        text = (completion.choices[0].message.content or "").strip()

        for hook in self.output_hooks:
            text = hook(text)

        text = await self._run_output_plugins(text)
        return text

    async def _run_input_plugins(self, user_message: str) -> str | None:
        if not self.plugins:
            return None
        try:
            from google.genai import types
        except ImportError:
            return None

        user_content = types.Content(
            role="user",
            parts=[types.Part.from_text(text=user_message)],
        )
        ctx = _MockInvocationContext()
        for plugin in self.plugins:
            cb = getattr(plugin, "on_user_message_callback", None)
            if cb is None:
                continue
            try:
                result = await cb(
                    invocation_context=ctx, user_message=user_content
                )
            except TypeError:
                result = cb(invocation_context=ctx, user_message=user_content)
            if result is None:
                continue
            return _content_to_text(result)
        return None

    async def _run_output_plugins(self, text: str) -> str:
        if not self.plugins or not text:
            return text
        try:
            from google.genai import types
        except ImportError:
            return text

        content = types.Content(
            role="model", parts=[types.Part.from_text(text=text)]
        )

        class _Resp:
            pass

        llm_response = _Resp()
        llm_response.content = content

        class _Ctx:
            pass

        for plugin in self.plugins:
            cb = getattr(plugin, "after_model_callback", None)
            if cb is None:
                continue
            try:
                out = await cb(callback_context=_Ctx(), llm_response=llm_response)
            except TypeError:
                out = cb(callback_context=_Ctx(), llm_response=llm_response)
            if out is not None and getattr(out, "content", None) is not None:
                llm_response = out
        return _content_to_text(llm_response.content) or text


def _content_to_text(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    parts = getattr(content, "parts", None) or []
    chunks = []
    for part in parts:
        t = getattr(part, "text", None)
        if t:
            chunks.append(t)
    return "".join(chunks)


def _make_pair(
    *,
    name: str,
    instruction: str,
    app_name: str,
    model: str,
    provider: str,
    client_kwargs: dict,
    plugins: list | None = None,
    input_hooks: list | None = None,
    output_hooks: list | None = None,
    temperature: float = 0.4,
    fallback_models: list[str] | None = None,
) -> tuple[OpenAIAgent, OpenAIRunner]:
    agent = OpenAIAgent(name=name, instruction=instruction, provider=provider)
    runner = OpenAIRunner(
        app_name=app_name,
        model=model,
        provider=provider,
        client_kwargs=client_kwargs,
        plugins=list(plugins or []),
        input_hooks=list(input_hooks or []),
        output_hooks=list(output_hooks or []),
        temperature=temperature,
        fallback_models=list(fallback_models or []),
    )
    return agent, runner


def create_blue_pair(
    *,
    name: str,
    instruction: str,
    app_name: str,
    plugins: list | None = None,
    input_hooks: list | None = None,
    output_hooks: list | None = None,
    temperature: float = 0.4,
) -> tuple[OpenAIAgent, OpenAIRunner]:
    """Blue Team — always OpenRouter liquid/lfm-2.5-2.6b."""
    return _make_pair(
        name=name,
        instruction=instruction,
        app_name=app_name,
        model=get_blue_model(),
        provider=get_blue_provider(),
        client_kwargs=blue_client_kwargs(),
        plugins=plugins,
        input_hooks=input_hooks,
        output_hooks=output_hooks,
        temperature=temperature,
        # Same locked model; OpenRouter currently serves it only on ":free".
        fallback_models=[f"{get_blue_model()}:free"],
    )


def create_openai_pair(
    *,
    name: str,
    instruction: str,
    app_name: str,
    plugins: list | None = None,
    input_hooks: list | None = None,
    output_hooks: list | None = None,
    temperature: float = 0.4,
    model: str | None = None,
) -> tuple[OpenAIAgent, OpenAIRunner]:
    """Red Team OpenAI path (default = soft model; advance may pass harder)."""
    return _make_pair(
        name=name,
        instruction=instruction,
        app_name=app_name,
        model=model or get_red_model(),
        provider=get_red_provider(),
        client_kwargs=red_openai_client_kwargs(),
        plugins=plugins,
        input_hooks=input_hooks,
        output_hooks=output_hooks,
        temperature=temperature,
    )
