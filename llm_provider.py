"""Provider-agnostic chat-LLM factory.

One place decides which backend builds the chat model, so the call sites
(nlu, facts, query, rag, app) don't care. Flip `config.LLM_PROVIDER` between
"openrouter" and "ollama" to switch the whole app — no other code changes.

JSON mode: nlu.py and facts.py need the model to emit a strict JSON object.
The mechanism differs per provider and is translated here:
    - Ollama           -> format="json"
    - OpenAI-compatible -> response_format={"type": "json_object"}
Not every free OpenRouter model honors response_format; if one ignores or
rejects it, the callers already degrade gracefully (nlu falls back to regex,
facts returns []), so we never hard-fail on it.
"""

import config


def make_chat(temperature: float = 0, json_mode: bool = False, timeout=None):
    """Build a chat model for the configured provider.

    Args:
        temperature: sampling temperature.
        json_mode:   request a strict JSON-object response (extractors).
        timeout:     per-call timeout seconds (defaults to config).
    """
    timeout = timeout or config.LLM_REQUEST_TIMEOUT_SEC
    provider = (getattr(config, "LLM_PROVIDER", "ollama") or "ollama").lower()

    if provider == "openrouter":
        # OpenRouter speaks the OpenAI wire protocol, so ChatOpenAI works with
        # a custom base_url + key. Key MUST come from the environment. Fail
        # fast with a clear message rather than letting the request go out
        # keyless and come back as an opaque 401 "Missing Authentication".
        if not config.OPENROUTER_API_KEY:
            raise RuntimeError(
                "OPENROUTER_API_KEY is not set in this shell. Run "
                "`export OPENROUTER_API_KEY=\"sk-or-v1-...\"` in THIS terminal "
                "(or add it to ~/.zshrc and restart the shell) before running."
            )
        from langchain_openai import ChatOpenAI
        model_kwargs = {}
        if json_mode:
            model_kwargs["response_format"] = {"type": "json_object"}
        return ChatOpenAI(
            model=config.OPENROUTER_MODEL,
            api_key=config.OPENROUTER_API_KEY,
            base_url=config.OPENROUTER_BASE_URL,
            temperature=temperature,
            timeout=timeout,
            max_retries=2,
            model_kwargs=model_kwargs,
        )

    # Default / fallback: Ollama (local or cloud).
    from langchain_ollama import ChatOllama
    kwargs = dict(
        model=config.LLM_MODEL,
        base_url=config.OLLAMA_BASE_URL,
        temperature=temperature,
        timeout=timeout,
    )
    if json_mode:
        kwargs["format"] = "json"
    return ChatOllama(**kwargs)
