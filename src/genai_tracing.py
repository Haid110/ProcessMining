# genai_tracing.py
#
# PURPOSE:
#   Provides traced_chat_completion(), a thin wrapper around MistralClient.chat()
#   that emits a structured OpenTelemetry span for every LLM call.
#
#   Each span contains:
#     - llm.request.model      : model name sent in the request
#     - llm.response.model     : model name echoed in the response
#     - llm.usage.prompt_tokens
#     - llm.usage.completion_tokens
#     - llm.usage.total_tokens
#
#   On failure the span is marked ERROR and the exception is recorded, then
#   re-raised so the caller can handle it.
#
# IMPORTANT — NO SIDE EFFECTS ON IMPORT:
#   This module deliberately contains no module-level code that opens files,
#   creates providers, or makes API calls. All of that lives in configure_tracing.py
#   (provider) and main.py (entry point). Importing this module is always safe.
#
# DEPENDENCIES (pin these in requirements.txt):
#   mistralai==0.1.8
#   opentelemetry-sdk>=1.20.0,<2.0.0
#
# COMPATIBLE MODEL NAMES (mistral-tiny was retired):
#   mistral-small-latest, mistral-medium-latest, mistral-large-latest,
#   open-mistral-7b, open-mixtral-8x7b

import os
from opentelemetry import trace
from mistralai.client import MistralClient

# ---------------------------------------------------------------------------
# Client initialisation
# ---------------------------------------------------------------------------
# MistralClient is created once at module load time. It reads the API key from
# the environment so the key is never hard-coded.
# Raises KeyError immediately if MISTRAL_API_KEY is not set, which is the
# clearest possible signal that the environment is misconfigured.
client = MistralClient(api_key=os.environ["MISTRAL_API_KEY"])

# Tracer is obtained from the *global* provider that configure_tracing.setup_tracing()
# registered. This must be called after setup_tracing() has run; if it is called
# before, OTel returns a no-op tracer and spans are silently discarded.
tracer = trace.get_tracer(__name__)


def traced_chat_completion(model: str, messages: list) -> object:
    """
    Call Mistral chat completion and record a span for the interaction.

    The span is named "mistral_chat_completion" — this exact string is used
    as the activity filter in convert_to_xes.py, so do not rename it without
    updating that script too.

    Args:
        model:    Mistral model identifier, e.g. "mistral-small-latest".
        messages: List of dicts with "role" and "content" keys.

    Returns:
        The raw Mistral ChatCompletionResponse object.

    Raises:
        MistralAPIException: Propagated from the Mistral client on API errors.
        Any other exception from the HTTP layer is also propagated.
    """
    # Start a span. The span stays open until the with-block exits, at which
    # point SimpleSpanProcessor exports it synchronously to traces.json.
    with tracer.start_as_current_span("mistral_chat_completion") as span:

        # --- Set request attributes BEFORE the API call ---------------------
        # Setting attributes early ensures they appear in the span even if the
        # API call raises an exception (attributes set after a raise are lost).
        span.set_attribute("llm.request.model", model)

        try:
            # --- Make the actual API call -----------------------------------
            response = client.chat(model=model, messages=messages)

            # --- Record response attributes ----------------------------------
            span.set_attribute("llm.response.model", response.model)

            if hasattr(response, "usage") and response.usage is not None:
                span.set_attribute(
                    "llm.usage.prompt_tokens",
                    response.usage.prompt_tokens
                )
                span.set_attribute(
                    "llm.usage.completion_tokens",
                    response.usage.completion_tokens
                )
                span.set_attribute(
                    "llm.usage.total_tokens",
                    response.usage.total_tokens
                )

            # Mark the span as successful (OTel default is UNSET, not OK).
            span.set_status(trace.StatusCode.OK)
            return response

        except Exception as exc:
            # --- Record the failure -----------------------------------------
            # record_exception() attaches exception.type, exception.message,
            # and exception.stacktrace as span events (visible in the JSON).
            span.record_exception(exc)
            span.set_status(trace.StatusCode.ERROR, str(exc))

            # Re-raise so main.py can decide whether to retry or abort.
            raise
