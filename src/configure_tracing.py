# configure_tracing.py
#
# PURPOSE:
#   Sets up a single, global OpenTelemetry TracerProvider that writes spans to
#   "traces.json" (JSONL format, one JSON object per line).
#
#   This module is the ONLY place where the provider and file handle are created.
#   main.py calls setup_tracing() once at startup, and shutdown_tracing() once
#   at the very end. No other module should touch the provider lifecycle.
#
# DEPENDENCIES:
#   opentelemetry-sdk>=1.20.0,<2.0.0
#   opentelemetry-instrumentation-mistralai==0.57.0   (optional, auto-instruments)
#   mistralai==0.1.8
#
# USAGE:
#   from configure_tracing import setup_tracing, shutdown_tracing
#   tracer = setup_tracing()
#   ...
#   shutdown_tracing()

import logging
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor, ConsoleSpanExporter

# MistralAiInstrumentor auto-patches MistralClient.chat() to emit spans.
# It is optional: if the package is missing or broken, we fall back to the
# manual spans defined in genai_tracing.py.
try:
    from opentelemetry.instrumentation.mistralai import MistralAiInstrumentor
    _HAS_MISTRAL_INSTRUMENTOR = True
except (ImportError, ModuleNotFoundError, AttributeError):
    MistralAiInstrumentor = None
    _HAS_MISTRAL_INSTRUMENTOR = False

# Module-level references so shutdown_tracing() can reach them.
_provider: TracerProvider | None = None
_log_file = None


def setup_tracing(service_name: str = "mistral-service") -> trace.Tracer:
    """
    Configure the global OTel TracerProvider and return a tracer.

    Spans are written to traces.json in JSONL format (append mode so that
    multiple runs accumulate cases for process mining).

    Args:
        service_name: Label that appears in each span's resource block.

    Returns:
        A Tracer bound to service_name.
    """
    global _provider, _log_file

    # --- 1. Open the output file in append + line-buffered mode ---------------
    # "a"  → multiple runs accumulate cases (good for process mining datasets).
    # buffering=1 → each line is flushed to disk immediately, so spans are not
    #               lost if the process crashes.
    _log_file = open("traces.json", "a", encoding="utf-8", buffering=1)

    # --- 2. Build the exporter ------------------------------------------------
    # ConsoleSpanExporter writes one JSON object per span to the file we just
    # opened. "Console" is a misnomer; it just writes to any file-like object.
    exporter = ConsoleSpanExporter(out=_log_file)

    # --- 3. Build the provider ------------------------------------------------
    # SimpleSpanProcessor exports each span synchronously as soon as it ends.
    # This is safer than BatchSpanProcessor for short-lived scripts because
    # there is no async buffer that could be lost on an early shutdown.
    _provider = TracerProvider()
    _provider.add_span_processor(SimpleSpanProcessor(exporter))

    # --- 4. Register as the global provider -----------------------------------
    # After this call, trace.get_tracer(...) anywhere in the process will use
    # this provider. genai_tracing.py must NOT create its own provider.
    trace.set_tracer_provider(_provider)

    # --- 5. Optionally enable auto-instrumentation ----------------------------
    # When active, every call to MistralClient.chat() automatically gets a span.
    # If you prefer full manual control (recommended for research), skip this
    # block and rely on the traced_chat_completion() wrapper in genai_tracing.py.
    if _HAS_MISTRAL_INSTRUMENTOR:
        try:
            MistralAiInstrumentor().instrument()
            logging.info("MistralAiInstrumentor active.")
        except Exception as exc:
            # Instrumentation failure should not crash the application.
            logging.warning("Mistral auto-instrumentation failed: %s", exc)
    else:
        logging.warning(
            "opentelemetry-instrumentation-mistralai not found; "
            "using manual spans only."
        )

    # NOTE: provider.shutdown() is intentionally NOT called here.
    #       Call shutdown_tracing() from main.py after all LLM work is done.
    return trace.get_tracer(service_name)


def shutdown_tracing() -> None:
    """
    Flush and close the TracerProvider and the output file.

    Must be called exactly once, at program exit, after all traced calls
    have completed. Calling it earlier will silently drop any spans that
    are still in flight.
    """
    global _provider, _log_file

    if _provider is not None:
        # flush() forces any buffered spans to the exporter before we close.
        _provider.shutdown()
        _provider = None

    if _log_file is not None and not _log_file.closed:
        _log_file.flush()
        _log_file.close()
        _log_file = None
