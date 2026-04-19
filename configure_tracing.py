# configure_tracing.py - Sets up the OpenTelemetry tracer provider with OTLP export
import logging
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor,BatchSpanProcessor, ConsoleSpanExporter
# from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
# from opentelemetry.sdk.resources import Resource

try:
    from opentelemetry.instrumentation.mistralai import MistralAiInstrumentor
except (ImportError, ModuleNotFoundError, AttributeError):
    MistralAiInstrumentor = None


def setup_tracing(service_name: str = "mistral-service") -> trace.Tracer:

    # 1. Open a file for writing
    log_file = open("traces.json", "a")

    # Setup the Provider
    provider = TracerProvider()
    
    # 2. Configure the exporter to write to that file
    # Note: This outputs one JSON object per line (JSONL format)
    # Setup the Exporter (Sending to Console for now)
    processor = BatchSpanProcessor(ConsoleSpanExporter(out=None)) # Or out=log_file
    provider.add_span_processor(processor)
    
    # 3. Set the global default tracer provider
    trace.set_tracer_provider(provider)

    # 4. Instrument Mistral (if available)
    if MistralAiInstrumentor is not None:
        try:
            MistralAiInstrumentor().instrument()
        except Exception as exc:
            logging.warning("Mistral instrumentation failed: %s", exc)
    else:
        logging.warning("Mistral instrumentation package not installed; skipping instrumentation")

    provider.shutdown() 

    # Close the file at the end
    log_file.close()


    return trace.get_tracer(service_name)

# Call this before you create your Mistral client
if __name__ == "__main__":
    setup_tracing()



# def setup_tracing(service_name: str) -> trace.Tracer:
#     # Define the service resource so spans are grouped correctly
#     resource = Resource.create({
#         "service.name": service_name,
#         "service.version": "1.0.0",
#     })

#     # Create the tracer provider with our resource
#     provider = TracerProvider(resource=resource)

#     # Configure the OTLP exporter - endpoint comes from environment variables
#     # Set OTEL_EXPORTER_OTLP_ENDPOINT and OTEL_EXPORTER_OTLP_HEADERS in your env
#     exporter = OTLPSpanExporter()

#     # Use batch processing to avoid blocking on every span
#     provider.add_span_processor(BatchSpanProcessor(exporter))

#     # Register this provider globally
#     trace.set_tracer_provider(provider)

#     return trace.get_tracer(service_name)
