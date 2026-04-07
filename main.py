# main.py - Complete example of GenAI tracing with OpenTelemetry
import os
from configure_tracing import setup_tracing
from genai_tracing import traced_chat_completion

# Set your OpenTelemetry endpoint before running
# export OTEL_EXPORTER_OTLP_ENDPOINT="https://otlp.oneuptime.com"
# export OTEL_EXPORTER_OTLP_HEADERS="x-oneuptime-token=YOUR_TOKEN"

def main():
    # Initialize the tracer
    tracer = setup_tracing("my-genai-service")

    # Define the conversation
    messages = [
        {"role": "system", "content": "You are a helpful coding assistant."},
        {"role": "user", "content": "Write a Python function to convert foot to cm."},
    ]

    # Make the traced LLM call
    result = traced_chat_completion(messages, model="mistral-large-latest", temperature=0.3)
    print(result)

    # Flush remaining spans before exit
    from opentelemetry import trace
    trace.get_tracer_provider().shutdown()

if __name__ == "__main__":
    main()