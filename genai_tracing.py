# # genai_tracing.py - Wraps OpenAI calls with full prompt/completion tracing
# import openai
# from opentelemetry import trace

# tracer = trace.get_tracer("genai-service")

# def traced_chat_completion(messages: list, model: str = "gpt-4", temperature: float = 0.7, max_tokens: int = 1024):
#     """Call OpenAI chat completion and record prompt/completion as span events."""

#     # Start a new span for this LLM call
#     with tracer.start_as_current_span("gen_ai.chat.completion") as span:
#         # Set standard GenAI attributes on the span
#         span.set_attribute("gen_ai.system", "openai")
#         span.set_attribute("gen_ai.request.model", model)
#         span.set_attribute("gen_ai.request.temperature", temperature)
#         span.set_attribute("gen_ai.request.max_tokens", max_tokens)

#         # Record the prompt as a span event
#         # Each message in the conversation gets its own event
#         for i, message in enumerate(messages):
#             span.add_event(
#                 "gen_ai.prompt",
#                 attributes={
#                     "gen_ai.prompt.role": message["role"],
#                     "gen_ai.prompt.content": message["content"],
#                     "gen_ai.prompt.index": i,
#                 },
#             )

#         # Make the actual API call
#         response = openai.chat.completions.create(
#             model=model,
#             messages=messages,
#             temperature=temperature,
#             max_tokens=max_tokens,
#         )

#         # Extract the completion
#         completion = response.choices[0].message.content

#         # Record the completion as a span event
#         span.add_event(
#             "gen_ai.completion",
#             attributes={
#                 "gen_ai.completion.role": "assistant",
#                 "gen_ai.completion.content": completion,
#                 "gen_ai.completion.finish_reason": response.choices[0].finish_reason,
#             },
#         )

#         # Record token usage for cost tracking
#         span.set_attribute("gen_ai.usage.input_tokens", response.usage.prompt_tokens)
#         span.set_attribute("gen_ai.usage.output_tokens", response.usage.completion_tokens)

#         return completion


# genai_tracing.py
from mistralai.client import MistralClient
#from mistralai.models.chat_completion import ChatMessage
from opentelemetry import trace
import os
#from opentelemetry.instrumentation.mistralai import MistralAiInstrumentor
#MistralAiInstrumentor().instrument() # This MUST come first

from mistralai.client import MistralClient
tracer = trace.get_tracer("genai-service")

# Initialize Mistral client
client = MistralClient(api_key=os.environ["MISTRAL_API_KEY"])

def traced_chat_completion(messages: list, model: str = "mistral-large-latest", temperature: float = 0.7, max_tokens: int = 1024):
    """Call Mistral chat completion and record prompt/completion as span events."""

    with tracer.start_as_current_span("gen_ai.chat.completion") as span:
        # Standard attributes (keep consistent for analytics)
        span.set_attribute("gen_ai.system", "mistral")
        span.set_attribute("gen_ai.request.model", model)
        span.set_attribute("gen_ai.request.temperature", temperature)
        span.set_attribute("gen_ai.request.max_tokens", max_tokens)

        # Convert messages → Mistral format
        mistral_messages = []
        for i, message in enumerate(messages):
            mistral_messages.append(
                #ChatMessage(role=message["role"], content=message["content"])
                {"role": message["role"], "content": message["content"]}
            )

            span.add_event(
                "gen_ai.prompt",
                attributes={
                    "gen_ai.prompt.role": message["role"],
                    "gen_ai.prompt.content": message["content"],
                    "gen_ai.prompt.index": i,
                },
            )

        # API Call
        response = client.chat(
            model=model,
            messages=mistral_messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )

        # Extract completion
        completion = response.choices[0].message.content

        # Completion event
        span.add_event(
            "gen_ai.completion",
            attributes={
                "gen_ai.completion.role": "assistant",
                "gen_ai.completion.content": completion,
                "gen_ai.completion.finish_reason": response.choices[0].finish_reason,
            },
        )

        # Token usage (Mistral format differs slightly)
        if hasattr(response, "usage") and response.usage:
            span.set_attribute("gen_ai.usage.input_tokens", response.usage.prompt_tokens)
            span.set_attribute("gen_ai.usage.output_tokens", response.usage.completion_tokens)

        return completion