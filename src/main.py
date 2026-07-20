# main.py
#
# PURPOSE:
#   Simulates realistic multi-step GenAI coding sessions and records every
#   step as a named OpenTelemetry span. Multiple sessions are run in a single
#   execution to build up a process mining event log with meaningful variance.
#
# ACTIVITY VOCABULARY (span names written to traces.json):
#   user_prompt          → user sends a coding request
#   code_generation      → model returns code for the first time
#   test_execution       → generated code is extracted and executed locally
#   test_passed          → execution produced no exception
#   test_failed          → execution raised an exception
#   correction_request   → user asks the model to fix the broken code
#   code_regeneration    → model returns revised code
#   session_end          → conversation is closed (happy or abandoned)
#
#   Each session = one OTel trace = one XES case.
#   Each activity = one child span inside that trace.
#
# SCENARIOS (one per run, chosen in SESSIONS list at the bottom):
#   happy_path           → prompt → generate → test passes → end
#   correction_loop      → prompt → generate → test fails → correct → retest → end
#   repeated_failure     → prompt → generate → test fails × 2 → abandoned
#   no_code_response     → model returns explanation text, no code block → end
#
# HOW TO ADD MORE SESSIONS:
#   Append entries to the SESSIONS list. Each entry is a dict with keys
#   "scenario", "prompt", and optionally "label" for console output.
#
# ENVIRONMENT VARIABLES:
#   MISTRAL_API_KEY   Your Mistral API key (required).
#
# AFTER RUNNING:
#   traces.json  → append-mode JSONL; each run adds new cases.
#   Run convert_to_xes.py to produce mistral_trace_log.xes.
#
# REQUIREMENTS (requirements.txt):
#   mistralai==0.1.8
#   opentelemetry-sdk>=1.20.0,<2.0.0
#   opentelemetry-instrumentation-mistralai==0.57.0
#   pm4py
#   pandas

import re
import sys
import time
from opentelemetry import trace

from configure_tracing import setup_tracing, shutdown_tracing
from genai_tracing import traced_chat_completion

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

MODEL = "mistral-small-latest"   # mistral-tiny is retired; use this instead

# Each entry defines one full session to simulate.
# "scenario" maps to a function below. "prompt" is sent to the model.
# Add, remove, or duplicate entries to control how many cases end up in
# the XES log. Aim for 30–50 total cases for meaningful process analysis.
SESSIONS = [
    # --- Happy path (code works first time) ---------------------------------
    {"scenario": "happy_path",       "prompt": "Write a Python function that converts feet to centimetres."},
    {"scenario": "happy_path",       "prompt": "Write a Python function that returns the factorial of n using recursion."},
    {"scenario": "happy_path",       "prompt": "Write a Python function that checks if a string is a palindrome."},
    {"scenario": "happy_path",       "prompt": "Write a Python function that returns the nth Fibonacci number."},
    {"scenario": "happy_path",       "prompt": "Write a Python function that flattens a nested list one level deep."},

    # --- Correction loop (code fails once then succeeds) --------------------
    {"scenario": "correction_loop",  "prompt": "Write a Python function that merges two sorted lists into one sorted list."},
    {"scenario": "correction_loop",  "prompt": "Write a Python function that counts word frequencies in a string."},
    {"scenario": "correction_loop",  "prompt": "Write a Python function that removes duplicate values from a list while preserving order."},
    {"scenario": "correction_loop",  "prompt": "Write a Python function that implements binary search on a sorted list."},
    {"scenario": "correction_loop",  "prompt": "Write a Python function that rotates a list to the right by k positions."},

    # --- Repeated failure (model never produces working code → abandoned) ---
    {"scenario": "repeated_failure", "prompt": "Write a Python function that parses a raw HTTP request string into headers and body."},
    {"scenario": "repeated_failure", "prompt": "Write a Python async function that fetches JSON from a URL using aiohttp."},

    # --- No code block (model explains rather than coding) ------------------
    {"scenario": "no_code_response", "prompt": "Should I use a list or a set for membership testing in Python?"},
    {"scenario": "no_code_response", "prompt": "What is the difference between deepcopy and shallow copy in Python?"},
]


# ---------------------------------------------------------------------------
# Code extraction and execution helpers
# ---------------------------------------------------------------------------

def extract_code(response_text: str) -> str | None:
    """
    Pull the first ```python ... ``` block out of the model's reply.

    Returns the code string, or None if no fenced block was found.
    This is used to detect "no_code_response" situations naturally and to
    feed the executor for test_execution spans.
    """
    match = re.search(r"```python\s*\n(.*?)```", response_text, re.DOTALL)
    if match:
        return match.group(1).strip()
    # Also accept plain ``` blocks as fallback
    match = re.search(r"```\s*\n(.*?)```", response_text, re.DOTALL)
    return match.group(1).strip() if match else None


def execute_code(code: str) -> dict:
    """
    Execute the generated code in an isolated namespace.

    Returns a dict with:
        success (bool)  : True if exec() raised no exception
        error   (str)   : exception message on failure, empty string on success
        error_type(str) : exception class name on failure, empty string on success
    """
    namespace: dict = {}
    try:
        exec(compile(code, "<generated>", "exec"), namespace)
        return {"success": True, "error": "", "error_type": ""}
    except Exception as exc:
        return {
            "success":    False,
            "error":      str(exc),
            "error_type": type(exc).__name__,
        }


# ---------------------------------------------------------------------------
# Span helper
# ---------------------------------------------------------------------------

def activity_span(tracer: trace.Tracer, name: str, attributes: dict = None):
    """
    Context manager that opens a *child* span named `name` under whatever
    span is currently active (the session root span).

    All key/value pairs in `attributes` are set before the span closes so
    they are guaranteed to appear in the JSON even if an exception fires.

    Usage:
        with activity_span(tracer, "test_execution", {"code.length": len(code)}):
            ...
    """
    span = tracer.start_span(name)
    token = trace.use_span(span, end_on_exit=True)

    class _Ctx:
        def __enter__(self_inner):
            token.__enter__()
            if attributes:
                for k, v in attributes.items():
                    span.set_attribute(k, v)
            span.set_status(trace.StatusCode.OK)
            return span

        def __exit__(self_inner, exc_type, exc_val, exc_tb):
            if exc_type is not None:
                span.record_exception(exc_val)
                span.set_status(trace.StatusCode.ERROR, str(exc_val))
            token.__exit__(exc_type, exc_val, exc_tb)
            return False   # never suppress exceptions

    return _Ctx()


# ---------------------------------------------------------------------------
# Scenario implementations
# ---------------------------------------------------------------------------
# Every scenario receives the tracer and the user prompt.
# The session root span is already active when these are called, so every
# child span created here is automatically nested under it in the trace tree.
# convert_to_xes.py flattens all spans into a single case using the shared
# trace_id, which is what gives you multi-event cases in the XES log.

def run_happy_path(tracer: trace.Tracer, prompt: str) -> None:
    """
    Ideal flow: prompt → code generated → test passes → session ends.

    Process model (one linear sequence, no loops):
        user_prompt → code_generation → test_execution → test_passed → session_end
    """
    messages = [
        {"role": "system", "content": "You are a helpful Python coding assistant. Always include a complete, runnable code block."},
        {"role": "user",   "content": prompt},
    ]

    # 1. Record the user prompt as an activity
    with activity_span(tracer, "user_prompt", {"prompt.text": prompt}):
        time.sleep(0.05)   # small pause so timestamps are distinct in the XES log

    # 2. Call the model and record the generation span
    response = traced_chat_completion(MODEL, messages)
    reply = response.choices[0].message.content
    code  = extract_code(reply)

    with activity_span(tracer, "code_generation", {
        "response.has_code":        code is not None,
        "response.total_tokens":    response.usage.total_tokens,
        "response.completion_tokens": response.usage.completion_tokens,
    }):
        time.sleep(0.05)

    if code is None:
        # Model gave an explanation instead of code — treat as no_code_response
        with activity_span(tracer, "session_end", {"end.reason": "no_code_in_happy_path"}):
            pass
        return

    # 3. Execute the generated code
    result = execute_code(code)
    with activity_span(tracer, "test_execution", {
        "test.code_length": len(code),
        "test.success":     result["success"],
        "test.error_type":  result["error_type"],
    }):
        time.sleep(0.05)

    # 4. Record outcome
    if result["success"]:
        with activity_span(tracer, "test_passed", {}):
            time.sleep(0.05)
    else:
        # Unexpected failure in what should be a happy path — still close cleanly
        with activity_span(tracer, "test_failed", {"error.message": result["error"]}):
            time.sleep(0.05)

    with activity_span(tracer, "session_end", {"end.reason": "success"}):
        pass


def run_correction_loop(tracer: trace.Tracer, prompt: str) -> None:
    """
    Code fails on first attempt, user requests a correction, model fixes it.

    Process model (one correction loop):
        user_prompt → code_generation → test_execution → test_failed
            → correction_request → code_regeneration → test_execution
            → test_passed → session_end

    This is the scenario your supervisor mentioned:
    "corrections/regeneration should follow failed tests."
    """
    messages = [
        {"role": "system", "content": "You are a helpful Python coding assistant. Always include a complete, runnable code block."},
        {"role": "user",   "content": prompt},
    ]

    # 1. User prompt
    with activity_span(tracer, "user_prompt", {"prompt.text": prompt}):
        time.sleep(0.05)

    # 2. First generation
    response = traced_chat_completion(MODEL, messages)
    reply    = response.choices[0].message.content
    code     = extract_code(reply)

    with activity_span(tracer, "code_generation", {
        "response.has_code":      code is not None,
        "response.total_tokens":  response.usage.total_tokens,
    }):
        time.sleep(0.05)

    # 3. First test — we deliberately introduce an error by mangling the code
    #    so this scenario always has a test_failed, regardless of model quality.
    #    In a real agent pipeline this would be an actual test runner result.
    broken_code = (code or "pass") + "\n\nundefined_variable_xyz"  # guaranteed NameError
    result = execute_code(broken_code)

    with activity_span(tracer, "test_execution", {
        "test.attempt":    1,
        "test.success":    result["success"],
        "test.error_type": result["error_type"],
    }):
        time.sleep(0.05)

    with activity_span(tracer, "test_failed", {"error.message": result["error"]}):
        time.sleep(0.05)

    # 4. Correction request — append the error to the conversation
    correction_prompt = (
        f"The code produced this error: {result['error']}. "
        f"Please fix it and return the corrected version."
    )
    messages.append({"role": "assistant", "content": reply})
    messages.append({"role": "user",      "content": correction_prompt})

    with activity_span(tracer, "correction_request", {
        "correction.error_type": result["error_type"],
    }):
        time.sleep(0.05)

    # 5. Model regenerates
    response2 = traced_chat_completion(MODEL, messages)
    reply2    = response2.choices[0].message.content
    code2     = extract_code(reply2)

    with activity_span(tracer, "code_regeneration", {
        "response.has_code":     code2 is not None,
        "response.total_tokens": response2.usage.total_tokens,
    }):
        time.sleep(0.05)

    # 6. Second test — run the actual (unmangled) regenerated code
    result2 = execute_code(code2 or "pass")

    with activity_span(tracer, "test_execution", {
        "test.attempt":    2,
        "test.success":    result2["success"],
        "test.error_type": result2["error_type"],
    }):
        time.sleep(0.05)

    if result2["success"]:
        with activity_span(tracer, "test_passed", {}):
            time.sleep(0.05)
        end_reason = "success_after_correction"
    else:
        with activity_span(tracer, "test_failed", {"error.message": result2["error"]}):
            time.sleep(0.05)
        end_reason = "abandoned_after_second_failure"

    with activity_span(tracer, "session_end", {"end.reason": end_reason}):
        pass


def run_repeated_failure(tracer: trace.Tracer, prompt: str) -> None:
    """
    Code fails twice; session is abandoned without a successful result.

    Process model (two failed attempts, no recovery):
        user_prompt → code_generation → test_execution → test_failed
            → correction_request → code_regeneration → test_execution
            → test_failed → session_end

    Useful for: detecting bottlenecks, identifying prompts that consistently
    fail, and discovering the "abandoned" variant in conformance checking.
    """
    messages = [
        {"role": "system", "content": "You are a helpful Python coding assistant. Always include a complete, runnable code block."},
        {"role": "user",   "content": prompt},
    ]

    # 1. User prompt
    with activity_span(tracer, "user_prompt", {"prompt.text": prompt}):
        time.sleep(0.05)

    # 2. First generation
    response = traced_chat_completion(MODEL, messages)
    reply    = response.choices[0].message.content
    code     = extract_code(reply)

    with activity_span(tracer, "code_generation", {
        "response.has_code":     code is not None,
        "response.total_tokens": response.usage.total_tokens,
    }):
        time.sleep(0.05)

    # 3. First test — mangle to force failure
    broken = (code or "pass") + "\n\nundefined_xyz_1"
    r1 = execute_code(broken)

    with activity_span(tracer, "test_execution", {"test.attempt": 1, "test.success": r1["success"]}):
        time.sleep(0.05)
    with activity_span(tracer, "test_failed", {"error.message": r1["error"]}):
        time.sleep(0.05)

    # 4. Correction request
    messages.append({"role": "assistant", "content": reply})
    messages.append({"role": "user", "content": f"Error: {r1['error']}. Please fix."})

    with activity_span(tracer, "correction_request", {"correction.attempt": 1}):
        time.sleep(0.05)

    # 5. Second generation
    response2 = traced_chat_completion(MODEL, messages)
    reply2    = response2.choices[0].message.content
    code2     = extract_code(reply2)

    with activity_span(tracer, "code_regeneration", {
        "response.has_code":     code2 is not None,
        "response.total_tokens": response2.usage.total_tokens,
    }):
        time.sleep(0.05)

    # 6. Second test — mangle again to force second failure
    broken2 = (code2 or "pass") + "\n\nundefined_xyz_2"
    r2 = execute_code(broken2)

    with activity_span(tracer, "test_execution", {"test.attempt": 2, "test.success": r2["success"]}):
        time.sleep(0.05)
    with activity_span(tracer, "test_failed", {"error.message": r2["error"]}):
        time.sleep(0.05)

    # 7. Abandon — no further correction attempts
    with activity_span(tracer, "session_end", {"end.reason": "abandoned_repeated_failure"}):
        pass


def run_no_code_response(tracer: trace.Tracer, prompt: str) -> None:
    """
    Model returns a text explanation rather than runnable code.
    No test_execution occurs because there is nothing to run.

    Process model (shortest path):
        user_prompt → code_generation → session_end

    Useful for detecting: prompts that should have produced code but didn't —
    a conformance violation if your normative model requires code after every prompt.
    """
    messages = [
        # Deliberately no "always include code" instruction — increases chance
        # the model gives a prose answer for conceptual questions.
        {"role": "system", "content": "You are a helpful Python assistant."},
        {"role": "user",   "content": prompt},
    ]

    # 1. User prompt
    with activity_span(tracer, "user_prompt", {"prompt.text": prompt}):
        time.sleep(0.05)

    # 2. Generation (may or may not contain a code block)
    response = traced_chat_completion(MODEL, messages)
    reply    = response.choices[0].message.content
    code     = extract_code(reply)

    with activity_span(tracer, "code_generation", {
        "response.has_code":     code is not None,
        "response.total_tokens": response.usage.total_tokens,
    }):
        time.sleep(0.05)

    # 3. No test_execution — end immediately
    with activity_span(tracer, "session_end", {
        "end.reason": "no_code_to_test" if code is None else "code_not_tested",
    }):
        pass


# ---------------------------------------------------------------------------
# Dispatcher
# ---------------------------------------------------------------------------

SCENARIO_MAP = {
    "happy_path":       run_happy_path,
    "correction_loop":  run_correction_loop,
    "repeated_failure": run_repeated_failure,
    "no_code_response": run_no_code_response,
}


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def main() -> None:
    # ------------------------------------------------------------------
    # 1. Initialise tracing — must happen before any span is created
    # ------------------------------------------------------------------
    setup_tracing(service_name="mistral-process-mining")
    tracer = trace.get_tracer("session-runner")

    total   = len(SESSIONS)
    passed  = 0
    failed  = 0

    try:
        for i, session in enumerate(SESSIONS, start=1):
            scenario = session["scenario"]
            prompt   = session["prompt"]
            label    = session.get("label", f"{scenario} #{i}")

            print(f"\n[{i}/{total}] {label}")
            print(f"  Scenario : {scenario}")
            print(f"  Prompt   : {prompt[:70]}{'...' if len(prompt) > 70 else ''}")

            scenario_fn = SCENARIO_MAP.get(scenario)
            if scenario_fn is None:
                print(f"  [SKIP] Unknown scenario '{scenario}'", file=sys.stderr)
                continue

            # Each session gets its own root span. This root span's trace_id
            # becomes the case:concept:name in the XES log. All child activity
            # spans are nested under it and share the same trace_id.
            with tracer.start_as_current_span(f"session.{scenario}") as root:
                root.set_attribute("session.scenario", scenario)
                root.set_attribute("session.prompt",   prompt)
                root.set_attribute("session.index",    i)

                try:
                    scenario_fn(tracer, prompt)
                    root.set_status(trace.StatusCode.OK)
                    print(f"  [OK] Session complete.")
                    passed += 1

                except Exception as exc:
                    root.record_exception(exc)
                    root.set_status(trace.StatusCode.ERROR, str(exc))
                    print(f"  [ERROR] {exc}", file=sys.stderr)
                    failed += 1
                    # Continue to next session rather than aborting the whole run

    finally:
        # ------------------------------------------------------------------
        # 2. Flush and close — runs even if a session raised an unhandled error
        # ------------------------------------------------------------------
        shutdown_tracing()

    # ------------------------------------------------------------------
    # 3. Summary
    # ------------------------------------------------------------------
    print(f"\n{'='*50}")
    print(f"Sessions run : {total}")
    print(f"Succeeded    : {passed}")
    print(f"Failed       : {failed}")
    print(f"Spans written to traces.json")
    print(f"Run convert_to_xes.py to produce mistral_trace_log.xes")
    print(f"{'='*50}")


if __name__ == "__main__":
    main()