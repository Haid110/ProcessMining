# convert_to_xes.py
#
# PURPOSE:
#   Reads the JSONL span file produced by main.py (traces.json) and converts
#   it into an XES event log (mistral_trace_log.xes) suitable for import into
#   process mining tools such as ProM, Disco, or pm4py itself.
#
# XES MAPPING:
#   OTel concept          → XES concept
#   ─────────────────────────────────────────────────
#   trace_id              → case:concept:name  (one process instance per trace)
#   span name             → concept:name       (activity label)
#   start_time            → time:timestamp
#   llm.request.model     → model              (custom attribute)
#   llm.usage.total_tokens→ tokens             (custom attribute)
#
# RUN THIS SCRIPT SEPARATELY after main.py has produced traces.json:
#   python convert_to_xes.py
#
# REQUIREMENTS:
#   pm4py
#   pandas

import json
import sys
import pandas as pd
import pm4py

# ---------------------------------------------------------------------------
# 1. Load spans from the JSONL file
# ---------------------------------------------------------------------------
# traces.json is written in JSONL-ish format by ConsoleSpanExporter:
# multiple JSON objects concatenated without a wrapping array. We use
# JSONDecoder.raw_decode() to extract them one by one without needing
# newline separators.

spans = []
decoder = json.JSONDecoder()

try:
    with open("traces.json", "r", encoding="utf-8") as f:
        content = f.read().strip()
except FileNotFoundError:
    print("[ERROR] traces.json not found. Run main.py first to generate spans.")
    sys.exit(1)

if not content:
    print("[ERROR] traces.json is empty. Run main.py first to generate spans.")
    sys.exit(1)

pos = 0
parse_errors = 0
while pos < len(content):
    # Skip whitespace between JSON objects (newlines, spaces, etc.)
    while pos < len(content) and content[pos].isspace():
        pos += 1
    if pos >= len(content):
        break
    try:
        obj, pos = decoder.raw_decode(content, pos)
        spans.append(obj)
    except json.JSONDecodeError as e:
        # Skip malformed entries (e.g. partial writes from a crashed process)
        # and continue parsing the rest of the file.
        parse_errors += 1
        pos += 1  # advance by one character and try again

print(f"Loaded {len(spans)} spans from traces.json ({parse_errors} parse errors skipped).")

# ---------------------------------------------------------------------------
# 2. Filter and flatten spans into a process mining event table
# ---------------------------------------------------------------------------
# Only spans named "mistral_chat_completion" are kept. This matches the span
# name set in genai_tracing.traced_chat_completion(). Add additional names
# here as you instrument more activity types (e.g. "user_prompt", "tool_call").
#
# Spans are skipped when:
#   - status_code is ERROR    → failed API calls have no meaningful attributes
#   - attributes dict is empty → span data is incomplete (usually also an error)
#
# Trace IDs arrive as hex strings like "0xd117e6935ca6bada3e6f398c9c47c92a".
# We strip the "0x" prefix for cleaner case IDs in the XES file.


ACTIVITY_NAMES = {
    "user_prompt",
    "code_generation",
    "test_execution",
    "test_passed",
    "test_failed",
    "correction_request",
    "code_regeneration",
    "session_end",
}

rows = []
skipped_error = 0
skipped_no_attrs = 0

for span in spans:
    # --- Activity filter ---
    if span.get("name") not in ACTIVITY_NAMES:
        continue

    # --- Skip failed spans ---
    status_code = span.get("status", {}).get("status_code", "")
    if status_code == "ERROR":
        skipped_error += 1
        continue

    # --- Skip spans with no attributes (incomplete data) ---
    attributes = span.get("attributes", {})
    if not attributes:
        skipped_no_attrs += 1
        continue

    # --- Normalise trace ID -------------------------------------------------
    # OTel serialises trace IDs as hex strings (with or without "0x" prefix).
    # Normalise to a plain lowercase hex string for consistent case IDs.
    raw_trace_id = span["context"]["trace_id"]
    trace_id = str(raw_trace_id).lower().replace("0x", "")

    # --- Parse timestamp ----------------------------------------------------
    # utc=True produces timezone-aware timestamps, which pm4py requires for
    # correct event ordering and conformance checking.
    timestamp = pd.to_datetime(span["start_time"], utc=True)

    rows.append({
        # Required pm4py columns
        "case:concept:name": trace_id,
        "concept:name":      span["name"],
        "time:timestamp":    timestamp,

        # Custom attributes — useful for filtering and performance analysis
        # in tools like Disco or ProM's performance spectrum plugin.
        "model":             attributes.get("llm.request.model", "unknown"),
        "response_model":    attributes.get("llm.response.model", "unknown"),
        "prompt_tokens":     attributes.get("llm.usage.prompt_tokens", 0),
        "completion_tokens": attributes.get("llm.usage.completion_tokens", 0),
        "tokens":            attributes.get("llm.usage.total_tokens", 0),

        # Duration in milliseconds — useful for performance analysis.
        # start_time and end_time are ISO 8601 strings; subtracting them gives
        # a timedelta which we convert to milliseconds.
        "duration_ms": (
            pd.to_datetime(span["end_time"], utc=True)
            - timestamp
        ).total_seconds() * 1000,
    })

# ---------------------------------------------------------------------------
# 3. Report and exit early if nothing usable was found
# ---------------------------------------------------------------------------
print(
    f"Events kept: {len(rows)} | "
    f"Skipped (ERROR status): {skipped_error} | "
    f"Skipped (no attributes): {skipped_no_attrs}"
)

if not rows:
    print(
        "[WARNING] No valid spans found. Check that:\n"
        "  1. main.py ran successfully (no ERROR spans).\n"
        "  2. The model name is valid (mistral-tiny is retired).\n"
        "  3. spans in traces.json are named 'mistral_chat_completion'."
    )
    sys.exit(0)

# ---------------------------------------------------------------------------
# 4. Build the DataFrame and export to XES
# ---------------------------------------------------------------------------
df = pd.DataFrame(rows)

# pm4py.format_dataframe() renames columns to the standard pm4py names and
# sorts events by timestamp within each case — required before writing XES.
df = pm4py.format_dataframe(
    df,
    case_id="case:concept:name",
    activity_key="concept:name",
    timestamp_key="time:timestamp",
)

output_path = "mistral_trace_log.xes"
pm4py.write_xes(df, output_path)

print(
    f"\nSuccess! Written {len(rows)} events across "
    f"{df['case:concept:name'].nunique()} cases to '{output_path}'.\n"
    f"Open this file in ProM, Disco, or use pm4py for further analysis."
)
