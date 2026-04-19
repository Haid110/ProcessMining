import json
import pandas as pd
import pm4py
from pm4py.objects.conversion.log import converter as log_converter

# 1. Load the JSON (assuming JSONL format or single entries)
spans = []
decoder = json.JSONDecoder()
with open("traces.json", "r", encoding="utf-8") as f:
    # If your file is one big JSON array, use json.load(f)
    # If it's multiple JSON objects (JSONL), use this loop:
    content = f.read().strip()
    pos = 0
    while pos < len(content):
        # Skip any whitespace between JSON objects
        while pos < len(content) and content[pos].isspace():
            pos += 1
        if pos >= len(content):
            break
        # Decode one JSON object and update position for the next
        obj, pos = decoder.raw_decode(content, pos)
        spans.append(obj)

# 2. Flatten relevant data for Process Mining
rows = []
for s in spans:
    if s.get("name") == "mistral_chat_completion":
        rows.append({
            "case:concept:name": s["context"]["trace_id"],  # Trace ID = Case ID
            "concept:name": s["name"],                     # Activity Name
            "time:timestamp": pd.to_datetime(s["start_time"]),
            "model": s["attributes"].get("llm.request.model"),
            "tokens": s["attributes"].get("llm.usage.total_tokens", 0)
        })
if not rows:
    print("No valid spans found in traces.json.")
else:
    df = pd.DataFrame(rows) #Convert to DataFrame and export to XES
    # Ensure PM4Py can read the columns correctly
    df = pm4py.format_dataframe(df, case_id='case:concept:name', activity_key='concept:name', timestamp_key='time:timestamp')
    pm4py.write_xes(df, "mistral_trace_log.xes")
    print(f"Successfully created mistral_trace_log.xes with {len(rows)} events.")


