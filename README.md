# GenAI Process Mining

> **Internship Research Project** — Università degli Studi di Messina  
> Exploring how interactions with generative AI coding agents can be captured
> as observability traces, transformed into structured event logs, and analysed
> using process mining techniques.

---

## Research Question

Can OpenTelemetry spans produced during GenAI coding sessions be meaningfully
converted into XES event logs and analysed with process mining algorithms to
reveal interaction patterns, bottlenecks, and conformance violations?

---

## Pipeline Overview

```
GenAI Coding Session
        │
        ▼
OpenTelemetry Spans       ← configure_tracing.py + genai_tracing.py
        │
        ▼
   traces.json            ← JSONL file, one span object per line
        │
        ▼
mistral_trace_log.xes     ← convert_to_xes.py
        │
        ▼
Process Mining Analysis   ← ProM / pm4py
  (Discovery, Conformance Checking, Declare Constraints)
```

---

## Repository Structure

```
GenAI_Process_Mining/
│
├── src/
│   ├── configure_tracing.py   # OpenTelemetry provider setup and teardown
│   ├── genai_tracing.py       # traced_chat_completion() wrapper for Mistral
│   ├── main.py                # session simulation engine — runs all scenarios
│   └── convert_to_xes.py      # converts traces.json → XES event log
│
├── data/
│   └── mistral_trace_log.xes  # XES log ready for ProM / pm4py
│
├── models/                    # saved process models (add after ProM analysis)
│   ├── inductive_model.pnml
│   └── heuristics_model.pnml
│
├── figures/                   # exported process map images (add after analysis)
│   ├── inductive_miner.png
│   └── heuristics_miner.png
│
├── requirements.txt
├── .gitignore
└── README.md
```

> `traces.json` is excluded from version control via `.gitignore` because it
> grows with every run and contains raw API response data. The converted
> `mistral_trace_log.xes` is committed instead as it is the reproducible
> analytical artefact.

---

## Activity Vocabulary

Each step of a coding session is recorded as a named OpenTelemetry span.
These span names become activity labels in the XES event log.

| Activity | Description |
|---|---|
| `user_prompt` | User sends a coding request to the model |
| `code_generation` | Model returns code for the first time |
| `test_execution` | Generated code is extracted and executed locally |
| `test_passed` | Execution completed without raising an exception |
| `test_failed` | Execution raised an exception |
| `correction_request` | User asks the model to fix the broken code |
| `code_regeneration` | Model returns a revised version of the code |
| `session_end` | Conversation is closed (successfully or abandoned) |

Each session = one OTel trace = one XES **case**. Each activity = one child span = one XES **event**.

---

## Simulated Scenarios

Four interaction patterns are simulated to produce a log with realistic process variance:

### `happy_path` — 5 sessions
Code works correctly on the first attempt.
```
user_prompt → code_generation → test_execution → test_passed → session_end
```

### `correction_loop` — 5 sessions
Code fails once; user requests a correction; model fixes it successfully.
```
user_prompt → code_generation → test_execution → test_failed
    → correction_request → code_regeneration → test_execution
    → test_passed → session_end
```

### `repeated_failure` — 2 sessions
Code fails twice; session is abandoned without a successful result.
```
user_prompt → code_generation → test_execution → test_failed
    → correction_request → code_regeneration → test_execution
    → test_failed → session_end
```

### `no_code_response` — 2 sessions
Model returns a text explanation rather than runnable code; no test step occurs.
```
user_prompt → code_generation → session_end
```

**Total: 14 sessions → ~80–100 spans → 14 XES cases**

---

## Setup

### 1. Clone the repository

```bash
git clone https://github.com/<your-username>/GenAI_Process_Mining.git
cd GenAI_Process_Mining
```

### 2. Create and activate the conda environment

```bash
conda create -n mistral python=3.11
conda activate mistral
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Set your Mistral API key

**Windows (Command Prompt):**
```cmd
set MISTRAL_API_KEY=5MNQbzChjDP1VkMhzcQ1Pf49Y3JDoBCx
```

**Windows (PowerShell):**
```powershell
$env:MISTRAL_API_KEY="5MNQbzChjDP1VkMhzcQ1Pf49Y3JDoBCx"
```

**macOS / Linux:**
```bash
export MISTRAL_API_KEY=5MNQbzChjDP1VkMhzcQ1Pf49Y3JDoBCx
```

> Get a free API key at [console.mistral.ai](https://console.mistral.ai).

---

## Running the Pipeline

### Step 1 — Clear any previous trace data

```bash
del data\traces.json        # Windows
rm data/traces.json         # macOS / Linux
```

Skip this step if you want to accumulate more cases across multiple runs.

### Step 2 — Run the session simulator

```bash
python src/main.py
```

Expected output:

```
[1/14] happy_path #1
  Scenario : happy_path
  Prompt   : Write a Python function that converts feet to centimetres.
  [OK] Session complete.

[2/14] happy_path #2
  ...

==================================================
Sessions run : 14
Succeeded    : 14
Failed       : 0
Spans written to traces.json
Run convert_to_xes.py to produce mistral_trace_log.xes
==================================================
```

### Step 3 — Convert spans to XES

```bash
python src/convert_to_xes.py
```

Expected output:

```
Loaded 98 spans from traces.json (0 parse errors skipped).
Events kept: 84 | Skipped (ERROR status): 0 | Skipped (no attributes): 14

Success! Written 84 events across 14 cases to 'mistral_trace_log.xes'.
```

> The 14 skipped spans are the session root spans (`session.happy_path` etc.),
> which are structural containers, not process activities.

### Step 4 — Open in a process mining tool

**ProM** (recommended for full analysis):
1. Download ProM from [promtools.org](https://promtools.org)
2. File → Import → select `data/mistral_trace_log.xes`
3. Run **Inductive Miner** or **Heuristics Miner** from the plugin list

**pm4py** (programmatic, in Python):
```python
import pm4py

log = pm4py.read_xes("data/mistral_trace_log.xes")

# Discover process model using Inductive Miner
net, im, fm = pm4py.discover_petri_net_inductive(log)
pm4py.view_petri_net(net, im, fm)

# View directly-follows graph
pm4py.view_dfg(log)
```

---

## Key Design Decisions

**Why `mistralai==0.1.8`?**  
The current Mistral SDK (v1.x) is incompatible with `opentelemetry-instrumentation-mistralai==0.57.0`, which was written against the old `MistralClient` interface. Pinning to 0.1.8 ensures the OTel instrumentation can correctly patch the client.

**Why `SimpleSpanProcessor` instead of `BatchSpanProcessor`?**  
`BatchSpanProcessor` buffers spans asynchronously. In a short-lived script, an early shutdown drops the buffer. `SimpleSpanProcessor` exports each span synchronously the moment it closes, so no data is lost even if the process crashes.

**Why is `traces.json` opened in append mode?**  
Multiple runs accumulate cases in a single file, increasing the dataset size for process mining without manual merging. Delete the file before a clean run.

**Why are `test_failed` spans deliberately forced in some scenarios?**  
The `correction_loop` and `repeated_failure` scenarios mangle generated code to guarantee a `NameError`. This ensures consistent process variant shapes regardless of model quality, making the event log suitable for repeatable process mining experiments.

---

## Known Limitations

- **Single LLM provider**: only Mistral is instrumented. Claude, Copilot, and Gemini would require separate OTel wrappers and API clients.
- **Synthetic test execution**: `exec()` runs code in an isolated namespace with no imports, so real-world library-dependent code will fail even when correct. A proper test runner (e.g. `pytest`) would be more realistic.
- **No user identity**: all sessions share the same `service.name`. Multi-user or multi-agent scenarios would need a `user.id` attribute on each span.
- **OTel ≠ process log**: OTel traces are designed for distributed systems debugging, not process analysis. The `trace_id` → `case_id` mapping is a pragmatic approximation; a native process logging layer would be more principled.
- **One correction attempt maximum**: scenarios currently model at most one correction loop. Real coding sessions can have many more iterations.

---

## Next Steps

- [ ] Run Inductive Miner and Heuristics Miner in ProM; compare resulting models
- [ ] Identify happy path, correction loop, and abandoned variants
- [ ] Define a normative Petri net and run token-based conformance checking
- [ ] Model Declare constraints (e.g. `test_execution` must follow `code_generation`)
- [ ] Extend to Claude and Gemini APIs for cross-agent comparison
- [ ] Increase session count to 50+ for statistically meaningful variant analysis

---

## Dependencies

```
mistralai==0.1.8
opentelemetry-sdk>=1.20.0,<2.0.0
opentelemetry-instrumentation-mistralai==0.57.0
pm4py
pandas
```

Full pinned versions in `requirements.txt`.

---

## Author

**Syed Fasih Haider**  
Internship project supervised by Prof. Merlino  
Università degli Studi di Messina
