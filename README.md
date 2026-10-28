# BRACE

**BRACE** (*Binding Runtime Authorization to Contextual Evidence*) is a runtime authorization layer for tool-using agents. It binds sensitive tool arguments to contextual evidence, enforces policy at execution time, and issues single-use execution permits tied to the exact action being executed.

This repository is a cleaned research-code release. It contains the authorization core, a planner-independent checklist adapter, and unit tests. Experiment outputs, API credentials, model caches, local paths, and third-party repository copies are intentionally excluded.

## Repository layout

```text
src/brace/authorization.py   Evidence extraction and runtime authorization
src/brace/execution_permit.py Exact-action, single-use execution permits
src/brace/planner_adapter.py Planner-independent checklist normalization
tests/                       Authorization and binding tests
docs/architecture.md         Short implementation overview
```

## Semantic evidence detector interface

BRACE does not infer trust from benchmark markers. The caller must provide a
semantic detector implementing `detect`. It receives the raw response plus
runtime-owned provenance and returns the trusted and untrusted portions:

```python
from brace import AuthorizationExecutorV2, DetectedEvidenceItem, EvidenceDetection

class SemanticDetector:
    def detect(self, *, text, source_tool, tool_call_id):
        result = call_your_llm_detector(text, source_tool, tool_call_id)
        return EvidenceDetection(
            trusted_text=result["trusted_text"],
            untrusted_text=result["untrusted_text"],
            structured=result.get("structured"),
            evidence_items=tuple(
                DetectedEvidenceItem(
                    record_id=item["record_id"],
                    entity=item.get("entity"),
                    field=item["field"],
                    value=item["value"],
                )
                for item in result.get("evidence_items", ())
            ),
        )

gate = AuthorizationExecutorV2(planner, SemanticDetector())
```

`source_tool` and `tool_call_id` are taken from the tool runtime rather than
from detector-generated text. The released authorization core uses the
detector output to construct provenance-bearing evidence records and then
binds sensitive arguments to those records.

Fine-grained plan conditions can require the complete relation rather than
mere value occurrence:

```json
{
  "user_id": {
    "source_tool": "search_contacts",
    "entity": "Alice",
    "field": "user_id"
  }
}
```

For such a condition, a value is accepted only when one detector evidence
item jointly matches the runtime source tool, call ID (when specified), record
ID (when specified), entity, field, and value. The matched lineage is retained
in the authorization event. String conditions such as
`"user_id": "search_contacts"` remain supported for older source-only plans.

## Installation

BRACE currently integrates with AgentDojo's message and function-runtime interfaces. Use Python 3.11 or newer, install the official AgentDojo checkout in the same environment, and then install this package:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e ".[dev]"
```

The checklist adapter is planner-independent and has no external planner dependency.

## Tests

After AgentDojo is available in the active environment:

```powershell
python -m pytest -q
```

All planner-adapter tests run without an external planner checkout.

## Reproducing the paper experiments

The detector interface is connected, but the concrete LLM detector and public experiment configurations should be added after the implementation used for the reported experiments is copied from the experiment machine and sanitized. See [RELEASE_CHECKLIST.md](RELEASE_CHECKLIST.md). The present tree should therefore be treated as a cleaned core-code package, not yet as a complete artifact for reproducing every table in the paper.

## License

BRACE is released under the Apache License 2.0. See [LICENSE](LICENSE).

