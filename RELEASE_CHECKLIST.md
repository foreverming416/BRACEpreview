# Checklist before publishing

The folder is cleaned and suitable for continued preparation, but the following items should be completed before the GitHub repository is presented as the paper artifact:

- [ ] Copy the LLM semantic detector used for the reported BRACE results from the other experiment machine, sanitize its configuration, and implement the connected detector interface.
- [x] Retire the marker-based provenance splitter in `authorization.py`; authorization now consumes semantic detector output.
- [ ] Add sanitized experiment launchers, model snapshots, seeds, benchmark versions, and exact evaluation commands.
- [ ] Record the tested AgentDojo release and benchmark configuration.
- [x] Release the original BRACE code under Apache License 2.0.
- [ ] Add authorship and citation metadata after the author list is finalized.
- [ ] Run all tests and a secret scan on the final tree.
- [ ] Review any sample data/results for benchmark redistribution restrictions before committing them.

This repository currently contains no API keys, full experiment logs, model caches, benchmark datasets, or result dumps.

