# Ghostrix

A text companion with memory that runs entirely on a local model. No cloud, no fallback. MIT licence.

Status: early build. The memory core is written and tested. The chat layer, the Ollama integration and the install script are not built yet.

## Memory core (`ghostrix/memory.py`)

- **Raw log:** every message kept in one SQLite file.
- **Session summaries:** each summary must link to at least one source message from the same session.
- **Retrieval:** top matches across summaries and raw snippets. The embedder is pluggable; the default is a deterministic stub, which tests plumbing, not quality.
- **Profile:** the companion proposes entries. Nothing enters the stable profile until the user accepts it.

## Tests

    python3 -m unittest discover -s tests -v

The 20-fact recall test runs on the stub embedder. Real memory and personality tests run with the local model.
