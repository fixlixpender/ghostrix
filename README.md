# Ghostrix

A text companion with memory that runs entirely on a local model. No cloud, no fallback. MIT licence.

Status: early build. Built and tested without a model: the memory core, character loading, config, prompt assembly, and the install check script. Not built yet: the chat loop and the Ollama integration. Those need a model to test.

## Memory core (`ghostrix/memory.py`)

- **Raw log:** every message kept in one SQLite file.
- **Session summaries:** each summary must link to at least one source message from the same session.
- **Retrieval:** top matches across summaries and raw snippets. The embedder is pluggable; the default is a deterministic stub, which tests plumbing, not quality.
- **Profile:** the companion proposes entries. Nothing enters the stable profile until the user accepts it.

## Character, config, prompt

- `examples/character.example.toml` is a character file: name, identity, system prompt.
- `ghostrix/config.py` holds defaults (Qwen3 8B chat model, embedding model nomic-embed-text). Unknown keys are errors.
- `ghostrix/prompt.py` builds the system prompt from character, accepted profile, and memories, within a character budget.

## Install check

    ./install.sh             # checks Ollama, pulls qwen3:latest and nomic-embed-text if missing
    ./install.sh --dry-run   # shows what would happen

## Tests

    python3 -m unittest discover -s tests -v

The 20-fact recall test runs on the stub embedder. Real memory and personality tests run with the local model.
