#!/usr/bin/env bash
# Ghostrix install check: Ollama present, chat and embedding models pulled.
#
# Environment overrides (for testing and custom setups):
#   OLLAMA_BIN              path to the ollama binary (default: ollama)
#   GHOSTRIX_CHAT_MODEL     default qwen3:latest
#   GHOSTRIX_EMBED_MODEL    default nomic-embed-text
#
# Flags:
#   --dry-run               print what would happen, change nothing
#
# Exit codes: 0 ready, 1 Ollama missing, 2 a pull failed.

set -euo pipefail

OLLAMA_BIN="${OLLAMA_BIN:-ollama}"
CHAT_MODEL="${GHOSTRIX_CHAT_MODEL:-qwen3:latest}"
EMBED_MODEL="${GHOSTRIX_EMBED_MODEL:-nomic-embed-text}"
DRY_RUN=0

for arg in "$@"; do
  case "$arg" in
    --dry-run) DRY_RUN=1 ;;
    *) echo "unknown argument: $arg" >&2; exit 1 ;;
  esac
done

if ! command -v "$OLLAMA_BIN" >/dev/null 2>&1; then
  echo "Ollama not found. Install it from https://ollama.com and run this again." >&2
  exit 1
fi

if [ "$DRY_RUN" -eq 1 ]; then
  echo "[dry-run] would check models via: $OLLAMA_BIN list"
  listing=""
else
  listing="$("$OLLAMA_BIN" list 2>/dev/null || true)"
fi

have_model() {
  # Match the model name as the first column, with or without an explicit tag.
  printf '%s\n' "$listing" | awk 'NR>1 {print $1}' | grep -Fxq "$1" \
    || printf '%s\n' "$listing" | awk 'NR>1 {print $1}' | grep -Fxq "$1:latest"
}

ensure_model() {
  local model="$1"
  if have_model "$model"; then
    echo "ok: $model already present"
    return 0
  fi
  if [ "$DRY_RUN" -eq 1 ]; then
    echo "[dry-run] would pull: $model"
    return 0
  fi
  echo "pulling: $model"
  if ! "$OLLAMA_BIN" pull "$model"; then
    echo "failed to pull $model" >&2
    exit 2
  fi
}

ensure_model "$CHAT_MODEL"
ensure_model "$EMBED_MODEL"

echo "Ghostrix models ready. (The chat command is not built yet.)"
