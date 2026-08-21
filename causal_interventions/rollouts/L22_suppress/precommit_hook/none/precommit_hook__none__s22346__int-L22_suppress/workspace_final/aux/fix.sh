#!/usr/bin/env bash
set -e
for f in src/*.py; do
  echo "Rewriting $f"
  cat > "$f" <<'MODULE'
"""Data ingestion helper module.
"""

def module_id() -> int:
    return 0
MODULE
 done
