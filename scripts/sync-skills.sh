#!/usr/bin/env bash
# Sync this repo's skills (top-level dirs containing SKILL.md) into
# Antigravity skill locations.
#
#   ./scripts/sync-skills.sh           # workspace scope: .agents/skills/
#   ./scripts/sync-skills.sh --global  # also install to ~/.gemini/antigravity/skills/
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WORKSPACE_DIR="$REPO_ROOT/.agents/skills"
GLOBAL_DIR="${HOME}/.gemini/antigravity/skills"

SKILLS=()
while IFS= read -r f; do
  SKILLS+=("$(basename "$(dirname "$f")")")
done < <(find "$REPO_ROOT" -mindepth 2 -maxdepth 2 -name SKILL.md -not -path "$REPO_ROOT/.agents/*" | sort)

if [[ ${#SKILLS[@]} -eq 0 ]]; then
  echo "No skills found (no top-level directory with a SKILL.md)." >&2
  exit 1
fi

sync_to() {
  local dest="$1"
  mkdir -p "$dest"
  for skill in "${SKILLS[@]}"; do
    rm -rf "$dest/$skill"
    cp -R "$REPO_ROOT/$skill" "$dest/$skill"
    echo "synced: $skill -> $dest/$skill"
  done
}

sync_to "$WORKSPACE_DIR"

if [[ "${1:-}" == "--global" ]]; then
  sync_to "$GLOBAL_DIR"
fi
