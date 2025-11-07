#!/usr/bin/env bash
set -euo pipefail

NEW_CODE_SRC="${1:-}"
if [[ -z "$NEW_CODE_SRC" || ! -d "$NEW_CODE_SRC" ]]; then
  echo "Usage: $0 /absolute/path/to/new_code_dir" >&2
  exit 1
fi

# 0) Safety checks
if ! git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
  echo "Not inside a git repo." >&2
  exit 1
fi
if [[ -n "$(git status --porcelain)" ]]; then
  echo "Working tree not clean. Commit or stash first." >&2
  exit 1
fi

# 1) Snapshot current state
git tag -a "pre-revamp-$(date +%Y%m%d-%H%M%S)" -m "Snapshot before revamp"

# 2) Work on a branch
BASE_BRANCH="$(git rev-parse --abbrev-ref HEAD)"
git checkout -b revamp || git checkout revamp

# 3) Optional: dry run to preview changes
echo "Dry-run of sync (no writes):"
rsync -a --delete --dry-run --exclude ".git" "$NEW_CODE_SRC"/ ./ | sed -e 's/^/  /'

read -p "Proceed with replacement? [y/N] " yn
[[ "$yn" == "y" || "$yn" == "Y" ]] || { echo "Aborted."; exit 1; }

# 4) Remove everything tracked + untracked, keep .git
git rm -r --cached . || true
git clean -fdx

# 5) Copy new tree in
rsync -a --delete --exclude ".git" "$NEW_CODE_SRC"/ ./

# 6) Housekeeping: keep your git config files if needed
# e.g., keep existing CODEOWNERS if you had one and new tree lacks it
if [[ -f ".github/CODEOWNERS.bak" ]]; then mv .github/CODEOWNERS.bak .github/CODEOWNERS; fi

# 7) Commit new tree
git add -A
git commit -m "Revamp: replace codebase with new slum-cluster–oriented implementation"

# 8) Push and show next steps
git push -u origin HEAD
echo
echo "Done. Open a PR from 'revamp' into '$BASE_BRANCH'."
echo "Tag created: $(git describe --tags --abbrev=0)"
