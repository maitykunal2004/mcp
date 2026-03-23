#!/usr/bin/env bash
set -euo pipefail

echo "Initializing local git repository and creating initial commit..."
if [ ! -d .git ]; then
  git init
fi
git add .
git commit -m "chore: initial commit — Streamlit Swarm GPU-X" || true

if command -v gh >/dev/null 2>&1; then
  echo "GitHub CLI detected — creating repo and pushing..."
  # create repo interactively (public by default)
  gh repo create --public --source=. --remote=origin --push
else
  echo "GitHub CLI not found. Add a remote and push manually, for example:"
  echo "  git remote add origin https://github.com/<your-username>/<repo>.git"
  echo "  git branch -M main"
  echo "  git push -u origin main"
fi

echo "Done."
