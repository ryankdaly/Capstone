#!/usr/bin/env bash
# scripts/dev-install.sh — Build and locally reinstall the hpema wheel.
#
# Usage:
#   bash scripts/dev-install.sh          # build + reinstall
#   bash scripts/dev-install.sh --clean  # uninstall only
#
# Requires: uv  (https://github.com/astral-sh/uv)
#
# Config isolation:
#   The installed `hpema` binary reads from ~/.hpema/hpema_config.yaml by default.
#   It ignores relative HPEMA_CONFIG env vars (like hpema_config.local.yaml) that
#   are only meaningful in a source checkout — so your dev config does NOT bleed in.
#
#   To test as a completely fresh user (no config, wizard auto-pops):
#     rm -f ~/.hpema/hpema_config.yaml ~/.hpema/.config_verified && hpema
#
#   Config in ~/.hpema/ survives reinstalls.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

GREEN='\033[0;32m'; BOLD='\033[1m'; NC='\033[0m'
ok()   { echo -e "${GREEN}✓  $*${NC}"; }
info() { echo -e "${BOLD}▶  $*${NC}"; }

if [[ "${1:-}" == "--clean" ]]; then
    info "Uninstalling hpema..."
    uv pip uninstall hpema -y 2>/dev/null && ok "Uninstalled" || echo "  (not installed)"
    exit 0
fi

# Uninstall existing (ignore if not present)
uv pip uninstall hpema -y 2>/dev/null || true

# Build a fresh wheel from current source
info "Building wheel..."
python -m build --wheel --outdir dist/ -q
WHEEL=$(ls dist/hpema-*.whl | tail -1)
ok "Built: $WHEEL"

# Install it
info "Installing..."
uv pip install "$WHEEL"
ok "Installed — run: hpema"
echo ""
echo "  Config persists at: ~/.hpema/hpema_config.yaml"
echo "  To uninstall:       bash scripts/dev-install.sh --clean"
