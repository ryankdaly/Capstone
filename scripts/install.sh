#!/usr/bin/env bash
# =============================================================================
# install.sh — End-user install from a HPEMA platform bundle
# =============================================================================
# Run this from the extracted bundle directory:
#   tar -xzf hpema-VERSION-linux-x64.tar.gz
#   cd hpema-VERSION-linux-x64
#   bash install.sh
# =============================================================================
set -euo pipefail

BUNDLE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
HPEMA_HOME="${HPEMA_HOME:-$HOME/.hpema}"

GREEN='\033[0;32m'; YELLOW='\033[1;33m'; BOLD='\033[1m'; NC='\033[0m'
ok()   { echo -e "${GREEN}✓  $*${NC}"; }
warn() { echo -e "${YELLOW}⚠  $*${NC}"; }
info() { echo -e "${BOLD}▶  $*${NC}"; }

echo ""
echo -e "${BOLD}HPEMA Installer${NC}"
echo "─────────────────────────────────────────"
echo ""

# ─── Python check ────────────────────────────────────────────────────────────
info "Checking Python..."
if ! command -v python3 &>/dev/null; then
    echo "✗ python3 not found. Install Python 3.11+ from https://python.org" >&2
    exit 1
fi
py_ver=$(python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
echo "  Found Python $py_ver"

# ─── Install wheel ───────────────────────────────────────────────────────────
info "Installing HPEMA..."
WHEEL=$(ls "$BUNDLE_DIR"/hpema-*.whl 2>/dev/null | head -1)
if [[ -z "$WHEEL" ]]; then
    echo "✗ No wheel found in bundle directory." >&2
    exit 1
fi
pip install --quiet "$WHEEL"
ok "HPEMA installed"

# ─── Install Dafny ───────────────────────────────────────────────────────────
if [[ -d "$BUNDLE_DIR/dafny" ]]; then
    info "Installing Dafny..."
    DAFNY_DEST="$HPEMA_HOME/dafny"
    mkdir -p "$DAFNY_DEST"
    cp -r "$BUNDLE_DIR/dafny/." "$DAFNY_DEST/"

    # Find the dafny executable
    DAFNY_BIN=$(find "$DAFNY_DEST" -name "dafny" ! -name "*.dll" ! -name "*.sh" -type f 2>/dev/null | head -1)
    if [[ -n "$DAFNY_BIN" ]]; then
        chmod +x "$DAFNY_BIN"
        mkdir -p "$HPEMA_HOME"
        echo "$DAFNY_BIN" > "$HPEMA_HOME/dafny_path"
        ok "Dafny installed: $DAFNY_BIN"
    else
        warn "Dafny binary not found in bundle — verification stage will require manual setup."
    fi
else
    warn "No Dafny bundle found — run 'hpema /setup' after install to configure."
fi

echo ""
ok "Done."
echo ""
echo "  Run:  hpema"
echo "  Then: /setup   to configure your API key"
echo ""
echo "  Config will be saved to: $HPEMA_HOME/hpema_config.yaml"
echo ""
