#!/usr/bin/env bash
# =============================================================================
# scripts/bundle.sh — Build distributable artifacts for HPEMA
# =============================================================================
#
# Produces (in dist/):
#   hpema-VERSION-py3-none-any.whl       Pure-Python wheel (upload to PyPI)
#   hpema-VERSION.tar.gz                 Source distribution
#   hpema-VERSION-linux-x64.tar.gz       Standalone bundle, Dafny embedded
#   hpema-VERSION-osx-x64.tar.gz         Standalone bundle, Dafny embedded
#   hpema-VERSION-windows-x64.zip        Standalone bundle, Dafny embedded
#
# Usage:
#   bash scripts/bundle.sh                        # full build
#   bash scripts/bundle.sh --wheel-only           # skip platform tarballs
#   bash scripts/bundle.sh --skip-tests           # skip pytest (not recommended)
#   DAFNY_VERSION=4.9.0 bash scripts/bundle.sh    # override Dafny version
#
# Requirements (install once):
#   pip install build twine
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
DIST="$ROOT/dist"

DAFNY_VERSION="${DAFNY_VERSION:-4.9.0}"
WHEEL_ONLY=false
SKIP_TESTS=false

for arg in "$@"; do
    case "$arg" in
        --wheel-only)  WHEEL_ONLY=true ;;
        --skip-tests)  SKIP_TESTS=true ;;
    esac
done

# Colour helpers
RED='\033[0;31m'; YELLOW='\033[1;33m'; GREEN='\033[0;32m'; BOLD='\033[1m'; NC='\033[0m'
info()  { echo -e "${BOLD}▶  $*${NC}"; }
ok()    { echo -e "${GREEN}✓  $*${NC}"; }
warn()  { echo -e "${YELLOW}⚠  $*${NC}"; }
die()   { echo -e "${RED}✗  $*${NC}" >&2; exit 1; }

cd "$ROOT"

# ─── Preflight checks ────────────────────────────────────────────────────────
info "Preflight checks"

# Python version
python_version=$(python3 --version 2>&1 | grep -oP '\d+\.\d+' | head -1)
python_major=$(echo "$python_version" | cut -d. -f1)
python_minor=$(echo "$python_version" | cut -d. -f2)
if [[ "$python_major" -lt 3 ]] || [[ "$python_major" -eq 3 && "$python_minor" -lt 11 ]]; then
    die "Python 3.11+ required, found $python_version"
fi
ok "Python $python_version"

# pyproject.toml exists
[[ -f "$ROOT/pyproject.toml" ]] || die "pyproject.toml not found — run from project root"

# Resolve version from pyproject.toml
VERSION=$(python3 -c "
import sys
if sys.version_info >= (3, 11):
    import tomllib
    with open('pyproject.toml', 'rb') as f:
        d = tomllib.load(f)
else:
    import tomli
    with open('pyproject.toml', 'rb') as f:
        d = tomli.load(f)
print(d['project']['version'])
" 2>/dev/null) || die "Could not read version from pyproject.toml"

[[ -n "$VERSION" ]] || die "version is empty in pyproject.toml"
info "Version: $VERSION"

# Build tools
python3 -m build --version &>/dev/null || die "'build' not installed. Run: pip install build"
python3 -m twine --version &>/dev/null || die "'twine' not installed. Run: pip install twine"

# Dafny (warn only — it's optional for wheel-only builds)
DAFNY_BIN=""
if command -v dafny &>/dev/null; then
    DAFNY_BIN="$(which dafny)"
    ok "Dafny detected: $DAFNY_BIN"
elif [[ -f "$ROOT/local/installs/dafny/dafny" ]]; then
    DAFNY_BIN="$ROOT/local/installs/dafny/dafny"
    ok "Dafny found in local/installs: $DAFNY_BIN"
else
    warn "Dafny not found. Verification tests will be skipped."
    warn "Install: brew install dotnet dafny  (macOS)  |  see docs/LOCAL_SETUP.md"
fi

# Git — check for uncommitted changes (warn only; allow dirty bundle for dev)
if git diff --quiet HEAD 2>/dev/null; then
    ok "Git working tree clean"
else
    warn "Uncommitted changes detected. Bundle will include them."
fi

echo ""

# ─── Tests ───────────────────────────────────────────────────────────────────
if [[ "$SKIP_TESTS" == "false" ]]; then
    info "Running unit tests  (integration tests skipped — they require live API keys)"
    if python3 -m pytest tests/unit/ -x -q -m "not integration" 2>&1; then
        ok "Unit tests passed"
    else
        die "Unit tests failed. Fix before bundling, or use --skip-tests (not recommended)."
    fi
    echo ""

    if [[ -n "$DAFNY_BIN" ]]; then
        info "Running Dafny smoke tests"
        if python3 -m pytest tests/ -x -q -m "not integration" -k "dafny" 2>&1; then
            ok "Dafny tests passed"
        else
            warn "Dafny tests failed — bundle will continue, but verify before releasing."
        fi
        echo ""
    fi
fi

# ─── Build Python wheel + sdist ──────────────────────────────────────────────
info "Building wheel and sdist"
mkdir -p "$DIST"
python3 -m build --wheel --sdist --outdir "$DIST"

WHEEL=$(ls "$DIST"/hpema-"${VERSION}"-*.whl 2>/dev/null | tail -1)
[[ -n "$WHEEL" ]] || die "Wheel not found after build"
ok "Wheel: $WHEEL"

# Twine check — validates PyPI metadata before you try to upload
info "Validating wheel with twine"
python3 -m twine check "$DIST"/hpema-"${VERSION}"-* && ok "twine check passed"

echo ""

if [[ "$WHEEL_ONLY" == "true" ]]; then
    ok "Done (--wheel-only). Artifacts in $DIST/"
    ls -lh "$DIST"/hpema-"${VERSION}"-*
    exit 0
fi

# ─── Platform bundles with Dafny embedded ────────────────────────────────────
info "Building platform bundles (Dafny ${DAFNY_VERSION})"
echo "  Each bundle = wheel + Dafny binary + install scripts"
echo "  These are for users who want a fully offline install."
echo ""

DAFNY_BASE="https://github.com/dafny-lang/dafny/releases/download/v${DAFNY_VERSION}"

declare -A PLATFORM_ASSETS=(
    ["linux-x64"]="dafny-${DAFNY_VERSION}-x64-ubuntu-20.04.zip"
    ["osx-x64"]="dafny-${DAFNY_VERSION}-x64-osx-11.zip"
    ["windows-x64"]="dafny-${DAFNY_VERSION}-x64-win.zip"
)

for PLATFORM in "${!PLATFORM_ASSETS[@]}"; do
    ASSET="${PLATFORM_ASSETS[$PLATFORM]}"
    DAFNY_CACHE="$DIST/dafny-cache-$ASSET"
    BUNDLE_DIR="$DIST/bundle-hpema-${VERSION}-${PLATFORM}"

    info "  Platform: $PLATFORM"

    # Download Dafny if not cached
    if [[ ! -f "$DAFNY_CACHE" ]]; then
        echo "    Downloading Dafny $DAFNY_VERSION for $PLATFORM..."
        if ! curl -fsSL "${DAFNY_BASE}/${ASSET}" -o "$DAFNY_CACHE"; then
            warn "    Failed to download Dafny for $PLATFORM — skipping this bundle."
            continue
        fi
    else
        echo "    Using cached Dafny: $DAFNY_CACHE"
    fi

    mkdir -p "$BUNDLE_DIR/dafny"
    unzip -q "$DAFNY_CACHE" -d "$BUNDLE_DIR/dafny"
    cp "$WHEEL" "$BUNDLE_DIR/"
    cp "$SCRIPT_DIR/install.sh"  "$BUNDLE_DIR/"
    cp "$SCRIPT_DIR/install.ps1" "$BUNDLE_DIR/"

    # Archive
    if [[ "$PLATFORM" == "windows-x64" ]]; then
        OUT="$DIST/hpema-${VERSION}-${PLATFORM}.zip"
        (cd "$DIST" && zip -qr "$OUT" "bundle-hpema-${VERSION}-${PLATFORM}/")
    else
        OUT="$DIST/hpema-${VERSION}-${PLATFORM}.tar.gz"
        tar -czf "$OUT" -C "$DIST" "bundle-hpema-${VERSION}-${PLATFORM}/"
    fi

    rm -rf "$BUNDLE_DIR"
    ok "  $OUT"
done

echo ""
ok "All artifacts ready in $DIST/"
echo ""
ls -lh "$DIST"/hpema-"${VERSION}"-*
echo ""
echo -e "${BOLD}Next step:${NC}  Test the wheel in a clean venv, then run:"
echo "  bash scripts/release.sh $VERSION"
