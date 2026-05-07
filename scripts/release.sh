#!/usr/bin/env bash
# =============================================================================
# scripts/release.sh — Tag, upload to PyPI, and create a GitHub Release
# =============================================================================
#
# Usage:
#   bash scripts/release.sh 0.2.0
#
# Prerequisites:
#   pip install twine
#   gh auth login                        (GitHub CLI — https://cli.github.com)
#   export PYPI_TOKEN=pypi-...           (from pypi.org/manage/account/token/)
#   dist/ must already contain artifacts built by bundle.sh
#
# Steps performed:
#   1. Sanity checks (version match, tag doesn't exist, dist/ has files)
#   2. twine check (validates metadata before upload attempt)
#   3. git tag + push
#   4. twine upload to PyPI
#   5. gh release create with all dist/ artifacts attached
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
DIST="$ROOT/dist"

VERSION="${1:-}"
if [[ -z "$VERSION" ]]; then
    echo "Usage: bash scripts/release.sh VERSION" >&2
    echo "Example: bash scripts/release.sh 0.1.0" >&2
    exit 1
fi

RED='\033[0;31m'; YELLOW='\033[1;33m'; GREEN='\033[0;32m'; BOLD='\033[1m'; NC='\033[0m'
info()  { echo -e "${BOLD}▶  $*${NC}"; }
ok()    { echo -e "${GREEN}✓  $*${NC}"; }
warn()  { echo -e "${YELLOW}⚠  $*${NC}"; }
die()   { echo -e "${RED}✗  $*${NC}" >&2; exit 1; }

cd "$ROOT"

# ─── Sanity checks ───────────────────────────────────────────────────────────
info "Sanity checks for v${VERSION}"

# Version matches pyproject.toml
PYPROJECT_VERSION=$(python3 -c "
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
")
if [[ "$PYPROJECT_VERSION" != "$VERSION" ]]; then
    die "pyproject.toml version is '$PYPROJECT_VERSION', expected '$VERSION'.\nUpdate [project].version and re-run bundle.sh first."
fi
ok "pyproject.toml version matches"

# Wheel exists for this version
WHEEL=$(ls "$DIST"/hpema-"${VERSION}"-*.whl 2>/dev/null | head -1)
[[ -n "$WHEEL" ]] || die "No wheel found in $DIST/ for version $VERSION. Run bundle.sh first."
ok "Wheel found: $WHEEL"

# Git tag doesn't already exist
if git tag | grep -qx "v${VERSION}"; then
    die "Tag v${VERSION} already exists. Delete it first: git tag -d v${VERSION}"
fi
ok "Tag v${VERSION} is available"

# PYPI_TOKEN is set
[[ -n "${PYPI_TOKEN:-}" ]] || die "PYPI_TOKEN env var not set. Export it before running release.sh."
ok "PYPI_TOKEN set"

# gh CLI available
command -v gh &>/dev/null || die "'gh' CLI not found. Install: https://cli.github.com"
gh auth status &>/dev/null || die "Not authenticated with gh. Run: gh auth login"
ok "GitHub CLI authenticated"

# twine available
python3 -m twine --version &>/dev/null || die "'twine' not installed. Run: pip install twine"

# Clean working tree
if ! git diff --quiet HEAD 2>/dev/null; then
    warn "Uncommitted changes in working tree. Continuing anyway."
fi

echo ""

# ─── twine check ─────────────────────────────────────────────────────────────
info "Validating distribution files"
python3 -m twine check "$DIST"/hpema-"${VERSION}"-*
ok "twine check passed"
echo ""

# ─── Git tag ─────────────────────────────────────────────────────────────────
info "Tagging v${VERSION}"
git tag -a "v${VERSION}" -m "Release v${VERSION}"
git push origin "v${VERSION}"
ok "Pushed tag v${VERSION}"
echo ""

# ─── Upload to PyPI ───────────────────────────────────────────────────────────
info "Uploading to PyPI"
python3 -m twine upload \
    --username "__token__" \
    --password "$PYPI_TOKEN" \
    "$DIST"/hpema-"${VERSION}"-*.whl \
    "$DIST"/hpema-"${VERSION}".tar.gz
ok "Uploaded to PyPI — https://pypi.org/project/hpema/${VERSION}/"
echo ""

# ─── GitHub Release ──────────────────────────────────────────────────────────
info "Creating GitHub Release v${VERSION}"

# Collect all artifacts for this version
ASSETS=()
while IFS= read -r f; do
    ASSETS+=("$f")
done < <(ls "$DIST"/hpema-"${VERSION}"-*.{tar.gz,zip,whl} "$DIST"/hpema-"${VERSION}".tar.gz 2>/dev/null | sort -u)

gh release create "v${VERSION}" \
    --title "HPEMA v${VERSION}" \
    --notes "$(cat <<EOF
## Install

\`\`\`bash
pip install hpema==${VERSION}
\`\`\`

Then run \`hpema\` and type \`/setup\` to configure your API key and Dafny path.

---

## Platform bundles (Dafny included — no internet required after download)

| Platform | Download |
|----------|----------|
| Linux x64 | \`hpema-${VERSION}-linux-x64.tar.gz\` |
| macOS x64  | \`hpema-${VERSION}-osx-x64.tar.gz\`   |
| Windows x64 | \`hpema-${VERSION}-windows-x64.zip\` |

Extract and run \`install.sh\` (Linux/macOS) or \`install.ps1\` (Windows).

---

See [docs/LOCAL_SETUP.md](docs/LOCAL_SETUP.md) for full setup instructions.
EOF
)" \
    "${ASSETS[@]}"

RELEASE_URL=$(gh release view "v${VERSION}" --json url -q .url)
ok "GitHub Release: $RELEASE_URL"
echo ""

ok "Release v${VERSION} complete."
echo ""
echo "  PyPI   : https://pypi.org/project/hpema/${VERSION}/"
echo "  GitHub : $RELEASE_URL"
echo ""
echo "  Verify: pip install hpema==${VERSION} && hpema --version"
