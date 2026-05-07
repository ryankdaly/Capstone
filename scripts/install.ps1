# =============================================================================
# install.ps1 — End-user install from a HPEMA Windows bundle
# =============================================================================
# Run from the extracted bundle directory (PowerShell):
#   Expand-Archive hpema-VERSION-windows-x64.zip .
#   cd hpema-VERSION-windows-x64
#   .\install.ps1
# =============================================================================
$ErrorActionPreference = "Stop"

$BundleDir = Split-Path -Parent $MyInvocation.MyCommand.Definition
$HpemaHome = if ($env:HPEMA_HOME) { $env:HPEMA_HOME } else { "$HOME\.hpema" }

Write-Host ""
Write-Host "HPEMA Installer" -ForegroundColor White -BackgroundColor DarkBlue
Write-Host "─────────────────────────────────────────"
Write-Host ""

# ─── Python check ────────────────────────────────────────────────────────────
Write-Host "▶  Checking Python..." -ForegroundColor Cyan
try {
    $pyVer = & python --version 2>&1
    Write-Host "   Found: $pyVer"
} catch {
    Write-Host "✗  python not found. Install Python 3.11+ from https://python.org" -ForegroundColor Red
    exit 1
}

# ─── Install wheel ───────────────────────────────────────────────────────────
Write-Host "▶  Installing HPEMA..." -ForegroundColor Cyan
$wheel = Get-ChildItem "$BundleDir\hpema-*.whl" -ErrorAction SilentlyContinue | Select-Object -First 1
if (-not $wheel) {
    Write-Host "✗  No wheel found in bundle directory." -ForegroundColor Red
    exit 1
}
& pip install --quiet $wheel.FullName
Write-Host "✓  HPEMA installed" -ForegroundColor Green

# ─── Install Dafny ───────────────────────────────────────────────────────────
$dafnyBundle = "$BundleDir\dafny"
if (Test-Path $dafnyBundle) {
    Write-Host "▶  Installing Dafny..." -ForegroundColor Cyan
    $DafnyDest = "$HpemaHome\dafny"
    New-Item -ItemType Directory -Force -Path $DafnyDest | Out-Null
    Copy-Item -Recurse -Force "$dafnyBundle\*" $DafnyDest

    $dafnyBin = Get-ChildItem -Recurse "$DafnyDest" -Filter "dafny.exe" -ErrorAction SilentlyContinue |
                Select-Object -First 1
    if ($dafnyBin) {
        New-Item -ItemType Directory -Force -Path $HpemaHome | Out-Null
        $dafnyBin.FullName | Out-File -FilePath "$HpemaHome\dafny_path" -Encoding ascii
        Write-Host "✓  Dafny installed: $($dafnyBin.FullName)" -ForegroundColor Green
    } else {
        Write-Host "⚠  Dafny binary not found in bundle." -ForegroundColor Yellow
    }
} else {
    Write-Host "⚠  No Dafny bundle — run 'hpema /setup' to configure." -ForegroundColor Yellow
}

Write-Host ""
Write-Host "✓  Done." -ForegroundColor Green
Write-Host ""
Write-Host "  Run:   hpema"
Write-Host "  Then:  /setup   to configure your API key"
Write-Host ""
Write-Host "  Config: $HpemaHome\hpema_config.yaml"
Write-Host ""
