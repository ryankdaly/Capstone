#!/usr/bin/env python3
"""Cross-platform HPEMA dev-install script.

Works on macOS, Linux, and Windows (no bash required).

Usage
-----
  python scripts/dev_install.py              # build wheel + reinstall
  python scripts/dev_install.py --editable   # editable install (fastest dev loop)
  python scripts/dev_install.py --clean      # uninstall only
  python scripts/dev_install.py --fresh      # uninstall + wipe config (simulate new user)

Modes
-----
  (no flag)    Build a wheel from current source, uninstall any existing hpema,
               install the new wheel. Reflects exactly what end users receive.
               Use this to test packaging, bundled data files, entry point behaviour.

  --editable   pip install -e . — one install, code changes are live immediately,
               no rebuild needed between iterations. Best for day-to-day development.

  --clean      Uninstall hpema. Does not touch config.

  --fresh      Uninstall + delete config/verified marker from the HPEMA home dir
               so the next `hpema` launch behaves exactly like a brand-new install.
               DOES NOT delete your API keys from the .env file.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# ── Colours (disabled on Windows unless ANSICON/WT_SESSION present) ──────────
_NO_COLOR = sys.platform == "win32" and not (
    os.environ.get("ANSICON") or os.environ.get("WT_SESSION") or os.environ.get("COLORTERM")
)

def _c(code: str, text: str) -> str:
    return text if _NO_COLOR else f"\033[{code}m{text}\033[0m"

def info(msg: str)  -> None: print(_c("1",      f"  ▶  {msg}"))
def ok(msg: str)    -> None: print(_c("32",     f"  ✓  {msg}"))
def warn(msg: str)  -> None: print(_c("33",     f"  ⚠  {msg}"))
def die(msg: str)   -> None: print(_c("31",     f"  ✗  {msg}"), file=sys.stderr); sys.exit(1)


# ── Pip command (prefer uv pip for speed, fall back to python -m pip) ────────
def _pip() -> list[str]:
    if shutil.which("uv"):
        return ["uv", "pip"]
    return [sys.executable, "-m", "pip"]


def _run(cmd: list[str], **kwargs) -> None:
    print(f"      $ {' '.join(str(c) for c in cmd)}")
    r = subprocess.run(cmd, **kwargs)
    if r.returncode != 0:
        sys.exit(r.returncode)


def _uninstall() -> None:
    subprocess.run([*_pip(), "uninstall", "hpema", "-y"],
                   capture_output=True)  # silence "not installed" noise


def _hpema_home() -> Path:
    """Mirror of backend.config.hpema_home() without importing the backend."""
    override = os.environ.get("HPEMA_HOME")
    if override:
        return Path(override)
    if sys.platform == "win32":
        appdata = os.environ.get("APPDATA")
        return Path(appdata) / "hpema" if appdata else Path.home() / "AppData" / "Roaming" / "hpema"
    return Path.home() / ".hpema"


# ── Actions ──────────────────────────────────────────────────────────────────

def clean() -> None:
    info("Uninstalling hpema...")
    _uninstall()
    ok("Uninstalled (config untouched)")


def fresh() -> None:
    """Uninstall + remove config so next launch looks like a new device."""
    home = _hpema_home()
    info("Uninstalling hpema...")
    _uninstall()

    targets = [
        home / "hpema_config.yaml",
        home / ".config_verified",
    ]
    removed = [t for t in targets if t.exists()]
    for t in removed:
        t.unlink()

    if removed:
        ok(f"Removed from {home}:")
        for t in removed:
            print(f"       {t.name}")
    else:
        ok(f"No config found in {home} — already clean")

    print()
    print("  Next `hpema` launch will auto-pop the setup wizard.")


def editable() -> None:
    info("Editable install (code changes are live immediately)...")
    _run([*_pip(), "install", "-e", ".[dev]"], cwd=ROOT)
    ok("hpema installed in editable mode")
    print()
    print(f"  Config dir : {_hpema_home()}")
    print("  Run        : hpema")
    print("  To remove  : python scripts/dev_install.py --clean")


def wheel_install() -> None:
    _uninstall()

    info("Building wheel from current source...")
    _run(
        [sys.executable, "-m", "build", "--wheel", "--outdir", "dist", "-q"],
        cwd=ROOT,
    )

    wheels = sorted(ROOT.glob("dist/hpema-*.whl"))
    if not wheels:
        die("No wheel found after build. Is 'build' installed? pip install build")
    wheel = wheels[-1]
    ok(f"Built: {wheel.name}")

    info("Installing...")
    _run([*_pip(), "install", str(wheel)])
    ok("hpema installed")
    print()
    print(f"  Config dir : {_hpema_home()}")
    print("  Run        : hpema")
    print()
    print("  Simulate fresh user  : python scripts/dev_install.py --fresh")
    print("  Rebuild after change : python scripts/dev_install.py")
    print("  Uninstall            : python scripts/dev_install.py --clean")


# ── Entry point ──────────────────────────────────────────────────────────────

def main() -> None:
    args = set(sys.argv[1:])
    if "--help" in args or "-h" in args:
        print(__doc__)
        return
    unknown = args - {"--clean", "--fresh", "--editable"}
    if unknown:
        die(f"Unknown flag(s): {unknown}\nUsage: python scripts/dev_install.py [--editable|--clean|--fresh]")

    # Require build tool for wheel mode
    if not args and shutil.which("python") or True:
        try:
            subprocess.run(
                [sys.executable, "-m", "build", "--version"],
                capture_output=True, check=True,
            )
        except subprocess.CalledProcessError:
            if "--editable" not in args and "--clean" not in args and "--fresh" not in args:
                die("'build' not installed. Run: pip install build")

    print()
    if "--clean" in args:
        clean()
    elif "--fresh" in args:
        fresh()
    elif "--editable" in args:
        editable()
    else:
        wheel_install()
    print()


if __name__ == "__main__":
    main()
