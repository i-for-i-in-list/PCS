#!/usr/bin/env python3
"""Create this copy's private Python environment and install its runtime dependency."""

import subprocess
import sys
import venv
from pathlib import Path


ROOT = Path(__file__).resolve().parent
ENV = ROOT / ".venv"


def main():
    if sys.version_info < (3, 9):
        print("Python 3.9 or newer is required.", file=sys.stderr)
        return 1
    try:
        print("Preparing environment: {}".format(ENV), flush=True)
        venv.EnvBuilder(with_pip=True).create(ENV)
        python = ENV / ("Scripts/python.exe" if sys.platform == "win32" else "bin/python")
        subprocess.run(
            [str(python), "-m", "pip", "--disable-pip-version-check", "install",
             "--no-cache-dir", "-r", str(ROOT / "requirements.txt")],
            cwd=ROOT, check=True,
        )
        subprocess.run([str(python), "-B", str(ROOT / "start.py"), "--check"],
                       cwd=ROOT, check=True)
    except (OSError, subprocess.CalledProcessError) as exc:
        print("Installation failed: {}".format(exc), file=sys.stderr)
        print("See INSTALL.md for Python, venv and network troubleshooting.", file=sys.stderr)
        return 1
    print("Installed. Run start.command (macOS), start.bat (Windows), or ./start.sh (Linux).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
