#!/usr/bin/env python3
"""Start the unchanged PCS WebUI from this delivery's local environment."""

import argparse
import os
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8001)
    parser.add_argument("--check", action="store_true", help="Check the environment and configs without starting a server")
    args = parser.parse_args()
    python = ROOT / ".venv" / ("Scripts/python.exe" if sys.platform == "win32" else "bin/python")
    if not python.is_file():
        print("This copy is not installed yet. Run install.command, install.bat, or python3 install.py.", file=sys.stderr)
        return 1
    if args.check:
        code = (
            "import sys, yaml, sqlite3; "
            "from dashboard_v2.server import build_config_payload; "
            "c=build_config_payload(); "
            "print('Python:', sys.version.split()[0]); "
            "print('PyYAML:', yaml.__version__); "
            "print('Active models:', len(c['models'])); "
            "print('Legacy models:', len(c['legacy_models'])); "
            "print('Runtimes:', len(c['runtimes'])); "
            "print('Environment and configuration check passed.')"
        )
        return subprocess.call([str(python), "-B", "-c", code], cwd=ROOT)
    if not 1 <= args.port <= 65535:
        parser.error("--port must be between 1 and 65535")
    os.chdir(ROOT)
    url_host = "127.0.0.1" if args.host == "0.0.0.0" else args.host
    print("Open http://{}:{} in your browser.".format(url_host, args.port), flush=True)
    print("Keep this terminal open. Press Ctrl+C to stop.", flush=True)
    os.execv(str(python), [str(python), "-B", str(ROOT / "dashboard_v2/server.py"),
                          "--host", args.host, "--port", str(args.port)])
    return 0


if __name__ == "__main__":
    sys.exit(main())
