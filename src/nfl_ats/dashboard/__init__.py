"""Human-friendly, local, read-only Streamlit dashboard for research artifacts.

Launch with ``nfl-ats dashboard``. The dashboard never writes to ``data/`` or
``artifacts/``; every page only reads.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

__all__ = ["launch_dashboard"]


def launch_dashboard(address: str = "127.0.0.1", port: int = 8501, headless: bool = False) -> int:
    entry = Path(__file__).resolve().parent / "app.py"
    command = [
        sys.executable,
        "-m",
        "streamlit",
        "run",
        str(entry),
        "--server.address",
        address,
        "--server.port",
        str(port),
        "--server.headless",
        str(headless).lower(),
    ]
    return subprocess.run(command, check=False).returncode
