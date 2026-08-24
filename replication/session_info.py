"""Record the interpreter, platform, and package versions used in replication."""

from __future__ import annotations

import importlib.metadata
import platform
import sys
from pathlib import Path


def session_information() -> str:
    """Return a plain-text description of the active environment."""
    packages = ["stsckm", "numpy", "pandas", "scipy", "scikit-learn", "matplotlib"]
    lines = [
        f"Python: {sys.version.replace(chr(10), ' ')}",
        f"Executable: {sys.executable}",
        f"Platform: {platform.platform()}",
    ]
    lines.extend(
        f"{package}: {importlib.metadata.version(package)}" for package in packages
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    """Write session information to the replication output directory."""
    output = Path(__file__).resolve().parent / "output" / "session_info.txt"
    output.parent.mkdir(parents=True, exist_ok=True)
    information = session_information()
    output.write_text(information, encoding="utf-8")
    print(information, end="")


if __name__ == "__main__":
    main()
