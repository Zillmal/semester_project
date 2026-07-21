#!/usr/bin/env python3
"""Install all Python packages required by the project."""

from pathlib import Path
import subprocess
import sys


# Locate project root ---------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]
REQUIREMENTS_FILE = PROJECT_ROOT / "requirements.txt"


# Validate requirements file -------------------------------------------------

if not REQUIREMENTS_FILE.exists():
    raise FileNotFoundError(
        f"Requirements file not found: {REQUIREMENTS_FILE}"
    )


# Install packages ------------------------------------------------------------

print(f"Installing packages from: {REQUIREMENTS_FILE}")

subprocess.check_call(
    [
        sys.executable,
        "-m",
        "pip",
        "install",
        "-r",
        str(REQUIREMENTS_FILE),
    ]
)

print("Python package setup completed successfully.")