"""Convenient launcher for Splendor Duel Round-Robin Tournament Evaluator."""

import sys
from pathlib import Path

# Ensure python/ directory is in sys.path
_python_dir = Path(__file__).resolve().parent
if str(_python_dir) not in sys.path:
    sys.path.insert(0, str(_python_dir))

from tournament import main

if __name__ == "__main__":
    main()
