# Run: $env:PYTHONPATH = (Get-Location)
import sys
from pathlib import Path

# Find project root (directory containing this file)
ROOT_DIR = Path(__file__).resolve().parent

# Add project root to sys.path if missing
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))
