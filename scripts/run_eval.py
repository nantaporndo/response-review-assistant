"""รัน evaluation กับ test set
ใช้: uv run python scripts/run_eval.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.evaluate import run

if __name__ == "__main__":
    run()