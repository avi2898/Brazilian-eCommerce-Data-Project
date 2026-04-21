"""
main.py
-------
Project runner for the Brazilian E-Commerce Analytics project.

Executes all four research questions in order.
Each question saves its output chart(s) to outputs/question_N/.

Usage:
    python main.py
    python main.py --question 2   # run a single question
"""

import argparse
import sys
import time
from pathlib import Path


def _separator(title: str) -> None:
    width = 60
    print("\n" + "=" * width)
    print(f"  {title}")
    print("=" * width)


def run_question(n: int) -> None:
    """Import and execute the main() function of a question module."""
    module_map = {
        1: ("src.question1", "Question 1 — Logistics Efficiency & Residual Analysis"),
        2: ("src.question2", "Question 2 — Sales Trends & Seasonality"),
        3: ("src.question3", "Question 3 — Seller Revenue Bootstrap Simulation"),
        4: ("src.question4", "Question 4 — Payment Methods: Risk vs Value"),
    }

    if n not in module_map:
        print(f"  [ERROR] Unknown question number: {n}")
        return

    module_path, label = module_map[n]
    _separator(label)

    import importlib
    t0 = time.time()
    try:
        mod = importlib.import_module(module_path)
        mod.main()
        elapsed = time.time() - t0
        print(f"\n  ✓ Question {n} completed in {elapsed:.1f}s")
    except Exception as exc:
        print(f"\n  ✗ Question {n} failed: {exc}")
        raise


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run Brazilian E-Commerce Analytics questions."
    )
    parser.add_argument(
        "--question", "-q",
        type=int,
        choices=[1, 2, 3, 4],
        default=None,
        help="Run a single question (1–4). Omit to run all.",
    )
    args = parser.parse_args()

    # Ensure outputs subdirectories exist
    root = Path(__file__).resolve().parent
    for i in range(1, 5):
        (root / "outputs" / f"question_{i}").mkdir(parents=True, exist_ok=True)

    print("\n  Brazilian E-Commerce Analytics Project")
    print("  ======================================")

    questions = [args.question] if args.question else [1, 2, 3, 4]
    t_start = time.time()

    for q in questions:
        run_question(q)

    total = time.time() - t_start
    _separator(f"All done — {len(questions)} question(s) in {total:.1f}s")


if __name__ == "__main__":
    main()
