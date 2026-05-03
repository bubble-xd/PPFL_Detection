from __future__ import annotations

from config import Config
from utils.benchmark import run_all_experiments


def main() -> None:
    output_dir = run_all_experiments(Config)
    print(f"Results saved to: {output_dir}")


if __name__ == "__main__":
    main()
