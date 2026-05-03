from __future__ import annotations

from .pipeline import run_layer_extraction


def main() -> None:
    result = run_layer_extraction()
    if "runs" in result:
        for run_result in result["runs"]:
            print(
                "Layer extraction results saved to: "
                f"{run_result['model']} -> {run_result['output_dir']}"
            )
        return
    print(f"Layer extraction results saved to: {result['output_dir']}")


if __name__ == "__main__":
    main()
