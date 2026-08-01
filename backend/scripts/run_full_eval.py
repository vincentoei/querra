"""Run the final baseline + best-model pipeline on the full Spider dev set and produce a results table."""

import json
import subprocess
from pathlib import Path

import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import PROCESSED_DEV


RUNS = [
    ("Base zero-shot", ["python", "scripts/run_baseline.py", "--mode", "zero", "--output", "outputs/base_zero.json"]),
    ("Prompt-engineered few-shot", ["python", "scripts/run_baseline.py", "--mode", "few", "--output", "outputs/base_few.json"]),
    ("Best model + Phase 1 pipeline", ["python", "scripts/run_baseline.py", "--adapter_path", "models/best-model", "--mode", "zero", "--self_correct", "--max_retries", "2", "--output", "outputs/qlora_best.json"]),
]


def main():
    table = []
    for name, cmd in RUNS:
        print(f"\n=== Running {name} ===")
        subprocess.run(cmd, cwd=Path(__file__).resolve().parent.parent, check=True)
        output_path = Path(cmd[cmd.index("--output") + 1])
        data = json.loads(output_path.read_text())
        metrics = data["metrics"]
        table.append(
            {
                "model": name,
                "exact_match": f"{metrics['exact_match']:.3f}",
                "execution_accuracy": f"{metrics['execution_accuracy']:.3f}",
                "avg_latency": f"{metrics['avg_latency']:.3f}",
            }
        )

    print("\n=== Results Table ===")
    print(f"{'Model':<35} {'Exact Match':>12} {'Exec Acc':>12} {'Avg Latency':>12}")
    for row in table:
        print(
            f"{row['model']:<35} {row['exact_match']:>12} {row['execution_accuracy']:>12} {row['avg_latency']:>12}"
        )

    with open("outputs/results_table.json", "w", encoding="utf-8") as f:
        json.dump(table, f, indent=2)


if __name__ == "__main__":
    main()
