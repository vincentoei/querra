"""Print a sample of failure cases for manual error analysis."""

import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--sample", type=int, default=30)
    args = parser.parse_args()

    data = json.loads(Path(args.input).read_text())
    failures = [p for p in data["predictions"] if not p["execution_match"]]

    print(f"Total: {len(data['predictions'])}, Failures: {len(failures)}")
    for i, p in enumerate(failures[: args.sample], 1):
        print(f"\n--- Failure {i} ---")
        print(f"DB: {p['db_id']}")
        print(f"Question: {p['question']}")
        print(f"Gold:    {p['gold']}")
        print(f"Predict: {p['pred']}")
        print(f"Components: {p['component_match']}")


if __name__ == "__main__":
    main()
