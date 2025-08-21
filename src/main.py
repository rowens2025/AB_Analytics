import argparse
from pathlib import Path
from .config import Config
from .ab_pipeline import run_pipeline

def parse_args():
    p = argparse.ArgumentParser(description="Marketing A/B pipeline → CSVs for Power BI")
    p.add_argument("--input", required=True, help="Path to input CSV (e.g., data/marketing_ab.csv)")
    p.add_argument("--output_dir", default="output", help="Directory for CSV outputs")
    return p.parse_args()

def main():
    args = parse_args()
    cfg = Config(input_path=Path(args.input), output_dir=Path(args.output_dir))
    s, v = run_pipeline(cfg)
    print(f"Wrote: {s}\nWrote: {v}")

if __name__ == "__main__":
    main()
