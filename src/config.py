from dataclasses import dataclass
from pathlib import Path

@dataclass
class Config:
    input_path: Path
    output_dir: Path
    experiment_col: str = "exp_id"        # Optional; will be derived if missing
    variant_col: str = "variant"
    converted_col: str = "converted"
    segment_col: str | None = "segment"   # Set to None if no segment
    revenue_col: str | None = "revenue_7d" # Optional

    @property
    def summary_csv(self) -> Path:
        return self.output_dir / "experiments_summary.csv"

    @property
    def variant_segment_csv(self) -> Path:
        return self.output_dir / "variant_segment.csv"
