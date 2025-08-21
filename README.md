# Marketing A/B Testing – Power BI Friendly Pipeline

This project computes **Frequentist** and **Bayesian** A/B test metrics from a marketing dataset
and exports tidy CSVs for **Power BI**. The idea is to host the CSVs on GitHub (free) and connect
Power BI via **Web connector** (Publish to web compatible).

## Project layout
```
ab_marketing_project/
  ├─ data/                 # Put raw CSVs here (e.g., marketing_ab.csv)
  ├─ output/               # Generated CSVs for Power BI
  ├─ src/
  │   ├─ main.py
  │   ├─ ab_pipeline.py
  │   ├─ util_bayes.py
  │   └─ config.py
  ├─ requirements.txt
  └─ .github/workflows/ab_csv.yml   # (optional) GitHub Action to auto-refresh outputs
```

## Setup (local)
1. **Install Python 3.10+** and pip.
2. In a terminal from the project root:
   ```bash
   python -m venv .venv
   .venv\Scripts\activate    # Windows
   # or: source .venv/bin/activate
   pip install -r requirements.txt
   ```
3. Place your dataset CSV in `data/` (e.g., `data/marketing_ab.csv`). Columns expected:
   - `user_id` (or session_id)
   - `variant` (values like `A`/`B`)
   - `converted` (0/1)
   - Optional: `segment` (e.g., device/region), `revenue_7d` (float)

4. Run the pipeline:
   ```bash
   python -m src.main --input data/marketing_ab.csv --output_dir output
   ```

### What you get
- `output/experiments_summary.csv` – per experiment with CR_A/CR_B, abs_lift, z_score, p_value,
  bayesian probability B>A, median_lift, 5–95% credible interval.
- `output/variant_segment.csv` – variant KPIs by segment (for bar charts/heatmaps in Power BI).

## Power BI
- **Get Data → Web**: Use your raw GitHub URLs for the two CSVs above.
- Build native visuals (tables, cards, bar/line). Publish-to-web will work (no Python visual needed).

## Optional: GitHub Actions (auto-refresh)
- The provided workflow `ab_csv.yml` runs the pipeline on a schedule and commits fresh CSVs.
- Ensure the repo allows workflow commits (default GITHUB_TOKEN is enough on non-protected branches).

Enjoy!
