from __future__ import annotations
import pandas as pd
import numpy as np
from dataclasses import dataclass
from math import sqrt
from pathlib import Path
from .config import Config
from .util_bayes import beta_binomial_compare

def _standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    - Lowercase/trim col names
    - Drop any index dump columns like 'Unnamed: 0' or empty header
    - Rename your dataset's headers to pipeline's expected names
    - Normalize values for `converted` (TRUE/FALSE -> 1/0) and `variant` (ad/psa -> A/B)
    - Optionally map 'most ads day' to a generic 'segment' for drilldowns
    """
    df = df.copy()

    # 1) Basic header clean
    df.columns = [str(c).strip().lower() for c in df.columns]

    # 2) Drop auto-index/blank columns (e.g., 'Unnamed: 0', '')
    to_drop = [c for c in df.columns if c == "" or c.startswith("unnamed")]
    if to_drop:
        df = df.drop(columns=to_drop, errors="ignore")

    # 3) Rename to expected schema
    rename_map = {
        # your file -> expected
        "user id": "user_id",
        "test group": "variant",
        # common fallbacks already handled:
        "userid": "user_id",
        "user": "user_id",
        "session": "user_id",
        "variant_group": "variant",
        "group": "variant",
        "converted_flag": "converted",
        "conversion": "converted",
        # optional: use 'most ads day' as the segment for drilldowns
        "most ads day": "segment",
        # keep extra fields like 'total ads', 'most ads hour' if you want later
    }
    df = df.rename(columns=rename_map)

    # 4) Normalize 'converted' to 0/1 (accept TRUE/FALSE/true/false/1/0)
    if "converted" in df.columns:
        df["converted"] = (
            df["converted"]
            .astype(str)
            .str.strip()
            .str.upper()
            .map({"TRUE": 1, "FALSE": 0, "1": 1, "0": 0})
            .fillna(0)
            .astype(int)
        )

    # 5) Normalize 'variant' to A/B (accept 'ad'/'psa', 'a'/'b')
    if "variant" in df.columns:
        df["variant"] = (
            df["variant"]
            .astype(str)
            .str.strip()
            .str.lower()
            .replace({"ad": "A", "psa": "B", "a": "A", "b": "B"})
            .str.upper()
        )

    return df

def _infer_experiment_id(df: pd.DataFrame) -> pd.DataFrame:
    # If there is no experiment column, create a single exp_id
    if "exp_id" not in df.columns:
        df = df.copy()
        df["exp_id"] = "EXP001"
    return df

def _validate(df: pd.DataFrame, cfg: Config):
    missing = [c for c in [cfg.variant_col, cfg.converted_col] if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    # At this point converted should already be 0/1 ints; enforce type
    df[cfg.converted_col] = pd.to_numeric(df[cfg.converted_col], errors="coerce").fillna(0).astype(int)

def ztest_props(sA: int, nA: int, sB: int, nB: int) -> tuple[float | None, float | None]:
    if min(nA, nB) == 0:
        return None, None
    pA, pB = sA / nA, sB / nB
    p = (sA + sB) / (nA + nB)
    denom = p * (1 - p) * (1 / nA + 1 / nB)
    if denom <= 0:
        return None, None
    z = (pB - pA) / sqrt(denom)
    # Two-sided p-value from normal approximation
    try:
        from scipy.stats import norm
        pval = 2 * (1 - norm.cdf(abs(z)))
    except Exception:
        pval = None
    return z, pval

def summarize_experiments(df: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    # Expect columns: exp_id, variant (A/B), converted (0/1), optional revenue_7d
    agg = df.groupby(["exp_id", cfg.variant_col]).agg(
        n=("user_id", "nunique") if "user_id" in df.columns else ("converted", "size"),
        successes=(cfg.converted_col, "sum"),
        rpu=(cfg.revenue_col, "mean") if (cfg.revenue_col and cfg.revenue_col in df.columns) else ("converted", "mean"),
    ).reset_index()

    # Pivot to A/B for z-test
    def compute_row(g: pd.DataFrame) -> pd.Series:
        A = g[g[cfg.variant_col].astype(str).str.upper().str.strip() == "A"]
        B = g[g[cfg.variant_col].astype(str).str.upper().str.strip() == "B"]
        if A.empty or B.empty:
            return pd.Series({"cr_A": None, "cr_B": None, "abs_lift": None, "z_score": None, "p_value": None,
                              "prob_B_wins": None, "median_lift": None, "p05": None, "p95": None})
        sA, nA = int(A["successes"]), int(A["n"])
        sB, nB = int(B["successes"]), int(B["n"])
        crA, crB = sA / nA if nA else None, sB / nB if nB else None
        abs_lift = (crB - crA) if (crA is not None and crB is not None) else None
        z, pval = ztest_props(sA, nA, sB, nB)
        bayes = beta_binomial_compare(sA, nA, sB, nB)
        return pd.Series({
            "cr_A": crA, "cr_B": crB, "abs_lift": abs_lift,
            "z_score": z, "p_value": pval,
            "prob_B_wins": bayes.prob_B_wins,
            "median_lift": bayes.median_lift,
            "p05": bayes.p05, "p95": bayes.p95
        })

    out = agg.groupby("exp_id").apply(compute_row).reset_index()
    return out

def variant_by_segment(df: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    if not cfg.segment_col or cfg.segment_col not in df.columns:
        return pd.DataFrame(columns=["exp_id", "segment", "variant", "n", "cr"])
    g = df.groupby(["exp_id", cfg.segment_col, cfg.variant_col]).agg(
        n=(cfg.converted_col, "size"),
        cr=(cfg.converted_col, "mean")
    ).reset_index()
    g = g.rename(columns={cfg.segment_col: "segment"})
    return g

def bayes_by_segment(df: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """
    Input df: columns [exp_id, segment, variant, n, cr] (from variant_by_segment)
    Output: one row per (exp_id, segment) with Bayesian lift CI and win prob for B vs A.
    """
    import numpy as np
    rows = []
    for (exp, seg), g in df.groupby(["exp_id", "segment"]):
        # ensure we have exactly one row for A and one for B
        A = g[g[cfg.variant_col].astype(str).str.upper().eq("A")]
        B = g[g[cfg.variant_col].astype(str).str.upper().eq("B")]
        if A.empty or B.empty:
            continue

        # pull scalars
        nA = int(A["n"].iloc[0])
        nB = int(B["n"].iloc[0])
        crA = float(A["cr"].iloc[0])
        crB = float(B["cr"].iloc[0])

        # derive successes from n * cr
        sA = int(round(crA * nA))
        sB = int(round(crB * nB))

        # Beta(1,1) posteriors
        aA, bA = 1 + sA, 1 + (nA - sA)
        aB, bB = 1 + sB, 1 + (nB - sB)

        draws = 100_000
        SA = np.random.beta(aA, bA, draws)
        SB = np.random.beta(aB, bB, draws)

        prob = float((SB > SA).mean())
        lift = (SB - SA) / SA  # relative lift

        rows.append({
            "exp_id": exp,
            "segment": seg,
            "prob_B_wins": prob,
            "median_lift": float(np.median(lift)),
            "p05": float(np.percentile(lift, 5)),
            "p95": float(np.percentile(lift, 95))
        })
    return pd.DataFrame(rows)

def run_pipeline(cfg: Config):
    # read_csv will bring the blank index as 'Unnamed: 0' if present; we drop it in _standardize_columns
    df = pd.read_csv(cfg.input_path)
    df = _standardize_columns(df)
    df = _infer_experiment_id(df)
    _validate(df, cfg)

    cfg.output_dir.mkdir(parents=True, exist_ok=True)

    summary = summarize_experiments(df, cfg)
    seg = variant_by_segment(df, cfg)
    bayes_seg = bayes_by_segment(seg, cfg)
    (bayes_seg.sort_values(["exp_id", "segment"])
        .to_csv(cfg.output_dir / "bayes_by_segment.csv", index=False))

    summary.to_csv(cfg.summary_csv, index=False)
    seg.to_csv(cfg.variant_segment_csv, index=False)
    
    return cfg.summary_csv, cfg.variant_segment_csv