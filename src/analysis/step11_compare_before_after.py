import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import os
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
from dotenv import load_dotenv
load_dotenv()

from config import CSV_PATH, BENCHMARK_DIR, LANGUAGES, metric_eval_dir

METRICS     = ["factualPrecision", "hallucinationRate", "factualRecall", "omissionRate"]
RESULTS_DIR = BENCHMARK_DIR / "results"


def _load_euph_ids() -> set:
    df = pd.read_csv(CSV_PATH, dtype=str)
    return set(df.loc[df["euphemismAnnotation"].notna(), "ID"].str.strip())


def _cohen_d_paired(diff: np.ndarray) -> float:
    sd = diff.std(ddof=1)
    return float(diff.mean() / sd) if sd > 0 else 0.0


def run(model_name: str):
    print(f"===== STEP 11: BEFORE vs AFTER REPLACEMENT | model={model_name} =====")
    euph_ids = _load_euph_ids()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    rows = []
    for lang in LANGUAGES:
        before_csv = metric_eval_dir(model_name, "standard",  lang) / "fact-based-annotation-metric-instance.csv"
        after_csv  = metric_eval_dir(model_name, "euphemism", lang) / "fact-based-annotation-metric-instance.csv"

        if not before_csv.exists() or not after_csv.exists():
            print(f"  [SKIP] Missing metric file for {lang}")
            continue

        before_df = pd.read_csv(before_csv)
        after_df  = pd.read_csv(after_csv)

        before_df["_uid"] = before_df["underlying_id"].astype(str).str.strip()
        after_df["_uid"]  = after_df["underlying_id"].astype(str).str.strip()

        before_df = before_df[before_df["_uid"].isin(euph_ids)].set_index("new_id")
        after_df  = after_df.set_index("new_id")
        common    = before_df.index.intersection(after_df.index)
        before_df = before_df.loc[common]
        after_df  = after_df.loc[common]

        for metric in METRICS:
            if metric not in before_df.columns or metric not in after_df.columns:
                continue

            x = before_df[metric].to_numpy(dtype=float)
            y = after_df[metric].reindex(before_df.index).to_numpy(dtype=float)
            mask = ~(np.isnan(x) | np.isnan(y))
            x, y = x[mask], y[mask]
            n    = len(x)

            w_stat = p_val = np.nan
            significant = False
            if n >= 2 and not np.allclose(x, y):
                try:
                    w_stat, p_val = wilcoxon(x, y)
                    significant = bool(p_val < 0.05)
                except Exception:
                    pass

            diff  = y - x
            d     = _cohen_d_paired(diff) if n >= 2 else np.nan
            delta = float(diff.mean()) if n > 0 else np.nan
            mark  = "**" if significant else ""

            rows.append({
                "lang":                     lang,
                "metric":                   metric,
                "n_pairs":                  n,
                "mean_before":              f"{x.mean():.4f}{mark}" if n > 0 else "",
                "std_before":               round(x.std(ddof=1), 4) if n > 1 else "",
                "mean_after":               f"{y.mean():.4f}{mark}" if n > 0 else "",
                "std_after":                round(y.std(ddof=1), 4) if n > 1 else "",
                "delta_after_minus_before": round(delta, 4) if not np.isnan(delta) else "",
                "cohen_d":                  round(d, 4) if not np.isnan(d) else "",
                "w_stat":                   round(float(w_stat), 4) if not np.isnan(w_stat) else "",
                "p_value":                  f"{p_val:.4E}" if not np.isnan(p_val) else "",
                "significant":              mark,
            })

    out = RESULTS_DIR / f"before_after_{model_name}.csv"
    pd.DataFrame(rows).to_csv(out, index=False, encoding="utf-8-sig")
    print(f"  Saved -> {out}")
    print("===== DONE =====")


if __name__ == "__main__":
    model = os.getenv("LLM_MODEL_NAME", "gpt-4o")
    run(model)
