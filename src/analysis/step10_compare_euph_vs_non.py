import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import os
import numpy as np
import pandas as pd
from scipy.stats import ttest_ind
from dotenv import load_dotenv
load_dotenv()

from config import CSV_PATH, BENCHMARK_DIR, LANGUAGES, metric_eval_dir

METRICS     = ["factualPrecision", "hallucinationRate", "factualRecall", "omissionRate"]
RESULTS_DIR = BENCHMARK_DIR / "results"


def _load_euph_ids() -> set:
    df = pd.read_csv(CSV_PATH, dtype=str)
    return set(df.loc[df["euphemismAnnotation"].notna(), "ID"].str.strip())


def _cohen_d(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = len(a), len(b)
    if na < 2 or nb < 2:
        return np.nan
    pooled = ((na - 1) * a.var(ddof=1) + (nb - 1) * b.var(ddof=1)) / (na + nb - 2)
    sd = np.sqrt(max(pooled, 0))
    return float((b.mean() - a.mean()) / sd) if sd > 0 else 0.0


def run(model_name: str):
    print(f"===== STEP 10: EUPHEMISTIC vs NON-EUPHEMISTIC | model={model_name} =====")
    euph_ids = _load_euph_ids()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    rows = []
    for lang in LANGUAGES:
        csv_path = metric_eval_dir(model_name, "standard", lang) / "fact-based-annotation-metric-instance.csv"
        if not csv_path.exists():
            print(f"  [SKIP] No metric file for {lang}")
            continue

        df = pd.read_csv(csv_path)
        df["_uid"] = df["underlying_id"].astype(str).str.strip()
        euph_df = df[df["_uid"].isin(euph_ids)]
        non_df  = df[~df["_uid"].isin(euph_ids)]

        for metric in METRICS:
            a = euph_df[metric].dropna().to_numpy() if metric in euph_df else np.array([])
            b = non_df[metric].dropna().to_numpy()  if metric in non_df  else np.array([])

            t_stat = p_val = np.nan
            significant = False
            if len(a) >= 2 and len(b) >= 2:
                t_stat, p_val = ttest_ind(a, b, equal_var=False)
                significant = bool(p_val < 0.05)

            d     = _cohen_d(a, b)
            delta = float(b.mean() - a.mean()) if len(a) > 0 and len(b) > 0 else np.nan
            mark  = "**" if significant else ""

            rows.append({
                "lang":                 lang,
                "metric":               metric,
                "n_euph":               len(a),
                "mean_euph":            f"{a.mean():.4f}{mark}" if len(a) > 0 else "",
                "std_euph":             round(a.std(ddof=1), 4) if len(a) > 1 else "",
                "n_non":                len(b),
                "mean_non":             f"{b.mean():.4f}{mark}" if len(b) > 0 else "",
                "std_non":              round(b.std(ddof=1), 4) if len(b) > 1 else "",
                "delta_non_minus_euph": round(delta, 4) if not np.isnan(delta) else "",
                "cohen_d":              round(d, 4) if not np.isnan(d) else "",
                "t_stat":               round(float(t_stat), 4) if not np.isnan(t_stat) else "",
                "p_value":              f"{p_val:.4E}" if not np.isnan(p_val) else "",
                "significant":          mark,
            })

    out = RESULTS_DIR / f"euph_vs_non_{model_name}.csv"
    pd.DataFrame(rows).to_csv(out, index=False, encoding="utf-8-sig")
    print(f"  Saved -> {out}")
    print("===== DONE =====")


if __name__ == "__main__":
    model = os.getenv("LLM_MODEL_NAME", "gpt-4o")
    run(model)
