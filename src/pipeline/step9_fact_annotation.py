import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import os
import pandas as pd
from dotenv import load_dotenv
load_dotenv()

from config import LANGUAGES, DOCTOR_FACTS_DIR
from config import common_facts_dir, hallucinated_dir, critical_omitted_dir, metric_eval_dir
from helpers import ensure_dir, read_lines, underlying_id, normalize_yn


def _count_yn(path: Path, target: str) -> int:
    return sum(1 for ln in read_lines(path) if normalize_yn(ln) == target)


def _safe_div(a, b) -> float:
    return float(a) / float(b) if b else 0.0


def run(model_name: str, mode: str = "standard"):
    print(f"===== STEP 9: FACT ANNOTATION | model={model_name} mode={mode} =====")

    for lang in LANGUAGES:
        c_dir   = common_facts_dir(model_name, mode, lang)
        h_dir   = hallucinated_dir(model_name, mode, lang)
        cr_dir  = critical_omitted_dir(model_name, mode, lang)
        out_dir = metric_eval_dir(model_name, mode, lang)
        ensure_dir(out_dir)

        all_ids: set[str] = set()
        for d in [c_dir, h_dir, cr_dir]:
            if d.exists():
                all_ids |= {fp.stem for fp in d.glob("*.txt")}

        rows = []
        for new_id in sorted(all_ids):
            uid = underlying_id(new_id)

            yes_c = _count_yn(c_dir  / f"{new_id}.txt", "yes") if c_dir.exists()  else 0
            no_c  = _count_yn(c_dir  / f"{new_id}.txt", "no")  if c_dir.exists()  else 0
            hall  = _count_yn(h_dir  / f"{new_id}.txt", "yes") if h_dir.exists()  else 0
            crit  = _count_yn(cr_dir / f"{new_id}.txt", "no")  if cr_dir.exists() else 0

            doc_path = DOCTOR_FACTS_DIR / f"{uid}.txt"
            ref_f = len([ln for ln in read_lines(doc_path) if ln]) if doc_path.exists() else 0
            sys_f = yes_c + no_c

            rows.append({
                "language":              lang,
                "model":                 model_name,
                "mode":                  mode,
                "new_id":                new_id,
                "underlying_id":         uid,
                "yesCommonFacts_count":  yes_c,
                "noCommonFacts_count":   no_c,
                "hallucinated_count":    hall,
                "criticalOmitted_count": crit,
                "referenceFacts":        ref_f,
                "systemFacts":           sys_f,
                "factualPrecision":      _safe_div(yes_c, sys_f),
                "hallucinationRate":     _safe_div(hall, sys_f),
                "factualRecall":         _safe_div(yes_c, ref_f),
                "omissionRate":          _safe_div(crit, ref_f),
            })

        df = pd.DataFrame(rows)
        out_csv = out_dir / "fact-based-annotation-metric-instance.csv"
        df.to_csv(out_csv, index=False, encoding="utf-8")
        print(f"  [{lang}] {len(df)} instances -> {out_csv.name}")

    print("===== DONE =====")


if __name__ == "__main__":
    model = os.getenv("LLM_MODEL_NAME", "gpt-4o")
    run(model, mode="standard")
    run(model, mode="euphemism")
