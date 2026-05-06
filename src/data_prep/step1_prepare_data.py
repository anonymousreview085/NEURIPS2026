import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
from config import DATA_DIR, CSV_PATH
from helpers import parse_euphemism_annotation


def create_mapping_dataset(df: pd.DataFrame, out_path: Path):
    euph = df[df["euphemismAnnotation"].notna()].copy()
    out = euph[["ID", "lang", "topic", "topic_english", "euphemismAnnotation"]].reset_index(drop=True)
    out.to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"[OK] euphemismMapping_dataset : {len(out)} rows -> {out_path}")


def create_terms_dataset(df: pd.DataFrame, out_path: Path):
    euph = df[df["euphemismAnnotation"].notna()].copy()
    seen: dict[tuple, dict] = {}

    for _, row in euph.iterrows():
        lang     = str(row["lang"]).strip()
        topic    = str(row["topic"]).strip()
        topic_en = str(row.get("topic_english", "")).strip()

        for term, direct in parse_euphemism_annotation(row["euphemismAnnotation"]):
            key = (lang, term.lower())
            if key not in seen:
                seen[key] = {
                    "lang":          lang,
                    "topic":         topic,
                    "topicEnglish":  topic_en,
                    "EuphemismTerm": term,
                    "DirectTerm":    direct,
                }

    rows = []
    for i, (_, info) in enumerate(sorted(seen.items()), start=1):
        rows.append({"EuphemismID": f"E-{i:05d}", **info})

    out = pd.DataFrame(
        rows,
        columns=["EuphemismID", "lang", "topic", "topicEnglish", "EuphemismTerm", "DirectTerm"],
    )
    out.to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"[OK] euphemismTerms_dataset   : {len(out)} unique terms -> {out_path}")


def main():
    print("===== STEP 1: PREPARE DATA =====")
    df = pd.read_csv(CSV_PATH, dtype=str)
    print(f"  Loaded: {len(df)} rows from {CSV_PATH.name}")
    create_mapping_dataset(df, DATA_DIR / "euphemismMapping_dataset.csv")
    create_terms_dataset(df,   DATA_DIR / "euphemismTerms_dataset.csv")
    print("===== DONE =====")


if __name__ == "__main__":
    main()
