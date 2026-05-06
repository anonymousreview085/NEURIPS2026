import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import json
import os
import pandas as pd
from dotenv import load_dotenv
load_dotenv()

from config import CSV_PATH, PROMPT_JSON, LANG_HEADER_MAP, LANGUAGES, prompts_dir
from helpers import ensure_dir, make_new_id, write_text, parse_euphemism_annotation, replace_euphemisms


def load_prompt_data() -> dict:
    with open(PROMPT_JSON, encoding="utf-8") as f:
        entries = json.load(f)
    return {
        e["language"].lower(): {
            "desc":            e.get("desc", "").strip(),
            "main":            e.get("main", "").strip(),
            "languageForcing": e.get("languageForcing", "").strip(),
        }
        for e in entries
    }


def build_prompt(lang: str, question_text: str, prompt_data: dict) -> str:
    cfg    = prompt_data[lang]
    header = LANG_HEADER_MAP[lang]
    parts  = [p for p in [cfg["desc"], cfg["main"], cfg["languageForcing"]] if p]
    return "\n\n".join(parts) + f"\n\n{header}:\n{question_text.strip()}"


def run(model_name: str, mode: str = "standard"):
    print(f"===== STEP 2: PROMPT BUILDER | mode={mode} model={model_name} =====")
    prompt_data = load_prompt_data()
    df = pd.read_csv(CSV_PATH, dtype=str)

    if mode == "euphemism":
        df = df[df["euphemismAnnotation"].notna()].copy()
    df = df[df["lang"].isin(LANGUAGES)].copy()

    total = 0
    for lang, grp in df.groupby("lang"):
        if lang not in prompt_data:
            print(f"  [WARN] lang '{lang}' not in prompt.json, skipped")
            continue

        out_dir = prompts_dir(model_name, mode, lang)
        ensure_dir(out_dir)

        for _, row in grp.iterrows():
            uid    = str(row["ID"]).strip()
            new_id = make_new_id(uid)
            q_text = str(row.get("question-annotator", "") or "").strip()
            if not q_text:
                continue

            if mode == "euphemism":
                pairs  = parse_euphemism_annotation(row.get("euphemismAnnotation", ""))
                q_text = replace_euphemisms(q_text, pairs)

            write_text(out_dir / f"{new_id}.txt", build_prompt(lang, q_text, prompt_data))
            total += 1

    print(f"  Written: {total} prompt files")
    print("===== DONE =====")


if __name__ == "__main__":
    model = os.getenv("LLM_MODEL_NAME", "gpt-4o")
    run(model, mode="standard")
    run(model, mode="euphemism")
