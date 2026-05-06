import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import os
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from dotenv import load_dotenv
load_dotenv()

from config import LANGUAGES, MAX_WORKERS, OPENAI_EVAL_MODEL, translate_dir, llm_facts_dir
from helpers import ensure_dir, file_ok, read_text, write_text, build_openai_client, retry, normalize_fact_lines

SYSTEM_MSG = {
    "role": "system",
    "content": (
        "You are tasked with extracting medical facts in S P O format (Subject Predicate Object). "
        "A fact is information expressible in a single sentence. "
        "Extract as many medical facts as possible. "
        "Output MUST be entirely in Indonesian. "
        "Format: numbered list, each line exactly 'S P O'. "
        "If no facts exist, output nothing."
    ),
}
USER_TMPL = "Extract all medical facts in S P O format from the INPUT below:\n\nINPUT\n{text}\n\nOutput only a numbered list following 'S P O', no extra text."


def _extract_one(client, sem: threading.Semaphore, in_path: Path, out_path: Path) -> str:
    fname = in_path.stem
    if file_ok(out_path):
        return f"skip:{fname}"
    try:
        text = read_text(in_path)
        if not text:
            write_text(out_path, "")
            return f"empty:{fname}"

        def _do():
            with sem:
                r = client.chat.completions.create(
                    model=OPENAI_EVAL_MODEL,
                    messages=[SYSTEM_MSG, {"role": "user", "content": USER_TMPL.format(text=text)}],
                )
                return r.choices[0].message.content.strip()

        raw   = retry(_do)
        facts = normalize_fact_lines(raw)
        write_text(out_path, "\n".join(facts))
        return f"done:{fname}"
    except Exception as e:
        return f"error:{fname}:{e}"


def run(model_name: str, mode: str = "standard"):
    print(f"===== STEP 5: EXTRACT LLM FACTS | model={model_name} mode={mode} =====")
    client = build_openai_client("OPENAI_API_KEY_TRANSLATE")
    sem    = threading.Semaphore(MAX_WORKERS)

    for lang in LANGUAGES:
        in_dir  = translate_dir(model_name, mode, lang)
        out_dir = llm_facts_dir(model_name, mode, lang)
        if not in_dir.exists():
            print(f"  [SKIP] No translated output for {lang}")
            continue
        ensure_dir(out_dir)
        files   = sorted(in_dir.glob("*.txt"))
        print(f"  [{lang}] {len(files)} files")
        results = []
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
            futs = {ex.submit(_extract_one, client, sem, fp, out_dir / fp.name): fp for fp in files}
            for fut in as_completed(futs):
                results.append(fut.result())
        done = sum(1 for r in results if r.startswith("done"))
        skip = sum(1 for r in results if r.startswith("skip"))
        err  = sum(1 for r in results if r.startswith("error"))
        print(f"    done={done}  skip={skip}  error={err}")

    print("===== DONE =====")


if __name__ == "__main__":
    model = os.getenv("LLM_MODEL_NAME", "gpt-4o")
    run(model, mode="standard")
    run(model, mode="euphemism")
