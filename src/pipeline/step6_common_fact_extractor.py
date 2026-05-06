import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import os
import time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from dotenv import load_dotenv
load_dotenv()

from config import LANGUAGES, MAX_WORKERS, MAX_MISMATCH_RETRY, OPENAI_EVAL_MODEL
from config import DOCTOR_ANSWERS_DIR, llm_facts_dir, common_facts_dir
from helpers import ensure_dir, file_ok, read_lines, write_text, underlying_id
from helpers import strip_numbering, count_numbered_lines, build_openai_client, retry

sem = threading.Semaphore(MAX_WORKERS)


def _ask(client, prompt: str) -> str:
    def _do():
        with sem:
            r = client.chat.completions.create(
                model=OPENAI_EVAL_MODEL,
                messages=[{"role": "user", "content": prompt}],
            )
            return r.choices[0].message.content.strip()
    return retry(_do)


def _process(client, lang: str, new_id: str, facts_dir: Path, out_dir: Path) -> str:
    fact_path   = facts_dir / f"{new_id}.txt"
    out_path    = out_dir   / f"{new_id}.txt"
    uid         = underlying_id(new_id)
    doctor_path = DOCTOR_ANSWERS_DIR / f"{uid}.txt"

    if file_ok(out_path):
        return f"skip:{new_id}"
    if not fact_path.exists():
        return f"missing_facts:{new_id}"
    if not doctor_path.exists():
        return f"missing_doctor:{new_id}"

    fact_lines  = read_lines(fact_path)
    doctor_text = doctor_path.read_text(encoding="utf-8").strip()

    if not fact_lines:
        write_text(out_path, "")
        return f"empty_facts:{new_id}"
    if not doctor_text:
        write_text(out_path, "")
        return f"empty_doctor:{new_id}"

    expected = len(fact_lines)
    parts = [f"Consider the INPUT below:\n\nINPUT: {doctor_text}\n"]
    for i, f in enumerate(fact_lines, 1):
        parts.append(f'{i}. Is the fact "{strip_numbering(f)}" stated or supported by the INPUT?')
    parts.append('\nWrite only "Yes" or "No" per fact:\n1. Yes\n2. No\n\nOUTPUT:')
    prompt = "\n".join(parts)

    final = None
    for _ in range(MAX_MISMATCH_RETRY):
        result = _ask(client, prompt)
        if count_numbered_lines(result) == expected:
            final = result
            break
        time.sleep(1.0)
    write_text(out_path, final or result)
    return f"done:{new_id}"


def run(model_name: str, mode: str = "standard"):
    print(f"===== STEP 6: COMMON FACT EXTRACTOR | model={model_name} mode={mode} =====")
    client = build_openai_client("OPENAI_API_KEY_TRANSLATE")

    for lang in LANGUAGES:
        facts_dir = llm_facts_dir(model_name, mode, lang)
        out_dir   = common_facts_dir(model_name, mode, lang)
        if not facts_dir.exists():
            print(f"  [SKIP] No llm-facts for {lang}")
            continue
        ensure_dir(out_dir)
        new_ids = [fp.stem for fp in sorted(facts_dir.glob("*.txt"))]
        print(f"  [{lang}] {len(new_ids)} files")
        results = []
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
            futs = {ex.submit(_process, client, lang, nid, facts_dir, out_dir): nid
                    for nid in new_ids}
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
