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
from config import DOCTOR_FACTS_DIR, translate_dir, critical_omitted_dir
from helpers import ensure_dir, file_ok, read_lines, write_text, underlying_id
from helpers import strip_numbering, count_numbered_lines, build_openai_client, retry

SYSTEM_MSG = {
    "role": "system",
    "content": (
        "You are a medical information reviewer. "
        "Assess whether each medical fact from the doctor is supported by the LLM output. "
        "Do NOT invent new facts. Only evaluate the provided facts."
    ),
}
USER_TMPL = (
    "INPUT FACTS (from Doctor):\n{doctor_facts}\n\n"
    "OUTPUT (LLM Answer):\n{llm_answer}\n\n"
    "For each numbered fact, answer whether it is CRITICAL INFORMATION but NOT supported by the OUTPUT.\n"
    'Write only "Yes" or "No" per fact:\n1. Yes\n2. No\n\nOUTPUT:'
)

sem = threading.Semaphore(MAX_WORKERS)


def _ask(client, user_prompt: str) -> str:
    def _do():
        with sem:
            r = client.chat.completions.create(
                model=OPENAI_EVAL_MODEL,
                messages=[SYSTEM_MSG, {"role": "user", "content": user_prompt}],
            )
            return r.choices[0].message.content.strip()
    return retry(_do)


def _process(client, new_id: str, llm_dir: Path, out_dir: Path) -> str:
    uid         = underlying_id(new_id)
    llm_path    = llm_dir / f"{new_id}.txt"
    doc_path    = DOCTOR_FACTS_DIR / f"{uid}.txt"
    out_path    = out_dir / f"{new_id}.txt"

    if file_ok(out_path):
        return f"skip:{new_id}"
    if not llm_path.exists():
        return f"missing_llm:{new_id}"
    if not doc_path.exists():
        return f"missing_doctor_facts:{new_id}"

    llm_answer   = llm_path.read_text(encoding="utf-8").strip()
    doctor_lines = read_lines(doc_path)

    if not doctor_lines:
        write_text(out_path, "")
        return f"empty_doctor_facts:{new_id}"
    if not llm_answer:
        write_text(out_path, "")
        return f"empty_llm:{new_id}"

    numbered = "\n".join(f"{i}. {strip_numbering(f)}" for i, f in enumerate(doctor_lines, 1))
    expected  = len(doctor_lines)
    user_p    = USER_TMPL.format(doctor_facts=numbered, llm_answer=llm_answer)

    final = None
    for _ in range(MAX_MISMATCH_RETRY):
        result = _ask(client, user_p)
        if count_numbered_lines(result) == expected:
            final = result
            break
        time.sleep(1.0)
    write_text(out_path, final or result)
    return f"done:{new_id}"


def run(model_name: str, mode: str = "standard"):
    print(f"===== STEP 8: CRITICAL OMITTED | model={model_name} mode={mode} =====")
    client = build_openai_client("OPENAI_API_KEY_TRANSLATE")

    for lang in LANGUAGES:
        llm_dir = translate_dir(model_name, mode, lang)
        out_dir = critical_omitted_dir(model_name, mode, lang)
        if not llm_dir.exists():
            print(f"  [SKIP] No translated output for {lang}")
            continue
        ensure_dir(out_dir)
        new_ids = [fp.stem for fp in sorted(llm_dir.glob("*.txt"))]
        print(f"  [{lang}] {len(new_ids)} files")
        results = []
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
            futs = {ex.submit(_process, client, nid, llm_dir, out_dir): nid for nid in new_ids}
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
