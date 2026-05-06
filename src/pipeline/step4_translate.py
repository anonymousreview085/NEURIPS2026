import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import os
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from dotenv import load_dotenv
load_dotenv()

from config import LANGUAGES, TRANSLATE_LANGS, MAX_WORKERS, OPENAI_TRANSLATE_MODEL
from config import inference_dir, translate_dir
from helpers import ensure_dir, file_ok, read_text, write_text, build_openai_client, openai_chat

SYSTEM_MSG = {
    "role": "system",
    "content": (
        "You are a professional translator. "
        "Translate the following text into natural, fluent Indonesian. "
        "Preserve meaning exactly. Do not add explanations."
    ),
}


def _translate_one(client, sem: threading.Semaphore, model: str,
                   in_path: Path, out_path: Path) -> str:
    fname = in_path.stem
    if file_ok(out_path):
        return f"skip:{fname}"
    try:
        text = read_text(in_path)
        if not text:
            return f"empty:{fname}"
        translated = openai_chat(client, model, [SYSTEM_MSG, {"role": "user", "content": text}], sem)
        write_text(out_path, translated)
        return f"done:{fname}"
    except Exception as e:
        return f"error:{fname}:{e}"


def run(model_name: str, mode: str = "standard"):
    print(f"===== STEP 4: TRANSLATE | model={model_name} mode={mode} =====")
    client = build_openai_client("OPENAI_API_KEY_TRANSLATE")
    sem    = threading.Semaphore(MAX_WORKERS)

    for lang in LANGUAGES:
        in_dir  = inference_dir(model_name, mode, lang)
        out_dir = translate_dir(model_name, mode, lang)
        if not in_dir.exists():
            print(f"  [SKIP] No inference output for {lang}")
            continue
        ensure_dir(out_dir)
        files = sorted(in_dir.glob("*.txt"))

        if lang not in TRANSLATE_LANGS:
            for fp in files:
                out = out_dir / fp.name
                if not file_ok(out):
                    write_text(out, read_text(fp))
            print(f"  [{lang}] copied {len(files)} files (no translation needed)")
            continue

        print(f"  [{lang}] translating {len(files)} files ...")
        results = []
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
            futs = {
                ex.submit(_translate_one, client, sem, OPENAI_TRANSLATE_MODEL,
                          fp, out_dir / fp.name): fp
                for fp in files
            }
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
