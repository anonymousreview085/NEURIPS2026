import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import os
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from dotenv import load_dotenv
load_dotenv()

from config import LANGUAGES, MAX_WORKERS, prompts_dir, inference_dir
from helpers import ensure_dir, file_ok, read_text, write_text, retry


def _build_openai(model_name: str):
    from openai import OpenAI
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    def infer(prompt: str) -> str:
        def _do():
            r = client.chat.completions.create(
                model=model_name,
                messages=[{"role": "user", "content": prompt}],
            )
            return r.choices[0].message.content.strip()
        return retry(_do)
    return infer


def _build_gemini(model_name: str):
    import google.generativeai as genai
    genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
    model = genai.GenerativeModel(model_name)
    sem   = threading.Semaphore(4)
    def infer(prompt: str) -> str:
        def _do():
            with sem:
                resp = model.generate_content(prompt)
                text = resp.text.strip()
                if not text:
                    raise ValueError("Empty Gemini response")
                return text
        return retry(_do)
    return infer


def _build_huggingface(model_name: str):
    from huggingface_hub import InferenceClient
    client = InferenceClient(model=model_name, token=os.getenv("HF_TOKEN"))
    def infer(prompt: str) -> str:
        def _do():
            out = client.text_generation(prompt, max_new_tokens=1024, do_sample=False)
            return out.strip() if isinstance(out, str) else str(out).strip()
        return retry(_do)
    return infer


def build_infer_fn(provider: str, model_name: str):
    if provider == "openai":
        return _build_openai(model_name)
    elif provider == "gemini":
        return _build_gemini(model_name)
    elif provider == "huggingface":
        return _build_huggingface(os.getenv("HF_MODEL_NAME", model_name))
    else:
        raise ValueError(f"Unknown LLM_PROVIDER: '{provider}'. Use openai / gemini / huggingface")


def _process(infer_fn, prompt_path: Path, out_path: Path) -> str:
    fname = prompt_path.stem
    if file_ok(out_path):
        return f"skip:{fname}"
    try:
        prompt = read_text(prompt_path)
        if not prompt:
            return f"empty_prompt:{fname}"
        result = infer_fn(prompt)
        if not result:
            return f"empty_result:{fname}"
        write_text(out_path, result)
        return f"done:{fname}"
    except Exception as e:
        return f"error:{fname}:{e}"


def run(model_name: str, mode: str = "standard"):
    provider = os.getenv("LLM_PROVIDER", "openai").lower()
    infer_fn = build_infer_fn(provider, model_name)
    print(f"===== STEP 3: INFERENCE | provider={provider} model={model_name} mode={mode} =====")

    for lang in LANGUAGES:
        in_dir  = prompts_dir(model_name, mode, lang)
        out_dir = inference_dir(model_name, mode, lang)
        if not in_dir.exists():
            print(f"  [SKIP] No prompts for {lang}")
            continue
        ensure_dir(out_dir)
        files   = sorted(in_dir.glob("*.txt"))
        print(f"  [{lang}] {len(files)} prompts")
        results = []
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
            futs = {ex.submit(_process, infer_fn, fp, out_dir / fp.name): fp for fp in files}
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
