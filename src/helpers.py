import os
import re
import time
import threading
from pathlib import Path
from typing import Optional

from config import MAX_RETRIES, RETRY_BACKOFF, MAX_WORKERS

def ensure_dir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p


def sanitize(s: str) -> str:
    return re.sub(r"[^a-zA-Z0-9._-]+", "_", str(s))


def file_ok(path: Path) -> bool:
    try:
        return path.exists() and path.stat().st_size > 0
    except Exception:
        return False


def read_lines(path: Path) -> list[str]:
    if not path.exists():
        return []
    with open(path, encoding="utf-8") as f:
        return [ln.strip() for ln in f if ln.strip()]


def read_text(path: Path) -> str:
    if not path.exists():
        return ""
    return path.read_text(encoding="utf-8").strip()


def write_text(path: Path, text: str):
    ensure_dir(path.parent)
    path.write_text(text, encoding="utf-8")

def make_new_id(underlying_id: str) -> str:
    """0-1 → D-0-1"""
    return f"D-{underlying_id}"


def underlying_id(new_id: str) -> str:
    """D-0-1 → 0-1"""
    parts = str(new_id).split("-")
    return "-".join(parts[1:]) if len(parts) >= 3 else new_id

LEAD_NUM_RE = re.compile(r"^\s*\d+[\).\s-]+\s*")


def strip_numbering(line: str) -> str:
    return LEAD_NUM_RE.sub("", line).strip()


def count_numbered_lines(text: str) -> int:
    if not text:
        return 0
    return sum(1 for ln in text.splitlines() if re.match(r"^\s*\d+[\).\s-]+", ln))


def normalize_yn(line: str) -> str:
    """'1. Yes' → 'yes',  '2. No' → 'no'"""
    s = LEAD_NUM_RE.sub("", line.strip()).lower()
    return s.strip()

def retry(fn, retries: int = MAX_RETRIES, backoff: float = RETRY_BACKOFF):
    last_err = None
    for i in range(retries):
        try:
            return fn()
        except Exception as e:
            last_err = e
            time.sleep(backoff ** i)
    raise last_err

def build_openai_client(api_key_env: str = "OPENAI_API_KEY"):
    from openai import OpenAI
    key = os.getenv(api_key_env)
    if not key:
        raise RuntimeError(f"{api_key_env} is not set in .env")
    return OpenAI(api_key=key)


def openai_chat(client, model: str, messages: list, semaphore: threading.Semaphore) -> str:
    def _do():
        with semaphore:
            r = client.chat.completions.create(model=model, messages=messages)
            return r.choices[0].message.content.strip()
    return retry(_do)

def parse_euphemism_annotation(annotation: str) -> list[tuple[str, str]]:
    """
    'gugurin kandungan → aborsi ; keturunan → anak'
    → [('gugurin kandungan', 'aborsi'), ('keturunan', 'anak')]
    """
    pairs = []
    if not annotation or str(annotation).strip().lower() in ("nan", "none", ""):
        return pairs
    for part in str(annotation).split(";"):
        part = part.strip()
        if "→" in part:
            euph, plain = part.split("→", 1)
            pairs.append((euph.strip(), plain.strip()))
    return pairs


def replace_euphemisms(text: str, pairs: list[tuple[str, str]]) -> str:
    """Replace each euphemism term with its plain equivalent (case-insensitive, word boundary)."""
    for euph, plain in sorted(pairs, key=lambda x: len(x[0]), reverse=True):
        text = re.sub(r"\b" + re.escape(euph) + r"\b", plain, text, flags=re.IGNORECASE)
    return text

LINE_PAT = re.compile(r"^\s*\d+[\).\s-]+(.+)$")


def normalize_fact_lines(raw_text: str) -> list[str]:
    """Normalize and deduplicate SPO fact lines from LLM output."""
    lines = []
    for raw in (raw_text or "").splitlines():
        raw = raw.strip()
        if not raw:
            continue
        m = LINE_PAT.match(raw)
        line = m.group(1).strip() if m else raw
        parts = re.split(r"\s*[-–—>→]+\s*", line)
        norm = f"{parts[0].strip()} - {parts[1].strip()} - {' - '.join(p.strip() for p in parts[2:])}" \
               if len(parts) >= 3 else line
        lines.append(norm)
    seen, out = set(), []
    for ln in lines:
        key = ln.lower()
        if key not in seen:
            seen.add(key)
            out.append(ln)
    return [f"{i+1}. {l}" for i, l in enumerate(out)]

def print_summary(tag: str, results: list[str]):
    done    = sum(1 for r in results if r.startswith("done:"))
    skipped = sum(1 for r in results if r.startswith("skip"))
    empty   = sum(1 for r in results if r.startswith("empty"))
    missing = sum(1 for r in results if r.startswith("missing"))
    errors  = sum(1 for r in results if r.startswith("error"))
    print(f"\n=== SUMMARY {tag} ===")
    print(f"  Done    : {done}")
    print(f"  Skipped : {skipped}")
    print(f"  Empty   : {empty}")
    print(f"  Missing : {missing}")
    print(f"  Errors  : {errors}")
