from pathlib import Path

BENCHMARK_DIR = Path(__file__).resolve().parent
DATA_DIR      = BENCHMARK_DIR.parent / "data"
CSV_PATH   = DATA_DIR / "euphemism_dataset.csv"
PROMPT_JSON = BENCHMARK_DIR / "prompt.json"

LANGUAGES       = ["aceh", "betawi", "bugis", "indonesia", "jawa", "minang", "sunda"]
TRANSLATE_LANGS = [l for l in LANGUAGES if l != "indonesia"]

LANG_HEADER_MAP = {
    "indonesia": "### PERTANYAAN",
    "jawa":      "### PITAKONAN",
    "sunda":     "### PANYAWALAN",
    "minang":    "### PERTANYOAN",
    "bugis":     "### PERRANYE",
    "betawi":    "### PERTANYAANNYE NIH",
    "aceh":      "### SOALON",
}

SHARED_DIR       = BENCHMARK_DIR / "shared"
DOCTOR_FACTS_DIR = SHARED_DIR / "doctor_facts"
DOCTOR_ANSWERS_DIR = SHARED_DIR / "doctor_answers"

RUNS_DIR = BENCHMARK_DIR / "runs"

def run_dir(model_name: str, mode: str = "standard") -> Path:
    return RUNS_DIR / model_name / mode

def prompts_dir(model_name, mode, lang):
    return run_dir(model_name, mode) / "1_prompts" / lang

def inference_dir(model_name, mode, lang):
    return run_dir(model_name, mode) / "2_inference" / lang

def translate_dir(model_name, mode, lang):
    return run_dir(model_name, mode) / "3_translated" / lang

def llm_facts_dir(model_name, mode, lang):
    return run_dir(model_name, mode) / "fact_extraction" / "llm-facts" / lang

def common_facts_dir(model_name, mode, lang):
    return run_dir(model_name, mode) / "fact_extraction" / "common-facts" / lang

def hallucinated_dir(model_name, mode, lang):
    return run_dir(model_name, mode) / "fact_extraction" / "hallucinated-extractor" / lang

def critical_omitted_dir(model_name, mode, lang):
    return run_dir(model_name, mode) / "fact_extraction" / "critical-omitted" / lang

def metric_eval_dir(model_name, mode, lang):
    return BENCHMARK_DIR / "metricEvaluation" / model_name / mode / lang

OPENAI_EVAL_MODEL      = "gpt-4o-mini"
OPENAI_TRANSLATE_MODEL = "gpt-4.1"
MAX_RETRIES            = 4
RETRY_BACKOFF          = 2.0
MAX_MISMATCH_RETRY     = 5
MAX_WORKERS            = 6
