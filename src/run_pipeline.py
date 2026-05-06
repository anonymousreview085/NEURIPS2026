import sys
import argparse
import os
from pathlib import Path
from dotenv import load_dotenv
load_dotenv()

BENCHMARK = Path(__file__).resolve().parent
sys.path.insert(0, str(BENCHMARK))


def _import(module_path: str, fn_name: str):
    import importlib
    mod = importlib.import_module(module_path)
    return getattr(mod, fn_name)


def main():
    parser = argparse.ArgumentParser(description="NeurIPS'26 Submission - Euphemism Benchmark Pipeline")
    parser.add_argument("--steps",  default="1,2,3,4,5,6,7,8,9,10,11",
                        help="Comma-separated step numbers (default: all)")
    parser.add_argument("--mode",   default="both",
                        choices=["standard", "euphemism", "both"])
    parser.add_argument("--model",  default=None)
    args = parser.parse_args()

    model  = args.model or os.getenv("LLM_MODEL_NAME", "gpt-4o")
    steps  = [int(s.strip()) for s in args.steps.split(",") if s.strip()]
    modes  = ["standard", "euphemism"] if args.mode == "both" else [args.mode]

    print("=" * 60)
    print("NeurIPS'26 Submission - Euphemism Benchmark Pipeline")
    print(f"  Model  : {model}")
    print(f"  Steps  : {steps}")
    print(f"  Mode(s): {modes}")
    print("=" * 60)

    if 1 in steps:
        _import("data_prep.step1_prepare_data", "main")()

    for mode in modes:
        if 2 in steps:
            _import("pipeline.step2_prompt_builder",   "run")(model, mode)
        if 3 in steps:
            _import("pipeline.step3_inference",        "run")(model, mode)
        if 4 in steps:
            _import("pipeline.step4_translate",        "run")(model, mode)
        if 5 in steps:
            _import("pipeline.step5_extract_llm_facts",    "run")(model, mode)
        if 6 in steps:
            _import("pipeline.step6_common_fact_extractor","run")(model, mode)
        if 7 in steps:
            _import("pipeline.step7_hallucinated_extractor","run")(model, mode)
        if 8 in steps:
            _import("pipeline.step8_critical_omitted", "run")(model, mode)
        if 9 in steps:
            _import("pipeline.step9_fact_annotation",  "run")(model, mode)

    if 10 in steps:
        _import("analysis.step10_compare_euph_vs_non",  "run")(model)
    if 11 in steps:
        _import("analysis.step11_compare_before_after", "run")(model)

    print("\n" + "=" * 60)
    print("Pipeline complete.")
    print("=" * 60)


if __name__ == "__main__":
    main()
