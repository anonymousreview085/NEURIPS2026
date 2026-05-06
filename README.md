# Can LLMs Handle Euphemism in Low-Resource Women’s Sexual and Reproductive Health?

This is the official implementation of Neurips'26 Submission for the paper above.

### Folder Structure

```
benchmark/
├── config.py
├── helpers.py
├── prompt.json
├── run_pipeline.py
├── requirements.txt
├── .env.example
│
├── data_prep/
│   └── step1_prepare_data.py
│
├── pipeline/
│   ├── step2_prompt_builder.py
│   ├── step3_inference.py
│   ├── step4_translate.py
│   ├── step5_extract_llm_facts.py
│   ├── step6_common_fact_extractor.py
│   ├── step7_hallucinated_extractor.py
│   ├── step8_critical_omitted.py
│   └── step9_fact_annotation.py
│
└── analysis/
    ├── step10_compare_euph_vs_non.py
    └── step11_compare_before_after.py
```

### Runtime Output

```
benchmark/
├── runs/{model_name}/
│   ├── standard/
│   │   ├── 1_prompts/{lang}/
│   │   ├── 2_inference/{lang}/
│   │   ├── 3_translated/{lang}/
│   │   └── fact_extraction/
│   │       ├── llm-facts/{lang}/
│   │       ├── common-facts/{lang}/
│   │       ├── hallucinated-extractor/{lang}/
│   │       └── critical-omitted/{lang}/
│   └── euphemism/  (same structure, euphemistic IDs only)
│
├── metricEvaluation/{model}/{mode}/{lang}/
│   └── fact-based-annotation-metric-instance.csv
│
└── results/
    ├── euph_vs_non_{model}.csv
    └── before_after_{model}.csv
```

---

## Setup

```bash
pip install -r benchmark/requirements.txt
cp benchmark/.env.example benchmark/.env
```

Edit `benchmark/.env`:

```env
LLM_PROVIDER=openai   # openai / gemini / huggingface
LLM_MODEL_NAME=gpt-4o

# if LLM_PROVIDER=openai
OPENAI_API_KEY=<Your OpenAI API Key>   

# if LLM_PROVIDER=gemini
GEMINI_API_KEY=<Your Gemini API Key>   

# if LLM_PROVIDER=huggingface
HF_TOKEN=<Your HuggingFace Token>   
HF_MODEL_NAME=<Your HuggingFace Model Name>   

# always required (translation + evaluation)
OPENAI_API_KEY_TRANSLATE=<Your OpenAI token>   
```

---

## Running

```bash
cd benchmark

python run_pipeline.py                                       # all steps, both modes
python run_pipeline.py --steps 2,3,4 --mode <standard/euphemism> --model <Your model>
python run_pipeline.py --steps 10,11                         # analysis only
```