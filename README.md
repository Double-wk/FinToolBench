# FinToolBench: Evaluating LLM Agents for Real-World Financial Tool Use

FinToolBench is a real-world, runnable benchmark for evaluating financial tool-using agents. It pairs 760 executable financial tools with 295 tool-required queries and measures not only execution success but also finance-critical compliance dimensions, including timeliness, intent type, and regulatory domain alignment. We also provide FATR, a finance-aware tool retrieval and reasoning baseline that improves stability and compliance. FinToolBench enables auditable financial tool execution and provides a reproducible evaluation protocol for finance-aware agents.

## Overview (This Release)

This open release contains the **evaluation pipeline**, **RapidAPI subscription helper**, and **minimal data** required to reproduce results. Full agent training/build scripts and other internal components are not included.

## Highlights

- **Runnable tool library**: 760 free-tier financial tools (RapidAPI + AkShare), normalized into a unified schema.
- **Tool-required questions**: 295 questions (166 single-tool, 129 multi-tool) adapted from FinanceBench and OpenFinData.
- **Trace-level evaluation**: capability metrics plus compliance mismatch rates from explicit finance attributes.
- **Baseline**: FATR (Finance-Aware Tool Routing) with attribute injection, caching, retries, and output compression.

## Benchmark Contents

### Tool Inventory

- **Sources**: RapidAPI endpoints and AkShare interfaces.
- **Filtering**: executability checks, deduplication, and authentication feasibility.
- **Normalization**: canonical parameter names, consistent date formats, and structured tool signatures.
- **Finance attributes**: `timeliness`, `intent_type`, and `regulatory_domains` for each tool.

Key files:
- `tools/tools_all_annotated.jsonl` (tool manifest + finance tags)

### Question Set

- **Sources**: FinanceBench (HF) and OpenFinData release.
- **Selection**: tool-required only, capped length, top-K tool retrieval, multi-sample verification.
- **Composition**: 166 single-tool + 129 multi-tool questions.

Key files:
- `data/question/select_data_real_remove_duplicates.jsonl` (benchmark questions)

## Evaluation

Each run produces a **tool trace** and a final answer. Evaluation separates **capability**, **correctness**, and **compliance**:

- **Capability**:
  - Tool Invocation Rate (TIR)
  - Tool Execution Success Rate (TESR)
  - Conditional Execution Rate (CER)
  - SoftScore (LLM judge / exact match),
  - CSS (correctness conditioned on successful execution)
- **Compliance mismatch**:
  - Timeliness (TMR), Intent (IMR), Domain (DMR)

## Baseline: FATR

FATR injects finance attributes into tool cards and stabilizes execution via caching, retries, and output compression. It is designed as a lightweight, reproducible baseline for finance-aware tool routing.

## What This Repo Includes (Open Parts)

This release is limited to the evaluation pipeline, subscription helper, and minimal tool data needed to reproduce results.

**Code**
- `code_bench/evaluate/` evaluation pipeline
- `code_bench/tools/tools_rapidapi_subscribe_url.py` subscribe from `tools/home_url.json`
- `code_bench/utils/model_requests.py` model request wrapper used by evaluation

**Data**
- `data/question/select_data_real_remove_duplicates.jsonl` benchmark questions
- `tools/tools_all_annotated.jsonl` tool manifest + finance tags
- `tools/home_url.json` RapidAPI home URLs for subscription

Other folders are kept internal.

## Execution Steps

1. Install dependencies:
```
pip install -r requirements.txt
```

2. Subscribe RapidAPI tools from `tools/home_url.json` (requires your RapidAPI account login):
```
python -u code_bench/tools/tools_rapidapi_subscribe_url.py
```

3. Configure LLM API (set your model endpoint and credentials in the code or environment as required).


4. Validate the released manifest and benchmark questions:
```
python -m code_bench.evaluate.validate_inputs \
  --tools tools/tools_all_annotated.jsonl \
  --questions data/question/select_data_real_remove_duplicates.jsonl
```

5. Run evaluation:
```
python -u code_bench/evaluate/run_relative_eval.py \
  --inputs data/result/result/result_model_name_full.jsonl \
  --output_dir data/eval/relative_model
```

`run_relative_eval.py` validates the tool manifest and result traces before
starting the judge calls. Pass `--skip_input_validation` only when evaluating
legacy traces that intentionally do not follow the released schema.

This writes:
- `<output_dir>/<setting>_results.jsonl`
- `<output_dir>/<setting>_metrics.json`
- `<output_dir>/all_metrics.json`

## RapidAPI Subscription (Optional)

Subscribe only the APIs listed in `tools/home_url.json`:

```
python -u code_bench/tools/tools_rapidapi_subscribe_url.py
```

If you want to see the browser UI, do not pass `--headless`.

Prerequisite: you must register and log in to your own RapidAPI account before running this script. The script relies on your RapidAPI login cookies to subscribe APIs.

## Notes

- Tool execution can drift over time; report evaluation date and tool manifest version when publishing results.
- Some RapidAPI endpoints may change or become unavailable over time.
