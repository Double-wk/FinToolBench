import argparse
import json
import logging
import os
from datetime import datetime
from typing import Dict

from evaluator import evaluate_dataset, load_jsonl_as_dict


os.makedirs("./logs", exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(filename)s - %(message)s",
    handlers=[
        logging.FileHandler(f"./logs/eval_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)


def write_jsonl(path: str, rows: list) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_json(path: str, data: Dict) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluator (no baseline)")
    parser.add_argument("--inputs", nargs="+", required=True, help="result jsonl files")
    parser.add_argument("--output_dir", required=True, help="output directory")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    logger.info("Start evaluation: inputs=%s output_dir=%s", args.inputs, args.output_dir)

    # judge = build_default_judge()

    all_metrics = {}
    for input_path in args.inputs:
        data = load_jsonl_as_dict(input_path)
        # results, metrics = evaluate_dataset(data, judge)
        results, metrics = evaluate_dataset(data)
        setting_name = os.path.splitext(os.path.basename(input_path))[0]
        out_results = os.path.join(args.output_dir, f"{setting_name}_results.jsonl")
        out_metrics = os.path.join(args.output_dir, f"{setting_name}_metrics.json")

        write_jsonl(out_results, results)
        write_json(out_metrics, metrics)
        all_metrics[setting_name] = metrics
        logger.info("Finished %s: metrics=%s", setting_name, metrics)

    write_json(os.path.join(args.output_dir, "all_metrics.json"), all_metrics)
    logger.info("Wrote all_metrics.json")


if __name__ == "__main__":
    main()


# nohup python -u code_bench/evaluate/run_relative_eval.py \
#   --inputs /Users/double/Documents/LLM/论文/fin_beachmark/data/result/result_ablation_all/result_Doubao-Seed-1.6_full.jsonl /Users/double/Documents/LLM/论文/fin_beachmark/data/result/result_ablation_all/result_Doubao-Seed-1.6_wo_injection.jsonl /Users/double/Documents/LLM/论文/fin_beachmark/data/result/result_ablation/result_GLM-4.7-Flash_full.jsonl /Users/double/Documents/LLM/论文/fin_beachmark/data/result/result_ablation/result_Qwen3-8B_full.jsonl\
#   --output_dir data/eval/relative_new >/Users/double/Documents/LLM/论文/fin_beachmark/eval_no_injection_new.log 2>&1 &



