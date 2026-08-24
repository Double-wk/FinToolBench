import argparse
import json
import logging
import os
from datetime import datetime
from typing import Dict

from evaluator import evaluate_dataset, load_jsonl_as_dict
from validate_inputs import validate_input_files


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
    parser.add_argument(
        "--tool_manifest",
        default=os.getenv("TOOL_METADATA_PATH", "tools/tools_all_annotated.jsonl"),
        help="tool metadata JSONL used for compliance evaluation",
    )
    parser.add_argument(
        "--skip_input_validation",
        action="store_true",
        help="skip manifest and trace schema validation before evaluation",
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    logger.info("Start evaluation: inputs=%s output_dir=%s", args.inputs, args.output_dir)

    os.environ["TOOL_METADATA_PATH"] = args.tool_manifest
    if not args.skip_input_validation:
        validation = validate_input_files(args.tool_manifest, trace_paths=args.inputs)
        if not validation.ok:
            for issue in validation.errors[:20]:
                logger.error("Input validation error: %s", issue.format())
            raise SystemExit(
                f"Input validation failed with {len(validation.errors)} error(s)."
            )
        logger.info("Input validation passed: %s", validation.stats)

    all_metrics = {}
    for input_path in args.inputs:
        data = load_jsonl_as_dict(input_path)
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
