import json
import logging
import os
import sys
import time
from typing import Dict, List, Optional, Tuple


current_file = os.path.abspath(__file__)
project_root = os.path.dirname(os.path.dirname(os.path.dirname(current_file)))  # 上三级
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from tqdm import tqdm

from code_bench.utils.model_requests import boyue_model_requests
from code_bench.evaluate.metrics_capability import (
    determine_answer_type,
    extract_tool_fields,
    judge_score_repeat,
    log_capability_summary,
    summarize_capability,
)
from code_bench.evaluate.metrics_compliance import (
    ComplianceJudge,
    evaluate_compliance_one,
    load_tool_metadata,
    summarize_compliance,
)


logger = logging.getLogger(__name__)

# 重复评测次数（可通过环境变量覆盖）
SCORE_REPEAT_K = int(os.getenv("SCORE_REPEAT_K", "3"))

# =========================
def load_jsonl_as_dict(path: str, id_key: str = "id") -> Dict[str, dict]:
    data: Dict[str, dict] = {}
    logger.info("Loading dataset from %s", path)
    with open(path, "r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
            except Exception as e:
                logger.warning(f"skip bad json line idx={idx}: {e}")
                continue
            qid = str(item.get(id_key, idx))
            data[qid] = item
    logger.info("Loaded %s examples", len(data))
    return data


# =========================
# Field extract
# =========================
def extract_question(example: dict) -> str:
    return str(example.get("question") or example.get("query") or "").strip()


def extract_answer(example: dict) -> str:
    return str(example.get("execution_result") or example.get("answer") or "").strip()


def extract_ground_truth(example: dict) -> str:
    return str(example.get("ground_truth") or "").strip()


class LLMJudge:
    def __init__(
        self,
        model: str,
        max_retries: int = 3,
        sleep_seconds: float = 0.0,
    ):
        self.model = model
        self.max_retries = max_retries
        self.sleep_seconds = sleep_seconds

    def call(self, prompt: str) -> str:
        for attempt in range(self.max_retries):
            try:
                if self.sleep_seconds:
                    time.sleep(self.sleep_seconds)

                response = boyue_model_requests(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt}],
                    available_tools=None,
                )
                return response or ""
            except Exception as e:
                logger.warning(f"judge retry {attempt + 1}/{self.max_retries} failed: {e}")
                if attempt == self.max_retries - 1:
                    return ""
        return ""


# =========================
# Core evaluate
# =========================
def evaluate_one(
    example: dict,
    judge: LLMJudge,
    tool_meta: dict,
    compliance_judge: ComplianceJudge,
) -> dict:
    qid = example.get("id")
    question = extract_question(example)
    answer = extract_answer(example)
    gold = extract_ground_truth(example)
    answer_type = determine_answer_type(gold)
    logger.info("[Eval Start] id=%s", qid)
    logger.info("Question: %s", question)
    (
        tool_invoked,
        select_tools,
        pass_label,
        pass_reason,
        max_step,
    ) = extract_tool_fields(example)
    if not tool_invoked:
        logger.info("[No Tool] id=%s", qid)
    elif pass_label != "passed":
        logger.info("[Tool Failed] id=%s reason=%s", qid, pass_reason)
    else:
        logger.info("[Tool Passed] id=%s steps=%s", qid, max_step)

    # score repeat (基于 ground_truth)
    logger.info("[Answer Type] id=%s type=%s", qid, answer_type)
    score_mean, label_majority, score_reason = judge_score_repeat(
        judge=judge,
        question=question,
        answer=answer,
        gold=gold,
        answer_type=answer_type,
        repeat_k=SCORE_REPEAT_K,
    )
    if score_mean is None:
        logger.warning("[Judge Failed] id=%s no valid score", qid)
    elif score_mean < 0.5:
        logger.info(
            "[Low Score] id=%s score=%s label=%s reason=%s",
            qid,
            score_mean,
            label_majority,
            score_reason,
        )


    result = {
        "id": example.get("id"),
        "setting": example.get("setting"),
        "question": question,
        "execution_result": answer,
        "ground_truth": gold,

        # tool ability
        "tool_invoked": tool_invoked,
        "select_tools": select_tools,
        "pass": pass_label,
        "pass_reason": pass_reason,

        # final answer quality (repeat3 stable)
        "score_mean": score_mean,
        "label_majority": label_majority,
        "score_reason": score_reason,
    }
    result["compliance"] = evaluate_compliance_one(result, tool_meta, compliance_judge)
    return result


def evaluate_dataset(
    data: Dict[str, dict],
) -> Tuple[List[dict], dict]:
    tool_meta_path = os.getenv("TOOL_METADATA_PATH", "tools/tools_all_annotated.jsonl")
    tool_meta = load_tool_metadata(tool_meta_path)
    judge = LLMJudge(model=os.getenv("JUDGE_MODEL", "ep-20251101221159-hhmrg"))
    compliance_judge = ComplianceJudge(model=os.getenv("JUDGE_MODEL", "ep-20251101221159-hhmrg"))

    results = []
    for qid, example in tqdm(data.items(), total=len(data), desc="Evaluating"):
        result = evaluate_one(example, judge, tool_meta, compliance_judge)
        results.append(result)

    cap_metrics = summarize_capability(results)
    compliance_metrics = summarize_compliance(results, tool_meta)
    metrics = {
        "cap": cap_metrics,
        "com": compliance_metrics,
    }
    log_capability_summary(logger, cap_metrics)
    return results, metrics
