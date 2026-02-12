import json
import logging
import os
import time
from typing import Dict, List, Optional

from code_bench.utils.model_requests import boyue_model_requests


logger = logging.getLogger(__name__)


def load_tool_metadata(path: str) -> Dict[str, dict]:
    meta: Dict[str, dict] = {}
    try:
        logger.info("Loading tool metadata from %s", path)
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                name = obj.get("name")
                if name:
                    tags = dict(obj.get("financial_tags", {}) or {})
                    tags["description"] = obj.get("description", "")
                    meta[name] = tags
    except FileNotFoundError:
        logger.warning("Tool metadata not found: %s", path)
        return {}
    logger.info("Loaded tool metadata: %s tools", len(meta))
    return meta


def _tool_tags(tool_meta: Dict[str, dict], tool_name: str) -> dict:
    return tool_meta.get(tool_name, {}) or {}


class ComplianceJudge:
    def __init__(
        self,
        model: Optional[str] = None,
        max_retries: int = 3,
        sleep_seconds: float = 0.0,
    ):
        self.model = model or os.getenv("COMPLIANCE_JUDGE_MODEL") or os.getenv("JUDGE_MODEL") or ""
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
                logger.warning("compliance judge retry %s/%s failed: %s", attempt + 1, self.max_retries, e)
                if attempt == self.max_retries - 1:
                    return ""
        return ""


def _parse_json_from_llm(text: str) -> Optional[dict]:
    if not text:
        return None
    text = text.strip()
    try:
        return json.loads(text)
    except Exception:
        pass
    try:
        start = text.index("{")
        end = text.rindex("}")
        return json.loads(text[start : end + 1])
    except Exception:
        return None


def _build_intent_prompt(question: str, tool_name: str, tags: dict) -> str:
    return f"""
你是金融工具合规评测员，请判断“问题需求”与“工具意图”是否匹配。
只考虑工具意图，不考虑是否调用成功。
工具意图仅取 informational / advisory / transactional 之一。

问题：{question}
工具名称：{tool_name}
工具意图(intent_type)：{tags.get("intent_type")}
工具描述：{tags.get("description") or ""}

请只输出JSON：
{{
  "match": "yes|no",
  "reason": "20字以内"
}}
""".strip()


def _build_domain_prompt(question: str, tool_name: str, tags: dict) -> str:
    return f"""
你是金融工具合规评测员，请判断“问题领域”与“工具监管领域”是否匹配。
只考虑领域是否一致或存在明确交集。

问题：{question}
工具名称：{tool_name}
工具监管领域(regulatory_domains)：{tags.get("regulatory_domains")}
工具描述：{tags.get("description") or ""}

请只输出JSON：
{{
  "match": "yes|no",
  "reason": "20字以内"
}}
""".strip()


def _build_freshness_prompt(question: str, tool_name: str, tags: dict) -> str:
    print(tags.get("description"))
    print(tags.get("update_frequency"))
    return f"""
你是金融工具合规评测员，请判断“问题的时效性需求”与“工具数据更新频率”是否匹配。
只输出是否匹配，忽略工具调用成功与否。

问题：{question}
工具名称：{tool_name}
工具更新频率(update_frequency)：{tags.get("update_frequency")}
工具描述：{tags.get("description") or ""}

请只输出JSON：
{{
  "match": "yes|no",
  "reason": "20字以内"
}}
""".strip()


def _judge_match(judge: ComplianceJudge, prompt: str) -> Optional[bool]:
    txt = judge.call(prompt)
    parsed = _parse_json_from_llm(txt) or {}
    val = parsed.get("match")
    if isinstance(val, str):
        v = val.strip().lower()
        if v == "yes":
            return True
        if v == "no":
            return False
    return None


def evaluate_compliance_one(result: dict, tool_meta: Dict[str, dict], judge: ComplianceJudge) -> dict:
    qid = result.get("id")
    question = result.get("question") or ""
    select_tools = result.get("select_tools") or []

    if not select_tools:
        logger.info("[Compliance Skip] id=%s no select_tools", qid)
        return {
            "eligible": False,
            "fmr": None,
            "imr": None,
            "dmr": None,
        }

    flags = {
        "eligible": True,
        "fmr": 0,
        "imr": 0,
        "dmr": 0,
    }

    # FMR (LLM judge per tool, any mismatch -> 1)
    fmr_any = False
    fmr_judged = False
    for tool in select_tools:
        tags = tool_meta.get(tool)
        if tags is None:
            logger.info("[FMR Skip] id=%s tool=%s no metadata", qid, tool)
            continue
        prompt = _build_freshness_prompt(question, tool, tags)
        matched = _judge_match(judge, prompt)
        if matched is None:
            logger.info("[FMR Skip] id=%s tool=%s judge_invalid", qid, tool)
            continue
        fmr_judged = True
        if not matched:
            fmr_any = True
            logger.info("[FMR] id=%s tool=%s mismatch", qid, tool)
    if fmr_judged and fmr_any:
        flags["fmr"] = 1

    # IMR (LLM judge per tool, any mismatch -> 1)
    imr_any = False
    imr_judged = False
    for tool in select_tools:
        tags = tool_meta.get(tool)
        if tags is None:
            logger.info("[IMR Skip] id=%s tool=%s no metadata", qid, tool)
            continue
        prompt = _build_intent_prompt(question, tool, tags)
        matched = _judge_match(judge, prompt)
        if matched is None:
            logger.info("[IMR Skip] id=%s tool=%s judge_invalid", qid, tool)
            continue
        imr_judged = True
        if not matched:
            imr_any = True
            logger.info("[IMR] id=%s tool=%s mismatch", qid, tool)
    if imr_judged and imr_any:
        flags["imr"] = 1

    # DMR (LLM judge per tool, any mismatch -> 1)
    dmr_any = False
    dmr_judged = False
    for tool in select_tools:
        tags = tool_meta.get(tool)
        if tags is None:
            logger.info("[DMR Skip] id=%s tool=%s no metadata", qid, tool)
            continue
        prompt = _build_domain_prompt(question, tool, tags)
        matched = _judge_match(judge, prompt)
        if matched is None:
            logger.info("[DMR Skip] id=%s tool=%s judge_invalid", qid, tool)
            continue
        dmr_judged = True
        if not matched:
            dmr_any = True
            logger.info("[DMR] id=%s tool=%s mismatch", qid, tool)
    if dmr_judged and dmr_any:
        flags["dmr"] = 1

    return flags


def summarize_compliance(results: List[dict], tool_meta: Dict[str, dict]) -> dict:
    if not results:
        return {}
    toolcall_denom = 0
    fmr_count = 0
    imr_count = 0
    dmr_count = 0

    for r in results:
        flags = r.get("compliance") or {}
        if not flags.get("eligible"):
            continue
        toolcall_denom += 1
        fmr_count += 1 if flags.get("fmr") else 0
        imr_count += 1 if flags.get("imr") else 0
        dmr_count += 1 if flags.get("dmr") else 0

    metrics = {
        "fmr": (fmr_count / toolcall_denom) if toolcall_denom else None,
        "imr": (imr_count / toolcall_denom) if toolcall_denom else None,
        "dmr": (dmr_count / toolcall_denom) if toolcall_denom else None,
    }
    logger.info(
        "[Compliance Summary] FMR=%.3f IMR=%.3f DMR=%.3f",
        metrics.get("fmr") or 0.0,
        metrics.get("imr") or 0.0,
        metrics.get("dmr") or 0.0,
    )
    return metrics
