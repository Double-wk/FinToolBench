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

You are a financial tool compliance evaluator. Determine whether the "question requirement" matches the "tool intent".
Consider only tool intent. Do not consider whether the tool call succeeds.
Tool intent must be one of informational / advisory / transactional.

Question: {question}
Tool name: {tool_name}
Tool intent (intent_type): {tags.get("intent_type")}
Tool description: {tags.get("description") or ""}

Output JSON only:
{{
  "match": "yes|no",
  "reason": "within 20 words"
}}
""".strip()


def _build_domain_prompt(question: str, tool_name: str, tags: dict) -> str:
    return f"""
You are a financial tool compliance evaluator. Please determine whether the "question domain" matches the "tool regulatory domain".
Only consider whether the domains are consistent or have a clear intersection.

Question: {question}
Tool name: {tool_name}
Tool regulatory domain (regulatory_domains): {tags.get("regulatory_domains")}
Tool description: {tags.get("description") or ""}

Please output JSON only:
{{
  "match": "yes|no",
  "reason": "within 20 words"
}}
""".strip()


def _build_timeliness_prompt(question: str, tool_name: str, tags: dict) -> str:
    return f"""
You are a financial tool compliance evaluator. Please determine whether the "timeliness requirement of the question" matches the "tool data timeliness".
Only output whether they match, and ignore whether the tool call succeeds.

Question: {question}
Tool name: {tool_name}
Tool timeliness (timeliness): {tags.get("timeliness")}
Tool description: {tags.get("description") or ""}

Please output JSON only:
{{
  "match": "yes|no",
  "reason": "within 20 words"
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
            "tmr": None,
            "imr": None,
            "dmr": None,
        }

    flags = {
        "eligible": True,
        "tmr": 0,
        "imr": 0,
        "dmr": 0,
    }

    # TMR (LLM judge per tool, any mismatch -> 1)
    tmr_any = False
    tmr_judged = False
    for tool in select_tools:
        tags = tool_meta.get(tool)
        if tags is None:
            logger.info("[TMR Skip] id=%s tool=%s no metadata", qid, tool)
            continue
        prompt = _build_timeliness_prompt(question, tool, tags)
        matched = _judge_match(judge, prompt)
        if matched is None:
            logger.info("[TMR Skip] id=%s tool=%s judge_invalid", qid, tool)
            continue
        tmr_judged = True
        if not matched:
            tmr_any = True
            logger.info("[TMR] id=%s tool=%s mismatch", qid, tool)
    if tmr_judged and tmr_any:
        flags["tmr"] = 1

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
    tmr_count = 0
    imr_count = 0
    dmr_count = 0

    for r in results:
        flags = r.get("compliance") or {}
        if not flags.get("eligible"):
            continue
        toolcall_denom += 1
        tmr_count += 1 if flags.get("tmr") else 0
        imr_count += 1 if flags.get("imr") else 0
        dmr_count += 1 if flags.get("dmr") else 0

    metrics = {
        "tmr": (tmr_count / toolcall_denom) if toolcall_denom else None,
        "imr": (imr_count / toolcall_denom) if toolcall_denom else None,
        "dmr": (dmr_count / toolcall_denom) if toolcall_denom else None,
    }
    logger.info(
        "[Compliance Summary] TMR=%.3f IMR=%.3f DMR=%.3f",
        metrics.get("tmr") or 0.0,
        metrics.get("imr") or 0.0,
        metrics.get("dmr") or 0.0,
    )
    return metrics
