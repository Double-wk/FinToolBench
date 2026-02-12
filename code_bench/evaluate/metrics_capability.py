import json
from collections import Counter
from typing import Any, List, Optional, Tuple


def normalize_tool_calls(raw: Any) -> List[dict]:
    if raw is None:
        return []
    if not isinstance(raw, list):
        return []
    item = []
    for per in raw:
        if isinstance(per, dict):
            per = dict(per)
            per.pop("long_output", None)
        item.append(per)
    return item


def example_has_tool_error(example: dict) -> Tuple[bool, str]:
    if example.get("error"):
        return True, f"example.error: {str(example.get('error'))[:120]}"

    tool_calls = normalize_tool_calls(example.get("tool_calls"))
    if not tool_calls:
        return True, "no tool_calls"

    max_step = max((tc.get("step", 0) for tc in tool_calls), default=0)
    last_round_calls = [(i, tc) for i, tc in enumerate(tool_calls) if tc.get("step", 0) == max_step]

    for i, tc in last_round_calls:
        output = tc.get("output")
        if isinstance(output, dict) and output.get("error"):
            return True, f"tool_call[{i}] output.error"
        if isinstance(output, str):
            low = output.lower()
            if "error" in low or "exception" in low or "traceback" in low:
                return True, f"tool_call[{i}] output error text"

    return False, ""


def tool_pass_label(example: dict) -> Tuple[str, str]:
    has_err, reason = example_has_tool_error(example)
    if has_err:
        return "failed", reason
    return "passed", "tool_calls ok"


def extract_tool_fields(example: dict) -> Tuple[bool, List[str], str, str, int]:
    tool_calls = normalize_tool_calls(example.get("tool_calls"))
    tool_invoked = bool(tool_calls)
    max_step = max((tc.get("step", 0) for tc in tool_calls), default=0)
    last_round_calls = [tc for tc in tool_calls if tc.get("step", 0) == max_step]
    select_tools = [tc.get("tool_name") for tc in last_round_calls if tc.get("tool_name")]
    pass_label, pass_reason = tool_pass_label(example)
    return tool_invoked, select_tools, pass_label, pass_reason, max_step


def _looks_like_json(text: str) -> bool:
    text = text.strip()
    if not text:
        return False
    if (text.startswith("{") and text.endswith("}")) or (text.startswith("[") and text.endswith("]")):
        try:
            json.loads(text)
            return True
        except Exception:
            return False
    return False


def _is_choice(text: str) -> bool:
    if not text:
        return False
    t = text.strip().upper()
    return len(t) == 1 and t in {"A", "B", "C", "D", "E"}


def _is_numeric(text: str) -> bool:
    if not text:
        return False
    t = text.strip()
    has_digit = any(ch.isdigit() for ch in t)
    if not has_digit:
        return False
    allowed = set("0123456789.,+-eE%亿万千百十兆/ ")
    return all((ch.isdigit() or ch in allowed) for ch in t)


def determine_answer_type(gold: str) -> str:
    if _looks_like_json(gold):
        return "criterium"
    if _is_choice(gold):
        return "choice"
    if _is_numeric(gold):
        return "numeric"
    return "other_str"


def parse_json_from_llm(text: str) -> Optional[dict]:
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


def build_score_prompt(question: str, answer: str, gold: str, answer_type: str) -> str:
    if answer_type == "numeric":
        return f"""
你是严格评测员，请判断“模型回答”的数值是否与“标准答案”数值等价（允许单位换算，不接受接近）。
评分规则：
- 1.0：数值等价
- 0.0：其他

问题：{question}
模型回答：{answer}
标准答案：{gold}

请只输出JSON：
{{
  "score": 0.0,
  "label": "correct|wrong",
  "reason": "20字以内"
}}
""".strip()
    if answer_type == "choice":
        return f"""
你是严格评测员，请判断“模型回答”的选项是否与“标准答案”选项一致。
评分规则：
- 1.0：选项完全一致
- 0.0：其他（包括多选或模糊）

问题：{question}
模型回答：{answer}
标准答案：{gold}

请只输出JSON：
{{
  "score": 0.0,
  "label": "correct|wrong",
  "reason": "20字以内"
}}
""".strip()
    if answer_type == "criterium":
        return f"""
你是严格评测员，请判断“模型回答”的多因子分析是否合理。
评估维度：
1) 覆盖关键因素（coverage）
2) 正负方向一致（direction）
3) 权重是否失真（calibration）

评分规则：
- 1.0：合理
- 0.5：部分合理
- 0.0：不合理

问题：{question}
模型回答：{answer}
标准答案：{gold}

请只输出JSON：
{{
  "score": 0.0,
  "label": "correct|partial|wrong",
  "reason": "20字以内"
}}
""".strip()
    return f"""
你是严格评测员，请根据“标准答案”判断“模型回答”的正确性。

评分规则：
- 1.0：完全正确（允许等价表达、单位换算）
- 0.5：部分正确（核心结论对，但缺关键点/存在小错误）
- 0.0：错误或答非所问

问题：{question}
模型回答：{answer}
标准答案：{gold}

请只输出JSON：
{{
  "score": 0.0,
  "label": "correct|partial|wrong",
  "reason": "20字以内"
}}
""".strip()


def judge_score_repeat(
    judge,
    question: str,
    answer: str,
    gold: str,
    answer_type: str,
    repeat_k: int = 3,
) -> Tuple[Optional[float], Optional[str], str]:
    if not gold:
        return None, None, ""

    scores: List[float] = []
    labels: List[str] = []
    reasons: List[str] = []

    prompt = build_score_prompt(question, answer, gold, answer_type)

    for _ in range(repeat_k):
        txt = judge.call(prompt)
        parsed = parse_json_from_llm(txt) or {}

        sc = parsed.get("score")
        try:
            sc = float(sc)
            if answer_type in {"numeric", "choice"}:
                sc = 1.0 if sc >= 0.5 else 0.0
            sc = max(0.0, min(1.0, sc))
            scores.append(sc)
        except Exception:
            pass

        lb = parsed.get("label")
        if isinstance(lb, str):
            lb = lb.strip().lower()
            if lb in {"correct", "partial", "wrong"}:
                labels.append(lb)

        rs = parsed.get("reason")
        if isinstance(rs, str) and rs.strip():
            reasons.append(rs.strip())

    score_mean = (sum(scores) / len(scores)) if scores else None
    label_majority = Counter(labels).most_common(1)[0][0] if labels else None
    reason = reasons[0] if reasons else ""
    return score_mean, label_majority, reason


def summarize_capability(results: List[dict]) -> dict:
    total = len(results)
    if total == 0:
        return {}

    invoked_count = sum(1 for r in results if r.get("tool_invoked"))
    pass_count = sum(1 for r in results if r.get("pass") == "passed")
    score_vals = [r.get("score_mean") for r in results if isinstance(r.get("score_mean"), (int, float))]
    avg_score: Optional[float] = sum(score_vals) / len(score_vals) if score_vals else None
    css_vals = [
        r.get("score_mean")
        for r in results
        if r.get("pass") == "passed" and isinstance(r.get("score_mean"), (int, float))
    ]
    css_score: Optional[float] = sum(css_vals) / len(css_vals) if css_vals else None

    return {
        "tir": invoked_count / total,
        "tesr": pass_count / total,
        "cer": (pass_count / invoked_count) if invoked_count else None,
        "css": css_score,
        "soft_score": avg_score,
    }


def log_capability_summary(logger, metrics: dict) -> None:
    logger.info(
        "[Summary] TIR=%.3f TESR=%.3f CER=%.3f Soft=%.3f CSS=%.3f",
        metrics.get("tir", 0.0) or 0.0,
        metrics.get("tesr", 0.0) or 0.0,
        metrics.get("cer", 0.0) or 0.0,
        metrics.get("soft_score", 0.0) or 0.0,
        metrics.get("css", 0.0) or 0.0,
    )
