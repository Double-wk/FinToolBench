import argparse
import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable


@dataclass
class ValidationIssue:
    path: str
    line: int
    message: str

    def format(self) -> str:
        location = f"{self.path}:{self.line}" if self.line else self.path
        return f"{location}: {self.message}"


@dataclass
class ValidationReport:
    errors: list[ValidationIssue] = field(default_factory=list)
    stats: dict[str, int] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return not self.errors

    def add_error(self, path: str, line: int, message: str) -> None:
        self.errors.append(ValidationIssue(path=path, line=line, message=message))

    def merge(self, other: "ValidationReport") -> None:
        self.errors.extend(other.errors)
        for key, value in other.stats.items():
            self.stats[key] = self.stats.get(key, 0) + value


def _read_jsonl(path: str) -> tuple[list[tuple[int, dict[str, Any]]], ValidationReport]:
    report = ValidationReport()
    rows: list[tuple[int, dict[str, Any]]] = []
    try:
        with open(path, "r", encoding="utf-8") as file:
            for line_no, line in enumerate(file, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    item = json.loads(line)
                except json.JSONDecodeError as exc:
                    report.add_error(path, line_no, f"invalid JSON: {exc.msg}")
                    continue
                if not isinstance(item, dict):
                    report.add_error(path, line_no, "line must decode to a JSON object")
                    continue
                rows.append((line_no, item))
    except FileNotFoundError:
        report.add_error(path, 0, "file does not exist")
    return rows, report


def _is_non_empty_str(value: Any) -> bool:
    return isinstance(value, str) and bool(value.strip())


def _validate_string_list(
    value: Any,
    *,
    field_name: str,
    path: str,
    line_no: int,
    report: ValidationReport,
    allow_empty: bool = False,
) -> list[str]:
    if not isinstance(value, list):
        report.add_error(path, line_no, f"{field_name} must be a list")
        return []
    if not value and not allow_empty:
        report.add_error(path, line_no, f"{field_name} must not be empty")
        return []
    invalid = [item for item in value if not _is_non_empty_str(item)]
    if invalid:
        report.add_error(path, line_no, f"{field_name} must contain only strings")
        return []
    return value


def validate_tool_manifest(path: str) -> tuple[set[str], ValidationReport]:
    rows, report = _read_jsonl(path)
    tool_names: set[str] = set()
    seen_names: set[str] = set()

    for line_no, item in rows:
        name = item.get("name")
        if not _is_non_empty_str(name):
            report.add_error(path, line_no, "name must be a non-empty string")
        elif name in seen_names:
            report.add_error(path, line_no, f"duplicate tool name: {name}")
        else:
            seen_names.add(name)
            tool_names.add(name)

        if not _is_non_empty_str(item.get("description")):
            report.add_error(path, line_no, "description must be a non-empty string")

        parameters = item.get("parameters")
        if not isinstance(parameters, dict):
            report.add_error(path, line_no, "parameters must be an object")
        else:
            if parameters.get("type") != "object":
                report.add_error(path, line_no, "parameters.type must be 'object'")
            properties = parameters.get("properties")
            if not isinstance(properties, dict):
                report.add_error(path, line_no, "parameters.properties must be an object")
                properties = {}
            required = parameters.get("required", [])
            required_fields = _validate_string_list(
                required,
                field_name="parameters.required",
                path=path,
                line_no=line_no,
                report=report,
                allow_empty=True,
            )
            missing_required = sorted(set(required_fields) - set(properties))
            if missing_required:
                report.add_error(
                    path,
                    line_no,
                    f"required parameters missing from properties: {missing_required}",
                )

        tags = item.get("financial_tags")
        if not isinstance(tags, dict):
            report.add_error(path, line_no, "financial_tags must be an object")
            continue
        for tag_name in ("timeliness", "intent_type"):
            if not _is_non_empty_str(tags.get(tag_name)):
                report.add_error(
                    path,
                    line_no,
                    f"financial_tags.{tag_name} must be a non-empty string",
                )
        _validate_string_list(
            tags.get("regulatory_domains"),
            field_name="financial_tags.regulatory_domains",
            path=path,
            line_no=line_no,
            report=report,
        )

    report.stats["tools"] = len(rows)
    return tool_names, report


def _validate_selected_tools(
    *,
    path: str,
    line_no: int,
    value: Any,
    tool_names: set[str],
    report: ValidationReport,
    field_name: str = "select_tools",
) -> list[str]:
    tools = _validate_string_list(
        value,
        field_name=field_name,
        path=path,
        line_no=line_no,
        report=report,
    )
    if tool_names:
        unknown = sorted(set(tools) - tool_names)
        if unknown:
            report.add_error(path, line_no, f"{field_name} contains unknown tools: {unknown}")
    return tools


def validate_question_file(path: str, tool_names: set[str]) -> ValidationReport:
    rows, report = _read_jsonl(path)
    seen_ids: set[str] = set()

    for line_no, item in rows:
        if item.get("id") is not None:
            question_id = str(item["id"])
            if question_id in seen_ids:
                report.add_error(path, line_no, f"duplicate question id: {question_id}")
            seen_ids.add(question_id)

        if not _is_non_empty_str(item.get("question") or item.get("query")):
            report.add_error(path, line_no, "question or query must be a non-empty string")
        if "answer" not in item and "ground_truth" not in item:
            report.add_error(path, line_no, "answer or ground_truth must be present")

        selected_tools = _validate_selected_tools(
            path=path,
            line_no=line_no,
            value=item.get("select_tools"),
            tool_names=tool_names,
            report=report,
        )
        tool_nums = item.get("tool_nums")
        if tool_nums is not None and tool_nums != len(selected_tools):
            report.add_error(
                path,
                line_no,
                f"tool_nums={tool_nums} does not match select_tools count={len(selected_tools)}",
            )

    report.stats["questions"] = len(rows)
    return report


def validate_trace_file(path: str, tool_names: set[str]) -> ValidationReport:
    rows, report = _read_jsonl(path)
    seen_ids: set[str] = set()

    for line_no, item in rows:
        if item.get("id") is not None:
            trace_id = str(item["id"])
            if trace_id in seen_ids:
                report.add_error(path, line_no, f"duplicate trace id: {trace_id}")
            seen_ids.add(trace_id)

        if not _is_non_empty_str(item.get("question") or item.get("query")):
            report.add_error(path, line_no, "question or query must be a non-empty string")
        if "execution_result" not in item and "answer" not in item:
            report.add_error(path, line_no, "execution_result or answer must be present")

        if "select_tools" in item:
            _validate_selected_tools(
                path=path,
                line_no=line_no,
                value=item.get("select_tools"),
                tool_names=tool_names,
                report=report,
            )

        tool_calls = item.get("tool_calls", [])
        if tool_calls is None:
            tool_calls = []
        if not isinstance(tool_calls, list):
            report.add_error(path, line_no, "tool_calls must be a list when present")
            continue
        for index, call in enumerate(tool_calls):
            if not isinstance(call, dict):
                report.add_error(path, line_no, f"tool_calls[{index}] must be an object")
                continue
            tool_name = call.get("tool_name")
            if not _is_non_empty_str(tool_name):
                report.add_error(path, line_no, f"tool_calls[{index}].tool_name is required")
            elif tool_names and tool_name not in tool_names:
                report.add_error(
                    path,
                    line_no,
                    f"tool_calls[{index}].tool_name unknown: {tool_name}",
                )
            step = call.get("step", 0)
            if not isinstance(step, int) or step < 0:
                report.add_error(
                    path,
                    line_no,
                    f"tool_calls[{index}].step must be a non-negative integer",
                )

    report.stats["traces"] = len(rows)
    return report


def validate_input_files(
    tool_manifest_path: str,
    *,
    trace_paths: Iterable[str] = (),
    question_paths: Iterable[str] = (),
) -> ValidationReport:
    tool_names, report = validate_tool_manifest(tool_manifest_path)
    for question_path in question_paths:
        report.merge(validate_question_file(question_path, tool_names))
    for trace_path in trace_paths:
        report.merge(validate_trace_file(trace_path, tool_names))
    return report


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Validate FinToolBench JSONL inputs")
    parser.add_argument(
        "--tools",
        default=os.getenv("TOOL_METADATA_PATH", "tools/tools_all_annotated.jsonl"),
        help="tool manifest JSONL path",
    )
    parser.add_argument("--questions", nargs="*", default=[], help="question JSONL files")
    parser.add_argument("--traces", nargs="*", default=[], help="evaluation trace JSONL files")
    parser.add_argument("--max-errors", type=int, default=20, help="maximum errors to print")
    return parser


def main() -> int:
    args = _build_parser().parse_args()
    report = validate_input_files(
        args.tools,
        trace_paths=args.traces,
        question_paths=args.questions,
    )
    if report.ok:
        stats = ", ".join(f"{key}={value}" for key, value in sorted(report.stats.items()))
        print(f"Validation passed ({stats})")
        return 0

    for issue in report.errors[: args.max_errors]:
        print(issue.format())
    remaining = len(report.errors) - args.max_errors
    if remaining > 0:
        print(f"... {remaining} more validation errors")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
