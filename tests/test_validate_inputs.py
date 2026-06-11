import json
import tempfile
import unittest
from pathlib import Path

from code_bench.evaluate.validate_inputs import (
    validate_input_files,
    validate_tool_manifest,
    validate_trace_file,
)


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    with open(path, "w", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")


class ValidateInputsTest(unittest.TestCase):
    def test_valid_manifest_and_trace_pass(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            tools_path = tmp / "tools.jsonl"
            trace_path = tmp / "trace.jsonl"
            _write_jsonl(
                tools_path,
                [
                    {
                        "name": "companies_balance_sheet_statements",
                        "description": "Balance sheet statements",
                        "parameters": {
                            "type": "object",
                            "properties": {"ticker": {"type": "string"}},
                            "required": ["ticker"],
                        },
                        "financial_tags": {
                            "timeliness": "as_filed",
                            "intent_type": "informational",
                            "regulatory_domains": ["equities"],
                        },
                    },
                ],
            )
            _write_jsonl(
                trace_path,
                [
                    {
                        "id": "q1",
                        "question": "What is FY2018 net PP&E for 3M?",
                        "execution_result": "$8.70",
                        "select_tools": ["companies_balance_sheet_statements"],
                        "tool_calls": [
                            {
                                "tool_name": "companies_balance_sheet_statements",
                                "step": 1,
                                "output": {"rows": []},
                            },
                        ],
                    },
                ],
            )

            report = validate_input_files(str(tools_path), trace_paths=[str(trace_path)])

            self.assertTrue(report.ok)
            self.assertEqual(report.stats["tools"], 1)
            self.assertEqual(report.stats["traces"], 1)

    def test_manifest_reports_required_parameter_not_in_properties(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tools_path = Path(tmpdir) / "tools.jsonl"
            _write_jsonl(
                tools_path,
                [
                    {
                        "name": "tool_a",
                        "description": "A tool",
                        "parameters": {
                            "type": "object",
                            "properties": {},
                            "required": ["ticker"],
                        },
                        "financial_tags": {
                            "timeliness": "daily",
                            "intent_type": "informational",
                            "regulatory_domains": ["equities"],
                        },
                    },
                ],
            )

            _, report = validate_tool_manifest(str(tools_path))

            self.assertFalse(report.ok)
            self.assertIn("required parameters missing", report.errors[0].message)

    def test_trace_reports_unknown_tool_and_invalid_step(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            trace_path = Path(tmpdir) / "trace.jsonl"
            _write_jsonl(
                trace_path,
                [
                    {
                        "question": "What is the latest price?",
                        "answer": "100",
                        "tool_calls": [{"tool_name": "unknown_tool", "step": -1}],
                    },
                ],
            )

            report = validate_trace_file(str(trace_path), {"known_tool"})

            messages = [issue.message for issue in report.errors]
            self.assertIn("tool_calls[0].tool_name unknown: unknown_tool", messages)
            self.assertIn(
                "tool_calls[0].step must be a non-negative integer",
                messages,
            )


if __name__ == "__main__":
    unittest.main()
