#!/usr/bin/env python3
"""
Convert detailed_log.jsonl to a readable HTML report.

Usage:
    python view_log.py detailed_log.jsonl -o report.html
    python view_log.py detailed_log.jsonl  # prints to stdout as formatted text
"""

import argparse
import html
import json
import sys


def load_jsonl(path: str) -> list[dict]:
    entries = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                entries.append(json.loads(line))
    return entries


def render_html(entries: list[dict], output_path: str):
    rows = []

    for entry in entries:
        sample_id = entry.get("sample_id", "?")
        question = entry.get("question", "")
        gt_answer = entry.get("gt_answer", "")
        prediction = entry.get("prediction", "")
        f1 = entry.get("f1", 0)
        em = entry.get("exact_match", False)
        error = entry.get("error")
        image_b64 = entry.get("image_base64", "")

        # Build image HTML
        if image_b64:
            image_html = f'<div class="sample-image"><img src="data:image/png;base64,{image_b64}" alt="Sample image"></div>'
        else:
            image_html = ""

        # Tool calls section
        tool_calls_html = ""
        for tc in entry.get("tool_calls", []):
            tool_name = tc.get("tool_name", "")
            args = json.dumps(tc.get("arguments", {}), indent=2, ensure_ascii=False)

            # What model actually sees (may be truncated)
            result_sent = tc.get("result_sent_to_model", "")
            if not result_sent:
                # Fallback for old logs
                result_full = tc.get("result", tc.get("result_full", {}))
                result_sent = json.dumps(result_full, indent=2, ensure_ascii=False)

            # Full result (for reference)
            result_full = tc.get("result_full", tc.get("result", {}))
            result_full_str = json.dumps(result_full, indent=2, ensure_ascii=False)

            tool_calls_html += f"""
            <div class="tool-call">
                <div class="tool-header">{html.escape(tool_name)}</div>
                <div class="tool-section">
                    <strong>Arguments:</strong>
                    <pre>{html.escape(args)}</pre>
                </div>
                <div class="tool-section">
                    <strong>Result Sent to Model ({len(result_sent)} chars):</strong>
                    <pre>{html.escape(result_sent)}</pre>
                </div>
                <details class="full-result">
                    <summary>Full Result ({len(result_full_str)} chars)</summary>
                    <pre>{html.escape(result_full_str)}</pre>
                </details>
            </div>
            """

        # Model responses section
        responses_html = ""
        for resp in entry.get("model_responses", []):
            iteration = resp.get("iteration", "?")
            response_text = resp.get("response", "")
            responses_html += f"""
            <div class="response">
                <div class="response-header">Response {iteration}</div>
                <pre>{html.escape(response_text)}</pre>
            </div>
            """

        # Status indicator
        if error:
            status_class = "error"
            status_text = "ERROR"
        elif em:
            status_class = "correct"
            status_text = "CORRECT"
        else:
            status_class = "incorrect"
            status_text = "INCORRECT"

        rows.append(f"""
        <div class="sample">
            <div class="sample-header">
                <span class="sample-id">Sample {sample_id}</span>
                <span class="status {status_class}">{status_text}</span>
                <span class="metrics">F1: {f1:.3f}</span>
            </div>

            {image_html}

            <div class="qa-section">
                <div class="question"><strong>Question:</strong> {html.escape(question)}</div>
                <div class="gt-answer"><strong>Ground Truth:</strong> {html.escape(gt_answer)}</div>
                <div class="prediction"><strong>Prediction:</strong> {html.escape(prediction[:500])}{"..." if len(prediction) > 500 else ""}</div>
                {f'<div class="error-msg"><strong>Error:</strong> {html.escape(str(error))}</div>' if error else ''}
            </div>

            <details class="tool-section">
                <summary>Tool Calls ({len(entry.get("tool_calls", []))})</summary>
                {tool_calls_html if tool_calls_html else "<p>No tool calls</p>"}
            </details>

            <details class="responses-section">
                <summary>Model Responses ({len(entry.get("model_responses", []))})</summary>
                {responses_html if responses_html else "<p>No responses logged</p>"}
            </details>
        </div>
        """)

    # Summary stats
    total = len(entries)
    correct = sum(1 for e in entries if e.get("exact_match"))
    errors = sum(1 for e in entries if e.get("error"))
    avg_f1 = sum(e.get("f1", 0) for e in entries) / total if total else 0
    total_tools = sum(len(e.get("tool_calls", [])) for e in entries)

    page = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>SimpleVQA Test Report</title>
    <style>
        body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
               background: #f5f5f5; margin: 0; padding: 20px; }}
        .container {{ max-width: 1200px; margin: 0 auto; }}
        h1 {{ color: #333; }}

        .summary {{ background: #fff; padding: 20px; border-radius: 8px; margin-bottom: 20px;
                   display: flex; gap: 30px; box-shadow: 0 1px 3px rgba(0,0,0,0.1); }}
        .summary-stat {{ text-align: center; }}
        .summary-stat .value {{ font-size: 2em; font-weight: bold; color: #2563eb; }}
        .summary-stat .label {{ color: #666; font-size: 0.9em; }}

        .sample {{ background: #fff; border-radius: 8px; padding: 20px; margin-bottom: 16px;
                  box-shadow: 0 1px 3px rgba(0,0,0,0.1); }}
        .sample-image {{ margin: 12px 0; text-align: center; }}
        .sample-image img {{ max-width: 400px; max-height: 400px; border-radius: 8px;
                            border: 1px solid #e5e7eb; }}
        .sample-header {{ display: flex; align-items: center; gap: 12px; margin-bottom: 16px;
                         padding-bottom: 12px; border-bottom: 1px solid #eee; }}
        .sample-id {{ font-weight: bold; font-size: 1.1em; }}
        .status {{ padding: 4px 12px; border-radius: 4px; font-size: 0.85em; font-weight: 500; }}
        .status.correct {{ background: #dcfce7; color: #166534; }}
        .status.incorrect {{ background: #fef3c7; color: #92400e; }}
        .status.error {{ background: #fee2e2; color: #991b1b; }}
        .metrics {{ color: #666; margin-left: auto; }}

        .qa-section {{ margin-bottom: 16px; }}
        .qa-section > div {{ margin-bottom: 8px; line-height: 1.5; }}
        .question {{ color: #1e40af; }}
        .gt-answer {{ color: #166534; }}
        .prediction {{ color: #7c3aed; }}
        .error-msg {{ color: #dc2626; }}

        details {{ margin-top: 12px; }}
        summary {{ cursor: pointer; font-weight: 500; color: #4b5563; padding: 8px;
                  background: #f9fafb; border-radius: 4px; }}
        summary:hover {{ background: #f3f4f6; }}

        .tool-call {{ background: #fffbeb; border: 1px solid #fcd34d; border-radius: 6px;
                     padding: 12px; margin: 12px 0; }}
        .tool-header {{ font-weight: bold; color: #92400e; margin-bottom: 8px; }}
        .tool-section {{ margin-top: 8px; }}
        .full-result {{ margin-top: 12px; }}
        .full-result summary {{ background: #fef3c7; }}
        .full-result pre {{ max-height: 500px; overflow-y: auto; }}

        .response {{ background: #f0fdf4; border: 1px solid #86efac; border-radius: 6px;
                    padding: 12px; margin: 12px 0; }}
        .response-header {{ font-weight: bold; color: #166534; margin-bottom: 8px; }}

        pre {{ background: #f8fafc; padding: 12px; border-radius: 4px; overflow-x: auto;
              font-size: 0.85em; white-space: pre-wrap; word-wrap: break-word; margin: 8px 0; }}
    </style>
</head>
<body>
    <div class="container">
        <h1>SimpleVQA Test Report</h1>

        <div class="summary">
            <div class="summary-stat">
                <div class="value">{correct}/{total}</div>
                <div class="label">Exact Match</div>
            </div>
            <div class="summary-stat">
                <div class="value">{avg_f1:.3f}</div>
                <div class="label">Avg F1</div>
            </div>
            <div class="summary-stat">
                <div class="value">{total_tools}</div>
                <div class="label">Total Tool Calls</div>
            </div>
            <div class="summary-stat">
                <div class="value">{errors}</div>
                <div class="label">Errors</div>
            </div>
        </div>

        {''.join(rows)}
    </div>
</body>
</html>
"""

    with open(output_path, "w") as f:
        f.write(page)
    print(f"Report saved to {output_path}")


def print_text(entries: list[dict]):
    for entry in entries:
        print("=" * 70)
        print(f"Sample {entry.get('sample_id', '?')}")
        print(f"  Question: {entry.get('question', '')}")
        print(f"  GT Answer: {entry.get('gt_answer', '')}")
        print(f"  Prediction: {entry.get('prediction', '')[:200]}")
        print(f"  F1: {entry.get('f1', 0):.3f}, EM: {entry.get('exact_match', False)}")

        if entry.get("error"):
            print(f"  ERROR: {entry.get('error')}")

        print(f"\n  Tool Calls ({len(entry.get('tool_calls', []))}):")
        for tc in entry.get("tool_calls", []):
            print(f"    - {tc.get('tool_name')}: {json.dumps(tc.get('arguments', {}))}")
            result = tc.get("result", {})
            if "results" in result:
                print(f"      Results: {len(result.get('results', []))} items")
            if "error" in result:
                print(f"      Error: {result.get('error')}")
        print()


def main():
    parser = argparse.ArgumentParser(description="View detailed log as HTML or text")
    parser.add_argument("logfile", help="Path to detailed_log.jsonl")
    parser.add_argument("-o", "--output", help="Output HTML file (omit for text output)")
    args = parser.parse_args()

    entries = load_jsonl(args.logfile)

    if args.output:
        render_html(entries, args.output)
    else:
        print_text(entries)


if __name__ == "__main__":
    main()
