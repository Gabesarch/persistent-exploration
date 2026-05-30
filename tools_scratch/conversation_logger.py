"""
HTML Conversation Logger for tool calling visualization.
Logs all model responses, tool calls, and tool results with embedded images.
"""
import base64
import html
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Optional


class ConversationLogger:
    def __init__(self, output_path: Optional[str] = None, title: str = "Tool Calling Conversation"):
        if output_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = f"conversation_{timestamp}.html"
        self.output_path = output_path
        self.title = title
        self.entries = []
        self.metadata = {}

    def set_metadata(self, **kwargs):
        """Set metadata like sample_idx, question, expected answer, etc."""
        self.metadata.update(kwargs)

    def log_user_message(self, content, images: list = None):
        """Log a user message with optional images."""
        self.entries.append({
            "type": "user",
            "content": content,
            "images": images or [],
        })

    def log_assistant_message(self, content: str, tool_calls: list = None):
        """Log assistant response with optional tool calls."""
        self.entries.append({
            "type": "assistant",
            "content": content,
            "tool_calls": tool_calls or [],
        })

    def log_tool_result(self, tool_name: str, tool_call_id: str, result: dict, image_b64: str = None):
        """Log tool execution result with optional image."""
        self.entries.append({
            "type": "tool",
            "tool_name": tool_name,
            "tool_call_id": tool_call_id,
            "result": result,
            "image_b64": image_b64,
        })

    def log_iteration(self, iteration: int):
        """Log iteration marker."""
        self.entries.append({
            "type": "iteration",
            "iteration": iteration,
        })

    def save(self):
        """Generate and save the HTML file."""
        html_content = self._generate_html()
        with open(self.output_path, "w", encoding="utf-8") as f:
            f.write(html_content)
        return self.output_path

    def _generate_html(self) -> str:
        css = """
        <style>
            * { box-sizing: border-box; }
            body {
                font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, sans-serif;
                max-width: 1200px;
                margin: 0 auto;
                padding: 20px;
                background: #f5f5f5;
                line-height: 1.6;
            }
            h1 { color: #333; border-bottom: 2px solid #007bff; padding-bottom: 10px; }
            .metadata {
                background: #e9ecef;
                padding: 15px;
                border-radius: 8px;
                margin-bottom: 20px;
            }
            .metadata h3 { margin-top: 0; }
            .metadata pre {
                background: #fff;
                padding: 10px;
                border-radius: 4px;
                white-space: pre-wrap;
                word-wrap: break-word;
                max-height: 300px;
                overflow-y: auto;
            }
            .iteration {
                background: #6c757d;
                color: white;
                padding: 8px 15px;
                border-radius: 4px;
                margin: 20px 0;
                font-weight: bold;
            }
            .message {
                margin: 15px 0;
                padding: 15px;
                border-radius: 8px;
                box-shadow: 0 1px 3px rgba(0,0,0,0.1);
            }
            .user {
                background: #e3f2fd;
                border-left: 4px solid #2196f3;
            }
            .assistant {
                background: #f3e5f5;
                border-left: 4px solid #9c27b0;
            }
            .tool {
                background: #e8f5e9;
                border-left: 4px solid #4caf50;
            }
            .role-label {
                font-weight: bold;
                text-transform: uppercase;
                font-size: 12px;
                margin-bottom: 10px;
                color: #666;
            }
            .user .role-label { color: #1565c0; }
            .assistant .role-label { color: #7b1fa2; }
            .tool .role-label { color: #2e7d32; }
            .content {
                white-space: pre-wrap;
                word-wrap: break-word;
                font-family: 'SF Mono', Monaco, 'Courier New', monospace;
                font-size: 14px;
                background: rgba(255,255,255,0.7);
                padding: 10px;
                border-radius: 4px;
            }
            .tool-calls {
                margin-top: 10px;
                padding: 10px;
                background: rgba(0,0,0,0.05);
                border-radius: 4px;
            }
            .tool-call {
                background: #fff3e0;
                padding: 10px;
                margin: 5px 0;
                border-radius: 4px;
                border-left: 3px solid #ff9800;
            }
            .tool-call-name {
                font-weight: bold;
                color: #e65100;
            }
            .tool-call-args {
                font-family: monospace;
                font-size: 13px;
                background: #fff;
                padding: 8px;
                border-radius: 4px;
                margin-top: 5px;
                overflow-x: auto;
            }
            .image-container {
                margin: 10px 0;
                text-align: center;
            }
            .image-container img {
                max-width: 100%;
                max-height: 600px;
                border: 1px solid #ddd;
                border-radius: 4px;
                box-shadow: 0 2px 8px rgba(0,0,0,0.1);
            }
            .image-label {
                font-size: 12px;
                color: #666;
                margin-top: 5px;
            }
            .final-answer {
                background: #fff9c4;
                border: 2px solid #fbc02d;
                padding: 15px;
                border-radius: 8px;
                margin-top: 20px;
            }
            .final-answer h3 { margin-top: 0; color: #f57f17; }
        </style>
        """

        body_parts = [f"<h1>{html.escape(self.title)}</h1>"]

        # Metadata section
        if self.metadata:
            body_parts.append('<div class="metadata">')
            body_parts.append('<h3>Sample Information</h3>')
            for key, value in self.metadata.items():
                if key == "question":
                    body_parts.append(f'<p><strong>{html.escape(key)}:</strong></p><pre>{html.escape(str(value))}</pre>')
                elif key == "image" and value:
                    body_parts.append(f'<p><strong>Input Image:</strong></p>')
                    body_parts.append(f'<div class="image-container"><img src="{value}" /></div>')
                else:
                    body_parts.append(f'<p><strong>{html.escape(key)}:</strong> {html.escape(str(value))}</p>')
            body_parts.append('</div>')

        # Conversation entries
        for entry in self.entries:
            if entry["type"] == "iteration":
                body_parts.append(f'<div class="iteration">Iteration {entry["iteration"]}</div>')

            elif entry["type"] == "user":
                body_parts.append('<div class="message user">')
                body_parts.append('<div class="role-label">User</div>')
                body_parts.append(f'<div class="content">{html.escape(entry["content"])}</div>')
                for img in entry.get("images", []):
                    body_parts.append(f'<div class="image-container"><img src="{img}" /><div class="image-label">User provided image</div></div>')
                body_parts.append('</div>')

            elif entry["type"] == "assistant":
                body_parts.append('<div class="message assistant">')
                body_parts.append('<div class="role-label">Assistant</div>')
                content = entry["content"] or "(no text content)"
                body_parts.append(f'<div class="content">{html.escape(content)}</div>')

                if entry.get("tool_calls"):
                    body_parts.append('<div class="tool-calls"><strong>Tool Calls:</strong>')
                    for tc in entry["tool_calls"]:
                        body_parts.append('<div class="tool-call">')
                        body_parts.append(f'<div class="tool-call-name">{html.escape(tc.get("name", "unknown"))}</div>')
                        args = tc.get("arguments", "{}")
                        try:
                            args_formatted = json.dumps(json.loads(args), indent=2)
                        except:
                            args_formatted = args
                        body_parts.append(f'<div class="tool-call-args">{html.escape(args_formatted)}</div>')
                        body_parts.append('</div>')
                    body_parts.append('</div>')
                body_parts.append('</div>')

            elif entry["type"] == "tool":
                body_parts.append('<div class="message tool">')
                body_parts.append(f'<div class="role-label">Tool Result: {html.escape(entry["tool_name"])}</div>')
                result_str = json.dumps(entry.get("result", {}), indent=2) if isinstance(entry.get("result"), dict) else str(entry.get("result", ""))
                body_parts.append(f'<div class="content">{html.escape(result_str)}</div>')

                if entry.get("image_b64"):
                    img_src = entry["image_b64"]
                    if not img_src.startswith("data:"):
                        img_src = f"data:image/png;base64,{img_src}"
                    body_parts.append(f'<div class="image-container"><img src="{img_src}" /><div class="image-label">Tool output image</div></div>')
                body_parts.append('</div>')

        return f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{html.escape(self.title)}</title>
    {css}
</head>
<body>
    {''.join(body_parts)}
</body>
</html>"""
