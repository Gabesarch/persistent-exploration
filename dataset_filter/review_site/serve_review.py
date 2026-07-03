#!/usr/bin/env python
"""Serve the PLAUSIBLE review site AND accept decision uploads.

Reviewers with Della access run their own copy from the review-site folder and forward
the port to their laptop; clicking "Submit to Della" on a page POSTs their decisions here,
which writes exports/review_<reviewer-slug>.json straight to scratch — no download + manual
re-upload. (The page's "Export JSON" download still works as an offline fallback.)

  cd /scratch/.../full_sync/manual_review
  python serve_review.py               # prints a localhost URL + the port to forward
  # VS Code (Remote-SSH to della): Ports -> Forward the printed port, open the URL,
  # click your name, review, click "Submit to Della".

Security: bound to localhost (reachable only through the reviewer's own SSH tunnel); the
only write endpoint is POST /submit, and the target filename is `review_<slug>.json` where
slug is sanitized to [a-z0-9-], so a submit can never write outside exports/. Dependency-free
(stdlib only).
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

MAX_BODY = 32 * 1024 * 1024  # 32 MB cap on an upload (a 200-card export is ~tens of KB)


def slugify(name) -> str:
    return re.sub(r"[^a-z0-9]+", "-", str(name).strip().lower()).strip("-")


class Handler(SimpleHTTPRequestHandler):
    def _json(self, code, obj):
        body = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self):
        if self.path.split("?", 1)[0].rstrip("/") != "/submit":
            return self._json(404, {"ok": False, "error": "unknown endpoint"})
        try:
            n = int(self.headers.get("Content-Length", 0))
        except ValueError:
            n = 0
        if n <= 0 or n > MAX_BODY:
            return self._json(413, {"ok": False, "error": "missing or too-large body"})
        try:
            payload = json.loads(self.rfile.read(n).decode("utf-8"))
        except Exception as e:
            return self._json(400, {"ok": False, "error": f"invalid JSON: {e}"})
        slug = slugify(payload.get("reviewer"))
        if not re.fullmatch(r"[a-z0-9-]+", slug or ""):
            return self._json(400, {"ok": False, "error": "missing/invalid reviewer name"})

        exports = Path(self.directory) / "exports"
        exports.mkdir(exist_ok=True)
        out = exports / f"review_{slug}.json"
        out.write_text(json.dumps(payload, indent=2))
        try:
            os.chmod(out, 0o664)  # group-readable so the coordinator can collect it
        except OSError:
            pass
        n_rev = payload.get("n_reviewed")
        print(f"[serve] wrote {out}  (n_reviewed={n_rev})")
        return self._json(200, {"ok": True, "file": f"exports/{out.name}", "n_reviewed": n_rev})

    def log_message(self, *a):  # keep the console to submit lines only
        pass


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--port", type=int, default=8000,
                    help="preferred port (tries the next 9 if busy; default 8000)")
    ap.add_argument("--dir", default=".",
                    help="review-site folder to serve (default: current dir)")
    ap.add_argument("--host", default="127.0.0.1",
                    help="bind address (default localhost — keep it so access is SSH-tunnel-only)")
    args = ap.parse_args()

    root = Path(args.dir).resolve()
    if not (root / "index.html").exists():
        sys.exit(f"[serve] no index.html in {root} — run from the review-site folder or pass --dir")

    exports = root / "exports"
    exports.mkdir(exist_ok=True)
    try:
        os.chmod(exports, 0o2775)  # setgid + group-writable: submits land group-readable
    except OSError:
        pass

    def handler(*a, **k):
        return Handler(*a, directory=str(root), **k)

    httpd = last = None
    for port in range(args.port, args.port + 10):
        try:
            httpd = ThreadingHTTPServer((args.host, port), handler)
            break
        except OSError as e:
            last = e
    if httpd is None:
        sys.exit(f"[serve] no free port in {args.port}-{args.port + 9}: {last}")

    print(f"[serve] serving {root}")
    print(f"[serve] open  http://localhost:{port}/   (forward port {port} in VS Code)")
    print("[serve] 'Submit to Della' -> exports/review_<name>.json ; Ctrl-C to stop")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\n[serve] stopped")


if __name__ == "__main__":
    main()
