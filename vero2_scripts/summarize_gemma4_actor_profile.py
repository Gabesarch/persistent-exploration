"""Summarize profiled actor phases without double-counting nested CUDA times."""

import json
import sys
from pathlib import Path


def summarize(directory):
    profiles = []
    for path in sorted((directory / "actor_profiles").glob("*.json")):
        report = json.loads(path.read_text())
        events = report.pop("events")
        if not any(e["device_total_ms"] > 0 for e in events):
            raise RuntimeError(f"Missing CUDA timing events in {path}; CPU-only profiling is insufficient")
        report["trace"] = str(path.with_suffix(".trace.json.gz"))
        report["python_function_times"] = str(path.with_suffix(".python.txt"))
        report["stages"] = [
            event
            for event in events
            if event["name"].startswith("gemma4/") and not event["name"].startswith("gemma4/module/")
        ]
        report["top_modules_inclusive"] = sorted(
            [event for event in events if event["name"].startswith("gemma4/module/")],
            key=lambda e: e["device_total_ms"],
            reverse=True,
        )[:20]
        report["top_operators_self_device_time"] = sorted(events, key=lambda e: e["device_self_ms"], reverse=True)[:30]
        profiles.append(report)
    benchmarks = [json.loads(path.read_text()) for path in sorted(directory.glob("padding_benchmark_rank*.json"))]
    expected = {(rank, call) for rank in (0, 1) for call in (1, 2)}
    for phase in ("compute_log_prob", "update_policy"):
        actual = {(p["rank"], p["call"]) for p in profiles if p["phase"] == phase}
        if not expected <= actual:
            raise RuntimeError(f"Incomplete {phase} profiles: expected {expected}, found {actual}")
    if len(benchmarks) != 2:
        raise RuntimeError("Missing same-trajectory padded/trimmed benchmark on one or more ranks")
    summary = dict(
        profiles=profiles,
        paired_padding_benchmarks=benchmarks,
        interpretation="Inclusive module/stage times overlap. Self device times are additive operator work, "
        "not wall time. Profiles include instrumentation overhead and smoke CPU offload; "
        "production batch throughput is not established by this small smoke.",
    )
    output = directory / "actor_profile_summary.json"
    output.write_text(json.dumps(summary, indent=2) + "\n")
    print(f"Actor timing summary: {output}")
    for p in profiles:
        print(p["phase"], "rank", p["rank"], "call", p["call"], "wall_seconds", p["wall_seconds"])
        for event in p["stages"]:
            print(
                "  ",
                event["name"],
                "CPU ms",
                round(event["cpu_total_ms"], 2),
                "CUDA ms",
                round(event["device_total_ms"], 2),
                "calls",
                event["calls"],
            )


if __name__ == "__main__":
    summarize(Path(sys.argv[1]))
