"""Opt-in actor CPU/CUDA profiling. No profiler or hooks run by default."""

import contextlib
import contextvars
import cProfile
import functools
import io
import json
import os
import pstats
import time
from pathlib import Path

import torch

_active = contextvars.ContextVar("gemma4_actor_profile", default=False)


def stage(name):
    return torch.profiler.record_function("gemma4/" + name) if _active.get() else contextlib.nullcontext()


def actor_stage(function):
    @functools.wraps(function)
    def wrapped(*args, **kwargs):
        with stage(function.__name__):
            return function(*args, **kwargs)

    return wrapped


def profile_actor(function):
    @functools.wraps(function)
    def wrapped(self, *args, **kwargs):
        if not self.config.fsdp_config.get("gemma4_profile_actor", False):
            return function(self, *args, **kwargs)
        directory = Path(os.environ["GEMMA4_ARTIFACT_DIR"]) / "actor_profiles"
        directory.mkdir(parents=True, exist_ok=True)
        rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
        key = "_gemma4_profile_" + function.__name__
        index = getattr(self, key, 0) + 1
        setattr(self, key, index)
        stem = directory / f"{function.__name__}_rank{rank}_call{index}"
        activities = [torch.profiler.ProfilerActivity.CPU]
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            activities.append(torch.profiler.ProfilerActivity.CUDA)
        if (
            function.__name__ == "update_policy"
            and index == 1
            and self.config.fsdp_config.get("gemma4_benchmark_padding", False)
        ):
            from .padding_benchmark import benchmark_padding

            benchmark_padding(self, args[0] if args else kwargs["data"])
        started = time.perf_counter()
        token = _active.set(True)
        python_profile = cProfile.Profile()
        try:
            with torch.profiler.profile(activities=activities, record_shapes=True) as profiler:
                with python_profile, stage(function.__name__):
                    result = function(self, *args, **kwargs)
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
        finally:
            _active.reset(token)
        elapsed = time.perf_counter() - started
        profiler.export_chrome_trace(str(stem) + ".trace.json.gz")
        python_profile.dump_stats(str(stem) + ".pstats")
        stream = io.StringIO()
        pstats.Stats(python_profile, stream=stream).sort_stats("cumulative").print_stats(100)
        pstats.Stats(python_profile, stream=stream).sort_stats("tottime").print_stats(100)
        Path(str(stem) + ".python.txt").write_text(stream.getvalue())
        events = profiler.key_averages()
        rows = [
            dict(
                name=e.key,
                calls=e.count,
                cpu_total_ms=e.cpu_time_total / 1000,
                cpu_self_ms=e.self_cpu_time_total / 1000,
                device_total_ms=e.device_time_total / 1000,
                device_self_ms=e.self_device_time_total / 1000,
            )
            for e in events
        ]
        report = dict(
            wall_seconds=elapsed,
            rank=rank,
            call=index,
            phase=function.__name__,
            events=rows,
            note="Profiling adds overhead. Nested totals overlap; use self times for additive attribution.",
        )
        Path(str(stem) + ".json").write_text(json.dumps(report, indent=2) + "\n")
        Path(str(stem) + ".txt").write_text(events.table(sort_by="self_device_time_total", row_limit=100))
        print(f"[Gemma4 profile] {stem}: {elapsed:.3f}s including profiler overhead", flush=True)
        return result

    return wrapped


def install_module_ranges(model):
    """Record major module forwards, including gradient-checkpoint recomputation."""
    classes = {
        "Gemma4TextModel",
        "Gemma4VisionModel",
        "Gemma4TextDecoderLayer",
        "Gemma4VisionEncoderLayer",
        "Gemma4TextAttention",
        "Gemma4TextMLP",
    }
    for name, module in model.named_modules():
        if type(module).__name__ not in classes:
            continue
        stack = []

        def pre(module, args, name=name, stack=stack):
            scope = stage("module/" + name)
            scope.__enter__()
            stack.append(scope)

        def post(module, args, output, stack=stack):
            stack.pop().__exit__(None, None, None)

        module.register_forward_pre_hook(pre)
        module.register_forward_hook(post, always_call=True)
