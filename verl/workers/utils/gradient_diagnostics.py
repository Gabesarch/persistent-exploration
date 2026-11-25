"""Full-parameter gradient Gram matrices without gathering FSDP parameters.

Five local gradient shards live temporarily on CPU. Scalar inner products are
reduced globally AFTER FSDP's gradient reduction, not norms of local gradients
before data-parallel averaging. Replicated parameter dimensions count once.
"""

import itertools
import json
import logging
import math
import random
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import torch
from torch.distributed.tensor import DTensor

from verl.workers.utils.diagnostic_safety import diagnostic_call
from verl.workers.utils.loss_diagnostics import COMPONENTS

logger = logging.getLogger(__name__)


def gradient_step_selected(config, global_step, first_update):
    if not config.get("enabled", False) or not config.get("gradient_enabled", False):
        return False
    if global_step is None:
        raise ValueError("Gradient diagnostics require global_steps metadata.")
    return (first_update and config.gradient_first_update) or int(global_step) % config.gradient_every_n_steps == 0


@contextmanager
def preserved_rng(device):
    python_state, numpy_state = random.getstate(), np.random.get_state()
    devices = [torch.cuda.current_device()] if torch.device(device).type == "cuda" else []
    try:
        with torch.random.fork_rng(devices=devices):
            yield
    finally:
        random.setstate(python_state)
        np.random.set_state(numpy_state)


def _local(tensor):
    return tensor.to_local() if isinstance(tensor, DTensor) else tensor


def _replication_weight(parameter):
    world = torch.distributed.get_world_size() if torch.distributed.is_initialized() else 1
    if not isinstance(parameter, DTensor):
        return 1.0 / world
    if parameter.device_mesh.size() != world:
        raise ValueError("Gradient diagnostics require a parameter mesh spanning the full actor world.")
    copies = 1
    for dim, placement in enumerate(parameter.placements):
        if placement.is_replicate():
            copies *= parameter.device_mesh.size(dim)
        elif not placement.is_shard():
            raise ValueError(f"Unsupported gradient placement: {placement}")
    return 1.0 / copies


class GradientProbe:
    def __init__(self, module, device, reconstruction_rtol=0.02):
        self.module = module
        self.device = device
        self.rtol = reconstruction_rtol
        self.names = [name for name, p in module.named_parameters() if p.requires_grad]
        self.shards = []
        self.metrics = {}
        self.gram = None
        self.losses = []
        self.storage_bytes = 0
        self.errors = []
        self.gradient_dtypes = set()
        self.metrics["valid"] = 1.0

    def invalidate(self, phase, message):
        """Quality failures are telemetry, never a veto of a training update."""
        self.metrics["valid"] = 0.0
        self.errors.append({"phase": phase, "message": str(message)})
        logger.warning("Optional gradient diagnostic %s: %s; training continues", phase, message)

    def local_call(self, function, phase):
        result, error = diagnostic_call(function, device=self.device, phase=phase)
        if error is not None:
            self.errors.append(error)
            self.metrics["valid"] = 0.0
            self.metrics["technical_error"] = 1.0
        return result, error is None

    def _parameters(self):
        params = [(name, p) for name, p in self.module.named_parameters() if p.requires_grad]
        if [name for name, _ in params] != self.names:
            raise RuntimeError("Trainable parameter inventory changed during gradient replay.")
        return [p for _, p in params]

    @torch.no_grad()
    def capture(self, loss_total, scale=1.0):
        """Capture an already reduced gradient, without touching the optimizer."""
        snapshot = []
        for p in self._parameters():
            if p.grad is None:
                snapshot.append(None)
            else:
                self.gradient_dtypes.add(str(p.grad.dtype))
                value = _local(p.grad).detach().to(device="cpu", copy=True).reshape(-1)
                # Preserve fp64 in analytic tests; use fp32 for other dtypes.
                if value.dtype != torch.float64:
                    value = value.float()
                value.div_(scale)
                snapshot.append(value)
                self.storage_bytes += value.numel() * value.element_size()
        self.shards.append(snapshot)
        self.losses.append(float(loss_total))

    @torch.no_grad()
    def _local_gram(self, scale):
        if len(self.shards) != len(COMPONENTS):
            raise RuntimeError("Five replay gradient snapshots are required.")
        gram = torch.zeros(6, 6, dtype=torch.float64)
        # Bound temporary RAM, including embeddings; never build a flat full model.
        chunk_size = 262144
        for index, p in enumerate(self._parameters()):
            n = _local(p).numel()
            actual = None if p.grad is None else _local(p.grad).detach().to(device="cpu").reshape(-1)
            parts = [snapshot[index] for snapshot in self.shards]
            for part in parts + [actual]:
                if part is not None and part.numel() != n:
                    raise RuntimeError("Parameter/gradient shard sizes changed during replay.")
            weight = _replication_weight(p)
            for start in range(0, n, chunk_size):
                count = min(chunk_size, n - start)
                rows = [torch.zeros(count, dtype=torch.float64) if v is None
                        else v[start:start + count].double() for v in parts + [actual]]
                rows[-1] = rows[-1] / scale
                block = torch.stack(rows)
                gram.addmm_(block, block.T, alpha=weight)
        return gram

    @torch.no_grad()
    def measure_before_clip(self, scale=1.0):
        gram, ok = self.local_call(lambda: self._local_gram(scale), "local_gram")
        # Release CPU snapshots on success AND failure. Never retain these until
        # the next optimizer minibatch after a failed optional measurement.
        self.shards.clear()
        if not ok:
            self.metrics["reconstruction_valid"] = 0.0
            return self.metrics
        gram = gram.to(self.device)
        if torch.distributed.is_initialized():
            torch.distributed.all_reduce(gram)
        gram = gram.cpu()
        self.gram = gram
        if not torch.isfinite(gram).all():
            self.metrics["finite"] = 0.0
            self.metrics["reconstruction_valid"] = 0.0
            self.invalidate("gram", "Nonfinite gradient Gram matrix")
            return self.metrics
        self.metrics["finite"] = 1.0
        self.metrics["uses_bfloat16_gradients"] = float("torch.bfloat16" in self.gradient_dtypes)
        before_summary = dict(self.metrics)
        _, ok = self.local_call(self._summarize_gram, "summarize_gram")
        if not ok:
            # A callback may have populated only some fields before failing on
            # one rank. Keep the metric inventory identical on every rank.
            self.metrics = {**before_summary, "valid": 0.0, "technical_error": 1.0,
                            "reconstruction_valid": 0.0}
        return self.metrics

    def _summarize_gram(self):
        gram = self.gram
        vectors = {name: torch.eye(6, dtype=torch.float64)[i] for i, name in enumerate(COMPONENTS)}
        vectors["canonical"] = vectors["canonical_positive"] + vectors["canonical_negative"]
        vectors["branch"] = vectors["branch_positive"] + vectors["branch_negative"]
        vectors["combined"] = torch.eye(6, dtype=torch.float64)[5]

        def dot(a, b):
            return float(a @ gram @ b)

        norms = {name: math.sqrt(max(0.0, dot(v, v))) for name, v in vectors.items()}
        self.metrics.update({f"{name}_norm_preclip": value for name, value in norms.items()})
        for a, b in itertools.combinations(vectors, 2):
            denom = norms[a] * norms[b]
            self.metrics[f"{a}__{b}_cosine_defined"] = float(denom > 0)
            self.metrics[f"{a}__{b}_cosine"] = (
                max(-1.0, min(1.0, dot(vectors[a], vectors[b]) / denom)) if denom > 0 else 0.0
            )
        total_sq = norms["combined"] ** 2
        for name, vector in vectors.items():
            self.metrics[f"{name}_projection_on_combined"] = (
                dot(vector, vectors["combined"]) / total_sq if total_sq > 0 else 0.0
            )
        self.metrics["projection_on_combined_defined"] = float(total_sq > 0)
        reconstructed = sum(vectors[k] for k in COMPONENTS)
        residual = reconstructed - vectors["combined"]
        error = math.sqrt(max(0.0, dot(residual, residual)))
        # Expose the unassigned residual explicitly rather than forcing the
        # five projections to sum to one by normalization or inventing a term.
        self.metrics["residual_norm_preclip"] = error
        self.metrics["residual_projection_on_combined"] = (
            dot(-residual, vectors["combined"]) / total_sq if total_sq > 0 else 0.0
        )
        component_norm_sum = sum(norms[k] for k in COMPONENTS)
        self.metrics["reconstruction_abs_error"] = error
        self.metrics["reconstruction_relative_to_component_norm_sum"] = error / max(component_norm_sum, 1e-30)
        self.metrics["reconstruction_relative_to_combined"] = error / max(norms["combined"], 1e-30)
        self.metrics["reconstructed_combined_norm"] = math.sqrt(max(0.0, dot(reconstructed, reconstructed)))
        self.metrics["cpu_gradient_storage_bytes_per_rank"] = float(self.storage_bytes)
        self.metrics["branch_to_canonical_norm_defined"] = float(norms["canonical"] > 0)
        for term in ("branch", "critic_ce"):
            self.metrics[f"{term}_to_canonical_norm_ratio"] = (
                norms[term] / norms["canonical"] if norms["canonical"] > 0 else 0.0
            )
        self.metrics["reconstruction_tolerance"] = self.rtol
        self.metrics["reconstruction_valid"] = float(error <= 1e-7 + self.rtol * component_norm_sum)
        if not self.metrics["reconstruction_valid"]:
            self.invalidate(
                "reconstruction", f"Independent backward residual {error / max(component_norm_sum, 1e-30):.6g} "
                f"exceeds quality tolerance {self.rtol:.6g}",
            )
        return self.metrics

    @torch.no_grad()
    def _local_postclip_sq(self):
        actual_sq = torch.zeros((), dtype=torch.float64, device=self.device)
        for p in self._parameters():
            if p.grad is not None:
                local = _local(p.grad).detach()
                # Accumulate each tensor norm in fp64, without copying a model.
                actual_sq += local.norm(dtype=torch.float64).square().to(self.device) * _replication_weight(p)
        return actual_sq

    @torch.no_grad()
    def after_clip(self, clip_reported_norm, max_norm):
        """Read real clipped gradients. A failed audit must not skip optimizer.step()."""
        actual_sq, ok = self.local_call(self._local_postclip_sq, "local_postclip_norm")
        if not ok:
            self.metrics["clipping_valid"] = 0.0
            return
        if torch.distributed.is_initialized():
            torch.distributed.all_reduce(actual_sq)
        post = math.sqrt(float(actual_sq))
        if not math.isfinite(post) or not math.isfinite(float(clip_reported_norm)):
            self.metrics["clipping_valid"] = 0.0
            self.invalidate(
                "clipping", "Nonfinite clipping diagnostic; optimizer retains its nonfinite-gradient handling",
            )
            return
        factor = min(1.0, float(max_norm) / (float(clip_reported_norm) + 1e-6))
        self.metrics["clip_multiplier"] = factor
        self.metrics["combined_norm_postclip_measured"] = post
        pre = self.metrics.get("combined_norm_preclip")
        if pre is None:
            self.metrics["clipping_valid"] = 0.0
            return
        self.metrics["clip_reported_norm_relative_error"] = abs(float(clip_reported_norm) - pre) / max(pre, 1e-30)
        self.metrics["clip_post_norm_relative_error"] = abs(post - pre * factor) / max(pre * factor, 1e-30)
        for key, value in list(self.metrics.items()):
            if key.endswith("_norm_preclip"):
                self.metrics[key.replace("_norm_preclip", "_norm_postclip_effective")] = value * factor
        self.metrics["clipping_valid"] = float(self.metrics["clip_reported_norm_relative_error"] <= 0.02
                                               and self.metrics["clip_post_norm_relative_error"] <= 0.02)
        if not self.metrics["clipping_valid"]:
            self.invalidate("clipping", "Measured clipping norms disagree with the optimizer's clip multiplier")

    def compare_losses(self, actual_losses):
        valid = len(self.losses) == len(COMPONENTS) and bool(actual_losses.get("reconstruction_valid", 1))
        if len(self.losses) == len(COMPONENTS):
            valid = valid and all(
                math.isfinite(self.losses[i])
                and abs(self.losses[i] - actual_losses[name]) <= 1e-6 + 1e-5 * abs(actual_losses[name])
                for i, name in enumerate(COMPONENTS)
            )
        self.metrics["replay_loss_valid"] = float(valid)
        if not valid:
            self.invalidate("replay_loss", "Replayed component losses do not match the actual loss breakdown")

    def write_record(self, output_dir, metadata):
        if not output_dir:
            return

        def write():
            if torch.distributed.is_initialized() and torch.distributed.get_rank() != 0:
                return
            path = Path(output_dir)
            path.mkdir(parents=True, exist_ok=True)
            record = {**metadata, "component_order": list(COMPONENTS) + ["actual_combined"],
                      "gram_preclip": self.gram.tolist() if self.gram is not None else None,
                      "metrics": self.metrics, "diagnostic_errors": self.errors,
                      "gradient_method": "independent_backward_same_training_precision",
                      "diagnostic_version": 2,
                      "gradient_dtypes": sorted(self.gradient_dtypes),
                      "quality_valid": bool(self.metrics.get("valid", 0))}

            def json_safe(value):
                if isinstance(value, float) and not math.isfinite(value):
                    return None
                if isinstance(value, dict):
                    return {k: json_safe(v) for k, v in value.items()}
                if isinstance(value, list):
                    return [json_safe(v) for v in value]
                return value

            line = json.dumps(json_safe(record), allow_nan=False) + "\n"
            with (path / f"step_{int(metadata['global_step']):06d}.jsonl").open("a") as stream:
                stream.write(line)

        _, ok = self.local_call(write, "write_record")
        self.metrics["record_write_failed"] = float(not ok)
