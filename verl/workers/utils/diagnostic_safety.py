"""Failure isolation for optional, read-only training diagnostics."""

import logging

import torch

logger = logging.getLogger(__name__)


def diagnostic_call(function, *, device, phase):
    """Run local work, then agree on success BEFORE the next collective.

    The callback must not contain distributed collectives. This protects against
    rank-local CPU/statistics/filesystem errors without letting peers enter a
    different collective. Errors in model/CUDA/NCCL execution are not recoverable
    by this helper and must retain the training framework's handling.
    """
    result, message = None, None
    try:
        result = function()
    except Exception as error:
        message = f"{type(error).__name__}: {error}"
        logger.warning("Optional diagnostic %s failed: %s", phase, message)
    failed = torch.tensor(int(message is not None), device=device, dtype=torch.int32)
    if torch.distributed.is_initialized():
        torch.distributed.all_reduce(failed, op=torch.distributed.ReduceOp.SUM)
    if failed.item():
        return None, {"phase": phase, "message": message or "Failed on another data-parallel rank",
                      "failed_ranks": int(failed.item())}
    return result, None
