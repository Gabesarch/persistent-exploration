"""Explicit diagnostic worker extension; never activated by importing Gemma support."""

import hashlib

import torch


class Gemma4SleepDiagnosticWorker:
    def gemma_snapshot(self):
        model = self.model_runner.model
        constants = {}
        for name, module in model.named_modules():
            if type(module).__name__ == "RMSNorm" and not module.has_weight:
                weight = module.weight.detach().cpu()
                constants[name] = {
                    "registered": "weight" in module._buffers,
                    "dtype": str(weight.dtype),
                    "finite": bool(weight.isfinite().all()),
                    "all_ones": bool((weight == 1).all()),
                    "first": weight.flatten()[:8].float().tolist(),
                }
        digest = hashlib.sha256()
        for name, param in sorted(model.named_parameters()):
            digest.update(name.encode())
            digest.update(param.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes())
        return {"constants": constants, "parameter_sha256": digest.hexdigest()}

    def gemma_register_constants(self):
        """Register constant norm weights so native vLLM sleep saves/restores them."""
        names = []
        model = self.model_runner.model
        if model.config.model_type != "gemma4":
            raise ValueError("This diagnostic is only for Gemma4")
        for name, module in model.named_modules():
            if type(module).__name__ == "RMSNorm" and not module.has_weight:
                # Reset after the deliberately unprotected sleep experiment.
                module.weight.fill_(1)
                if "weight" not in module._buffers:
                    weight = module.weight
                    del module.weight
                    module.register_buffer("weight", weight, persistent=False)
                names.append(name)
        return names

    def gemma_reload_source(self, path):
        """Replay source tensors in buckets, with the same post-load step as training."""
        from safetensors import safe_open
        from vllm.model_executor.model_loader.utils import process_weights_after_loading

        model = self.model_runner.model
        count, bucket, size = 0, [], 0
        with safe_open(path, framework="pt", device="cpu") as source:
            for name in source.keys():
                tensor = source.get_tensor(name)
                if size + tensor.nbytes > 6144 * 1024**2 and bucket:
                    model.load_weights(bucket)
                    torch.cuda.synchronize()
                    bucket, size = [], 0
                bucket.append((name, tensor.to(self.device)))
                size += tensor.nbytes
                count += 1
            if bucket:
                model.load_weights(bucket)
                torch.cuda.synchronize()
        process_weights_after_loading(model, self.model_runner.vllm_config.model_config, self.device)
        return {"source_tensors": count}
