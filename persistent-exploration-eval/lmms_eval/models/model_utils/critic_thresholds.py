"""Per-domain evaluation cutoffs, shared by submission tags and request routing."""

import argparse
import json
import math

DOMAINS = ("visual_probe", "hstar", "pinpoint", "screenspot")


def validate_threshold(value):
    if isinstance(value, bool):
        raise ValueError("Critic thresholds must be numbers, not booleans")
    value = float(value)
    if not math.isfinite(value) or not 0 <= value <= 1:
        raise ValueError("Critic thresholds must be finite numbers in [0, 1]")
    return value


def parse_domain_thresholds(raw):
    values = json.loads(raw) if raw else {}
    if not isinstance(values, dict) or set(values) - set(DOMAINS):
        raise ValueError(f"Critic domain thresholds must be an object with keys from {DOMAINS}")
    return {key: validate_threshold(value) for key, value in values.items()}


def threshold_for_task(task, thresholds, fallback):
    for prefix, domain in (
        ("visual_probe_", "visual_probe"), ("hstarbench_", "hstar"),
        ("pinpoint_bench_", "pinpoint"), ("screenspotpro_", "screenspot"),
        ("screenspot_", "screenspot"),
    ):
        if task.startswith(prefix):
            return thresholds.get(domain, fallback)
    return fallback


def threshold_tag(thresholds, fallback):
    def token(value):
        return str(validate_threshold(value)).replace(".", "p").replace("-", "m")

    parts = [f"{short}{token(thresholds[domain])}" for short, domain in
             zip(("vp", "hs", "pp", "ss"), DOMAINS, strict=True)]
    return "critic_" + "_".join(parts) + "_fb" + token(fallback)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--fallback", required=True)
    for domain in DOMAINS:
        parser.add_argument("--" + domain.replace("_", "-"), required=True)
    args = parser.parse_args()
    thresholds = {domain: validate_threshold(getattr(args, domain)) for domain in DOMAINS}
    print(json.dumps(thresholds, sort_keys=True, separators=(",", ":")))
    print(threshold_tag(thresholds, args.fallback))
