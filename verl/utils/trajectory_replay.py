"""Sanitized visual evidence and bounded, checkpointed trajectory-judge replay.

Only the training controller writes manifests. Rollout workers may atomically
publish immutable image assets before discarded answer branches are rolled back.
"""

from __future__ import annotations

import hashlib
import heapq
import json
import math
import os
import random
import re
from collections import defaultdict, deque
from collections.abc import Callable
from pathlib import Path
from typing import Any
from uuid import uuid4

from PIL import Image

DEFAULT_JUDGE_SYSTEM_PROMPT = (
    "Judge whether the candidate answer fully answers the original question correctly, "
    "using the supplied visual evidence. A means fully correct; R means incorrect or "
    "only partially correct. Reason before deciding. Output exactly one nonempty "
    "<think>...</think> block followed by exactly one <answer>A</answer> or "
    "<answer>R</answer> block. Use no tools and put no other text outside these blocks."
)
_RESPONSE = re.compile(
    r"\s*<think>(?P<think>.*?)</think>\s*<answer>\s*(?P<label>A|R)\s*</answer>\s*",
    flags=re.DOTALL,
)
_RESERVED_TAG = re.compile(r"</?(?:think|answer|tool_call|tool_response)(?:\s|>)")
_SOURCE_KEYS = {"request_id", "uid", "sample_index", "rollout_n", "candidate_ordinal", "data_source"}


def extract_candidate_answer(text: str) -> str | None:
    """Keep the last boxed answer, or the last answer body, never old reasoning."""
    bodies = re.findall(r"<answer>(.*?)</answer>", str(text), flags=re.DOTALL)
    if not bodies:
        return None
    answer = bodies[-1].strip()
    if not answer or _RESERVED_TAG.search(answer):
        return None
    boxes = []
    for match in re.finditer(r"\\boxed\s*\{", answer):
        depth = 1
        end = match.end()
        while end < len(answer) and depth:
            if answer[end] == "{" and (end == 0 or answer[end - 1] != "\\"):
                depth += 1
            elif answer[end] == "}" and (end == 0 or answer[end - 1] != "\\"):
                depth -= 1
            end += 1
        if depth == 0:
            boxes.append(answer[match.start() : end])
    return boxes[-1] if boxes else answer


def extract_question_text(messages: list[dict[str, Any]]) -> str | None:
    """Read only original user text; never assistant or tool history."""
    for message in messages:
        if not isinstance(message, dict) or message.get("role") != "user":
            continue
        content = message.get("content", "")
        if isinstance(content, str):
            text = content
        elif isinstance(content, list):
            text = "\n".join(str(item["text"]) for item in content if isinstance(item, dict) and "text" in item)
        else:
            continue
        if text.strip():
            return text.strip()
    return None


def select_candidate_evidence(
    initial_images: Any, tool_images: Any, candidate_turn: int, tool_call_count: int
) -> tuple[str, list[Any]]:
    """The sole initial-image exception is a direct, first-turn answer."""
    if int(candidate_turn) == 1 and int(tool_call_count) == 0:
        return "initial", list(initial_images or [])
    return "tool_outputs", list(tool_images or [])


def load_judge_system_prompt(value: str | os.PathLike | None = None) -> str:
    if value is None:
        return DEFAULT_JUDGE_SYSTEM_PROMPT
    value = os.path.expandvars(os.path.expanduser(str(value)))
    if "\n" not in value and len(value) < 4096:
        try:
            if Path(value).is_file():
                text = Path(value).read_text(encoding="utf-8").strip()
                if not text:
                    raise ValueError("Judge system prompt file is empty")
                return text
        except OSError:
            if value.startswith("/") or value.endswith(".txt"):
                raise
        if value.startswith("/") or value.endswith(".txt"):
            raise FileNotFoundError(f"Judge system prompt file does not exist: {value}")
    return value


def _image_block(image: Any) -> dict[str, Any]:
    if isinstance(image, Image.Image):
        value, width, height = image, image.width, image.height
    else:
        ref = image if isinstance(image, dict) else {"path": str(image)}
        value = str(ref["path"])
        if "width" in ref and "height" in ref:
            width, height = int(ref["width"]), int(ref["height"])
        else:
            with Image.open(value) as opened:
                width, height = opened.size
    # qwen_vl_utils uses these dimensions rather than silently resizing evidence.
    return {"type": "image", "image": value, "resized_width": width, "resized_height": height}


def build_judge_messages(
    question: str, answer: str, images: list[Any], system_prompt: str | os.PathLike | None = None
) -> list[dict[str, Any]]:
    content = [{"type": "text", "text": f"Original question:\n{question.strip()}\n\nVisual evidence:\n"}]
    content.extend(_image_block(image) for image in images)
    content.append({"type": "text", "text": f"\nCandidate answer:\n{answer.strip()}"})
    return [
        {"role": "system", "content": load_judge_system_prompt(system_prompt)},
        {"role": "user", "content": content},
    ]


def parse_judge_response(response_text: str) -> str | None:
    match = _RESPONSE.fullmatch(response_text)
    if not match or not match.group("think").strip() or _RESERVED_TAG.search(match.group("think")):
        return None
    return match.group("label")


def judge_rewards(
    response_text: str,
    label: str,
    generated_tokens: int,
    normal_budget: int = 4096,
    overlong_buffer: int = 1024,
    format_weight: float = 0.2,
) -> dict[str, Any]:
    if label not in {"A", "R"}:
        raise ValueError("Judge ground-truth label must be A or R")
    if normal_budget <= 0 or overlong_buffer <= 0 or not 0.0 < format_weight < 1.0:
        raise ValueError("Judge budgets must be positive and format_weight must lie in (0, 1)")
    if generated_tokens < 0:
        raise ValueError("generated_tokens must be nonnegative")
    answer_blocks = re.findall(r"<answer>(.*?)</answer>", response_text, flags=re.DOTALL)
    decision = None
    if response_text.count("<answer>") == response_text.count("</answer>") == len(answer_blocks) == 1:
        answer = answer_blocks[0].strip()
        if answer in {"A", "R"}:
            decision = answer
    raw_format = float(parse_judge_response(response_text) is not None)
    accuracy = float(decision == label)
    penalty = -max(0, int(generated_tokens) - int(normal_budget)) / int(overlong_buffer)
    adjusted_format = raw_format + penalty / format_weight
    return {
        "score": (1.0 - format_weight) * accuracy + format_weight * raw_format + penalty,
        "accuracy": accuracy,
        "format": adjusted_format,
        "raw_format": raw_format,
        "overlong_reward": penalty,
        "overlong": penalty < 0.0,
        "decision": decision,
        "format_valid": bool(raw_format),
    }


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    try:
        with temporary.open("w", encoding="utf-8") as stream:
            json.dump(value, stream, ensure_ascii=False, sort_keys=True, allow_nan=False)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _persist_image(root: Path, image: Any) -> dict[str, Any]:
    if isinstance(image, Image.Image):
        rgb = image.convert("RGB")
    else:
        path = image["path"] if isinstance(image, dict) else image
        with Image.open(path) as opened:
            rgb = opened.convert("RGB").copy()
    digest = hashlib.sha256(f"RGB:{rgb.width}:{rgb.height}:".encode() + rgb.tobytes()).hexdigest()
    path = root / "images" / f"{digest}.png"
    if not path.is_file():
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(f".{digest}.{uuid4().hex}.tmp")
        try:
            rgb.save(temporary, format="PNG")
            os.replace(temporary, path)
        finally:
            temporary.unlink(missing_ok=True)
    return {"path": str(path), "width": rgb.width, "height": rgb.height, "sha256": digest}


def persist_candidate(
    root: str | os.PathLike,
    *,
    question: str,
    answer: str,
    initial_images: Any,
    tool_images: Any,
    candidate_turn: int,
    tool_call_count: int,
    accuracy: float,
    source_step: int,
    source_ids: dict[str, Any] | None = None,
    entry_id: str | None = None,
) -> dict[str, Any] | None:
    """Worker-side capture; the controller later commits returned metadata."""
    if not question or not answer or _RESERVED_TAG.search(answer):
        return None
    try:
        accuracy = float(accuracy)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(accuracy) or not 0.0 <= accuracy <= 1.0 or int(candidate_turn) < 1:
        return None
    evidence_mode, selected = select_candidate_evidence(initial_images, tool_images, candidate_turn, tool_call_count)
    if not selected:
        return None
    root = Path(root).expanduser().resolve()
    source = {}
    for key, value in (source_ids or {}).items():
        if key not in _SOURCE_KEYS:
            continue
        if hasattr(value, "item"):
            value = value.item()
        if value is None or isinstance(value, str | int | float | bool):
            source[key] = value
    entry = {
        "question": str(question).strip(),
        "answer": str(answer).strip(),
        "images": [_persist_image(root, image) for image in selected],
        "label": "A" if accuracy >= 1.0 else "R",
        "accuracy": accuracy,
        "turn": int(candidate_turn),
        "source_step": int(source_step),
        "source_ids": source,
        "evidence_mode": evidence_mode,
    }
    entry["entry_id"] = (
        entry_id
        or hashlib.sha256(json.dumps(entry, sort_keys=True, ensure_ascii=False, allow_nan=False).encode()).hexdigest()
    )
    return entry


class ReplayCache:
    """Class-balanced FIFO storage with turn diversity and exact resume snapshots."""

    SNAPSHOT_NAME = "trajectory_replay.json"
    SCHEMA_VERSION = 1

    def __init__(self, root: str | os.PathLike, capacity: int = 8192, seed: int = 42):
        if isinstance(capacity, bool) or int(capacity) < 2 or int(capacity) % 2:
            raise ValueError("Replay capacity must be a positive even integer of at least two")
        self.root = Path(root).expanduser().resolve()
        self.capacity = int(capacity)
        self.seed = int(seed)
        self.entries: list[dict[str, Any]] = []
        self.step = 0
        self._snapshot_refs: dict[Path, tuple[int, int, int, set[str]]] = {}
        self.reload()

    @property
    def manifest_path(self) -> Path:
        return self.root / "manifest.json"

    def __len__(self) -> int:
        return len(self.entries)

    def _encode(self, entries: list[dict[str, Any]] | None = None) -> dict[str, Any]:
        serialized = []
        paths = {}
        for entry in self.entries if entries is None else entries:
            encoded = {**entry, "images": []}
            for image in entry["images"]:
                if image["path"] not in paths:
                    paths[image["path"]] = str(Path(image["path"]).resolve().relative_to(self.root))
                encoded["images"].append({**image, "path": paths[image["path"]]})
            serialized.append(encoded)
        return {
            "schema_version": self.SCHEMA_VERSION,
            "capacity": self.capacity,
            "seed": self.seed,
            "step": self.step,
            "entries": serialized,
        }

    def _decode(self, payload: dict[str, Any]) -> None:
        if payload.get("schema_version") != self.SCHEMA_VERSION:
            raise ValueError("Unsupported trajectory replay manifest schema")
        if int(payload["capacity"]) != self.capacity or int(payload["seed"]) != self.seed:
            raise ValueError("Replay resume capacity/seed differ from the saved cache")
        entries = []
        paths = {}
        for encoded in payload["entries"]:
            entry = {**encoded, "images": []}
            for image in encoded["images"]:
                if image["path"] not in paths:
                    path = (self.root / image["path"]).resolve()
                    if not path.is_relative_to(self.root / "images"):
                        raise ValueError("Replay image reference must belong to this cache's assets")
                    if not path.is_file():
                        raise FileNotFoundError(f"Missing persistent replay image: {path}")
                    paths[image["path"]] = str(path)
                entry["images"].append({**image, "path": paths[image["path"]]})
            entries.append(entry)
        self._validate_entries(entries)
        self.entries, self.step = entries, int(payload["step"])

    def _validate_entries(self, entries: list[dict[str, Any]]) -> None:
        seen = set()
        counts = {"A": 0, "R": 0}
        for entry in entries:
            if entry["entry_id"] in seen:
                raise ValueError("Replay manifest contains duplicate entry IDs")
            seen.add(entry["entry_id"])
            if entry["label"] not in counts:
                raise ValueError("Replay label must be A or R")
            counts[entry["label"]] += 1
            if not entry["question"] or not entry["answer"] or not entry["images"]:
                raise ValueError("Replay entries require question, answer and visual evidence")
        if max(counts.values()) > self.capacity // 2:
            raise ValueError("Replay manifest exceeds a label's capacity")

    def reload(self) -> None:
        if self.manifest_path.is_file():
            self._decode(json.loads(self.manifest_path.read_text(encoding="utf-8")))
        else:
            self.entries, self.step = [], 0

    def sample(
        self, count: int, step: int, *, eligible: Callable[[dict[str, Any]], bool] | None = None
    ) -> list[dict[str, Any]]:
        """Sample distinct balanced pairs, scanning past ineligible turn entries.

        ``eligible`` receives a detached entry and may attach prompt metadata.
        Class quotas apply after filtering, so sparse classes do not prevent
        searching the more populated class for valid replacement candidates.
        """
        by_label: dict[str, dict[int, list[dict[str, Any]]]] = {"A": defaultdict(list), "R": defaultdict(list)}
        for entry in self.entries:
            if int(entry["source_step"]) >= int(step):
                continue
            by_label[entry["label"]][int(entry["turn"])].append(entry)
        target_pairs = min(max(0, int(count)) // 2, *(sum(map(len, turns.values())) for turns in by_label.values()))
        rng = random.Random(f"trajectory-replay:{self.seed}:{int(step)}")
        selected = {"A": [], "R": []}
        for label in ("A", "R"):
            buckets = {turn: list(entries) for turn, entries in sorted(by_label[label].items())}
            turns = list(buckets)
            rng.shuffle(turns)
            for entries in buckets.values():
                rng.shuffle(entries)
            while len(selected[label]) < target_pairs and any(buckets.values()):
                for turn in turns:
                    if buckets[turn]:
                        entry = json.loads(json.dumps(buckets[turn].pop()))
                        if eligible is None or eligible(entry):
                            selected[label].append(entry)
                        if len(selected[label]) == target_pairs:
                            break
        pairs = min(len(selected["A"]), len(selected["R"]))
        sampled = selected["A"][:pairs] + selected["R"][:pairs]
        rng.shuffle(sampled)
        return sampled

    def commit(self, candidates: list[dict[str, Any]], step: int) -> None:
        if int(step) < self.step:
            raise ValueError("Replay cache cannot commit an earlier step without restoring its checkpoint")
        entries = {entry["entry_id"]: entry for entry in self.entries}
        for candidate in candidates:
            if candidate is not None:
                entries.setdefault(candidate["entry_id"], candidate)
        retained = list(entries.values())
        for label in ("A", "R"):
            rows = [entry for entry in retained if entry["label"] == label]
            age = lambda entry: (int(entry["source_step"]), entry["entry_id"])
            grouped = defaultdict(list)
            for entry in rows:
                grouped[int(entry["turn"])].append(entry)
            buckets = {turn: deque(sorted(values, key=age)) for turn, values in grouped.items()}
            heap = [(-len(values), age(values[0]), turn) for turn, values in buckets.items()]
            heapq.heapify(heap)
            remaining = len(rows)
            while remaining > self.capacity // 2:
                _, _, turn = heapq.heappop(heap)
                buckets[turn].popleft()
                remaining -= 1
                if buckets[turn]:
                    heapq.heappush(heap, (-len(buckets[turn]), age(buckets[turn][0]), turn))
            rows = [entry for values in buckets.values() for entry in values]
            retained = [entry for entry in retained if entry["label"] != label] + rows
        retained.sort(key=lambda entry: (int(entry["source_step"]), entry["entry_id"]))
        self._validate_entries(retained)
        previous_entries, previous_step = self.entries, self.step
        self.entries, self.step = retained, int(step)
        try:
            _atomic_json(self.manifest_path, self._encode())
        except Exception:
            self.entries, self.step = previous_entries, previous_step
            raise

    def snapshot(self, checkpoint_dir: str | os.PathLike) -> None:
        _atomic_json(Path(checkpoint_dir) / self.SNAPSHOT_NAME, self._encode())

    def restore(self, checkpoint_dir: str | os.PathLike) -> None:
        path = Path(checkpoint_dir) / self.SNAPSHOT_NAME
        if not path.is_file():
            raise FileNotFoundError(f"Replay-enabled resume requires checkpoint cache snapshot: {path}")
        self._decode(json.loads(path.read_text(encoding="utf-8")))
        _atomic_json(self.manifest_path, self._encode())

    def garbage_collect(self) -> int:
        """Call only after rollout workers finish; retained snapshots pin assets."""
        referenced = {image["path"] for entry in self.entries for image in entry["images"]}
        manifests = set(self.root.parent.glob(f"global_step_*/{self.SNAPSHOT_NAME}"))
        for stale in self._snapshot_refs.keys() - manifests:
            del self._snapshot_refs[stale]
        for manifest in manifests:
            stat = manifest.stat()
            cached = self._snapshot_refs.get(manifest)
            if cached is None or cached[:3] != (stat.st_ino, stat.st_mtime_ns, stat.st_size):
                payload = json.loads(manifest.read_text(encoding="utf-8"))
                paths = {
                    os.path.abspath(self.root / image["path"])
                    for entry in payload["entries"]
                    for image in entry["images"]
                }
                cached = (stat.st_ino, stat.st_mtime_ns, stat.st_size, paths)
                self._snapshot_refs[manifest] = cached
            referenced.update(cached[3])
        removed = 0
        for path in (self.root / "images").glob("*.png"):
            if str(path) not in referenced:
                path.unlink()
                removed += 1
        return removed
