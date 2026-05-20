# Copyright 2026 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import hashlib
import json
import re
from typing import Any


_SAFE_ACTOR_NAME_RE = re.compile(r"[^A-Za-z0-9_-]+")
_DUPLICATE_DASH_RE = re.compile(r"-+")


def _sanitize_actor_name_part(value: str) -> str:
    sanitized = _SAFE_ACTOR_NAME_RE.sub("-", value.strip()).strip("-").lower()
    sanitized = _DUPLICATE_DASH_RE.sub("-", sanitized)
    return sanitized or "tool"


def make_tool_actor_name(
    *,
    tool_name: str,
    pool_kind: str,
    num_workers: int,
    rate_limit: int | float,
    enable_global_rate_limit: bool,
    extra: dict[str, Any] | None = None,
) -> str:
    """Build a stable Ray actor name scoped to a tool pool configuration."""
    payload = {
        "tool_name": tool_name,
        "pool_kind": pool_kind,
        "num_workers": int(num_workers),
        "rate_limit": rate_limit,
        "enable_global_rate_limit": bool(enable_global_rate_limit),
    }
    if extra:
        payload["extra"] = extra

    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    digest = hashlib.sha1(encoded.encode("utf-8")).hexdigest()[:12]
    prefix = _sanitize_actor_name_part(f"{tool_name}-{pool_kind}")
    return f"{prefix}-{digest}"
