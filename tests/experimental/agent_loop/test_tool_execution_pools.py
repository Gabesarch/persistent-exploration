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

import re

from verl.tools import sandbox_fusion_tools, search_tool
from verl.tools.utils.ray_actor_utils import make_tool_actor_name


def test_make_tool_actor_name_is_stable_and_ray_safe():
    actor_name = make_tool_actor_name(
        tool_name="Search Tool!",
        pool_kind="execution pool",
        num_workers=16,
        rate_limit=32,
        enable_global_rate_limit=True,
    )
    same_actor_name = make_tool_actor_name(
        tool_name="Search Tool!",
        pool_kind="execution pool",
        num_workers=16,
        rate_limit=32,
        enable_global_rate_limit=True,
    )
    different_actor_name = make_tool_actor_name(
        tool_name="Search Tool!",
        pool_kind="execution pool",
        num_workers=17,
        rate_limit=32,
        enable_global_rate_limit=True,
    )

    assert actor_name == same_actor_name
    assert actor_name != different_actor_name
    assert re.fullmatch(r"[A-Za-z0-9_-]+", actor_name)
    assert actor_name.startswith("search-tool-execution-pool-")


class _FakeRemoteClass:
    def __init__(self, calls):
        self.calls = calls

    def options(self, **kwargs):
        self.calls.append(("options", kwargs))
        return self

    def remote(self, **kwargs):
        self.calls.append(("remote", kwargs))
        return "actor-handle"


class _FakeRay:
    def __init__(self):
        self.calls = []

    def remote(self, cls):
        self.calls.append(("remote_class", cls))
        return _FakeRemoteClass(self.calls)


def test_search_execution_pool_uses_named_reusable_actor(monkeypatch):
    fake_ray = _FakeRay()
    monkeypatch.setattr(search_tool, "ray", fake_ray)

    actor = search_tool.init_search_execution_pool(
        num_workers=7,
        enable_global_rate_limit=True,
        rate_limit=13,
        tool_name="search",
    )

    assert actor == "actor-handle"
    options = dict(fake_ray.calls[1][1])
    remote_kwargs = dict(fake_ray.calls[2][1])
    assert options["get_if_exists"] is True
    assert options["max_concurrency"] == 7
    assert options["name"].startswith("search-execution-pool-")
    assert remote_kwargs["rate_limiter_actor_name"].startswith("search-rate-limiter-")
    assert remote_kwargs["rate_limiter_actor_name"] != "rate-limiter"


def test_sandbox_execution_pool_uses_named_reusable_actor(monkeypatch):
    fake_ray = _FakeRay()
    monkeypatch.setattr(sandbox_fusion_tools, "ray", fake_ray)

    actor = sandbox_fusion_tools.init_execution_pool(
        num_workers=3,
        enable_global_rate_limit=True,
        rate_limit=5,
        tool_name="code_interpreter",
    )

    assert actor == "actor-handle"
    options = dict(fake_ray.calls[1][1])
    remote_kwargs = dict(fake_ray.calls[2][1])
    assert options["get_if_exists"] is True
    assert options["max_concurrency"] == 3
    assert options["name"].startswith("code_interpreter-execution-pool-")
    assert remote_kwargs["rate_limiter_actor_name"].startswith("code_interpreter-rate-limiter-")
    assert remote_kwargs["rate_limiter_actor_name"] != "rate-limiter"
