import asyncio
import json

import numpy as np
import torch
from omegaconf import OmegaConf

from verl import DataProto
from verl.experimental.reward_loop.reward_manager.dapo import DAPORewardManager


class _FakeTokenizer:
    def decode(self, *_args, **_kwargs):
        return "decoded response"


def test_dapo_reward_manager_logs_reward_assignment_without_changing_reward(tmp_path):
    config = OmegaConf.create(
        {
            "trainer": {
                "comprehensive_debug_logging": {
                    "enabled": True,
                    "dir": str(tmp_path / "debug_logs"),
                    "save_cropped_images": True,
                    "save_source_images": False,
                    "max_text_chars": None,
                }
            },
            "reward": {
                "reward_kwargs": {
                    "overlong_buffer_cfg": {
                        "enable": False,
                        "len": 0,
                        "penalty_factor": 0.0,
                        "log": False,
                    },
                    "max_resp_len": 8,
                }
            },
        }
    )

    async def compute_score(**_kwargs):
        return {"score": 0.6, "accuracy": 0.5, "format": 1.0}

    manager = DAPORewardManager(config, _FakeTokenizer(), compute_score)
    data = DataProto.from_dict(
        tensors={
            "responses": torch.tensor([[1, 2, 0]], dtype=torch.long),
            "attention_mask": torch.tensor([[1, 1, 0]], dtype=torch.long),
        },
        non_tensors={
            "data_source": np.array(["debug_source"], dtype=object),
            "reward_model": np.array([{"ground_truth": "42"}], dtype=object),
            "extra_info": np.array([{"reward_type": "numeric", "tolerance": 0.01}], dtype=object),
            "uid": np.array(["sample-1"], dtype=object),
            "tool_extra_fields": np.array([{"request_id": "req-1"}], dtype=object),
        },
    )

    async def run_single_on_current_loop():
        manager.loop = asyncio.get_running_loop()
        return await manager.run_single(data)

    result = asyncio.run(run_single_on_current_loop())

    assert result == {
        "reward_score": 0.6,
        "reward_extra_info": {"score": 0.6, "accuracy": 0.5, "format": 1.0},
    }
    log_path = tmp_path / "debug_logs" / "reward_assignments.jsonl"
    record = json.loads(log_path.read_text())
    assert record["event"] == "reward_assignment"
    assert record["request_id"] == "req-1"
    assert record["ground_truth"] == "42"
    assert record["score"] == 0.6
    assert record["accuracy"] == 0.5
    assert record["format"] == 1.0
    assert record["final_reward"] == 0.6
