# Copyright 2025 Bytedance Ltd. and/or its affiliates
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

import json
import tempfile
import unittest
from pathlib import Path

from verl.trainer.ppo.ray_trainer import RayPPOTrainer


class TestRayTrainerValidationDumpOnCPU(unittest.TestCase):
    def setUp(self):
        self.trainer = RayPPOTrainer.__new__(RayPPOTrainer)
        self.trainer.global_steps = 7
        self.trainer.image_key = None

    def _read_dump(self, dump_dir: str):
        dump_path = Path(dump_dir) / "7.jsonl"
        with dump_path.open() as f:
            return [json.loads(line) for line in f]

    def test_dump_generations_without_image_refs_preserves_schema(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            self.trainer._dump_generations(
                inputs=["prompt"],
                outputs=["answer"],
                gts=["gt"],
                scores=[1.0],
                reward_extra_infos_dict={"reward": [1.0], "ignored": []},
                dump_path=tmpdir,
            )

            [entry] = self._read_dump(tmpdir)
            self.assertEqual(
                set(entry.keys()),
                {"input", "output", "gts", "score", "step", "reward"},
            )
            self.assertEqual(entry["input"], "prompt")
            self.assertEqual(entry["reward"], 1.0)

    def test_dump_generations_with_valid_image_refs_includes_paths(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            self.trainer._dump_generations(
                inputs=["prompt_a", "prompt_b"],
                outputs=["answer_a", "answer_b"],
                gts=["gt_a", "gt_b"],
                scores=[1.0, 0.5],
                reward_extra_infos_dict={"reward": [1.0, 0.5]},
                dump_path=tmpdir,
                image_paths=["img_a.png", Path("img_b.png")],
            )

            entries = self._read_dump(tmpdir)
            self.assertEqual(entries[0]["images"], "img_a.png")
            self.assertEqual(entries[1]["images"], "img_b.png")

    def test_dump_generations_with_invalid_image_refs_nulls_entry(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            self.trainer._dump_generations(
                inputs=["prompt_a", "prompt_b"],
                outputs=["answer_a", "answer_b"],
                gts=["gt_a", "gt_b"],
                scores=[1.0, 0.5],
                reward_extra_infos_dict={"reward": [1.0, 0.5]},
                dump_path=tmpdir,
                image_paths=[{"bad": "payload"}, ["img_b.png", Path("img_c.png")]],
            )

            entries = self._read_dump(tmpdir)
            self.assertIsNone(entries[0]["images"])
            self.assertEqual(entries[1]["images"], ["img_b.png", "img_c.png"])

    def test_dump_generations_uses_alternate_image_key(self):
        self.trainer.image_key = "frames"

        with tempfile.TemporaryDirectory() as tmpdir:
            self.trainer._dump_generations(
                inputs=["prompt"],
                outputs=["answer"],
                gts=["gt"],
                scores=[1.0],
                reward_extra_infos_dict={"reward": [1.0]},
                dump_path=tmpdir,
                image_paths=["frame_a.png"],
            )

            [entry] = self._read_dump(tmpdir)
            self.assertIn("frames", entry)
            self.assertNotIn("images", entry)
            self.assertEqual(entry["frames"], "frame_a.png")

    def test_collect_image_references_prefers_image_key_then_falls_back(self):
        self.trainer.image_key = "frames"

        refs = self.trainer._collect_image_references(
            [
                {"frames": [Path("frame_a.png"), "frame_b.png"], "images": "ignored.png"},
                {"images": "fallback.png"},
                None,
                {"frames": {"bad": "payload"}},
            ]
        )

        self.assertEqual(refs[0], ["frame_a.png", "frame_b.png"])
        self.assertEqual(refs[1], "fallback.png")
        self.assertIsNone(refs[2])
        self.assertIsNone(refs[3])


if __name__ == "__main__":
    unittest.main()
