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

import numpy as np
from omegaconf import OmegaConf

from verl.utils.crop_cache import CROP_IMAGE_REFS_KEY
from verl.trainer.ppo.ray_trainer import RayPPOTrainer


class _FakeBatch:
    def __init__(self, non_tensor_batch, batch_size):
        self.non_tensor_batch = non_tensor_batch
        self._batch_size = batch_size

    def __len__(self):
        return self._batch_size


class TestRayTrainerValidationDumpOnCPU(unittest.TestCase):
    def setUp(self):
        self.trainer = RayPPOTrainer.__new__(RayPPOTrainer)
        self.trainer.global_steps = 7
        self.trainer.image_key = None
        self.trainer.comprehensive_debug_options = {"max_text_chars": None}
        self.trainer.config = OmegaConf.create({"trainer": {"save_validation_crops": False}})
        self.trainer._warned_save_validation_crops_without_dir = False

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

    def test_dump_generations_with_crop_refs_includes_refs(self):
        crop_refs = [[{"path": "crops/step_7/sample_a/crop_0.png", "bbox_2d": [1, 2, 3, 4]}]]

        with tempfile.TemporaryDirectory() as tmpdir:
            self.trainer._dump_generations(
                inputs=["prompt"],
                outputs=["answer"],
                gts=["gt"],
                scores=[1.0],
                reward_extra_infos_dict={"reward": [1.0]},
                dump_path=tmpdir,
                crop_image_refs=crop_refs,
            )

            [entry] = self._read_dump(tmpdir)
            self.assertEqual(entry[CROP_IMAGE_REFS_KEY], crop_refs[0])

    def test_copy_validation_crop_refs_for_batch_copies_and_rewrites_refs(self):
        self.trainer.config = OmegaConf.create({"trainer": {"save_validation_crops": True}})

        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)
            source_crop = tmpdir_path / "cache" / "crop.png"
            source_crop.parent.mkdir()
            source_crop.write_bytes(b"crop-bytes")

            crop_refs = np.empty(2, dtype=object)
            crop_refs[0] = [
                {
                    "path": str(source_crop),
                    "order": 2,
                    "bbox_2d": [1, 2, 3, 4],
                    "tool_name": "image_zoom_in_tool",
                }
            ]
            crop_refs[1] = [{"path": str(tmpdir_path / "cache" / "missing.png"), "order": 0}]
            batch = _FakeBatch(
                {
                    CROP_IMAGE_REFS_KEY: crop_refs,
                    "uid": np.array(["sample/a", "sample_b"], dtype=object),
                },
                batch_size=2,
            )

            copied_refs = self.trainer._copy_validation_crop_refs_for_batch(batch, str(tmpdir_path / "validation"))

            self.assertEqual(len(copied_refs), 2)
            self.assertEqual(len(copied_refs[0]), 1)
            self.assertEqual(copied_refs[0][0]["source_path"], str(source_crop))
            self.assertEqual(copied_refs[0][0]["step"], 7)
            self.assertEqual(copied_refs[0][0]["bbox_2d"], [1, 2, 3, 4])
            copied_path = Path(copied_refs[0][0]["path"])
            self.assertTrue(copied_path.exists())
            self.assertEqual(copied_path.read_bytes(), b"crop-bytes")
            self.assertIn("validation/crops/step_7/sample_sample_a/crop_2.png", copied_path.as_posix())
            self.assertEqual(copied_refs[1], [])

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
