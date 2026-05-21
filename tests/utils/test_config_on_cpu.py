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

import unittest
from dataclasses import dataclass, field
from pathlib import Path

from omegaconf import OmegaConf

from verl.base_config import BaseConfig
from verl.utils import omega_conf_to_dataclass


@dataclass
class TestDataclass(BaseConfig):
    hidden_size: int = 0
    activation: str = "relu"


@dataclass
class TestTrainConfig(BaseConfig):
    batch_size: int = 0
    model: TestDataclass = field(default_factory=TestDataclass)
    override_config: dict = field(default_factory=dict)


_cfg_str = """train_config:
  _target_: tests.utils.test_config_on_cpu.TestTrainConfig
  batch_size: 32
  model:
    hidden_size: 768
    activation: relu
  override_config: {}"""


class TestConfigOnCPU(unittest.TestCase):
    """Test cases for configuration utilities on CPU.

    Test Plan:
    1. Test basic OmegaConf to dataclass conversion for simple nested structures
    2. Test nested OmegaConf to dataclass conversion for complex hierarchical configurations
    3. Verify all configuration values are correctly converted and accessible
    """

    def setUp(self):
        self.config = OmegaConf.create(_cfg_str)

    def test_omega_conf_to_dataclass(self):
        sub_cfg = self.config.train_config.model
        cfg = omega_conf_to_dataclass(sub_cfg, TestDataclass)
        self.assertEqual(cfg.hidden_size, 768)
        self.assertEqual(cfg.activation, "relu")
        assert isinstance(cfg, TestDataclass)

    def test_nested_omega_conf_to_dataclass(self):
        cfg = omega_conf_to_dataclass(self.config.train_config, TestTrainConfig)
        self.assertEqual(cfg.batch_size, 32)
        self.assertEqual(cfg.model.hidden_size, 768)
        self.assertEqual(cfg.model.activation, "relu")
        assert isinstance(cfg, TestTrainConfig)
        assert isinstance(cfg.model, TestDataclass)

    def test_ppo_trainer_save_validation_crops_defaults_false(self):
        repo_root = Path(__file__).resolve().parents[2]
        cfg = OmegaConf.load(repo_root / "verl" / "trainer" / "config" / "ppo_trainer.yaml")
        self.assertFalse(cfg.trainer.save_validation_crops)
        self.assertEqual(cfg.trainer.comprehensive_debug_logging.log_every_n_steps, 1)


class TestPrintCfgCommand(unittest.TestCase):
    """Test suite for the print_cfg.py command-line tool."""

    def test_command_with_override(self):
        """Test that the command runs without error when overriding config values."""
        import os
        import subprocess
        import tempfile

        repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
        env = os.environ.copy()

        with tempfile.TemporaryDirectory() as save_dir, tempfile.TemporaryDirectory() as image_root:
            env.update(
                {
                    "VERL_ROOT": repo_root,
                    "SAVE_PATH": save_dir,
                    "IMAGE_ROOT": image_root,
                    "ROOT_PATH": save_dir,
                }
            )

            result = subprocess.run(
                [
                    "python3",
                    "scripts/print_cfg.py",
                    "--cfg",
                    "job",
                    "--config-path",
                    os.path.join(repo_root, "examples", "gaze_rl", "config"),
                    "--config-name",
                    "gspo_chart_qwen3vl_onpolicy",
                    "trainer.experiment_name=test_override",
                ],
                cwd=repo_root,
                capture_output=True,
                text=True,
                env=env,
            )

        # Verify the command exited successfully
        self.assertEqual(result.returncode, 0, f"Command failed with stderr: {result.stderr}")

        # Verify the output contains expected config information
        self.assertIn("trainer:", result.stdout)
        self.assertIn("experiment_name: test_override", result.stdout)

    def test_gaze_rl_config_resolves_with_launcher_env(self):
        """Test that the gaze RL config resolves with the same env shape as its launcher."""
        import os
        import subprocess
        import tempfile

        repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
        env = os.environ.copy()

        with tempfile.TemporaryDirectory() as save_dir, tempfile.TemporaryDirectory() as image_root:
            env.update(
                {
                    "VERL_ROOT": repo_root,
                    "SAVE_PATH": save_dir,
                    "IMAGE_ROOT": image_root,
                    "ROOT_PATH": save_dir,
                }
            )

            result = subprocess.run(
                [
                    "python3",
                    "scripts/print_cfg.py",
                    "--cfg",
                    "job",
                    "--config-path",
                    os.path.join(repo_root, "examples", "gaze_rl", "config"),
                    "--config-name",
                    "gspo_chart_qwen3vl_onpolicy",
                ],
                cwd=repo_root,
                capture_output=True,
                text=True,
                env=env,
            )

        self.assertEqual(result.returncode, 0, f"Command failed with stderr: {result.stderr}")
        self.assertIn("validation_data_dir", result.stdout)
        self.assertIn("image_key: images", result.stdout)


if __name__ == "__main__":
    unittest.main()
