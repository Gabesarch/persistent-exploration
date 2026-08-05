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

    def test_gdpo_stability_defaults_off_and_branch_configs_opt_in(self):
        repo_root = Path(__file__).resolve().parents[2]
        base_cfg = OmegaConf.load(repo_root / "verl" / "trainer" / "config" / "ppo_trainer.yaml")
        self.assertFalse(base_cfg.algorithm.gdpo_stability_enabled)
        self.assertEqual(base_cfg.algorithm.gdpo_component_std_floor, 0.05)
        self.assertEqual(base_cfg.algorithm.gdpo_component_adv_clip, 5.0)
        self.assertEqual(base_cfg.algorithm.gdpo_final_std_floor, 0.05)
        self.assertEqual(base_cfg.algorithm.gdpo_final_adv_clip, 10.0)

        branch_configs = [
            "gdpo_hdpotool_gspo_chart_qwen3vl_visual_probe_cropzoom_cpueff_hardonly_"
            "answer_retry_branch_donly_llmjudge_forcedprefill.yaml",
            "gdpo_hdpotool_gspo_chart_qwen3vl_visual_probe_cropzoom_debug256_cpueff_"
            "answer_retry_branch_donly_llmjudge_forcedprefill.yaml",
        ]
        for config_name in branch_configs:
            cfg = OmegaConf.load(repo_root / "vero2_scripts" / "config" / config_name)
            self.assertEqual(cfg.algorithm.gdpo_baseline_weight_key, "gdpo_baseline_weight")
            self.assertTrue(cfg.algorithm.gdpo_stability_enabled)
            self.assertEqual(cfg.algorithm.gdpo_component_std_floor, 0.05)
            self.assertEqual(cfg.algorithm.gdpo_component_adv_clip, 5.0)
            self.assertEqual(cfg.algorithm.gdpo_final_std_floor, 0.05)
            self.assertEqual(cfg.algorithm.gdpo_final_adv_clip, 10.0)

    def test_branch_strata_recipes_enable_guarded_gdpo_and_legacy_actor(self):
        repo_root = Path(__file__).resolve().parents[2]
        branch_strata_configs = [
            "gdpo_hdpotool_gspo_chart_qwen3vl_visual_probe_cropzoom_cpueff_hardonly_"
            "answer_retry_branch_donly_llmjudge_forcedprefill_branchstrata.yaml",
            "gdpo_hdpotool_gspo_chart_qwen3vl_visual_probe_cropzoom_debug256_cpueff_"
            "answer_retry_branch_donly_llmjudge_forcedprefill_branchstrata.yaml",
        ]
        for config_name in branch_strata_configs:
            cfg = OmegaConf.load(repo_root / "vero2_scripts" / "config" / config_name)
            self.assertTrue(cfg.algorithm.gdpo_stability_enabled)
            self.assertTrue(cfg.actor_rollout_ref.actor.policy_loss.branch_strata.enabled)
            self.assertEqual(cfg.actor_rollout_ref.actor.policy_loss.branch_strata.branch_fraction, 0.2)
            self.assertTrue(cfg.actor_rollout_ref.actor.policy_loss.branch_strata.cap_by_parent_prevalence)
            self.assertTrue(cfg.actor_rollout_ref.actor.policy_loss.branch_strata.parent_balance)
            self.assertEqual(cfg.trainer.use_legacy_worker_impl, "enable")

    def test_noforceanswer_recipe_caps_retries_and_subsamples_branches(self):
        repo_root = Path(__file__).resolve().parents[2]
        config_name = (
            "gdpo_hdpotool_gspo_chart_qwen3vl_visual_probe_cropzoom_cpueff_hardonly_"
            "answer_retry_branch_donly_llmjudge_forcedprefill_branchstrata_base_"
            "noforceanswer_te015_turn4k_inferenceval.yaml"
        )
        cfg = OmegaConf.load(repo_root / "vero2_scripts" / "config" / config_name)

        self.assertEqual(cfg.actor_rollout_ref.rollout.multi_turn.conditional_answer_retry_max_retries, 4)
        strata = cfg.actor_rollout_ref.actor.policy_loss.branch_strata
        self.assertEqual(strata.branch_fraction, 0.2)
        self.assertTrue(strata.subsampling.enabled)
        self.assertEqual(strata.subsampling.max_branches_per_parent, 1)
        self.assertEqual(strata.subsampling.seed, 42)
        reward_kwargs = cfg.reward.custom_reward_function.reward_kwargs
        self.assertTrue(reward_kwargs.assistant_turn_overlong_enabled)
        self.assertEqual(reward_kwargs.assistant_turn_overlong_max_turns, 11)
        self.assertEqual(reward_kwargs.assistant_turn_overlong_buffer_turns, 2)
        self.assertEqual(reward_kwargs.assistant_turn_overlong_penalty_factor, 1.0)

    def test_randomturncap_recipe_is_isolated_and_configured(self):
        repo_root = Path(__file__).resolve().parents[2]
        cfg = OmegaConf.load(
            repo_root / "vero2_scripts" / "config" / "branchstrata20_fresh_randomturncap.yaml"
        )
        multi_turn = cfg.actor_rollout_ref.rollout.multi_turn
        self.assertEqual(multi_turn.max_assistant_turns, 11)
        self.assertTrue(multi_turn.randomized_max_assistant_turns_enabled)
        self.assertEqual(multi_turn.randomized_max_assistant_turns_early_window, 5)
        self.assertEqual(multi_turn.randomized_max_assistant_turns_per_turn_probability, 0.05)
        self.assertEqual(multi_turn.randomized_max_assistant_turns_seed, 42)
        self.assertFalse(multi_turn.randomized_max_assistant_turns_enable_validation)
        reward_kwargs = cfg.reward.custom_reward_function.reward_kwargs
        self.assertEqual(reward_kwargs.tool_efficiency_concave_k, 10.0)
        self.assertEqual(reward_kwargs.tool_efficiency_concave_q, 0.75)
        self.assertEqual(
            cfg.trainer.experiment_name,
            "branchstrata20_fresh_randomturncap_p05_x5_q075_20260803",
        )


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
