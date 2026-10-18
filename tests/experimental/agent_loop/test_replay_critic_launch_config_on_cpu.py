"""The feature launcher is isolated and changes only the reviewed run settings."""

import json
import os
import subprocess
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[3]
CONFIG_NAME = "branchstrata20_fresh_lr1e5_critic_replay_schedule_v1"
SCRIPTS = ROOT / "vero2_scripts"


def test_shared_configuration_defaults_keep_features_disabled():
    trainer = yaml.safe_load((ROOT / "verl/trainer/config/ppo_trainer.yaml").read_text())["trainer"]
    rollout = yaml.safe_load((ROOT / "verl/trainer/config/rollout/rollout.yaml").read_text())["multi_turn"]
    assert trainer["trajectory_replay"]["enabled"] is False
    assert trainer["trajectory_replay"]["batch_mode"] == "replace"
    assert rollout["prompt_group_schedule_enabled"] is False
    assert rollout["conditional_answer_retry_critic_mode"] == "legacy"


def _flatten(value, prefix=""):
    if isinstance(value, dict):
        result = {}
        for key, child in value.items():
            result.update(_flatten(child, f"{prefix}.{key}" if prefix else key))
        return result
    return {prefix: value}


def test_new_configuration_preserves_base_except_authorized_features():
    base = yaml.safe_load((SCRIPTS / "config/branchstrata20_fresh_lr1e5_critic.yaml").read_text())
    feature = yaml.safe_load((SCRIPTS / f"config/{CONFIG_NAME}.yaml").read_text())
    old, new = _flatten(base), _flatten(feature)
    changed = {key for key in old.keys() | new.keys() if old.get(key) != new.get(key)}
    authorized = {
        "actor_rollout_ref.actor.policy_loss.retry_critic.enabled",
        "actor_rollout_ref.rollout.multi_turn.conditional_answer_retry_critic_oracle_supervision_enabled",
        "actor_rollout_ref.rollout.multi_turn.conditional_answer_retry_critic_mode",
        "actor_rollout_ref.rollout.multi_turn.conditional_answer_retry_accept_partial_credit",
        "actor_rollout_ref.rollout.multi_turn.prompt_group_schedule_enabled",
        "trainer.experiment_name",
    }
    assert changed - authorized == {key for key in changed if key.startswith("trainer.trajectory_replay.")}
    assert authorized <= changed
    assert feature["data"]["train_batch_size"] == 128
    assert feature["actor_rollout_ref"]["rollout"]["n"] == 8
    assert feature["actor_rollout_ref"]["actor"]["policy_loss"]["retry_critic"]["enabled"] is False
    replay = feature["trainer"]["trajectory_replay"]
    assert replay["enabled"] and replay["prompt_fraction"] == 0.2 and replay["capacity"] == 8192
    assert replay["batch_mode"] == "append"
    assert replay["normal_budget"] == 4096 and replay["overlong_buffer"] == 1024
    assert replay["schedule_single_fraction"] == 0.10 and replay["schedule_double_fraction"] == 0.05
    assert (replay["schedule_first_turn"], replay["schedule_last_turn"], replay["schedule_min_gap"]) == (2, 9, 2)
    assert feature["trainer"]["experiment_name"] == f"{CONFIG_NAME}_20261010"
    assert feature["trainer"]["default_local_dir"] == base["trainer"]["default_local_dir"]
    assert feature["trainer"]["bootstrap_from_path"] is None


def test_main_launcher_pins_paths_before_and_after_environment_setup(tmp_path):
    checkout = tmp_path / "feature checkout"
    (checkout / "vero2_scripts/config").mkdir(parents=True)
    (checkout / f"vero2_scripts/config/{CONFIG_NAME}.yaml").write_text("{}\n")
    before = tmp_path / "before.json"
    after = tmp_path / "after.json"
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    python_stub = bin_dir / "python3"
    python_stub.write_text(
        "#!/usr/bin/python3\nimport json, os, sys\n"
        "from pathlib import Path\n"
        "Path(os.environ['PATH_TEST_AFTER']).write_text(json.dumps({'root': os.environ['REPO_ROOT'], "
        "'verl': os.environ['VERL_ROOT'], 'pythonpath': os.environ['PYTHONPATH'], 'argv': sys.argv[1:]}))\n"
    )
    python_stub.chmod(0o755)
    (checkout / "set_paths.sh").write_text(
        'printf \'{"root":"%s","verl":"%s"}\' "$REPO_ROOT" "$VERL_ROOT" > "$PATH_TEST_BEFORE"\n'
        "export REPO_ROOT=/old/checkout VERL_ROOT=/old/checkout\n"
    )
    env = dict(
        os.environ,
        VERO2_REPO_ROOT=str(checkout),
        VERL_ROOT="/inherited/checkout",
        SLURM_SUBMIT_DIR="/another/submit/directory",
        PATH=f"{bin_dir}:{os.environ['PATH']}",
        PATH_TEST_BEFORE=str(before),
        PATH_TEST_AFTER=str(after),
    )
    subprocess.run(
        ["bash", str(SCRIPTS / f"run_{CONFIG_NAME}.sh"), "trainer.total_training_steps=1"],
        cwd=tmp_path,
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )
    assert json.loads(before.read_text()) == {"root": str(checkout), "verl": str(checkout)}
    observed = json.loads(after.read_text())
    assert observed["root"] == observed["verl"] == str(checkout)
    assert observed["pythonpath"].split(":")[0] == str(checkout)
    assert f"--config-path={checkout}/vero2_scripts/config" in observed["argv"]
    assert f"--config-name={CONFIG_NAME}" in observed["argv"]
    assert observed["argv"][-1] == "trainer.total_training_steps=1"


def test_launcher_shell_syntax_and_smoke_resume_contract():
    for suffix in ("", "_smoke4gpu"):
        script = SCRIPTS / f"run_{CONFIG_NAME}{suffix}.sh"
        subprocess.run(["bash", "-n", str(script)], check=True)
    smoke = (SCRIPTS / f"run_{CONFIG_NAME}_smoke4gpu.sh").read_text()
    for setting in (
        "data.train_batch_size=20",
        "trainer.total_training_steps=3",
        "trainer.total_training_steps=4",
        "trainer.save_freq=1",
        "trainer.test_freq=1",
        "trainer.resume_from_path=$CHECKPOINT_DIR/global_step_3",
    ):
        assert setting in smoke
    assert 'SMOKE_RUN_ROOT="${REPLAY_SMOKE_RUN_ROOT:-$SAVE_PATH/replay_smoke}"' in smoke
    assert 'RUN_DIR="$SMOKE_RUN_ROOT/' in smoke
    assert 'RUN_DIR="$REPO_ROOT/logs/' not in smoke
    assert "--gres=gpu:4" in smoke and "--partition=gpu-test" in smoke and "--constraint=gpu80" in smoke
