# Training

Train Qwen3-VL-8B or Gemma 4 E4B with Persistent Exploration. The recipes combine visual search with zoom tools, continued exploration after failed answers, and an auxiliary answer/retry objective.

Run all commands from the repository root after completing [environment setup](../README.md#installation). The default recipes use eight GPUs.

## Data Setup

### Training Data

The [released dataset](https://huggingface.co/datasets/gsarch/Persistent-Exploration-RL-Data) contains 2,042 VisualProbe training examples and their images. Download and export it:

```bash
export ROOT_PATH=/path/to/runtime
export HF_HOME="$ROOT_PATH/hf_cache"

hf download gsarch/Persistent-Exploration-RL-Data --repo-type dataset \
  --revision 07fcdb7be28759fe278507aca49d1a38382456d4 \
  --local-dir "$ROOT_PATH/datasets/persistent-exploration-release"

python "$ROOT_PATH/datasets/persistent-exploration-release/export_verl.py" \
  "$ROOT_PATH/datasets/persistent-exploration-train"

export TRAIN_FILES="$ROOT_PATH/datasets/persistent-exploration-train/chart_search_counting_visual_probe_train_rollout_score_leq05_all.verl.jsonl"
export IMAGE_ROOT="$ROOT_PATH/datasets/persistent-exploration-train"
```

### Validation Data

Prepare the VisualProbe validation splits:

```bash
python persistent-exploration-rl/prepare_validation.py \
  "$ROOT_PATH/datasets/visual-probe-validation"

export VAL_FILES="$ROOT_PATH/datasets/visual-probe-validation/visual_probe_validation.verl.jsonl"
```

The exporter downloads the Easy, Medium, and Hard benchmark splits and writes the validation JSONL and images. These examples are used for validation and are separate from the training examples.

## Qwen3-VL-8B

Download the base model, then start training with the data paths set above:

```bash
conda activate persistent-exploration-qwen
hf download Qwen/Qwen3-VL-8B-Instruct \
  --local-dir "$ROOT_PATH/models/Qwen3-VL-8B-Instruct"

export MODEL_PATH="$ROOT_PATH/models/Qwen3-VL-8B-Instruct"
bash persistent-exploration-rl/train.sh
```

The launcher loads the [Qwen training configuration](config/branchstrata20_fresh_lr1e5_critic_massbounds.yaml).

## Gemma 4 E4B

Download the Gemma base model and launch training with the same data paths:

```bash
conda activate persistent-exploration-gemma
hf download google/gemma-4-E4B-it \
  --local-dir "$ROOT_PATH/models/gemma-4-E4B-it"

export GEMMA4_MODEL_PATH="$ROOT_PATH/models/gemma-4-E4B-it"
bash persistent-exploration-rl/train_gemma.sh
```

The launcher loads the [Gemma training configuration](config/gemma4_e4b.yaml), including its tool interface and prompts. Authenticate with `hf auth login` if the model download requires access to your Hugging Face account.

## Checkpoints

Checkpoints are written to `$ROOT_PATH/outputs/checkpoints/<experiment_name>` by default. Training resumes from the latest checkpoint in that directory. Set a new experiment name in the training configuration to start a separate run.

See the [evaluation guide](../persistent-exploration-eval/README.md#evaluate-a-local-checkpoint) to evaluate a saved checkpoint.

## Method

Persistent Exploration continues a failed search from its existing context, requires further tool exploration, and trains on both the original and retry trajectories. An auxiliary objective teaches the model to decide whether to accept an answer or continue searching.

The recipes bound the contributions of retry branches and the auxiliary objective. These settings are defined in the Qwen and Gemma training configurations linked above.
