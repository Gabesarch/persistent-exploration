"""Real cached Gemma tokenizer/processor protocol checks (opt-in runtime)."""

import asyncio
import json
import os
from pathlib import Path

import pytest
import torch
from PIL import Image

from verl.models.gemma4 import protocol as p
from verl.models.gemma4.processing import multimodal_inputs

MODEL = os.environ.get("GEMMA4_MODEL", "/scratch/gpfs/ZHUANGL/gs1693/hf_cache/google/gemma-4-E4B-it")
ROOT = Path(__file__).resolve().parents[3]
CALL = '<|tool_call>call:image_zoom_in_tool{bbox_2d:[100,200,700,800],img_idx:0,label:<|"|>region<|"|>}<tool_call|>'


@pytest.fixture(scope="module")
def processor():
    from verl.utils.tokenizer import hf_processor

    proc = hf_processor(MODEL)
    assert proc is not None and proc.__class__.__name__ == "Gemma4Processor"
    return proc


def test_native_prompt_and_crop_suffix(processor):
    import yaml

    schema = yaml.safe_load((ROOT / "vero_eval/lmms_eval/tasks/_tool_templates/_toolset_zoom_gemma_yaml").read_text())[
        "metadata"
    ]["tool_schema_overrides"]["image_zoom_in_tool"]
    tools = [{"type": "function", "function": {"name": "image_zoom_in_tool", **schema}}]
    original = Image.new("RGB", (2000, 1000))
    crop = Image.new("RGB", (1200, 600))
    messages = [
        {"role": "system", "content": (ROOT / "vero_eval/prompts/tools/system_gemma_zoom.txt").read_text().strip()},
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": "\nWhat is shown?\nImage 0 is available as img_idx=0."},
            ],
        },
    ]
    initial = p.render(processor, messages, tools)
    assert initial == processor.apply_chat_template(
        messages, tools=tools, tokenize=False, add_generation_prompt=True, enable_thinking=True, preserve_thinking=True
    )
    call = p.parse_calls(CALL)[0]
    text = (
        "Zoomed in on the image to the region [100, 200, 700, 800] with label region. "
        "The cropped view is now available as img_idx 1."
    )
    full_messages = messages + [
        {
            "role": "assistant",
            "content": "",
            "reasoning": "Inspect the region.",
            "tool_calls": [{"type": "function", "function": call}],
        },
        {"role": "tool", "name": call["name"], "content": text},
        {
            "role": "user",
            "content": [{"type": "image"}, {"type": "text", "text": "Above are the tool-returned image(s)."}],
        },
    ]
    full = p.render(processor, full_messages, tools)
    suffix = p.tool_continuation(processor, [call], [text], [crop], stopped_with_response=True)
    expected = p.encode(processor, full.split(p.TOOL_RESPONSE, 1)[1], [crop])
    assert suffix == expected
    assert processor.tokenizer.bos_token_id not in suffix
    assert processor.tokenizer.convert_tokens_to_ids(p.TOOL_RESPONSE) not in suffix
    assert (
        p.tool_continuation(processor, [call], [text], [crop], stopped_with_response=False)
        == [processor.tokenizer.convert_tokens_to_ids(p.TOOL_RESPONSE)] + suffix
    )
    ids = torch.tensor([p.encode(processor, full, [original, crop])])
    padded = torch.nn.functional.pad(ids, (7, 13), value=processor.tokenizer.pad_token_id)
    mm = multimodal_inputs(processor, padded, [original, crop])
    assert mm["mm_token_type_ids"].shape == padded.shape
    assert torch.equal(mm["mm_token_type_ids"], (padded == processor.image_token_id).long())
    assert 0 < mm["mm_token_type_ids"].sum().item() <= 2240
    assert mm["pixel_values"].shape[0] == mm["image_position_ids"].shape[0] == 2
    from verl.utils.lazy_multi_modal import (
        _compute_multi_modal_inputs_for_sample,
        should_rehydrate_multi_modal_inputs,
        strip_heavy_multi_modal_inputs,
    )

    stripped = strip_heavy_multi_modal_inputs(mm)
    assert should_rehydrate_multi_modal_inputs([stripped])
    rebuilt = _compute_multi_modal_inputs_for_sample(
        tokenizer=processor.tokenizer,
        processor=processor,
        input_ids=padded[0],
        multi_modal_data={"images": [original, crop]},
    )
    for key in mm:
        assert torch.equal(mm[key], rebuilt[key]), key
    generation = json.loads((Path(MODEL) / "generation_config.json").read_text())
    assert processor.tokenizer.convert_tokens_to_ids(p.TOOL_RESPONSE) in generation["eos_token_id"]


def test_coords_and_native_crop_provenance(processor, tmp_path):
    import yaml

    from verl.experimental.agent_loop.tool_agent_loop import AgentData
    from verl.tools.gemma4_zoom_tool import Gemma4ZoomTool
    from verl.tools.schemas import OpenAIFunctionToolSchema

    schema = OpenAIFunctionToolSchema.model_validate(
        yaml.safe_load((ROOT / "vero2_scripts/config/tool_config/gemma4_image_zoom_in_tool.yaml").read_text())["tools"][
            0
        ]["tool_schema"]
    )
    tool = Gemma4ZoomTool(
        {
            "coordinate_system": "normalized_1000",
            "cropped_min_pixels": 518400,
            "cropped_max_pixels": 1048576,
            "use_original_resolution_source": True,
            "image_indexing": "multi_index",
        },
        schema,
    )
    source = tmp_path / "source.png"
    Image.new("RGB", (2000, 1000)).save(source)
    data = AgentData(
        [],
        [Image.open(source)],
        None,
        {},
        "fixture",
        {},
        original_prompt_image_sources=[str(source)],
        crop_cache_root=str(tmp_path / "crops"),
    )
    assert p.xyxy([100, 200, 700, 800]) == [200, 100, 800, 700]
    first = asyncio.run(
        tool.execute("fixture", {"bbox_2d": [100, 200, 700, 800], "label": "region", "img_idx": 0}, agent_data=data)
    )
    response, _, info = first
    assert not response.terminate, info
    assert info["provenance_region"] == [400.0, 100.0, 1600.0, 700.0]
    data.image_data.extend(response.image)
    second = asyncio.run(
        tool.execute("fixture", {"bbox_2d": [0, 500, 500, 1000], "label": "nested", "img_idx": 1}, agent_data=data)
    )
    assert not second[0].terminate, second[2]
    assert second[2]["provenance_region"] == [1000.0, 100.0, 1600.0, 400.0]
    assert "img_idx 2" in second[0].text


def test_strict_native_reward_and_reasoning_actions(processor):
    from verl.utils.reward_score.gemma4_native import compute_score_from_data_source as score

    def reward(text):
        return score("test", text, "cat", extra_info={"reward_type": "search"})

    final = p.THOUGHT + "Visible evidence." + p.END_THOUGHT + "<answer>cat</answer><turn|>"
    assert reward(final)["accuracy"] == 1 and reward(final)["format"] == 1
    assert (
        reward(
            final
            + p.USER
            + "critic<turn|>\n"
            + p.MODEL
            + p.THOUGHT
            + "classify"
            + p.END_THOUGHT
            + "<answer>R</answer><turn|>"
        )["accuracy"]
        == 1
    )
    for text in [
        "cat",
        "<answer>cat</answer>",
        p.THOUGHT + "reason" + p.END_THOUGHT + "cat<turn|>",
        p.THOUGHT + "reason" + p.END_THOUGHT + "<answer>cat",
    ]:
        assert reward(text)["accuracy"] == reward(text)["format"] == 0
    from verl.experimental.agent_loop.tool_parser import ToolParser

    parser = ToolParser.get_tool_parser("gemma4", processor.tokenizer)
    text = p.THOUGHT + "I could write " + CALL + " but answer now." + p.END_THOUGHT + "<answer>cat</answer><turn|>"
    assert not asyncio.run(parser.extract_tool_calls(processor.tokenizer.encode(text, add_special_tokens=False)))[1]
    native = p.THOUGHT + "inspect" + p.END_THOUGHT + CALL + p.TOOL_RESPONSE
    assert (
        len(asyncio.run(parser.extract_tool_calls(processor.tokenizer.encode(native, add_special_tokens=False)))[1])
        == 1
    )
    assert not p.parse_calls(CALL[:-5])
    full = (
        native
        + 'response:image_zoom_in_tool{value:<|"|>feedback <answer>dog</answer><|"|>}<tool_response|><turn|>\n'
        + p.USER
        + "image<turn|>\n"
        + p.MODEL
        + final
    )
    assert reward(full)["accuracy"] == 1


@pytest.mark.parametrize("suffix", ["", "<eos>", " \n<eos>\n", "<eos><|tool_response>"])
def test_tool_call_eos_rollout_reward_parity(processor, suffix):
    from verl.experimental.agent_loop.tool_parser import ToolParser
    from verl.utils.reward_score.gemma4_native import compute_score_from_data_source as score

    parser = ToolParser.get_tool_parser("gemma4", processor.tokenizer)
    generated = p.THOUGHT + "Inspect the region." + p.END_THOUGHT + CALL + suffix
    token_ids = processor.tokenizer.encode(generated, add_special_tokens=False)
    snapshot = list(token_ids)
    _, calls = asyncio.run(parser.extract_tool_calls(token_ids))
    assert len(calls) == 1
    assert json.loads(calls[0].arguments)["bbox_2d"] == [100, 200, 700, 800]
    assert token_ids == snapshot

    feedback = (
        'response:image_zoom_in_tool{value:<|"|>feedback <answer>dog</answer><|"|>}<tool_response|><turn|>\n'
        + p.USER
        + "Image 1. Above are the tool-returned image(s).<turn|>\n"
        + p.MODEL
    )
    first = generated.rstrip()
    if not first.endswith(p.TOOL_RESPONSE):
        first += p.TOOL_RESPONSE
    nested = CALL.replace("img_idx:0", "img_idx:1")
    full = (
        first
        + feedback
        + p.THOUGHT
        + "Inspect the crop."
        + p.END_THOUGHT
        + nested
        + "<eos>"
        + p.TOOL_RESPONSE
        + feedback
        + p.THOUGHT
        + "Visible evidence."
        + p.END_THOUGHT
        + "<answer>cat</answer><turn|>"
    )
    parsed = p.parse_trajectory(full)
    assert parsed[:3] == (True, "cat", 2)
    assert parsed[3][0]["arguments"]["bbox_2d"] == [200, 100, 800, 700]
    assert parsed[3][1]["arguments"]["img_idx"] == 1
    for label in ("A", "R"):
        sidecar = p.USER + "critic<turn|>\n" + p.MODEL + p.THOUGHT + "classify" + p.END_THOUGHT
        reward = score(
            "test", full + sidecar + f"<answer>{label}</answer><turn|>", "cat", extra_info={"reward_type": "search"}
        )
        assert reward["accuracy"] == reward["format"] == 1
    bad_final = full.replace("<answer>cat</answer>", "cat")
    reward = score("test", bad_final, "cat", extra_info={"reward_type": "search"})
    assert reward["accuracy"] == reward["format"] == 0


@pytest.mark.parametrize(
    "action",
    [
        "<eos>" + CALL,
        CALL + "<eos><eos>",
        CALL + "<eos>extra prose",
        CALL + "extra prose<eos>",
        CALL + p.TOOL_RESPONSE + "<eos>",
        CALL + "<eos>" + CALL,
        CALL.replace("<tool_call|>", "<eos>"),
        CALL.replace("[100,200,700,800]", "[700,200,100,800]") + "<eos>",
    ],
)
def test_tool_call_eos_rejects_malformed_boundaries(processor, action):
    from verl.experimental.agent_loop.tool_parser import ToolParser

    parser = ToolParser.get_tool_parser("gemma4", processor.tokenizer)
    text = p.THOUGHT + "Inspect." + p.END_THOUGHT + action
    ids = processor.tokenizer.encode(text, add_special_tokens=False)
    assert not asyncio.run(parser.extract_tool_calls(ids))[1]
    # In a full trajectory, content after TOOL_RESPONSE belongs to the
    # environment; malformed model suffixes are checked before that boundary.
    if p.TOOL_RESPONSE not in action:
        assert not p.parse_trajectory(
            text
            + p.TOOL_RESPONSE
            + "feedback<turn|>\n"
            + p.MODEL
            + p.THOUGHT
            + "Done."
            + p.END_THOUGHT
            + "<answer>cat</answer><turn|>"
        )[0]


def test_tool_call_eos_inside_label_is_not_removed():
    call = CALL.replace("region", "literal <eos> label")
    assert p.parse_calls(call + "<eos>")[0]["arguments"]["label"] == "literal <eos> label"


def test_critic_masks_forcing_and_rollback(processor):
    from verl.experimental.agent_loop.gemma4_tool_agent_loop import Gemma4ToolAgentLoop
    from verl.experimental.agent_loop.tool_agent_loop import AgentData
    from verl.utils.forced_action_tags import (
        FORCE_ACTION_AFTER_THINK_IDS_OPTIONS_KEY,
        FORCE_ACTION_TAG_IDS_KEY,
        ForceActionTagAfterThinkRequestProcessor,
    )

    loop = object.__new__(Gemma4ToolAgentLoop)
    loop.tokenizer = processor.tokenizer
    loop.processor = processor
    loop.conditional_answer_retry_critic_enabled = True
    loop.conditional_answer_retry_critic_oracle_supervision_enabled = True
    loop.conditional_answer_retry_critic_question = "Accept or retry?"
    loop.conditional_answer_retry_critic_assistant_prefill = p.THOUGHT + "classify\n" + p.END_THOUGHT + "<answer>\n"
    loop._retry_critic_accept_token_id = processor.tokenizer.encode("A", add_special_tokens=False)[0]
    loop._retry_critic_retry_token_id = processor.tokenizer.encode("R", add_special_tokens=False)[0]
    loop.response_length = 4096
    loop.force_final_answer_prefix = ""
    for label in ("A", "R"):
        data = AgentData([], [], None, {}, "fixture", {})
        data.prompt_ids = [22, 23]
        data.response_mask = [1, 1]
        data.retry_sft_mask = [0, 0]
        data.retry_critic_mask = [0, 0]
        data.response_logprobs = [-0.1, -0.2]
        snapshot = loop._snapshot_generation_state(data)
        assert asyncio.run(
            loop._append_retry_critic_supervision(data, grade={"accuracy": float(label == "A")}, label=label)
        )
        assert sum(data.retry_critic_mask) == 1 and sum(data.response_mask) == 2
        at = data.retry_critic_mask.index(1)
        assert processor.tokenizer.decode([data.prompt_ids[at]]) == label
        assert not data.response_mask[at]
        loop._rollback_generation_state(data, snapshot)
        assert data.prompt_ids == [22, 23] and data.response_mask == [1, 1] and sum(data.retry_critic_mask) == 0
    for tag in ("<tool_call>", "<answer>"):
        extra = loop._with_forced_action_extra_args({}, tag)["extra_args"]
        force = ForceActionTagAfterThinkRequestProcessor(
            extra[FORCE_ACTION_AFTER_THINK_IDS_OPTIONS_KEY], extra[FORCE_ACTION_TAG_IDS_KEY]
        )
        marker = processor.tokenizer.encode(p.END_THOUGHT, add_special_tokens=False)
        generated = []
        for expected in extra[FORCE_ACTION_TAG_IDS_KEY]:
            assert force._forced_token_id(marker, generated) == expected
            generated.append(expected)
        assert force._forced_token_id(marker, generated) is None


def test_vllm_image_expansion_matches_transformers(processor):
    from vllm.config import ModelConfig
    from vllm.multimodal import MULTIMODAL_REGISTRY
    from vllm.multimodal.processing.context import TimingContext
    from vllm.multimodal.processing.inputs import ProcessorInputs

    from verl.models.gemma4.processing import collapse_image_tokens

    config = ModelConfig(
        model=MODEL,
        tokenizer=MODEL,
        dtype="bfloat16",
        max_model_len=8192,
        mm_processor_kwargs={"max_soft_tokens": 1120},
        limit_mm_per_prompt={"image": 4, "audio": 0, "video": 0},
    )
    mmproc = MULTIMODAL_REGISTRY.create_processor(config, tokenizer=processor.tokenizer)
    images = [Image.new("RGB", (2000, 1000)), Image.new("RGB", (1200, 600))]
    text = (
        "<bos><|turn>user\n<|image|>question<turn|>\n"
        + p.MODEL
        + p.THOUGHT
        + "look"
        + p.END_THOUGHT
        + CALL
        + p.TOOL_RESPONSE
        + 'response:image_zoom_in_tool{value:<|"|>crop<|"|>}<tool_response|><turn|>\n'
        + p.USER
        + "<|image|>Above are the tool-returned image(s).<turn|>\n"
        + p.MODEL
    )
    expanded = p.encode(processor, text, images)
    compact = collapse_image_tokens(expanded, processor)
    assert compact.count(processor.image_token_id) == 2
    items = mmproc.info.parse_mm_data({"image": images})
    output = mmproc.apply(
        ProcessorInputs(prompt=compact, mm_data_items=items, hf_processor_mm_kwargs={"max_soft_tokens": 1120}),
        TimingContext(enabled=False),
    )
    assert output["prompt_token_ids"] == expanded
    assert len(output["mm_placeholders"]["image"]) == 2


def test_original_recipe_unchanged_and_gemma_overrides():
    import subprocess

    import yaml

    for file in (
        "vero2_scripts/run_branchstrata20_fresh_lr1e5_critic.sh",
        "vero2_scripts/config/branchstrata20_fresh_lr1e5_critic.yaml",
        "vero2_scripts/run_gdpo_fresh_lr1e5_noexplore_accformat.sh",
        "vero2_scripts/config/gdpo_fresh_lr1e5_noexplore_accformat.yaml",
    ):
        assert (ROOT / file).read_bytes() == subprocess.check_output(["git", "show", "HEAD:" + file], cwd=ROOT)
    original = yaml.safe_load((ROOT / "vero2_scripts/config/branchstrata20_fresh_lr1e5_critic.yaml").read_text())
    gemma = yaml.safe_load(
        (ROOT / "vero2_scripts/config/branchstrata20_fresh_lr1e5_critic_gemma4_e4b_recipe.yaml").read_text()
    )
    for key in ("algorithm", "critic"):
        assert original[key] == gemma[key]
    for key in (
        "train_files",
        "val_files",
        "train_batch_size",
        "max_prompt_length",
        "max_response_length",
        "per_batch_domain_weights",
    ):
        assert original["data"][key] == gemma["data"][key]
    old = original["actor_rollout_ref"]
    new = gemma["actor_rollout_ref"]
    for key in ("optim", "policy_loss", "ppo_mini_batch_size", "clip_ratio_low", "clip_ratio_high"):
        assert old["actor"][key] == new["actor"][key]
    for key in ("n", "temperature", "top_p", "top_k", "max_model_len"):
        assert old["rollout"][key] == new["rollout"][key]
    for key, value in old["rollout"]["multi_turn"].items():
        if key not in {"format", "tool_config_path", "conditional_answer_retry_critic_assistant_prefill"}:
            assert value == new["rollout"]["multi_turn"][key], key


@pytest.mark.parametrize("crop_first", [False, True], ids=["retry_initial_answer", "retry_after_crop"])
def test_native_controller_fixture_wrong_answer_crop_retry_and_accept(processor, tmp_path, monkeypatch, crop_first):
    from hydra import compose, initialize_config_dir

    from verl.experimental.agent_loop.agent_loop import DictConfigWrap
    from verl.experimental.agent_loop.gemma4_tool_agent_loop import Gemma4ToolAgentLoop
    from verl.utils.dataset.gemma4_dataset import Gemma4Dataset
    from verl.workers.rollout.replica import TokenOutput

    for key, value in {
        "VERL_ROOT": str(ROOT),
        "HF_HOME": str(Path(MODEL).parents[1]),
        "SAVE_PATH": str(tmp_path),
        "IMAGE_ROOT": "",
        "VERO2_CROP_CACHE_ROOT": str(tmp_path / "crops"),
    }.items():
        monkeypatch.setenv(key, value)
    with initialize_config_dir(config_dir=str(ROOT / "vero2_scripts/config"), version_base=None):
        cfg = compose(config_name="branchstrata20_fresh_lr1e5_critic_gemma4_e4b_smoke")

    # Keep fixture artifacts; the server is deterministic, the controller/tool/reward are real.
    class Server:
        turn = 0
        requests = []

        async def generate(self, **kw):
            self.requests.append({**kw, "prompt_ids": list(kw["prompt_ids"])})
            prompt = processor.tokenizer.decode(kw["prompt_ids"], skip_special_tokens=False)
            if "-llm-judge-" in kw["request_id"]:
                assert "<|think|>" not in prompt
                equivalent = (
                    "cat" in prompt.split("[PREDICTION]")[-1] if "[PREDICTION]" in prompt else "cat" in prompt[-100:]
                )
                # First rubric request rejects; second accepts, independently of wording.
                judge_count = sum("-llm-judge-" in r["request_id"] for r in self.requests)
                equivalent = judge_count > 1
                text = '"is_equivalent":' + str(equivalent).lower() + ',"partial_credit":false}<turn|>'
            else:
                self.turn += 1
                prefix = "" if not prompt.endswith(p.MODEL) else p.THOUGHT
                if crop_first and self.turn == 1:
                    text = prefix + "Inspect first." + p.END_THOUGHT + CALL + p.TOOL_RESPONSE
                elif self.turn == 1 + int(crop_first):
                    text = prefix + "Evidence." + p.END_THOUGHT + "<answer>dog</answer><turn|>"
                elif self.turn == 2 + int(crop_first):
                    call = CALL.replace("img_idx:0", "img_idx:1") if crop_first else CALL
                    text = prefix + "Inspect." + p.END_THOUGHT + call + p.TOOL_RESPONSE
                else:
                    text = prefix + "Clear evidence." + p.END_THOUGHT + "<answer>cat</answer><turn|>"
            ids = processor.tokenizer.encode(text, add_special_tokens=False)
            return TokenOutput(token_ids=ids, log_probs=[-0.1] * len(ids))

    image = tmp_path / "original.png"
    Image.new("RGB", (2000, 1000)).save(image)
    messages = [
        {"role": "system", "content": (ROOT / "vero_eval/prompts/tools/system_gemma_zoom.txt").read_text().strip()},
        {
            "role": "user",
            "content": [
                {"type": "image", "image": str(image)},
                {"type": "text", "text": "\nWhat is shown?\nImage 0 is available as img_idx=0."},
            ],
        },
    ]
    server = Server()

    async def run():
        loop = Gemma4ToolAgentLoop(
            DictConfigWrap(cfg), server, processor.tokenizer, processor, Gemma4Dataset, DictConfigWrap(cfg.data)
        )
        schema = loop.tool_schemas[0]["function"]["parameters"]["properties"]["bbox_2d"]
        assert schema["items"] == {"type": "number"} and schema["minItems"] == schema["maxItems"] == 4
        return await asyncio.wait_for(
            loop.run(
                {"temperature": 1.0, "top_p": 1.0, "top_k": -1, "logprobs": True},
                raw_prompt=messages,
                reward_model={"ground_truth": "cat"},
                data_source="fixture",
                extra_info={"reward_type": "search"},
                _debug_trajectory={"step": 1, "validate": False},
            ),
            timeout=30,
        )

    output = asyncio.run(run())
    assert server.turn == 3 + int(crop_first)
    assert len(output.branch_outputs) == 1
    branch = output.branch_outputs[0]
    assert branch.extra_fields["retry_critic_label"] == "R" and output.extra_fields["retry_critic_label"] == "A"
    for result in (output, branch):
        assert sum(result.retry_critic_mask) == 1
        assert all(
            not p_mask for p_mask, c_mask in zip(result.response_mask, result.retry_critic_mask, strict=False) if c_mask
        )
        assert len(result.response_ids) == len(result.response_mask) == len(result.response_logprobs)
    text = processor.tokenizer.decode(output.response_ids, skip_special_tokens=False)
    assert "<answer>dog</answer>" not in text
    assert "Above are the tool-returned image(s)." in text
    assert "<|tool_response><|tool_response>" not in text
    solver_requests = [r for r in server.requests if "-llm-judge-" not in r["request_id"]]
    previous_ids = solver_requests[int(crop_first)]["prompt_ids"]
    retry_ids = solver_requests[1 + int(crop_first)]["prompt_ids"]
    # Rollback keeps the exact token prefix, including prior sampled reasoning,
    # calls, crop returns, and image expansions. Only the native prefill is added.
    assert retry_ids[: len(previous_ids)] == previous_ids
    prefill = processor.tokenizer.decode(retry_ids[len(previous_ids) :], skip_special_tokens=False)
    assert prefill == p.THOUGHT + cfg.actor_rollout_ref.rollout.multi_turn.conditional_answer_retry_assistant_prefill
    retry_prompt = processor.tokenizer.decode(retry_ids, skip_special_tokens=False)
    assert retry_prompt.count("<bos>") == retry_prompt.count("<|turn>system") == 1
    if crop_first:
        assert "Inspect first." in retry_prompt
        assert retry_prompt.count("Above are the tool-returned image(s).") == 1
        assert text.count("Above are the tool-returned image(s).") == 2


def test_gemma_replicated_buffers_merge_without_concatenation(tmp_path):
    from types import SimpleNamespace

    from verl.model_merger.fsdp_model_merger import FSDPModelMerger

    merger = object.__new__(FSDPModelMerger)
    merger.config = SimpleNamespace(local_dir=str(tmp_path))
    merger.model_config = SimpleNamespace(model_type="gemma4")
    for rank in range(2):
        torch.save(
            {
                "model.layers.0.layer_scalar": torch.tensor([1.0]),
                "model.vision.std_bias": torch.tensor([0.0, 1.0]),
                "model.weight": torch.ones(1, 2) * rank,
            },
            tmp_path / f"model_world_size_2_rank_{rank}.pt",
        )
    state = merger._load_and_merge_state_dicts(2, 2, (2,), ("fsdp",))
    assert state["model.layers.0.layer_scalar"].shape == (1,)
    assert state["model.vision.std_bias"].shape == (2,)
    assert state["model.weight"].shape == (2, 2)
    shard = torch.load(tmp_path / "model_world_size_2_rank_1.pt", weights_only=False)
    shard["model.layers.0.layer_scalar"] += 0.0001  # This difference would vanish in BF16.
    torch.save(shard, tmp_path / "model_world_size_2_rank_1.pt")
    with pytest.raises(ValueError, match="replicated buffer differs"):
        merger._load_and_merge_state_dicts(2, 2, (2,), ("fsdp",))


def test_weight_transfer_preserves_buffers_and_rejects_unequal_replicas(monkeypatch):
    from verl.models.gemma4 import weight_transfer as transfer

    state = {
        "model.layers.0.layer_scalar": torch.tensor([1.0], dtype=torch.bfloat16),
        "model.observer.input_min": torch.tensor(float("inf")),
        "model.weight": torch.ones(2, 2),
    }
    original = dict(state)
    monkeypatch.setattr(transfer.dist, "is_initialized", lambda: True)
    monkeypatch.setattr(transfer.dist, "get_world_size", lambda: 2)

    def gather_equal(result, fingerprint):
        result[:] = [fingerprint, fingerprint]

    monkeypatch.setattr(transfer.dist, "all_gather_object", gather_equal)
    assert transfer.verify_replicated_buffers(state)["buffers"] == 2
    assert all(value is original[key] for key, value in state.items())

    def gather_unequal(result, fingerprint):
        result[:] = [fingerprint, "different"]

    monkeypatch.setattr(transfer.dist, "all_gather_object", gather_unequal)
    with pytest.raises(ValueError, match="differ across weight-transfer"):
        transfer.verify_replicated_buffers(state)


def test_original_source_dataset_adapter(processor, tmp_path):
    from omegaconf import OmegaConf

    from verl.utils.dataset.gemma4_dataset import Gemma4Dataset

    source = (
        "/scratch/gpfs/ZHUANGL/gs1693/vero2_datasets/train_rl_datasets/"
        "chart_search_counting_visual_probe_train_rollout_score_leq05_all.verl.jsonl"
    )
    config = OmegaConf.create(
        {
            "cache_dir": str(tmp_path / "dataset_cache"),
            "filter_overlong_prompts": False,
            "system_prompt": str(ROOT / "vero_eval/prompts/tools/system_gemma_zoom.txt"),
            "max_pixels": None,
            "shuffle": False,
            "seed": 42,
            "return_raw_chat": True,
            "return_multi_modal_inputs": False,
        }
    )
    dataset = Gemma4Dataset(source, processor.tokenizer, config, processor=processor, max_samples=4)
    assert len(dataset) == 4
    row = dataset[0]
    messages = row["raw_prompt"]
    assert messages[0]["content"] == (ROOT / "vero_eval/prompts/tools/system_gemma_zoom.txt").read_text().strip()
    images, _ = asyncio.run(dataset.process_vision_info(messages, config=config))
    block = next(
        c for m in messages if isinstance(m.get("content"), list) for c in m["content"] if c.get("type") == "image"
    )
    with Image.open(block["image"]) as original:
        assert images[0].size == original.size
    assert images[0].mode == "RGB"
    prompt = p.render(processor, messages)
    assert prompt.count("Image 0 is available as img_idx=0.") == 1
    raw = json.loads(Path(source).read_text().splitlines()[0])
    question = raw["prompt"][0]["content"].replace("<image>", "").strip()
    expected = [
        {"role": "system", "content": (ROOT / "vero_eval/prompts/tools/system_gemma_zoom.txt").read_text().strip()},
        {
            "role": "user",
            "content": [{"type": "image"}, {"type": "text", "text": question + "\nImage 0 is available as img_idx=0."}],
        },
    ]
    assert prompt == p.render(processor, expected)


@pytest.mark.parametrize("wrapped", [False, True])
def test_smoke_audit_checks_every_loss_and_reports_missing_live_critic(tmp_path, monkeypatch, wrapped):
    import importlib.util
    from types import SimpleNamespace

    spec = importlib.util.spec_from_file_location("gemma_audit", ROOT / "vero2_scripts/validate_gemma4_training.py")
    audit_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(audit_module)
    run = tmp_path / "run"
    checkpoint = run / "global_step_2/actor/model_world_size_1_rank_0.pt"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.touch()
    (run / "debug_logs").mkdir()
    (tmp_path / "resolved.yaml").write_text(f"trainer:\n  default_local_dir: {run}\n")
    trajectory = {
        "trajectory": {"step": 2},
        "main": {"extra_fields": {"retry_critic_label": "A"}},
        "branches": [{"extra_fields": {"retry_critic_label": "R"}}],
        "crop_image_refs": ["fixture.png"],
    }
    trace_path = run / "debug_logs/gemma4_raw_trajectories.jsonl"
    trace_path.write_text(json.dumps(trajectory) + "\n")

    def reload_ok(*args, **kwargs):
        (tmp_path / "checkpoint_audit.json").write_text('{"finite_reload_forward":true}')
        return SimpleNamespace(returncode=0, stdout="fixture reload success")

    monkeypatch.setattr(audit_module.subprocess, "run", reload_ok)
    lines = ["step:2 - actor/grad_norm:0.5", "actor/pg_loss:nan", "actor/pg_loss:0.1"]
    for group in ("text", "vision"):
        for suffix in ("grad_norm", "update_max_abs", "param_bf16", "grad_fp32", "grad_bf16"):
            lines.append(f"actor/gemma4_{group}_{suffix}:1.0")
    if wrapped:
        import re

        lines = [re.sub(r"(actor/[\w/]+:)(nan|[\d.]+)", r"\1np.float64(\2)", line) for line in lines]
    log_path = tmp_path / "training.log"
    log_path.write_text("\n".join(lines))
    with pytest.raises(SystemExit, match="acceptance incomplete"):
        audit_module.audit(tmp_path)
    report = json.loads((tmp_path / "acceptance.json").read_text())
    assert not report["live_validation_passed"]
    assert "Missing or nonfinite loss evidence." in report["limitations"]
    log_path.write_text("\n".join(line for line in lines if "nan" not in line))
    audit_module.audit(tmp_path)
    assert json.loads((tmp_path / "acceptance.json").read_text())["live_validation_passed"]
    log_path.write_text(
        "\n".join(line.replace("0.5", "0.0").replace("1.0", "0.0") for line in lines if "nan" not in line)
    )
    with pytest.raises(SystemExit, match="acceptance incomplete"):
        audit_module.audit(tmp_path)
    zero_report = json.loads((tmp_path / "acceptance.json").read_text())
    assert zero_report["gradient_norms"] == [0.0]
    assert "All-zero gradient signal." in zero_report["limitations"]
    log_path.write_text("\n".join(line for line in lines if "nan" not in line))
    trajectory["main"]["extra_fields"].clear()
    trajectory["branches"][0]["extra_fields"].clear()
    trace_path.write_text(json.dumps(trajectory) + "\n")
    with pytest.raises(SystemExit, match="acceptance incomplete"):
        audit_module.audit(tmp_path)
    assert (
        "No live critic supervision observed." in json.loads((tmp_path / "acceptance.json").read_text())["limitations"]
    )
