"""Original-resolution RGB dataset adapter for Gemma (no Qwen image transform)."""

import io

from PIL import Image

from verl.utils.dataset.rl_dataset import RLHFDataset


class Gemma4Dataset(RLHFDataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self.system_prompt:
            self.system_prompt = self.system_prompt.strip()

    def _build_messages(self, example, *, remove_prompt=True):
        messages = super()._build_messages(example, remove_prompt=remove_prompt)
        for message in messages:
            content = message.get("content")
            # Native Gemma renders a system content list with an extra space.
            # The evaluation supplies a stripped string, so preserve that form.
            if message["role"] == "system" and isinstance(content, list):
                message["content"] = "".join(item.get("text", "") for item in content).strip()
            if (
                message["role"] == "user"
                and isinstance(content, list)
                and any(x.get("type") == "image" for x in content)
            ):
                # Jinja trims each content block separately. Put the hint in the
                # question's text block so its separating newline survives.
                texts = [item for item in content if item.get("type") == "text"]
                if texts:
                    texts[-1]["text"] = texts[-1]["text"].rstrip() + "\nImage 0 is available as img_idx=0."
                else:
                    content.append({"type": "text", "text": "Image 0 is available as img_idx=0."})
                break
        return messages

    @staticmethod
    async def process_vision_info(messages, image_patch_size=None, config=None):
        images = []
        for message in messages:
            content = message.get("content", [])
            if not isinstance(content, list):
                continue
            for item in content:
                if item.get("type") == "video":
                    raise ValueError("Gemma critic recipe supports images only")
                if item.get("type") != "image":
                    continue
                source = item.get("image", item.get("path"))
                if isinstance(source, dict):
                    source = io.BytesIO(source["bytes"]) if source.get("bytes") else source["path"]
                if isinstance(source, bytes):
                    source = io.BytesIO(source)
                if isinstance(source, str):
                    source = source.removeprefix("file://")
                image = source if isinstance(source, Image.Image) else Image.open(source)
                images.append(image.convert("RGB"))
        return images or None, None
