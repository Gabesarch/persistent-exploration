"""Native y/x boundary around the existing source-aware x/y crop tool."""

from verl.models.gemma4.protocol import xyxy
from verl.tools.image_zoom_in_tool import ImageZoomInTool


class Gemma4ZoomTool(ImageZoomInTool):
    async def execute(self, instance_id, parameters, **kwargs):
        native = parameters.get("bbox_2d")
        canonical = xyxy(native)
        response, reward, info = await super().execute(instance_id, {**parameters, "bbox_2d": canonical}, **kwargs)
        info.update(model_bbox_yxyx=native, canonical_bbox_xyxy=canonical)
        if not response.terminate:
            response.text = f"Zoomed in on the image to the region {native} with label {parameters['label']}."
            count = self._processed_image_count(kwargs.get("agent_data"))
            if count is not None:
                response.text += f" The cropped view is now available as img_idx {count}."
        return response, reward, info
