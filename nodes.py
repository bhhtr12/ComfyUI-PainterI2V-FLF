import torch
import comfy.model_management
import comfy.utils
import comfy.clip_vision 
import node_helpers
from comfy_api.latest import io, ComfyExtension
from typing_extensions import override

class PainterI2V(io.ComfyNode):
    
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="PainterI2V",
            category="conditioning/video_models",
            inputs=[
                io.Conditioning.Input("positive"),
                io.Conditioning.Input("negative"),
                io.Vae.Input("vae"),
                io.Int.Input("width", default=832, min=16, max=4096, step=16),
                io.Int.Input("height", default=480, min=16, max=4096, step=16),
                io.Int.Input("length", default=81, min=1, max=4096, step=4),
                io.Int.Input("batch_size", default=1, min=1, max=4096),
                io.Float.Input("motion_amplitude", default=1.15, min=1.0, max=2.0, step=0.05),
                io.ClipVisionOutput.Input("clip_vision_output", optional=True), # Treated as Start CV
                io.ClipVisionOutput.Input("clip_vision_output_end", optional=True), # Treated as End CV
                io.Image.Input("start_image", optional=True),
                io.Image.Input("end_image", optional=True),
            ],
            outputs=[
                io.Conditioning.Output(display_name="positive"),
                io.Conditioning.Output(display_name="negative"),
                io.Latent.Output(display_name="latent"),
            ]
        )

    @classmethod
    def execute(cls, positive, negative, vae, width, height, length, batch_size,
                motion_amplitude=1.15, start_image=None, end_image=None, 
                clip_vision_output=None, clip_vision_output_end=None) -> io.NodeOutput:
        
        # 1. 严格的零latent初始化
        latent_time_dim = ((length - 1) // 4) + 1
        latent = torch.zeros([batch_size, 16, latent_time_dim, height // 8, width // 8], 
                           device=comfy.model_management.intermediate_device())
        
        image = torch.ones((length, height, width, 3), dtype=torch.float32, device=comfy.model_management.intermediate_device()) * 0.5
        
        has_image_input = False

        # --- start Image ---
        if start_image is not None:
            has_image_input = True
            start_image = start_image[:1]
            start_image = comfy.utils.common_upscale(
                start_image.movedim(-1, 1), width, height, "bilinear", "center"
            ).movedim(1, -1)
            image[0] = start_image[0]

        # --- end Image ---
        if end_image is not None:
            has_image_input = True
            end_image = end_image[:1]
            end_image = comfy.utils.common_upscale(
                end_image.movedim(-1, 1), width, height, "bilinear", "center"
            ).movedim(1, -1)
            image[-1] = end_image[0]

        if has_image_input:
            concat_latent_image = vae.encode(image[:, :, :, :3])
            
            mask = torch.ones((1, 1, latent.shape[2], concat_latent_image.shape[-2], 
                             concat_latent_image.shape[-1]), 
                            device=concat_latent_image.device, dtype=concat_latent_image.dtype)
            
            if start_image is not None:
                mask[:, :, 0] = 0.0
            
            if end_image is not None:
                mask[:, :, -1] = 0.0
            
            # 2. 运动幅度增强 (Motion/Brightness Fix)
            if motion_amplitude > 1.0:
                base_latent = concat_latent_image[:, :, 0:1]
                
                if end_image is not None:
                    # if end image exist, we only amplify the middle grey frames. last frame stays safe
                    middle_latent = concat_latent_image[:, :, 1:-1]
                    end_latent = concat_latent_image[:, :, -1:]
                    
                    diff = middle_latent - base_latent
                    diff_mean = diff.mean(dim=(1, 3, 4), keepdim=True)
                    diff_centered = diff - diff_mean
                    scaled_middle = base_latent + diff_centered * motion_amplitude + diff_mean
                    scaled_middle = torch.clamp(scaled_middle, -6, 6)
                    
                    # reconstruct
                    concat_latent_image = torch.cat([base_latent, scaled_middle, end_latent], dim=2)
                else:
                    # original behavior
                    gray_latent = concat_latent_image[:, :, 1:]
                    
                    diff = gray_latent - base_latent
                    diff_mean = diff.mean(dim=(1, 3, 4), keepdim=True)
                    diff_centered = diff - diff_mean
                    scaled_latent = base_latent + diff_centered * motion_amplitude + diff_mean
                    
                    scaled_latent = torch.clamp(scaled_latent, -6, 6)
                    concat_latent_image = torch.cat([base_latent, scaled_latent], dim=2)
            
            # 3. 注入到conditioning
            positive = node_helpers.conditioning_set_values(
                positive, {"concat_latent_image": concat_latent_image, "concat_mask": mask}
            )
            negative = node_helpers.conditioning_set_values(
                negative, {"concat_latent_image": concat_latent_image, "concat_mask": mask}
            )

            # 4. 参考帧增强 (Keeping original Painter logic for Start Image)
            if start_image is not None:
                ref_latent = vae.encode(start_image[:, :, :, :3])
                positive = node_helpers.conditioning_set_values(positive, {"reference_latents": [ref_latent]}, append=True)
                negative = node_helpers.conditioning_set_values(negative, {"reference_latents": [torch.zeros_like(ref_latent)]}, append=True)

        # clip vision just in case
        final_clip_vision = None
        
        if clip_vision_output is not None:
            final_clip_vision = clip_vision_output
            
        if clip_vision_output_end is not None:
            if final_clip_vision is not None:
                states = torch.cat([final_clip_vision.penultimate_hidden_states, 
                                  clip_vision_output_end.penultimate_hidden_states], dim=-2)
                final_clip_vision = comfy.clip_vision.Output()
                final_clip_vision.penultimate_hidden_states = states
            else:
                final_clip_vision = clip_vision_output_end

        if final_clip_vision is not None:
            positive = node_helpers.conditioning_set_values(positive, {"clip_vision_output": final_clip_vision})
            negative = node_helpers.conditioning_set_values(negative, {"clip_vision_output": final_clip_vision})

        out_latent = {}
        out_latent["samples"] = latent
        return io.NodeOutput(positive, negative, out_latent)


class PainterI2VExtension(ComfyExtension):
    @override
    async def get_node_list(self) -> list[type[io.ComfyNode]]:
        return [PainterI2V]

async def comfy_entrypoint() -> PainterI2VExtension:
    return PainterI2VExtension()


# 节点注册映射
NODE_CLASS_MAPPINGS = {
    "PainterI2V": PainterI2V,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "PainterI2V": "PainterI2V (Wan2.2 Slow-Motion Fix)",
}
