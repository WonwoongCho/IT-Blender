
from attention_processor import (
    AttnProcessor2_0 as AttnProcessor,
)
from attention_processor import (
    BlendedAttnProcessor2_0 as BlendedAttnProcessor,
)
from PIL import Image

from typing import List
import torch
import os
from torchvision import transforms

import shutil
from huggingface_hub import hf_hub_download

def get_generator(seed, device):

    if seed is not None:
        if isinstance(seed, list):
            generator = [torch.Generator(device).manual_seed(seed_item) for seed_item in seed]
        else:
            generator = torch.Generator(device).manual_seed(seed)
    else:
        generator = None

    return generator


class ITBlender:
    def __init__(self, sd_pipe, ba_ckpt, device, start_end_layers):
        self.device = device

        self.ba_ckpt = ba_ckpt
        self.start_end_layers = start_end_layers
        
        self.pipe = sd_pipe.to(self.device)
        self.set_it_blender()
        
        self.transform = transforms.Compose([
            transforms.Resize(512, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.CenterCrop(512),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])
        ])

        self.load_it_blender()

    def set_it_blender(self):
        unet = self.pipe.unet
        attn_procs = {}
        SD_15_sa_name_list = ['down_blocks.0.attentions.0.transformer_blocks.0.attn1.processor', 'down_blocks.0.attentions.1.transformer_blocks.0.attn1.processor', 'down_blocks.1.attentions.0.transformer_blocks.0.attn1.processor', 'down_blocks.1.attentions.1.transformer_blocks.0.attn1.processor', 'down_blocks.2.attentions.0.transformer_blocks.0.attn1.processor', 'down_blocks.2.attentions.1.transformer_blocks.0.attn1.processor', 'mid_block.attentions.0.transformer_blocks.0.attn1.processor', 'up_blocks.1.attentions.0.transformer_blocks.0.attn1.processor', 'up_blocks.1.attentions.1.transformer_blocks.0.attn1.processor', 'up_blocks.1.attentions.2.transformer_blocks.0.attn1.processor', 'up_blocks.2.attentions.0.transformer_blocks.0.attn1.processor', 'up_blocks.2.attentions.1.transformer_blocks.0.attn1.processor', 'up_blocks.2.attentions.2.transformer_blocks.0.attn1.processor', 'up_blocks.3.attentions.0.transformer_blocks.0.attn1.processor', 'up_blocks.3.attentions.1.transformer_blocks.0.attn1.processor', 'up_blocks.3.attentions.2.transformer_blocks.0.attn1.processor']
        # temp = []
        for name in unet.attn_processors.keys():
            # cross_attention_dim = None if name.endswith("attn1.processor") else unet.config.cross_attention_dim
            if name.endswith("attn1.processor"):
                cross_attention_dim = -1
            else:
                cross_attention_dim = None
            if name.startswith("mid_block"):
                hidden_size = unet.config.block_out_channels[-1]
            elif name.startswith("up_blocks"):
                block_id = int(name[len("up_blocks.")])
                hidden_size = list(reversed(unet.config.block_out_channels))[block_id]
            elif name.startswith("down_blocks"):
                block_id = int(name[len("down_blocks.")])
                hidden_size = unet.config.block_out_channels[block_id]
            if cross_attention_dim is None:
                attn_procs[name] = AttnProcessor(is_training=False)
            else:
                sa_layer_idx = SD_15_sa_name_list.index(name)
                if sa_layer_idx >= self.start_end_layers[0] and sa_layer_idx < self.start_end_layers[1]:
                    attn_procs[name] = BlendedAttnProcessor(
                        hidden_size=hidden_size).to(self.device, dtype=torch.float16
                    )
                else:
                    attn_procs[name] = AttnProcessor()

        unet.set_attn_processor(attn_procs)

    def load_it_blender(self):
        try:
            state_dict = torch.load(self.ba_ckpt, map_location="cpu")
        except:
            model_file = hf_hub_download(
                repo_id="Wonwoong/IT-Blender",
                filename="sd15/it-blender.bin" # adjust the filename as needed
            )
            os.makedirs("/".join(self.ba_ckpt.split("/")[:-1]), exist_ok=True)
            shutil.copy(model_file, self.ba_ckpt)
            state_dict = torch.load(self.ba_ckpt, map_location="cpu")
            
        ba_layers = torch.nn.ModuleList(self.pipe.unet.attn_processors.values())
        try:
            ba_layers.load_state_dict(state_dict["it_blender"])
        except:
            print("Make sure that the loaded parameters exactly match the current network settings.")
            ba_layers.load_state_dict(state_dict["it_blender"], strict=False)

    def set_scale(self, scale):
        for attn_processor in self.pipe.unet.attn_processors.values():
            if isinstance(attn_processor, BlendedAttnProcessor):
                attn_processor.scale = scale

    def generate(
        self,
        pil_image=None,
        prompt=None,
        negative_prompt=None,
        scale=1.0,
        num_samples=4,
        seed=None,
        guidance_scale=7.5,
        num_inference_steps=30,
        attention_mask = None,
        fix_latents=False,
        **kwargs,
    ):
        self.set_scale(scale)

        if pil_image is not None:
            num_prompts = 1 if isinstance(pil_image, Image.Image) else len(pil_image)

        if prompt is None:
            prompt = "best quality, high quality"
        if negative_prompt is None:
            negative_prompt = "monochrome, lowres, bad anatomy, worst quality, low quality"

        if not isinstance(prompt, List):
            prompt = [prompt] * num_prompts
        if not isinstance(negative_prompt, List):
            negative_prompt = [negative_prompt] * num_prompts
        if len(prompt) != len(negative_prompt):
            negative_prompt = negative_prompt * len(prompt)

        with torch.inference_mode():
            prompt_embeds_, negative_prompt_embeds_ = self.pipe.encode_prompt(
                prompt,
                device=self.device,
                num_images_per_prompt=num_samples,
                do_classifier_free_guidance=True,
                negative_prompt=negative_prompt,
            )
            prompt_embeds = torch.cat([prompt_embeds_], dim=1)
            negative_prompt_embeds = torch.cat([negative_prompt_embeds_], dim=1)
        
        generator = get_generator(seed, self.device)

        clean_image = self.transform(pil_image.convert("RGB"))
        
        kwargs = {
            "clean_image":clean_image,
            "attention_mask":attention_mask,
            "fix_latents":fix_latents
        }
        
        images = self.pipe(
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            guidance_scale=guidance_scale,
            num_inference_steps=num_inference_steps,
            generator=generator,
            **kwargs,
        ).images
        
        return images