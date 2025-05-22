import torch
from diffusers import FluxPipeline
import os
from attention_processor import FluxBlendedAttnProcessor2_0
from glob import glob
from PIL import Image

from torchvision import transforms
import torchvision.transforms.functional as F

import numpy as np
import random

from utils_sample import set_seed, image_grid
import argparse

parser = argparse.ArgumentParser(description="IT-Blender with FLUX")

parser.add_argument("--scale", type=float, default=0.6, help="scale for Blended Attention")
parser.add_argument("--num_ref", type=int, default=1, help="Number of reference images")
parser.add_argument("--num_samples_per_ref", type=int, default=3, help="Number of samples to generate per a reference image")
parser.add_argument("--seed", type=int, default=42, help="random seed")
parser.add_argument("--obj", type=str, default="", help="an object to generate, e.g., monster cartoon character, dragon, sneakers, and handbag")
parser.add_argument("--temperature", type=float, default=1.2, help='''Temperature before softmax. Only used if num_ref > 1.
        Set greater than 1.0 if the result does not clearly apply multiple reference images.
        Setting greater than 1.5 can negatively affect the generation quality. 
        See appendices of our paper for further details.''')

# Parse arguments
args = parser.parse_args()

img_root_path = "assets/laion_square"
img_path_list = glob(os.path.join(img_root_path, "*.png"))
img_path_list = sorted(img_path_list)

num_ref = args.num_ref
scale = args.scale
num_samples_per_ref = args.num_samples_per_ref
seed = args.seed
temperature = args.temperature

set_seed(seed)

if args.obj == "":
    objs = ["monster cartoon character", "owl cartoon character"]
else:
    objs = [args.obj]

dtype = torch.bfloat16 # VRAM usage: 26726MiB
pipe = FluxPipeline.from_pretrained("black-forest-labs/FLUX.1-dev", torch_dtype=dtype)


pipe.enable_model_cpu_offload()

blended_attn_procs = {}
for name, _ in pipe.transformer.attn_processors.items():
    if "single" in name: # for single stream
        blended_attn_procs[name] = FluxBlendedAttnProcessor2_0(3072, ba_scale=scale, num_ref=num_ref, temperature=temperature)
    else:
        blended_attn_procs[name] = pipe.transformer.attn_processors[name]


pipe.transformer.set_attn_processor(blended_attn_procs)
pipe.to(dtype)

load_path = f"models/FLUX"
pretrained_blended_attn_weights = torch.load(os.path.join(load_path, "it-blender.bin"), map_location=pipe._execution_device)

key_changed_blended_attn_weights = {}
for key, value in pretrained_blended_attn_weights.items():
    block_idx = int(key.split(".")[0]) - 21 # for single stream
    k_or_v = key.split("_")[2]
    changed_key = f'single_transformer_blocks.{block_idx}.attn.processor.blended_attention_{k_or_v}_proj.weight' # for single stream
    key_changed_blended_attn_weights[changed_key] = value.to(dtype)
    
missing_keys, unexpected_keys = pipe.transformer.load_state_dict(key_changed_blended_attn_weights, strict=False)


for obj in objs:
    
    if num_ref > 1:
        out_path = os.path.join("outputs", "FLUX", f"scale{scale}_{obj}_nref{num_ref}_temp{temperature}")
    else:
        out_path = os.path.join("outputs", "FLUX", f"scale{scale}_{obj}_nref{num_ref}")
    os.makedirs(out_path, exist_ok=True)

    prompt = f"A photo of a {obj}"
    prompt += ", imaginative, creative, design"

    image_list = []
    
    for i, img_path in enumerate(img_path_list):
        image = Image.open(img_path).convert('RGB')
        image = image.resize((512, 512))
        image_list.append(image)

        if len(image_list) == num_ref:
            out_list = []
            for j in range(num_samples_per_ref):
                out = pipe(
                    prompt=prompt,
                    height=512,
                    width=512,
                    max_sequence_length=256,
                    generator=torch.Generator().manual_seed(seed+10*j),
                    it_blender_image=image_list
                ).images[0]
                out_list.append(out)

            out = image_grid(image_list + out_list, 1, num_samples_per_ref + num_ref)

            out.save(os.path.join(out_path, f"out{i}.png"))
            image_list = []
