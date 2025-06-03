import torch
from diffusers import StableDiffusionPipeline, StableDiffusionImg2ImgPipeline, StableDiffusionInpaintPipelineLegacy, DDIMScheduler, AutoencoderKL
from PIL import Image

from it_blender_sd15 import ITBlender
import os
from glob import glob
from natsort import natsorted

import numpy as np
import random

from utils_sample import set_seed, image_grid, resize_and_center_crop, resize_and_add_margin
import argparse

parser = argparse.ArgumentParser(description="IT-Blender with FLUX")

parser.add_argument("--scale", type=float, default=0.25, help="A scale for Blended Attention.")
parser.add_argument("--num_samples_per_ref", type=int, default=4, help="The number of samples to generate per a reference image.")
parser.add_argument("--seed", type=int, default=42, help="A random seed.")
parser.add_argument("--obj", type=str, default="", help="An object to generate, e.g., monster cartoon character, dragon, sneakers, and handbag.")
parser.add_argument("--ref_preprocessing", type=str, default="resize_addmargin", help='''
Two image preprocessing algorithms are provided to deal with both square and rectangular reference images; 
Select either one of \"resize_centercrop\" or \"resize_addmargin\". 
Default is resize_addmargin.''')

# Parse arguments
args = parser.parse_args()

img_root_path = "assets/laion_square"
img_path_list = glob(os.path.join(img_root_path, "*.png"))
img_path_list = sorted(img_path_list)

scale_ba = args.scale
num_samples_per_ref = args.num_samples_per_ref
seed = args.seed

num_samples_per_ref = 4

if args.obj == "":
    obj = "motorcycle"
else:
    obj = args.obj

prompt = f"best quality, high quality, a photo of a {obj}, 4k, detailed"

start_end_layers = [0, 16]
set_seed(seed)

root_path = f"models/sd15"

ckpt_path = os.path.join(root_path, "it-blender.bin")
out_path = os.path.join("outputs", "sd15", f"scale{scale_ba}_{obj}_layer{start_end_layers[0]}_{start_end_layers[1]}")
os.makedirs(out_path, exist_ok=True)

base_model_path = "runwayml/stable-diffusion-v1-5"
vae_model_path = "stabilityai/sd-vae-ft-mse"

device = "cuda"

noise_scheduler = DDIMScheduler(
    num_train_timesteps=1000,
    beta_start=0.00085,
    beta_end=0.012,
    beta_schedule="scaled_linear",
    clip_sample=False,
    set_alpha_to_one=False,
    steps_offset=1,
)

vae = AutoencoderKL.from_pretrained(vae_model_path).to(dtype=torch.float16)

# load SD pipeline, VRAM usage: 6364MiB
pipe = StableDiffusionPipeline.from_pretrained(
    base_model_path,
    torch_dtype=torch.float16,
    scheduler=noise_scheduler,
    vae=vae,
    feature_extractor=None,
    safety_checker=None
)

it_blender = ITBlender(pipe, ckpt_path, device, start_end_layers)

for i, img_path in enumerate(img_path_list):
    # read image prompt
    image = Image.open(img_path).convert('RGB')
    if args.ref_preprocessing == "resize_centercrop":
        image = resize_and_center_crop(image, target_size=512)
    elif args.ref_preprocessing == "resize_addmargin":
        image = resize_and_add_margin(image, target_size=512)
    else:
        raise ValueError("Not implemented preprocessing method. Choose either one of resize_centercrop or resize_addmargin.")

    images = [image]
    images += it_blender.generate(pil_image=image, num_samples=num_samples_per_ref, num_inference_steps=50, seed=seed,
            prompt=prompt, scale=scale_ba, fix_latents=False)

    grid = image_grid(images, 1, num_samples_per_ref+1)
    grid.save(os.path.join(out_path,f"out{i}.png"))