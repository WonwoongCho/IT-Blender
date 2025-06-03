# IT-Blender


<img src='./assets/main_figure_update.png' width='75%' />
<br>

<a href="https://imagineforme.github.io/"><img alt="Build" src="https://img.shields.io/badge/Project%20Page-ITBlender-yellow"></a> 
<a href="https://huggingface.co/Yuanshi/OminiControl"><img src="https://img.shields.io/badge/🤗_HuggingFace-Model-ffbd45.svg" alt="HuggingFace"></a>
<a href="https://huggingface.co/spaces/Yuanshi/OminiControl"><img src="https://img.shields.io/badge/🤗_HuggingFace-Demo-ffbd45.svg" alt="HuggingFace"></a>
<br>
<a href="https://arxiv.org/abs/2411.15098"><img src="https://img.shields.io/badge/ariXv-ITBlender-A42C25.svg" alt="arXiv"></a>

> **Imagine for Me: Creative Conceptual Blending of Real Images and Text via Blended Attention**
> <br>
> [Wonwoong Cho*](https://wonwoongcho.github.io), 
> [Yanxia Zhang**](https://www.yanxiazhang.com/), 
> [Yan-Ying Chen**](https://www.tri.global/about-us/dr-yan-ying-chen)
> [David Inouye*](https://www.davidinouye.com/)
> <br>
> \* Elmore Family School of Electrical and Computer Engineering, Purdue University
> <br>
> \*\* Toyota Research Institute


## Features

IT-Blender is a T2I diffusion adapter that can automate the blending process of visual and textual concepts to enhance human creativity.

* **Preserving detailed visual concepts from a reference image**:  We leverage the denoising network (both UNet-based and DiT-based) as an image encoder to maintain the details of visual concepts.
* **Disentangling textual and visual concepts**:  We design a novel Blended Attention on top of the image self-attention module, where textual concepts are physically separated, encouraging disentanglement of textual and visual concepts.


## News
- **2025-06-12**: ⭐️ Sampling codes of [IT-Blender](https://arxiv.org/abs/2503.08280) on both SD 1.5 and FLUX have been released.

## Quick Start
### Setup
1. **Environment setup**
```bash
conda create -n itblender python=3.12
conda activate itblender
```
2. **Requirements installation**
```bash
pip install -r requirements.txt
```

### Usage example
1. **FLUX (`FLUX.1-dev`)**
```
python sample_flux.py
```
- Peak VRAM usage is 26726MiB with a single gpu.
- Options
  - ``--scale``: A scale for Blended Attention. The default value is 0.6. A value between 0.5 and 0.8 is recommended. (float, default=0.6)
  - ``--num_ref``: The number of reference images. (int, default=1)
  - ``--num_samples_per_ref`` : The number of samples to generate per a reference image. (int, default=3)
  - ``--seed`` : A random seed. (int, default=42)
  - ``--obj`` : An object to generate, e.g., monster cartoon character, dragon, sneakers, and handbag. (str, default="", which is replaced with ``["monster cartoon character", "owl cartoon character"]``)
  - ``--temperature`` : A temperature before softmax. Only used if num_ref > 1.
        Set greater than 1.0 (low temp) if the result does not clearly apply multiple reference images.
        This sharpens the softmax distribution, possibly helping to prevent ambiguous mixtures of visual concepts.
        Setting greater than 1.0 can negatively affect the generation quality. A value less than 1.5 is recommended. 
        See appendices of our paper for further details. (float, default=1.0)
  - ``--ref_preprocessing`` : Two image preprocessing algorithms are provided to deal with both square and rectangular reference images; Select either one of \"resize_centercrop\" or \"resize_addmargin\". (str, default=``resize_addmargin``)


2. **StableDiffusion (`SD 1.5`)**
```
python sample_sd15.py
```
- Peak VRAM usage is 6364 MiB with a single gpu.
- Options
  - ``--scale``: A scale for Blended Attention. The default value is 0.25. A value between 0.2 and 0.3 is recommended. (float, default=0.25)
  - ``--num_samples_per_ref`` : The number of samples to generate per a reference image. (int, default=4)
  - ``--seed`` : A random seed. (int, default=42)
  - ``--obj`` : An object to generate, e.g., monster cartoon character, dragon, sneakers, and handbag. (str, default="", which is replaced with ``"motorcycle"``)
  - ``--ref_preprocessing`` : Two image preprocessing algorithms are provided to deal with both square and rectangular reference images; Select either one of \"resize_centercrop\" or \"resize_addmargin\". (str, default=``resize_addmargin``)


## Pretrained Models

| Model                                                                                            | Base model     | Description                                                                                                                                                 | Resolution   |
| ------------------------------------------------------------------------------------------------ | -------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------ |
| [`IT-Blender FLUX`](https://huggingface.co/Yuanshi/OminiControl/tree/main/experimental) | FLUX.1-dev | The model used in the paper. 1.43 GB.                                                                                                                               | (512, 512)   |
| [`IT-Blender StableDiffusion`](https://huggingface.co/Yuanshi/OminiControl/tree/main/omini)      | SD 1.5 | The model used in the paper. 99.1 MB.                                                                                                         | (512, 512)   |

## Feasible Design 

The blended results are more practical and feasible when the given cross-modal concepts are semantically close. More examples are provided in our project page.

<img src='./assets/feasible_first_image.png' width='75%' />

## Societal Impact

- **Positive societal impact**: IT-Blender can augment human creativity, especially for people in creative industries, e.g., design and marketing. With IT-Blender, designers might be able to have better final
design outcome by exploring wide design space in the ideation stage.
- **Negative societal impact**: IT-Blender can be used to apply the design of an existing product to the new products. The user must be aware of the fact that they can infringe on the company’s intellectual property if a specific texture pattern or material combination is registered. *We encourage users to use IT-Blender to augment creativity in the ideation stage, rather than directly having a final design outcome.*

## Citation
```
@article{tan2024ominicontrol,
  title={OminiControl: Minimal and Universal Control for Diffusion Transformer},
  author={Tan, Zhenxiong and Liu, Songhua and Yang, Xingyi and Xue, Qiaochu and Wang, Xinchao},
  journal={arXiv preprint arXiv:2411.15098},
  year={2024}
}