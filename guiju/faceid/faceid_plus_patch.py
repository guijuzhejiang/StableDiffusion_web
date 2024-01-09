# coding=utf-8
# @Time : 2024/1/9 下午4:21
# @File : faceid_plus_patch.py
from typing import List

import torch


def patch_generate(
        self,
        face_image=None,
        faceid_embeds=None,
        prompt=None,
        negative_prompt=None,
        scale=1.0,
        num_samples=4,
        seed=None,
        guidance_scale=7.5,
        num_inference_steps=30,
        s_scale=1.0,
        shortcut=False,
        **kwargs,
):
    self.set_scale(scale)

    num_prompts = faceid_embeds.size(0)

    if prompt is None:
        prompt = "best quality, high quality"
    if negative_prompt is None:
        negative_prompt = "monochrome, lowres, bad anatomy, worst quality, low quality"

    if not isinstance(prompt, List):
        prompt = [prompt] * num_prompts
    if not isinstance(negative_prompt, List):
        negative_prompt = [negative_prompt] * num_prompts

    image_prompt_embeds, uncond_image_prompt_embeds = self.get_image_embeds(faceid_embeds, face_image, s_scale,
                                                                            shortcut)

    bs_embed, seq_len, _ = image_prompt_embeds.shape
    image_prompt_embeds = image_prompt_embeds.repeat(1, num_samples, 1)
    image_prompt_embeds = image_prompt_embeds.view(bs_embed * num_samples, seq_len, -1)
    uncond_image_prompt_embeds = uncond_image_prompt_embeds.repeat(1, num_samples, 1)
    uncond_image_prompt_embeds = uncond_image_prompt_embeds.view(bs_embed * num_samples, seq_len, -1)

    with torch.inference_mode():
        # prompt_embeds_, negative_prompt_embeds_ = self.pipe.encode_prompt(
        #     prompt,
        #     device=self.device,
        #     num_images_per_prompt=num_samples,
        #     do_classifier_free_guidance=True,
        #     negative_prompt=negative_prompt,
        # )
        # prompt_embeds_, negative_prompt_embeds_ = self.get_pipeline_embeds(prompt[0], negative_prompt[0], self.device)
        # prompt_embeds_ = prompt_embeds_.repeat(1, num_samples, 1)
        # prompt_embeds_ = prompt_embeds_.view(bs_embed * num_samples, seq_len, -1)
        # negative_prompt_embeds_ = negative_prompt_embeds_.repeat(1, num_samples, 1)
        # negative_prompt_embeds_ = negative_prompt_embeds_.view(bs_embed * num_samples, seq_len, -1)

        prompt_embeds_compel = self.compel.build_conditioning_tensor(prompt[0])
        negative_prompt_embeds_compel = self.compel.build_conditioning_tensor(negative_prompt[0])
        prompt_embeds_ = prompt_embeds_compel.repeat(num_samples, 1, 1)
        negative_prompt_embeds_ = negative_prompt_embeds_compel.repeat(num_samples, 1, 1)

        prompt_embeds = torch.cat([prompt_embeds_, image_prompt_embeds], dim=1)
        negative_prompt_embeds = torch.cat([negative_prompt_embeds_, uncond_image_prompt_embeds], dim=1)

    generator = torch.Generator(self.device).manual_seed(seed) if seed is not None else None
    images = self.pipe(
        prompt_embeds=prompt_embeds,
        negative_prompt_embeds=negative_prompt_embeds,
        guidance_scale=guidance_scale,
        num_inference_steps=num_inference_steps,
        generator=generator,
        **kwargs,
    ).images

    return images
