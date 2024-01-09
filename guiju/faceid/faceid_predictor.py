# coding=utf-8
# @Time : 2023/11/3 下午12:55
# @File : facer_parsing.py
import torch
from diffusers import StableDiffusionPipeline, DDIMScheduler, AutoencoderKL
from insightface.utils import face_align
from guiju.faceid.faceid_plus_patch import patch_generate
from ip_adapter.ip_adapter_faceid import IPAdapterFaceIDPlus

# patch
IPAdapterFaceIDPlus.generate = patch_generate


class FaceIDPredictor:
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # base_model_path = "/media/zzg/GJ_disk01/pretrained_model/Yntec_ChilloutMix"   #手不好
    # base_model_path = "/media/zzg/GJ_disk01/pretrained_model/stablediffusionapi_chilloutmix"   #手不好
    # base_model_path = "/media/zzg/GJ_disk01/pretrained_model/SG161222/Realistic_Vision_V6.0_B1_noVAE"       #偶尔会脸崩891022477.jpg
    # base_model_path = "/media/zzg/GJ_disk01/pretrained_model/SG161222/Realistic_Vision_V5.1_noVAE"              #还原度高，手容易崩坏
    # base_model_path = "/media/zzg/GJ_disk01/pretrained_model/SG161222/Realistic_Vision_V5.0_noVAE"
    # base_model_path = "/media/zzg/GJ_disk01/pretrained_model/SG161222/Realistic_Vision_V4.0_noVAE"
    # base_model_path = "/media/zzg/GJ_disk01/pretrained_model/jzli/PerfectWorld-v5"
    # base_model_path = "/media/zzg/GJ_disk01/pretrained_model/Lykon/dreamshaper-8"
    # base_model_path = "/media/zzg/GJ_disk01/pretrained_model/emilianJR/chilloutmix_NiPrunedFp32Fix"   #加载模型报错
    # base_model_path = "/media/zzg/GJ_disk01/pretrained_model/jzli/majicMIX-realistic-7"               #还原度高，手挺好
    base_model_path = "/home/ray/Workspace/model/faceid/epiCRealism"  # 还原度高，手容易崩坏
    vae_model_path = "/home/ray/Workspace/model/faceid/sd-vae-ft-mse"
    image_encoder_path = "/home/ray/Workspace/model/faceid/image_encoder"
    ip_ckpt = "/home/ray/Workspace/model/faceid/ip-adapter-faceid-plus_sd15.bin"

    def __init__(self, face_analyser):
        self.face_analyser = face_analyser

        noise_scheduler = DDIMScheduler(
            num_train_timesteps=1000,
            beta_start=0.00085,
            beta_end=0.012,
            beta_schedule="scaled_linear",
            clip_sample=False,
            set_alpha_to_one=False,
            steps_offset=1,
        )
        # 如果低模带有vae,StableDiffusionPipeline可以不加载vae模型
        vae = AutoencoderKL.from_pretrained(self.vae_model_path).to(dtype=torch.float16)
        pipe = StableDiffusionPipeline.from_pretrained(
            self.base_model_path,
            torch_dtype=torch.float16,
            scheduler=noise_scheduler,
            vae=vae,
            feature_extractor=None,
            safety_checker=None,
            # custom_pipeline="lpw_stable_diffusion",
            # custom_pipeline="stable_diffusion_mega",
        )

        self.ip_model = IPAdapterFaceIDPlus(pipe, self.image_encoder_path, self.ip_ckpt, self.device)
        # 跟踪输入照片脸部特征的程度,默认1.0,如果不想一模一样可调低
        self.ip_model.set_scale(1.0)

        pipe.to(self.device)

    def __call__(self, image_arr, prompt, negative_prompt, batch_size, *args, **kwargs):
        faces = self.face_analyser.get(image_arr)
        faceid_embed = torch.from_numpy(faces[0].normed_embedding).unsqueeze(0)
        face_image = face_align.norm_crop(image_arr, landmark=faces[0].kps,
                                          image_size=224)  # you can also segment the face

        image = self.ip_model.generate(
            prompt=prompt,
            negative_prompt=negative_prompt,
            face_image=face_image,
            faceid_embeds=faceid_embed,
            width=512,
            height=768,
            num_inference_steps=30,
            num_samples=batch_size,
            truncation=False,
            # guidance_scale=7.5,
            # scale=1.0,
            # seed=2023,
        )
        # print(image)
        return image


if __name__ == '__main__':
    pass
    # res = FaceParsing()(Image.open('/home/ray/Workspace/project/stable_diffusion/StableDiffusion_web/tmp/hair_origin_wGqoHn_save.png'), keep='True')
    # res.show()