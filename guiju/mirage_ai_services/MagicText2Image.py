# coding=utf-8
# @Time : 2023/12/15 下午3:14
# @File : magic_wallpapaper.py
import copy
import os
import random
import string

import numpy as np
from PIL import Image

from guiju.mirage_ai_services.MagicMirage import lora_mirage_dict
from lora_config import reference_dir
from modules.sd_samplers_kdiffusion import samplers_k_diffusion

common_prompts = ['masterpiece', 'best quality', 'breathtaking {prompt} . award-winning', 'professional',
                  '<lora:add-detail-xl:1>', '<lora:xl_more_art-full_v1:0.8>', '<lora:WowifierXL-V2:0.6>',
                  'extremely detailed']
sd_negative_prompt = 'lowres,bad anatomy,bad hands,text,error,missing fingers,extra digit,fewer digits,cropped,worst quality,low quality,normal quality,jpeg artifacts,signature,watermark,username,blurry,artist name'
text2image_style_prompts = {
    0: {'label': 'None',
        'prompt': '',
        'disallow': []},
    1: {'label': '胶片',
        'prompt': 'analog film photo {prompt} . faded film,desaturated,35mm photo,grainy,vignette,vintage,Kodachrome,Lomography,stained,found footage',
        'disallow': []},
    2: {'label': '动画',
        'prompt': 'anime artwork {prompt} . anime style,key visual,vibrant,studio anime',
        'disallow': []},
    3: {'label': '电影',
        'prompt': 'cinematic film still {prompt} . shallow depth of field,vignette,high budget,bokeh,cinemascope,moody,epic,gorgeous,film grain,grainy',
        'disallow': []},
    4: {'label': '漫画书',
        'prompt': 'comic {prompt} . graphic illustration,comic art,graphic novel art,vibrant',
        'disallow': []},
    5: {'label': '手工粘土',
        'prompt': 'play-doh style {prompt} . sculpture,clay art,centered composition,Claymation',
        'disallow': []},
    6: {'label': '梦幻',
        'prompt': 'ethereal fantasy concept art of  {prompt} . magnificent,celestial,ethereal,painterly,epic,majestic,magical,fantasy art,cover art,dreamy',
        'disallow': []},
    7: {'label': '等距',
        'prompt': 'isometric style {prompt} . vibrant,beautiful,crisp,intricate',
        'disallow': []},
    8: {'label': '线条艺术',
        'prompt': 'line art drawing {prompt} . professional,sleek,modern,minimalist,graphic,line art,vector graphics',
        'disallow': ['<lora:xl_more_art-full_v1:0.8>','<lora:WowifierXL-V2:0.6>','extremely detailed']},
    9: {'label': '多边形',
        'prompt': 'low-poly style {prompt} . low-poly game art,polygon mesh,jagged,blocky,wireframe edges,centered composition',
        'disallow': []},
    10: {'label': '霓虹朋克',
         'prompt': 'neonpunk style {prompt} . cyberpunk,vaporwave,neon,vibes,vibrant,stunningly beautiful,crisp,sleek,ultramodern,magenta highlights,purple shadows,high contrast,cinematic,intricate,professional',
         'disallow': []},
    11: {'label': '折纸',
         'prompt': 'origami style {prompt} . paper art,pleated paper,folded,origami art,pleats,cut and fold,centered composition',
         'disallow': []},
    12: {'label': '摄影',
         'prompt': 'cinematic photo {prompt} . 35mm photograph,film,bokeh,professional,4k',
         'disallow': []},
    13: {'label': '像素风',
         'prompt': 'pixel-art {prompt} . low-res,blocky,pixel art style,8-bit graphics',
         'disallow': ['<lora:xl_more_art-full_v1:0.8>','<lora:WowifierXL-V2:0.6>','extremely detailed']},
    14: {'label': '汽车',
         'prompt': 'Automotive advertisement style {prompt} . Sleek,dynamic,professional,commercial,vehicle-focused,high-resolution,Car Photography',
         'disallow': []},
    15: {'label': '时尚杂志',
         'prompt': 'Fashion editorial style {prompt} . High fashion,trendy,stylish,editorial,magazine style,professional',
         'disallow': []},
    16: {'label': '食品摄影',
         'prompt': 'Food photography style {prompt} . Appetizing,professional,culinary,high-resolution,commercial,Food Photography',
         'disallow': []},
    17: {'label': '奢侈品',
         'prompt': 'Luxury product style {prompt} . Elegant,sophisticated,high-end,luxurious,professional',
         'disallow': []},
    18: {'label': '房屋',
         'prompt': 'Real estate photography style {prompt} . Professional,inviting,well-lit,high-resolution,property-focused,commercial,Architecture Photography',
         'disallow': []},
    19: {'label': '小商品包装',
         'prompt': 'Retail packaging style {prompt} . Vibrant,enticing,commercial,product-focused,eye-catching,professional',
         'disallow': []},
    20: {'label': '抽象派',
         'prompt': 'abstract expressionist painting {prompt} . energetic brushwork,bold colors,abstract forms,expressive,emotional',
         'disallow': ['<lora:xl_more_art-full_v1:0.8>','<lora:WowifierXL-V2:0.6>','extremely detailed']},
    21: {'label': '装饰艺术',
         'prompt': 'Art Deco style {prompt} . geometric shapes,bold colors,luxurious,elegant,decorative,symmetrical,ornate,Interior Photography',
         'disallow': []},
    22: {'label': '色块拼图',
         'prompt': 'cubist artwork {prompt} . geometric shapes,abstract,innovative,revolutionary',
         'disallow': []},
    23: {'label': '涂鸦',
         'prompt': 'graffiti style {prompt} . street art,vibrant,urban,tag,mural',
         'disallow': []},
    24: {'label': '印象派',
         'prompt': 'impressionist painting {prompt} . loose brushwork,vibrant color,light and shadow play,captures feeling over form',
         'disallow': ['<lora:xl_more_art-full_v1:0.8>','<lora:WowifierXL-V2:0.6>','extremely detailed']},
    25: {'label': '点彩画',
         'prompt': 'pointillism style {prompt} . composed entirely of small,distinct dots of color,vibrant',
         'disallow': []},
    26: {'label': '流行艺术',
         'prompt': 'Pop Art style {prompt} . bright colors,bold outlines,popular culture themes,ironic or kitsch',
         'disallow': ['<lora:xl_more_art-full_v1:0.8>','<lora:WowifierXL-V2:0.6>','extremely detailed']},
    27: {'label': '迷幻',
         'prompt': 'psychedelic style {prompt} . vibrant colors,swirling patterns,abstract forms,surreal,trippy',
         'disallow': ['<lora:xl_more_art-full_v1:0.8>','<lora:WowifierXL-V2:0.6>','extremely detailed']},
    28: {'label': '蒸汽朋克',
         'prompt': 'steampunk style {prompt} . antique,mechanical,brass and copper tones,gears,intricate',
         'disallow': []},
    29: {'label': '水彩',
         'prompt': 'watercolor painting {prompt} . vibrant,beautiful,painterly,textural,artistic',
         'disallow': []},
    30: {'label': '生物赛博',
         'prompt': 'biomechanical cyberpunk {prompt} . cybernetics,human-machine fusion,dystopian,organic meets artificial,intricate',
         'disallow': []},
    31: {'label': '赛博机器人',
         'prompt': 'cybernetic robot {prompt} . android,AI,machine,metal,wires,tech,futuristic',
         'disallow': []},
    32: {'label': '赛博城市',
         'prompt': 'cyberpunk cityscape {prompt} . neon lights,alleys,skyscrapers,futuristic,vibrant colors,high contrast',
         'disallow': []},
    33: {'label': '未来主义',
         'prompt': 'futuristic style {prompt} . sleek,modern,ultramodern,high tech',
         'disallow': []},
    34: {'label': '复古赛博',
         'prompt': "retro cyberpunk {prompt} . 80's inspired,synthwave,neon,vibrant,retro futurism",
         'disallow': []},
    35: {'label': '复古科幻',
         'prompt': 'retro-futuristic {prompt} . vintage sci-fi,50s and 60s style,atomic age,vibrant',
         'disallow': []},
    36: {'label': '科幻',
         'prompt': 'sci-fi style {prompt} . futuristic,technological,alien worlds,space themes,advanced civilizations',
         'disallow': []},
    37: {'label': '泡泡龙',
         'prompt': 'Bubble Bobble style {prompt} . 8-bit,cute,pixelated,fantasy,vibrant,reminiscent of Bubble Bobble game',
         'disallow': []},
    38: {'label': '赛博游戏',
         'prompt': 'cyberpunk game style {prompt} . neon,dystopian,futuristic,digital,vibrant,high contrast,reminiscent of cyberpunk genre video games',
         'disallow': []},
    39: {'label': '格斗游戏',
         'prompt': 'fighting game style {prompt} . dynamic,vibrant,action-packed,detailed character design,reminiscent of fighting video games',
         'disallow': []},
    40: {'label': 'GTA',
         'prompt': 'GTA-style artwork {prompt} . satirical,exaggerated,pop art style,vibrant colors,iconic characters,action-packed',
         'disallow': []},
    41: {'label': '马里奥',
         'prompt': 'Super Mario style {prompt} . vibrant,cute,cartoony,fantasy,playful,reminiscent of Super Mario series',
         'disallow': []},
    42: {'label': '体素风',
         'prompt': 'Minecraft style {prompt} . blocky,pixelated,vibrant colors,recognizable characters and objects,game assets',
         'disallow': []},
    43: {'label': '宝可梦',
         'prompt': 'Pokémon style {prompt} . vibrant,cute,anime,fantasy,reminiscent of Pokémon series',
         'disallow': []},
    44: {'label': 'Street Fighter',
         'prompt': 'Street Fighter style {prompt} . vibrant,dynamic,arcade,2D fighting game,reminiscent of Street Fighter series',
         'disallow': []},
    45: {'label': 'disco',
         'prompt': 'disco-themed {prompt} . vibrant,groovy,retro 70s style,shiny disco balls,neon lights,dance floor',
         'disallow': []},
    46: {'label': '世界末日',
         'prompt': 'dystopian style {prompt} . bleak,post-apocalyptic,somber,dramatic',
         'disallow': []},
    47: {'label': '童话',
         'prompt': 'fairy tale {prompt} . magical,fantastical,enchanting,storybook style',
         'disallow': []},
    48: {'label': '哥特式',
         'prompt': 'gothic style {prompt},mysterious,haunting,dramatic,ornate',
         'disallow': []},
    49: {'label': '摇滚',
         'prompt': 'grunge style {prompt} . textured,distressed,vintage,edgy,punk rock vibe,dirty,noisy',
         'disallow': []},
    50: {'label': '恐怖',
         'prompt': 'horror-themed {prompt} . eerie,unsettling,spooky,suspenseful,grim',
         'disallow': []},
    51: {'label': '可爱',
         'prompt': 'kawaii style {prompt} . cute,adorable,brightly colored,cheerful,anime influence',
         'disallow': []},
    52: {'label': '魔幻',
         'prompt': 'lovecraftian horror {prompt} . eldritch,cosmic horror,unknown,mysterious,surreal',
         'disallow': []},
    53: {'label': '阴森','prompt': 'macabre style {prompt} . dark,gothic,grim,haunting',
         'disallow': []},
    54: {'label': '摩天高楼',
         'prompt': 'metropolis-themed {prompt} . urban,cityscape,skyscrapers,modern,futuristic',
         'disallow': []},
    55: {'label': '单色',
         'prompt': 'monochrome {prompt} . black and white,contrast,tone,texture',
         'disallow': []},
    56: {'label': '航海',
         'prompt': 'nautical-themed {prompt} . sea,ocean,ships,maritime,beach,marine life',
         'disallow': []},
    57: {'label': '宇宙太空',
         'prompt': 'space-themed {prompt} . cosmic,celestial,stars,galaxies,nebulas,planets,science fiction',
         'disallow': []},
    58: {'label': '染色玻璃',
         'prompt': 'stained glass style {prompt} . vibrant,beautiful,translucent,intricate',
         'disallow': []},
    59: {'label': '时尚赛博',
         'prompt': 'techwear fashion {prompt} . futuristic,cyberpunk,urban,tactical,sleek',
         'disallow': []},
    60: {'label': '部落',
         'prompt': 'tribal style {prompt} . indigenous,ethnic,traditional patterns,bold,natural colors',
         'disallow': []},
    61: {'label': '复杂单色',
         'prompt': 'zentangle {prompt} . intricate,abstract,monochrome,patterns,meditative',
         'disallow': []},
    62: {'label': '平面剪纸',
         'prompt': 'flat papercut style {prompt} . silhouette,clean cuts,paper,sharp edges,minimalist,color block',
         'disallow': ['<lora:xl_more_art-full_v1:0.8>','<lora:WowifierXL-V2:0.6>','extremely detailed']},
    63: {'label': '立体剪纸',
         'prompt': 'kirigami representation of {prompt} . 3D,paper folding,paper cutting,Japanese,intricate,symmetrical,precision,clean lines',
         'disallow': ['<lora:xl_more_art-full_v1:0.8>','<lora:WowifierXL-V2:0.6>','extremely detailed']},
    64: {'label': '纸浆',
         'prompt': 'paper mache representation of {prompt} . 3D,sculptural,textured,handmade,vibrant,fun',
         'disallow': ['<lora:xl_more_art-full_v1:0.8>','<lora:WowifierXL-V2:0.6>','extremely detailed']},
    65: {'label': '剪纸拼图',
         'prompt': 'papercut collage of {prompt} . mixed media,textured paper,overlapping,asymmetrical,abstract,vibrant',
         'disallow': ['<lora:xl_more_art-full_v1:0.8>','<lora:WowifierXL-V2:0.6>','extremely detailed']},
    66: {'label': '地外文明',
         'prompt': 'alien-themed {prompt} . extraterrestrial,cosmic,otherworldly,mysterious,sci-fi',
         'disallow': []},
    67: {'label': '黑白电影',
         'prompt': 'film noir style {prompt} . monochrome,high contrast,dramatic shadows,1940s style,mysterious,cinematic',
         'disallow': []},
    68: {'label': '高清',
         'prompt': 'HDR photo of {prompt} . High dynamic range,vivid,rich details,clear shadows and highlights,realistic,intense,enhanced contrast,Hyperdetailed Photography',
         'disallow': []}}


class MagicText2Image(object):
    operator = None
    sd_model_name = 'juggernautXL_v9Rundiffusionphoto2'

    def __init__(self, operator):
        self.operator = operator

    def __call__(self, *args, **kwargs):
        if self.operator.shared.sd_model.sd_checkpoint_info.model_name != self.sd_model_name:
            self.operator.shared.change_sd_model(self.sd_model_name)
        # read params
        # params, user_id, input_image, pic_name
        params = kwargs['params']
        user_id = kwargs['user_id']
        pic_name = kwargs['pic_name']

        _batch_size = int(params['batchSize'])
        _output_width = int(params['width'])
        _output_height = int(params['height'])
        _style = int(params['style'])
        _prompt = params['prompt']
        _selected_aspect = _output_width / _output_height

        if self.operator.update_progress(20):
            return {'success': True}

        # img2img generate bg
        prompt_styles = None
        steps = 20
        sampler_index = 16  # sampling method modules/sd_samplers_kdiffusion.py
        restore_faces = False
        tiling = False
        n_iter = 1
        cfg_scale = 6
        seed = -1.0
        subseed = -1.0
        subseed_strength = 0
        seed_resize_from_h = 0
        seed_resize_from_w = 0
        seed_enable_extras = False
        override_settings_texts = []

        # celery_task.update_state(state='PROGRESS', meta={'progress': 50})
        if self.operator.update_progress(50):
            return {'success': True}

        task_id = f"task({''.join([random.choice(string.ascii_letters) for c in range(15)])})"
        _common_prompt = ','.join([x for x in common_prompts if x not in text2image_style_prompts[_style]['disallow']])
        sd_positive_prompt = ','.join([text2image_style_prompts[_style]['prompt'], _prompt, _common_prompt])

        print("-------------------txt2image logger-----------------")
        print(f"sd_positive_prompt: {sd_positive_prompt}")
        print(f"sd_negative_prompt: {sd_negative_prompt}")
        print(f"Sampling method: {samplers_k_diffusion[sampler_index]}")
        self.operator.logging(
            f"[{pic_name}]:\nsd_positive_prompt: {sd_positive_prompt}\nsd_negative_prompt: {sd_negative_prompt}\nSampling method: {samplers_k_diffusion[sampler_index]}",
            f"logs/sd_webui.log")

        # controlnet args
        controlnet_args_unit1 = self.operator.scripts.scripts_img2img.alwayson_scripts[
            self.operator.cnet_idx].get_default_ui_unit()
        # depth
        controlnet_args_unit1.enabled = False
        controlnet_args_unit2 = copy.deepcopy(controlnet_args_unit1)
        controlnet_args_unit3 = copy.deepcopy(controlnet_args_unit1)
        sam_args = [0,
                    False, {}, {},  # adetail args
                    controlnet_args_unit1, controlnet_args_unit2, controlnet_args_unit3,  # controlnet args
                    # sam
                    False,  # inpaint_upload_enable
                    False, 0, None,
                    [],
                    0,  # sam_output_chosen_mask
                    False, [], [], False, 0, 1, False, False, 0, None, [], -2, False, [],
                    '<ul>\n<li><code>CFG Scale</code>should be 2 or lower.</li>\n</ul>\n',
                    True, True, '',
                    # tiled diffsuion
                    False, 'MultiDiffusion', False, True,
                    1024, 1024, 64, 64, 32, 8, 'None', 2, False, 10, 1, 1,
                    64, False, False, False, False, False, 0.4, 0.4, 0.2, 0.2, '', '', 'Background', 0.2, -1.0,
                    False, 0.4, 0.4, 0.2, 0.2, '', '', 'Background', 0.2, -1.0, False, 0.4, 0.4, 0.2, 0.2, '',
                    '', 'Background', 0.2, -1.0, False, 0.4, 0.4, 0.2, 0.2, '', '', 'Background', 0.2, -1.0,
                    False, 0.4, 0.4, 0.2, 0.2, '', '', 'Background', 0.2, -1.0, False, 0.4, 0.4, 0.2, 0.2, '',
                    '', 'Background', 0.2, -1.0, False, 0.4, 0.4, 0.2, 0.2, '', '', 'Background', 0.2, -1.0,
                    False, 0.4, 0.4, 0.2, 0.2, '', '', 'Background', 0.2, -1.0,
                    # tiled_vae
                    False, 256, 48, True, True, True,
                    False
                    ]

        # 生成
        txt2img_res = self.operator.txt2img.txt2img(task_id,
                                                    sd_positive_prompt,
                                                    sd_negative_prompt,
                                                    prompt_styles,
                                                    steps,
                                                    sampler_index,
                                                    restore_faces,
                                                    tiling,
                                                    n_iter,
                                                    _batch_size,  # batch size
                                                    cfg_scale,
                                                    seed, subseed, subseed_strength, seed_resize_from_h,
                                                    seed_resize_from_w,
                                                    seed_enable_extras,
                                                    _output_height,
                                                    _output_width,
                                                    False,  # enable_hr
                                                    0.7,  # denoising_strength
                                                    2.0,  # hr_scale
                                                    'Latent',  # hr_upscaler"
                                                    0,  # hr_second_pass_steps
                                                    0,  # hr_resize_x
                                                    0,  # hr_resize_y
                                                    sampler_index,  # hr_sampler_index
                                                    '',  # hr_prompt
                                                    '',  # hr_negative_prompt,
                                                    override_settings_texts,
                                                    *sam_args)

        return [x.convert('RGB') for x in txt2img_res[0][:_batch_size]]
